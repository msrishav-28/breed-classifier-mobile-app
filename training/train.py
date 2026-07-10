#!/usr/bin/env python3
"""Trains the cattle/buffalo breed classifier and exports the TFLite model
the Android app consumes.

The exported artifacts implement the app's model contract (docs/MODEL.md):

  * ``breed_classifier.tflite`` — float32 input [1, 224, 224, 3] with raw RGB
    pixel values in 0..255 (preprocessing is baked into the model), float32
    output [1, N] of class probabilities (softmax).
  * ``labels.txt`` — N class labels, one per line, in output order.

Dataset layout (standard directory-per-class):

    dataset/
      Gir/            *.jpg
      Red_Sindhi/     *.jpg
      Murrah/         *.jpg
      ...

Directory names become model labels; use the breed names from
``app/src/main/assets/data/breed_mapping.csv`` (spaces may be written as
underscores — the app matches them case- and separator-insensitively).

Usage:

    python train.py --data-dir /path/to/dataset --output-dir build
    python train.py --data-dir ... --epochs 20 --fine-tune-epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = 224
VALIDATION_SPLIT = 0.2
SEED = 1337


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset root: one sub-directory per breed")
    parser.add_argument("--output-dir", type=Path, default=Path("build"),
                        help="Where model artifacts are written")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs for the frozen-backbone phase")
    parser.add_argument("--fine-tune-epochs", type=int, default=8,
                        help="Epochs for the fine-tuning phase (0 disables)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-layers", type=int, default=60,
                        help="How many trailing backbone layers to unfreeze")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency class weighting")
    return parser.parse_args()


def load_datasets(data_dir: Path, batch_size: int):
    common = dict(
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
    )
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir, subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **common)

    class_names = list(train_ds.class_names)

    # Augmentation lives in the data pipeline, NOT in the model: the exported
    # TFLite graph must contain only inference ops (see docs/MODEL.md).
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomContrast(0.15),
            keras.layers.RandomBrightness(0.15),
        ],
        name="augmentation",
    )

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.cache()
        .shuffle(1000)
        .map(lambda x, y: (augmentation(x, training=True), y),
             num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> keras.Model:
    # include_preprocessing=True bakes 0..255 rescaling into the graph, which
    # is what lets the app feed raw pixel values.
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        include_preprocessing=True,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="breed_classifier")


def compile_model(model: keras.Model, learning_rate: float, label_smoothing: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(3, name="top3")],
    )


def compute_class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    image_extensions = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".gif", ".bmp"}
    counts = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        count = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        counts.append(count)
    total = sum(counts)
    if total == 0 or any(count == 0 for count in counts):
        return {}
    return {
        index: total / (len(class_names) * count)
        for index, count in enumerate(counts)
    }


def unfreeze_top_layers(model: keras.Model, layer_count: int) -> None:
    backbone = next(l for l in model.layers if l.name.startswith("efficientnetv2"))
    backbone.trainable = True
    for layer in backbone.layers[:-layer_count]:
        layer.trainable = False
    # BatchNorm statistics must stay frozen during fine-tuning.
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def export_tflite(model: keras.Model, class_names: list[str], output_dir: Path) -> Path:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "breed_classifier.tflite"
    model_path.write_bytes(tflite_model)
    (output_dir / "labels.txt").write_text("\n".join(class_names) + "\n")
    return model_path


def verify_export(model_path: Path, num_classes: int) -> None:
    """Sanity-checks the exported model against the app contract."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    expected_input = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
    if list(inp["shape"]) != expected_input:
        sys.exit(f"Export failed contract: input shape {inp['shape']} != {expected_input}")
    if inp["dtype"].__name__ != "float32":
        sys.exit(f"Export failed contract: input dtype {inp['dtype']} != float32")
    if int(out["shape"][-1]) != num_classes:
        sys.exit(f"Export failed contract: {out['shape'][-1]} outputs for {num_classes} labels")
    print(f"Contract OK: float32 {expected_input} -> [1, {num_classes}]")


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        sys.exit(f"Dataset directory not found: {args.data_dir}")

    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    class_weights = None if args.no_class_weights else compute_class_weights(args.data_dir, class_names)
    if class_weights:
        print("Class weights:", json.dumps(class_weights, indent=2))

    model = build_model(len(class_names))
    compile_model(model, args.learning_rate, args.label_smoothing)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
    histories = []

    print("\n=== Phase 1: training classifier head ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    histories.append({"phase": "frozen_head", "history": history.history})

    if args.fine_tune_epochs > 0:
        print("\n=== Phase 2: fine-tuning backbone ===")
        unfreeze_top_layers(model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_learning_rate, args.label_smoothing)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        histories.append({"phase": "fine_tune", "history": history.history})

    loss, accuracy, top3 = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {accuracy:.3f} (top-3: {top3:.3f})")

    model_path = export_tflite(model, class_names, args.output_dir)
    verify_export(model_path, len(class_names))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"val_accuracy": round(float(accuracy), 4),
         "val_top3_accuracy": round(float(top3), 4),
         "val_loss": round(float(loss), 4),
         "classes": class_names,
         "label_smoothing": args.label_smoothing,
         "class_weights": class_weights or {}},
        indent=2,
    ))
    (args.output_dir / "history.json").write_text(json.dumps(histories, indent=2))

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nExported {model_path} ({size_mb:.1f} MB)")
    print(f"Labels:   {args.output_dir / 'labels.txt'}")
    print(f"Metrics:  {metrics_path}")
    print("\nTo bundle into the app:")
    print(f"  cp {model_path} app/src/main/assets/models/breed_classifier.tflite")
    print(f"  cp {args.output_dir / 'labels.txt'} app/src/main/assets/models/labels.txt")


if __name__ == "__main__":
    main()
