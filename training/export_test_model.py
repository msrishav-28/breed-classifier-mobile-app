#!/usr/bin/env python3
"""Builds the tiny TFLite model used by the app's instrumented tests.

The exported model implements exactly the production model contract
(float32 [1,224,224,3] raw RGB in, float32 [1,N] softmax out — see
docs/MODEL.md) but uses a very small CNN so the fixture stays a few
hundred kilobytes and can live in version control under
app/src/androidTest/assets/. It is trained on a procedural texture
dataset (training/make_fixture_dataset.py output or any directory-per-
class dataset) only so that tests get deterministic, confident
predictions; it does not recognise animals and must never ship in a
release build.

Usage:
    python export_test_model.py --data-dir <dataset> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = 224
SEED = 20260704


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_tiny_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    for filters in (16, 32, 64, 64):
        x = keras.layers.Conv2D(filters, 3, strides=2, padding="same",
                                use_bias=False)(x)
        x = keras.layers.BatchNormalization(momentum=0.9)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="test_fixture_classifier")


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(SEED)

    common = dict(
        validation_split=0.2,
        seed=SEED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    train_ds = keras.utils.image_dataset_from_directory(
        args.data_dir, subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir, subset="validation", **common)
    class_names = list(train_ds.class_names)

    model = build_tiny_model(len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds.cache(), validation_data=val_ds.cache(),
              epochs=args.epochs)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    if accuracy < 0.9:
        sys.exit(f"Fixture model too weak for deterministic tests: "
                 f"val_accuracy={accuracy:.3f} (< 0.9)")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "breed_classifier.tflite"
    model_path.write_bytes(tflite_model)
    (args.output_dir / "labels.txt").write_text("\n".join(class_names) + "\n")

    size_kb = model_path.stat().st_size / 1024
    print(f"val_accuracy={accuracy:.3f}  size={size_kb:.0f} KB")
    print(f"Exported {model_path}")


if __name__ == "__main__":
    main()
