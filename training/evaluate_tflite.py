#!/usr/bin/env python3
"""Evaluate an exported TFLite model on an untouched directory-per-class test set.

Preprocessing intentionally mirrors ``TfLiteBreedClassifier`` in the Android
app: center-crop to square, bilinear resize to 224x224, RGB float32 pixels in
raw 0..255 range. Do not add Keras/EfficientNet preprocessing here; it must be
baked into the model graph.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import tensorflow as tf

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".gif", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images-per-class", type=int)
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [label for label in labels if label]


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[None, ...]


def softmax_if_needed(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64)
    if np.all(scores >= 0.0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        return scores.astype(np.float32)
    scores -= scores.max()
    exp = np.exp(scores)
    return (exp / exp.sum()).astype(np.float32)


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


def precision_recall_f1(confusion: np.ndarray, labels: list[str]) -> list[ClassMetrics]:
    metrics = []
    for i, label in enumerate(labels):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append(
            ClassMetrics(
                label=label,
                support=int(confusion[i, :].sum()),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return metrics


def calibration_bins(rows: list[dict[str, object]], bin_count: int = 10) -> list[dict[str, float]]:
    bins = []
    total = len(rows)
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            row for row in rows
            if lower <= float(row["top1_confidence"]) < upper
            or (index == bin_count - 1 and math.isclose(float(row["top1_confidence"]), upper))
        ]
        if not members:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0,
                    "ece_component": 0.0,
                }
            )
            continue
        accuracy = sum(bool(row["top1_correct"]) for row in members) / len(members)
        mean_confidence = sum(float(row["top1_confidence"]) for row in members) / len(members)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "ece_component": abs(accuracy - mean_confidence) * len(members) / total,
            }
        )
    return bins


def write_confusion_csv(path: Path, labels: list[str], confusion: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row.tolist()])


def write_confusion_png(path: Path, labels: list[str], confusion: np.ndarray) -> None:
    cell = 72
    margin_left = 180
    margin_top = 180
    size = (margin_left + cell * len(labels) + 40, margin_top + cell * len(labels) + 40)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max(int(confusion.max()), 1)

    for index, label in enumerate(labels):
        x = margin_left + index * cell + 4
        draw.text((x, 20), label[:16], fill="black", font=font)
        y = margin_top + index * cell + 24
        draw.text((10, y), label[:24], fill="black", font=font)

    for actual in range(len(labels)):
        for predicted in range(len(labels)):
            value = int(confusion[actual, predicted])
            intensity = 255 - int(210 * value / max_value)
            fill = (intensity, intensity, 255)
            x0 = margin_left + predicted * cell
            y0 = margin_top + actual * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=fill, outline=(180, 180, 180))
            draw.text((x0 + 8, y0 + 26), str(value), fill="black", font=font)

    draw.text((margin_left, 8), "Predicted", fill="black", font=font)
    draw.text((10, margin_top - 24), "Actual", fill="black", font=font)
    image.save(path)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    if not labels:
        raise SystemExit(f"No labels in {args.labels}")

    label_index = {normalize_name(label): i for i, label in enumerate(labels)}
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(path for path in args.test_dir.iterdir() if path.is_dir()):
        key = normalize_name(class_dir.name)
        if key not in label_index:
            raise SystemExit(f"Test class {class_dir.name!r} is not in labels.txt")
        files = image_files(class_dir)
        if args.max_images_per_class is not None:
            files = files[: args.max_images_per_class]
        samples.extend((path, label_index[key]) for path in files)
    if not samples:
        raise SystemExit(f"No test images found under {args.test_dir}")

    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expected_input = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
    if list(input_details["shape"]) != expected_input or input_details["dtype"] != np.float32:
        raise SystemExit(
            f"Input contract mismatch: {input_details['dtype']} "
            f"{input_details['shape']} != float32 {expected_input}"
        )
    if int(output_details["shape"][-1]) != len(labels):
        raise SystemExit("Output class count does not match labels.txt")

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int32)
    rows: list[dict[str, object]] = []
    total_latency_ms = 0.0
    for path, actual in samples:
        array = preprocess(path)
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_latency_ms += elapsed_ms
        scores = softmax_if_needed(output)
        ranking = np.argsort(scores)[::-1]
        predicted = int(ranking[0])
        top3 = [int(item) for item in ranking[:3]]
        confusion[actual, predicted] += 1
        rows.append(
            {
                "image": str(path),
                "actual": labels[actual],
                "top1": labels[predicted],
                "top1_confidence": float(scores[predicted]),
                "top1_correct": predicted == actual,
                "top3": "|".join(labels[item] for item in top3),
                "top3_correct": actual in top3,
                "latency_ms": elapsed_ms,
            }
        )

    total = len(rows)
    top1 = sum(bool(row["top1_correct"]) for row in rows) / total
    top3 = sum(bool(row["top3_correct"]) for row in rows) / total
    class_metrics = precision_recall_f1(confusion, labels)
    bins = calibration_bins(rows)
    ece = sum(bin_row["ece_component"] for bin_row in bins)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in class_metrics)
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", labels, confusion)
    write_confusion_png(args.output_dir / "confusion_matrix.png", labels, confusion)
    metrics = {
        "model": str(args.model),
        "labels": labels,
        "test_dir": str(args.test_dir),
        "sample_count": total,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mean_latency_ms_python_tflite": total_latency_ms / total,
        "min_recall": min(metric.recall for metric in class_metrics),
        "per_class": [asdict(metric) for metric in class_metrics],
        "calibration": {
            "expected_calibration_error": ece,
            "bins": bins,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-3 accuracy: {top3:.4f}")
    print(f"Minimum class recall: {metrics['min_recall']:.4f}")
    print(f"Mean Python TFLite latency: {metrics['mean_latency_ms_python_tflite']:.1f} ms")
    print(f"Calibration ECE: {ece:.4f}")
    print(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
