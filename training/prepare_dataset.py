#!/usr/bin/env python3
"""Prepare a real-photo breed dataset for training and held-out evaluation.

The script expects a directory-per-class source dataset, validates class names
against the bundled app catalog, removes corrupt/tiny/near-duplicate images,
and writes:

    output/
      trainval/<Breed>/*.jpg
      test/<Breed>/*.jpg
      manifest.csv
      summary.json
      provenance.md

``trainval`` is intended for ``training/train.py``; the untouched ``test`` set
is intended for ``training/evaluate_tflite.py`` after export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".gif", ".bmp"}
DEFAULT_ALIAS = {
    # Common spelling drift in public datasets.
    "jaffrabadi": "Jaffarabadi",
    "nili ravi": "Nili-Ravi",
    "niliravi": "Nili-Ravi",
    "krishnavalley": "Krishna Valley",
    "red sindhi": "Red Sindhi",
    "redsindhi": "Red Sindhi",
    "khillar": "Khillari",
    "bhadwari": "Bhadawari",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        dest="source_dirs",
        required=True,
        help="Source dataset root. Repeat to merge multiple sources before splitting.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("app/src/main/assets/data/breed_mapping.csv"),
    )
    parser.add_argument("--source-name", default="unknown")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument(
        "--min-usable",
        type=int,
        default=150,
        help="Classes below this cleaned count are excluded from train/test output.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=300,
        help="Reported target count for production readiness; does not exclude by itself.",
    )
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=3,
        help="Maximum dHash Hamming distance treated as a near duplicate.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional catalog breed to include. Repeatable. Defaults to all matched breeds.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="SOURCE=CATALOG",
        help="Extra class-name alias. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def compact_name(value: str) -> str:
    return normalize_name(value).replace(" ", "")


def strip_category_prefix(value: str) -> str:
    normalized = value.replace("-", "_").strip("_")
    for prefix in ("Cattle_", "Buffalo_", "Cow_"):
        if normalized.lower().startswith(prefix.lower()):
            return normalized[len(prefix):]
    return normalized


def canonical_label(value: str) -> str:
    return strip_category_prefix(value).replace("_", " ").replace("-", " ").strip()


def read_catalog(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [row["breed_name"].strip() for row in reader]
    mapping: dict[str, str] = {}
    for name in names:
        mapping[normalize_name(name)] = name
        mapping[compact_name(name)] = name
    return mapping


def parse_aliases(values: Iterable[str]) -> dict[str, str]:
    aliases = {normalize_name(k): v for k, v in DEFAULT_ALIAS.items()}
    aliases.update({compact_name(k): v for k, v in DEFAULT_ALIAS.items()})
    for raw in values:
        if "=" not in raw:
            sys.exit(f"Invalid --alias {raw!r}; expected SOURCE=CATALOG")
        source, target = raw.split("=", 1)
        aliases[normalize_name(source)] = target.strip()
        aliases[compact_name(source)] = target.strip()
    return aliases


def resolve_label(raw: str, catalog: dict[str, str], aliases: dict[str, str]) -> str | None:
    label = canonical_label(raw)
    keys = [normalize_name(label), compact_name(label)]
    for key in keys:
        if key in aliases:
            return aliases[key]
        if key in catalog:
            return catalog[key]
    return None


def image_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def direct_image_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def discover_class_dirs(source_dir: Path) -> list[Path]:
    direct = [
        path
        for path in source_dir.iterdir()
        if path.is_dir() and direct_image_files(path)
    ]
    if direct:
        return sorted(direct, key=lambda p: p.name.lower())

    nested = [
        path
        for path in source_dir.rglob("*")
        if path.is_dir() and direct_image_files(path)
    ]
    return sorted(nested, key=lambda p: p.name.lower())


def dhash(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def stable_stem(source: Path, label: str) -> str:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return f"{label.replace(' ', '_').replace('-', '_')}_{digest}.jpg"


@dataclass
class ClassSummary:
    raw_class: str
    resolved_label: str | None
    raw_count: int = 0
    kept_count: int = 0
    corrupt_count: int = 0
    tiny_count: int = 0
    duplicate_count: int = 0
    included: bool = False
    trainval_count: int = 0
    test_count: int = 0
    decision: str = ""


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved == cwd or resolved == resolved.anchor:
        sys.exit(f"Refusing unsafe output directory: {resolved}")
    if path.exists():
        if not overwrite:
            sys.exit(f"Output directory already exists: {path} (pass --overwrite)")
        shutil.rmtree(path)
    (path / "trainval").mkdir(parents=True)
    (path / "test").mkdir(parents=True)


def save_jpeg(source: Path, destination: Path, min_short_side: int) -> tuple[int, int, int]:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if min(width, height) < min_short_side:
            raise ValueError("tiny")
        converted = image.convert("RGB")
        image_hash = dhash(converted)
        destination.parent.mkdir(parents=True, exist_ok=True)
        converted.save(destination, "JPEG", quality=92, optimize=True)
        return width, height, image_hash


def main() -> None:
    args = parse_args()
    for source_dir in args.source_dirs:
        if not source_dir.is_dir():
            sys.exit(f"Source directory not found: {source_dir}")
    if not args.catalog.is_file():
        sys.exit(f"Catalog not found: {args.catalog}")

    catalog = read_catalog(args.catalog)
    aliases = parse_aliases(args.alias)
    include = {normalize_name(name) for name in args.include}
    ensure_clean_output(args.output_dir, args.overwrite)

    rng = random.Random(args.seed)
    seen_hashes: list[tuple[int, str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    summaries: list[ClassSummary] = []
    kept_by_label: dict[str, list[dict[str, object]]] = {}

    for source_dir in args.source_dirs:
        for class_dir in discover_class_dirs(source_dir):
            files = image_files(class_dir)
            label = resolve_label(class_dir.name, catalog, aliases)
            summary = ClassSummary(
                raw_class=f"{source_dir.name}/{class_dir.name}",
                resolved_label=label,
                raw_count=len(files),
            )
            summaries.append(summary)

            if label is None:
                summary.decision = "excluded: no catalog label match"
                continue
            if include and normalize_name(label) not in include:
                summary.decision = "excluded: not in explicit include list"
                continue

            for source in sorted(files):
                row: dict[str, object] = {
                    "source": str(source),
                    "raw_class": summary.raw_class,
                    "resolved_label": label,
                    "status": "kept",
                    "reason": "",
                    "width": "",
                    "height": "",
                    "hash": "",
                    "duplicate_of": "",
                    "split": "",
                    "output": "",
                }
                try:
                    with Image.open(source) as probe:
                        probe = ImageOps.exif_transpose(probe)
                        width, height = probe.size
                        if min(width, height) < args.min_short_side:
                            raise ValueError("tiny")
                        image_hash = dhash(probe.convert("RGB"))
                except (UnidentifiedImageError, OSError):
                    summary.corrupt_count += 1
                    row["status"] = "dropped"
                    row["reason"] = "corrupt"
                    manifest_rows.append(row)
                    continue
                except ValueError:
                    summary.tiny_count += 1
                    row["status"] = "dropped"
                    row["reason"] = f"short side < {args.min_short_side}"
                    row["width"] = width
                    row["height"] = height
                    manifest_rows.append(row)
                    continue

                duplicate = next(
                    (
                        existing
                        for existing in seen_hashes
                        if hamming(image_hash, existing[0]) <= args.hash_distance
                    ),
                    None,
                )
                if duplicate is not None:
                    summary.duplicate_count += 1
                    row["status"] = "dropped"
                    row["reason"] = f"dHash distance <= {args.hash_distance}"
                    row["width"] = width
                    row["height"] = height
                    row["hash"] = f"{image_hash:016x}"
                    row["duplicate_of"] = duplicate[2]
                    manifest_rows.append(row)
                    continue

                seen_hashes.append((image_hash, label, str(source)))
                summary.kept_count += 1
                row["width"] = width
                row["height"] = height
                row["hash"] = f"{image_hash:016x}"
                kept_by_label.setdefault(label, []).append(row)

    included_labels: set[str] = set()
    included_classes: list[dict[str, object]] = []
    for label, rows in sorted(kept_by_label.items()):
        rng.shuffle(rows)
        if len(rows) < args.min_usable:
            for summary in summaries:
                if summary.resolved_label == label and not summary.decision:
                    summary.decision = f"excluded: cleaned count {len(rows)} < {args.min_usable}"
            manifest_rows.extend(rows)
            continue
        if len(rows) <= args.test_count:
            for summary in summaries:
                if summary.resolved_label == label and not summary.decision:
                    summary.decision = f"excluded: cleaned count {len(rows)} <= test-count {args.test_count}"
            manifest_rows.extend(rows)
            continue

        included_labels.add(label)
        decision = (
            "included"
            if len(rows) >= args.target_count
            else f"included below target {args.target_count}; report as limited data"
        )
        included_classes.append(
            {
                "label": label,
                "raw_count": sum(s.raw_count for s in summaries if s.resolved_label == label),
                "cleaned_count": len(rows),
                "trainval_count": len(rows) - args.test_count,
                "test_count": args.test_count,
                "decision": decision,
            }
        )
        for summary in summaries:
            if summary.resolved_label == label and not summary.decision:
                summary.included = True
                summary.decision = "contributed to included class"
        for index, row in enumerate(rows):
            split = "test" if index < args.test_count else "trainval"
            destination = (
                args.output_dir
                / split
                / label.replace(" ", "_").replace("-", "_")
                / stable_stem(Path(str(row["source"])), label)
            )
            save_jpeg(Path(str(row["source"])), destination, args.min_short_side)
            row["split"] = split
            row["output"] = str(destination)
            manifest_rows.append(row)

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source",
            "raw_class",
            "resolved_label",
            "status",
            "reason",
            "width",
            "height",
            "hash",
            "duplicate_of",
            "split",
            "output",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_json = {
        "source_name": args.source_name,
        "source_url": args.source_url,
        "license": args.license,
        "source_dirs": [str(path) for path in args.source_dirs],
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "min_short_side": args.min_short_side,
        "hash_distance": args.hash_distance,
        "test_count_per_class": args.test_count,
        "min_usable": args.min_usable,
        "target_count": args.target_count,
        "included_classes": included_classes,
        "classes": [asdict(summary) for summary in sorted(summaries, key=lambda s: s.raw_class)],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    excluded = [
        s for s in summaries
        if s.resolved_label not in included_labels and not s.included
    ]
    provenance = [
        f"# Dataset Preparation: {args.source_name}",
        "",
        f"- Source: {args.source_url or args.source_name}",
        f"- License: {args.license}",
        f"- Source directories: {', '.join(f'`{path}`' for path in args.source_dirs)}",
        f"- Output directory: `{args.output_dir}`",
        f"- Cleaning: corrupt/tiny images dropped; dHash near-duplicates dropped with distance <= {args.hash_distance}.",
        f"- Hold-out test split: {args.test_count} images per included class.",
        "",
        "## Included Classes",
        "",
        "| Label | Raw | Cleaned | Train/Val | Test | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for summary in sorted(included_classes, key=lambda s: str(s["label"])):
        provenance.append(
            f"| {summary['label']} | {summary['raw_count']} | {summary['cleaned_count']} | "
            f"{summary['trainval_count']} | {summary['test_count']} | {summary['decision']} |"
        )
    provenance.extend(["", "## Excluded Classes", "", "| Raw class | Resolved label | Raw | Cleaned | Decision |", "|---|---|---:|---:|---|"])
    for summary in sorted(excluded, key=lambda s: s.raw_class.lower()):
        provenance.append(
            f"| {summary.raw_class} | {summary.resolved_label or ''} | "
            f"{summary.raw_count} | {summary.kept_count} | {summary.decision} |"
        )
    (args.output_dir / "provenance.md").write_text(
        "\n".join(provenance) + "\n",
        encoding="utf-8",
    )

    print(f"Included {len(included_classes)} classes.")
    for summary in sorted(included_classes, key=lambda s: str(s["label"])):
        print(
            f"  {summary['label']}: cleaned={summary['cleaned_count']}, "
            f"trainval={summary['trainval_count']}, test={summary['test_count']}"
        )
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    print(f"Provenance: {args.output_dir / 'provenance.md'}")


if __name__ == "__main__":
    main()
