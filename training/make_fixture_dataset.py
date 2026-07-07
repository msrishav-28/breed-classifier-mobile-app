"""Generates a synthetic-but-learnable image dataset for pipeline
verification. Each class has a distinct procedural appearance (base hue,
pattern type, pattern scale) with heavy per-image randomness, so a
classifier must genuinely learn to separate them — validating training
dynamics, export, and on-device inference mechanics end to end.

The class names reuse real breed labels ONLY so the app's catalog
integration can be exercised in tests; the resulting model recognises
procedural textures, not animals, and must never ship as the production
model."""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 256
PER_CLASS = 130
SEED = 20260704

# label -> (base RGB, pattern, pattern scale). Base colors are chosen to be
# well separated so even the tiny fixture CNN converges reliably.
CLASSES = {
    "Gir":        ((196, 128, 60), "spots",   26),
    "Sahiwal":    ((160, 96, 36),  "spots",   52),
    "Red_Sindhi": ((118, 30, 30),  "stripes", 22),
    "Tharparkar": ((212, 206, 198), "stripes", 44),
    "Ongole":     ((232, 208, 160), "rings",   36),
    "Murrah":     ((40, 38, 42),   "rings",   64),
    "Nili_Ravi":  ((70, 58, 112),  "grid",    30),
    "Surti":      ((88, 112, 66),  "grid",    56),
}


def jitter(rgb, amount):
    return tuple(
        max(0, min(255, c + random.randint(-amount, amount))) for c in rgb
    )


def draw_pattern(draw, pattern, scale, color):
    if pattern == "spots":
        for _ in range(int((SIZE / scale) ** 2 * 1.6)):
            x, y = random.uniform(0, SIZE), random.uniform(0, SIZE)
            r = random.uniform(scale * 0.25, scale * 0.55)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif pattern == "stripes":
        angle = random.uniform(-0.5, 0.5)
        step = scale
        for i in range(-SIZE, 2 * SIZE, step):
            offset = math.tan(angle) * SIZE
            draw.line([(i, 0), (i + offset, SIZE)], fill=color,
                      width=max(2, scale // 4))
    elif pattern == "rings":
        for _ in range(int((SIZE / scale) ** 2 * 1.2)):
            x, y = random.uniform(0, SIZE), random.uniform(0, SIZE)
            r = random.uniform(scale * 0.3, scale * 0.6)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color,
                         width=max(2, scale // 8))
    elif pattern == "grid":
        step = scale
        w = max(2, scale // 6)
        for i in range(0, SIZE, step):
            draw.line([(i, 0), (i, SIZE)], fill=color, width=w)
            draw.line([(0, i), (SIZE, i)], fill=color, width=w)


def make_image(base, pattern, scale):
    bg = jitter(base, 14)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)

    # Patterns are always darker than the background so class appearance
    # stays consistent; brightness of the darkening varies per image.
    contrast = -random.randint(45, 85)
    pattern_color = tuple(max(0, min(255, c + contrast)) for c in bg)
    draw_pattern(draw, pattern, scale + random.randint(-4, 4), pattern_color)

    img = img.rotate(random.uniform(-15, 15), expand=False,
                     fillcolor=jitter(base, 16))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.9)))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    for label, (base, pattern, scale) in CLASSES.items():
        class_dir = args.output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(PER_CLASS):
            make_image(base, pattern, scale).save(
                class_dir / f"{label.lower()}_{i:03d}.jpg", quality=88
            )
        print(f"{label}: {PER_CLASS} images")
    print(f"Dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
