"""Render a single PDF showing each YOLO data augmentation.

Each panel applies one transform to a sample image at YOLO's default magnitude
so you can see what the parameter does before tuning it during training. The
output is one multi-image grid saved as a PDF.

Run from the project root after capturing frames:
    python scripts/visualize_augmentations.py

The script needs at least 4 images (mosaic uses 4, mixup uses 2) in the input
directory.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Each entry: (label_for_panel, transform_key, magnitude_used_for_demo).
# The magnitudes here are picked to be visually clear, not the YOLO defaults.
# Real YOLO training applies these as random magnitudes within bounds, so a
# value like hsv_h=0.015 produces subtle variation each step. The demo uses
# the maximum magnitude so students can see what each knob actually does.
AUGMENTATIONS = [
    ("original",         None,          None),
    ("hsv_h = 0.5",      "hsv_h",       0.5),
    ("hsv_s = 0.7",      "hsv_s",       0.7),
    ("hsv_v = 0.4",      "hsv_v",       0.4),
    ("degrees = 30",     "degrees",     30.0),
    ("translate = 0.2",  "translate",   0.2),
    ("scale = 0.5",      "scale",       0.5),
    ("shear = 15",       "shear",       15.0),
    ("perspective = 0.001", "perspective", 0.001),
    ("flipud = 1.0",     "flipud",      1.0),
    ("fliplr = 1.0",     "fliplr",      1.0),
    ("mosaic = 1.0",     "mosaic",      1.0),
    ("mixup = 1.0",      "mixup",       1.0),
]


def apply_hsv(img: np.ndarray, h_gain: float = 0, s_gain: float = 0, v_gain: float = 0) -> np.ndarray:
    """Shift hue/saturation/value to mimic YOLO's HSV augmentation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + int(180 * h_gain)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + s_gain), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + v_gain), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_affine(
    img: np.ndarray,
    degrees: float = 0,
    translate: float = 0,
    scale: float = 1.0,
    shear: float = 0,
    perspective: float = 0,
) -> np.ndarray:
    """Apply rotate, translate, scale, shear, perspective transforms in sequence.

    Each step is a separate matrix so the code stays readable. YOLO composes
    these into a single matrix internally, but the visual result is similar.
    """
    h, w = img.shape[:2]
    border = (114, 114, 114)  # YOLO's default gray fill

    # Rotate + scale around the center, then add translation.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, scale)
    M[0, 2] += translate * w
    M[1, 2] += translate * h
    out = cv2.warpAffine(img, M, (w, h), borderValue=border)

    if shear:
        rad = math.radians(shear)
        shear_M = np.array([[1, math.tan(rad), 0],
                            [math.tan(rad), 1, 0]], dtype=np.float32)
        out = cv2.warpAffine(out, shear_M, (w, h), borderValue=border)

    if perspective:
        persp_M = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [perspective, perspective, 1]], dtype=np.float32)
        out = cv2.warpPerspective(out, persp_M, (w, h), borderValue=border)

    return out


def apply_mosaic(images: list[np.ndarray], target_size: int = 640) -> np.ndarray:
    """Combine 4 images into a single grid, resizing each to a quadrant."""
    s = target_size // 2
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    positions = [(0, 0), (s, 0), (0, s), (s, s)]
    for img, (x, y) in zip(images[:4], positions):
        canvas[y:y + s, x:x + s] = cv2.resize(img, (s, s))
    return canvas


def apply_mixup(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend two images. YOLO uses Beta-distribution alpha; 0.5 is a clear demo value."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.addWeighted(img1, alpha, img2_resized, 1 - alpha, 0)


def load_images(image_dir: Path) -> list[np.ndarray]:
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) < 4:
        raise RuntimeError(
            f"Need at least 4 images in {image_dir} (found {len(paths)}). "
            "Mosaic uses 4 images and mixup uses 2."
        )
    return [cv2.imread(str(p)) for p in paths[:5]]


def render_panel(
    base: np.ndarray,
    extras: list[np.ndarray],
    key: str | None,
    value: float | None,
) -> np.ndarray:
    if key is None:
        return base
    if key == "hsv_h":
        return apply_hsv(base, h_gain=value)
    if key == "hsv_s":
        return apply_hsv(base, s_gain=value)
    if key == "hsv_v":
        return apply_hsv(base, v_gain=value)
    if key == "degrees":
        return apply_affine(base, degrees=value)
    if key == "translate":
        return apply_affine(base, translate=value)
    if key == "scale":
        return apply_affine(base, scale=value)
    if key == "shear":
        return apply_affine(base, shear=value)
    if key == "perspective":
        return apply_affine(base, perspective=value)
    if key == "flipud":
        return cv2.flip(base, 0)
    if key == "fliplr":
        return cv2.flip(base, 1)
    if key == "mosaic":
        return apply_mosaic([base] + extras[:3])
    if key == "mixup":
        return apply_mixup(base, extras[0])
    raise ValueError(f"Unknown augmentation key: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="data/captured",
                        help="Directory of source images")
    parser.add_argument("--output", default="augmentations.pdf",
                        help="Output PDF file path")
    parser.add_argument("--cols", type=int, default=3,
                        help="Number of grid columns")
    args = parser.parse_args()

    images = load_images(Path(args.image_dir))
    base = images[0]
    extras = images[1:]

    panels = [(label, render_panel(base, extras, key, value))
              for label, key, value in AUGMENTATIONS]

    rows = math.ceil(len(panels) / args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (label, img) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("YOLO Data Augmentations", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
