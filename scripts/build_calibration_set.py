"""
Build DJI quantization calibration set from training-v2 images.

Uses all 274 val images (unseen during training) + random sample from
train to reach target count. Output is a flat folder of 640x640 JPEGs
ready to zip and upload to DJI AI Training Platform.

Usage:
    python scripts/build_calibration_set.py [--count 500] [--output calibration/]
"""

import argparse
import random
import shutil
from pathlib import Path

DATASET = Path("D:/Projects/wildlife-census/training-v2/images")
VAL_DIR = DATASET / "val"
TRAIN_DIR = DATASET / "train"


def collect_images(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Target calibration image count")
    parser.add_argument("--output", default="D:/Projects/wildlife-census/calibration", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    val_images = collect_images(VAL_DIR)
    train_images = collect_images(TRAIN_DIR)

    selected = list(val_images)  # always include all val images
    needed = args.count - len(selected)

    if needed > 0:
        sampled = random.sample(train_images, min(needed, len(train_images)))
        selected += sampled

    selected = selected[:args.count]
    random.shuffle(selected)

    print(f"Val images:   {len(val_images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Calibration:  {len(selected)} images -> {out}")

    for i, src in enumerate(selected):
        dst = out / f"calib_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)

    print(f"Done. {len(selected)} images written to {out}")
    print(f"\nNext: zip for DJI upload")
    print(f"  cd D:/Projects/wildlife-census")
    print(f"  zip -r calibration.zip calibration/")


if __name__ == "__main__":
    main()
