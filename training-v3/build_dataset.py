"""
SAI Wildlife Census — Training Dataset Builder v3
Merges thermal-only datasets into a unified YOLOv8 training set.

Included sources:
  bambi       — 19,252 thermal deer frames (red deer, roe deer, boar)
  awir-deer   — 53 thermal white-tailed deer (AWIR *_R.JPG only)
  thermal-deer      — 322 thermal deer images
  thermal-drone-deer-fox — 778 thermal drone (deer class only, person dropped)
  roe-deer    — 361 thermal roe deer

All classes remapped to: 0 = deer

Output: D:/Projects/wildlife-census/training-v3/
  images/{train,val,test}/
  labels/{train,val,test}/
  dataset.yaml
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

SRC  = Path("D:/Projects/wildlife-census/datasets")
DEST = Path("D:/Projects/wildlife-census/training-v3")

# ---------------------------------------------------------------------------
# Source definitions
# Each entry: (name, images_dir, labels_dir, split, deer_class_ids)
#   split = "pre"  → already has train/valid/test subfolders
#   split = "flat" → single flat folder, will be split 80/10/10
#   deer_class_ids = set of class IDs in this dataset that map to deer (0)
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "bambi",
        "base": SRC / "bambi" / "extracted",
        "split": "pre",
        "val_dir": "valid",
        "deer_ids": {0},
    },
    {
        "name": "awir",
        "base": SRC / "awir" / "Deer",
        "split": "flat",
        "deer_ids": {1},
        "filter": lambda p: p.name.endswith("_R.JPG"),
    },
    {
        "name": "td",
        "base": SRC / "thermal-deer",
        "split": "pre",
        "val_dir": "valid",
        "deer_ids": {0},
    },
    {
        "name": "tdf",
        "base": SRC / "thermal-drone-deer-fox",
        "split": "pre",
        "val_dir": "valid",
        "deer_ids": {0},
    },
    {
        "name": "rd",
        "base": SRC / "roe-deer",
        "split": "pre",
        "val_dir": "valid",
        "deer_ids": {0},
    },
]


def remap_label(src_label: Path, dst_label: Path, deer_ids: set) -> bool:
    """Read YOLO label, keep only deer boxes remapped to class 0. Returns True if any boxes kept."""
    if not src_label.exists():
        return False
    lines = src_label.read_text().strip().splitlines()
    kept = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        if cls in deer_ids:
            kept.append("0 " + " ".join(parts[1:]))
    if kept:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(kept) + "\n")
        return True
    return False


def copy_pair(img_src: Path, lbl_src: Path, split: str, prefix: str, deer_ids: set) -> bool:
    """Copy image+label to destination split, remapping classes. Returns True on success."""
    img_ext = img_src.suffix
    stem = f"{prefix}_{img_src.stem}"

    dst_img = DEST / "images" / split / (stem + img_ext)
    dst_lbl = DEST / "labels" / split / (stem + ".txt")

    if dst_img.exists():
        return True

    if not remap_label(lbl_src, dst_lbl, deer_ids):
        return False

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, dst_img)
    return True


def process_presplit(src: dict):
    """Process a dataset that already has train/val/test splits."""
    base = src["base"]
    prefix = src["name"]
    deer_ids = src["deer_ids"]
    val_dir = src.get("val_dir", "valid")

    split_map = {"train": "train", val_dir: "val", "test": "test"}

    copied = skipped = 0
    for src_split, dst_split in split_map.items():
        img_dir = base / src_split / "images"
        lbl_dir = base / src_split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if copy_pair(img_path, lbl_path, dst_split, prefix, deer_ids):
                copied += 1
            else:
                skipped += 1
    return copied, skipped


def process_flat(src: dict):
    """Process a flat dataset (no pre-split), split 80/10/10."""
    base = src["base"]
    prefix = src["name"]
    deer_ids = src["deer_ids"]
    filter_fn = src.get("filter", None)

    all_imgs = sorted(p for p in base.iterdir() if p.suffix.upper() in {".JPG", ".JPEG", ".PNG"})
    if filter_fn:
        all_imgs = [p for p in all_imgs if filter_fn(p)]

    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    n_train = n - n_val - n_test

    splits = (
        [("train", p) for p in all_imgs[:n_train]] +
        [("val",   p) for p in all_imgs[n_train:n_train + n_val]] +
        [("test",  p) for p in all_imgs[n_train + n_val:]]
    )

    lbl_dir = base
    copied = skipped = 0
    for dst_split, img_path in splits:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if copy_pair(img_path, lbl_path, dst_split, prefix, deer_ids):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def write_yaml(counts: dict):
    train_count = counts.get("train", 0)
    val_count   = counts.get("val", 0)
    test_count  = counts.get("test", 0)

    yaml = f"""# SAI Wildlife Census — Thermal Deer Detection v3
# Built by build_dataset.py — do not edit manually
# Total: {train_count} train / {val_count} val / {test_count} test

path: D:/Projects/wildlife-census/training-v3
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: deer
"""
    (DEST / "dataset.yaml").write_text(yaml)
    print(f"Wrote dataset.yaml  ({train_count} train / {val_count} val / {test_count} test)")


def count_split(split: str) -> int:
    d = DEST / "images" / split
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main():
    print("=" * 60)
    print("SAI Wildlife Census — Dataset Builder v3")
    print(f"Output: {DEST}")
    print("=" * 60)

    for split in ("train", "val", "test"):
        (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_copied = total_skipped = 0

    for src in SOURCES:
        print(f"\n[{src['name']}] Processing {src['base'].name} ...")
        if src["split"] == "pre":
            c, s = process_presplit(src)
        else:
            c, s = process_flat(src)
        print(f"  copied={c}  skipped(no-deer-box)={s}")
        total_copied += c
        total_skipped += s

    counts = {sp: count_split(sp) for sp in ("train", "val", "test")}
    write_yaml(counts)

    print(f"\n{'=' * 60}")
    print(f"Done.  Total copied={total_copied}  skipped={total_skipped}")
    print(f"  train : {counts['train']:,}")
    print(f"  val   : {counts['val']:,}")
    print(f"  test  : {counts['test']:,}")
    print(f"  TOTAL : {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
