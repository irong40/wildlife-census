"""
SAI Wildlife Census — YOLOv8 Training v3
Dataset: 77,067 thermal deer images (73,923 train / 1,567 val / 1,577 test)
GPU: RTX 5070 (12.8 GB VRAM)

Run: python train.py
Resume: python train.py --resume
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATASET  = "D:/Projects/wildlife-census/training-v3/dataset.yaml"
RUNS_DIR = "D:/Projects/wildlife-census/runs"
RUN_NAME = "deer-thermal-v3"
WEIGHTS  = "yolov8s.pt"


def train(resume: bool = False):
    if resume:
        last = Path(RUNS_DIR) / RUN_NAME / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {last}")
        model = YOLO(str(last))
        print(f"Resuming from {last}")
    else:
        model = YOLO(WEIGHTS)
        print(f"Starting fresh from {WEIGHTS}")

    results = model.train(
        data=DATASET,
        epochs=100,
        imgsz=640,
        batch=32,               # RTX 5070 12.8 GB — fits comfortably at YOLOv8s/640
        device=0,
        workers=8,
        patience=15,            # early stop if no mAP50 improvement for 15 epochs
        save=True,
        save_period=10,
        project=RUNS_DIR,
        name=RUN_NAME,
        exist_ok=resume,
        pretrained=True,
        resume=resume,

        # Augmentation tuned for overhead thermal imagery
        hsv_h=0.005,            # minimal hue — thermal is near-grayscale
        hsv_s=0.2,
        hsv_v=0.4,              # brightness variance covers hot/cold ambient temps
        degrees=180,            # full rotation — drone can approach from any angle
        translate=0.1,
        scale=0.5,              # altitude variance 50-400 ft
        fliplr=0.5,
        flipud=0.5,             # overhead — no canonical up
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Logging
        plots=True,
        val=True,
        verbose=True,
    )

    best = Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best model : {best}")
    print(f"mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'see results/'):.4f}")
    print(f"\nNext step  : python export_onnx.py")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
