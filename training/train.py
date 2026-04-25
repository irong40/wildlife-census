"""
SAI Wildlife Census — YOLO Training Script
Trains YOLOv8 on thermal deer detection data.
Run: python train.py
"""
from ultralytics import YOLO

# Start from pre-trained YOLOv8s (transfer learning)
model = YOLO("yolov8s.pt")

# Train
results = model.train(
    data="D:/Projects/wildlife-census/training/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,           # RTX 5070
    patience=20,         # early stopping if no improvement for 20 epochs
    save=True,
    save_period=25,      # checkpoint every 25 epochs
    project="D:/Projects/wildlife-census/runs",
    name="deer-thermal-v1",
    pretrained=True,
    augment=True,
    # Augmentation tuned for aerial thermal
    hsv_h=0.01,          # minimal hue shift (thermal is grayscale-ish)
    hsv_s=0.3,
    hsv_v=0.3,
    degrees=15,          # slight rotation (drone angle variance)
    translate=0.1,
    scale=0.3,           # scale augmentation (altitude variance)
    fliplr=0.5,
    flipud=0.2,          # vertical flip (overhead perspective)
    mosaic=0.8,
)

print(f"\nTraining complete!")
print(f"Best model: D:/Projects/wildlife-census/runs/deer-thermal-v1/weights/best.pt")
