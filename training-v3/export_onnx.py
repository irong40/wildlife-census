"""
SAI Wildlife Census — Export best.pt to ONNX for Manifold 3 deployment.
Run after training: python export_onnx.py

Output: D:/Projects/wildlife-census/models/deer-thermal-v3.onnx
"""

from pathlib import Path
from ultralytics import YOLO
import shutil

BEST_PT  = Path("D:/Projects/wildlife-census/runs/deer-thermal-v3/weights/best.pt")
MODELS   = Path("D:/Projects/wildlife-census/models")


def export():
    if not BEST_PT.exists():
        raise FileNotFoundError(f"Train first: {BEST_PT}")

    model = YOLO(str(BEST_PT))
    out = model.export(
        format="onnx",
        imgsz=640,
        opset=12,           # TensorRT 8.5.2 on JetPack 5.1.3 requires opset <= 16; 12 is safe
        simplify=True,
        dynamic=False,      # fixed batch=1 for TensorRT engine build
        half=False,         # FP32 ONNX → TensorRT FP16 conversion done on-device via trtexec
    )

    MODELS.mkdir(exist_ok=True)
    dest = MODELS / "deer-thermal-v3.onnx"
    shutil.copy2(out, dest)
    print(f"Exported: {dest}")
    print(f"\nDeploy to Manifold 3:")
    print(f"  scp {dest} manifold:/opt/sai-census/models/")
    print(f"  trtexec --onnx=deer-thermal-v3.onnx --saveEngine=deer-thermal-v3.engine --fp16")


if __name__ == "__main__":
    export()
