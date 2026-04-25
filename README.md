# SAI Thermal Wildlife Detector

YOLOv8-based thermal infrared detection model for drone-based wildlife census operations, developed by [Sentinel Aerial Inspections](https://sentinelaerial.com) (Hampton Roads, VA).

Designed for real-time onboard inference on the DJI M4T + Manifold 3 via PSDK 3.x — detections are georeferenced and logged during flight, delivering a completed count report within minutes of landing.

## Performance

| Metric | Value |
|---|---|
| mAP50 | **98.2%** |
| mAP50-95 | 78.6% |
| Precision | 92.7% |
| Recall | 94.7% |
| Architecture | YOLOv8n |
| Input | 640 x 640 thermal (normalized to RGB) |
| Classes | 1 — `animal` (deer / ungulate) |

## Repository Structure

```
wildlife-census/
  training/            YOLOv8 v1 training config and dataset split
  training-v2/         v2 training config (BAMBI + AWIR, 1826 frames)
  training-v3/         v3 config and export script
  scripts/
    export_onnx.py     Export .pt → ONNX (opset 12, TRT-compatible)
    build_calibration_set.py   Build DJI quantization calibration set
  sai-census-app/      C++ onboard application (PSDK 3.x + TensorRT)
  proposal/            DJI algorithm submission proposal
  datasets/            Download instructions (data not committed)
```

## Training Data

- **BAMBI** — 1,826 thermal frames of European ungulates (benchmark dataset)
- **USGS AWIR** — 53 white-tailed deer thermal frames from mid-Atlantic US field surveys
- Split: 1,594 train / 274 val

## Export to ONNX

```bash
pip install ultralytics onnx onnxsim

python scripts/export_onnx.py \
  --model runs/detect/runs/wildlife-thermal-v2/weights/best.pt \
  --imgsz 640 \
  --opset 12 \
  --output models/
```

Output: `models/deer-thermal-v2.onnx` (44.7 MB, opset 12)

## Build Calibration Set

```bash
python scripts/build_calibration_set.py --count 500 --output calibration/
```

Copies 274 val images + 226 random train samples into a flat folder for DJI platform quantization.

## Onboard Application

The C++ PSDK 3.x application lives in a companion repository: **[sai-census-app](https://github.com/irong40/sai-census-app)**

- TensorRT FP16 inference pipeline
- PSDK thermal stream subscription
- Real-time georeferenced detection events
- DJI AI Box overlay and census widget
- Ground station count map via OcuSync

## Deployment Architecture

```
M4T Thermal Sensor
  → PSDK 3.x stream
  → Manifold 3 (Jetson Orin)
  → YOLOv8n ONNX → TensorRT FP16
  → Detection events (GeoJSON + thumbnails)
  → OcuSync relay → Ground station tablet (live count map)
```

## Use Case

State wildlife agencies, federal land managers, and ecological research institutions running drone-based population surveys. Target: white-tailed deer census in mid-Atlantic US temperate forest.

Current pipeline without onboard inference: 4-8 hours post-flight analyst review per mission.  
With onboard inference: count report available at landing.

## Operator

**Adam Pierce** — Founder, Sentinel Aerial Inspections  
FAA Part 107 | CISSP | CISA | Veteran-Owned  
Hampton Roads, Virginia  
info@faithandharmonyllc.com
