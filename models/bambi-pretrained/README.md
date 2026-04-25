# BAMBI - UAV Thermal Wildlife Detection Model

**Source:** https://huggingface.co/cpraschl/bambi-thermal-detection
**License:** AGPL-3.0
**Downloaded:** 2026-03-31

## Model Overview

The BAMBI project uses camera drones with AI to automatically monitor wildlife.
This is a **YOLO11-based thermal object detection model** trained on airborne
thermal video data for wildlife detection.

## Detection Classes

**Single class, class-agnostic detection:**
- `0: animal` -- General wildlife detection (all mammalian species)

## Training Dataset

- **Total frames:** 19,252 thermal video frames from 225 videos
- **Train:** 15,730 images (~80%)
- **Validation:** 1,696 images (~10%)
- **Test:** 1,826 images (~10%)
- **Collection period:** April 2022 - March 2025
- **Total flights:** 400+

### Species in Training Data
- Red deer (Cervus elaphus)
- Wild boar (Sus scrofa)
- Roe deer (Capreolus capreolus)
- Fallow deer
- Chamois
- Alpine ibex
- Wolves

## Performance Metrics

| Metric     | Value  |
|------------|--------|
| mAP50      | ~0.97  |
| mAP50-95   | ~0.90  |
| Precision  | >0.93  |
| Recall     | ~0.92  |

## Usage

```python
from ultralytics import YOLO

# Load the pre-trained BAMBI model
model = YOLO("thermal_animal_detector.pt")

# Run inference on a thermal image
results = model("path/to/thermal_image.jpg")

# Run inference with confidence threshold
results = model("path/to/thermal_image.jpg", conf=0.5)
```

## Dataset Sources
- Zenodo: https://doi.org/10.5281/zenodo.15773102
- Roboflow: https://app.roboflow.com/dseducation/bambi-bounding_box-20250523/1

## Fine-Tuning Notes

This model serves as our production baseline for the Virginia DWR deer census
pipeline. Fine-tuning on our own thermal drone footage (white-tailed deer,
Chesapeake/VA habitat) should further improve accuracy for our specific use case.
