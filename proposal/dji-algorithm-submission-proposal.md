# DJI AI Algorithm Submission — Solution Proposal
## Thermal Wildlife Detection for Drone-Based Census Operations

**Submitted by:** Adam Pierce, Founder — CISSP, CISA, FAA Part 107  
**Organization:** Sentinel Aerial Inspections / Faith & Harmony LLC  
**Location:** Hampton Roads, Virginia  
**Contact:** info@faithandharmonyllc.com | (757) 843-8772  
**Date:** April 2026

---

## 1. Algorithm Overview

**Algorithm Name:** SAI Thermal Wildlife Detector v2  
**Algorithm Type:** Infrared Detection Model (Onboard / Manifold Algorithm)  
**Supported Platform:** DJI Matrice 4 Series (primary: M4T)  
**Industry:** Forestry / Agriculture  
**Deployment Mode:** Onboard — real-time edge inference on Manifold 3

This algorithm detects wildlife (deer/ungulates) in thermal infrared imagery captured by the DJI M4T's integrated radiometric sensor. It is designed for drone-based population census operations: the model runs onboard during flight, geotagging each detection in real time and delivering a completed count report within minutes of landing rather than days.

---

## 2. Problem Statement

Traditional drone wildlife census workflows require a post-flight analyst to manually review hundreds or thousands of thermal frames and count heat signatures by eye. On a county-scale white-tailed deer survey, this review takes 4-8 hours per flight. For state wildlife agencies and ecological research organizations, this labor cost is the primary constraint on survey frequency and geographic scope.

The bottleneck is not flight capacity — it is post-flight processing.

An onboard thermal detection model running at the airframe eliminates that bottleneck. Detections are georeferenced, counted, and logged during the flight. The deliverable is available when the aircraft lands.

---

## 3. Technical Specifications

| Parameter | Value |
|---|---|
| Model architecture | YOLOv8n (nano) |
| Input format | RGB-normalized thermal frames, 640 x 640 px |
| ONNX opset | 12 |
| Input tensor | [1, 3, 640, 640] |
| Output tensor | [1, 5, 8400] — (x, y, w, h, confidence) x 8400 anchors |
| Classes | 1 — animal (deer / ungulate) |
| Model size (ONNX) | 44.7 MB |
| Training framework | Ultralytics YOLOv8 8.4.33, PyTorch 2.10 |
| Training hardware | NVIDIA RTX 5060 Ti (CUDA 12.8) |

**Training dataset:**
- BAMBI (Benchmark Animal dataset for Machine-learning Benchmarking in Infrared) — 1,826 thermal frames from European ungulate surveys
- USGS Aerial Wildlife Image Repository (AWIR) — 53 white-tailed deer thermal frames from mid-Atlantic US
- Combined split: 1,594 train / 274 val images
- Single-class detection (all ungulates labeled as `animal` for maximum cross-species coverage)

**Training configuration:**
- Transfer learning from v1 deer-thermal checkpoint
- 100 epochs, batch 16, image size 640, AMP enabled
- Augmentation: mosaic 0.8, mixup 0.1, horizontal/vertical flip, HSV jitter, rotation ±15°, erasing 0.4

---

## 4. Performance Metrics

Evaluated on held-out validation split (274 images, not seen during training):

| Metric | Value |
|---|---|
| mAP50 | **98.2%** |
| mAP50-95 | 78.6% |
| Precision | 92.7% |
| Recall | 94.7% |
| Best epoch | 48 / 100 |

These results represent a production-ready detector. The 98.2% mAP50 reflects detection of animals in varied thermal contrast conditions including partial canopy occlusion, group clustering, and mixed-temperature backgrounds. Training on the BAMBI dataset — which includes solar-heated terrain, warm rock outcroppings, and sun-baked ground cover — specifically reduces false positives from non-biological heat sources, a known challenge in forested thermal survey environments.

---

## 5. Use Case — Drone-Based Wildlife Census

**Operator:** Sentinel Aerial Inspections (SAI), Hampton Roads VA  
**Aircraft:** DJI M4T (integrated 640 x 512 thermal + RGB)  
**Compute target:** DJI Manifold 3 (NVIDIA Jetson Orin-class, onboard via PSDK 3.x)

**Operational scenario:**

1. Operator programs a transect or grid mission in DJI Pilot 2 / FlightHub 2
2. Aircraft launches and begins the survey pattern
3. Manifold 3 subscribes to the M4T thermal stream via PSDK and runs the detection model at up to 30 fps (pending final on-device benchmark; Manifold 3 unit arriving late April 2026)
4. Each detection fires a georeferenced event: species label, bounding box, GPS coordinates, UTC timestamp, confidence score
5. Running count and detection map update on the ground-station tablet in near real time
6. On detection of a high-density cluster above a configurable threshold, the autonomy framework triggers an orbit branch for a confirmation count
7. At mission end, a GeoJSON of all detections and a summary count report are available immediately — no post-flight analyst review required for routine surveys

**Target customers:**
- State wildlife agencies (Virginia DWR deer census program — contact initiated)
- Ecological research institutions and universities
- Agricultural operations managing deer pressure on crops
- Federal land managers (USFS, NPS, USFWS) — population monitoring

**Deployment geography:** Mid-Atlantic US (Virginia, North Carolina, Maryland); model generalizes to any temperate-forest ungulate population

---

## 6. Differentiators

**Edge inference, no cloud required.** Raw thermal imagery never leaves the airframe. Only inferred outputs (counts, GeoJSON, annotated thumbnails) are transmitted or stored. This is critical for agencies with data-sovereignty requirements and for classified-adjacent sites where cloud transit of aerial imagery is not permitted.

**Operator credentials.** SAI is operated by a CISSP/CISA/Part 107-certified principal — a combination that enables deployment in law-enforcement-adjacent, corrections, and federal engagements where most independent operators cannot qualify.

**Proven training data.** BAMBI is a peer-reviewed benchmark dataset. AWIR is USGS-sourced. The v2 model is not trained on synthetic or augmented-only data — it is trained on real thermal aerial wildlife imagery from field deployments.

**Scalable service model.** At current pricing ($5K-$15K per county-scale engagement), reclaiming 30-40% post-flight labor through onboard inference improves margin by $1.5K-$6K per contract and makes multi-county, recurring-annual survey contracts tractable.

---

## 7. Deployment Architecture

```
M4T Thermal Sensor
       |
   PSDK 3.x stream
       |
   Manifold 3 (Jetson Orin)
       |
   YOLOv8n ONNX → TensorRT FP16 engine
       |
   Detection events (GeoJSON + thumbnails)
       |
   OcuSync / 4G relay
       |
   Ground station tablet (live count map)
       +
   SD card (encrypted raw frames, optional)
```

The ONNX file exported at opset 12 is converted on-device using `trtexec --fp16` to a hardware-specific TensorRT engine on the Manifold 3. The C++ inference application is implemented using DJI PSDK 3.x and exposes a census widget overlay on the DJI RC Pro display.

**Hardware verification:** Manifold 3 unit arriving ~April 30, 2026. Screen recording and live inference demo will be completed before the May 10 submission deadline.

---

## 8. Files Submitted

| File | Description |
|---|---|
| `deer-thermal-v2.onnx` | Exported detection model, opset 12, 44.7 MB |
| `calibration.zip` | 500 calibration images for DJI platform quantization (274 val + 226 train samples) |
| This document | Solution proposal |

---

## 9. Requested Support from DJI

SAI is submitting this algorithm as a working system, not a concept. The following support from DJI's engineering and ecosystem teams would accelerate deployment:

- **Quantization guidance** — confirmation that the submitted ONNX (opset 12, FP32) and 500-image calibration set meet DJI AI Training Platform requirements for INT8/FP16 quantization, or feedback on any format adjustments needed
- **PSDK thermal stream specs** — clarification on the M4T radiometric stream frame format (16-bit vs. normalized 8-bit) as delivered to a Manifold-hosted PSDK application, to ensure the preprocessing pipeline matches training-time normalization
- **Algorithm listing** — if the model is accepted, inclusion in the DJI AI algorithm marketplace to support discovery by state wildlife agencies and ecological research institutions

---

## 10. Contact & Organization

**Adam Pierce** — Founder, Sentinel Aerial Inspections  
Faith & Harmony LLC | Hampton Roads, Virginia  
FAA Part 107 | CISSP | CISA | Veteran-Owned  
info@faithandharmonyllc.com | (757) 843-8772  
sentinelaerial.com

*Skywatch.AI aviation liability policy active (Policy #41066191-00, $1M coverage, effective 2026-04-15)*
