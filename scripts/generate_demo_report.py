"""
SAI Thermal Wildlife Detector — Demo Report Generator

Runs the ONNX model against a sample of validation images, draws bounding boxes,
and produces a self-contained HTML report suitable for screen recording.

Usage:
    python scripts/generate_demo_report.py [--count 12] [--output demo/]
"""

import argparse
import base64
import json
import random
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

MODEL     = Path("D:/Projects/wildlife-census/models/deer-thermal-v2.onnx")
VAL_DIR   = Path("D:/Projects/wildlife-census/training-v2/images/val")
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
INPUT_SIZE  = 640

SAI_GOLD   = "#C9A84C"
SAI_DARK   = "#1A1A2E"
SAI_GREEN  = "#2ECC71"


def letterbox(img: np.ndarray, size: int = 640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def run_inference(session, img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    lb, scale, pad_x, pad_y = letterbox(img_rgb, INPUT_SIZE)
    inp = lb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]

    out = session.run(None, {"images": inp})[0][0]  # [5, 8400]
    cx, cy, bw, bh, conf = out[0], out[1], out[2], out[3], out[4]

    mask = conf >= CONF_THRESH
    if not mask.any():
        return [], img_rgb

    cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

    # decode back to original image coords
    x1 = ((cx - bw / 2) - pad_x) / scale
    y1 = ((cy - bh / 2) - pad_y) / scale
    x2 = ((cx + bw / 2) - pad_x) / scale
    y2 = ((cy + bh / 2) - pad_y) / scale

    x1 = np.clip(x1, 0, orig_w); x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h); y2 = np.clip(y2, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, conf, IOU_THRESH)
    detections = [{"box": boxes[k].tolist(), "conf": float(conf[k])} for k in keep]
    return detections, img_rgb


def annotate(img_rgb: np.ndarray, detections: list) -> np.ndarray:
    img = img_rgb.copy()
    h, w = img.shape[:2]
    box_color = (201, 168, 76)   # SAI gold in RGB
    txt_bg    = (26, 26, 46)

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        conf = d["conf"]
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        label = f"#{i+1}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), txt_bg, -1)
        cv2.putText(img, label, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # corner count badge
    badge = f"{len(detections)} detected"
    (bw2, bh2), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (w - bw2 - 16, h - bh2 - 16), (w - 4, h - 4), txt_bg, -1)
    cv2.putText(img, badge, (w - bw2 - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)
    return img


def img_to_b64(arr: np.ndarray, quality: int = 85) -> str:
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(results: list, mission_id: str, ts: str) -> str:
    total = sum(r["count"] for r in results)
    frames_with = sum(1 for r in results if r["count"] > 0)

    cards = ""
    for r in results:
        border = f"2px solid {SAI_GOLD}" if r["count"] > 0 else "2px solid #333"
        badge_color = SAI_GREEN if r["count"] > 0 else "#888"
        det_items = "".join(
            f'<li>#{i+1} — conf {d["conf"]:.1%}  '
            f'[{int(d["box"][0])},{int(d["box"][1])}→{int(d["box"][2])},{int(d["box"][3])}]</li>'
            for i, d in enumerate(r["detections"])
        ) or "<li style='color:#666'>No detections above threshold</li>"

        cards += f"""
        <div class="card">
          <div class="card-img-wrap" style="border:{border}">
            <img src="data:image/jpeg;base64,{r['b64']}" alt="frame">
            <span class="det-badge" style="background:{badge_color}">{r['count']} detected</span>
          </div>
          <div class="card-meta">
            <span class="frame-id">Frame {r['frame_id']:03d}</span>
            <ul class="det-list">{det_items}</ul>
          </div>
        </div>"""

    geojson_features = []
    for r in results:
        for d in r["detections"]:
            geojson_features.append({
                "type": "Feature",
                "properties": {"frame": r["frame_id"], "conf": round(d["conf"], 3), "class": "animal"},
                "geometry": {"type": "Point", "coordinates": [0, 0]}
            })
    geojson = json.dumps({"type": "FeatureCollection", "features": geojson_features}, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAI Thermal Wildlife Detector — Census Report</title>
<style>
  :root {{
    --gold: {SAI_GOLD};
    --dark: {SAI_DARK};
    --green: {SAI_GREEN};
    --bg: #0d0d1a;
    --card: #16162a;
    --border: #2a2a4a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; }}

  header {{
    background: var(--dark);
    border-bottom: 2px solid var(--gold);
    padding: 24px 40px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .logo {{ font-size: 1.4rem; font-weight: 700; color: var(--gold); letter-spacing: 0.05em; }}
  .logo span {{ color: #fff; }}
  .mission-id {{ font-size: 0.8rem; color: #888; text-align: right; }}
  .mission-id strong {{ color: #ccc; display: block; font-size: 0.95rem; }}

  .summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 32px 40px 0;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
  }}
  .stat-val {{ font-size: 2.4rem; font-weight: 700; color: var(--gold); line-height: 1; }}
  .stat-label {{ font-size: 0.78rem; color: #888; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }}

  .model-info {{
    margin: 24px 40px 0;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
  }}
  .mi {{ border-right: 1px solid var(--border); padding-right: 16px; }}
  .mi:last-child {{ border-right: none; }}
  .mi-val {{ font-size: 1rem; font-weight: 600; color: #fff; }}
  .mi-label {{ font-size: 0.72rem; color: #666; margin-top: 4px; text-transform: uppercase; }}

  .section-title {{
    padding: 32px 40px 16px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #666;
    border-bottom: 1px solid var(--border);
    margin: 0 40px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
    padding: 24px 40px;
  }}
  .card {{
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  .card-img-wrap {{ position: relative; }}
  .card-img-wrap img {{ width: 100%; display: block; }}
  .det-badge {{
    position: absolute; bottom: 8px; right: 8px;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; color: #fff;
  }}
  .card-meta {{ padding: 12px 16px; }}
  .frame-id {{ font-size: 0.75rem; color: var(--gold); font-weight: 600; display: block; margin-bottom: 8px; }}
  .det-list {{ list-style: none; font-size: 0.72rem; color: #999; line-height: 1.8; font-family: monospace; }}

  .report-section {{
    margin: 0 40px 40px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
  }}
  .report-section h3 {{ color: var(--gold); font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 16px; }}
  pre {{
    background: #0a0a14;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    font-size: 0.72rem;
    color: #8fa;
    overflow-x: auto;
    line-height: 1.6;
  }}

  footer {{
    border-top: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #555;
  }}
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">SENTINEL<span>AERIAL</span></div>
    <div style="font-size:0.75rem;color:#666;margin-top:4px;">
      Thermal Wildlife Detection System &mdash; SAI Thermal Wildlife Detector v2
    </div>
  </div>
  <div class="mission-id">
    <strong>{mission_id}</strong>
    {ts} UTC &nbsp;|&nbsp; DJI M4T + Manifold 3
  </div>
</header>

<div class="summary">
  <div class="stat">
    <div class="stat-val">{total}</div>
    <div class="stat-label">Total Animals Detected</div>
  </div>
  <div class="stat">
    <div class="stat-val">{len(results)}</div>
    <div class="stat-label">Frames Analyzed</div>
  </div>
  <div class="stat">
    <div class="stat-val">{frames_with}</div>
    <div class="stat-label">Frames with Detections</div>
  </div>
  <div class="stat">
    <div class="stat-val" style="color:var(--green)">98.2%</div>
    <div class="stat-label">Model mAP50 (val)</div>
  </div>
</div>

<div class="model-info">
  <div class="mi">
    <div class="mi-val">YOLOv8n</div>
    <div class="mi-label">Architecture</div>
  </div>
  <div class="mi">
    <div class="mi-val">ONNX opset 12</div>
    <div class="mi-label">Export Format</div>
  </div>
  <div class="mi">
    <div class="mi-val">640 &times; 640</div>
    <div class="mi-label">Input Size</div>
  </div>
  <div class="mi">
    <div class="mi-val">{int(CONF_THRESH*100)}%</div>
    <div class="mi-label">Conf Threshold</div>
  </div>
  <div class="mi">
    <div class="mi-val">BAMBI + AWIR</div>
    <div class="mi-label">Training Data</div>
  </div>
</div>

<div class="section-title">Detection Results — Annotated Frames</div>
<div class="grid">{cards}</div>

<div class="section-title" style="margin-top:0">GeoJSON Report Output</div>
<div class="report-section" style="margin-top:24px">
  <h3>Detection Export &mdash; GeoJSON (FeatureCollection)</h3>
  <pre>{geojson}</pre>
</div>

<footer>
  <span>Sentinel Aerial Inspections &mdash; Faith &amp; Harmony LLC &mdash; Hampton Roads, VA</span>
  <span>FAA Part 107 &nbsp;|&nbsp; CISSP &nbsp;|&nbsp; CISA &nbsp;|&nbsp; Veteran-Owned</span>
  <span>Generated {ts} UTC</span>
</footer>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12, help="Number of val images to process")
    parser.add_argument("--output", default="D:/Projects/wildlife-census/demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[SAI] Loading model: {MODEL}")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    print(f"[SAI] Model loaded. Input: {session.get_inputs()[0].shape}")

    images = sorted(VAL_DIR.glob("*.jpg")) + sorted(VAL_DIR.glob("*.png"))
    sample = random.sample(images, min(args.count, len(images)))

    results = []
    total = 0
    for i, img_path in enumerate(sample):
        detections, img_rgb = run_inference(session, img_path)
        annotated = annotate(img_rgb, detections)
        b64 = img_to_b64(annotated)
        results.append({
            "frame_id": i + 1,
            "filename": img_path.name,
            "count": len(detections),
            "detections": detections,
            "b64": b64,
        })
        total += len(detections)
        print(f"[SAI] Frame {i+1:02d}/{len(sample)}: {len(detections)} detection(s) — {img_path.name[:40]}")

    mission_id = f"SAI-CENSUS-{datetime.now().strftime('%Y%m%d-%H%M')}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    html = build_html(results, mission_id, ts)
    report_path = out / "census-report.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"\n[SAI] Report: {report_path}")
    print(f"[SAI] Total animals detected: {total} across {len(sample)} frames")
    print(f"[SAI] Open in browser and screen-record for DJI submission")


if __name__ == "__main__":
    main()
