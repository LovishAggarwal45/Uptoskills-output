# SentinelGrid — Occlusion-Resistant 3D Crowd Counting

An end-to-end multi-camera crowd-counting and surveillance pipeline for synchronized, calibrated camera views.

SentinelGrid detects people in multiple camera feeds, projects detected foot-points onto a common ground plane using camera homographies, and fuses overlapping observations into a unified bird's-eye occupancy map.

The system records timestamped events, raises configurable capacity alerts, and exports event statistics as a CSV report.

## What is Included

| Requirement | Implementation |
|---|---|
| Overlapping calibrated CCTV feeds | Multi-image `/api/analyze` endpoint with per-camera homographies |
| Camera geometry | `/api/geometry` calculates the fundamental matrix and image-to-ground homographies from camera poses |
| Deep person detection | YOLO person detector fine-tuned on WILDTRACK |
| Unified ground-plane count | Confidence-prioritized spatial fusion across camera views |
| Alerts and event statistics | Configurable capacity limit with persistent `data/events.csv` |
| Visual result | Web dashboard with individual camera views, bird's-eye map, metrics, and CSV download |

## Project Structure

```text
SentinelGrid/
│
├── backend/
│   └── main.py
│
├── config/
│   ├── calibrations.example.json
│   └── calibrations.json
│
├── data/
│   └── events.csv
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── train_wildtrack_colab.ipynb
│
├── scripts/
│   └── calibration.py
│
├── requirements.txt
├── README.md
└── .gitignore