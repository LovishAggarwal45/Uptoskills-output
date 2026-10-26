# 🚗 Wrong-Way Driving & Lane Discipline Violation Detection

**AI-Powered Intelligent Traffic Monitoring & Violation Detection System**

The **Wrong-Way Driving & Lane Discipline Violation Detection** system is an AI-driven computer vision application built to monitor traffic, track vehicle trajectories, detect vehicles traveling against traffic flow, and identify lane straddling/discipline violations in real-time.

It integrates YOLOv8 vehicle detection with multi-object tracking, direction estimation algorithms, and a web application interface for easy analysis and reporting.

---

## ✨ Features

- 🚘 **Vehicle Detection & Tracking:** Detects vehicles (cars, buses, trucks, motorcycles) using YOLOv8 and tracks their movements across frames.
- 📐 **Trajectory & Direction Estimation:** Analyzes vehicle paths to determine motion vectors and heading directions.
- 🛣️ **Lane Segmentation & Direction Mapping:** Maps traffic lanes and enforces directional flow logic per lane.
- ⚠️ **Violation Logic & Persistence Threshold:** Detects wrong-way driving and lane straddling using persistence thresholding to minimize false alerts.
- 🎬 **Violation Clip Recorder:** Automatically cuts and saves short video clips when violations are detected.
- 📊 **CSV Data Logging:** Logs comprehensive violation details including vehicle type, lane ID, timestamp, and confidence score.
- 💻 **Web Application & Dashboard:** Interactive Flask web interface to upload traffic footage and review real-time violation analytics.

---

## 🤖 AI Models & Core Components

| Component | File / Model | Purpose |
| :--- | :--- | :--- |
| **Object Detection** | `yolov8n.pt` | Real-time vehicle detection |
| **Detection & Tracking** | `detector_tracker.py` | Integrates YOLO detection with object tracking algorithms |
| **Lane Management** | `lane_manager.py` | Handles lane segmentation and direction mapping |
| **Trajectory Estimation**| `trajectory_estimator.py` | Computes vehicle trajectory vectors and direction angles |
| **Violation Engine** | `violation_detector.py` | Evaluates persistent violation rules (Wrong-way & Lane discipline) |
| **Clip Recorder** | `clip_recorder.py` | Generates and exports video clips for flagged violations |
| **CSV Logger** | `csv_logger.py` | Records structured violation metadata into CSV format |

---

## 🔄 Workflow

```text
📷 Upload / Stream Traffic Video
               ↓
🤖 Vehicle Detection (YOLOv8) & Tracking
               ↓
📐 Trajectory Estimation & Lane Mapping
               ↓
⚠️ Violation Logic Evaluation (Wrong-Way / Lane Straddling)
               ↓
🎬 Record Violation Clips  +  📊 Log CSV Data
               ↓
🖥️ Interactive Web Dashboard & Report Generation
