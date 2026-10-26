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
---

## 🧠 Technologies & Tools

 * Language: Python 3.x
 * Computer Vision & AI: OpenCV, Ultralytics YOLOv8, PyTorch
 * Data Processing: NumPy, Pandas
 * Web Framework: Flask, HTML5/CSS3
 * Visualization: Matplotlib / OpenCV Drawing Utilities
---

## 📁 Project Structure

Wrong-Way-Driving-and-Lane-Discipline-Violation-Detection/
│
├── app.py                      # Flask web application entry point
├── main.py                     # CLI execution script
├── config.py                   # Global configuration & parameters
│
├── detector_tracker.py         # Object detection and tracking module
├── lane_manager.py             # Lane segmentation and flow direction map
├── trajectory_estimator.py     # Trajectory and direction estimation
├── violation_detector.py       # Core violation detection logic
├── clip_recorder.py            # Video clip recorder for violation events
├── csv_logger.py               # CSV logging utility
├── visualizer.py               # Video annotation & visualization helpers
├── analyze_violations.py       # Post-processing analytics script
│
├── yolov8n.pt                  # YOLOv8 target model weights
├── requirements.txt            # Python dependencies
├── run_web.bat                 # Windows batch execution script
├── run_web.ps1                 # PowerShell execution script
│
├── output/                     # Directory for generated results
│   ├── violation_clips/        # Saved video clips of detected violations
│   ├── annotated_traffic.mp4   # Annotated full output video
│   ├── report.html             # Generated HTML summary report
│   └── violations.csv          # Exported CSV log files
│
├── templates/                  # Web dashboard HTML templates
│   └── index.html
│
├── tests/                      # Automated test scripts
│   └── test_pipeline.py
│
└── uploads/                    # Uploaded input video files
---

## ▶️ Run Locally

1. Clone the repository
git clone [https://github.com/pruthijeetamlan-ai/Wrong-Way-Driving-and-Lane-Discipline-Violation-Detection.git](https://github.com/pruthijeetamlan-ai/Wrong-Way-Driving-and-Lane-Discipline-Violation-Detection.git)
cd Wrong-Way-Driving-and-Lane-Discipline-Violation-Detection

2. Install dependencies
pip install -r requirements.txt

3. Run via CLI
python main.py

4. Run Web Application
python app.py

Alternatively on Windows, double-click run_web.bat or execute ./run_web.ps1 in PowerShell.
Open http://localhost:5000 in your browser to view the web dashboard.
---

## 🚀 Future Enhancements

 * 📹 Live RTSP Camera Stream Integration: Support for direct IP camera feed streaming.
 * 🚨 Real-Time Notification System: Instant email or Webhook alerts upon violation detection.
 * 🌙 Night Vision & Adverse Weather Optimization: Fine-tuned detection for low-light conditions.
 * 🚘 License Plate Recognition (ANPR): Automated vehicle plate identification during violation recording.
---
   
## 👩‍💻 Developed By

Amlan Pruthijeet

Internship(Outskill)

Task T230 - Wrong-Way Driving & Lane Discipline Violation Detection System

Domain: Artificial Intelligence & Computer Vision
---
