# Wrong-Way Driving & Lane Discipline Violation Detection

An intelligent AI traffic monitoring system that detects:
- **Wrong-Way Driving** — vehicles moving against the traffic flow direction
- **Lane Discipline Violations** — vehicles straddling lane boundaries

Built with **YOLOv8 + ByteTrack** for robust multi-vehicle tracking, **Shapely** for precise lane geometry, and **OpenCV** for video I/O and annotation.

---

## Quick Start

### 1. Launch the Local Web Application (Recommended)
You can start the complete interactive web app locally with live streaming, controls, and video clip player:
```bash
python app.py
```
Or double-click `run_web.bat`. Then open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### 2. CLI: Run on your own video
```bash
python main.py --video your_traffic_video.mp4
```

### 3. Run on the included sample
```bash
python main.py --video sample_traffic.mp4
```

### 4. Run with live display
```bash
python main.py --video sample_traffic.mp4 --display
```

### 5. Generate Violation Audit Report & HTML Dashboard
```bash
python analyze_violations.py --csv output/violations_real.csv --report output/report.html
```

### 6. Run Test Suite
```bash
python tests/test_pipeline.py
```

---

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `sample_traffic.mp4` | Input video file / webcam index / RTSP stream |
| `--output-video` | `output/annotated_traffic.mp4` | Output annotated video |
| `--output-csv` | `output/violations.csv` | Violations log CSV |
| `--clips-dir` | `output/violation_clips/` | Per-violation clip directory |
| `--display` | off | Show live OpenCV window |
| `--max-frames` | none | Limit frames processed |
| `--skip-calib` | off | Skip auto-calibration (for live streams) |

---

## Output

| File | Description |
|------|-------------|
| `output/annotated_traffic.mp4` | Full video with lanes, trails, arrows, HUD & alerts |
| `output/violations.csv` | vehicle_type, lane, timestamp, confidence, deviation_deg |
| `output/violation_clips/clip_VIO-*.mp4` | 3-second clips per violation (pre+post roll) |

---

## System Architecture

```
Video → YOLO Detect → ByteTrack → Trajectory Smoothing
     → Lane Polygon Intersection
     → Heading vs Learned Flow Comparison   → Wrong-Way check
     → Multi-Lane Overlap Ratio             → Straddling check
     → Persistence Threshold Filter
     → Alert + Clip + CSV Log
```

---

## Project Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestrator with auto-calibration |
| `config.py` | All tunable parameters + lane definitions |
| `detector_tracker.py` | YOLOv8 + ByteTrack vehicle detection |
| `trajectory_estimator.py` | Smoothed heading estimation per vehicle |
| `lane_manager.py` | Lane polygon geometry + flow learning |
| `violation_detector.py` | Persistence-threshold violation logic |
| `clip_recorder.py` | Pre/post-roll violation clip recorder |
| `csv_logger.py` | Structured CSV violation logger |
| `visualizer.py` | HUD overlays, trails, flashing alerts |
| `analyze_violations.py` | Violation analytics summary & interactive HTML report generator |
| `tests/test_pipeline.py` | Automated unit & integration test suite |
| `generate_sample_video.py` | Synthetic test video generator |
| `requirements.txt` | Python dependencies |

---

## Adapting Lane Definitions

Edit `DEFAULT_LANES_720P` in `config.py` to match your camera layout:

```python
LaneDefinition(
    lane_id=1,
    name="Lane 1 (Left)",
    polygon=[(0, 0), (427, 0), (427, 720), (0, 720)],   # Shapely polygon
    expected_heading_deg=340.0,   # 270=North, 90=South, 0=East, 180=West
    allowed_angle_deviation_deg=110.0,
)
```

The auto-calibration phase will refine these headings automatically from the first 120 frames.

---

## Key Parameters (config.py)

```python
wrong_way_persistence_frames = 8    # frames before WW alert fires
wrong_way_angle_threshold    = 100  # degree deviation to classify as wrong-way
straddle_persistence_frames  = 15   # frames before straddle alert fires
straddle_overlap_min_ratio   = 0.15 # 15% overlap in 2nd lane = straddling
pre_roll_frames              = 45   # ~1.5s clip buffer before alert
post_roll_frames             = 45   # ~1.5s clip recording after alert
```
