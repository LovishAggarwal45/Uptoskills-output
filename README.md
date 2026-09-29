# Self-Supervised Video Masked Autoencoder for Crowd Outlier Localization

An AI-driven crowd surveillance system designed to detect and localize spatio-temporal crowd anomalies (e.g., panic dispersion, counter-flow movement, sudden gatherings, local disturbances) in continuous wide-angle video feeds using a **Self-Supervised Video Masked Autoencoder (VideoMAE)**.

---

## 🌟 Key Features

1. **3D Spatio-Temporal Tubelet Extraction**: Converts video sequences $(C \times T \times H \times W)$ into 3D tube patches with 3D sine-cosine positional encodings.
2. **Self-Supervised VideoMAE Architecture**:
   - Trains exclusively on **normal crowd dynamics** (75% tubelet masking ratio).
   - Lightweight Vision Transformer (ViT) Encoder-Decoder architecture optimized for CPU and low-latency hardware execution.
3. **Spatio-Temporal Outlier Localization**:
   - Calculates patch-level reconstruction Mean Squared Error (MSE) between masked inputs and context predictions.
   - Interpolates spatio-temporal MSE values to generate continuous pixel-level heatmaps.
4. **Real-Time Heatmap Video Overlay & Alerting**:
   - Overlays TURBO colormap heatmaps with dynamic transparency ($\alpha$-blending).
   - Draws bounding boxes around peak anomalous zones.
   - Includes HUD status banner, timestamping, frame tracking, and dynamic risk gauge.
5. **Automated Event Logging & Analytics**:
   - Generates structured CSV logs (`anomaly_events.csv`): `timestamp`, `frame_idx`, `location_bbox`, `detected_condition`, `confidence_score`, `risk_level`.
   - Generates visual analytics plots: `anomaly_timeline.png` (reconstruction error vs. time) and `spatial_heatmap_summary.png` (cumulative spatial risk distribution).

---

## 📁 Repository Structure

```
crowd_anomaly_videomae/
├── config.yaml                     # System, model, and anomaly threshold configurations
├── requirements.txt                # Dependency specifications
├── main.py                         # Unified CLI pipeline entry point
├── trainer.py                      # VideoMAE training routine on normal crowd video
├── test_pipeline.py                # End-to-end integration and unit tests
│
├── models/
│   ├── __init__.py
│   └── videomae.py                 # 3D Patch embedding, 3D PosEncoding, ViT Encoder/Decoder
│
├── dataset/
│   ├── __init__.py
│   ├── video_dataset.py            # Video loader & sliding-window clip sampler
│   └── generate_sample_video.py    # Synthetic wide-angle crowd video generator
│
├── detector/
│   ├── __init__.py
│   └── outlier_detector.py         # Outlier localization engine & video renderer
│
└── analytics/
    ├── __init__.py
    └── report_generator.py         # CSV event logging & timeline/spatial plot renderer
```

---

## ⚙️ Installation & Prerequisites

### 1. Requirements
Ensure Python 3.9+ is installed. Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `torch >= 2.0.0`
- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `matplotlib >= 3.7.0`
- `pyyaml >= 6.0`
- `scipy >= 1.10.0`

---

## 🚀 Quick Start Guide

Run the full end-to-end pipeline (Data Generation $\to$ Training $\to$ Outlier Detection $\to$ Heatmap Video Overlay + CSV Report + Plot Analytics):

```bash
python main.py --mode run_all
```

---

## 💡 Pipeline Commands

### 1. Generate Synthetic Benchmark Videos
Generates normal crowd flow video (`data/sample_crowd_normal.mp4`) and test video with injected crowd anomaly events (`data/sample_crowd_test.mp4`):

```bash
python main.py --mode generate_data
```

### 2. Train VideoMAE Model on Normal Crowd Video
Trains the self-supervised VideoMAE model exclusively on normal crowd video clips:

```bash
python main.py --mode train --video_path data/sample_crowd_normal.mp4 --epochs 15
```

### 3. Run Outlier Localization & Generate Heatmap Video
Evaluates the test video using the trained VideoMAE checkpoint, renders pixel-level spatio-temporal heatmaps, and logs events:

```bash
python main.py --mode detect --video_path data/sample_crowd_test.mp4
```

### 4. Run System Unit & Integration Tests
Execute the complete automated test suite:

```bash
python test_pipeline.py
```

---

## 📊 Outputs & Artifacts

After running detection, the following artifacts are generated in the `outputs/` directory:

| File Path | Description |
| :--- | :--- |
| `outputs/output_annotated.mp4` | Video with overlaid pixel-level anomaly heatmaps, bounding boxes, risk gauge, and alert banner |
| `outputs/anomaly_events.csv` | Structured CSV report of all anomaly trigger events with timestamps and spatial bounding boxes |
| `outputs/anomaly_timeline.png` | Time-series plot of reconstruction MSE anomaly score vs. video time with threshold line |
| `outputs/spatial_heatmap_summary.png` | Cumulative spatial risk heatmap across the surveillance field of view |

---

## 🛠️ Configuration Options (`config.yaml`)

Key configurable parameters in `config.yaml`:

```yaml
data:
  image_size: [128, 128]         # Processing resolution
  num_frames: 16                 # Temporal sequence depth
  tubelet_size: 2                # 3D temporal patch depth
  patch_size: 16                 # Spatial patch size

model:
  embed_dim: 128                 # Transformer embedding dimension
  encoder_depth: 4               # ViT encoder depth
  decoder_depth: 2               # ViT decoder depth
  masking_ratio: 0.75            # SSL tubelet masking ratio

detection:
  anomaly_threshold: 0.031       # Anomaly trigger threshold (Reconstruction MSE)
  heatmap_alpha: 0.5             # Heatmap overlay opacity
  alert_cooldown_frames: 10      # Cooldown frame interval between CSV log entries
```
