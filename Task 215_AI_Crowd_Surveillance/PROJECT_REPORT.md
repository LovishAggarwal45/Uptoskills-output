# AI-Based Crowd Surveillance System: Technical & Academic Research Report

## Weakly-Supervised Spatio-Temporal Action Localization for Long CCTV Videos

**Author**: Academic & Engineering Portfolio Research Project  
**Date**: September 2026  
**Repository**: `ai_crowd_surveillance_project`

---

## 1. Abstract

Automated detection and localization of abnormal crowd actions in CCTV surveillance is a challenging computer vision problem because real surveillance footage is long, untrimmed, and difficult to annotate at frame level. This report presents the design and implementation of a Weakly-Supervised Spatio-Temporal Action Localization (WSSTAL) system for long surveillance videos.

The framework uses video-level supervision within a Multiple Instance Learning architecture and combines RGB appearance features with dense optical-flow motion features. A temporal Transformer models relationships between video segments, while temporal filtering and threshold-based event grouping provide event interval localization.

Spatial regions are estimated using motion-energy based pseudo-localization rather than ground-truth bounding-box supervision. The system also provides automated event aggregation, confidence reporting, annotated video generation, timeline visualization, and CSV-based audit reporting.

Controlled synthetic experiments are used for pipeline and model development, while selected UCF-Crime videos are used as an out-of-domain real-world inference demonstration. The real-world experiments highlight the importance of domain adaptation and confidence calibration when transferring models from controlled training data to genuine surveillance footage.

---

## 2. Introduction & Background

Public safety and facility management systems increasingly rely on closed-circuit television networks operating 24 hours a day, 7 days a week.

Human surveillance operators must continuously monitor large volumes of video footage, making automated video analysis useful for identifying potentially relevant events and reducing the amount of footage requiring manual inspection.

While computer vision systems promise automated triage, classic action recognition architectures such as standard 3D CNNs generally assume pre-trimmed, short video clips where the action occupies a significant portion of the temporal duration.

Long, unedited surveillance videos, however, are untrimmed. Violent or anomalous actions may occur only briefly within long periods of normal background activity.

This creates three major challenges:

1. The system must determine **what** action is occurring.

2. It must determine **when** the action occurs.

3. It must estimate **where** the relevant motion or activity occurs.

The proposed system addresses these requirements using a weakly supervised spatio-temporal architecture designed for long surveillance videos.

---

## 3. Problem Statement

Given an untrimmed surveillance video $\mathcal{V}$ of arbitrary duration $T_{\text{video}}$, the objective is to build a system that:

### 3.1 Action Classification

Classifies the action category:

$$
Y \in \{0, 1, \dots, C-1\}
$$

where $C$ represents the number of supported action classes.

### 3.2 Temporal Localization

Localizes the temporal event interval:

$$
[t_{\text{start}}, t_{\text{end}}]
$$

indicating when the detected event occurs.

### 3.3 Spatial Localization

Identifies a spatial region:

$$
(x, y, w, h)
$$

or an attention/motion region indicating where relevant activity occurs.

### 3.4 Alert Generation

Generates an alert when the detection confidence exceeds a configured threshold $\theta$.

### 3.5 Weak Supervision

Operates under a weak-supervision setting in which training can use video-level labels rather than requiring dense frame-level temporal and spatial annotations.

The system is designed as an automated video-analysis and triage tool. Model-generated detections are not intended to independently establish that a crime has occurred.

---

## 4. Objectives

The major objectives of the project are:

- **End-to-End Modular Pipeline**: Implement video streaming/decoding, two-stream feature extraction, temporal self-attention encoding, MIL pooling, temporal filtering, spatial ROI extraction, alerting, and reporting.

- **Scientific Honesty & Verification**: Explicitly distinguish pseudo-localization bounding boxes from ground-truth annotations and report only metrics that are actually computed from available evaluation data.

- **CPU-Compatible Execution**: The complete pipeline is executable on a multi-core consumer CPU without requiring a multi-GPU server environment, while retaining compatibility with CUDA-enabled execution when available.

- **Long-Video Processing**: Support untrimmed surveillance videos rather than requiring actions to occupy the complete video.

- **Automated Event Reporting**: Generate event intervals, confidence values, spatial pseudo-localizations, annotated videos, timeline dashboards, and machine-readable CSV reports.

- **Portfolio-Ready Engineering**: Maintain a modular research-oriented implementation suitable for academic demonstration, experimentation, and further development.

---

## 5. Existing Challenges in Surveillance Vision

### 5.1 Annotation Dilemma

Annotating bounding boxes and action labels at high frame rates for long surveillance videos creates a substantial annotation burden.

For example, a one-hour video recorded at 30 frames per second contains:

$$
60 \times 60 \times 30 = 108,000
$$

frames.

Providing detailed spatial and temporal annotations for every frame is therefore expensive and time-consuming.

---

### 5.2 Temporal Sparsity

Anomalous events may occupy only a small portion of a long surveillance recording.

The system must therefore distinguish between:

- Normal background activity.
- Short-duration anomalous activity.
- Repeated or multiple events.
- Motion that is visually significant but not necessarily anomalous.

This creates a challenging temporal localization problem.

---

### 5.3 Motion Ambiguity

Fast movement does not necessarily indicate an abnormal or violent action.

For example, a person running to catch a bus can generate strong optical-flow responses without representing violence.

Therefore, the system combines:

- RGB appearance information.
- Optical-flow motion information.
- Temporal context.

This two-stream representation helps the model consider both scene appearance and motion characteristics.

---

### 5.4 Domain Shift

Models trained on controlled or synthetic data may encounter substantially different visual characteristics when applied to real-world surveillance footage.

Differences can include:

- Camera viewpoint.
- Lighting.
- Resolution.
- Compression.
- Crowd density.
- Background appearance.
- Motion patterns.
- Camera movement.

The UCF-Crime inference experiments conducted in this project demonstrate the importance of considering this domain-shift problem.

---

## 6. Proposed System Overview

The proposed framework treats each untrimmed video as a bag of $S$ sequential temporal snippets:

$$
\{S_1, S_2, \dots, S_S\}
$$

Each snippet contains a sequence of sampled frames.

The current implementation uses clips containing:

$$
T = 16
$$

frames.

The system follows a two-stream neural architecture.

### 6.1 Spatial Branch

The spatial branch extracts visual semantic information from RGB frames using a ResNet-18 backbone.

The current implementation uses ImageNet-pretrained ResNet-18 features.

The resulting RGB representation has dimensionality:

$$
f_{\text{rgb}} \in \mathbb{R}^{512}
$$

---

### 6.2 Motion Branch

The motion branch extracts movement information using dense Farneback optical flow.

For each frame pair, the optical-flow field provides horizontal and vertical displacement components:

$$
(u,v)
$$

These motion features are processed through a motion convolutional encoder to obtain:

$$
f_{\text{flow}} \in \mathbb{R}^{128}
$$

Optical flow provides motion information between frames but does not by itself provide semantic understanding of whether that motion represents a specific action.

---

### 6.3 Feature Fusion

The RGB and motion representations are concatenated:

$$
[f_{\text{rgb}} \| f_{\text{flow}}]
$$

and projected into a shared latent space with dimensionality:

$$
d = 256
$$

---

### 6.4 Temporal Transformer

The fused segment representations are processed using a multi-head temporal Transformer.

The Transformer models relationships between different temporal snippets of the same surveillance video.

---

### 6.5 Multiple Instance Learning

The complete video is treated as a bag of temporal instances.

The model produces segment-level anomaly scores and action predictions.

Top-$k$ temporal instances are used for anomaly aggregation, while temporal attention weights are used for action classification.

---

### 6.6 Event Generation

The inference pipeline converts segment predictions into event-level outputs through:

1. Temporal smoothing.
2. Confidence thresholding.
3. Contiguous interval grouping.
4. Merge-gap processing.
5. Minimum-duration filtering.
6. Spatial motion-region extraction.
7. Alert generation.
8. Visualization and CSV reporting.

---

## 7. Mathematical Formulations

### 7.1 Optical Flow Formulation

Dense optical flow satisfies the brightness constancy assumption:

$$
I(x + \Delta x, y + \Delta y, t + \Delta t)
=
I(x, y, t)
$$

Using the Farneback polynomial expansion method, displacement vectors $(u,v)$ are computed at pixel coordinates $(x,y)$.

Motion magnitude and direction are then derived as:

$$
M(x,y)
=
\sqrt{u(x,y)^2 + v(x,y)^2}
$$

and

$$
\theta(x,y)
=
\text{arctan2}(v(x,y),u(x,y))
$$

where:

- $M(x,y)$ represents motion magnitude.
- $\theta(x,y)$ represents motion direction.

---

### 7.2 Two-Stream Feature Fusion

For temporal segment $t$:

$$
f_{\text{rgb}}^{(t)}
=
\phi_{\text{ResNet}}
(I_{\text{rgb}}^{(t)})
\in \mathbb{R}^{512}
$$

and:

$$
f_{\text{flow}}^{(t)}
=
\psi_{\text{MotionNet}}
(I_{\text{flow}}^{(t)})
\in \mathbb{R}^{128}
$$

The representations are concatenated and projected into the fusion space:

$$
f_{\text{fused}}^{(t)}
=
\text{LayerNorm}
\left(
\mathbf{W}_f
[
f_{\text{rgb}}^{(t)}
\|
f_{\text{flow}}^{(t)}
]
+
\mathbf{b}_f
\right)
\in
\mathbb{R}^{256}
$$

---

### 7.3 Multi-Head Temporal Self-Attention

Positional encodings are injected into the fused representation:

$$
X = F + P
$$

where:

$$
P \in \mathbb{R}^{S \times d}
$$

The attention operation is:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
$$

where:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

This enables the model to incorporate temporal relationships between different segments.

---

### 7.4 MIL Ranking Loss & Regularizers

Let $\mathcal{B}_a$ represent an abnormal video bag and $\mathcal{B}_n$ represent a normal video bag.

The segment-level anomaly score is:

$$
s_t
=
\sigma
(
\mathbf{w}_a^\top h_t
)
\in [0,1]
$$

The ranking loss is formulated as:

$$
\mathcal{L}_{\text{rank}}
=
\max
\left(
0,
1
-
\max_{t \in \mathcal{B}_a}s_t
+
\max_{t \in \mathcal{B}_n}s_t
\right)
$$

Temporal smoothness is encouraged using:

$$
\mathcal{L}_{\text{smooth}}
=
\sum_{t=1}^{S-1}
(s_{t+1}-s_t)^2
$$

Sparsity is encouraged using:

$$
\mathcal{L}_{\text{sparse}}
=
\sum_{t=1}^{S}s_t
$$

The combined objective is represented as:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{BCE}}
(Y_{\text{video}},y)
+
\mathcal{L}_{\text{CE}}
(P_{\text{class}},c)
+
\mathcal{L}_{\text{rank}}
+
\lambda_s\mathcal{L}_{\text{smooth}}
+
\lambda_p\mathcal{L}_{\text{sparse}}
$$

---

## 8. Temporal & Spatial Localization Methodology

### 8.1 Temporal Localization

The temporal localization pipeline consists of the following stages:

#### Step 1 — Temporal Smoothing

Continuous 1D Gaussian smoothing is applied to snippet anomaly scores.

This reduces isolated frame/segment-level fluctuations.

---

#### Step 2 — Confidence Thresholding

Segments satisfying:

$$
s_t \geq \theta_{\text{conf}}
$$

are considered candidate anomalous segments.

The current configuration uses:

$$
\theta_{\text{conf}} = 0.52
$$

---

#### Step 3 — Contiguous Segment Clustering

Adjacent positive snippets are grouped into continuous candidate event intervals.

---

#### Step 4 — Merge-Gap Bridging

Candidate intervals separated by less than:

$$
T_{\text{merge}} = 2.0
\text{ seconds}
$$

can be merged into a single event.

---

#### Step 5 — Minimum Duration Filtering

Candidate events shorter than:

$$
T_{\text{min}} = 1.0
\text{ second}
$$

are discarded to reduce isolated detections.

---

### 8.2 Spatial Localization

The current system does not use manually annotated ground-truth bounding boxes for spatial supervision.

Instead, spatial regions are generated using optical-flow motion energy.

The procedure is:

1. The optical-flow field around the temporal event center is converted into motion magnitude.

2. Motion magnitude is processed to identify active regions.

3. Morphological processing connects nearby motion contours.

4. Bounding rectangles are generated around detected motion regions.

5. The resulting regions are reported as **Motion-Energy Based Spatial Pseudo-Localizations**.

These bounding boxes should not be interpreted as ground-truth person or action bounding boxes.

---

### 8.3 Spatial Localization Terminology

The system uses the following terminology:

**Motion-Energy Based Spatial Pseudo-Localization**

rather than:

- Ground-truth bounding box.
- Human-annotated action box.
- Verified person bounding box.

This distinction is maintained to avoid overstating the spatial localization capability of the current implementation.

---

## 9. Experimental Protocol & Evaluation Metrics

The project uses two distinct experimental stages.

### 9.1 Controlled Synthetic Benchmark

The controlled synthetic benchmark is used for:

- Model development.
- Architecture testing.
- Training experiments.
- Pipeline verification.
- Temporal localization testing.
- Unit testing.
- Debugging.
- Reproducibility checks.

The synthetic benchmark is not treated as evidence of real-world CCTV performance.

---

### 9.2 Real-World UCF-Crime Demonstration

Selected UCF-Crime surveillance videos are used to examine how the current model behaves when exposed to real-world surveillance footage from a different data distribution.

The current checkpoint used for this demonstration was trained using the project's synthetic training data and was not trained or fine-tuned on UCF-Crime.

Therefore, these experiments are classified as:

**Out-of-Domain Real-World Inference Demonstration**

rather than formal UCF-Crime accuracy evaluation.

---

### 9.3 Evaluation Metrics

Where appropriate ground-truth temporal annotations are available, the evaluation engine can assess:

#### Frame-Level ROC-AUC

Measures the area under the Receiver Operating Characteristic curve across temporal samples.

---

#### Frame-Level PR-AUC

Measures the area under the Precision-Recall curve and is useful for imbalanced anomaly-detection problems.

---

#### Accuracy

The proportion of correctly classified samples at the selected detection threshold.

---

#### Precision

The proportion of detected positive samples that correspond to positive ground-truth samples.

---

#### Recall

The proportion of ground-truth positive samples successfully detected.

---

#### F1 Score

The harmonic mean of precision and recall:

$$
F_1
=
2
\frac{
\text{Precision}\times\text{Recall}
}{
\text{Precision}+\text{Recall}
}
$$

---

#### Temporal Intersection over Union

For predicted event interval $E_{\text{pred}}$ and ground-truth event interval $E_{\text{gt}}$:

$$
tIoU
=
\frac{
|E_{\text{pred}}\cap E_{\text{gt}}|
}{
|E_{\text{pred}}\cup E_{\text{gt}}|
}
$$

This measures the temporal overlap between predicted and ground-truth events.

---

## 10. Experimental Results

The experimental evaluation is divided into two distinct stages to maintain scientific validity.

---

### 10.1 Controlled Synthetic Benchmark

The initial development and model verification experiments were conducted using the project's controlled synthetic surveillance benchmark.

This stage was used to verify:

- End-to-end training and inference functionality.
- Two-stream RGB and optical-flow feature extraction.
- Temporal Transformer processing.
- Multiple Instance Learning aggregation.
- Temporal event localization.
- Motion-energy based spatial pseudo-localization.
- Alert generation and CSV reporting.
- CPU-compatible execution.

The synthetic benchmark was primarily used for engineering validation and controlled model experimentation.

Therefore, synthetic benchmark results are not presented as evidence of real-world CCTV performance.

---

### 10.2 Real-World UCF-Crime Inference Demonstration

A subset of the UCF-Crime real-world surveillance dataset was subsequently used to examine how the trained system behaves when exposed to genuine surveillance footage from a different data distribution.

The current model checkpoint was trained using the project's synthetic training data and was **not trained or fine-tuned on UCF-Crime**.

Therefore, these experiments constitute an:

**Out-of-Domain Real-World Inference Demonstration**

rather than a formal UCF-Crime accuracy evaluation.

Three videos were processed.

| Video | Ground-Truth Category | Duration | Predicted Action | Reported Confidence | Detected Events |
|---|---|---:|---|---:|---:|
| `Fighting002_x264.mp4` | Fighting | 89.60 s | Panic_Dispersal | 54.5% | 2 |
| `Fighting003_x264.mp4` | Fighting | 103.40 s | Panic_Dispersal | 55.0% | 6 |
| `Normal_Videos_006_x264.mp4` | Normal | 15.00 s | Panic_Dispersal | 53.7% | 1 |

The experiments demonstrate that the complete inference pipeline can process real-world untrimmed surveillance videos and generate:

- Temporal event intervals.
- Predicted action categories.
- Confidence values.
- Motion-energy based spatial pseudo-localizations when reliable motion regions are detected.
- Annotated surveillance videos.
- Timeline visualization dashboards.
- Machine-readable CSV event reports.

---

### 10.3 Observed Domain-Shift Behavior

The real-world experiments demonstrated a clear domain-shift limitation.

The current model produced `Panic_Dispersal` predictions on both Fighting-category videos and the Normal-category video, with reported confidence values in approximately the 52–55% range.

This behavior indicates that the synthetic training distribution does not adequately represent all visual and motion characteristics present in genuine surveillance footage.

The result should therefore be interpreted as evidence of the current model's sensitivity to domain shift rather than as evidence of successful real-world classification.

The experiment also demonstrates why simply lowering or raising the alert threshold to suppress unwanted detections would not constitute a scientifically valid solution.

A more appropriate future approach is to train and evaluate the model using representative real-world surveillance data and to perform proper confidence calibration.

---

### 10.4 Generated Real-World Artifacts

The following artifacts were successfully generated during the real-world demonstration.

#### Annotated Videos

```text
outputs/annotated_videos/Fighting002_x264_annotated.mp4

outputs/annotated_videos/Fighting003_x264_annotated.mp4

outputs/annotated_videos/Normal_Videos_006_x264_annotated.mp4
### 10.5 Real-World Demonstration Summary

The real-world demonstration confirms that the complete inference pipeline can process genuine surveillance videos and produce temporal localization, confidence scores, spatial pseudo-localizations, annotated videos, and timeline dashboards.

The three demonstrated UCF-Crime videos produced the following observations:

- `Fighting002_x264.mp4`: Fighting-category video, with two localized `Panic_Dispersal` events.
- `Fighting003_x264.mp4`: Fighting-category video, with six localized `Panic_Dispersal` events.
- `Normal_Videos_006_x264.mp4`: Normal-category video, with one localized `Panic_Dispersal` event.

The model produced confidence values in approximately the 52–55% range for the reported events.

Because the current checkpoint was trained using the project's synthetic benchmark rather than UCF-Crime training data, these results are treated as an **Out-of-Domain Real-World Inference Demonstration** rather than a formal UCF-Crime accuracy evaluation.

The results demonstrate the complete operational pipeline while also exposing the domain-shift problem between synthetic training data and genuine surveillance footage.

---

## 11. Limitations & Ethical Boundary

### 11.1 Synthetic Training Data

The current model development experiments use a controlled synthetic surveillance benchmark. Synthetic data is useful for verifying the architecture, training loop, inference pipeline, localization logic, visualization, and reporting system, but it does not reproduce the full diversity of genuine CCTV footage.

### 11.2 Domain Shift

The UCF-Crime demonstration showed that a model trained on synthetic data can produce abnormal-action predictions when applied to genuine surveillance footage. However, the observed predictions should not be interpreted as evidence of real-world classification accuracy.

The current results therefore highlight the need for representative real-world training and validation data.

### 11.3 Spatial Localization

The system currently produces **Motion-Energy Based Spatial Pseudo-Localizations**. These regions are generated from optical-flow motion information and connected-component analysis.

They are not manually annotated ground-truth bounding boxes.

Therefore, spatial localization results should not be interpreted as verified identification of a specific person or object responsible for an event.

### 11.4 Threshold Selection

The alert threshold is part of the experimental configuration. Changing the threshold only to suppress unwanted predictions on selected videos would introduce evaluation bias.

Proper threshold selection should instead be performed using an independent validation set and documented calibration procedure.

### 11.5 Ethical Boundary

The system is intended to support surveillance analysis by generating alerts and visual evidence for human review.

An automated prediction should not be treated as legal proof that a crime has occurred or that a particular person committed an offence.

Final interpretation and action should remain with appropriately authorized human operators.

---

## 12. Implementation & Reproducibility

The project is implemented as a modular Python-based surveillance analysis pipeline.

### 12.1 Main Components

```text
src/
├── data/
├── preprocessing/
├── features/
├── models/
├── localization/
├── training/
├── evaluation/
├── inference/
├── alerts/
└── visualization/
## 13. Conclusion

This project presents a weakly supervised spatio-temporal action localization framework for analyzing long, untrimmed surveillance videos.

The proposed system combines RGB appearance information, optical-flow motion information, feature fusion, temporal Transformer encoding, Multiple Instance Learning, temporal localization, motion-energy based spatial pseudo-localization, alert generation, and visualization.

The controlled synthetic experiments were used to validate the implementation and establish a reproducible development benchmark. The subsequent UCF-Crime experiments demonstrated that the complete pipeline can process genuine surveillance footage and produce interpretable outputs such as event intervals, confidence values, spatial pseudo-localizations, annotated videos, and timeline dashboards.

At the same time, the real-world demonstration revealed a significant domain-shift limitation: a model trained only on synthetic data does not provide sufficient evidence for reliable real-world action classification.

Therefore, the current system should be considered a functional research prototype and surveillance-analysis framework rather than a production-ready crime detection system.

Future development should focus on training with representative real-world surveillance data, improving confidence calibration, evaluating temporal localization against official annotations, improving spatial localization, and conducting systematic experiments across multiple surveillance environments.

The project establishes a complete foundation for further development toward robust weakly supervised surveillance video understanding.