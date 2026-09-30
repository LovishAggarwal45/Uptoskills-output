"""
End-to-End Surveillance Inference Pipeline.

Consumes long untrimmed CCTV videos, extracts RGB and Optical Flow features,
runs the WSSTAL model, localizes temporal events, extracts spatial ROIs,
triggers configurable alerts, generates CSV audit logs, and produces
annotated video + timeline charts.

Spatial localization is explicitly treated as
Motion-Energy Based Spatial Pseudo-Localization.
"""

import cv2
import json
import yaml
import torch
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine
from src.localization.temporal_localizer import TemporalLocalizer
from src.localization.spatial_localizer import SpatialLocalizer
from src.alerts.alert_manager import AlertManager
from src.alerts.report_generator import ReportGenerator
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.timeline_plotter import TimelinePlotter


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


class SurveillanceInferencePipeline:
    """
    End-to-end inference pipeline for long CCTV videos.

    Pipeline:

        CCTV Video
            ↓
        Video Sampling
            ↓
        RGB + Optical Flow
            ↓
        Two-Stream WSSTAL Model
            ↓
        Temporal Localization
            ↓
        Motion-Energy Spatial Pseudo-Localization
            ↓
        Alert Generation
            ↓
        CSV + JSON + Timeline + Annotated Video
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        # ------------------------------------------------------------
        # 1. Load configuration
        # ------------------------------------------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # ------------------------------------------------------------
        # 2. Device setup
        # ------------------------------------------------------------
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"Inference device: {self.device}")

        # ------------------------------------------------------------
        # 3. Initialize WSSTAL model
        # ------------------------------------------------------------
        m_cfg = self.config["model"]
        d_cfg = self.config["dataset"]

        self.model = TwoStreamWSSTALNet(
            num_classes=d_cfg.get("num_classes", 3),
            rgb_feature_dim=m_cfg.get("rgb_feature_dim", 512),
            flow_feature_dim=m_cfg.get("flow_feature_dim", 128),
            fusion_dim=m_cfg.get("fusion_dim", 256),
            encoder_layers=m_cfg.get("encoder_layers", 2),
            nheads=m_cfg.get("nheads", 4)
        ).to(self.device)

        # ------------------------------------------------------------
        # 4. Load trained checkpoint
        # ------------------------------------------------------------
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading weights from {checkpoint_path}...")

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)

            print("Checkpoint loaded successfully.")

        else:
            print(
                "WARNING: No fine-tuned checkpoint was supplied. "
                "Running with initialized model weights."
            )

        self.model.eval()

        # ------------------------------------------------------------
        # 5. Initialize processing configuration
        # ------------------------------------------------------------
        v_cfg = self.config["video"]
        l_cfg = self.config["localization"]
        a_cfg = self.config["alerts"]

        # ------------------------------------------------------------
        # 6. Optical flow engine
        # ------------------------------------------------------------
        self.flow_engine = OpticalFlowEngine(
            target_size=tuple(v_cfg["flow_size"])
        )

        # ------------------------------------------------------------
        # 7. Temporal localization engine
        # ------------------------------------------------------------
        self.temporal_localizer = TemporalLocalizer(
            threshold=l_cfg["threshold"],
            temporal_smoothing_kernel=l_cfg.get(
                "temporal_smoothing_kernel",
                5
            ),
            min_duration_seconds=l_cfg["min_duration_seconds"],
            merge_gap_seconds=l_cfg["merge_gap_seconds"],
            classes=d_cfg["classes"]
        )

        # ------------------------------------------------------------
        # 8. Spatial pseudo-localization engine
        # ------------------------------------------------------------
        self.spatial_localizer = SpatialLocalizer(
            min_box_area_ratio=l_cfg.get(
                "min_box_area_ratio",
                0.005
            ),
            threshold_ratio=l_cfg.get(
                "spatial_threshold",
                0.35
            )
        )

        # ------------------------------------------------------------
        # 9. Alert manager
        # ------------------------------------------------------------
        self.alert_manager = AlertManager(
            confidence_threshold=a_cfg["confidence_threshold"],
            log_file=a_cfg["log_file"]
        )

        # ------------------------------------------------------------
        # 10. CSV report generator
        # ------------------------------------------------------------
        self.report_generator = ReportGenerator(
            output_csv=a_cfg["csv_report"]
        )

        # ------------------------------------------------------------
        # 11. Visualization components
        # ------------------------------------------------------------
        self.annotator = VideoAnnotator()
        self.timeline_plotter = TimelinePlotter()

    # =================================================================
    # PROCESS VIDEO
    # =================================================================

    def process_video(
        self,
        video_path: str,
        output_annotated: bool = True,
        output_plot: bool = True
    ) -> Dict[str, Any]:

        video_path = Path(video_path)
        video_id = video_path.stem

        if not video_path.exists():
            raise FileNotFoundError(
                f"Input video does not exist: {video_path}"
            )

        print("\n" + "=" * 70)
        print(f"PROCESSING SURVEILLANCE VIDEO: {video_path.name}")
        print("=" * 70)

        # ------------------------------------------------------------
        # Step 1: Initialize video reader
        # ------------------------------------------------------------
        v_cfg = self.config["video"]

        reader = VideoReader(
            str(video_path),
            target_fps=v_cfg["target_fps"],
            target_size=tuple(v_cfg["image_size"])
        )

        meta = reader.get_metadata()

        print(
            f"Video Info: "
            f"{meta['native_frame_count']} frames @ "
            f"{meta['native_fps']:.1f} FPS, "
            f"{meta['duration_seconds']:.2f}s"
        )

        # ------------------------------------------------------------
        # Step 2: Extract video clips
        # ------------------------------------------------------------
        print("\n[1/7] Extracting RGB clips and dense optical flow...")

        clips = list(
            reader.stream_clips(
                clip_length=v_cfg["clip_length"],
                stride=v_cfg["stride"]
            )
        )

        if not clips:
            raise RuntimeError(
                "No valid clips could be sampled from the video."
            )

        rgb_list = []
        flow_list = []
        timestamps = []

        # Raw middle optical-flow maps used for
        # spatial pseudo-localization.
        clip_flows_raw = []

        for clip in clips:

            # --------------------------------------------------------
            # RGB normalization
            # --------------------------------------------------------
            norm_rgb = (
                clip["frames"].astype(np.float32) / 255.0
                - IMAGENET_MEAN
            ) / IMAGENET_STD

            # --------------------------------------------------------
            # Dense optical flow
            # --------------------------------------------------------
            flow_data = self.flow_engine.compute_clip_flow(
                clip["frames"]
            )

            # --------------------------------------------------------
            # Store tensors
            # --------------------------------------------------------
            rgb_list.append(
                np.transpose(
                    norm_rgb,
                    (0, 3, 1, 2)
                )
            )

            flow_list.append(
                flow_data["flow"]
            )

            timestamps.append(
                (
                    clip["start_time"],
                    clip["end_time"]
                )
            )

            # --------------------------------------------------------
            # Keep middle flow map for spatial ROI extraction
            # --------------------------------------------------------
            middle_index = len(flow_data["flow"]) // 2

            clip_flows_raw.append(
                flow_data["flow"][middle_index]
            )

        # ------------------------------------------------------------
        # Step 3: Prepare model tensors
        # ------------------------------------------------------------
        print("\n[2/7] Preparing model tensors...")

        rgb_t = torch.from_numpy(
            np.stack(rgb_list, axis=0)
        ).unsqueeze(0).to(self.device)

        flow_t = torch.from_numpy(
            np.stack(flow_list, axis=0)
        ).unsqueeze(0).to(self.device)

        # ------------------------------------------------------------
        # Step 4: Model inference
        # ------------------------------------------------------------
        print("\n[3/7] Running spatio-temporal inference network...")

        with torch.no_grad():
            outputs = self.model(
                rgb_t,
                flow_t
            )

        anom_scores = (
            outputs["segment_anomaly_scores"]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        action_probs = (
            outputs["segment_action_probs"]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        video_anom = float(
            outputs["video_anomaly_score"]
            .item()
        )

        # ------------------------------------------------------------
        # Step 5: Temporal localization
        # ------------------------------------------------------------
        print("\n[4/7] Performing temporal action localization...")

        detected_events = self.temporal_localizer.localize_events(
            anom_scores,
            action_probs,
            timestamps
        )

        print(
            f"Localized {len(detected_events)} "
            f"temporal event interval(s)."
        )

        # ------------------------------------------------------------
        # Step 6: Spatial pseudo-localization
        # ------------------------------------------------------------
        print(
            "\n[5/7] Performing motion-energy "
            "spatial pseudo-localization..."
        )

        representative_bboxes = {}

        for event in detected_events:

            start_segment = event["start_segment"]
            end_segment = event["end_segment"]

            mid_segment = (
                start_segment + end_segment
            ) // 2

            # Safety check
            if not (
                0 <= mid_segment < len(clip_flows_raw)
            ):
                representative_bboxes[
                    event["event_id"]
                ] = None

                continue

            flow_sample = clip_flows_raw[mid_segment]

            boxes = self.spatial_localizer.extract_motion_rois(
                flow_sample,
                frame_shape=(
                    meta["height"],
                    meta["width"]
                )
            )

            if boxes:
                # Use the strongest detected motion ROI.
                representative_bboxes[
                    event["event_id"]
                ] = boxes[0]["bbox_norm"]

            else:
                # IMPORTANT:
                # Never invent a bounding box when no reliable
                # motion region was found.
                representative_bboxes[
                    event["event_id"]
                ] = None

        # ------------------------------------------------------------
        # Step 7: Alerts and CSV report
        # ------------------------------------------------------------
        print("\n[6/7] Generating alerts and event report...")

        alerts = self.alert_manager.process_events(
            video_id,
            detected_events,
            representative_bboxes
        )

        self.report_generator.write_events(
            video_id,
            detected_events,
            representative_bboxes
        )

        # ------------------------------------------------------------
        # Timeline visualization
        # ------------------------------------------------------------
        mid_times = [
            (start + end) / 2
            for start, end in timestamps
        ]

        plot_file = None

        if output_plot:

            plot_file = (
                self.timeline_plotter
                .plot_surveillance_timeline(
                    video_id=video_id,
                    timestamps=mid_times,
                    anomaly_scores=anom_scores,
                    action_probs=action_probs,
                    detected_events=detected_events,
                    threshold=self.config[
                        "localization"
                    ]["threshold"],
                    classes=self.config[
                        "dataset"
                    ]["classes"]
                )
            )

            print(
                f"Diagnostic Timeline Plot: {plot_file}"
            )

        # ------------------------------------------------------------
        # Render annotated video
        # ------------------------------------------------------------
        annotated_video_path = None

        if output_annotated:

            print(
                "\nRendering annotated surveillance video..."
            )

            output_video_dir = Path(
                "outputs/annotated_videos"
            )

            output_video_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            annotated_video_path = (
                output_video_dir
                / f"{video_id}_annotated.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            cap = cv2.VideoCapture(
                str(video_path)
            )

            orig_fps = (
                cap.get(cv2.CAP_PROP_FPS)
                or 16
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            writer = cv2.VideoWriter(
                str(annotated_video_path),
                fourcc,
                orig_fps,
                (width, height)
            )

            frame_idx = 0

            while cap.isOpened():

                ret, bgr_frame = cap.read()

                if not ret:
                    break

                current_timestamp = (
                    frame_idx / orig_fps
                )

                # ----------------------------------------------------
                # Find active event
                # ----------------------------------------------------
                active_event = None

                for event in detected_events:

                    if (
                        event["start_time"]
                        <= current_timestamp
                        <= event["end_time"]
                    ):
                        active_event = event
                        break

                # ----------------------------------------------------
                # Alert state
                # ----------------------------------------------------
                is_alert = (
                    active_event is not None
                    and active_event["confidence"]
                    >= self.config[
                        "alerts"
                    ]["confidence_threshold"]
                )

                # ----------------------------------------------------
                # Current action
                # ----------------------------------------------------
                action_name = (
                    active_event["action"]
                    if active_event
                    else "Normal"
                )

                # ----------------------------------------------------
                # Current confidence
                # ----------------------------------------------------
                if active_event:

                    action_confidence = (
                        active_event["confidence"]
                    )

                else:

                    action_confidence = (
                        1.0
                        - max(
                            0.0,
                            float(
                                np.interp(
                                    current_timestamp,
                                    mid_times,
                                    anom_scores
                                )
                            )
                        )
                    )

                # ----------------------------------------------------
                # Spatial pseudo-localization boxes
                # ----------------------------------------------------
                active_boxes = []

                if (
                    active_event is not None
                    and active_event["event_id"]
                    in representative_bboxes
                ):

                    bbox = representative_bboxes[
                        active_event["event_id"]
                    ]

                    # Only draw a box if a genuine motion ROI
                    # was extracted.
                    if bbox is not None:

                        bx, by, bw, bh = bbox

                        active_boxes.append(
                            {
                                "bbox_pixel": (
                                    int(bx * width),
                                    int(by * height),
                                    int(bw * width),
                                    int(bh * height)
                                ),
                                "bbox_norm": (
                                    bx,
                                    by,
                                    bw,
                                    bh
                                )
                            }
                        )

                # ----------------------------------------------------
                # Annotate frame
                # ----------------------------------------------------
                annotated_frame = (
                    self.annotator.annotate_frame(
                        bgr_frame,
                        timestamp=current_timestamp,
                        current_action=action_name,
                        confidence=action_confidence,
                        is_alert=is_alert,
                        active_event=active_event,
                        bboxes=(
                            active_boxes
                            if is_alert
                            else None
                        )
                    )
                )

                writer.write(
                    annotated_frame
                )

                frame_idx += 1

            cap.release()
            writer.release()

            print(
                f"Annotated CCTV Video: "
                f"{annotated_video_path}"
            )

        # ------------------------------------------------------------
        # Save prediction JSON
        # ------------------------------------------------------------
        output_prediction_dir = Path(
            "outputs/predictions"
        )

        output_prediction_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        prediction_file = (
            output_prediction_dir
            / f"{video_id}_predictions.json"
        )

        prediction_data = {
            "video_id": video_id,
            "duration": meta[
                "duration_seconds"
            ],
            "video_anomaly_score": round(
                video_anom,
                4
            ),
            "timestamps": mid_times,
            "anomaly_scores": [
                round(float(score), 4)
                for score in anom_scores
            ],
            "action_probs": [
                [
                    round(float(probability), 4)
                    for probability in row
                ]
                for row in action_probs
            ],
            "detected_events": detected_events,
            "alerts_triggered": alerts
        }

        with open(
            prediction_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                prediction_data,
                f,
                indent=2
            )

        # ------------------------------------------------------------
        # Executive surveillance summary
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print(
            f"SURVEILLANCE INFERENCE SUMMARY: "
            f"{video_id}"
        )
        print("=" * 70)

        if detected_events:

            for event in detected_events:

                print(
                    f"  * WHAT HAPPENED?      : "
                    f"{event['action']}"
                )

                print(
                    f"  * WHEN DID IT HAPPEN? : "
                    f"{event['start_time_str']} -> "
                    f"{event['end_time_str']} "
                    f"({event['duration']}s)"
                )

                print(
                    f"  * HOW CONFIDENT?      : "
                    f"{event['confidence'] * 100:.1f}% "
                    f"(Peak: "
                    f"{event['peak_confidence'] * 100:.1f}%)"
                )

                bbox = representative_bboxes.get(
                    event["event_id"]
                )

                if bbox is not None:

                    bx, by, bw, bh = bbox

                    print(
                        f"  * WHERE DID IT HAPPEN? : "
                        f"Bounding Box "
                        f"[x={bx:.2f}, "
                        f"y={by:.2f}, "
                        f"w={bw:.2f}, "
                        f"h={bh:.2f}] "
                        f"(Motion-Energy "
                        f"Pseudo-Localization)"
                    )

                else:

                    print(
                        "  * WHERE DID IT HAPPEN? : "
                        "No reliable spatial ROI detected"
                    )

        else:

            print(
                "  * WHAT HAPPENED?      : "
                "Normal Crowd Flow "
                "(No Abnormal Actions Detected)"
            )

            print(
                f"  * VIDEO DURATION      : "
                f"{meta['duration_seconds']:.2f} seconds"
            )

            print(
                f"  * ANOMALY SCORE       : "
                f"{video_anom * 100:.1f}% "
                f"(Below threshold)"
            )

        print("=" * 70)

        return prediction_data

