"""
Rigorous Evaluation Engine for Weakly Supervised Action Localization.

Computes:
1. Frame-level anomaly ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1
2. Segment-level Temporal IoU (tIoU)
3. Video-level 3-class action classification accuracy
4. Per-class precision, recall, and F1
5. Confusion matrix

The anomaly/localization evaluation and action-classification evaluation
are reported separately because they measure different capabilities.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine
from src.localization.temporal_localizer import TemporalLocalizer


def compute_temporal_iou(
    interval_a: Tuple[float, float],
    interval_b: Tuple[float, float]
) -> float:
    """Computes 1D Temporal Intersection over Union (tIoU)."""

    start_a, end_a = interval_a
    start_b, end_b = interval_b

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    inter = max(0.0, inter_end - inter_start)

    union = (
        (end_a - start_a)
        + (end_b - start_b)
        - inter
    )

    return inter / union if union > 0 else 0.0


class WSSTALEvaluator:

    def __init__(
        self,
        ground_truth_file: str,
        threshold: float = 0.60
    ):
        self.gt_file = Path(ground_truth_file)

        if not self.gt_file.exists():
            raise FileNotFoundError(
                f"Ground truth file missing: {self.gt_file}"
            )

        with open(self.gt_file, "r") as f:
            self.ground_truth = json.load(f)

        self.threshold = threshold

    def evaluate_predictions(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        all_gt_frames = []
        all_pred_frames = []
        video_tious = []

        dt = 0.25

        for res in test_results:

            vid_id = res["video_id"]

            if vid_id not in self.ground_truth:
                continue

            gt_info = self.ground_truth[vid_id]

            dur = max(
                res["duration"],
                gt_info.get("duration", 10.0)
            )

            num_bins = max(
                4,
                int(np.ceil(dur / dt))
            )

            time_bins = np.linspace(
                0,
                dur,
                num_bins
            )

            # --------------------------------------------------
            # 1. Ground-truth anomaly vector
            # --------------------------------------------------

            gt_vec = np.zeros(
                num_bins,
                dtype=int
            )

            gt_events = gt_info.get(
                "events",
                []
            )

            for g_ev in gt_events:

                s_idx = int(
                    g_ev["start_time"] / dt
                )

                e_idx = int(
                    np.ceil(
                        g_ev["end_time"] / dt
                    )
                )

                gt_vec[
                    s_idx:min(e_idx, num_bins)
                ] = 1

            # --------------------------------------------------
            # 2. Predicted anomaly scores
            # --------------------------------------------------

            pred_times = np.array(
                res["timestamps"]
            )

            pred_scores = np.array(
                res["anomaly_scores"]
            )

            if len(pred_times) > 1:

                interp_scores = np.interp(
                    time_bins,
                    pred_times,
                    pred_scores,
                    left=pred_scores[0],
                    right=pred_scores[-1]
                )

            else:

                interp_scores = (
                    np.ones(num_bins)
                    * pred_scores[0]
                )

            all_gt_frames.extend(
                gt_vec.tolist()
            )

            all_pred_frames.extend(
                interp_scores.tolist()
            )

            # --------------------------------------------------
            # 3. Temporal IoU
            # --------------------------------------------------

            det_events = res.get(
                "detected_events",
                []
            )

            if gt_events and det_events:

                for d_ev in det_events:

                    d_int = (
                        d_ev["start_time"],
                        d_ev["end_time"]
                    )

                    max_iou = max(
                        [
                            compute_temporal_iou(
                                d_int,
                                (
                                    g["start_time"],
                                    g["end_time"]
                                )
                            )
                            for g in gt_events
                        ]
                    )

                    video_tious.append(
                        max_iou
                    )

            elif not gt_events and not det_events:

                video_tious.append(1.0)

            elif (
                (not gt_events and det_events)
                or
                (gt_events and not det_events)
            ):

                video_tious.append(0.0)

        # ------------------------------------------------------
        # Frame-level anomaly metrics
        # ------------------------------------------------------

        y_true = np.array(
            all_gt_frames
        )

        y_pred = np.array(
            all_pred_frames
        )

        y_pred_bin = (
            y_pred >= self.threshold
        ).astype(int)

        unique_classes = np.unique(
            y_true
        )

        if len(unique_classes) > 1:

            roc_auc = float(
                roc_auc_score(
                    y_true,
                    y_pred
                )
            )

            pr_auc = float(
                average_precision_score(
                    y_true,
                    y_pred
                )
            )

        else:

            roc_auc = float("nan")
            pr_auc = float("nan")

        acc = float(
            accuracy_score(
                y_true,
                y_pred_bin
            )
        )

        prec, rec, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred_bin,
                average="binary",
                zero_division=0
            )
        )

        mean_tiou = (
            float(np.mean(video_tious))
            if video_tious
            else 0.0
        )

        tiou_arr = np.array(
            video_tious
        )

        hit_tiou_03 = (
            float(
                np.mean(
                    tiou_arr >= 0.3
                )
            )
            if len(tiou_arr) > 0
            else 0.0
        )

        hit_tiou_05 = (
            float(
                np.mean(
                    tiou_arr >= 0.5
                )
            )
            if len(tiou_arr) > 0
            else 0.0
        )

        return {
            "evaluation_status":
                "REAL_MEASURED_EVALUATION",

            "detection_threshold":
                self.threshold,

            "total_evaluated_frames":
                int(len(y_true)),

            "positive_anomaly_frames":
                int(np.sum(y_true)),

            "negative_normal_frames":
                int(
                    len(y_true)
                    - np.sum(y_true)
                ),

            "accuracy":
                round(acc, 4),

            "precision":
                round(float(prec), 4),

            "recall":
                round(float(rec), 4),

            "f1_score":
                round(float(f1), 4),

            "roc_auc":
                round(roc_auc, 4)
                if not np.isnan(roc_auc)
                else "N/A",

            "pr_auc":
                round(pr_auc, 4)
                if not np.isnan(pr_auc)
                else "N/A",

            "mean_temporal_iou":
                round(mean_tiou, 4),

            "temporal_iou_at_0.3":
                round(hit_tiou_03, 4),

            "temporal_iou_at_0.5":
                round(hit_tiou_05, 4)
        }


def evaluate_action_classification(
    test_results: List[Dict[str, Any]],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluate video-level 3-class action classification.

    Each test video contributes exactly one prediction based on
    the model's video_action_logits.
    """

    y_true = []
    y_pred = []

    detailed_results = []

    for res in test_results:

        true_label = int(
            res["label_id"]
        )

        logits = np.array(
            res["video_action_logits"]
        )

        predicted_label = int(
            np.argmax(logits)
        )

        y_true.append(
            true_label
        )

        y_pred.append(
            predicted_label
        )

        detailed_results.append(
            {
                "video_id":
                    res["video_id"],

                "ground_truth":
                    class_names[true_label],

                "predicted":
                    class_names[predicted_label],

                "correct":
                    bool(
                        true_label
                        == predicted_label
                    )
            }
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(len(class_names))
            ),
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        )
    )

    per_class = {}

    for i, class_name in enumerate(
        class_names
    ):

        per_class[class_name] = {
            "precision":
                round(
                    float(precision[i]),
                    4
                ),

            "recall":
                round(
                    float(recall[i]),
                    4
                ),

            "f1_score":
                round(
                    float(f1[i]),
                    4
                ),

            "support":
                int(support[i])
        }

    return {
        "classification_evaluation":
            "REAL_MEASURED_VIDEO_LEVEL",

        "num_test_videos":
            int(len(y_true)),

        "correct_predictions":
            int(np.sum(y_true == y_pred)),

        "classification_accuracy":
            round(accuracy, 4),

        "per_class_metrics":
            per_class,

        "confusion_matrix":
            cm.tolist(),

        "class_names":
            class_names,

        "video_predictions":
            detailed_results
    }


def evaluate_test_set(
    checkpoint_path,
    config_path,
    output_dir="reports/metrics"
):

    out_dir = Path(
        output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    splits_dir = Path(
        config["dataset"]["splits_dir"]
    )

    test_csv = (
        splits_dir
        / "test.csv"
    )

    gt_file = (
        splits_dir
        / "temporal_ground_truth.json"
    )

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading checkpoint {checkpoint_path} "
        f"onto {device}..."
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    m_cfg = config["model"]
    d_cfg = config["dataset"]

    model = TwoStreamWSSTALNet(
        num_classes=d_cfg.get(
            "num_classes",
            3
        ),

        rgb_feature_dim=m_cfg.get(
            "rgb_feature_dim",
            512
        ),

        flow_feature_dim=m_cfg.get(
            "flow_feature_dim",
            128
        ),

        fusion_dim=m_cfg.get(
            "fusion_dim",
            256
        ),

        encoder_layers=m_cfg.get(
            "encoder_layers",
            2
        ),

        nheads=m_cfg.get(
            "nheads",
            4
        ),

        pretrained=m_cfg.get(
            "pretrained",
            True
        ),

        freeze_rgb_backbone=m_cfg.get(
            "freeze_rgb_backbone",
            False
        )

    ).to(device)

    # ----------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------

    flow_engine = OpticalFlowEngine(
        target_size=tuple(
            config["video"]["flow_size"]
        )
    )

    localizer = TemporalLocalizer(
        threshold=config[
            "localization"
        ]["threshold"],

        min_duration_seconds=config[
            "localization"
        ]["min_duration_seconds"],

        merge_gap_seconds=config[
            "localization"
        ]["merge_gap_seconds"]
    )

    df_test = pd.read_csv(
        test_csv
    )

    test_results = []

    print(
        f"Evaluating model on "
        f"{len(df_test)} test surveillance videos..."
    )

    IMAGENET_MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(
        1, 1, 3
    )

    IMAGENET_STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(
        1, 1, 3
    )

    # ----------------------------------------------------------
    # Test videos
    # ----------------------------------------------------------

    for idx, row in df_test.iterrows():

        vpath = Path(
            row["path"]
        )

        reader = VideoReader(
            str(vpath),

            target_fps=config[
                "video"
            ]["target_fps"],

            target_size=tuple(
                config[
                    "video"
                ]["image_size"]
            )
        )

        clips = list(
            reader.stream_clips(
                clip_length=config[
                    "video"
                ]["clip_length"],

                stride=config[
                    "video"
                ]["stride"]
            )
        )

        rgb_list = []
        flow_list = []
        timestamps = []

        for clip in clips:

            norm_rgb = (
                clip["frames"].astype(
                    np.float32
                )
                / 255.0
                - IMAGENET_MEAN
            ) / IMAGENET_STD

            flow_data = (
                flow_engine.compute_clip_flow(
                    clip["frames"]
                )
            )

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

        rgb_t = torch.from_numpy(
            np.stack(
                rgb_list,
                axis=0
            )
        ).unsqueeze(0).to(device)

        flow_t = torch.from_numpy(
            np.stack(
                flow_list,
                axis=0
            )
        ).unsqueeze(0).to(device)

        # ------------------------------------------------------
        # Model inference
        # ------------------------------------------------------

        with torch.no_grad():

            outputs = model(
                rgb_t,
                flow_t
            )

        anom_scores = (
            outputs[
                "segment_anomaly_scores"
            ]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        action_probs = (
            outputs[
                "segment_action_probs"
            ]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        video_action_logits = (
            outputs[
                "video_action_logits"
            ]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # ------------------------------------------------------
        # Temporal localization
        # ------------------------------------------------------

        det_events = (
            localizer.localize_events(
                anom_scores,
                action_probs,
                timestamps
            )
        )

        mid_times = [
            (t[0] + t[1]) / 2
            for t in timestamps
        ]

        # ------------------------------------------------------
        # Store complete test result
        # ------------------------------------------------------

        test_results.append(
            {
                "video_id":
                    row["video_id"],

                "label_id":
                    int(row["label_id"]),

                "label":
                    row["label"],

                "duration":
                    float(row["duration"]),

                "timestamps":
                    mid_times,

                "anomaly_scores":
                    anom_scores.tolist(),

                "action_probs":
                    action_probs.tolist(),

                "video_action_logits":
                    video_action_logits.tolist(),

                "detected_events":
                    det_events
            }
        )

    # ==========================================================
    # EVALUATION 1 — ANOMALY / TEMPORAL LOCALIZATION
    # ==========================================================

    evaluator = WSSTALEvaluator(
        str(gt_file),
        threshold=config[
            "localization"
        ]["threshold"]
    )

    anomaly_metrics = (
        evaluator.evaluate_predictions(
            test_results
        )
    )

    # ==========================================================
    # EVALUATION 2 — 3-CLASS ACTION CLASSIFICATION
    # ==========================================================

    class_names = d_cfg.get(
        "classes",
        [
            "Normal",
            "Fighting",
            "Panic_Dispersal"
        ]
    )

    classification_metrics = (
        evaluate_action_classification(
            test_results,
            class_names
        )
    )

    # ==========================================================
    # COMBINED REPORT
    # ==========================================================

    metrics = {
        **anomaly_metrics,
        **classification_metrics
    }

    # ----------------------------------------------------------
    # Print anomaly report
    # ----------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        "SURVEILLANCE EVALUATION REPORT "
        "(TEST SPLIT)"
    )
    print("=" * 65)

    print("\n[ANOMALY / TEMPORAL LOCALIZATION]")

    anomaly_keys = [
        "evaluation_status",
        "detection_threshold",
        "total_evaluated_frames",
        "positive_anomaly_frames",
        "negative_normal_frames",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mean_temporal_iou",
        "temporal_iou_at_0.3",
        "temporal_iou_at_0.5"
    ]

    for key in anomaly_keys:

        print(
            f"  * {key:<26}: "
            f"{metrics[key]}"
        )

    # ----------------------------------------------------------
    # Print classification report
    # ----------------------------------------------------------

    print(
        "\n[3-CLASS ACTION CLASSIFICATION]"
    )

    print(
        f"  * classification_accuracy : "
        f"{classification_metrics['classification_accuracy']}"
    )

    print(
        f"  * correct_predictions     : "
        f"{classification_metrics['correct_predictions']}"
        f"/"
        f"{classification_metrics['num_test_videos']}"
    )

    print("\n  Per-class metrics:")

    for class_name, values in (
        classification_metrics[
            "per_class_metrics"
        ].items()
    ):

        print(
            f"    {class_name}: "
            f"Precision={values['precision']:.4f}, "
            f"Recall={values['recall']:.4f}, "
            f"F1={values['f1_score']:.4f}, "
            f"Support={values['support']}"
        )

    print("\n  Confusion matrix:")

    cm = classification_metrics[
        "confusion_matrix"
    ]

    print(
        "    "
        + " | ".join(
            class_names
        )
    )

    for class_name, row_values in zip(
        class_names,
        cm
    ):

        print(
            f"    {class_name:<18} "
            + " | ".join(
                str(v)
                for v in row_values
            )
        )

    print("\n  Video-level predictions:")

    for item in classification_metrics[
        "video_predictions"
    ]:

        status = (
            "CORRECT"
            if item["correct"]
            else "WRONG"
        )

        print(
            f"    {item['video_id']}: "
            f"GT={item['ground_truth']} | "
            f"Pred={item['predicted']} | "
            f"{status}"
        )

    print("=" * 65)

    # ----------------------------------------------------------
    # Save JSON
    # ----------------------------------------------------------

    report_json = (
        out_dir
        / "test_evaluation_report.json"
    )

    with open(
        report_json,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------

    df_metrics = pd.DataFrame(
        [
            {
                **anomaly_metrics,
                "classification_accuracy":
                    classification_metrics[
                        "classification_accuracy"
                    ],
                "correct_classifications":
                    classification_metrics[
                        "correct_predictions"
                    ],
                "total_test_videos":
                    classification_metrics[
                        "num_test_videos"
                    ]
            }
        ]
    )

    df_metrics.to_csv(
        out_dir
        / "test_evaluation_summary.csv",
        index=False
    )

    # ----------------------------------------------------------
    # Save video-level classification CSV
    # ----------------------------------------------------------

    prediction_df = pd.DataFrame(
        classification_metrics[
            "video_predictions"
        ]
    )

    prediction_df.to_csv(
        out_dir
        / "video_classification_predictions.csv",
        index=False
    )

    print(
        f"\nSaved evaluation metrics to "
        f"{report_json}"
    )

    return metrics