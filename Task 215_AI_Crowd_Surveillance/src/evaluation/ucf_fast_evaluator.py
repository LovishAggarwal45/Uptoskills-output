"""
Fast UCF-Crime binary evaluation.

Evaluates:
    Normal  = 0
    Anomaly = 1

For speed, one representative 16-frame clip is sampled from each video.

Outputs:
    - accuracy
    - precision
    - recall
    - F1
    - ROC-AUC
    - PR-AUC
    - confusion matrix
    - per-video predictions CSV
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
# RGB preparation
# -------------------------------------------------------------------

def prepare_rgb(frames):
    """
    Input:
        frames: [T, H, W, 3]

    Model expects:
        [B, N, T, 3, H, W]

    Here:
        B = 1
        N = 1
    """

    processed = []

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    for frame in frames:

        frame = frame.astype(np.float32) / 255.0

        frame = (frame - mean) / std

        frame = torch.from_numpy(frame)

        frame = frame.permute(2, 0, 1)

        processed.append(frame)

    # [T, 3, H, W]
    rgb = torch.stack(
        processed,
        dim=0,
    )

    # [1, 1, T, 3, H, W]
    rgb = rgb.unsqueeze(0).unsqueeze(0)

    return rgb


# -------------------------------------------------------------------
# Optical-flow preparation
# -------------------------------------------------------------------

def prepare_flow(flow):
    """
    Input:
        flow: [T, 2, H, W]

    Model expects:
        [B, N, T, 2, H, W]

    Here:
        B = 1
        N = 1
    """

    flow = torch.from_numpy(
        flow.astype(np.float32)
    )

    # [1, 1, T, 2, H, W]
    flow = flow.unsqueeze(0).unsqueeze(0)

    return flow


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def load_model(checkpoint, config, device):

    model_cfg = config["model"]

    dataset_cfg = config["dataset"]

    model = TwoStreamWSSTALNet(
        num_classes=dataset_cfg.get(
            "num_classes",
            2,
        ),

        rgb_feature_dim=model_cfg.get(
            "rgb_feature_dim",
            512,
        ),

        flow_feature_dim=model_cfg.get(
            "flow_feature_dim",
            128,
        ),

        fusion_dim=model_cfg.get(
            "fusion_dim",
            256,
        ),

        encoder_layers=model_cfg.get(
            "encoder_layers",
            2,
        ),

        nheads=model_cfg.get(
            "nheads",
            4,
        ),

        dropout=model_cfg.get(
            "dropout",
            0.15,
        ),

        mil_top_k=model_cfg.get(
            "mil_top_k",
            3,
        ),

        pretrained=model_cfg.get(
            "pretrained",
            True,
        ),

        freeze_rgb_backbone=model_cfg.get(
            "freeze_rgb_backbone",
            True,
        ),
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    # ---------------------------------------------------------------
    # Extract state dictionary
    # ---------------------------------------------------------------

    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint_data, dict)
        and "state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data[
            "state_dict"
        ]

    else:
        state_dict = checkpoint_data

    # ---------------------------------------------------------------
    # Remove possible DataParallel "module." prefix
    # ---------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    # ---------------------------------------------------------------
    # Load model weights
    # ---------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(device)

    model.eval()

    return model


# -------------------------------------------------------------------
# Representative clip selection
# -------------------------------------------------------------------

def get_representative_clip(
    reader,
    clip_length=16,
):
    """
    Select one clip near the middle of the video.

    This is intentionally used for fast
    submission-time evaluation.
    """

    clips = list(
        reader.stream_clips(
            clip_length=clip_length,
            stride=clip_length,
        )
    )

    if not clips:
        return None

    middle_index = len(clips) // 2

    return clips[middle_index]


# -------------------------------------------------------------------
# Single-video prediction
# -------------------------------------------------------------------

@torch.no_grad()
def predict_video(
    model,
    video_path,
    flow_engine,
    device,
    clip_length=16,
    threshold=0.52,
):
    """
    Predict one video.

    IMPORTANT:
        threshold is explicitly passed into this function so that
        the per-video predicted_label uses the same threshold as
        the final evaluation.
    """

    reader = VideoReader(
        video_path,
        target_fps=16,
        target_size=(224, 224),
    )

    clip = get_representative_clip(
        reader,
        clip_length=clip_length,
    )

    if clip is None:
        return None

    # ---------------------------------------------------------------
    # RGB frames
    # ---------------------------------------------------------------

    rgb_frames = clip["frames"]

    # ---------------------------------------------------------------
    # Optical flow
    # ---------------------------------------------------------------

    flow_result = flow_engine.compute_clip_flow(
        rgb_frames
    )

    flow = flow_result["flow"]

    # ---------------------------------------------------------------
    # Convert inputs to tensors
    # ---------------------------------------------------------------

    rgb_tensor = prepare_rgb(
        rgb_frames
    )

    flow_tensor = prepare_flow(
        flow
    )

    rgb_tensor = rgb_tensor.to(device)

    flow_tensor = flow_tensor.to(device)

    # ---------------------------------------------------------------
    # Model inference
    # ---------------------------------------------------------------

    outputs = model(
        rgb_tensor,
        flow_tensor,
    )

    # ---------------------------------------------------------------
    # Anomaly score
    # ---------------------------------------------------------------

    anomaly_score = float(
        outputs["video_anomaly_score"]
        .detach()
        .cpu()
        .item()
    )

    # ---------------------------------------------------------------
    # Action probabilities
    # ---------------------------------------------------------------

    action_probs = (
        F.softmax(
            outputs["video_action_logits"],
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()[0]
    )

    # ---------------------------------------------------------------
    # IMPORTANT FIX
    #
    # Use the threshold supplied by main().
    # Previously this variable was undefined.
    # ---------------------------------------------------------------

    predicted_label = int(
        anomaly_score >= threshold
    )

    # ---------------------------------------------------------------
    # Return prediction
    # ---------------------------------------------------------------

    return {
        "anomaly_score": anomaly_score,

        "predicted_label": predicted_label,

        "predicted_class": (
            "Anomaly"
            if predicted_label == 1
            else "Normal"
        ),

        "clip_start": float(
            clip["start_time"]
        ),

        "clip_end": float(
            clip["end_time"]
        ),

        "action_prob_normal": (
            float(action_probs[0])
            if len(action_probs) > 0
            else 0.0
        ),

        "action_prob_anomaly": (
            float(action_probs[1])
            if len(action_probs) > 1
            else 0.0
        ),
    }


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "ucf_binary/"
            "best_model.pt"
        ),
    )

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    parser.add_argument(
        "--config",
        default="configs/ucf_binary.yaml",
    )

    # ---------------------------------------------------------------
    # Test CSV
    # ---------------------------------------------------------------

    parser.add_argument(
        "--test-csv",
        default=(
            "datasets/"
            "metadata/"
            "ucf_binary_test.csv"
        ),
    )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default=(
            "reports/"
            "metrics/"
            "ucf_binary_fast"
        ),
    )

    # ---------------------------------------------------------------
    # Maximum videos
    # ---------------------------------------------------------------

    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help=(
            "0 = evaluate all videos."
        ),
    )

    # ---------------------------------------------------------------
    # Classification threshold
    # ---------------------------------------------------------------

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.52,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------

    config = load_config(
        args.config
    )

    # ---------------------------------------------------------------
    # Select device
    # ---------------------------------------------------------------

    device_name = (
        config
        .get("runtime", {})
        .get("device", "auto")
    )

    if device_name == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device_name
        )

    # ---------------------------------------------------------------
    # Evaluation information
    # ---------------------------------------------------------------

    print("=" * 70)
    print(
        "UCF-CRIME FAST BINARY EVALUATION"
    )
    print("=" * 70)

    print(
        f"Device     : {device}"
    )

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    print(
        f"Test CSV   : {args.test_csv}"
    )

    print(
        f"Threshold  : {args.threshold}"
    )

    print(
        "Sampling   : "
        "1 representative clip/video"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # Load test metadata
    # ---------------------------------------------------------------

    df = pd.read_csv(
        args.test_csv
    )

    if args.max_videos > 0:

        df = df.head(
            args.max_videos
        ).copy()

    print(
        f"Videos to evaluate: {len(df)}"
    )

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model = load_model(
        args.checkpoint,
        config,
        device,
    )

    # ---------------------------------------------------------------
    # Optical flow engine
    # ---------------------------------------------------------------

    flow_engine = OpticalFlowEngine(
        target_size=(112, 112)
    )

    # ---------------------------------------------------------------
    # Prediction storage
    # ---------------------------------------------------------------

    results = []

    # ---------------------------------------------------------------
    # Evaluate every video
    # ---------------------------------------------------------------

    for index, row in df.iterrows():

        video_id = str(
            row["video_id"]
        )

        video_path = Path(
            str(row["path"])
        )

        if not video_path.is_absolute():

            video_path = (
                ROOT / video_path
            )

        true_label = int(
            row["label"]
        )

        print(
            f"[{len(results) + 1}/{len(df)}] "
            f"{video_id}",
            end=" ... ",
            flush=True,
        )

        try:

            # -------------------------------------------------------
            # IMPORTANT FIX:
            # Pass args.threshold into predict_video().
            # -------------------------------------------------------

            prediction = predict_video(
                model=model,
                video_path=video_path,
                flow_engine=flow_engine,
                device=device,
                clip_length=16,
                threshold=args.threshold,
            )

            if prediction is None:

                print("NO CLIP")

                continue

            # -------------------------------------------------------
            # Add metadata
            # -------------------------------------------------------

            prediction["video_id"] = (
                video_id
            )

            prediction["video_path"] = (
                str(video_path)
            )

            prediction["true_label"] = (
                true_label
            )

            prediction["true_class"] = (
                "Anomaly"
                if true_label == 1
                else "Normal"
            )

            prediction["ucf_category"] = str(
                row.get(
                    "ucf_category",
                    "",
                )
            )

            results.append(
                prediction
            )

            print(
                f"true={true_label} "
                f"pred={prediction['predicted_label']} "
                f"score="
                f"{prediction['anomaly_score']:.4f}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    # ---------------------------------------------------------------
    # Check results
    # ---------------------------------------------------------------

    if not results:

        print(
            "\nNo successful predictions."
        )

        return

    # ---------------------------------------------------------------
    # Convert results to DataFrame
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------------
    # Ground-truth labels
    # ---------------------------------------------------------------

    y_true = (
        results_df["true_label"]
        .astype(int)
        .to_numpy()
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Final predictions also use args.threshold.
    #
    # This must match the threshold used inside
    # predict_video().
    # ---------------------------------------------------------------

    y_pred = (
        results_df[
            "anomaly_score"
        ]
        .to_numpy()
        >= args.threshold
    ).astype(int)

    # ---------------------------------------------------------------
    # Anomaly scores
    # ---------------------------------------------------------------

    scores = (
        results_df[
            "anomaly_score"
        ]
        .to_numpy()
    )

    # ---------------------------------------------------------------
    # Classification metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # ROC-AUC and PR-AUC
    # ---------------------------------------------------------------

    if len(
        np.unique(y_true)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true,
            scores,
        )

        pr_auc = average_precision_score(
            y_true,
            scores,
        )

    else:

        roc_auc = None
        pr_auc = None

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------------
    # Metrics dictionary
    # ---------------------------------------------------------------

    metrics = {

        "evaluation":
            "UCF-Crime fast binary evaluation",

        "sampling":
            "one representative middle clip per video",

        "threshold":
            args.threshold,

        "num_videos":
            int(len(results_df)),

        "normal_videos":
            int((y_true == 0).sum()),

        "anomaly_videos":
            int((y_true == 1).sum()),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),

        "pr_auc":
            (
                float(pr_auc)
                if pr_auc is not None
                else None
            ),

        "confusion_matrix": {

            "TN":
                int(tn),

            "FP":
                int(fp),

            "FN":
                int(fn),

            "TP":
                int(tp),
        },
    }

    # ---------------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------------

    csv_path = (
        output_dir
        / "ucf_fast_predictions.csv"
    )

    json_path = (
        output_dir
        / "ucf_fast_metrics.json"
    )

    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Final results
    # ---------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "FINAL UCF TEST RESULTS"
    )

    print("=" * 70)

    print(
        f"Videos evaluated : "
        f"{len(results_df)}"
    )

    print(
        f"Normal           : "
        f"{(y_true == 0).sum()}"
    )

    print(
        f"Anomaly          : "
        f"{(y_true == 1).sum()}"
    )

    print()

    print(
        f"Accuracy         : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score         : "
        f"{f1 * 100:.2f}%"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC          : "
            f"{roc_auc:.4f}"
        )

        print(
            f"PR-AUC            : "
            f"{pr_auc:.4f}"
        )

    print()

    print(
        "Confusion Matrix"
    )

    print(
        "----------------"
    )

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    print()

    print(
        f"Predictions saved: "
        f"{csv_path}"
    )

    print(
        f"Metrics saved    : "
        f"{json_path}"
    )

    print("=" * 70)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()