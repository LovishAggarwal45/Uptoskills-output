"""
UCF-Crime Multi-Clip Binary Evaluation.

Experiment 3:
    - Samples multiple temporal clips from each video.
    - Runs the existing trained model on every sampled clip.
    - Uses the maximum clip anomaly score as the video-level score.
    - Does NOT tune the threshold on the test set.

This evaluator is intentionally separate from ucf_fast_evaluator.py
so the previous 88.68% experiment remains reproducible.
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_rgb(frames):
    """
    Input:
        [T, H, W, 3]

    Output:
        [1, 1, T, 3, H, W]
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

    rgb = torch.stack(processed, dim=0)

    return rgb.unsqueeze(0).unsqueeze(0)


def prepare_flow(flow):
    """
    Input:
        [T, 2, H, W]

    Output:
        [1, 1, T, 2, H, W]
    """

    flow = torch.from_numpy(
        flow.astype(np.float32)
    )

    return flow.unsqueeze(0).unsqueeze(0)


def load_model(checkpoint, config, device):

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model = TwoStreamWSSTALNet(
        num_classes=dataset_cfg.get("num_classes", 2),
        rgb_feature_dim=model_cfg.get(
            "rgb_feature_dim", 512
        ),
        flow_feature_dim=model_cfg.get(
            "flow_feature_dim", 128
        ),
        fusion_dim=model_cfg.get(
            "fusion_dim", 256
        ),
        encoder_layers=model_cfg.get(
            "encoder_layers", 2
        ),
        nheads=model_cfg.get(
            "nheads", 4
        ),
        dropout=model_cfg.get(
            "dropout", 0.15
        ),
        mil_top_k=model_cfg.get(
            "mil_top_k", 3
        ),
        pretrained=model_cfg.get(
            "pretrained", True
        ),
        freeze_rgb_backbone=model_cfg.get(
            "freeze_rgb_backbone", True
        ),
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data["model_state_dict"]

    elif (
        isinstance(checkpoint_data, dict)
        and "state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data["state_dict"]

    else:
        state_dict = checkpoint_data

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def select_temporal_clips(
    reader,
    clip_length=16,
    num_clips=3,
):
    """
    Select approximately uniform temporal clips.

    For three clips this gives approximately:
        early
        middle
        late
    """

    clips = list(
        reader.stream_clips(
            clip_length=clip_length,
            stride=clip_length,
        )
    )

    if not clips:
        return []

    if len(clips) <= num_clips:
        return clips

    indices = np.linspace(
        0,
        len(clips) - 1,
        num=num_clips,
        dtype=int,
    )

    indices = np.unique(indices)

    return [
        clips[int(index)]
        for index in indices
    ]


@torch.no_grad()
def predict_video(
    model,
    video_path,
    flow_engine,
    device,
    clip_length=16,
    num_clips=3,
):
    reader = VideoReader(
        video_path,
        target_fps=16,
        target_size=(224, 224),
    )

    clips = select_temporal_clips(
        reader,
        clip_length=clip_length,
        num_clips=num_clips,
    )

    if not clips:
        return None

    clip_results = []

    for clip in clips:

        rgb_frames = clip["frames"]

        flow_result = flow_engine.compute_clip_flow(
            rgb_frames
        )

        flow = flow_result["flow"]

        rgb_tensor = prepare_rgb(
            rgb_frames
        ).to(device)

        flow_tensor = prepare_flow(
            flow
        ).to(device)

        outputs = model(
            rgb_tensor,
            flow_tensor,
        )

        anomaly_score = float(
            outputs["video_anomaly_score"]
            .detach()
            .cpu()
            .item()
        )

        action_probs = (
            F.softmax(
                outputs["video_action_logits"],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()[0]
        )

        clip_results.append(
            {
                "anomaly_score": anomaly_score,
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
        )

    # Maximum score represents the strongest
    # anomaly evidence anywhere in the sampled clips.
    best_clip = max(
        clip_results,
        key=lambda x: x["anomaly_score"],
    )

    video_score = best_clip["anomaly_score"]

    return {
        "anomaly_score": video_score,
        "best_clip_start": best_clip["clip_start"],
        "best_clip_end": best_clip["clip_end"],
        "num_clips_evaluated": len(clip_results),
        "clip_scores": [
            item["anomaly_score"]
            for item in clip_results
        ],
        "action_prob_normal": (
            best_clip["action_prob_normal"]
        ),
        "action_prob_anomaly": (
            best_clip["action_prob_anomaly"]
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "ucf_binary_finetune/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/ucf_binary.yaml",
    )

    parser.add_argument(
        "--csv",
        default=(
            "datasets/metadata/"
            "ucf_binary_val.csv"
        ),
        help=(
            "CSV containing videos to evaluate."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "ucf_multiclip_val"
        ),
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help=(
            "0 = evaluate all videos."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help=(
            "Number of uniformly distributed "
            "temporal clips per video."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(
        args.config
    )

    device_name = config.get(
        "runtime", {}
    ).get(
        "device",
        "auto",
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

    print("=" * 70)
    print(
        "UCF-CRIME MULTI-CLIP BINARY EVALUATION"
    )
    print("=" * 70)

    print(f"Device       : {device}")
    print(
        f"Checkpoint   : {args.checkpoint}"
    )
    print(
        f"CSV          : {args.csv}"
    )
    print(
        f"Threshold    : {args.threshold}"
    )
    print(
        f"Temporal clips/video : "
        f"{args.num_clips}"
    )
    print(
        "Aggregation  : maximum clip score"
    )

    print("=" * 70)

    df = pd.read_csv(
        args.csv
    )

    if args.max_videos > 0:
        df = df.head(
            args.max_videos
        ).copy()

    print(
        f"Videos to evaluate: {len(df)}"
    )

    model = load_model(
        args.checkpoint,
        config,
        device,
    )

    flow_engine = OpticalFlowEngine(
        target_size=(112, 112)
    )

    results = []

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

            prediction = predict_video(
                model=model,
                video_path=video_path,
                flow_engine=flow_engine,
                device=device,
                clip_length=16,
                num_clips=args.num_clips,
            )

            if prediction is None:
                print("NO CLIPS")
                continue

            prediction[
                "video_id"
            ] = video_id

            prediction[
                "video_path"
            ] = str(video_path)

            prediction[
                "true_label"
            ] = true_label

            prediction[
                "true_class"
            ] = (
                "Anomaly"
                if true_label == 1
                else "Normal"
            )

            prediction[
                "ucf_category"
            ] = str(
                row.get(
                    "ucf_category",
                    "",
                )
            )

            prediction[
                "predicted_label"
            ] = int(
                prediction[
                    "anomaly_score"
                ] >= args.threshold
            )

            prediction[
                "predicted_class"
            ] = (
                "Anomaly"
                if prediction[
                    "predicted_label"
                ] == 1
                else "Normal"
            )

            results.append(
                prediction
            )

            print(
                f"true={true_label} "
                f"pred="
                f"{prediction['predicted_label']} "
                f"score="
                f"{prediction['anomaly_score']:.4f} "
                f"clips="
                f"{prediction['num_clips_evaluated']}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    if not results:

        print(
            "\nNo successful predictions."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    y_true = (
        results_df[
            "true_label"
        ]
        .astype(int)
        .to_numpy()
    )

    scores = (
        results_df[
            "anomaly_score"
        ]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        scores >= args.threshold
    ).astype(int)

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {

        "evaluation":
            "UCF-Crime multi-clip binary evaluation",

        "sampling":
            "uniformly distributed temporal clips",

        "num_clips_per_video":
            int(args.num_clips),

        "aggregation":
            "maximum clip anomaly score",

        "threshold":
            float(args.threshold),

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

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }

    csv_path = (
        output_dir
        / "ucf_multiclip_predictions.csv"
    )

    json_path = (
        output_dir
        / "ucf_multiclip_metrics.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

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

    print()
    print("=" * 70)
    print(
        "FINAL MULTI-CLIP RESULTS"
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


if __name__ == "__main__":
    main()