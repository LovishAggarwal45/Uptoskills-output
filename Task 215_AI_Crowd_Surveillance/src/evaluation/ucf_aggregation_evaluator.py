from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UCF-Crime temporal aggregation evaluator"
    )

    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="Existing multi-clip prediction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["MAX", "MEAN", "TOP2_MEAN"],
        default="TOP2_MEAN",
        help="Temporal aggregation method.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Classification threshold selected from validation.",
    )

    return parser.parse_args()


def parse_clip_scores(value):
    if isinstance(value, list):
        return [float(x) for x in value]

    scores = ast.literal_eval(str(value))

    if not isinstance(scores, list):
        raise ValueError(
            f"Invalid clip score list: {value}"
        )

    return [float(x) for x in scores]


def aggregate_scores(scores, method):

    scores = np.asarray(scores, dtype=float)

    if method == "MAX":
        return float(np.max(scores))

    if method == "MEAN":
        return float(np.mean(scores))

    if method == "TOP2_MEAN":
        top_two = np.sort(scores)[-2:]
        return float(np.mean(top_two))

    raise ValueError(
        f"Unsupported aggregation method: {method}"
    )


def main():

    args = parse_args()

    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"\nPrediction CSV not found:\n"
            f"{predictions_csv}"
        )

    df = pd.read_csv(predictions_csv)

    required = [
        "clip_scores",
        "true_label",
        "video_id",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Parse clip scores
    # --------------------------------------------------------

    df["clip_scores_parsed"] = (
        df["clip_scores"].apply(parse_clip_scores)
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    df["aggregated_score"] = (
        df["clip_scores_parsed"].apply(
            lambda scores: aggregate_scores(
                scores,
                args.method,
            )
        )
    )

    df["predicted_label"] = (
        df["aggregated_score"] >= args.threshold
    ).astype(int)

    df["predicted_class"] = np.where(
        df["predicted_label"] == 1,
        "Anomaly",
        "Normal",
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    y_true = (
        df["true_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["predicted_label"]
        .astype(int)
        .to_numpy()
    )

    scores = (
        df["aggregated_score"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    pr_auc = average_precision_score(
        y_true,
        scores,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_columns = [
        "video_id",
        "video_path",
        "true_label",
        "true_class",
        "ucf_category",
        "clip_scores",
        "aggregated_score",
        "predicted_label",
        "predicted_class",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in df.columns
    ]

    prediction_output = (
        output_dir /
        "ucf_aggregation_predictions.csv"
    )

    df[prediction_columns].to_csv(
        prediction_output,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "experiment": "Experiment 4",
        "evaluation": "held_out_test",
        "aggregation_method": args.method,
        "threshold": args.threshold,
        "videos_evaluated": int(len(df)),
        "normal_videos": int((y_true == 0).sum()),
        "anomaly_videos": int((y_true == 1).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics_output = (
        output_dir /
        "ucf_aggregation_metrics.json"
    )

    with open(
        metrics_output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("=" * 80)
    print("UCF-CRIME EXPERIMENT 4 FINAL TEST")
    print("=" * 80)

    print(
        f"Prediction CSV : {predictions_csv}"
    )

    print(
        f"Aggregation    : {args.method}"
    )

    print(
        f"Threshold      : {args.threshold:.2f}"
    )

    print()

    print(
        f"Videos evaluated : {len(df)}"
    )

    print(
        f"Normal           : {(y_true == 0).sum()}"
    )

    print(
        f"Anomaly          : {(y_true == 1).sum()}"
    )

    print()

    print(
        f"Accuracy         : {accuracy:.4%}"
    )

    print(
        f"Precision        : {precision:.4%}"
    )

    print(
        f"Recall           : {recall:.4%}"
    )

    print(
        f"F1 Score         : {f1:.4%}"
    )

    print(
        f"ROC-AUC          : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC           : {pr_auc:.4f}"
    )

    print()

    print("Confusion Matrix")
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
        "Predictions saved to:"
    )
    print(prediction_output)

    print()

    print(
        "Metrics saved to:"
    )
    print(metrics_output)

    print("=" * 80)


if __name__ == "__main__":
    main()