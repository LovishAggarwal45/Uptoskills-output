from pathlib import Path
import ast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_CSV = Path(
    r"outputs\ucf_multiclip_val\ucf_multiclip_predictions.csv"
)

OUTPUT_CSV = Path(
    r"outputs\ucf_multiclip_val\temporal_aggregation_analysis.csv"
)

THRESHOLDS = np.round(np.arange(0.05, 0.91, 0.05), 2)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"\nValidation prediction file not found:\n{VAL_CSV}\n\n"
        "Run the multi-clip validation evaluator first."
    )

df = pd.read_csv(VAL_CSV)

required_columns = [
    "clip_scores",
    "true_label",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# PARSE CLIP SCORES
# ============================================================

def parse_clip_scores(value):
    """
    Convert the CSV string representation of a list
    into a Python list of floating-point scores.
    """
    if isinstance(value, list):
        return [float(x) for x in value]

    scores = ast.literal_eval(str(value))

    if not isinstance(scores, list):
        raise ValueError(
            f"Expected list of clip scores, got: {type(scores)}"
        )

    return [float(x) for x in scores]


df["clip_scores_parsed"] = df["clip_scores"].apply(parse_clip_scores)


# Verify that every video has three clips.
clip_counts = df["clip_scores_parsed"].apply(len)

if not (clip_counts == 3).all():
    print("WARNING: Not every video contains exactly 3 clips.")
    print(clip_counts.value_counts().sort_index())


y_true = df["true_label"].astype(int).to_numpy()


# ============================================================
# TEMPORAL AGGREGATION METHODS
# ============================================================

def aggregate_scores(scores, method):
    scores = np.asarray(scores, dtype=float)

    if method == "MAX":
        return float(np.max(scores))

    if method == "MEAN":
        return float(np.mean(scores))

    if method == "TOP2_MEAN":
        top_two = np.sort(scores)[-2:]
        return float(np.mean(top_two))

    raise ValueError(f"Unknown aggregation method: {method}")


METHODS = [
    "MAX",
    "MEAN",
    "TOP2_MEAN",
]


for method in METHODS:
    df[f"score_{method.lower()}"] = df[
        "clip_scores_parsed"
    ].apply(
        lambda scores: aggregate_scores(scores, method)
    )


# ============================================================
# DISPLAY BASIC SCORE INFORMATION
# ============================================================

print("=" * 80)
print("UCF-CRIME EXPERIMENT 4")
print("TEMPORAL AGGREGATION COMPARISON")
print("=" * 80)

print(f"Validation CSV : {VAL_CSV}")
print(f"Videos         : {len(df)}")
print(f"Normal         : {(y_true == 0).sum()}")
print(f"Anomaly        : {(y_true == 1).sum()}")
print()

for method in METHODS:
    scores = df[f"score_{method.lower()}"].to_numpy()

    print(
        f"{method:10s} "
        f"range = {scores.min():.4f} - {scores.max():.4f} "
        f"mean = {scores.mean():.4f}"
    )

print()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for method in METHODS:

    scores = df[f"score_{method.lower()}"].to_numpy()

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    for threshold in THRESHOLDS:

        y_pred = (scores >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
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
        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "method": method,
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )


results_df = pd.DataFrame(results)


# ============================================================
# PRINT RESULTS
# ============================================================

for method in METHODS:

    method_df = results_df[
        results_df["method"] == method
    ].copy()

    print("=" * 80)
    print(f"{method} AGGREGATION")
    print("=" * 80)

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Bal.Acc':>10}"
    )

    for _, row in method_df.iterrows():

        print(
            f"{row['threshold']:10.2f} "
            f"{row['accuracy']:10.4f} "
            f"{row['precision']:11.4f} "
            f"{row['recall']:10.4f} "
            f"{row['f1']:10.4f} "
            f"{row['balanced_accuracy']:10.4f}"
        )

    # Best accuracy
    best_accuracy = method_df.loc[
        method_df["accuracy"].idxmax()
    ]

    # Best F1
    best_f1 = method_df.loc[
        method_df["f1"].idxmax()
    ]

    # Best balanced accuracy
    best_balanced = method_df.loc[
        method_df["balanced_accuracy"].idxmax()
    ]

    print()
    print("BEST ACCURACY")
    print(
        f"Threshold : {best_accuracy['threshold']:.2f}\n"
        f"Accuracy  : {best_accuracy['accuracy']:.4f}\n"
        f"Precision : {best_accuracy['precision']:.4f}\n"
        f"Recall    : {best_accuracy['recall']:.4f}\n"
        f"F1        : {best_accuracy['f1']:.4f}\n"
        f"TN={int(best_accuracy['tn'])} "
        f"FP={int(best_accuracy['fp'])} "
        f"FN={int(best_accuracy['fn'])} "
        f"TP={int(best_accuracy['tp'])}"
    )

    print()
    print("BEST F1")
    print(
        f"Threshold : {best_f1['threshold']:.2f}\n"
        f"Accuracy  : {best_f1['accuracy']:.4f}\n"
        f"Precision : {best_f1['precision']:.4f}\n"
        f"Recall    : {best_f1['recall']:.4f}\n"
        f"F1        : {best_f1['f1']:.4f}"
    )

    print()
    print("BEST BALANCED ACCURACY")
    print(
        f"Threshold : {best_balanced['threshold']:.2f}\n"
        f"Accuracy  : {best_balanced['accuracy']:.4f}\n"
        f"Precision : {best_balanced['precision']:.4f}\n"
        f"Recall    : {best_balanced['recall']:.4f}\n"
        f"F1        : {best_balanced['f1']:.4f}\n"
        f"Balanced Accuracy : "
        f"{best_balanced['balanced_accuracy']:.4f}"
    )

    print()


# ============================================================
# OVERALL COMPARISON
# ============================================================

print("=" * 80)
print("OVERALL VALIDATION COMPARISON")
print("=" * 80)

summary_rows = []

for method in METHODS:

    method_df = results_df[
        results_df["method"] == method
    ]

    best = method_df.loc[
        method_df["accuracy"].idxmax()
    ]

    summary_rows.append(
        {
            "method": method,
            "best_threshold": best["threshold"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "balanced_accuracy": best["balanced_accuracy"],
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
            "tn": int(best["tn"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tp": int(best["tp"]),
        }
    )

summary_df = pd.DataFrame(summary_rows)

print(
    summary_df[
        [
            "method",
            "best_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print("=" * 80)
print("EXPERIMENT 4 COMPLETE")
print("=" * 80)
print(f"Detailed results saved to:")
print(OUTPUT_CSV)
print()
print(
    "IMPORTANT: Threshold selection is performed using "
    "validation data only."
)
print(
    "The held-out test set must NOT be used to select "
    "the aggregation method or threshold."
)
print("=" * 80)