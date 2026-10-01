from pathlib import Path

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

CSV_PATH = Path(
    r"outputs\ucf_multiclip_val\ucf_multiclip_predictions.csv"
)

THRESHOLDS = np.round(np.arange(0.05, 0.91, 0.05), 2)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nValidation prediction file not found:\n{CSV_PATH}\n\n"
        "Run the multi-clip validation evaluator first."
    )

df = pd.read_csv(CSV_PATH)

required_columns = [
    "anomaly_score",
    "true_label",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

y_true = df["true_label"].astype(int).to_numpy()
scores = df["anomaly_score"].astype(float).to_numpy()

print("=" * 70)
print("UCF-CRIME MULTI-CLIP VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

print(f"CSV file        : {CSV_PATH}")
print(f"Validation videos: {len(df)}")
print(f"Normal videos   : {(y_true == 0).sum()}")
print(f"Anomaly videos  : {(y_true == 1).sum()}")
print()

# Check score range
print(
    f"Score range     : {scores.min():.4f} - {scores.max():.4f}"
)

print()


# ============================================================
# THRESHOLD SWEEP
# ============================================================

results = []

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
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )


results_df = pd.DataFrame(results)


# ============================================================
# PRINT THRESHOLD TABLE
# ============================================================

print("THRESHOLD SWEEP")
print("-" * 70)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>11} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'Bal.Acc':>10}"
)

for _, row in results_df.iterrows():

    print(
        f"{row['threshold']:10.2f} "
        f"{row['accuracy']:10.4f} "
        f"{row['precision']:11.4f} "
        f"{row['recall']:10.4f} "
        f"{row['f1']:10.4f} "
        f"{row['balanced_accuracy']:10.4f}"
    )


# ============================================================
# BEST THRESHOLDS
# ============================================================

best_accuracy = results_df.loc[
    results_df["accuracy"].idxmax()
]

best_precision = results_df.loc[
    results_df["precision"].idxmax()
]

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_balanced = results_df.loc[
    results_df["balanced_accuracy"].idxmax()
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_best_result(title, row):
    print()
    print(title)
    print("-" * 50)

    print(f"Threshold          : {row['threshold']:.2f}")
    print(f"Accuracy           : {row['accuracy']:.4f}")
    print(f"Precision          : {row['precision']:.4f}")
    print(f"Recall             : {row['recall']:.4f}")
    print(f"F1                 : {row['f1']:.4f}")
    print(
        f"Balanced Accuracy : "
        f"{row['balanced_accuracy']:.4f}"
    )

    print(
        f"Confusion Matrix   : "
        f"TN={int(row['tn'])} "
        f"FP={int(row['fp'])} "
        f"FN={int(row['fn'])} "
        f"TP={int(row['tp'])}"
    )


# ============================================================
# PRINT BEST RESULTS
# ============================================================

print()
print("=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)

print_best_result(
    "Best Accuracy",
    best_accuracy,
)

print_best_result(
    "Best Precision",
    best_precision,
)

print_best_result(
    "Best Recall",
    best_recall,
)

print_best_result(
    "Best F1",
    best_f1,
)

print_best_result(
    "Best Balanced Accuracy",
    best_balanced,
)


# ============================================================
# THRESHOLD-INDEPENDENT METRICS
# ============================================================

try:
    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    pr_auc = average_precision_score(
        y_true,
        scores,
    )

    print()
    print("=" * 70)
    print("THRESHOLD-INDEPENDENT METRICS")
    print("=" * 70)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

except ValueError as exc:
    print()
    print("Could not calculate ROC-AUC / PR-AUC:")
    print(exc)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

output_path = Path(
    r"outputs\ucf_multiclip_val\ucf_multiclip_threshold_analysis.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    output_path,
    index=False,
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Threshold analysis saved to:\n{output_path}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "The selected threshold must be chosen using the "
    "validation set only."
)

print(
    "Do NOT use the held-out test set to select the threshold."
)

print("=" * 70)
