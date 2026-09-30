import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

CSV_PATH = r"outputs\ucf_finetune_val\ucf_fast_predictions.csv"
df = pd.read_csv(CSV_PATH)

y_true = df["true_label"].astype(int)
scores = df["anomaly_score"].astype(float)

print("=" * 90)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 90)
print(f"Validation videos : {len(df)}")
print()

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'Bal.Acc':>10}"
)

print("-" * 90)

results = []

for threshold in [i / 100 for i in range(10, 91, 5)]:

    y_pred = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )
    balanced_accuracy = balanced_accuracy_score(
        y_true, y_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred
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

    print(
        f"{threshold:10.2f} "
        f"{accuracy:10.4f} "
        f"{precision:10.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f} "
        f"{balanced_accuracy:10.4f}"
    )

print()
print("=" * 90)
print("BEST THRESHOLDS")
print("=" * 90)

best_f1 = max(results, key=lambda x: x["f1"])
best_balanced = max(
    results,
    key=lambda x: x["balanced_accuracy"]
)
best_accuracy = max(
    results,
    key=lambda x: x["accuracy"]
)

for name, result in [
    ("Best F1", best_f1),
    ("Best Balanced Accuracy", best_balanced),
    ("Best Accuracy", best_accuracy),
]:

    print(f"\n{name}")
    print("-" * 40)
    print(f"Threshold          : {result['threshold']:.2f}")
    print(f"Accuracy           : {result['accuracy']:.4f}")
    print(f"Precision          : {result['precision']:.4f}")
    print(f"Recall             : {result['recall']:.4f}")
    print(f"F1                 : {result['f1']:.4f}")
    print(
        f"Balanced Accuracy  : "
        f"{result['balanced_accuracy']:.4f}"
    )
    print(
        f"Confusion Matrix   : "
        f"TN={result['tn']} "
        f"FP={result['fp']} "
        f"FN={result['fn']} "
        f"TP={result['tp']}"
    )