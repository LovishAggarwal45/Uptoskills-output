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
)


VAL_CSV = Path(
    r"outputs\ucf_multiclip_val\ucf_multiclip_predictions.csv"
)

OUTPUT_CSV = Path(
    r"outputs\ucf_multiclip_val\temporal_persistence_analysis.csv"
)

THRESHOLDS = np.round(
    np.arange(0.01, 0.91, 0.01),
    2,
)

PERSISTENCE_RULES = [1, 2, 3]


def parse_scores(value):
    return [
        float(x)
        for x in ast.literal_eval(str(value))
    ]


if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


df = pd.read_csv(VAL_CSV)

required = ["clip_scores", "true_label"]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


df["scores"] = df["clip_scores"].apply(parse_scores)

y_true = (
    df["true_label"]
    .astype(int)
    .to_numpy()
)


results = []


for threshold in THRESHOLDS:

    for required_positive_clips in PERSISTENCE_RULES:

        predictions = []

        for scores in df["scores"]:

            high_score_count = sum(
                score >= threshold
                for score in scores
            )

            prediction = int(
                high_score_count
                >= required_positive_clips
            )

            predictions.append(prediction)

        y_pred = np.asarray(
            predictions,
            dtype=int,
        )

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

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "required_positive_clips":
                    required_positive_clips,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy":
                    balanced_accuracy,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )


results_df = pd.DataFrame(results)


print("=" * 80)
print("UCF-CRIME TEMPORAL PERSISTENCE ANALYSIS")
print("=" * 80)

print(
    f"Validation videos : {len(df)}"
)

print(
    f"Normal            : {(y_true == 0).sum()}"
)

print(
    f"Anomaly           : {(y_true == 1).sum()}"
)

print()


for rule in PERSISTENCE_RULES:

    rule_df = results_df[
        results_df["required_positive_clips"] == rule
    ]

    best = rule_df.loc[
        rule_df["accuracy"].idxmax()
    ]

    best_f1 = rule_df.loc[
        rule_df["f1"].idxmax()
    ]

    best_balanced = rule_df.loc[
        rule_df["balanced_accuracy"].idxmax()
    ]

    print("=" * 80)
    print(
        f"{rule}-OF-3 PERSISTENCE RULE"
    )
    print("=" * 80)

    print(
        f"Best Accuracy Threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {best['accuracy']:.4f}"
    )

    print(
        f"Precision : {best['precision']:.4f}"
    )

    print(
        f"Recall    : {best['recall']:.4f}"
    )

    print(
        f"F1        : {best['f1']:.4f}"
    )

    print(
        f"Balanced  : "
        f"{best['balanced_accuracy']:.4f}"
    )

    print(
        f"TN={int(best['tn'])} "
        f"FP={int(best['fp'])} "
        f"FN={int(best['fn'])} "
        f"TP={int(best['tp'])}"
    )

    print()

    print(
        f"Best F1 Threshold : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {best_f1['accuracy']:.4f}"
    )

    print(
        f"Precision : {best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : {best_f1['recall']:.4f}"
    )

    print(
        f"F1        : {best_f1['f1']:.4f}"
    )

    print()

    print(
        f"Best Balanced Accuracy "
        f"Threshold : "
        f"{best_balanced['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_balanced['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_balanced['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_balanced['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_balanced['f1']:.4f}"
    )

    print(
        f"Balanced  : "
        f"{best_balanced['balanced_accuracy']:.4f}"
    )

    print()


# ------------------------------------------------------------
# Best validation configuration by accuracy
# ------------------------------------------------------------

best_overall = results_df.loc[
    results_df["accuracy"].idxmax()
]

print("=" * 80)
print("BEST VALIDATION CONFIGURATION")
print("=" * 80)

print(
    f"Persistence rule : "
    f"{int(best_overall['required_positive_clips'])}-of-3"
)

print(
    f"Threshold        : "
    f"{best_overall['threshold']:.2f}"
)

print(
    f"Accuracy         : "
    f"{best_overall['accuracy']:.4f}"
)

print(
    f"Precision        : "
    f"{best_overall['precision']:.4f}"
)

print(
    f"Recall           : "
    f"{best_overall['recall']:.4f}"
)

print(
    f"F1               : "
    f"{best_overall['f1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_overall['balanced_accuracy']:.4f}"
)

print(
    f"TN={int(best_overall['tn'])} "
    f"FP={int(best_overall['fp'])} "
    f"FN={int(best_overall['fn'])} "
    f"TP={int(best_overall['tp'])}"
)


OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print(
    f"Results saved to:\n{OUTPUT_CSV}"
)

print("=" * 80)