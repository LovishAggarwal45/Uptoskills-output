
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("outputs/evaluation")
base.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "ocr_accuracy_summary.csv")

metrics = [
    ("Exact Match Accuracy (%)", "exact_match_accuracy.png"),
    ("Character Accuracy (%)", "character_accuracy.png")
]

for metric, filename in metrics:
    plt.figure(figsize=(9, 5))

    plt.bar(df["Method"], df[metric])

    plt.title(metric)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=10)

    for i, value in enumerate(df[metric]):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(base / filename, dpi=300)
    plt.close()

print("Evaluation charts generated successfully!")
print("Saved in:", base)
