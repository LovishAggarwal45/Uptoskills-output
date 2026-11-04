
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_FILE = Path("outputs/anpr_realesrgan_results.csv")
GROUND_TRUTH_FILE = Path(
    "outputs/evaluation/ground_truth_verified.csv"
)

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# NORMALIZATION
# ==========================================

def normalize_plate(value):
    if pd.isna(value):
        return ""

    # Preserve O and 0 as different characters.
    return "".join(
        c for c in str(value).upper()
        if c.isalnum()
    )


def exact_match(prediction, actual):
    return normalize_plate(prediction) == normalize_plate(actual)


def character_accuracy(prediction, actual):
    """Character-level accuracy using edit distance."""
    pred = normalize_plate(prediction)
    truth = normalize_plate(actual)

    if not truth:
        return 0.0

    previous = list(range(len(truth) + 1))

    for i, char_p in enumerate(pred, start=1):
        current = [i]

        for j, char_t in enumerate(truth, start=1):
            cost = 0 if char_p == char_t else 1

            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    distance = previous[-1]
    return max(0.0, 1 - distance / len(truth))


# ==========================================
# LOAD DATA
# ==========================================

if not RESULTS_FILE.exists():
    raise FileNotFoundError(RESULTS_FILE)

if not GROUND_TRUTH_FILE.exists():
    raise FileNotFoundError(GROUND_TRUTH_FILE)

results = pd.read_csv(RESULTS_FILE)

ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
# Keep only manually verified ground-truth labels
ground_truth["actual_plate"] = (
    ground_truth["actual_plate"].fillna("").astype(str).str.strip()
)

ground_truth = ground_truth[
    ground_truth["actual_plate"] != ""
].copy()

print("Verified ground-truth rows:", len(ground_truth))

required_results = [
    "frame",
    "car_id",
    "original_prediction",
    "enhanced_prediction",
    "final_voted_plate"
]

required_gt = ["frame", "car_id", "actual_plate"]

for column in required_results:
    if column not in results.columns:
        raise ValueError(f"Missing results column: {column}")

for column in required_gt:
    if column not in ground_truth.columns:
        raise ValueError(f"Missing ground truth column: {column}")

# Prevent accidental duplicate ground-truth labels.
if ground_truth.duplicated(["frame", "car_id"]).any():
    raise ValueError("Duplicate frame + car_id in ground truth.")

# ==========================================
# MATCH RECORDS
# ==========================================

results["frame"] = pd.to_numeric(results["frame"], errors="coerce")
ground_truth["frame"] = pd.to_numeric(
    ground_truth["frame"], errors="coerce"
)

results["car_id"] = pd.to_numeric(results["car_id"], errors="coerce")
ground_truth["car_id"] = pd.to_numeric(
    ground_truth["car_id"], errors="coerce"
)

# Keep only ground-truth columns needed for evaluation
ground_truth_eval = ground_truth[
    ["frame", "car_id", "actual_plate"]
].copy()

data = results.merge(
    ground_truth_eval,
    on=["frame", "car_id"],
    how="inner",
    validate="many_to_one"
)
# Exclude ambiguous duplicate observation
data = data[
    ~((data["frame"] == 336) & (data["car_id"] == 36))
].copy()

print("Evaluation observations:", len(data))

if data.empty:
    raise ValueError(
        "No matching records. Check video, frame numbers and car IDs."
    )

# ==========================================
# EVALUATE EACH METHOD
# ==========================================

methods = {
    "Original OCR": "original_prediction",
    "Enhanced OCR": "enhanced_prediction",
    "Multi-frame Voting": "final_voted_plate"
}

for label, column in methods.items():
    data[label + "_exact"] = data.apply(
        lambda row: exact_match(
            row[column], row["actual_plate"]
        ),
        axis=1
    )

    data[label + "_character_accuracy"] = data.apply(
        lambda row: character_accuracy(
            row[column], row["actual_plate"]
        ),
        axis=1
    )

# ==========================================
# SUMMARY
# ==========================================

summary = []

for label in methods:
    exact_col = label + "_exact"
    char_col = label + "_character_accuracy"

    accuracy = data[exact_col].mean() * 100
    char_accuracy = data[char_col].mean() * 100

    summary.append({
        "Method": label,
        "Matched Observations": len(data),
        "Correct Plates": int(data[exact_col].sum()),
        "Exact Match Accuracy (%)": round(accuracy, 2),
        "Character Accuracy (%)": round(char_accuracy, 2)
    })

summary_df = pd.DataFrame(summary)

print("\n========== OCR EVALUATION ==========\n")
print(summary_df.to_string(index=False))

# ==========================================
# IMPROVEMENT
# ==========================================

original_acc = summary_df.iloc[0]["Exact Match Accuracy (%)"]
enhanced_acc = summary_df.iloc[1]["Exact Match Accuracy (%)"]
voting_acc = summary_df.iloc[2]["Exact Match Accuracy (%)"]

print("\n========== IMPROVEMENT ==========")
print(f"Original OCR: {original_acc}%")
print(f"Enhanced OCR: {enhanced_acc}%")
print(f"Voting OCR:   {voting_acc}%")
print(f"Enhancement improvement: {enhanced_acc - original_acc:+.2f} percentage points")

# ==========================================
# SAVE REPORTS
# ==========================================

summary_df.to_csv(
    OUTPUT_DIR / "ocr_accuracy_summary.csv",
    index=False
)

data.to_csv(
    OUTPUT_DIR / "ocr_frame_level_evaluation.csv",
    index=False
)

print("\nEvaluation files saved:")
print(OUTPUT_DIR / "ocr_accuracy_summary.csv")
print(OUTPUT_DIR / "ocr_frame_level_evaluation.csv")
