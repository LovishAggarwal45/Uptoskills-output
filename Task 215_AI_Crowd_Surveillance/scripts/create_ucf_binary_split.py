#!/usr/bin/env python3

"""
Create train/validation splits for the UCF-Crime binary experiment.

Input:
    datasets/metadata/ucf_binary_metadata.csv

Output:
    datasets/metadata/ucf_binary_train.csv
    datasets/metadata/ucf_binary_val.csv
    datasets/metadata/ucf_binary_test.csv

Important:
    - Existing test videos are NEVER moved into training/validation.
    - The 150 normal videos remain test-only.
    - Only the 641 anomaly training videos are split into train/validation.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "ucf_binary_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
)

TRAIN_OUTPUT = OUTPUT_DIR / "ucf_binary_train.csv"
VAL_OUTPUT = OUTPUT_DIR / "ucf_binary_val.csv"
TEST_OUTPUT = OUTPUT_DIR / "ucf_binary_test.csv"


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UCF-CRIME BINARY TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal metadata rows: {len(df)}")

    # --------------------------------------------------------
    # Separate official train candidates and official test
    # --------------------------------------------------------

    anomaly_train = df[
        (df["split"] == "train")
        & (df["label"] == 1)
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(f"Anomaly training candidates: {len(anomaly_train)}")
    print(f"Untouched test videos:       {len(test_df)}")

    # --------------------------------------------------------
    # Stratified train / validation split
    # --------------------------------------------------------
    #
    # UCF category is used for stratification.
    #
    # This keeps the category distribution approximately
    # consistent between train and validation.
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        anomaly_train,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        stratify=anomaly_train["ucf_category"],
    )

    # --------------------------------------------------------
    # Sort for reproducibility
    # --------------------------------------------------------

    train_df = train_df.sort_values(
        ["ucf_category", "video_id"]
    ).reset_index(drop=True)

    val_df = val_df.sort_values(
        ["ucf_category", "video_id"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["class_name", "ucf_category", "video_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    val_df.to_csv(
        VAL_OUTPUT,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print("\nDataset sizes:")
    print(f"  Train      : {len(train_df)}")
    print(f"  Validation : {len(val_df)}")
    print(f"  Test       : {len(test_df)}")

    print("\nTrain / validation category distribution:")

    distribution = pd.DataFrame(
        {
            "train": train_df["ucf_category"].value_counts(),
            "validation": val_df["ucf_category"].value_counts(),
        }
    ).fillna(0).astype(int)

    print(distribution.to_string())

    print("\nTest class distribution:")

    print(
        test_df["class_name"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_ids = set(train_df["video_id"])
    val_ids = set(val_df["video_id"])
    test_ids = set(test_df["video_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\nLeakage checks:")
    print(f"  Train ∩ Validation : {len(train_val_overlap)}")
    print(f"  Train ∩ Test       : {len(train_test_overlap)}")
    print(f"  Validation ∩ Test  : {len(val_test_overlap)}")

    print("\nOutput files:")

    print(f"  {TRAIN_OUTPUT}")
    print(f"  {VAL_OUTPUT}")
    print(f"  {TEST_OUTPUT}")

    print("\nImportant:")
    print(
        "The test set has not been used to create the "
        "training or validation split."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()