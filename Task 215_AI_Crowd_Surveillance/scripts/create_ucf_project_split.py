from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42

ANOMALY_TRAIN = 512
ANOMALY_VAL = 129

NORMAL_TRAIN = 75
NORMAL_VAL = 25
NORMAL_TEST = 50


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "ucf_binary_metadata.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "metadata"


def print_distribution(name, df):
    print(f"\n{name}")
    print("-" * 50)
    print(f"Total: {len(df)}")

    print(df["label"].value_counts().sort_index())

    anomaly_df = df[df["label"] == 1]

    if len(anomaly_df) > 0 and "ucf_category" in anomaly_df.columns:
        print("\nAnomaly categories:")
        print(
            anomaly_df["ucf_category"]
            .value_counts()
            .sort_index()
        )


def main():

    print("=" * 70)
    print("UCF-CRIME PROJECT-SPECIFIC BINARY SPLIT")
    print("=" * 70)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    print(f"\nLoaded metadata: {len(df)} rows")

    # ---------------------------------------------------------
    # Separate anomaly and normal videos
    # ---------------------------------------------------------

    anomaly_train_candidates = df[
        (df["label"] == 1)
        & (df["split"] == "train")
    ].copy()

    anomaly_test = df[
        (df["label"] == 1)
        & (df["split"] == "test")
    ].copy()

    normal = df[df["label"] == 0].copy()

    print("\nAvailable data")
    print("-" * 50)
    print(
        f"Anomaly training candidates : "
        f"{len(anomaly_train_candidates)}"
    )
    print(f"Anomaly test                : {len(anomaly_test)}")
    print(f"Normal total                : {len(normal)}")

    if len(anomaly_train_candidates) != 641:
        raise ValueError(
            "Expected 641 anomaly training candidates, "
            f"found {len(anomaly_train_candidates)}."
        )

    if len(anomaly_test) != 109:
        raise ValueError(
            "Expected 109 anomaly test videos, "
            f"found {len(anomaly_test)}."
        )

    if len(normal) != 150:
        raise ValueError(
            "Expected 150 normal videos, "
            f"found {len(normal)}."
        )

    # ---------------------------------------------------------
    # Create anomaly train/validation split
    #
    # Stratification preserves the distribution of UCF
    # anomaly categories as much as possible.
    # ---------------------------------------------------------

    if "ucf_category" in anomaly_train_candidates.columns:

        anomaly_train, anomaly_val = train_test_split(
            anomaly_train_candidates,
            train_size=ANOMALY_TRAIN,
            test_size=ANOMALY_VAL,
            random_state=SEED,
            stratify=anomaly_train_candidates["ucf_category"],
        )

    else:

        anomaly_train, anomaly_val = train_test_split(
            anomaly_train_candidates,
            train_size=ANOMALY_TRAIN,
            test_size=ANOMALY_VAL,
            random_state=SEED,
            shuffle=True,
        )

    anomaly_train = anomaly_train.reset_index(drop=True)
    anomaly_val = anomaly_val.reset_index(drop=True)

    # ---------------------------------------------------------
    # Split the 150 normal videos
    # ---------------------------------------------------------

    rng = np.random.default_rng(SEED)

    shuffled_indices = rng.permutation(len(normal))

    normal = (
        normal
        .iloc[shuffled_indices]
        .reset_index(drop=True)
    )

    normal_train = normal.iloc[
        :NORMAL_TRAIN
    ].copy()

    normal_val = normal.iloc[
        NORMAL_TRAIN:
        NORMAL_TRAIN + NORMAL_VAL
    ].copy()

    normal_test = normal.iloc[
        NORMAL_TRAIN + NORMAL_VAL:
        NORMAL_TRAIN + NORMAL_VAL + NORMAL_TEST
    ].copy()

    # ---------------------------------------------------------
    # Verify sizes
    # ---------------------------------------------------------

    assert len(anomaly_train) == 512
    assert len(anomaly_val) == 129
    assert len(anomaly_test) == 109

    assert len(normal_train) == 75
    assert len(normal_val) == 25
    assert len(normal_test) == 50

    # ---------------------------------------------------------
    # Build final project-specific splits
    # ---------------------------------------------------------

    train_df = pd.concat(
        [anomaly_train, normal_train],
        ignore_index=True,
    )

    val_df = pd.concat(
        [anomaly_val, normal_val],
        ignore_index=True,
    )

    test_df = pd.concat(
        [anomaly_test, normal_test],
        ignore_index=True,
    )

    # Shuffle rows
    train_df = (
        train_df
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Leakage check
    # ---------------------------------------------------------

    path_column = None

    for candidate in [
        "video_path",
        "path",
        "filepath",
        "file_path",
    ]:
        if candidate in df.columns:
            path_column = candidate
            break

    if path_column is not None:

        train_paths = set(train_df[path_column])
        val_paths = set(val_df[path_column])
        test_paths = set(test_df[path_column])

        assert not train_paths & val_paths
        assert not train_paths & test_paths
        assert not val_paths & test_paths

        print("\nNo video-path leakage detected.")

    # ---------------------------------------------------------
    # Backup previous generated split
    # ---------------------------------------------------------

    train_file = OUTPUT_DIR / "ucf_binary_train.csv"
    val_file = OUTPUT_DIR / "ucf_binary_val.csv"
    test_file = OUTPUT_DIR / "ucf_binary_test.csv"

    if train_file.exists():
        backup = (
            OUTPUT_DIR
            / "ucf_binary_train_previous_backup.csv"
        )
        train_file.replace(backup)

    if val_file.exists():
        backup = (
            OUTPUT_DIR
            / "ucf_binary_val_previous_backup.csv"
        )
        val_file.replace(backup)

    if test_file.exists():
        backup = (
            OUTPUT_DIR
            / "ucf_binary_test_previous_backup.csv"
        )
        test_file.replace(backup)

    # ---------------------------------------------------------
    # Save final splits
    # ---------------------------------------------------------

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Clearly named copies
    train_df.to_csv(
        OUTPUT_DIR / "ucf_binary_project_train.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR / "ucf_binary_project_val.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "ucf_binary_project_test.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print_distribution("TRAIN", train_df)
    print_distribution("VALIDATION", val_df)
    print_distribution("TEST", test_df)

    print("\n" + "=" * 70)
    print("SPLIT CREATION COMPLETED")
    print("=" * 70)

    print("\nFinal split:")
    print(f"  Train      : {len(train_df)} videos")
    print(f"  Validation : {len(val_df)} videos")
    print(f"  Test       : {len(test_df)} videos")

    print("\nExpected:")
    print("  Train      : 75 Normal + 512 Anomaly = 587")
    print("  Validation : 25 Normal + 129 Anomaly = 154")
    print("  Test       : 50 Normal + 109 Anomaly = 159")


if __name__ == "__main__":
    main()
