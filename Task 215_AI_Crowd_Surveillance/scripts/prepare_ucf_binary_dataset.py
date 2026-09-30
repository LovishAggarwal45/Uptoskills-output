#!/usr/bin/env python3

"""
Prepare a binary UCF-Crime metadata file.

Binary labels:
    0 = Normal
    1 = Anomaly

Sources:
    - Official UCF-Crime anomaly train split
    - Official UCF-Crime anomaly test split
    - Testing_Normal_Videos

Important:
    This script only includes videos that actually exist locally.
    It does not download or modify video files.
"""

from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "metadata"
OUTPUT_FILE = OUTPUT_DIR / "ucf_binary_metadata.csv"

# Local UCF-Crime anomaly videos
ANOMALY_ROOTS = [
    DOWNLOADS / "Anomaly-Videos-Part-1" / "Anomaly-Videos-Part-1",
    DOWNLOADS / "Anomaly-Videos-Part-2" / "Anomaly-Videos-Part-2",
    DOWNLOADS / "Anomaly-Videos-Part-3" / "Anomaly-Videos-Part-3",
]

# Local normal videos
NORMAL_ROOT = (
    DOWNLOADS
    / "Testing_Normal_Videos"
    / "Testing_Normal_Videos_Anomaly"
)

# Official split files
ANOMALY_TRAIN_FILE = DOWNLOADS / "Anomaly_Train.txt"

ANOMALY_TEST_FILE = (
    DOWNLOADS
    / "UCF_Splits"
    / "Anomaly_Detection_splits"
    / "Anomaly_Test.txt"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name: str) -> str:
    """
    Normalize a relative UCF-Crime path for matching.
    """
    return name.replace("\\", "/").strip().lower()


def load_split_file(path: Path):
    """
    Read a UCF-Crime split file.

    Returns:
        set of normalized relative video paths
    """
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    entries = set()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Split files may contain extra whitespace.
            parts = line.split()

            # The video path is normally the first field.
            relative_path = parts[0]

            entries.add(normalize_name(relative_path))

    return entries


def build_video_index(roots):
    """
    Index all local anomaly videos.

    Returns:
        dictionary:
            normalized relative path -> actual local Path
    """
    index = {}

    for root in roots:
        if not root.exists():
            print(f"[WARNING] Missing anomaly root: {root}")
            continue

        for video in root.rglob("*.mp4"):
            relative = video.relative_to(root)

            key = normalize_name(str(relative))

            # Keep the first occurrence if a duplicate exists.
            if key not in index:
                index[key] = video

    return index


def build_normal_index(root):
    """
    Index local normal videos.

    Returns:
        dictionary:
            filename -> actual local Path
    """
    index = {}

    if not root.exists():
        raise FileNotFoundError(
            f"Normal-video directory not found:\n{root}"
        )

    for video in root.rglob("*.mp4"):
        key = normalize_name(video.name)

        if key not in index:
            index[key] = video

    return index


def category_from_path(relative_path: str) -> str:
    """
    Extract UCF-Crime anomaly category.

    Example:
        Fighting/Fighting001_x264.mp4
        -> Fighting
    """
    parts = relative_path.replace("\\", "/").split("/")

    if len(parts) >= 2:
        return parts[0]

    return "Unknown"


# ============================================================
# MAIN DATASET PREPARATION
# ============================================================

def main():

    print("=" * 70)
    print("UCF-CRIME BINARY DATASET PREPARATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load official split files
    # --------------------------------------------------------

    print("\n[1/5] Loading official split files...")

    train_entries = load_split_file(ANOMALY_TRAIN_FILE)
    test_entries = load_split_file(ANOMALY_TEST_FILE)

    print(f"Official anomaly train entries : {len(train_entries)}")
    print(f"Official anomaly test entries  : {len(test_entries)}")

    # --------------------------------------------------------
    # 2. Index local anomaly videos
    # --------------------------------------------------------

    print("\n[2/5] Indexing local anomaly videos...")

    anomaly_index = build_video_index(ANOMALY_ROOTS)

    print(f"Local anomaly videos found     : {len(anomaly_index)}")

    # --------------------------------------------------------
    # 3. Match official splits to local files
    # --------------------------------------------------------

    print("\n[3/5] Matching official anomaly splits...")

    rows = []

    matched_train = 0
    matched_test = 0

    unmatched_train = []
    unmatched_test = []

    # -----------------------------
    # Anomaly training videos
    # -----------------------------

    for relative_path in sorted(train_entries):

        local_path = anomaly_index.get(relative_path)

        if local_path is None:
            unmatched_train.append(relative_path)
            continue

        rows.append(
            {
                "video_id": local_path.stem,
                "path": str(local_path.resolve()),
                "split": "train",
                "label": 1,
                "class_name": "Anomaly",
                "source": "UCF-Crime",
                "ucf_category": category_from_path(relative_path),
                "evaluation_type": "binary_anomaly_detection",
            }
        )

        matched_train += 1

    # -----------------------------
    # Anomaly test videos
    # -----------------------------

    for relative_path in sorted(test_entries):

        local_path = anomaly_index.get(relative_path)

        if local_path is None:
            unmatched_test.append(relative_path)
            continue

        rows.append(
            {
                "video_id": local_path.stem,
                "path": str(local_path.resolve()),
                "split": "test",
                "label": 1,
                "class_name": "Anomaly",
                "source": "UCF-Crime",
                "ucf_category": category_from_path(relative_path),
                "evaluation_type": "binary_anomaly_detection",
            }
        )

        matched_test += 1

    # --------------------------------------------------------
    # 4. Add normal evaluation videos
    # --------------------------------------------------------

    print("\n[4/5] Indexing normal evaluation videos...")

    normal_index = build_normal_index(NORMAL_ROOT)

    normal_count = 0

    for filename, local_path in sorted(normal_index.items()):

        rows.append(
            {
                "video_id": local_path.stem,
                "path": str(local_path.resolve()),
                "split": "test",
                "label": 0,
                "class_name": "Normal",
                "source": "UCF-Crime",
                "ucf_category": "Normal",
                "evaluation_type": "binary_anomaly_detection",
            }
        )

        normal_count += 1

    # --------------------------------------------------------
    # 5. Write metadata
    # --------------------------------------------------------

    print("\n[5/5] Writing metadata...")

    fieldnames = [
        "video_id",
        "path",
        "split",
        "label",
        "class_name",
        "source",
        "ucf_category",
        "evaluation_type",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(f"\nOutput file:")
    print(OUTPUT_FILE)

    print("\nAnomaly:")
    print(f"  Train matched : {matched_train}")
    print(f"  Test matched  : {matched_test}")

    print("\nNormal:")
    print(f"  Test videos   : {normal_count}")

    print("\nTotal metadata rows:")
    print(f"  {len(rows)}")

    print("\nExpected from your current local files:")
    print("  Anomaly train : 641")
    print("  Anomaly test  : 109")
    print("  Normal test   : 150")

    print("\nUnmatched official anomaly entries:")
    print(f"  Train : {len(unmatched_train)}")
    print(f"  Test  : {len(unmatched_test)}")

    print("\nSplit composition:")

    train_rows = [
        r for r in rows
        if r["split"] == "train"
    ]

    test_rows = [
        r for r in rows
        if r["split"] == "test"
    ]

    print(f"  Train anomaly : {sum(r['label'] == 1 for r in train_rows)}")

    print(
        f"  Test anomaly  : "
        f"{sum(r['label'] == 1 for r in test_rows)}"
    )

    print(
        f"  Test normal   : "
        f"{sum(r['label'] == 0 for r in test_rows)}"
    )

    print("\nIMPORTANT:")
    print(
        "The 150 normal videos are used only for evaluation, "
        "not training."
    )

    print(
        "This metadata prepares a binary anomaly-detection "
        "experiment and does not claim spatial ground-truth boxes."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()