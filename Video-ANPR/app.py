
from __future__ import annotations

import io
import os
import re
import sys
import subprocess
import zipfile
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MAIN_PY = ROOT / "main.py"

UPLOAD_DIR = ROOT / "uploads"
INPUT_VIDEO = UPLOAD_DIR / "car.mp4"

OUTPUT_DIR = Path(
    os.environ.get("ANPR_OUTPUT_DIR", str(ROOT / "outputs"))
)

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = ROOT / OUTPUT_DIR

# Your current project output structure
OUTPUT_VIDEO = OUTPUT_DIR / "anpr_realesrgan_output.mp4"
RESULTS_CSV = OUTPUT_DIR / "anpr_realesrgan_results.csv"

PLATE_DIR = OUTPUT_DIR / "plate_crops"
ENHANCED_DIR = OUTPUT_DIR / "enhanced_plates"

# Support earlier run-specific folders if root files are absent.
if not RESULTS_CSV.exists():
    run_dirs = sorted(
        OUTPUT_DIR.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for folder in run_dirs:
        candidate = folder / "anpr_realesrgan_results.csv"
        if candidate.exists():
            RESULTS_CSV = candidate
            break

if not OUTPUT_VIDEO.exists():
    for folder in sorted(
        OUTPUT_DIR.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ):
        candidate = folder / "anpr_realesrgan_output.mp4"
        if candidate.exists():
            OUTPUT_VIDEO = candidate
            break

INDIAN_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
]

st.set_page_config(
    page_title="ANPR + Real-ESRGAN",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 ANPR + Real-ESRGAN Dashboard")
st.caption(
    "Automatic Number Plate Recognition | YOLO | SORT | "
    "Real-ESRGAN | EasyOCR | Multi-frame Voting"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_plate(value):
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in ("", "NONE", "NAN", "UNREADABLE"):
        return ""

    return re.sub(r"[^A-Z0-9]", "", text)


def valid_indian_format(value):
    text = normalize_plate(value)

    return bool(text) and any(
        pattern.fullmatch(text)
        for pattern in INDIAN_PATTERNS
    )


def readable(value):
    return bool(normalize_plate(value))


def clean_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def mode_plate(series):
    values = [
        normalize_plate(value)
        for value in series.fillna("").tolist()
    ]

    values = [v for v in values if v]

    if not values:
        return ""

    return Counter(values).most_common(1)[0][0]


def to_bool(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def read_results():
    if not RESULTS_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            RESULTS_CSV,
            encoding="utf-8-sig"
        )
    except Exception as exc:
        st.error(f"Unable to read results CSV: {exc}")
        return pd.DataFrame()


def vehicle_summary(df):
    if df.empty or "car_id" not in df.columns:
        return pd.DataFrame()

    rows = []

    for car_id, group in df.groupby("car_id", dropna=False):

        row = {"car_id": car_id}

        for column in [
            "original_ocr",
            "enhanced_ocr",
            "best_ocr_text",
            "final_voted_plate"
        ]:
            if column in group.columns:
                row[column] = mode_plate(group[column])

        if "vote_count" in group.columns:
            row["vote_count"] = pd.to_numeric(
                group["vote_count"],
                errors="coerce"
            ).max()

        if "final_confidence" in group.columns:
            row["final_confidence"] = pd.to_numeric(
                group["final_confidence"],
                errors="coerce"
            ).max()

        if "format_valid" in group.columns:
            row["format_valid"] = to_bool(
                group["format_valid"]
            ).any()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3. GROUND-TRUTH TEMPLATE
# ============================================================

def make_ground_truth_template(df):

    if df.empty or not {"frame", "car_id"}.issubset(df.columns):
        template = pd.DataFrame(
            columns=["frame", "car_id", "actual_plate"]
        )

    else:
        template = df[["frame", "car_id"]].copy()

        template["frame"] = pd.to_numeric(
            template["frame"],
            errors="coerce"
        )

        template = template.dropna(
            subset=["frame", "car_id"]
        )

        template["frame"] = template["frame"].astype(int)

        template = template.drop_duplicates(
            subset=["frame", "car_id"]
        )

        template["actual_plate"] = ""

    return template.to_csv(
        index=False
    ).encode("utf-8-sig")


# ============================================================
# 4. OCR ACCURACY EVALUATION
# ============================================================

def evaluate_ground_truth(results, ground_truth):

    required_results = {"frame", "car_id"}
    required_gt = {"frame", "car_id", "actual_plate"}

    res = results.copy()
    gt = ground_truth.copy()

    res.columns = res.columns.astype(str).str.strip()
    gt.columns = gt.columns.astype(str).str.strip()

    if not required_results.issubset(res.columns):
        raise ValueError(
            "Results CSV must contain frame and car_id."
        )

    if not required_gt.issubset(gt.columns):
        raise ValueError(
            "Ground truth must contain frame, car_id, actual_plate."
        )

    # Normalize frame and vehicle IDs
    res["frame"] = pd.to_numeric(
        res["frame"], errors="coerce"
    )

    gt["frame"] = pd.to_numeric(
        gt["frame"], errors="coerce"
    )

    res["car_id"] = clean_id(res["car_id"])
    gt["car_id"] = clean_id(gt["car_id"])

    res = res.dropna(
        subset=["frame", "car_id"]
    ).copy()

    gt = gt.dropna(
        subset=["frame", "car_id"]
    ).copy()

    res["frame"] = res["frame"].astype(int)
    gt["frame"] = gt["frame"].astype(int)

    # Normalize actual labels
    gt["actual_plate"] = (
        gt["actual_plate"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    gt["actual_norm"] = gt["actual_plate"].map(
        normalize_plate
    )

    gt = gt[gt["actual_norm"] != ""].copy()

    if gt.empty:
        raise ValueError(
            "No verified actual_plate labels found."
        )

    # Detect conflicting annotations
    label_counts = gt.groupby(
        ["frame", "car_id"]
    )["actual_norm"].nunique()

    conflicts = label_counts[label_counts > 1]

    if not conflicts.empty:
        raise ValueError(
            "Conflicting ground-truth labels exist for "
            "the same frame and vehicle ID."
        )

    # Remove exact duplicate annotations
    gt = gt.drop_duplicates(
        subset=["frame", "car_id", "actual_norm"]
    )

    # Handle duplicate result rows.
    # Keep the row with the highest final confidence.
    if "final_confidence" in res.columns:
        res["_final_conf"] = pd.to_numeric(
            res["final_confidence"],
            errors="coerce"
        ).fillna(0)
    else:
        res["_final_conf"] = 0

    if "enhanced_ocr_confidence" in res.columns:
        res["_enhanced_conf"] = pd.to_numeric(
            res["enhanced_ocr_confidence"],
            errors="coerce"
        ).fillna(0)
    else:
        res["_enhanced_conf"] = 0

    res = res.sort_values(
        ["_final_conf", "_enhanced_conf"],
        ascending=False
    )

    res = res.drop_duplicates(
        subset=["frame", "car_id"],
        keep="first"
    )

    # Frame-level matching
    merged = res.merge(
        gt[["frame", "car_id", "actual_norm", "actual_plate"]],
        on=["frame", "car_id"],
        how="inner",
        validate="one_to_one"
    )

    if merged.empty:
        raise ValueError(
            "No matching frame and car ID pairs. "
            "Check that ground truth belongs to this results CSV."
        )

    # IMPORTANT:
    # Use final_voted_plate from your actual results CSV.
    source_columns = {
        "original_ocr": "original_prediction",
        "enhanced_ocr": "enhanced_prediction",
        "final_voted_plate": "final_prediction"
    }

    for source, target in source_columns.items():

        if source in merged.columns:
            merged[target] = merged[source].map(
                normalize_plate
            )
        else:
            merged[target] = ""

    for method in ["original", "enhanced", "final"]:

        merged[f"{method}_exact_match"] = (
            merged[f"{method}_prediction"]
            == merged["actual_norm"]
        )

    return merged, "Frame-aligned observations"


def calculate_character_accuracy(predictions, actuals):

    similarities = []

    for prediction, actual in zip(predictions, actuals):

        a = normalize_plate(prediction)
        b = normalize_plate(actual)

        if not b:
            continue

        # Levenshtein edit distance
        previous = list(range(len(b) + 1))

        for i, char_a in enumerate(a, start=1):

            current = [i]

            for j, char_b in enumerate(b, start=1):

                current.append(
                    min(
                        current[-1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + (char_a != char_b)
                    )
                )

            previous = current

        distance = previous[-1]

        similarity = max(
            0,
            1 - distance / max(len(a), len(b), 1)
        )

        similarities.append(similarity)

    if not similarities:
        return 0.0

    return round(
        sum(similarities) / len(similarities) * 100,
        2
    )


def create_accuracy_summary(evaluated):

    methods = [
        ("Original OCR", "original"),
        ("Enhanced OCR", "enhanced"),
        ("Multi-frame Voting", "final")
    ]

    rows = []

    for label, prefix in methods:

        correct = int(
            evaluated[f"{prefix}_exact_match"].sum()
        )

        total = len(evaluated)

        accuracy = (
            correct / total * 100
            if total else 0
        )

        character_accuracy = calculate_character_accuracy(
            evaluated[f"{prefix}_prediction"],
            evaluated["actual_norm"]
        )

        rows.append({
            "Method": label,
            "Matched Observations": total,
            "Correct Plates": correct,
            "Exact Match Accuracy (%)": round(accuracy, 2),
            "Character Accuracy (%)": character_accuracy
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. RUN ANPR PIPELINE
# ============================================================

def run_pipeline(uploaded_video=None):

    if not MAIN_PY.exists():
        st.error(f"main.py not found: {MAIN_PY}")
        return False

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if uploaded_video is not None:

        with INPUT_VIDEO.open("wb") as file:
            file.write(uploaded_video.getbuffer())

    if not INPUT_VIDEO.exists():
        st.error(
            f"No input video found. Upload a video or place it at {INPUT_VIDEO}"
        )
        return False

    env = os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"
    env["ANPR_VIDEO_PATH"] = str(INPUT_VIDEO.resolve())
    env["ANPR_OUTPUT_DIR"] = str(OUTPUT_DIR.resolve())

    status = st.status(
        "Running ANPR pipeline...",
        expanded=True
    )

    log_box = status.empty()

    progress = st.progress(
        0,
        text="Starting pipeline"
    )

    logs = []

    try:

        process = subprocess.Popen(
            [sys.executable, str(MAIN_PY)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):

            line = line.rstrip()

            if not line:
                continue

            logs.append(line)
            logs = logs[-150:]

            log_box.code(
                "\n".join(logs[-60:]),
                language="text"
            )

            match = re.search(
                r"Processed\s+(\d+)\s*/\s*(\d+)\s*frames",
                line,
                flags=re.I
            )

            if match:

                done = int(match.group(1))
                total = max(1, int(match.group(2)))

                progress.progress(
                    min(done / total, 1.0),
                    text=f"Processed {done}/{total} frames"
                )

        exit_code = process.wait()

        st.session_state["pipeline_logs"] = logs

        if exit_code == 0:

            progress.progress(
                1.0,
                text="Pipeline complete"
            )

            status.update(
                label="Pipeline completed successfully",
                state="complete",
                expanded=False
            )

            st.session_state["last_run"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            return True

        status.update(
            label=f"Pipeline failed: exit code {exit_code}",
            state="error",
            expanded=True
        )

        st.error("Pipeline failed. Check the logs.")

        return False

    except Exception as exc:

        status.update(
            label="Unable to start pipeline",
            state="error",
            expanded=True
        )

        st.exception(exc)

        return False


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("ANPR Controls")

uploaded_video = st.sidebar.file_uploader(
    "Upload input video",
    type=["mp4", "mov", "avi", "mkv"]
)

if INPUT_VIDEO.exists():
    st.sidebar.success(
        f"Default video ready: {INPUT_VIDEO.name}"
    )
else:
    st.sidebar.info("Upload a video to begin.")

if st.sidebar.button(
    "▶ Run ANPR Pipeline",
    type="primary",
    use_container_width=True
):

    if run_pipeline(uploaded_video):
        st.rerun()

if st.sidebar.button(
    "↻ Refresh Saved Outputs",
    use_container_width=True
):
    st.rerun()

st.sidebar.caption(
    "The pipeline runs only when you press Run."
)


# ============================================================
# 7. LOAD RESULTS
# ============================================================

df = read_results()

video_exists = (
    OUTPUT_VIDEO.exists()
    and OUTPUT_VIDEO.stat().st_size > 0
)

csv_exists = (
    RESULTS_CSV.exists()
    and RESULTS_CSV.stat().st_size > 0
)

expected_columns = {
    "frame": pd.NA,
    "time_sec": pd.NA,
    "car_id": pd.NA,
    "original_ocr": "",
    "original_ocr_confidence": 0.0,
    "enhanced_ocr": "",
    "enhanced_ocr_confidence": 0.0,
    "best_ocr_text": "",
    "final_voted_plate": "",
    "vote_count": 0,
    "final_confidence": 0.0,
    "format_valid": False,
    "final_status": ""
}

for column, default in expected_columns.items():

    if column not in df.columns:
        df[column] = default

if not df.empty:

    for column in [
        "frame",
        "time_sec",
        "car_id",
        "original_ocr_confidence",
        "enhanced_ocr_confidence",
        "vote_count",
        "final_confidence"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["format_valid_bool"] = to_bool(
        df["format_valid"]
    )

    df["final_readable"] = df[
        "final_voted_plate"
    ].map(readable)

else:

    df["format_valid_bool"] = pd.Series(dtype=bool)
    df["final_readable"] = pd.Series(dtype=bool)


# ============================================================
# 8. DASHBOARD TABS
# ============================================================

overview, explorer, video_tab, accuracy, crops, help_tab = st.tabs([
    "Overview",
    "Results Explorer",
    "Video & Downloads",
    "OCR Accuracy",
    "Crops & Validation",
    "Help"
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with overview:

    st.subheader("Pipeline Overview")

    if df.empty:

        st.warning("No detection results available yet.")

    else:

        metrics = st.columns(4)

        metrics[0].metric(
            "Detection Rows",
            f"{len(df):,}"
        )

        metrics[1].metric(
            "Tracked Vehicles",
            f"{df['car_id'].nunique(dropna=True):,}"
        )

        metrics[2].metric(
            "Original OCR Rows",
            f"{int(df['original_ocr'].map(readable).sum()):,}"
        )

        metrics[3].metric(
            "Enhanced OCR Rows",
            f"{int(df['enhanced_ocr'].map(readable).sum()):,}"
        )

        metrics2 = st.columns(3)

        metrics2[0].metric(
            "Readable Final Rows",
            f"{int(df['final_readable'].sum()):,}"
        )

        metrics2[1].metric(
            "Rows With Votes",
            f"{int((df['vote_count'].fillna(0) > 0).sum()):,}"
        )

        metrics2[2].metric(
            "Indian Format Valid",
            f"{int(df['format_valid_bool'].sum()):,}"
        )

        st.caption(
            "These are detection counts, not accuracy measurements. "
            "OCR confidence is not accuracy."
        )

        st.markdown("### Most Common Final OCR Results")

        counts = (
            df["final_voted_plate"]
            .fillna("")
            .map(normalize_plate)
            .replace("", "UNREADABLE")
            .value_counts()
            .head(12)
        )

        st.bar_chart(counts)

        st.markdown("### Per-Vehicle Summary")

        st.dataframe(
            vehicle_summary(df),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Output File Status")

    st.write(
        "Annotated Video:",
        "Available" if video_exists else f"Missing: {OUTPUT_VIDEO}"
    )

    st.write(
        "Results CSV:",
        "Available" if csv_exists else f"Missing: {RESULTS_CSV}"
    )


# ============================================================
# TAB 2: RESULTS EXPLORER
# ============================================================

with explorer:

    st.subheader("Search and Filter Detections")

    if df.empty:

        st.info("No result rows available.")

    else:

        c1, c2, c3, c4 = st.columns(4)

        vehicle_ids = sorted(
            df["car_id"].dropna().unique().tolist()
        )

        selected_ids = c1.multiselect(
            "Vehicle IDs",
            vehicle_ids
        )

        format_filter = c2.selectbox(
            "Indian Format",
            ["All", "Valid", "Invalid"]
        )

        result_filter = c3.selectbox(
            "Final Result",
            ["All", "Readable", "Unreadable"]
        )

        query = c4.text_input(
            "Search Plate",
            placeholder="TN09AB1234"
        )

        filtered = df.copy()

        if selected_ids:
            filtered = filtered[
                filtered["car_id"].isin(selected_ids)
            ]

        if format_filter == "Valid":
            filtered = filtered[
                filtered["format_valid_bool"]
            ]

        elif format_filter == "Invalid":
            filtered = filtered[
                ~filtered["format_valid_bool"]
            ]

        if result_filter == "Readable":
            filtered = filtered[
                filtered["final_readable"]
            ]

        elif result_filter == "Unreadable":
            filtered = filtered[
                ~filtered["final_readable"]
            ]

        if query.strip():

            needle = normalize_plate(query)

            search_columns = [
                col for col in [
                    "original_ocr",
                    "enhanced_ocr",
                    "best_ocr_text",
                    "final_voted_plate"
                ]
                if col in filtered.columns
            ]

            mask = pd.Series(
                False,
                index=filtered.index
            )

            for col in search_columns:

                mask |= filtered[col].fillna("").map(
                    normalize_plate
                ).str.contains(
                    re.escape(needle),
                    na=False
                )

            filtered = filtered[mask]

        display_columns = [
            col for col in [
                "frame",
                "time_sec",
                "car_id",
                "original_ocr",
                "original_ocr_confidence",
                "enhanced_ocr",
                "enhanced_ocr_confidence",
                "best_ocr_text",
                "final_voted_plate",
                "vote_count",
                "final_confidence",
                "format_valid",
                "final_status"
            ]
            if col in filtered.columns
        ]

        st.caption(
            f"Showing {len(filtered)} of {len(df)} observations"
        )

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Filtered CSV",
            data=filtered.drop(
                columns=[
                    "format_valid_bool",
                    "final_readable"
                ],
                errors="ignore"
            ).to_csv(index=False).encode("utf-8-sig"),
            file_name="anpr_filtered_results.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 3: VIDEO AND DOWNLOADS
# ============================================================

with video_tab:

    st.subheader("Annotated Output Video")

    left, right = st.columns([1.5, 1])

    with left:

        if video_exists:

            st.video(str(OUTPUT_VIDEO))

            st.download_button(
                "⬇ Download Annotated Video",
                data=OUTPUT_VIDEO.read_bytes(),
                file_name=OUTPUT_VIDEO.name,
                mime="video/mp4"
            )

        else:

            st.warning(
                f"Output video not found: {OUTPUT_VIDEO}"
            )

    with right:

        if csv_exists:

            st.download_button(
                "⬇ Download Results CSV",
                data=RESULTS_CSV.read_bytes(),
                file_name=RESULTS_CSV.name,
                mime="text/csv"
            )

        else:

            st.warning("Results CSV not found.")

        st.markdown("### Crop Files")

        original_count = (
            len(list(PLATE_DIR.glob("*.jpg")))
            if PLATE_DIR.exists()
            else 0
        )

        enhanced_count = (
            len(list(ENHANCED_DIR.glob("*.jpg")))
            if ENHANCED_DIR.exists()
            else 0
        )

        st.write("Original Crops:", original_count)
        st.write("Enhanced Crops:", enhanced_count)

    if st.button(
        "Prepare Review ZIP",
        disabled=not (video_exists and csv_exists)
    ):

        buffer = io.BytesIO()

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as archive:

            archive.write(
                OUTPUT_VIDEO,
                OUTPUT_VIDEO.name
            )

            archive.write(
                RESULTS_CSV,
                RESULTS_CSV.name
            )

            archive.writestr(
                "README.txt",
                "ANPR review bundle. Verify all plate text "
                "against the source video. OCR confidence "
                "is not the same as accuracy."
            )

        st.session_state["review_zip"] = buffer.getvalue()

    if "review_zip" in st.session_state:

        st.download_button(
            "⬇ Download Review ZIP",
            data=st.session_state["review_zip"],
            file_name="anpr_review_bundle.zip",
            mime="application/zip"
        )


# ============================================================
# TAB 4: OCR ACCURACY
# ============================================================

with accuracy:

    st.subheader("OCR Accuracy Evaluation")

    st.info(
        "Upload manually verified ground-truth labels. "
        "Accuracy is calculated using frame and vehicle ID. "
        "OCR confidence is not accuracy."
    )

    st.download_button(
        "⬇ Download Ground-Truth Template",
        data=make_ground_truth_template(df),
        file_name="ground_truth_template.csv",
        mime="text/csv"
    )

    st.caption(
        "Fill actual_plate using the source video. "
        "Required columns: frame, car_id, actual_plate."
    )

    ground_truth_file = st.file_uploader(
        "Upload Completed Ground-Truth CSV",
        type=["csv"],
        key="ground_truth_csv"
    )

    if ground_truth_file is not None:

        try:

            ground_truth_df = pd.read_csv(
                ground_truth_file
            )

            st.markdown("### Ground-Truth Data")

            st.dataframe(
                ground_truth_df.head(30),
                use_container_width=True,
                hide_index=True
            )

            evaluated, evaluation_level = evaluate_ground_truth(
                df,
                ground_truth_df
            )

            summary = create_accuracy_summary(
                evaluated
            )

            st.caption(
                f"{evaluation_level} | "
                f"Matched observations: {len(evaluated)}"
            )

            metric_cols = st.columns(3)

            for col, (_, row) in zip(
                metric_cols,
                summary.iterrows()
            ):

                col.metric(
                    row["Method"],
                    f"{row['Exact Match Accuracy (%)']:.2f}%",
                    f"{int(row['Correct Plates'])}/"
                    f"{int(row['Matched Observations'])} correct"
                )

            st.markdown("### Accuracy Comparison")

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                summary.set_index("Method")[
                    [
                        "Exact Match Accuracy (%)",
                        "Character Accuracy (%)"
                    ]
                ]
            )

            st.markdown("### Frame-Level Evaluation")

            st.dataframe(
                evaluated,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇ Download Evaluation Details",
                data=evaluated.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="anpr_accuracy_evaluation.csv",
                mime="text/csv"
            )

            st.download_button(
                "⬇ Download Evaluation Summary",
                data=summary.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="anpr_accuracy_summary.csv",
                mime="text/csv"
            )

            if len(evaluated) < 30:

                st.warning(
                    "Small evaluation sample. Clearly report "
                    "the sample size in your project report."
                )

        except Exception as exc:

            st.error(
                f"Evaluation could not be completed: {exc}"
            )


# ============================================================
# TAB 5: CROPS AND VALIDATION
# ============================================================

with crops:

    st.subheader("Original vs Enhanced Plate Crops")

    if df.empty or df["car_id"].dropna().empty:

        st.info(
            "Run the pipeline or load a results CSV to view crops."
        )

    else:

        ids = sorted(
            df["car_id"].dropna().unique().tolist()
        )

        selected_id = st.selectbox(
            "Select Vehicle ID",
            ids
        )

        vehicle_rows = df[
            df["car_id"] == selected_id
        ].copy()

        if "frame" in vehicle_rows.columns:

            vehicle_rows = vehicle_rows.sort_values(
                "frame"
            )

        max_rows = min(20, len(vehicle_rows))

        if max_rows <= 1:
           num_rows = 1
        else:
           num_rows = st.slider(
        "Number of detections",
        min_value=1,
        max_value=max_rows,
        value=min(5, max_rows)
    )

        for _, row in vehicle_rows.head(num_rows).iterrows():

            frame = row.get("frame")
            car_id = row.get("car_id")

            if pd.isna(frame) or pd.isna(car_id):
                continue

            frame_id = int(frame)
            car_text = str(int(car_id))

            original_crop = (
                PLATE_DIR
                / f"frame_{frame_id:06d}_car_{car_text}.jpg"
            )

            enhanced_crop = (
                ENHANCED_DIR
                / f"enhanced_frame_{frame_id:06d}_car_{car_text}.jpg"
            )

            st.markdown(
                f"### Frame {frame_id} | Vehicle {car_text}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.caption(
                    f"Original OCR: {row.get('original_ocr', '')}"
                )

                st.caption(
                    f"Confidence: "
                    f"{row.get('original_ocr_confidence', '')}"
                )

                if original_crop.exists():

                    st.image(
                        str(original_crop),
                        caption="Original Plate Crop",
                        use_container_width=True
                    )

                else:

                    st.caption("Original crop unavailable.")

            with col2:

                st.caption(
                    f"Enhanced OCR: {row.get('enhanced_ocr', '')}"
                )

                st.caption(
                    f"Confidence: "
                    f"{row.get('enhanced_ocr_confidence', '')}"
                )

                if enhanced_crop.exists():

                    st.image(
                        str(enhanced_crop),
                        caption="Enhanced Plate Crop",
                        use_container_width=True
                    )

                else:

                    st.caption("Enhanced crop unavailable.")

            st.write(
                "Final Voted Plate:",
                row.get("final_voted_plate", "")
            )

            st.write(
                "Vote Count:",
                row.get("vote_count", "")
            )

            st.write(
                "Format Valid:",
                row.get("format_valid", "")
            )

            st.divider()

    st.markdown("### Test Indian Number Plate Format")

    manual_plate = st.text_input(
        "Enter plate number",
        placeholder="TN09AB1234"
    )

    if manual_plate.strip():

        cleaned = normalize_plate(manual_plate)

        if valid_indian_format(cleaned):

            st.success(
                f"{cleaned} matches a configured Indian format pattern. "
                "This does not verify official registration."
            )

        else:

            st.warning(
                f"{cleaned} does not match the configured patterns."
            )


# ============================================================
# TAB 6: HELP
# ============================================================

with help_tab:

    st.subheader("Dashboard Guide")

    if st.session_state.get("last_run"):

        st.write(
            "Last successful run:",
            st.session_state["last_run"]
        )

    if st.session_state.get("pipeline_logs"):

        with st.expander("Latest Pipeline Logs"):

            st.code(
                "\n".join(
                    st.session_state["pipeline_logs"]
                ),
                language="text"
            )

    st.markdown("""
    ### How to use

    1. Upload an MP4, MOV, AVI or MKV video.
    2. Click Run ANPR Pipeline.
    3. Wait for vehicle detection, tracking, plate detection,
       enhancement and OCR to finish.
    4. Open Results Explorer to inspect detections.
    5. Open Video & Downloads to view the annotated video.
    6. Open OCR Accuracy to upload verified ground truth.
    7. Open Crops & Validation to compare original and
       enhanced plate crops.

    ### Metric definitions

    - OCR confidence is the OCR engine score, not accuracy.
    - Exact-match accuracy requires the complete plate to match.
    - Character accuracy measures normalized edit similarity.
    - Vote count represents observations used by voting.
    - Format-valid means the text matches a configured pattern.
    - Format validation does not prove official registration.
    - Repeated frames are observations, not independent vehicles.
    """)

    st.code(
        "python -m pip install streamlit pandas\n"
        "python -m streamlit run app.py",
        language="powershell"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "ANPR + Real-ESRGAN | Verify detected plate text against "
    "the source video before treating it as confirmed."
)
