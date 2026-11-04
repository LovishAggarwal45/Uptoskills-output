
from pathlib import Path
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "outputs" / "anpr_realesrgan_output.mp4"
CSV = ROOT / "outputs" / "anpr_realesrgan_results.csv"
GROUND_TRUTH = ROOT / "ground_truth.csv"
EVALUATION = ROOT / "anpr_accuracy_evaluation.csv"

checks = []

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    checks.append(bool(passed))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


print("\n===== ANPR FINAL DELIVERABLE AUDIT =====\n")

# 1. Check output files exist and aren't empty.
report("Annotated video exists", VIDEO.is_file())
report("Annotated video is non-empty",
       VIDEO.is_file() and VIDEO.stat().st_size > 0)
report("Results CSV exists", CSV.is_file())
report("Results CSV is non-empty",
       CSV.is_file() and CSV.stat().st_size > 0)

# 2. Check that OpenCV can open and decode the output video.
if VIDEO.is_file() and VIDEO.stat().st_size > 0:
    cap = cv2.VideoCapture(str(VIDEO))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    decoded_frames = 0

    if opened:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is not None and frame.size > 0:
                decoded_frames += 1

    cap.release()

    report("Video opens in OpenCV", opened)
    report("Video FPS and frame count are positive",
           fps > 0 and reported_frames > 0,
           f"FPS={fps:.2f}, reported frames={reported_frames}")
    report("Video frames decode successfully",
           decoded_frames > 0,
           f"Decoded frames={decoded_frames}")
    report("Video appears fully decodable",
           decoded_frames == reported_frames,
           f"Decoded={decoded_frames}, reported={reported_frames}")
else:
    report("Video opens in OpenCV", False, "Video missing or empty")

# 3. Check CSV columns and data.
if CSV.is_file() and CSV.stat().st_size > 0:
    try:
        df = pd.read_csv(CSV, encoding="utf-8-sig")
        required = [
            "frame", "time_sec", "car_id",
            "original_ocr", "original_ocr_confidence",
            "enhanced_ocr", "enhanced_ocr_confidence",
            "final_voted_plate", "vote_count",
            "final_confidence", "format_valid",
        ]
        missing = [col for col in required if col not in df.columns]

        report("CSV can be read", True, f"Rows={len(df)}")
        report("Required CSV columns exist",
               not missing,
               f"Missing={missing}" if missing else "All required columns present")
        report("CSV contains result rows", len(df) > 0)

        if len(df) > 0 and not missing:
            for col in [
                "frame", "time_sec", "car_id", "vote_count",
                "original_ocr_confidence", "enhanced_ocr_confidence",
                "final_confidence",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            report(
                "Timestamps are populated and non-negative",
                df["time_sec"].notna().all()
                and (df["time_sec"] >= 0).all(),
            )

            readable = (
                df["final_voted_plate"].fillna("").astype(str)
                .str.strip().str.upper()
                .ne("")
                & df["final_voted_plate"].fillna("").astype(str)
                .str.strip().str.upper().ne("UNREADABLE")
            )

            report(
                "Some final plate candidates are present",
                bool(readable.any()),
                f"Readable rows={int(readable.sum())}/{len(df)}",
            )

            votes = df["vote_count"].fillna(0)
            report(
                "Vote counts are non-negative",
                (votes >= 0).all(),
                f"Rows with votes={(votes > 0).sum()}",
            )

            for col in [
                "original_ocr_confidence",
                "enhanced_ocr_confidence",
                "final_confidence",
            ]:
                vals = df[col].dropna()
                report(
                    f"{col} values are within 0–1",
                    ((vals >= 0) & (vals <= 1)).all(),
                )

            flags = (
                df["format_valid"].fillna(False).astype(str).str.lower()
                .isin(["true", "1", "yes"])
            )
            report(
                "Format-valid flags are present",
                "format_valid" in df.columns,
                f"Valid rows={int(flags.sum())}",
            )

            print("\nTop final plate candidates:")
            print(
                df["final_voted_plate"].fillna("UNREADABLE")
                .value_counts().head(10).to_string()
            )

            print("\nCSV columns:")
            print(", ".join(df.columns))

    except Exception as exc:
        report("CSV can be read", False, str(exc))
else:
    report("CSV can be read", False, "CSV missing or empty")

# 4. Optional evaluation files.
report(
    "Ground-truth CSV available",
    GROUND_TRUTH.is_file(),
    "Optional; needed for accuracy evaluation",
)
report(
    "Downloaded accuracy evaluation available",
    EVALUATION.is_file(),
    "Optional; download it from the dashboard if generated",
)

print("\n===== AUDIT SUMMARY =====")
print(f"Passed checks: {sum(checks)}/{len(checks)}")
print(
    "Technical file checks completed. "
    "Manual review of plate correctness is still required."
)
