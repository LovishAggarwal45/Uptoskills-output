
import pandas as pd
from pathlib import Path

path = Path("outputs/evaluation/ground_truth_verified.csv")

if not path.exists():
    raise FileNotFoundError(
        "Please save your completed annotations as ground_truth_verified.csv"
    )

df = pd.read_csv(path)

required = ["frame", "car_id", "actual_plate"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["actual_plate"] = df["actual_plate"].fillna("").astype(str).str.strip()

missing = df[df["actual_plate"] == ""]
duplicates = df[df.duplicated(["frame", "car_id"], keep=False)]

print("\n===== GROUND TRUTH VALIDATION =====")
print("Total observations:", len(df))
print("Verified observations:", len(df) - len(missing))
print("Missing annotations:", len(missing))
print("Duplicate frame/car IDs:", len(duplicates))

if len(missing):
    print("\nPlease verify these rows:")
    print(missing[["frame", "car_id"]].to_string(index=False))

elif len(duplicates):
    print("\nPlease resolve duplicate frame/car ID entries.")

else:
    print("\nGround truth validation successful!")
