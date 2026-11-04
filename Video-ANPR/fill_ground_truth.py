
import pandas as pd
from pathlib import Path

path = Path("outputs/evaluation/ground_truth_current_run.csv")

df = pd.read_csv(path)

plate_map = {
    6: "KA02MN1826",
    36: "KA02MN1826",
    62: "KA02MM9091",
    # Add car 60 only after confirming its plate visually.
    # IDs 39, 40 and 87 remain blank.
}

df["actual_plate"] = df["car_id"].map(plate_map).fillna("")

df.to_csv(path, index=False)

print("Ground truth updated!")
print("Total observations:", len(df))
print("Labelled observations:", df["actual_plate"].ne("").sum())
print(df.groupby("car_id")["actual_plate"].first())
