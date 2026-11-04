
import pandas as pd
from pathlib import Path

path = Path(
    "outputs/evaluation/ground_truth_verified.csv"
)

df = pd.read_csv(path)
df["actual_plate"] = df["actual_plate"].astype("string")
# Only plates that were clearly readable
verified = {
    (294, 20): "TN10CD5678",

    (324, 32): "TN10CD5678",
    (327, 32): "TN10CD5678",
    (330, 32): "TN10CD5678",
    (333, 32): "TN10CD5678",
    (336, 32): "TN10CD5678",
    (339, 32): "TN10CD5678",

    (336, 36): "TN09AB1234",
    (339, 36): "TN09AB1234",
    (357, 36): "TN09AB1234",
    (360, 36): "TN09AB1234",
    (363, 36): "TN09AB1234",
    (366, 36): "TN09AB1234",
    (369, 36): "TN09AB1234",

    (381, 38): "TN09AB1234",
    (543, 52): "TN09AB1234",
}

for (frame, car_id), plate in verified.items():
    mask = (
        (df["frame"] == frame) &
        (df["car_id"] == car_id)
    )
    df.loc[mask, "actual_plate"] = plate

df.to_csv(path, index=False)

print("Verified labels saved:", len(verified))
print("Remaining labels:", (df["actual_plate"].fillna("").str.strip() == "").sum())
print("Saved:", path)
