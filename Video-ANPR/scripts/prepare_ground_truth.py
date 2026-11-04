
import pandas as pd
from pathlib import Path

results_path = Path("outputs/anpr_realesrgan_results.csv")
output_path = Path(
    "outputs/evaluation/ground_truth_annotation_template.csv"
)

df = pd.read_csv(results_path)

# Keep useful columns for manual annotation
columns = [
    "frame",
    "car_id",
    "original_crop",
    "enhanced_crop",
    "final_voted_plate"
]

template = df[columns].copy()

# Add an empty column for manually verified labels
template["actual_plate"] = ""

# Remove duplicate frame/vehicle observations
template = template.drop_duplicates(
    subset=["frame", "car_id"]
)

template = template.sort_values(
    ["car_id", "frame"]
)

output_path.parent.mkdir(parents=True, exist_ok=True)
template.to_csv(output_path, index=False)

print("Ground-truth annotation template created!")
print("Total observations:", len(template))
print("Unique vehicle IDs:", template["car_id"].nunique())
print("Saved to:", output_path)
print(template.head(10).to_string(index=False))
