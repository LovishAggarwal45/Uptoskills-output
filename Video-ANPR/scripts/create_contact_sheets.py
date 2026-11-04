
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
import math

CSV_PATH = Path(
    "outputs/evaluation/ground_truth_annotation_template.csv"
)

OUTPUT_DIR = Path("outputs/evaluation/contact_sheets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

THUMB_WIDTH = 300
THUMB_HEIGHT = 130
ITEMS_PER_PAGE = 12

for car_id, group in df.groupby("car_id"):

    group = group.sort_values("frame").reset_index(drop=True)

    total_pages = math.ceil(len(group) / ITEMS_PER_PAGE)

    for page in range(total_pages):

        start = page * ITEMS_PER_PAGE
        end = min(start + ITEMS_PER_PAGE, len(group))
        subset = group.iloc[start:end]

        columns = 3
        rows = math.ceil(len(subset) / columns)

        sheet = Image.new(
            "RGB",
            (columns * THUMB_WIDTH, rows * THUMB_HEIGHT),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for index, (_, row) in enumerate(subset.iterrows()):

            image_path = Path(str(row["original_crop"]))

            x = (index % columns) * THUMB_WIDTH
            y = (index // columns) * THUMB_HEIGHT

            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((280, 85))

                sheet.paste(image, (x + 10, y + 5))
            else:
                draw.text(
                    (x + 10, y + 10),
                    "Crop image missing",
                    fill="red"
                )

            label = (
                f"Car ID: {car_id} | Frame: {row['frame']}\n"
                f"Predicted: {row['final_voted_plate']}"
            )

            draw.text(
                (x + 10, y + 95),
                label,
                fill="black"
            )

        output_path = OUTPUT_DIR / (
            f"vehicle_{car_id}_page_{page + 1}.jpg"
        )

        sheet.save(output_path, quality=95)

    print(
        f"Vehicle {car_id}: "
        f"{len(group)} observations, {total_pages} pages"
    )

print("\nContact sheets created successfully!")
print("Location:", OUTPUT_DIR)
