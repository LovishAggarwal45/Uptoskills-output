
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "outputs" / "anpr_realesrgan_results.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "contact_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Results CSV not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

required = ["frame", "car_id"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing CSV column: {col}")

image_col = (
    "enhanced_crop" if "enhanced_crop" in df.columns
    else "original_crop"
)

if image_col not in df.columns:
    raise ValueError("CSV needs enhanced_crop or original_crop column.")

for car_id, group in df.groupby("car_id"):
    tiles = []

    for _, row in group.iterrows():
        crop_path = Path(str(row[image_col]))

        if not crop_path.is_absolute():
            crop_path = ROOT / crop_path

        if not crop_path.exists():
            continue

        try:
            img = Image.open(crop_path).convert("RGB")
            img.thumbnail((340, 100))

            tile = Image.new("RGB", (360, 145), "white")
            x = (360 - img.width) // 2
            tile.paste(img, (x, 5))

            draw = ImageDraw.Draw(tile)
            draw.text(
                (8, 110),
                f"Car ID: {car_id} | Frame: {int(row['frame'])}",
                fill="black"
            )

            prediction = str(
                row.get("final_voted_plate", "Not available")
            )
            draw.text(
                (8, 127),
                f"Predicted: {prediction}",
                fill="black"
            )

            tiles.append(tile)

        except Exception as error:
            print(f"Skipping {crop_path}: {error}")

    if not tiles:
        print(f"No readable crops for car ID {car_id}")
        continue

    columns = 3
    rows = (len(tiles) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * 360, rows * 145),
        "#eeeeee"
    )

    for index, tile in enumerate(tiles):
        x = (index % columns) * 360
        y = (index // columns) * 145
        sheet.paste(tile, (x, y))

    output_file = OUTPUT_DIR / f"car_{car_id}_contact_sheet.png"
    sheet.save(output_file)

    print("Created:", output_file)

print("\nContact sheets created successfully!")
