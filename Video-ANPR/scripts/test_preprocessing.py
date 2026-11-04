
import cv2
from utils import reader

path = r"outputs\enhanced_plates\enhanced_frame_000000_car_2.jpg"

image = cv2.imread(path)

if image is None:
    print("ERROR: Could not load image:", path)
    raise SystemExit(1)

height, width = image.shape[:2]
image = cv2.resize(
    image,
    (width * 4, height * 4),
    interpolation=cv2.INTER_CUBIC,
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
).apply(gray)

_, threshold = cv2.threshold(
    clahe,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

variants = {
    "GRAY": gray,
    "CONTRAST": clahe,
    "THRESHOLD": threshold,
    "INVERTED": 255 - threshold,
}

for name, processed_image in variants.items():
    print(f"\n--- {name} ---")

    results = reader.readtext(processed_image)

    if not results:
        print("No text detected")
        continue

    for result in results:
        print(
            "Text:", repr(result[1]),
            "| Confidence:", round(float(result[2]), 3),
        )
