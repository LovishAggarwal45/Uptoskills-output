
import cv2
from utils import reader

path = r"outputs\enhanced_plates\enhanced_frame_000000_car_2.jpg"

image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(
    gray, None, fx=4, fy=4,
    interpolation=cv2.INTER_CUBIC
)

configs = [
    {"contrast_ths": 0.05, "adjust_contrast": 0.7},
    {"contrast_ths": 0.1, "adjust_contrast": 0.9},
    {"contrast_ths": 0.2, "adjust_contrast": 0.5},
]

for config in configs:
    print("\nConfig:", config)
    results = reader.readtext(
        gray,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        detail=1,
        paragraph=False,
        **config
    )

    for result in results:
        print(
            "Text:", repr(result[1]),
            "| Confidence:", round(float(result[2]), 3)
        )
