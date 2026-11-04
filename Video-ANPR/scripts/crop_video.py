
import cv2
from pathlib import Path

INPUT_VIDEO = "uploads/ANPR.mp4"
OUTPUT_VIDEO = "cropped_video.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Remove the top overlay area
crop_top = 160

if height <= crop_top:
    raise ValueError("Video is too short for this crop.")

output_height = height - crop_top

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, output_height)
)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Crop the top 160 pixels
    cropped = frame[crop_top:, 0:width]

    writer.write(cropped)

cap.release()
writer.release()

print("Video cropping completed!")
print("Saved as:", OUTPUT_VIDEO)
print("Output resolution:", width, "x", output_height)
print("FPS:", fps)
