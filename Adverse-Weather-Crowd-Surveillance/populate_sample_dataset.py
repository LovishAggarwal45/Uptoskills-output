import os
import shutil
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 1. 01_shibuya_crossing_clear_crowd.jpg
src_shibuya = os.path.join(SAMPLE_DIR, "real_clear_shibuya_crowd.jpg")
if os.path.exists(src_shibuya):
    shutil.copyfile(src_shibuya, os.path.join(SAMPLE_DIR, "01_shibuya_crossing_clear_crowd.jpg"))

# 2. 02_heavy_fog_pedestrians.jpg
src_fog = os.path.join(SAMPLE_DIR, "real_heavy_fog_crowd.jpg")
if os.path.exists(src_fog):
    shutil.copyfile(src_fog, os.path.join(SAMPLE_DIR, "02_heavy_fog_pedestrians.jpg"))

# 3. 03_monsoon_rain_umbrellas.jpg
src_rain = os.path.join(SAMPLE_DIR, "real_monsoon_rain_crowd.jpg")
if os.path.exists(src_rain):
    shutil.copyfile(src_rain, os.path.join(SAMPLE_DIR, "03_monsoon_rain_umbrellas.jpg"))

# 4. 04_times_square_sun_glare.jpg
src_glare = os.path.join(SAMPLE_DIR, "real_lens_glare_crowd.jpg")
if os.path.exists(src_glare):
    shutil.copyfile(src_glare, os.path.join(SAMPLE_DIR, "04_times_square_sun_glare.jpg"))

# 5. 05_empty_bridge_fog_control.jpg
src_empty = os.path.join(SAMPLE_DIR, "empty_bridge_control.jpg")
if os.path.exists(src_empty):
    shutil.copyfile(src_empty, os.path.join(SAMPLE_DIR, "05_empty_bridge_fog_control.jpg"))

# 6. 06_empty_plaza_clear_control.jpg (empty architecture with 0 people)
img_empty = np.zeros((480, 640, 3), dtype=np.uint8)
for y in range(220):
    img_empty[y, :] = (int(200 - y*0.2), int(180 - y*0.1), int(160 + y*0.1))
for y in range(220, 480):
    img_empty[y, :] = (90, 95, 100)
cv2.rectangle(img_empty, (80, 100), (220, 220), (120, 110, 105), -1)
cv2.rectangle(img_empty, (300, 70), (520, 220), (105, 100, 95), -1)
cv2.imwrite(os.path.join(SAMPLE_DIR, "06_empty_plaza_clear_control.jpg"), img_empty)

# 7. 07_pedestrian_walkway_corridor.jpg (from real CCTV footage frame)
src_ogv = os.path.join(SAMPLE_DIR, "real_pedestrian_crowd.ogv")
if os.path.exists(src_ogv):
    cap = cv2.VideoCapture(src_ogv)
    for _ in range(50):
        ret, fr = cap.read()
    if ret:
        cv2.imwrite(os.path.join(SAMPLE_DIR, "07_pedestrian_walkway_corridor.jpg"), fr)
    cap.release()

print("Populated rich sample dataset in sample_data/ with 7 distinct crowd & control scenes!")
