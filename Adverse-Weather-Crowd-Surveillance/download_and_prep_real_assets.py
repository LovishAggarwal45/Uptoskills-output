import os
import time
import urllib.request
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

headers = {'User-Agent': 'AdverseWeatherCrowdResearch/2.0 (mailto:researcher@university.edu)'}

DIRECT_ASSETS = [
    (
        "real_clear_shibuya_crowd.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/1/10/Ground_level_at_Shibuya_Crossing_in_Shibuya%2C_Tokyo%2C_Japan%2C_2024_May.jpg"
    ),
    (
        "real_monsoon_rain_crowd.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c0/People_with_Umbrellas_Walking_into_Hanzhong_Street%2C_Taipei_20160610.jpg"
    ),
    (
        "real_heavy_fog_crowd.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/94/Fog_in_Prague%2C_Czech_Republic.jpg"
    ),
    (
        "real_lens_glare_crowd.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/97/Times_Square%2C_Manhattan%2C_New_York_%287237736496%29.jpg"
    )
]

def download_and_crop(url, out_name, target_w=640, target_h=480):
    out_path = os.path.join(SAMPLE_DIR, out_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"Already exists: {out_path}")
        return cv2.imread(out_path)
        
    print(f"Downloading {out_name} from {url}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
    if img is None:
        print(f"Failed to decode image from {url}")
        return None
        
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    x1 = (nw - target_w) // 2
    y1 = (nh - target_h) // 2
    cropped = resized[y1:y1+target_h, x1:x1+target_w]
    
    cv2.imwrite(out_path, cropped)
    print(f"Successfully saved {out_path} ({cropped.shape})")
    time.sleep(1.0)
    return cropped

def main():
    for name, url in DIRECT_ASSETS:
        download_and_crop(url, name)
        
    # For real_lens_glare_crowd, let's ensure realistic specular sun glare is present on Times Square:
    glare_path = os.path.join(SAMPLE_DIR, "real_lens_glare_crowd.jpg")
    if os.path.exists(glare_path):
        img = cv2.imread(glare_path)
        h, w = img.shape[:2]
        # Add optical specular bloom on upper corner
        cx, cy = int(w * 0.8), int(h * 0.15)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        bloom = np.exp(-dist_sq / (2 * (120**2))) * 255.0
        halo = np.exp(-dist_sq / (2 * (250**2))) * 110.0
        glare = np.clip(bloom + halo, 0, 255)[:, :, None]
        img_glare = np.clip(img.astype(np.float32) + glare, 0, 255).astype(np.uint8)
        cv2.imwrite(glare_path, img_glare)
        print("Applied optical lens flare on real Times Square crowd image.")

if __name__ == "__main__":
    main()
