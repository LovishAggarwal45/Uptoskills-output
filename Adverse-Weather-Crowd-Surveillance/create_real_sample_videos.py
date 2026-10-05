import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
src_video = os.path.join(SAMPLE_DIR, "real_pedestrian_crowd.ogv")

def apply_fog(frame, beta=0.75):
    """Applies Koschmieder atmospheric scattering to real video frame."""
    h, w = frame.shape[:2]
    # Depth gradient
    y_coords = np.linspace(0.2, 0.85, h)[:, None]
    t = np.clip(y_coords, 0.15, 0.90)
    t_3ch = np.repeat(t[:, :, None], 3, axis=2)
    airlight = np.array([215, 220, 230], dtype=np.float32)
    
    fr_f = frame.astype(np.float32)
    hazy = fr_f * t_3ch + airlight * (1.0 - t_3ch)
    return np.clip(hazy, 0, 255).astype(np.uint8)

def apply_rain(frame, frame_idx=0):
    """Applies monsoon directional rain streaks to real video frame."""
    h, w = frame.shape[:2]
    canvas = (frame.astype(np.float32) * 0.88 + 12).astype(np.uint8)
    
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    np.random.seed(42 + frame_idx * 7)
    for _ in range(900):
        rx = np.random.randint(0, w)
        ry = np.random.randint(0, h - 30)
        l = np.random.randint(18, 30)
        cv2.line(rain_layer, (rx, ry), (rx, ry + l), 255, 1)
        
    blur = cv2.GaussianBlur(rain_layer, (1, 5), 0)
    mask = blur.astype(np.float32) / 255.0 * 0.7
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c] * (1.0 - mask) + 230 * mask, 0, 255)
    return canvas

def create_videos():
    print(f"Reading real human pedestrian footage from {src_video}...")
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError("Could not open source pedestrian video")
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15.0
    num_frames = 60  # 4 seconds of surveillance footage
    
    fog_out = os.path.join(SAMPLE_DIR, "surveillance_fog_sample.mp4")
    rain_out = os.path.join(SAMPLE_DIR, "surveillance_rain_sample.mp4")
    
    # Use avc1 / mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_fog = cv2.VideoWriter(fog_out, fourcc, fps, (w, h))
    vw_rain = cv2.VideoWriter(rain_out, fourcc, fps, (w, h))
    
    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        fr_fog = apply_fog(frame, beta=0.75 + 0.05 * np.sin(idx * 0.2))
        fr_rain = apply_rain(frame, frame_idx=idx)
        
        vw_fog.write(fr_fog)
        vw_rain.write(fr_rain)
        
    cap.release()
    vw_fog.release()
    vw_rain.release()
    
    print(f"Created real fog surveillance video: {fog_out} ({os.path.getsize(fog_out)} bytes)")
    print(f"Created real rain surveillance video: {rain_out} ({os.path.getsize(rain_out)} bytes)")

if __name__ == "__main__":
    create_videos()
