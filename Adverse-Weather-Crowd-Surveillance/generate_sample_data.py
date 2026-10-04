import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

def generate_base_scene(w=640, h=480):
    """Generates a rich outdoor architectural plaza scene with ground perspective."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Sky / background building
    for y in range(int(h * 0.45)):
        ratio = y / (h * 0.45)
        color = (int(180 + 30 * ratio), int(160 + 40 * ratio), int(140 + 50 * ratio))
        img[y, :] = color
        
    # Buildings on horizon
    cv2.rectangle(img, (50, 80), (180, int(h * 0.45)), (110, 105, 100), -1)
    cv2.rectangle(img, (220, 60), (380, int(h * 0.45)), (95, 90, 85), -1)
    cv2.rectangle(img, (420, 100), (580, int(h * 0.45)), (120, 115, 110), -1)
    
    # Plaza ground with perspective cobblestone / pavement
    for y in range(int(h * 0.45), h):
        depth_factor = (y - h * 0.45) / (h * 0.55)
        base_gray = int(90 + 70 * depth_factor)
        img[y, :] = (base_gray, base_gray, base_gray + 5)
        
        # Pavement grid perspective lines
        if y % max(4, int(20 * depth_factor)) == 0:
            img[y, :] = np.clip(img[y, :].astype(int) - 15, 0, 255).astype(np.uint8)
            
    # Converging perspective walkway lines
    for x_offset in [-200, -100, 0, 100, 200]:
        pt1 = (int(w / 2 + x_offset * 0.2), int(h * 0.45))
        pt2 = (int(w / 2 + x_offset * 1.8), h)
        cv2.line(img, pt1, pt2, (80, 80, 85), 1)
        
    return img

def add_pedestrians(img, frame_num=0, crowd_density=65):
    """Draws realistic pedestrian clusters with depth perspective and motion."""
    h, w, _ = img.shape
    canvas = img.copy()
    np.random.seed(42)  # Consistent base positions
    
    # Generate crowd distribution
    for i in range(crowd_density):
        # Depth perspective: y in [h*0.48, h*0.92]
        base_y = np.random.uniform(h * 0.48, h * 0.92)
        # Perspective scale: people farther away are much smaller
        scale = (base_y - h * 0.45) / (h * 0.55)  # 0.1 to 1.0
        
        # Speed varies with depth and direction
        direction = 1 if (i % 2 == 0) else -1
        speed = (1.5 + (i % 3) * 0.8) * direction * (0.4 + 0.6 * scale)
        base_x = (np.random.uniform(50, w - 50) + frame_num * speed) % (w - 60) + 30
        
        ped_h = int(55 * scale)
        ped_w = max(4, int(18 * scale))
        head_r = max(2, int(6 * scale))
        
        # Pedestrian color palette (jackets, trousers)
        colors = [
            ((40, 50, 70), (20, 20, 30)),
            ((60, 40, 40), (30, 30, 40)),
            ((30, 60, 50), (20, 25, 25)),
            ((80, 70, 50), (35, 30, 25)),
            ((50, 50, 60), (25, 25, 30))
        ]
        torso_c, leg_c = colors[i % len(colors)]
        
        px, py = int(base_x), int(base_y)
        
        # Shadow on ground
        cv2.ellipse(canvas, (px, py), (int(ped_w * 0.9), int(ped_w * 0.3)), 0, 0, 360, (50, 50, 50), -1)
        # Legs
        cv2.rectangle(canvas, (px - ped_w//3, py - ped_h//2), (px + ped_w//3, py), leg_c, -1)
        # Torso / jacket
        cv2.rectangle(canvas, (px - ped_w//2, py - ped_h + head_r*2), (px + ped_w//2, py - ped_h//2), torso_c, -1)
        # Head
        cv2.circle(canvas, (px, py - ped_h + head_r), head_r, (180, 160, 140), -1)
        
    return canvas

def apply_fog_degradation(img, beta=0.85):
    """Applies Koschmieder atmospheric scattering model: I(x) = J(x)*t(x) + A*(1 - t(x))."""
    h, w, _ = img.shape
    # Transmission map decreases with distance to horizon
    y_coords = np.linspace(0, 1, h)[:, None]
    # Far away (top/horizon) has dense fog t ~ 0.25; near ground t ~ 0.75
    t_map = np.clip(0.20 + 0.65 * y_coords, 0.15, 0.95)
    t_3ch = np.repeat(t_map[:, :, None], 3, axis=2)
    
    airlight = np.array([215, 220, 230], dtype=np.float32)
    img_f = img.astype(np.float32)
    hazy = img_f * t_3ch + airlight * (1.0 - t_3ch)
    return np.clip(hazy, 0, 255).astype(np.uint8)

def apply_rain_degradation(img, streak_count=1200):
    """Applies monsoon vertical rain streaks and atmospheric dampening."""
    h, w, _ = img.shape
    canvas = (img.astype(np.float32) * 0.88 + 15).astype(np.uint8)  # Overcast attenuation
    
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    for _ in range(streak_count):
        rx = np.random.randint(0, w)
        ry = np.random.randint(0, h - 35)
        length = np.random.randint(18, 35)
        slant = np.random.randint(-2, 2)
        thickness = 1
        cv2.line(rain_layer, (rx, ry), (rx + slant, ry + length), 255, thickness)
        
    # Gaussian blur along streaks
    blurred_rain = cv2.GaussianBlur(rain_layer, (3, 7), 0)
    streak_mask = blurred_rain.astype(np.float32) / 255.0 * 0.75
    
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c] * (1.0 - streak_mask) + 225 * streak_mask, 0, 255).astype(np.uint8)
        
    return canvas

def apply_glare_degradation(img):
    """Applies lens flare / specular bloom over-exposure."""
    h, w, _ = img.shape
    canvas = img.copy().astype(np.float32)
    
    # Bright glare center at upper-right
    cx, cy = int(w * 0.75), int(h * 0.25)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    
    # Specular bloom radial falloff
    bloom = np.exp(-dist_sq / (2 * (110**2))) * 255.0
    halo = np.exp(-dist_sq / (2 * (240**2))) * 120.0
    glare = np.clip(bloom + halo, 0, 255)[:, :, None]
    
    canvas = canvas + glare
    return np.clip(canvas, 0, 255).astype(np.uint8)

def main():
    print("[SAMPLE-GEN] Generating sample surveillance datasets for offline testing...")
    
    # 1. Base clean scene
    base = generate_base_scene(640, 480)
    
    # 2. Generate Still Test Images
    clear_img = add_pedestrians(base, frame_num=0, crowd_density=60)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "clear_plaza_crowd.png"), cv2.cvtColor(clear_img, cv2.COLOR_RGB2BGR))
    
    fog_img = apply_fog_degradation(add_pedestrians(base, frame_num=5, crowd_density=72))
    cv2.imwrite(os.path.join(SAMPLE_DIR, "foggy_plaza_crowd.png"), cv2.cvtColor(fog_img, cv2.COLOR_RGB2BGR))
    
    rain_img = apply_rain_degradation(add_pedestrians(base, frame_num=10, crowd_density=80))
    cv2.imwrite(os.path.join(SAMPLE_DIR, "monsoon_rain_crowd.png"), cv2.cvtColor(rain_img, cv2.COLOR_RGB2BGR))
    
    glare_img = apply_glare_degradation(add_pedestrians(base, frame_num=15, crowd_density=65))
    cv2.imwrite(os.path.join(SAMPLE_DIR, "lens_glare_crowd.png"), cv2.cvtColor(glare_img, cv2.COLOR_RGB2BGR))
    
    print("  Created 4 sample images in sample_data/")
    
    # 3. Generate Video Clips (30 frames @ 5 FPS = 6 seconds each)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Video 1: Foggy Plaza Surveillance Clip
    fog_vid_path = os.path.join(SAMPLE_DIR, "surveillance_fog_sample.mp4")
    vw_fog = cv2.VideoWriter(fog_vid_path, fourcc, 5.0, (640, 480))
    for f in range(25):
        fr = add_pedestrians(base, frame_num=f, crowd_density=68 + (f % 5))
        fr_fog = apply_fog_degradation(fr, beta=0.85 + 0.05 * np.sin(f * 0.2))
        vw_fog.write(cv2.cvtColor(fr_fog, cv2.COLOR_RGB2BGR))
    vw_fog.release()
    print(f"  Created video: {fog_vid_path}")
    
    # Video 2: Monsoon Rain Surveillance Clip
    rain_vid_path = os.path.join(SAMPLE_DIR, "surveillance_rain_sample.mp4")
    vw_rain = cv2.VideoWriter(rain_vid_path, fourcc, 5.0, (640, 480))
    for f in range(25):
        fr = add_pedestrians(base, frame_num=f, crowd_density=75 + (f % 8))
        fr_rain = apply_rain_degradation(fr, streak_count=1300 + (f % 200))
        vw_rain.write(cv2.cvtColor(fr_rain, cv2.COLOR_RGB2BGR))
    vw_rain.release()
    print(f"  Created video: {rain_vid_path}")
    
    print("[SAMPLE-GEN] Sample dataset creation complete!")

if __name__ == "__main__":
    main()
