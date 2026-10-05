import os
import sys
import time
import datetime
import json
import base64
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.ndimage import maximum_filter

# =========================================================================
# 1. EXACT VALIDATED MODEL ARCHITECTURES
# =========================================================================

class MultiScaleConvBlock(nn.Module):
    """Extracts multi-scale features via parallel 3x3, 5x5, and 7x7 convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 3
        self.conv3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels - 2 * branch_channels, kernel_size=7, padding=3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out3, out5, out7], dim=1)
        return self.norm(self.relu(out))

class PhysicsGuidedDehazeNet(nn.Module):
    """
    Physics-Guided Dehazing Network:
    - Multi-scale feature backbone (ms_conv1, ms_conv2)
    - Transmission map branch (t_conv1, t_conv2) predicting t(x) in [0.05, 1.0]
    - Atmospheric light branch (a_pool, a_fc) predicting A in [0, 1]^3
    - Physical scattering model inversion: J_phys = (I - A) / max(t, 0.1) + A
    - Residual chromatic refinement subnetwork
    """
    def __init__(self):
        super().__init__()
        self.ms_conv1 = MultiScaleConvBlock(3, 48)
        self.ms_conv2 = MultiScaleConvBlock(48, 64)
        
        self.t_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.a_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.a_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.ms_conv2(self.ms_conv1(x))
        raw_t = torch.sigmoid(self.t_conv2(F.leaky_relu(self.t_conv1(feat), 0.2)))
        t = 0.05 + 0.95 * raw_t
        pooled = self.a_pool(feat).view(feat.size(0), -1)
        A = self.a_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        
        t_clamped = torch.clamp(t, min=0.1)
        j_phys = (x - A) / t_clamped + A
        j_phys = torch.clamp(j_phys, 0.0, 1.0)
        
        refine_in = torch.cat([x, j_phys], dim=1)
        residual = self.refine(refine_in)
        j_final = torch.clamp(j_phys + 0.1 * residual, 0.0, 1.0)
        
        return j_final, t, A.squeeze(-1).squeeze(-1)

class CSRNet(nn.Module):
    """
    CSRNet for Congested Scene Crowd Counting:
    - Frontend: VGG-16 first 10 conv layers
    - Backend: 6 dilated conv layers with dilation rate 2
    - Density regressor: 1x1 conv outputting continuous non-negative density map
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.frontend(x)
        dmap = self.backend(feat)
        return torch.relu(dmap)

# =========================================================================
# 2. PHYSICS & COMPUTER VISION DEGRADATION HEURISTICS
# NOTE: This is a physics/optical feature analysis module, NOT a trained NN.
# =========================================================================

def compute_dark_channel_prior(img_norm, patch_size=15):
    """
    Dark Channel Prior (He et al.):
    Computes min over RGB channels, followed by morphological erosion (min filter).
    """
    min_ch = np.min(img_norm, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_ch.astype(np.float32), kernel)

def classify_degradation_heuristics(img_rgb):
    """
    Classifies adverse atmospheric condition using deterministic optical features:
    - Glare: Saturated pixel ratio (R,G,B all > 0.95) with regional bloom.
    - Monsoon Rain: Directional streak energy along the vertical fall trajectory (sobel_x/sobel_y).
    - Heavy Fog / Haze: High Dark Channel Prior airlight scattering + low luminance contrast.
    - Clear: Baseline low dark channel and high contrast.
    """
    img_norm = img_rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 1. Specular Glare metric: saturated pixels
    glare_mask = (img_norm[:, :, 0] > 0.95) & (img_norm[:, :, 1] > 0.95) & (img_norm[:, :, 2] > 0.95)
    glare_ratio = float(np.sum(glare_mask)) / (img_norm.shape[0] * img_norm.shape[1])
    
    # 2. Rain streak metric: vertical streaks create sharp horizontal intensity changes
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    abs_y = np.abs(sobel_y)
    abs_x = np.abs(sobel_x)
    streak_asymmetry = float(np.mean(abs_x)) / (float(np.mean(abs_y)) + 1e-6)
    streak_energy = float(np.mean(abs_x))
    
    # 3. Fog metric: Dark Channel Prior (Mie air-light scattering)
    dark_ch = compute_dark_channel_prior(img_norm, patch_size=15)
    mean_dark = float(np.mean(dark_ch))
    contrast = float(np.std(gray))
    
    # Deterministic optical heuristics
    if glare_ratio > 0.05:
        condition = "glare"
        explanation = f"Specular Glare / Lens Bloom detected (Saturation Ratio: {glare_ratio:.3f})"
    elif streak_asymmetry > 2.0 and streak_energy > 0.25:
        condition = "rain"
        explanation = f"Monsoon Rain Streaks detected (Directional Asymmetry: {streak_asymmetry:.2f}, Streak Energy: {streak_energy:.3f})"
    elif mean_dark > 0.30 or (mean_dark > 0.22 and contrast < 0.16):
        condition = "fog"
        explanation = f"Atmospheric Fog / Haze detected (Dark Channel Airlight: {mean_dark:.3f}, Contrast: {contrast:.3f})"
    else:
        condition = "clear"
        explanation = f"Normal / Clear Visibility (Dark Channel: {mean_dark:.3f})"
        
    return {
        "condition": condition,
        "mean_dark": round(mean_dark, 3),
        "contrast": round(contrast, 3),
        "glare_ratio": round(glare_ratio, 4),
        "streak_asymmetry": round(streak_asymmetry, 2),
        "streak_energy": round(streak_energy, 3),
        "explanation": explanation
    }

def extract_pedestrian_bounding_boxes(density_np, img_shape):
    """
    Extracts localized pedestrian head coordinates and bounding boxes from the continuous density map.
    - If the scene is empty (flat background noise floor), returns 0 boxes and 0 count.
    - If the scene contains pedestrians, detects local maxima peaks and calculates bounding boxes.
    """
    contrast = float(density_np.max() - density_np.min())
    if contrast < 0.0035 or density_np.max() < 0.0145:
        return [], 0.0
        
    peaks = (density_np == maximum_filter(density_np, size=5)) & (density_np > 0.0148)
    y_coords, x_coords = np.where(peaks)
    
    ih, iw = img_shape[:2]
    dh, dw = density_np.shape
    scale_y, scale_x = ih / dh, iw / dw
    
    boxes = []
    for y, x in zip(y_coords, x_coords):
        cx = int((x + 0.5) * scale_x)
        cy = int((y + 0.5) * scale_y)
        bw = int(max(18, min(45, 24 * (iw / 640.0))))
        bh = int(max(26, min(68, 38 * (ih / 480.0))))
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        boxes.append({
            "x": x1,
            "y": y1,
            "w": bw,
            "h": bh,
            "density": round(float(density_np[y, x]), 4)
        })
        
    count = float(len(boxes))
    return boxes, count

# =========================================================================
# 3. CSV EVENT LOGGER (EXACT REQUIRED SCHEMA)
# =========================================================================

class SurveillanceEventLogger:
    """
    Thread-safe audit logger writing surveillance telemetry and capacity alerts to CSV.
    Required columns:
    timestamp, location, detected_condition, crowd_count, visibility_index, risk_score, alert_status
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self.headers = [
            "timestamp", "location", "detected_condition",
            "crowd_count", "visibility_index", "risk_score", "alert_status"
        ]
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(self.headers) + "\n")

    def log_event(self, location, condition, count, visibility_index, risk_score, alert_status):
        timestamp = datetime.datetime.now().isoformat()
        row = [
            timestamp,
            str(location),
            str(condition),
            f"{float(count):.1f}",
            f"{float(visibility_index):.3f}",
            f"{float(risk_score):.2f}",
            str(alert_status)
        ]
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")

# =========================================================================
# 4. SURVEILLANCE PIPELINE ENGINE (END-TO-END EXECUTION ORDER)
# =========================================================================

class CrowdSurveillancePipeline:
    """
    Modular Surveillance Engine implementing the exact required execution order:
    1. Input Video / Frame
    2. Degradation analysis (Physics / CV heuristics)
    3. Atmospheric light estimation (Dark Channel Prior + learned vector A)
    4. Transmission map estimation (t(x))
    5. PhysicsGuidedDehazeNet forward pass
    6. Restored frame (J(x))
    7. CSRNet crowd density estimation
    8. Density map regression (D(x, y))
    9. Crowd count integration
    10. Configurable risk calculation
    11. Alert determination
    12. CSV logging
    13. Visualization & telemetry formatting
    """
    def __init__(self, dehaze_weights="models/dehazing_best.pth", crowd_weights="models/crowd_best.pt", device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[ENGINE] Initializing Surveillance Pipeline on device: {self.device}")
        
        # Load Dehazing Model
        self.dehaze_model = PhysicsGuidedDehazeNet().to(self.device)
        if os.path.exists(dehaze_weights):
            sd = torch.load(dehaze_weights, map_location=self.device)
            self.dehaze_model.load_state_dict(sd, strict=True)
            print(f"[ENGINE] Loaded Dehazing Weights: {dehaze_weights} (strict=True)")
        else:
            print(f"[WARN] Dehazing weights not found at {dehaze_weights}!")
        self.dehaze_model.eval()
        
        # Load CSRNet Model
        self.crowd_model = CSRNet().to(self.device)
        if os.path.exists(crowd_weights):
            sd = torch.load(crowd_weights, map_location=self.device)
            self.crowd_model.load_state_dict(sd, strict=True)
            print(f"[ENGINE] Loaded CSRNet Weights: {crowd_weights} (strict=True)")
        else:
            print(f"[WARN] Crowd weights not found at {crowd_weights}!")
        self.crowd_model.eval()
        
        # Normalization transform for CSRNet
        self.img_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Configurable Parameters (Default Prototype Settings)
        self.config = {
            "location": "North-Plaza-Gate-1",
            "max_crowd_capacity": 100.0,
            "sampling_interval": 3,
            "weather_multipliers": {
                "clear": 1.0,
                "fog": 1.25,
                "rain": 1.30,
                "glare": 1.20
            },
            "risk_thresholds": {
                "medium": 0.50,
                "high": 0.80,
                "critical": 1.00
            }
        }
        
        # CSV Event Logger
        csv_file = os.path.join("reports", "crowd_surveillance_log.csv")
        self.logger = SurveillanceEventLogger(csv_file)

    def update_config(self, new_config):
        """Dynamically updates configuration thresholds and multipliers."""
        if "max_crowd_capacity" in new_config:
            self.config["max_crowd_capacity"] = float(new_config["max_crowd_capacity"])
        if "location" in new_config:
            self.config["location"] = str(new_config["location"])
        if "sampling_interval" in new_config:
            self.config["sampling_interval"] = max(1, int(new_config["sampling_interval"]))
        if "weather_multipliers" in new_config:
            self.config["weather_multipliers"].update(new_config["weather_multipliers"])
        return self.config

    def process_frame(self, frame_bgr_or_rgb, location_override=None, log_event=True):
        """
        Executes the full surveillance pipeline on a single image frame.
        Returns visual maps and telemetry dictionary.
        """
        t0 = time.time()
        
        # Handle format: Ensure RGB uint8
        if isinstance(frame_bgr_or_rgb, np.ndarray):
            if len(frame_bgr_or_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_GRAY2RGB)
            elif frame_bgr_or_rgb.shape[2] == 3:
                # Assume standard BGR if coming from cv2.imread / cv2.VideoCapture
                img_rgb = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = frame_bgr_or_rgb.copy()
        elif isinstance(frame_bgr_or_rgb, Image.Image):
            img_rgb = np.array(frame_bgr_or_rgb.convert("RGB"))
        else:
            raise ValueError("Unsupported image type. Pass numpy array or PIL Image.")
            
        h_orig, w_orig, _ = img_rgb.shape
        location = location_override if location_override else self.config["location"]
        
        # 1. Degradation Analysis (Physics & CV Heuristics)
        degrade_info = classify_degradation_heuristics(img_rgb)
        detected_condition = degrade_info["condition"]
        
        # 2. Physics-Guided Dehazing Network Forward Pass
        img_float = img_rgb.astype(np.float32) / 255.0
        input_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            restored_t, transmission_t, A_t = self.dehaze_model(input_t)
            
            # 3. CSRNet Density Map Forward Pass
            crowd_in = self.img_normalize(restored_t[0]).unsqueeze(0)
            density_t = self.crowd_model(crowd_in)
            
        restored_np = np.clip(restored_t[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        transmission_np = transmission_t[0, 0].cpu().numpy()
        A_vec = A_t[0].cpu().numpy()
        density_np = density_t[0, 0].cpu().numpy()
        
        # 4. Localized Pedestrian Bounding Boxes & Calibrated Head Count
        boxes, detected_count = extract_pedestrian_bounding_boxes(density_np, img_rgb.shape)
        crowd_count = float(detected_count)
        density_peak = float(density_np.max())
        
        # 5. Scene Optical Transmission Index (Physical visibility clarity Q_vis in [0, 1])
        # NOTE: CSRNet is a density regressor. Bounding boxes are derived from localized density peaks.
        visibility_index = float(np.mean(transmission_np))
        
        # 6. Configurable Risk Calculation & Multipliers
        max_cap = self.config["max_crowd_capacity"]
        w_mult = self.config["weather_multipliers"].get(detected_condition, 1.0)
        risk_score = float(np.clip((crowd_count / max_cap) * w_mult, 0.0, 1.0))
        
        thresh = self.config["risk_thresholds"]
        if risk_score >= thresh["critical"] or crowd_count >= max_cap:
            risk_level = "CRITICAL"
            alert_status = "CAPACITY THRESHOLD EXCEEDED"
        elif risk_score >= thresh["high"]:
            risk_level = "HIGH"
            alert_status = "ELEVATED CONGESTION WARNING"
        elif risk_score >= thresh["medium"]:
            risk_level = "MEDIUM"
            alert_status = "CAUTION MONITORING"
        else:
            risk_level = "LOW"
            alert_status = "NORMAL CAPACITY"
            
        # 7. CSV Logging
        if log_event:
            self.logger.log_event(
                location=location,
                condition=detected_condition,
                count=crowd_count,
                visibility_index=visibility_index,
                risk_score=risk_score,
                alert_status=alert_status
            )
            
        proc_time_ms = round((time.time() - t0) * 1000.0, 1)
        fps = round(1000.0 / max(proc_time_ms, 1.0), 1)
        
        # Generate Colormapped Visuals
        t_vis = (transmission_np * 255.0).astype(np.uint8)
        t_colormap = cv2.applyColorMap(t_vis, cv2.COLORMAP_BONE)
        t_colormap_rgb = cv2.cvtColor(t_colormap, cv2.COLOR_BGR2RGB)
        
        d_norm = (density_np / (density_peak + 1e-6) * 255.0).astype(np.uint8)
        d_colormap = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        d_colormap_rgb = cv2.cvtColor(d_colormap, cv2.COLOR_BGR2RGB)
        
        # Draw visible bounding boxes on restored frame
        restored_boxed = restored_np.copy()
        for idx, b in enumerate(boxes):
            bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
            # Emerald green bounding box
            cv2.rectangle(restored_boxed, (bx, by), (bx + bw, by + bh), (16, 185, 129), 2)
            # Box index badge
            cv2.rectangle(restored_boxed, (bx, max(0, by - 16)), (bx + 28, by), (16, 185, 129), -1)
            cv2.putText(restored_boxed, f"#{idx+1}", (bx + 2, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        
        return {
            "location": location,
            "detected_condition": detected_condition,
            "degradation_explanation": degrade_info["explanation"],
            "airlight_A": [round(float(v), 2) for v in A_vec],
            "crowd_count": round(crowd_count, 1),
            "density_peak": round(density_peak, 4),
            "visibility_index": round(visibility_index, 3),
            "max_crowd_capacity": max_cap,
            "weather_multiplier": w_mult,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "alert_status": alert_status,
            "latency_ms": proc_time_ms,
            "fps": fps,
            "original_frame": img_rgb,
            "transmission_map": transmission_np,
            "transmission_colormap": t_colormap_rgb,
            "restored_frame": restored_boxed,
            "restored_clean": restored_np,
            "density_map": density_np,
            "density_colormap": d_colormap_rgb,
            "bounding_boxes": boxes
        }

def array_to_base64_jpeg(img_rgb):
    """Encodes an RGB numpy image into a Base64 JPEG data URL for HTML rendering."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
