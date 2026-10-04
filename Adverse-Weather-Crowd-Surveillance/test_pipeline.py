import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("RUNNING ARCHITECTURE & COMPONENT SANITY TESTS")
print("=" * 60)

# Test 1: Physics-Guided Dehaze Network
class MultiScaleConvBlock(nn.Module):
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

dehaze_net = PhysicsGuidedDehazeNet()
dummy_hazy = torch.rand(2, 3, 256, 256)
j_out, t_out, a_out = dehaze_net(dummy_hazy)
assert j_out.shape == (2, 3, 256, 256), f"Unexpected j_out shape {j_out.shape}"
assert t_out.shape == (2, 1, 256, 256), f"Unexpected t_out shape {t_out.shape}"
assert a_out.shape == (2, 3), f"Unexpected a_out shape {a_out.shape}"
assert (t_out >= 0.05).all() and (t_out <= 1.0).all(), "Transmission out of physical bounds [0.05, 1.0]"
print("[PASS] Test 1: PhysicsGuidedDehazeNet forward pass verified.")

# Test 2: CSRNet Model
import torchvision.models as models
vgg = models.vgg16(weights=None)
frontend = nn.Sequential(*list(vgg.features.children())[:23])
backend = nn.Sequential(
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

class MiniCSRNet(nn.Module):
    def __init__(self, frontend, backend):
        super().__init__()
        self.frontend = frontend
        self.backend = backend
    def forward(self, x):
        return F.relu(self.backend(self.frontend(x)))

crowd_net = MiniCSRNet(frontend, backend)
dummy_crowd_img = torch.rand(2, 3, 384, 512)
dmap_out = crowd_net(dummy_crowd_img)
assert dmap_out.shape == (2, 1, 384 // 8, 512 // 8), f"Unexpected dmap shape {dmap_out.shape}"
assert (dmap_out >= 0.0).all(), "Density map must be strictly non-negative"
print("[PASS] Test 2: CSRNet density map output verified.")

# Test 3: Density Map Generator Geometry & Count Conservation
from scipy.spatial import KDTree
def test_density_map_conservation():
    h, w = 120, 160
    points = [(30, 40), (32, 45), (100, 80), (105, 85), (50, 90)]
    pts = np.array(points, dtype=np.float32)
    tree = KDTree(pts)
    distances, _ = tree.query(pts, k=min(4, len(pts)))
    sigmas = np.clip(0.3 * np.mean(distances[:, 1:], axis=1), 3.0, 15.0)
    dmap = np.zeros((h, w), dtype=np.float32)
    for i, (px, py) in enumerate(pts):
        ix, iy = int(round(px)), int(round(py))
        sigma = sigmas[i]
        rad = int(round(3 * sigma))
        x_min, x_max = max(0, ix - rad), min(w, ix + rad + 1)
        y_min, y_max = max(0, iy - rad), min(h, iy + rad + 1)
        y_g, x_g = np.ogrid[y_min - iy : y_max - iy, x_min - ix : x_max - ix]
        k = np.exp(-(x_g**2 + y_g**2) / (2.0 * sigma**2))
        k /= k.sum()
        dmap[y_min:y_max, x_min:x_max] += k
    integrated_count = float(dmap.sum())
    assert abs(integrated_count - len(points)) < 0.05, f"Count conservation error: {integrated_count} vs {len(points)}"
    print(f"[PASS] Test 3: Density Map geometry & integration verified: Integrated sum = {integrated_count:.4f} for {len(points)} heads.")

test_density_map_conservation()

# Test 4: Dynamic Risk Engine
def evaluate_crowd_risk(crowd_count, weather_condition="clear", max_capacity=100):
    multipliers = {"clear": 1.0, "fog": 1.25, "rain": 1.30, "snow": 1.40}
    w_factor = multipliers.get(weather_condition.lower(), 1.0)
    raw_ratio = crowd_count / float(max_capacity)
    risk_score = np.clip(raw_ratio * w_factor, 0.0, 1.0)
    if risk_score >= 1.0 or crowd_count >= max_capacity:
        level, alert = "CRITICAL", "CAPACITY THRESHOLD EXCEEDED"
    elif risk_score >= 0.80:
        level, alert = "HIGH", "ELEVATED CONGESTION WARNING"
    elif risk_score >= 0.50:
        level, alert = "MEDIUM", "CAUTION MONITORING"
    else:
        level, alert = "LOW", "NORMAL CAPACITY"
    return {"risk_score": float(risk_score), "level": level, "alert": alert}

r1 = evaluate_crowd_risk(30, "clear", 100)
assert r1["level"] == "LOW"
r2 = evaluate_crowd_risk(75, "fog", 100)
assert r2["level"] == "HIGH", f"Expected HIGH, got {r2['level']}"
r3 = evaluate_crowd_risk(80, "rain", 100)
assert r3["level"] == "CRITICAL", f"Expected CRITICAL, got {r3['level']}"
print("[PASS] Test 4: Dynamic adverse-weather risk engine verified.")

print("=" * 60)
print("ALL TESTS PASSED WITH 100% SUCCESS!")
print("=" * 60)
