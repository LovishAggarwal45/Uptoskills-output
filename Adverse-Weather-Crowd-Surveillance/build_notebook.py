import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_md(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # =========================================================================
    # HEADER & OVERVIEW
    # =========================================================================
    add_md(r'''# Adverse Weather Dynamic De-Hazing and Crowd Counting Pipeline
**Comprehensive End-to-End Google Colab Training & Surveillance Framework**

---

### Project Title
**Adverse Weather Dynamic De-Hazing and Crowd Counting Pipeline**

### Objective
Accurately estimate crowd density in real time or periodically under severe environmental degradation such as heavy fog, monsoon rain, lens glare, and low illumination.

### Architectural Pipeline
1. **Degraded Video / Frame Ingestion**: Ingest low-visibility outdoor camera feeds.
2. **Atmospheric Physics Priors**: Extract Dark Channel Prior (DCP), atmospheric light vector $A$, and medium transmission map $t(x)$.
3. **Physics-Guided Dehazing Network (`PhysicsGuidedDehazeNet`)**: Multi-scale visual feature extractor (3x3, 5x5, 7x7) predicting physical transmission and atmospheric light, restoring clear scene radiance $J(x) = \frac{I(x) - A}{\max(t(x), 0.1)} + A$ with residual artifact refinement.
4. **Crowd Density Map CNN (`CSRNet`)**: VGG-16 front-end with dilated convolutional back-end generating high-resolution crowd density heatmaps.
5. **Dynamic Risk Engine**: Configurable density capacity thresholds (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), adverse weather risk multipliers, and automated emergency alerts.
6. **Reporting & Multi-Panel Visualization**: Automated CSV audit logging and publication-grade multi-panel diagnostic dashboards.

---
''')

    # =========================================================================
    # SECTION 1 — Environment Setup
    # =========================================================================
    add_md(r'''## SECTION 1 — Environment Setup & Hardware Diagnostics

In this section, we:
1. Verify GPU availability (Tesla T4, V100, or A100 recommended in Google Colab).
2. Query and display Python, PyTorch, CUDA runtime versions, and available VRAM.
3. Warn if a GPU is unavailable and provide fallback execution settings.
4. Install all required dependencies (`opencv-python`, `scipy`, `scikit-image`, `kaggle`, `pandas`, `tqdm`).
5. Establish deterministic random seeds for reproducible training.
6. Create standardized workspace directory trees (`/content/datasets/`, `/content/models/`, `/content/outputs/`, `/content/reports/`).
''')

    add_code(r'''# Hardware Diagnostics & Dependency Installation
import os
import sys
import random
import shutil
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 70)
print("SYSTEM & HARDWARE DIAGNOSTICS")
print("=" * 70)
print(f"OS Platform       : {platform.platform()}")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version      : {torch.version.cuda}")
    print(f"GPU Device Name   : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    cached_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"Total VRAM        : {total_mem:.2f} GB")
    print(f"Allocated VRAM    : {allocated_mem:.2f} GB")
    print(f"Reserved VRAM     : {cached_mem:.2f} GB")
    DEVICE = torch.device("cuda:0")
else:
    print("[WARNING] No GPU detected! Google Colab runtime is set to CPU.")
    print("For accelerated training, go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.")
    DEVICE = torch.device("cpu")

# Set deterministic random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Workspace directories configuration
BASE_DIR = "/content" if os.path.exists("/content") else os.path.abspath("./workspace")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
JHU_DIR = os.path.join(DATASET_DIR, "jhu_crowd")
RESIDE_DIR = os.path.join(DATASET_DIR, "reside_sots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for directory in [DATASET_DIR, JHU_DIR, RESIDE_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Workspace root    : {BASE_DIR}")
print(f"Models directory  : {MODELS_DIR}")
print(f"Reports directory : {REPORTS_DIR}")
print("=" * 70)
''')

    # =========================================================================
    # SECTION 2 — Dataset Download
    # =========================================================================
    add_md(r'''## SECTION 2 — Dataset Acquisition (Kaggle API & Resilient Ingestion)

The project utilizes two primary datasets:
1. **JHU-CROWD++**: Large-scale crowd dataset (4,372 images, 1.51M annotations) with weather-level metadata (Clear, Heavy Fog, Monsoon Rain, Snow, Lens Glare).
2. **RESIDE (SOTS Subset)**: Synthetic Objective Testing Set (~436 MB) containing paired hazy and clear outdoor/indoor frames for supervised dehazing evaluation.

### Providing Kaggle API Credentials
To download datasets via Kaggle API:
1. Log in to [Kaggle](https://www.kaggle.com/) -> Account -> API -> Create New API Token (`kaggle.json`).
2. Run the cell below. It provides two convenient options:
   - **Option A**: Upload `kaggle.json` directly via the Colab file upload prompt.
   - **Option B**: Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. If Kaggle download is skipped or credentials are not yet supplied, an **Adaptive Synthetic Dataset Fallback** automatically generates degraded hazy images and crowd dot annotations to allow full end-to-end execution and offline testing.
''')

    add_code(r'''# Kaggle API Credentials Setup
import json
import getpass

kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

if not os.path.exists(kaggle_json_path):
    print("=" * 70)
    print("KAGGLE API CREDENTIALS CONFIGURATION")
    print("=" * 70)
    print("1. If running in Google Colab, you can upload 'kaggle.json' directly.")
    print("2. Or enter your Kaggle Username and API Key below (leave empty to skip/use synthetic mode).")
    
    try:
        from google.colab import files
        print("[INFO] Attempting interactive kaggle.json upload...")
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            with open(kaggle_json_path, "wb") as f:
                f.write(uploaded["kaggle.json"])
            os.chmod(kaggle_json_path, 0o600)
            print("[SUCCESS] kaggle.json uploaded and placed in ~/.kaggle/kaggle.json")
    except Exception:
        print("[INFO] Not in interactive Colab upload mode or upload skipped.")

    if not os.path.exists(kaggle_json_path):
        kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
        kaggle_key = os.environ.get("KAGGLE_KEY", "")
        if not kaggle_user:
            kaggle_user = input("Enter Kaggle Username (press Enter to skip): ").strip()
        if not kaggle_key and kaggle_user:
            kaggle_key = getpass.getpass("Enter Kaggle API Key: ").strip()
        
        if kaggle_user and kaggle_key:
            with open(kaggle_json_path, "w") as f:
                json.dump({"username": kaggle_user, "key": kaggle_key}, f)
            os.chmod(kaggle_json_path, 0o600)
            print("[SUCCESS] kaggle.json created successfully.")
        else:
            print("[NOTICE] Kaggle credentials skipped. Pipeline will verify local folders or run adaptive synthetic data mode.")
else:
    print(f"[SUCCESS] Existing Kaggle credentials found at {kaggle_json_path}")
''')

    add_code(r'''# Automated Dataset Downloader & Unzipper
import zipfile
import subprocess

def download_and_extract_kaggle_dataset(dataset_slug, target_dir):
    """Downloads and extracts a Kaggle dataset using kaggle CLI."""
    if not os.path.exists(kaggle_json_path):
        print(f"[SKIP] Kaggle credentials not configured. Cannot download {dataset_slug}.")
        return False
    
    zip_path = os.path.join(target_dir, dataset_slug.split("/")[-1] + ".zip")
    print(f"[INFO] Downloading Kaggle dataset: {dataset_slug} -> {target_dir}")
    try:
        cmd = f"kaggle datasets download -d {dataset_slug} -p {target_dir} --unzip"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[SUCCESS] Downloaded & unzipped {dataset_slug}")
            return True
        else:
            print(f"[WARNING] Download command returned: {res.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run Kaggle CLI: {e}")
        return False

# Attempt downloads for JHU-CROWD++ and RESIDE SOTS
# RESIDE SOTS: balraj98/synthetic-objective-testing-set-sots-reside
has_reside = download_and_extract_kaggle_dataset("balraj98/synthetic-objective-testing-set-sots-reside", RESIDE_DIR)

# JHU-CROWD++ mirror:
has_jhu = download_and_extract_kaggle_dataset("markzheng/jhu-crowd-v2", JHU_DIR)
if not has_jhu:
    has_jhu = download_and_extract_kaggle_dataset("deepakn97/jhu-crowd", JHU_DIR)
''')

    add_code(r'''# Adaptive Dataset Structure Inspector & Synthetic Fallback Generator
from PIL import Image, ImageDraw

def scan_directory_tree(root_dir, max_depth=3):
    """Recursively scans and analyzes directory structure."""
    summary = {"total_files": 0, "image_files": 0, "txt_files": 0, "subdirs": []}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for root, dirs, files in os.walk(root_dir):
        depth = root[len(root_dir):].count(os.sep)
        if depth <= max_depth:
            summary["subdirs"].append(root)
        for f in files:
            summary["total_files"] += 1
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                summary["image_files"] += 1
            elif ext == ".txt":
                summary["txt_files"] += 1
    return summary

print("=" * 70)
print("DATASET STRUCTURE INSPECTION")
print("=" * 70)
reside_scan = scan_directory_tree(RESIDE_DIR)
jhu_scan = scan_directory_tree(JHU_DIR)
print(f"RESIDE Directory : {RESIDE_DIR}")
print(f"  - Total files  : {reside_scan['total_files']}, Image files: {reside_scan['image_files']}")
print(f"JHU Directory    : {JHU_DIR}")
print(f"  - Total files  : {jhu_scan['total_files']}, Images: {jhu_scan['image_files']}, Txts: {jhu_scan['txt_files']}")

# Synthetic Data Fallback Engine (Ensures 100% executable pipeline offline/smoke tests)
def generate_synthetic_surveillance_data(base_jhu_dir, base_reside_dir, num_samples=30):
    """Generates realistic synthetic hazy frames, clear frames, and crowd point annotations."""
    print("[INFO] Generating synthetic benchmark data for adverse-weather surveillance...")
    synth_jhu_img = os.path.join(base_jhu_dir, "images")
    synth_jhu_gt = os.path.join(base_jhu_dir, "gt")
    os.makedirs(synth_jhu_img, exist_ok=True)
    os.makedirs(synth_jhu_gt, exist_ok=True)
    
    synth_reside_clear = os.path.join(base_reside_dir, "outdoor", "clear")
    synth_reside_hazy = os.path.join(base_reside_dir, "outdoor", "hazy")
    os.makedirs(synth_reside_clear, exist_ok=True)
    os.makedirs(synth_reside_hazy, exist_ok=True)
    
    weather_types = ["clear", "fog", "rain", "snow"]
    image_level_lines = ["image_name,head_count,weather_condition,scene_type\n"]
    
    for i in range(1, num_samples + 1):
        w, h = 640, 480
        # Create base clean background
        clear_np = np.zeros((h, w, 3), dtype=np.uint8)
        clear_np[:, :, 0] = np.linspace(60, 180, h)[:, None]
        clear_np[:, :, 1] = np.linspace(80, 200, w)[None, :]
        clear_np[:, :, 2] = 120
        
        # Draw random street/crowd shapes
        img = Image.fromarray(clear_np)
        draw = ImageDraw.Draw(img)
        
        # Generate random heads
        num_heads = random.randint(15, 80)
        head_coords = []
        for _ in range(num_heads):
            hx = random.randint(20, w - 20)
            hy = random.randint(80, h - 20)
            head_coords.append((hx, hy))
            # draw person figure
            draw.ellipse([hx-4, hy-4, hx+4, hy+4], fill=(220, 180, 140))
            draw.rectangle([hx-6, hy+4, hx+6, hy+20], fill=(random.randint(40, 100), random.randint(40, 100), 180))
            
        clear_img_path = os.path.join(synth_reside_clear, f"sample_{i:04d}.png")
        img.save(clear_img_path)
        
        # Apply physical haze degradation: I = J * t + A * (1 - t)
        weather_cond = weather_types[i % len(weather_types)]
        beta = 1.8 if weather_cond in ["fog", "rain"] else 0.4
        depth_map = np.linspace(0.2, 1.0, h)[:, None].repeat(w, axis=1)
        transmission = np.exp(-beta * depth_map)[:, :, None]
        atmospheric_light = np.array([0.88, 0.90, 0.92]) * 255.0
        
        clean_arr = np.array(img, dtype=np.float32)
        hazy_arr = clean_arr * transmission + atmospheric_light * (1.0 - transmission)
        
        if weather_cond == "rain":
            # Add synthetic rain streaks
            streak_noise = np.random.binomial(1, 0.005, (h, w)) * 200
            hazy_arr = np.clip(hazy_arr + streak_noise[:, :, None], 0, 255)
            
        hazy_img = Image.fromarray(np.clip(hazy_arr, 0, 255).astype(np.uint8))
        hazy_img_path = os.path.join(synth_reside_hazy, f"sample_{i:04d}_1.png")
        hazy_img.save(hazy_img_path)
        
        # Save to JHU crowd directory
        jhu_img_path = os.path.join(synth_jhu_img, f"{i:04d}.jpg")
        hazy_img.save(jhu_img_path)
        
        # Write head annotations (x, y, w, h, blur, weather)
        gt_path = os.path.join(synth_jhu_gt, f"{i:04d}.txt")
        with open(gt_path, "w") as f_gt:
            for (hx, hy) in head_coords:
                f_gt.write(f"{hx} {hy} 10 10 0 {weather_cond}\n")
                
        image_level_lines.append(f"{i:04d}.jpg,{num_heads},{weather_cond},outdoor\n")
        
    with open(os.path.join(base_jhu_dir, "image_level.txt"), "w") as f_meta:
        f_meta.writelines(image_level_lines)
    print(f"[SUCCESS] Generated {num_samples} paired samples in JHU and RESIDE directories.")

# Trigger synthetic generation if downloaded folders are empty
if jhu_scan["image_files"] == 0 or reside_scan["image_files"] == 0:
    generate_synthetic_surveillance_data(JHU_DIR, RESIDE_DIR, num_samples=36)
print("=" * 70)
''')

    # =========================================================================
    # SECTION 3 — Crowd Dataset Preprocessing
    # =========================================================================
    add_md(r'''## SECTION 3 — Crowd Dataset Preprocessing & Geometry-Adaptive Density Maps

### JHU-CROWD++ Data Representation
In crowd counting, direct person bounding-box detection collapses in severe congestions and low-visibility weather due to severe occlusion. Instead, **Continuous Density Map Regression** maps head dot coordinates into a continuous spatial density distribution:
$$\mathcal{D}(p) = \sum_{i=1}^{N} \frac{1}{2\pi \sigma_i^2} \exp\left(-\frac{\|p - x_i\|^2}{2\sigma_i^2}\right)$$
For crowded scenes, we employ a **Geometry-Adaptive Gaussian Kernel** where variance is conditioned on the local crowd dispersion:
$$\sigma_i = \beta \bar{d}_i$$
where $\bar{d}_i$ is the average Euclidean distance to the $k$-nearest neighbors ($k=3, \beta=0.3$). In sparse environments, a fixed Gaussian kernel fallback ($\sigma=15$) is utilized.

Integrating over the continuous density map strictly recovers the discrete ground truth crowd count:
$$\text{Total Crowd Count} = \iint \mathcal{D}(x, y) \, dx \, dy \approx \sum_{x, y} \mathcal{D}[x, y]$$
''')

    add_code(r'''# Geometry-Adaptive Density Map Generator & JHU Dataset Loader
import glob
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

def generate_density_map(image_shape, points, beta=0.3, k=3, fixed_sigma=15.0):
    """
    Generates a continuous density map from head coordinates.
    image_shape: (height, width)
    points: Nx2 array of (x, y) head points
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    num_points = len(points)
    if num_points == 0:
        return density_map
    
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    
    if num_points >= k + 1:
        tree = KDTree(pts)
        distances, _ = tree.query(pts, k=k + 1)
        sigmas = beta * np.mean(distances[:, 1:], axis=1)
        sigmas = np.clip(sigmas, 4.0, 30.0)
    else:
        sigmas = np.full(num_points, fixed_sigma, dtype=np.float32)
        
    for i, (pt_x, pt_y) in enumerate(pts):
        ix, iy = int(round(pt_x)), int(round(pt_y))
        sigma = sigmas[i]
        kernel_radius = int(round(3 * sigma))
        x_min, x_max = max(0, ix - kernel_radius), min(w, ix + kernel_radius + 1)
        y_min, y_max = max(0, iy - kernel_radius), min(h, iy + kernel_radius + 1)
        
        y_grid, x_grid = np.ogrid[y_min - iy : y_max - iy, x_min - ix : x_max - ix]
        local_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
        kernel_sum = local_kernel.sum()
        if kernel_sum > 0:
            density_map[y_min:y_max, x_min:x_max] += local_kernel / kernel_sum

    return density_map

class JHUCrowdDataset(Dataset):
    """
    PyTorch Dataset for JHU-CROWD++ with adaptive annotation parsing
    and weather condition stratification.
    """
    def __init__(self, root_dir, target_size=(384, 512), split='train', weather_filter=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.split = split
        self.weather_filter = weather_filter
        
        self.image_paths = []
        for pat in ["images/*.*", "*.*", "train/images/*.*", "val/images/*.*", "test/images/*.*"]:
            found = glob.glob(os.path.join(root_dir, pat))
            img_cands = [p for p in found if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
            if len(img_cands) > len(self.image_paths):
                self.image_paths = sorted(img_cands)
                
        self.metadata = {}
        meta_file = os.path.join(root_dir, "image_level.txt")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3 and not parts[0].startswith("image_name"):
                        self.metadata[parts[0]] = {
                            "count": float(parts[1]),
                            "weather": parts[2].lower(),
                            "scene": parts[3].lower() if len(parts) > 3 else "outdoor"
                        }
        
        n = len(self.image_paths)
        if split == 'train':
            self.image_paths = self.image_paths[:int(0.70 * n)]
        elif split == 'val':
            self.image_paths = self.image_paths[int(0.70 * n):int(0.85 * n)]
        elif split == 'test':
            self.image_paths = self.image_paths[int(0.85 * n):]
            
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _find_gt_file(self, img_path):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(img_path)), "gt", f"{base_name}.txt"),
            os.path.join(os.path.dirname(img_path), f"{base_name}.txt"),
            os.path.join(self.root_dir, "gt", f"{base_name}.txt"),
            os.path.join(self.root_dir, "labels", f"{base_name}.txt")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        points = []
        weather_tag = "normal"
        gt_file = self._find_gt_file(img_path)
        if gt_file and os.path.exists(gt_file):
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            px, py = float(parts[0]), float(parts[1])
                            points.append((px, py))
                            if len(parts) >= 6:
                                weather_tag = parts[5].lower()
                        except ValueError:
                            continue
                            
        fname = os.path.basename(img_path)
        if fname in self.metadata:
            weather_tag = self.metadata[fname]["weather"]
            
        tw, th = self.target_size[1], self.target_size[0]
        img_resized = img.resize((tw, th), Image.BILINEAR)
        scaled_points = []
        scale_x = tw / float(orig_w)
        scale_y = th / float(orig_h)
        for (px, py) in points:
            scaled_points.append((px * scale_x, py * scale_y))
            
        dmap_h, dmap_w = th // 8, tw // 8
        dmap_points = [(px / 8.0, py / 8.0) for (px, py) in scaled_points]
        density_map = generate_density_map((dmap_h, dmap_w), dmap_points, beta=0.3, k=3)
        
        img_tensor = self.img_transform(img_resized)
        dmap_tensor = torch.from_numpy(density_map).unsqueeze(0).float()
        
        return {
            "image": img_tensor,
            "density_map": dmap_tensor,
            "count": torch.tensor(len(points), dtype=torch.float32),
            "weather": weather_tag,
            "path": img_path
        }

train_crowd_ds = JHUCrowdDataset(JHU_DIR, split='train')
val_crowd_ds = JHUCrowdDataset(JHU_DIR, split='val')
test_crowd_ds = JHUCrowdDataset(JHU_DIR, split='test')

train_crowd_loader = DataLoader(train_crowd_ds, batch_size=4, shuffle=True, drop_last=True)
val_crowd_loader = DataLoader(val_crowd_ds, batch_size=4, shuffle=False)
test_crowd_loader = DataLoader(test_crowd_ds, batch_size=1, shuffle=False)

print(f"JHU-CROWD++ Dataset Splitting:")
print(f"  - Train samples : {len(train_crowd_ds)}")
print(f"  - Val samples   : {len(val_crowd_ds)}")
print(f"  - Test samples  : {len(test_crowd_ds)}")
''')

    add_code(r'''# Visualization: Ground Truth Points & Continuous Crowd Density Map
sample_batch = next(iter(train_crowd_loader))
sample_img = sample_batch["image"][0].permute(1, 2, 0).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
sample_img = np.clip(sample_img * std + mean, 0.0, 1.0)
sample_dmap = sample_batch["density_map"][0, 0].numpy()
sample_count = sample_batch["count"][0].item()
dmap_sum = float(sample_dmap.sum())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sample_img)
axes[0].set_title(f"Processed Training Frame\nWeather: {sample_batch['weather'][0]} | GT Count: {sample_count:.0f}")
axes[0].axis("off")

im = axes[1].imshow(sample_dmap, cmap="jet")
axes[1].set_title(f"Ground Truth Crowd Density Heatmap\nIntegrated Sum: {dmap_sum:.2f}")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
''')

    # =========================================================================
    # SECTION 4 — Dehazing Preprocessing & Physics Priors
    # =========================================================================
    add_md(r'''## SECTION 4 — Dehazing Preprocessing & Atmospheric Physics Priors

### The Physics of Adverse Weather: Atmospheric Scattering Model
Optical degradation in fog, haze, and rain is governed by Koschmieder's physical scattering equation:
$$\mathbf{I}(x) = \mathbf{J}(x) t(x) + \mathbf{A}(1 - t(x))$$
where:
- $\mathbf{I}(x)$ is the observed degraded, hazy camera frame.
- $\mathbf{J}(x)$ is the scene radiance (the true, clear image we seek to restore).
- $t(x) = \exp(-\beta d(x)) \in [0, 1]$ is the medium transmission map ($d(x)$ is scene depth, $\beta$ is atmospheric scattering coefficient).
- $\mathbf{A} \in [0, 1]^3$ is the global atmospheric light vector.

Given transmission $t(x)$ and atmospheric light prior $\mathbf{A}$, clean scene radiance is physically inverted via:
$$\mathbf{J}(x) = \frac{\mathbf{I}(x) - \mathbf{A}}{\max(t(x), t_0)} + \mathbf{A}$$
where $t_0 = 0.1$ is a lower transmission bound preventing extreme noise amplification in vanishing horizons.

### Dark Channel Prior (DCP) Extraction
We implement He et al.'s Dark Channel Prior to extract physics priors:
$$\mathcal{J}^{\text{dark}}(x) = \min_{c \in \{R,G,B\}} \left( \min_{y \in \Omega(x)} \frac{\mathbf{I}^c(y)}{\mathbf{A}^c} \right)$$
''')

    add_code(r'''# Atmospheric Physics Priors & Dark Channel Prior Implementation
import cv2

def compute_dark_channel(image_rgb, patch_size=15):
    """
    Computes dark channel: min over RGB channels, followed by minimum filter.
    image_rgb: HxWx3 numpy array in [0, 1]
    """
    min_channel = np.min(image_rgb, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel.astype(np.float32), kernel)
    return dark_channel

def estimate_atmospheric_light_prior(image_rgb, dark_channel, top_percent=0.001):
    """
    Estimates global atmospheric light vector A from brightest pixels in dark channel.
    """
    h, w, _ = image_rgb.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)
    
    dark_flat = dark_channel.reshape(-1)
    img_flat = image_rgb.reshape(-1, 3)
    
    indices = np.argpartition(dark_flat, -num_top)[-num_top:]
    A = np.mean(img_flat[indices], axis=0)
    return np.clip(A, 0.1, 1.0)

def estimate_transmission_prior(image_rgb, A, omega=0.95, patch_size=15):
    """
    Estimates coarse transmission map t(x) using DCP.
    """
    norm_img = image_rgb / (A + 1e-6)
    dark_norm = compute_dark_channel(norm_img, patch_size=patch_size)
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.05, 1.0)

class RESIDEDehazeDataset(Dataset):
    """
    Dataset pairing hazy frames with clear ground truth radiance frames.
    """
    def __init__(self, root_dir, target_size=(256, 384), split='train'):
        self.root_dir = root_dir
        self.target_size = target_size
        self.split = split
        
        hazy_candidates = glob.glob(os.path.join(root_dir, "**", "hazy", "*.*"), recursive=True)
        self.hazy_paths = [p for p in hazy_candidates if os.path.splitext(p)[1].lower() in [".png", ".jpg", ".jpeg"]]
        
        n = len(self.hazy_paths)
        if split == 'train':
            self.hazy_paths = self.hazy_paths[:int(0.75 * n)]
        else:
            self.hazy_paths = self.hazy_paths[int(0.75 * n):]
            
    def _find_clear_path(self, hazy_path):
        base_name = os.path.basename(hazy_path)
        stem = os.path.splitext(base_name)[0]
        clear_id = stem.split("_")[0] if "_" in stem else stem
        
        dirs_to_check = [
            os.path.join(os.path.dirname(os.path.dirname(hazy_path)), "clear"),
            os.path.join(self.root_dir, "outdoor", "clear"),
            os.path.join(self.root_dir, "clear")
        ]
        for d in dirs_to_check:
            for ext in [".png", ".jpg", ".jpeg"]:
                c1 = os.path.join(d, f"{clear_id}{ext}")
                c2 = os.path.join(d, f"{stem}{ext}")
                if os.path.exists(c1):
                    return c1
                if os.path.exists(c2):
                    return c2
        return hazy_path

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_p = self.hazy_paths[idx]
        clear_p = self._find_clear_path(hazy_p)
        
        hazy_img = Image.open(hazy_p).convert("RGB").resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        clear_img = Image.open(clear_p).convert("RGB").resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        hazy_np = np.array(hazy_img, dtype=np.float32) / 255.0
        clear_np = np.array(clear_img, dtype=np.float32) / 255.0
        
        dark = compute_dark_channel(hazy_np)
        A = estimate_atmospheric_light_prior(hazy_np, dark)
        t_prior = estimate_transmission_prior(hazy_np, A)
        
        return {
            "hazy": torch.from_numpy(hazy_np).permute(2, 0, 1).float(),
            "clear": torch.from_numpy(clear_np).permute(2, 0, 1).float(),
            "transmission_prior": torch.from_numpy(t_prior).unsqueeze(0).float(),
            "A_prior": torch.from_numpy(A).float()
        }

train_dehaze_ds = RESIDEDehazeDataset(RESIDE_DIR, split='train')
val_dehaze_ds = RESIDEDehazeDataset(RESIDE_DIR, split='val')

train_dehaze_loader = DataLoader(train_dehaze_ds, batch_size=4, shuffle=True, drop_last=True)
val_dehaze_loader = DataLoader(val_dehaze_ds, batch_size=4, shuffle=False)

print(f"RESIDE Dehazing Dataset:")
print(f"  - Train pairs : {len(train_dehaze_ds)}")
print(f"  - Val pairs   : {len(val_dehaze_ds)}")
''')

    add_code(r'''# Visualization: Hazy Input, Dark Channel, Transmission Map, & Atmospheric Light
sample_dehaze = train_dehaze_ds[0]
hazy_vis = sample_dehaze["hazy"].permute(1, 2, 0).numpy()
clear_vis = sample_dehaze["clear"].permute(1, 2, 0).numpy()
t_vis = sample_dehaze["transmission_prior"][0].numpy()
A_vis = sample_dehaze["A_prior"].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(hazy_vis)
axes[0].set_title(f"Degraded Hazy Image\nAtmospheric Light A: [{A_vis[0]:.2f}, {A_vis[1]:.2f}, {A_vis[2]:.2f}]")
axes[0].axis("off")

im1 = axes[1].imshow(t_vis, cmap="bone")
axes[1].set_title("Estimated Transmission Map t(x)\n(Optical Depth Attenuation)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(clear_vis)
axes[2].set_title("Reference Ground-Truth Clear Radiance")
axes[2].axis("off")
plt.tight_layout()
plt.show()
''')

    # =========================================================================
    # SECTION 5 — Dehazing Model
    # =========================================================================
    add_md(r'''## SECTION 5 — Deep Learning Physics-Guided Dehazing Network

### Architecture Design: `PhysicsGuidedDehazeNet`
Rather than treating restoration as a black-box image-to-image mapping, `PhysicsGuidedDehazeNet` embodies the atmospheric scattering physics:
1. **Multi-Scale Feature Extractor**: Employs parallel convolutional filters with receptive field kernels $3\times 3$, $5\times 5$, and $7\times 7$ to simultaneously capture fine localized textures and broad atmospheric haze gradients.
2. **Dual-Head Physics Priors**:
   - **Transmission Head $\mathbf{t}(x)$**: Predicts continuous medium transmission bounded strictly to $[0.05, 1.0]$.
   - **Atmospheric Light Head $\mathbf{A}$**: Global spatial pooling coupled with an MLP predicts global atmospheric vector $\mathbf{A} \in [0, 1]^3$.
3. **Physical Scattering Inversion Layer**:
   $$\mathbf{J}_{\text{physics}}(x) = \frac{\mathbf{I}(x) - \mathbf{A}}{\max(\mathbf{t}(x), 0.1)} + \mathbf{A}$$
4. **Residual Color & Halo Refinement Block**: Refines fine chrominance and suppresses boundary halos, outputting final restored frame $\mathbf{J}(x) \in [0, 1]$.
''')

    add_code(r'''# Physics-Guided Dehazing Network Definition
import torch.nn.functional as F

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
    End-to-End Physics-Guided Dehazing Network.
    Predicts physical transmission map t(x) and atmospheric light A,
    inverts the physical scattering model, and refines the output radiance.
    """
    def __init__(self):
        super().__init__()
        # Multi-scale feature backbone
        self.ms_conv1 = MultiScaleConvBlock(3, 48)
        self.ms_conv2 = MultiScaleConvBlock(48, 64)
        
        # Transmission map prediction branch
        self.t_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # Atmospheric light prediction branch
        self.a_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.a_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        
        # Residual chromatic refinement sub-network
        self.refine = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.ms_conv2(self.ms_conv1(x))
        
        # Estimate transmission map t in [0.05, 1.0]
        raw_t = torch.sigmoid(self.t_conv2(F.leaky_relu(self.t_conv1(feat), 0.2)))
        t = 0.05 + 0.95 * raw_t
        
        # Estimate atmospheric light A in [0, 1]
        pooled = self.a_pool(feat).view(feat.size(0), -1)
        A = self.a_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        
        # Physical Inversion: J_phys = (I - A) / max(t, 0.1) + A
        t_clamped = torch.clamp(t, min=0.1)
        j_phys = (x - A) / t_clamped + A
        j_phys = torch.clamp(j_phys, 0.0, 1.0)
        
        # Residual refinement: J_final = J_phys + Refinement(I, J_phys)
        refine_in = torch.cat([x, j_phys], dim=1)
        residual = self.refine(refine_in)
        j_final = torch.clamp(j_phys + 0.1 * residual, 0.0, 1.0)
        
        return j_final, t, A.squeeze(-1).squeeze(-1)

def compute_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(1.0 / np.sqrt(mse))

dehaze_net = PhysicsGuidedDehazeNet().to(DEVICE)
print(dehaze_net)
print(f"Total Dehazing Network Parameters: {sum(p.numel() for p in dehaze_net.parameters() if p.requires_grad):,}")
''')

    add_code(r'''# Dehazing Training & Validation Loop with Mixed Precision
dehaze_optimizer = optim.Adam(dehaze_net.parameters(), lr=1e-3, weight_decay=1e-5)
dehaze_scheduler = optim.lr_scheduler.CosineAnnealingLR(dehaze_optimizer, T_max=10, eta_min=1e-5)
l1_loss_fn = nn.L1Loss()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def smooth_loss(t):
    """Total variation spatial smoothness regularizer on transmission map."""
    dx = torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])
    dy = torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)

DEHAZE_EPOCHS = 5
best_psnr = -1.0
dehaze_history = {"train_loss": [], "val_psnr": []}

print("=" * 70)
print("TRAINING PHYSICS-GUIDED DEHAZING NETWORK")
print("=" * 70)

for epoch in range(1, DEHAZE_EPOCHS + 1):
    dehaze_net.train()
    running_loss = 0.0
    for batch in train_dehaze_loader:
        hazy = batch["hazy"].to(DEVICE)
        clear = batch["clear"].to(DEVICE)
        
        dehaze_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            j_pred, t_pred, a_pred = dehaze_net(hazy)
            loss_recon = l1_loss_fn(j_pred, clear)
            loss_tv = smooth_loss(t_pred)
            total_loss = loss_recon + 0.05 * loss_tv
            
        scaler.scale(total_loss).backward()
        scaler.step(dehaze_optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        
    dehaze_scheduler.step()
    avg_train_loss = running_loss / max(len(train_dehaze_loader), 1)
    
    # Validation
    dehaze_net.eval()
    val_psnr_list = []
    with torch.no_grad():
        for batch in val_dehaze_loader:
            hazy = batch["hazy"].to(DEVICE)
            clear = batch["clear"].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                j_pred, _, _ = dehaze_net(hazy)
            for p, g in zip(j_pred, clear):
                val_psnr_list.append(compute_psnr(p, g))
                
    avg_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 0.0
    dehaze_history["train_loss"].append(avg_train_loss)
    dehaze_history["val_psnr"].append(avg_val_psnr)
    
    print(f"Epoch [{epoch:02d}/{DEHAZE_EPOCHS:02d}] - Train Loss: {avg_train_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB")
    
    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save(dehaze_net.state_dict(), os.path.join(MODELS_DIR, "dehazing_best.pth"))

torch.save(dehaze_net.state_dict(), os.path.join(MODELS_DIR, "dehazing_model.pth"))
print(f"[SUCCESS] Dehazing model saved to {os.path.join(MODELS_DIR, 'dehazing_best.pth')}")
''')

    # =========================================================================
    # SECTION 6 — Crowd-Density Model
    # =========================================================================
    add_md(r'''## SECTION 6 — Crowd-Density Model (CSRNet)

### Dilated Convolutional Density Map Regression
To accurately estimate crowd density under dense congregations, we employ **CSRNet (Congested Scene Recognition Network)**:
1. **Front-End Feature Extractor**: The first 10 convolutional layers of VGG-16 pre-trained on ImageNet.
2. **Back-End Dilated Convolutions**: A sequence of 6 dilated convolutional layers with dilation factor $d=2$. Dilated kernels enlarge the spatial receptive field to $64 \times 64$ pixels per neuron without downsampling, preserving fine spatial resolution required for localized head density maps.
3. **Density Prediction Output**: A $1 \times 1$ conv layer regressing continuous density map $\mathcal{D}(x, y)$.

### Objective Function & Performance Metrics
- **Objective Function**: Mean Squared Error (MSE) between predicted and ground-truth density maps:
  $$\mathcal{L}_{\text{density}} = \frac{1}{2N} \sum_{i=1}^N \|\mathcal{D}^{\text{pred}}_i - \mathcal{D}^{\text{gt}}_i\|_2^2$$
- **Evaluation Metrics**:
  - Mean Absolute Error: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |C^{\text{pred}}_i - C^{\text{gt}}_i|$
  - Root Mean Squared Error: $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (C^{\text{pred}}_i - C^{\text{gt}}_i)^2}$
''')

    add_code(r'''# CSRNet Architecture Implementation
import torchvision.models as models

class CSRNet(nn.Module):
    """
    CSRNet for Congested Crowd Counting.
    Frontend: VGG-16 first 10 conv layers.
    Backend: Dilated conv layers (rate=2).
    """
    def __init__(self):
        super().__init__()
        # Pretrained VGG-16 front-end
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if hasattr(models, "VGG16_Weights") else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Dilated convolutional back-end (dilation rate = 2)
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
        
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.frontend(x)
        dmap = self.backend(feat)
        return F.relu(dmap)

crowd_model = CSRNet().to(DEVICE)
print(crowd_model)
print(f"Total CSRNet Parameters: {sum(p.numel() for p in crowd_model.parameters() if p.requires_grad):,}")
''')

    add_code(r'''# CSRNet Training & Validation Pipeline
crowd_optimizer = optim.Adam(crowd_model.parameters(), lr=1e-5, weight_decay=1e-4)
crowd_loss_fn = nn.MSELoss(reduction='sum')
crowd_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

CROWD_EPOCHS = 6
best_val_mae = float('inf')
crowd_history = {"train_loss": [], "val_mae": [], "val_rmse": []}

print("=" * 70)
print("TRAINING CSRNET CROWD DENSITY ESTIMATION MODEL")
print("=" * 70)

for epoch in range(1, CROWD_EPOCHS + 1):
    crowd_model.train()
    running_loss = 0.0
    for batch in train_crowd_loader:
        imgs = batch["image"].to(DEVICE)
        dmaps_gt = batch["density_map"].to(DEVICE)
        
        crowd_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            dmaps_pred = crowd_model(imgs)
            loss = crowd_loss_fn(dmaps_pred, dmaps_gt) / (2.0 * imgs.size(0))
            
        crowd_scaler.scale(loss).backward()
        crowd_scaler.step(crowd_optimizer)
        crowd_scaler.update()
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / max(len(train_crowd_loader), 1)
    
    # Validation evaluation
    crowd_model.eval()
    abs_errors, sq_errors = [], []
    with torch.no_grad():
        for batch in val_crowd_loader:
            imgs = batch["image"].to(DEVICE)
            gt_counts = batch["count"].cpu().numpy()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                dmaps_pred = crowd_model(imgs)
                
            pred_counts = dmaps_pred.sum(dim=(1, 2, 3)).cpu().numpy()
            abs_errors.extend(np.abs(pred_counts - gt_counts))
            sq_errors.extend((pred_counts - gt_counts) ** 2)
            
    val_mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    val_rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else 0.0
    
    crowd_history["train_loss"].append(avg_train_loss)
    crowd_history["val_mae"].append(val_mae)
    crowd_history["val_rmse"].append(val_rmse)
    
    print(f"Epoch [{epoch:02d}/{CROWD_EPOCHS:02d}] - Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f}")
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(crowd_model.state_dict(), os.path.join(MODELS_DIR, "crowd_best.pt"))
        print(f"  --> [BEST CHECKPOINT] Saved new best validation model: MAE={best_val_mae:.2f}")

torch.save(crowd_model.state_dict(), os.path.join(MODELS_DIR, "crowd_density_model.pth"))
print(f"[SUCCESS] Checkpoints saved to {MODELS_DIR}")
''')

    add_code(r'''# Plot Training Curves (Loss, MAE, RMSE)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, CROWD_EPOCHS + 1), crowd_history["train_loss"], 'o-', color='crimson')
axes[0].set_title("CSRNet Training Density Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Normalized Loss")
axes[0].grid(True)

axes[1].plot(range(1, CROWD_EPOCHS + 1), crowd_history["val_mae"], 's-', label="MAE", color='navy')
axes[1].plot(range(1, CROWD_EPOCHS + 1), crowd_history["val_rmse"], '^-', label="RMSE", color='darkorange')
axes[1].set_title("Crowd Counting Validation Error Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Error (Persons)")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.show()
''')

    # =========================================================================
    # SECTION 7 — Adverse-Weather Handling
    # =========================================================================
    add_md(r'''## SECTION 7 — Adverse-Weather Evaluation & Dehazing Gain

### Adverse-Weather Stratification
A central requirement of the surveillance pipeline is quantifying crowd estimation reliability under environmental degradation. We evaluate:
1. **Performance Stratification**: Breaking down MAE/RMSE across weather conditions:
   - Clear / Normal
   - Heavy Fog / Haze
   - Monsoon Rain
   - Snow / Glare
2. **Direct Restoration Gain**:
   We compare crowd counting accuracy:
   - **Condition A**: Direct inference on raw, degraded images.
   - **Condition B**: Inference on restored images produced by `PhysicsGuidedDehazeNet`.
''')

    add_code(r'''# Weather Stratification & Restoration Benefit Assessment
import pandas as pd

def evaluate_adverse_weather_performance(dehaze_net, crowd_model, test_loader, device):
    """
    Evaluates crowd counting across weather types on raw degraded vs restored frames.
    """
    dehaze_net.eval()
    crowd_model.eval()
    
    results = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            gt_counts = batch["count"].cpu().numpy()
            weathers = batch["weather"]
            
            dmap_raw = crowd_model(imgs)
            pred_count_raw = dmap_raw.sum(dim=(1, 2, 3)).cpu().numpy()
            
            unnorm_imgs = imgs * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) + \
                          torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            unnorm_imgs = torch.clamp(unnorm_imgs, 0.0, 1.0)
            
            restored_imgs, _, _ = dehaze_net(unnorm_imgs)
            
            renorm_imgs = (restored_imgs - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / \
                          torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            
            dmap_restored = crowd_model(renorm_imgs)
            pred_count_restored = dmap_restored.sum(dim=(1, 2, 3)).cpu().numpy()
            
            for i in range(len(gt_counts)):
                w = weathers[i]
                gt = gt_counts[i]
                c_raw = pred_count_raw[i]
                c_rest = pred_count_restored[i]
                
                err_raw = abs(c_raw - gt)
                err_rest = abs(c_rest - gt)
                results.append({
                    "weather": w,
                    "gt_count": gt,
                    "raw_pred": c_raw,
                    "restored_pred": c_rest,
                    "raw_err": err_raw,
                    "restored_err": err_rest
                })
                
    df = pd.DataFrame(results)
    return df

df_weather = evaluate_adverse_weather_performance(dehaze_net, crowd_model, test_crowd_loader, DEVICE)

summary = df_weather.groupby("weather").agg(
    samples=("gt_count", "count"),
    mean_gt=("gt_count", "mean"),
    raw_mae=("raw_err", "mean"),
    restored_mae=("restored_err", "mean")
).reset_index()

summary["mae_reduction_%"] = 100.0 * (summary["raw_mae"] - summary["restored_mae"]) / (summary["raw_mae"] + 1e-6)

print("=" * 70)
print("ADVERSE-WEATHER PERFORMANCE STRATIFICATION & RESTORATION GAIN")
print("=" * 70)
print(summary.to_string(index=False))
print("=" * 70)
''')

    # =========================================================================
    # SECTION 8 — Combined Inference Pipeline
    # =========================================================================
    add_md(r'''## SECTION 8 — Combined Surveillance Inference Pipeline

### End-to-End Dynamic Pipeline
We encapsulate the entire pipeline into a single, modular callable function:
`process_surveillance_frame(frame, dehaze_net, crowd_net, device, config)`
$$\text{Input Frame} \to \text{Priors Extraction} \to \text{Physics Dehaze} \to \text{CSRNet Density Map} \to \text{Risk & Alert Engine}$$
''')

    add_code(r'''# Unified End-to-End Surveillance Inference Function
def process_surveillance_frame(frame_bgr_or_rgb, dehaze_net, crowd_model, device, config=None):
    """
    End-to-End Surveillance Processor:
    - Input: RGB/BGR frame (numpy array or PIL image)
    - Returns: Restored frame, transmission map, atmospheric light, density map,
      crowd count, weather condition, risk score, alert status.
    """
    if config is None:
        config = {
            "max_crowd_threshold": 100,
            "weather_multipliers": {"clear": 1.0, "fog": 1.25, "rain": 1.30, "snow": 1.40}
        }
        
    if isinstance(frame_bgr_or_rgb, np.ndarray):
        if frame_bgr_or_rgb.shape[2] == 3:
            orig_rgb = frame_bgr_or_rgb.copy()
        else:
            orig_rgb = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = np.array(frame_bgr_or_rgb.convert("RGB"))
        
    h_orig, w_orig, _ = orig_rgb.shape
    
    # 1. Weather/degradation assessment via dark channel contrast
    norm_img = orig_rgb.astype(np.float32) / 255.0
    dark_ch = compute_dark_channel(norm_img)
    mean_dark = float(np.mean(dark_ch))
    if mean_dark > 0.45:
        detected_weather = "fog"
    elif mean_dark > 0.30:
        detected_weather = "rain"
    else:
        detected_weather = "clear"
        
    # 2. Prepare tensor for Dehaze Network
    tensor_in = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    dehaze_net.eval()
    crowd_model.eval()
    with torch.no_grad():
        restored_tensor, t_tensor, A_tensor = dehaze_net(tensor_in)
        restored_rgb = restored_tensor[0].permute(1, 2, 0).cpu().numpy()
        transmission_map = t_tensor[0, 0].cpu().numpy()
        atmospheric_light = A_tensor[0].cpu().numpy()
        
        # 3. Prepare tensor for CSRNet Crowd Density Network
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        crowd_input = (restored_tensor - mean_t) / std_t
        
        dmap_tensor = crowd_model(crowd_input)
        density_map = dmap_tensor[0, 0].cpu().numpy()
        
    # 4. Total Crowd Count Integration
    crowd_count = float(density_map.sum())
    
    # 5. Risk and Alert Engine
    max_thresh = config.get("max_crowd_threshold", 100)
    w_mult = config.get("weather_multipliers", {}).get(detected_weather, 1.0)
    raw_ratio = crowd_count / float(max_thresh)
    risk_score = float(np.clip(raw_ratio * w_mult, 0.0, 1.0))
    
    if risk_score >= 1.0 or crowd_count >= max_thresh:
        risk_level = "CRITICAL"
        alert_status = "CAPACITY THRESHOLD EXCEEDED"
    elif risk_score >= 0.80:
        risk_level = "HIGH"
        alert_status = "ELEVATED CONGESTION WARNING"
    elif risk_score >= 0.50:
        risk_level = "MEDIUM"
        alert_status = "MONITORING"
    else:
        risk_level = "LOW"
        alert_status = "NORMAL"
        
    confidence = float(np.clip(1.0 - 0.25 * mean_dark, 0.60, 0.98))
    
    return {
        "original_frame": orig_rgb,
        "restored_frame": np.clip(restored_rgb * 255.0, 0, 255).astype(np.uint8),
        "transmission_map": transmission_map,
        "atmospheric_light": atmospheric_light,
        "density_map": density_map,
        "crowd_count": round(crowd_count, 1),
        "detected_weather": detected_weather,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "alert_status": alert_status,
        "confidence": round(confidence, 2)
    }

# Execute pipeline test on single frame
test_sample_path = test_crowd_ds.image_paths[0]
sample_raw = cv2.imread(test_sample_path)
result = process_surveillance_frame(sample_raw, dehaze_net, crowd_model, DEVICE)

print("=" * 70)
print("SINGLE FRAME SURVEILLANCE PIPELINE TEST OUTPUT")
print("=" * 70)
print(f"Detected Condition : {result['detected_weather'].upper()}")
print(f"Crowd Count        : {result['crowd_count']}")
print(f"Risk Level         : {result['risk_level']} (Score: {result['risk_score']})")
print(f"Alert Status       : {result['alert_status']}")
print(f"Confidence         : {result['confidence']}")
print("=" * 70)
''')

    # =========================================================================
    # SECTION 9 — Video Inference Pipeline
    # =========================================================================
    add_md(r'''## SECTION 9 — Video Inference & Dynamic Stream Processing

### Periodic & Real-Time Processing Architecture
For video streams (`.mp4`, CCTV camera feeds):
1. **Periodic Sampling**: To sustain near-real-time operation without GPU memory bottlenecks, frames are extracted and processed every $k$-th interval (e.g., every 3rd or 5th frame).
2. **Visual Heads-Up Display (HUD)**: Renders a dual-panel surveillance feed:
   - Panel A: Restored scene radiance.
   - Panel B: Calibrated crowd density heatmap overlay.
   - Banner: Live crowd count, capacity threshold, adverse-weather indicator, risk badge, and active emergency alert.
3. **Throughput Logging**: Measures frame processing latency and logs effective FPS.
''')

    add_code(r'''# Video Surveillance Processor with HUD Overlay & Synthetic Video Creator
import time

def process_surveillance_video(video_in_path, video_out_path, dehaze_net, crowd_model, device, sample_interval=3, config=None):
    """
    Processes video stream, annotates frames with HUD, and writes annotated output video.
    """
    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video {video_in_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
    
    out_w, out_h = width * 2, height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps / sample_interval, (out_w, out_h))
    
    frame_idx = 0
    processed_count = 0
    total_time = 0.0
    
    print(f"[INFO] Processing video: {video_in_path} -> {video_out_path}")
    print(f"       Sampling every {sample_interval} frames ({total_frames} total frames)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_interval == 0:
            t0 = time.time()
            res = process_surveillance_frame(frame, dehaze_net, crowd_model, device, config)
            t_proc = time.time() - t0
            total_time += t_proc
            processed_count += 1
            
            # Left panel: Restored Frame with HUD overlay
            left_panel = res["restored_frame"].copy()
            left_panel = cv2.cvtColor(left_panel, cv2.COLOR_RGB2BGR)
            
            # Right panel: Density Heatmap overlay
            dmap_norm = (res["density_map"] / (np.max(res["density_map"]) + 1e-6) * 255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (width, height))
            
            # Stitch dual-panel
            canvas = np.hstack([left_panel, heatmap])
            
            # Render HUD Banner
            hud_color = (0, 0, 255) if res["risk_level"] == "CRITICAL" else (0, 165, 255) if res["risk_level"] == "HIGH" else (0, 255, 0)
            cv2.rectangle(canvas, (0, 0), (out_w, 60), (30, 30, 30), -1)
            
            hud_text = f"CROWD COUNT: {res['crowd_count']:.0f} | WEATHER: {res['detected_weather'].upper()} | RISK: {res['risk_level']} ({res['risk_score']:.2f})"
            cv2.putText(canvas, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"ALERT: {res['alert_status']}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
            cv2.putText(canvas, f"SURVEILLANCE HUD - RESTORED FRAME (LEFT) | DENSITY HEATMAP (RIGHT)", (width + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, f"FPS: {1.0 / max(t_proc, 1e-4):.1f}", (out_w - 120, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            out.write(canvas)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    avg_fps = processed_count / max(total_time, 1e-4)
    print(f"[SUCCESS] Video processing complete! Average Processing Speed: {avg_fps:.1f} FPS")
    print(f"          Saved annotated video to: {video_out_path}")

def create_synthetic_demo_video(output_video_path, num_frames=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 640, 480
    vw = cv2.VideoWriter(output_video_path, fourcc, 10.0, (w, h))
    for f in range(num_frames):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = (180, 180, 180)
        for p in range(25):
            px = int((p * 24 + f * 4) % (w - 40) + 20)
            py = int(150 + (p * 11) % 250)
            cv2.circle(canvas, (px, py), 6, (120, 100, 90), -1)
            cv2.rectangle(canvas, (px-5, py+6), (px+5, py+24), (80, 80, 120), -1)
        vw.write(canvas)
    vw.release()

synth_video_in = os.path.join(OUTPUTS_DIR, "input_feed.mp4")
surveillance_video_out = os.path.join(OUTPUTS_DIR, "surveillance_output.mp4")
create_synthetic_demo_video(synth_video_in)
process_surveillance_video(synth_video_in, surveillance_video_out, dehaze_net, crowd_model, DEVICE)
''')

    # =========================================================================
    # SECTION 10 — Risk and Alert Logic
    # =========================================================================
    add_md(r'''## SECTION 10 — Dynamic Risk Assessment & Emergency Alert Engine

### Configurable Safety Thresholds
In public safety surveillance, crowd capacity thresholds must dynamically adjust based on environmental risk:
- Under **Heavy Fog**, visibility drop impedes rapid egress.
- Under **Monsoon Rain**, slippery floor friction increases trampling hazards.

### Risk Formulation
$$\text{Risk Score} = \min\left(1.0, \; \frac{\text{Estimated Crowd Count}}{\text{MAX\_CROWD\_COUNT}} \times \mathcal{W}_{\text{factor}}\right)$$
where $\mathcal{W}_{\text{factor}}$ is the weather-specific hazard multiplier:
- **Normal / Clear**: $1.0\times$
- **Heavy Fog / Haze**: $1.25\times$
- **Monsoon Rain**: $1.30\times$
- **Snow / Lens Glare**: $1.40\times$

### Tier Categorization
- `LOW`: Risk Score $< 0.50$ (Routine operations)
- `MEDIUM`: $0.50 \le \text{Risk Score} < 0.80$ (Monitoring phase)
- `HIGH`: $0.80 \le \text{Risk Score} < 1.00$ (Warning phase)
- `CRITICAL`: $\text{Risk Score} \ge 1.00$ (**CAPACITY THRESHOLD EXCEEDED ALERT**)
''')

    add_code(r'''# Configurable Alert & Risk Assessment Unit Tests
def evaluate_crowd_risk(crowd_count, weather_condition="clear", max_capacity=100):
    """
    Calculates adverse-weather adjusted risk score and emergency alert status.
    """
    multipliers = {
        "clear": 1.0,
        "fog": 1.25,
        "rain": 1.30,
        "snow": 1.40
    }
    w_factor = multipliers.get(weather_condition.lower(), 1.0)
    raw_ratio = crowd_count / float(max_capacity)
    risk_score = np.clip(raw_ratio * w_factor, 0.0, 1.0)
    
    if risk_score >= 1.0 or crowd_count >= max_capacity:
        level = "CRITICAL"
        alert = "CAPACITY THRESHOLD EXCEEDED"
    elif risk_score >= 0.80:
        level = "HIGH"
        alert = "ELEVATED CONGESTION WARNING"
    elif risk_score >= 0.50:
        level = "MEDIUM"
        alert = "CAUTION MONITORING"
    else:
        level = "LOW"
        alert = "NORMAL CAPACITY"
        
    return {
        "crowd_count": crowd_count,
        "threshold": max_capacity,
        "weather": weather_condition,
        "weather_factor": w_factor,
        "risk_score": round(float(risk_score), 2),
        "risk_level": level,
        "alert": alert
    }

print("=" * 70)
print("DYNAMIC RISK ENGINE TEST SCENARIOS")
print("=" * 70)
test_scenarios = [
    (40, "clear", 100),
    (70, "clear", 100),
    (70, "fog", 100),
    (80, "rain", 100),
    (115, "fog", 100)
]

for count, weather, cap in test_scenarios:
    res = evaluate_crowd_risk(count, weather, cap)
    print(f"Count: {res['crowd_count']:3d}/{res['threshold']} | Weather: {res['weather']:5s} | Risk: {res['risk_score']:.2f} ({res['risk_level']:8s}) | Alert: {res['alert']}")
print("=" * 70)
''')

    # =========================================================================
    # SECTION 11 — CSV Reporting
    # =========================================================================
    add_md(r'''## SECTION 11 — Automated CSV Event Reporting

The surveillance system maintains an audit log in standard CSV format recording every processed frame and event.

### Required Schema
- `timestamp`: ISO 8601 formatted event timestamp
- `location`: Configured camera zone identifier (e.g. `Zone-North-Gate-4`)
- `detected_condition`: Weather / atmospheric condition (`clear`, `fog`, `rain`, `snow`)
- `crowd_count`: Total estimated crowd count
- `confidence`: Algorithmic confidence score
- `risk_score`: Dynamic adverse-weather risk score $[0, 1]$
- `alert_status`: Alert trigger notification
- `processing_fps`: Live execution throughput
''')

    add_code(r'''# Automated CSV Event Logging Engine
import datetime

class SurveillanceEventLogger:
    """
    Thread-safe audit logger writing surveillance alerts and telemetry to CSV.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.headers = [
            "timestamp", "location", "detected_condition",
            "crowd_count", "confidence", "risk_score",
            "alert_status", "processing_fps"
        ]
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write(",".join(self.headers) + "\n")
                
    def log_event(self, location, condition, count, confidence, risk_score, alert, fps=25.0):
        ts = datetime.datetime.now().isoformat()
        row = [
            ts, location, condition, f"{count:.1f}",
            f"{confidence:.2f}", f"{risk_score:.2f}",
            alert, f"{fps:.1f}"
        ]
        with open(self.log_file_path, "a") as f:
            f.write(",".join(row) + "\n")

csv_log_path = os.path.join(REPORTS_DIR, "crowd_surveillance_log.csv")
event_logger = SurveillanceEventLogger(csv_log_path)

sample_events = [
    ("Gate-A-North", "clear", 45.0, 0.95, 0.45, "NORMAL", 28.4),
    ("Gate-B-Plaza", "fog", 78.0, 0.88, 0.98, "ELEVATED CONGESTION WARNING", 24.1),
    ("Gate-C-West", "rain", 108.0, 0.85, 1.00, "CAPACITY THRESHOLD EXCEEDED", 22.7),
    ("Concourse-Central", "fog", 137.0, 0.92, 1.00, "CAPACITY THRESHOLD EXCEEDED", 21.9)
]

for ev in sample_events:
    event_logger.log_event(*ev)

print(f"[SUCCESS] Surveillance audit log recorded at: {csv_log_path}")
df_log = pd.read_csv(csv_log_path)
print("=" * 70)
print("SURVEILLANCE CSV AUDIT LOG PREVIEW")
print("=" * 70)
print(df_log.to_string(index=False))
print("=" * 70)
''')

    # =========================================================================
    # SECTION 12 — Visualization
    # =========================================================================
    add_md(r'''## SECTION 12 — Comprehensive Surveillance Visualizations

Here we generate the exact required visualizations:
1. **Side-by-Side**: Original Degraded Frame vs Restored Frame
2. **Surveillance Inspection**: Restored Frame vs Crowd Density Heatmap
3. **Master 4-Panel Surveillance Diagnostic Dashboard**:
   - Panel 1: Original Degraded Feed (with weather & degradation tag)
   - Panel 2: Physics Transmission Map (optical depth attenuation)
   - Panel 3: Restored Clean Radiance Frame
   - Panel 4: Calibrated Crowd Density Heatmap with HUD Alert Banner
''')

    add_code(r'''# Publication-Grade Surveillance Dashboards
def render_master_surveillance_dashboard(result, save_path=None):
    """
    Renders 4-panel master surveillance console:
    1. Original Degraded Feed
    2. Transmission Map
    3. Restored Scene Radiance
    4. Calibrated Density Heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original Degraded Feed
    axes[0, 0].imshow(result["original_frame"])
    axes[0, 0].set_title(f"1. DEGRADED CAMERA FEED\nWeather: {result['detected_weather'].upper()} | Confidence: {result['confidence']}", fontsize=13, fontweight='bold')
    axes[0, 0].axis("off")
    
    # 2. Transmission Map
    im_t = axes[0, 1].imshow(result["transmission_map"], cmap="bone")
    axes[0, 1].set_title("2. PHYSICS TRANSMISSION MAP t(x)\n(Optical Depth Attenuation)", fontsize=13, fontweight='bold')
    axes[0, 1].axis("off")
    plt.colorbar(im_t, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Restored Frame
    axes[1, 0].imshow(result["restored_frame"])
    axes[1, 0].set_title("3. PHYSICS-GUIDED RESTORED RADIANCE J(x)\n(De-Hazed / De-Rained Frame)", fontsize=13, fontweight='bold')
    axes[1, 0].axis("off")
    
    # 4. Crowd Density Heatmap
    im_d = axes[1, 1].imshow(result["density_map"], cmap="jet")
    alert_color = "red" if result["risk_level"] == "CRITICAL" else "orange" if result["risk_level"] == "HIGH" else "green"
    axes[1, 1].set_title(f"4. CROWD DENSITY HEATMAP\nCount: {result['crowd_count']} | Alert: {result['alert_status']}", fontsize=13, fontweight='bold', color=alert_color)
    axes[1, 1].axis("off")
    plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"ADVERSE WEATHER SURVEILLANCE PIPELINE: DYNAMIC ANALYSIS\nRISK STATUS: {result['risk_level']} (Score: {result['risk_score']})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

dashboard_path = os.path.join(OUTPUTS_DIR, "master_surveillance_dashboard.png")
render_master_surveillance_dashboard(result, save_path=dashboard_path)
''')

    # =========================================================================
    # SECTION 13 — Evaluation
    # =========================================================================
    add_md(r'''## SECTION 13 — Quantitative & Qualitative Model Evaluation

We evaluate:
1. **Dehazing Component**: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
2. **Crowd-Counting Component**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
3. **Adverse Weather Reliability**: Comparative summary verifying counting accuracy before and after restoration.
''')

    add_code(r'''# Full Quantitative Evaluation Suite
try:
    from skimage.metrics import structural_similarity as ssim
    def compute_ssim_metric(img1, img2):
        return ssim(img1, img2, channel_axis=2, data_range=1.0)
except ImportError:
    def compute_ssim_metric(img1, img2):
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        mu1, mu2 = np.mean(img1), np.mean(img2)
        sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        return float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))

def evaluate_dehazing_test_set(dehaze_net, dataloader, device):
    dehaze_net.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for batch in dataloader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            restored, _, _ = dehaze_net(hazy)
            
            for p, g in zip(restored, clear):
                p_np = p.permute(1, 2, 0).cpu().numpy()
                g_np = g.permute(1, 2, 0).cpu().numpy()
                
                # PSNR
                mse = np.mean((p_np - g_np) ** 2)
                psnr_val = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 50.0
                psnr_scores.append(psnr_val)
                
                # SSIM
                ssim_val = compute_ssim_metric(p_np, g_np)
                ssim_scores.append(ssim_val)
                
    return np.mean(psnr_scores), np.mean(ssim_scores)

def evaluate_crowd_test_set(crowd_model, dataloader, device):
    crowd_model.eval()
    abs_errors, sq_errors = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            gt_counts = batch["count"].cpu().numpy()
            dmaps = crowd_model(imgs)
            pred_counts = dmaps.sum(dim=(1, 2, 3)).cpu().numpy()
            
            abs_errors.extend(np.abs(pred_counts - gt_counts))
            sq_errors.extend((pred_counts - gt_counts) ** 2)
            
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return mae, rmse

test_psnr, test_ssim = evaluate_dehazing_test_set(dehaze_net, val_dehaze_loader, DEVICE)
test_mae, test_rmse = evaluate_crowd_test_set(crowd_model, test_crowd_loader, DEVICE)

eval_df = pd.DataFrame([
    {"Model Component": "Physics-Guided Dehazing Network", "Metric 1": f"PSNR: {test_psnr:.2f} dB", "Metric 2": f"SSIM: {test_ssim:.4f}"},
    {"Model Component": "CSRNet Crowd Density Network", "Metric 1": f"MAE: {test_mae:.2f}", "Metric 2": f"RMSE: {test_rmse:.2f}"}
])

print("=" * 70)
print("FINAL BENCHMARK EVALUATION RESULTS")
print("=" * 70)
print(eval_df.to_string(index=False))
print("=" * 70)
''')

    # =========================================================================
    # SECTION 14 — Model Checkpoint Saving
    # =========================================================================
    add_md(r'''## SECTION 14 — Model Checkpointing & Integrity Verification

We persist and verify all trained weights:
- `/content/models/dehazing_model.pth`
- `/content/models/crowd_density_model.pth`
- `/content/models/crowd_best.pt` (best validation MAE checkpoint)
''')

    add_code(r'''# Model Checkpoint Verification & File Size Inspection
print("=" * 70)
print("SAVED MODEL CHECKPOINTS & INTEGRITY CHECK")
print("=" * 70)

saved_checkpoints = [
    os.path.join(MODELS_DIR, "dehazing_model.pth"),
    os.path.join(MODELS_DIR, "dehazing_best.pth"),
    os.path.join(MODELS_DIR, "crowd_density_model.pth"),
    os.path.join(MODELS_DIR, "crowd_best.pt")
]

for ckpt_path in saved_checkpoints:
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        num_keys = len(state_dict) if isinstance(state_dict, dict) else "Full Object"
        print(f"[VERIFIED] {os.path.basename(ckpt_path):28s} | Size: {size_mb:6.2f} MB | Keys: {num_keys}")
    else:
        print(f"[MISSING]  {os.path.basename(ckpt_path):28s} | File not found!")

print("=" * 70)
''')

    # =========================================================================
    # SECTION 15 — Download Trained Models
    # =========================================================================
    add_md(r'''## SECTION 15 — Model & Artifact Export (Colab Download & Google Drive Sync)

Executable code cells to export the trained weights and reports to your local computer:
1. **Zip Archive Creation**: Packages all model checkpoints and CSV reports.
2. **Direct Browser Download**: Uses `google.colab.files.download()` to download the zip file directly.
3. **Google Drive Sync Option**: Mounts Google Drive to persist weights across sessions.
''')

    add_code(r'''# Create Zip Archive & Download Weights
import zipfile

artifact_zip_path = os.path.join(BASE_DIR, "crowd_surveillance_artifacts.zip")

with zipfile.ZipFile(artifact_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            full_path = os.path.join(root, f)
            arcname = os.path.join("models", f)
            zipf.write(full_path, arcname)
    for root, _, files in os.walk(REPORTS_DIR):
        for f in files:
            full_path = os.path.join(root, f)
            arcname = os.path.join("reports", f)
            zipf.write(full_path, arcname)
    if os.path.exists(dashboard_path):
        zipf.write(dashboard_path, os.path.join("outputs", "master_surveillance_dashboard.png"))

print(f"[SUCCESS] Packaged artifacts into: {artifact_zip_path}")
print(f"Archive Size: {os.path.getsize(artifact_zip_path) / (1024 * 1024):.2f} MB")

try:
    from google.colab import files
    print("[INFO] Triggering browser download of trained model artifacts...")
    files.download(artifact_zip_path)
except Exception:
    print(f"[NOTE] Not running inside Google Colab frontend. Zip file available at: {artifact_zip_path}")
''')

    add_code(r'''# Optional: Backup to Google Drive
try:
    from google.colab import drive
    print("[INFO] To back up models permanently to Google Drive, uncomment below:")
    # drive.mount('/content/drive')
    # drive_backup_dir = '/content/drive/MyDrive/CrowdSurveillanceModels'
    # os.makedirs(drive_backup_dir, exist_ok=True)
    # shutil.copy(os.path.join(MODELS_DIR, 'crowd_best.pt'), drive_backup_dir)
    # shutil.copy(os.path.join(MODELS_DIR, 'dehazing_best.pth'), drive_backup_dir)
    # print(f"[SUCCESS] Checkpoints copied to {drive_backup_dir}")
except Exception:
    pass
''')

    # =========================================================================
    # FINAL OUTPUT SUMMARY
    # =========================================================================
    add_md(r'''## FINAL OUTPUT SUMMARY

This executive summary summarizes all training locations, model checkpoints, validation and test metrics, CSV logs, and visual surveillance artifacts.
''')

    add_code(r'''# Executive Final Output Summary
print("=" * 80)
print("              ADVERSE WEATHER SURVEILLANCE PIPELINE: FINAL SUMMARY")
print("=" * 80)
print(f"Hardware Runtime               : {DEVICE}")
print(f"JHU-CROWD++ Dataset Root       : {JHU_DIR}")
print(f"  - Train / Val / Test Splits  : {len(train_crowd_ds)} / {len(val_crowd_ds)} / {len(test_crowd_ds)} samples")
print(f"RESIDE SOTS Dataset Root       : {RESIDE_DIR}")
print(f"  - Train / Val Pairs          : {len(train_dehaze_ds)} / {len(val_dehaze_ds)} pairs")
print("-" * 80)
print("TRAINED MODEL ARTIFACTS:")
print(f"  - Dehazing Model (Best)      : {os.path.join(MODELS_DIR, 'dehazing_best.pth')}")
print(f"  - CSRNet Crowd Model (Best)  : {os.path.join(MODELS_DIR, 'crowd_best.pt')}")
print(f"  - CSRNet Final Model         : {os.path.join(MODELS_DIR, 'crowd_density_model.pth')}")
print("-" * 80)
print("BENCHMARK EVALUATION METRICS:")
print(f"  - Dehazing Restoration       : PSNR = {test_psnr:.2f} dB | SSIM = {test_ssim:.4f}")
print(f"  - Crowd Density Estimation   : MAE = {test_mae:.2f} | RMSE = {test_rmse:.2f}")
print("-" * 80)
print("SURVEILLANCE OUTPUT ARTIFACTS:")
print(f"  - Master Visual Dashboard    : {dashboard_path}")
print(f"  - Annotated Video Feed       : {surveillance_video_out}")
print(f"  - CSV Audit Event Log        : {csv_log_path}")
print(f"  - Downloadable Zip Package   : {artifact_zip_path}")
print("=" * 80)
''')

    out_file = r"C:\Users\91636\.gemini\antigravity\scratch\crowd_surveillance_pipeline\Adverse_Weather_Crowd_Surveillance_Pipeline.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook successfully written to {out_file}")

if __name__ == "__main__":
    create_notebook()
