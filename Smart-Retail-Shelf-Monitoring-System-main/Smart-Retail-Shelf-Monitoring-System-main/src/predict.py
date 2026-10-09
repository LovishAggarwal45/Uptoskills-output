from pathlib import Path
from ultralytics import YOLO


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

if not MODEL_DIR.exists():
    for candidate in [
        BASE_DIR.parent / "models",
        BASE_DIR / "Smart-Retail-Shelf-Monitoring-System-main" / "models",
        BASE_DIR.parent / "Smart-Retail-Shelf-Monitoring-System-main" / "models"
    ]:
        if candidate.exists():
            MODEL_DIR = candidate
            break

PRODUCT_MODEL_PATH = MODEL_DIR / "product_best.pt"
SKU_MODEL_PATH = MODEL_DIR / "sku_best.pt"
VOID_MODEL_PATH = MODEL_DIR / "void_best.pt"


# ============================================================
# LOAD MODELS
# ============================================================

product_model = YOLO(str(PRODUCT_MODEL_PATH))
sku_model = YOLO(str(SKU_MODEL_PATH))
void_model = YOLO(str(VOID_MODEL_PATH))


# ============================================================
# PRODUCT DETECTION
# ============================================================

def detect_products(image, conf=0.25):

    results = product_model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    return results[0]


# ============================================================
# SKU DETECTION
# ============================================================

def detect_skus(image, conf=0.20):

    results = sku_model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    return results[0]


# ============================================================
# EMPTY SPACE DETECTION
# ============================================================

def detect_voids(image, conf=0.25):

    results = void_model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    return results[0]


# ============================================================
# RUN ALL MODELS
# ============================================================

def run_all_models(
    image,
    product_conf=0.25,
    sku_conf=0.20,
    void_conf=0.25
):

    product_result = detect_products(
        image,
        conf=product_conf
    )

    sku_result = detect_skus(
        image,
        conf=sku_conf
    )

    void_result = detect_voids(
        image,
        conf=void_conf
    )

    return {
        "product": product_result,
        "sku": sku_result,
        "void": void_result
    }