from collections import Counter


# ============================================================
# EXTRACT YOLO DETECTIONS
# ============================================================

def extract_detections(result):

    detections = []

    if result is None or result.boxes is None:
        return detections

    for i in range(len(result.boxes)):

        cls_id = int(
            result.boxes.cls[i].item()
        )

        confidence = float(
            result.boxes.conf[i].item()
        )

        bbox = result.boxes.xyxy[i].tolist()

        detections.append({
            "class_id": cls_id,
            "confidence": confidence,
            "bbox": bbox
        })

    return detections


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def analyze_products(result):

    detections = extract_detections(result)

    counts = Counter(
        d["class_id"]
        for d in detections
    )

    return {
        "total": len(detections),
        "unique_types": len(counts),
        "counts": dict(counts),
        "detections": detections
    }


# ============================================================
# SKU ANALYSIS
# ============================================================

def analyze_skus(result):

    detections = extract_detections(result)

    counts = Counter(
        d["class_id"]
        for d in detections
    )

    return {
        "total": len(detections),
        "unique_skus": len(counts),
        "counts": dict(counts),
        "detections": detections
    }


# ============================================================
# VOID / EMPTY SPACE ANALYSIS
# ============================================================

def analyze_voids(result):

    detections = extract_detections(result)

    return {
        "total": len(detections),
        "detections": detections
    }


# ============================================================
# PRODUCT STOCK STATUS
# ============================================================

def get_product_status(count):

    if count == 0:
        return "OUT OF STOCK"

    elif count <= 2:
        return "LOW"

    else:
        return "FULL"


# ============================================================
# SKU STOCK STATUS
# ============================================================

def get_sku_status(count):

    if count == 0:
        return "OUT OF STOCK"

    elif count <= 2:
        return "LOW"

    else:
        return "FULL"


# ============================================================
# OVERALL SHELF STATUS
# ============================================================

def get_overall_status(product_count, void_count):

    # --------------------------------------------------------
    # No products detected
    # --------------------------------------------------------

    if product_count <= 0:
        return "EMPTY"


    # --------------------------------------------------------
    # Calculate percentage of empty spaces
    # --------------------------------------------------------

    empty_ratio = void_count / product_count


    # --------------------------------------------------------
    # 0% - 20% empty
    # Shelf is mostly stocked
    # --------------------------------------------------------

    if empty_ratio <= 0.20:
        return "FULL"


    # --------------------------------------------------------
    # More than 20% and up to 50% empty
    # Shelf needs attention
    # --------------------------------------------------------

    elif empty_ratio <= 0.50:
        return "LOW"


    # --------------------------------------------------------
    # More than 50% empty
    # Shelf is critically under-stocked
    # --------------------------------------------------------

    else:
        return "EMPTY"


# ============================================================
# COMPLETE SHELF ANALYSIS
# ============================================================

def analyze_shelf(results):

    # ========================================================
    # PRODUCT ANALYSIS
    # ========================================================

    products = analyze_products(
        results["product"]
    )


    # ========================================================
    # SKU ANALYSIS
    # ========================================================

    skus = analyze_skus(
        results["sku"]
    )


    # ========================================================
    # VOID ANALYSIS
    # ========================================================

    voids = analyze_voids(
        results["void"]
    )


    # ========================================================
    # PRODUCT STATUS
    # ========================================================

    product_status = get_product_status(
        products["total"]
    )


    # ========================================================
    # SKU STATUS
    # ========================================================

    sku_status = {}

    for sku_id, count in skus["counts"].items():

        sku_status[sku_id] = {
            "count": count,
            "status": get_sku_status(count)
        }


    # ========================================================
    # OVERALL SHELF STATUS
    # ========================================================

    overall_status = get_overall_status(
        products["total"],
        voids["total"]
    )


    # ========================================================
    # FINAL ANALYSIS RESULT
    # ========================================================

    return {

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        "summary": {

            "total_products":
                products["total"],

            "unique_product_types":
                products["unique_types"],

            "total_sku_detections":
                skus["total"],

            "unique_skus":
                skus["unique_skus"],

            "empty_spaces":
                voids["total"],

            "product_status":
                product_status,

            "overall_status":
                overall_status
        },


        # ----------------------------------------------------
        # PRODUCTS
        # ----------------------------------------------------

        "products":
            products,


        # ----------------------------------------------------
        # SKUS
        # ----------------------------------------------------

        "skus": {

            "analysis":
                skus,

            "status":
                sku_status
        },


        # ----------------------------------------------------
        # EMPTY SPACES
        # ----------------------------------------------------

        "voids":
            voids
    }
