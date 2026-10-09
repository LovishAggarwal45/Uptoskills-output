"""
Professional Shelf Visualizer for Retail Shelf Monitoring
Renders high-definition bounding boxes, labels, and translucent void overlays.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Modern Retail Color Palette (RGB)
COLOR_PALETTE = {
    "product": {
        "border": (14, 165, 233),      # Vivid Sky Blue
        "fill": (14, 165, 233, 40),    # Translucent
        "tag": (14, 165, 233),
        "text": (255, 255, 255),
        "label_prefix": "Product"
    },
    "sku": {
        "border": (168, 85, 247),      # Electric Purple
        "fill": (168, 85, 247, 40),
        "tag": (168, 85, 247),
        "text": (255, 255, 255),
        "label_prefix": "SKU"
    },
    "void": {
        "border": (239, 68, 68),       # Crimson Warning
        "fill": (239, 68, 68, 70),     # Highlight empty space
        "tag": (220, 38, 38),
        "text": (255, 255, 255),
        "label_prefix": "EMPTY"
    }
}


def get_default_font(size=14):
    """Attempt to load a crisp TrueType font, falling back to default."""
    font_names = ["segoeui.ttf", "arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_annotated_shelf(
    image: Image.Image,
    results: dict,
    show_products: bool = True,
    show_skus: bool = True,
    show_voids: bool = True,
    box_thickness: int = 2,
    show_confidence: bool = True,
    highlight_voids: bool = True,
    target_layer_only: str = "All"
) -> Image.Image:
    """
    Renders enterprise-grade annotations with semi-transparent highlights,
    anti-aliased crisp borders, and clean status tags.
    """
    if image is None:
        return None

    # Base image in RGBA for translucent overlays
    base_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Text drawing layer
    text_img = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    
    font = get_default_font(max(12, int(min(base_rgba.size) * 0.022)))
    width, height = base_rgba.size

    # Layer visibility configuration
    layers_to_render = []
    
    if (target_layer_only in ["All", "Products"]) and show_products and "product" in results:
        layers_to_render.append(("product", results["product"]))
        
    if (target_layer_only in ["All", "SKUs"]) and show_skus and "sku" in results:
        layers_to_render.append(("sku", results["sku"]))
        
    if (target_layer_only in ["All", "Voids"]) and show_voids and "void" in results:
        layers_to_render.append(("void", results["void"]))

    # 1. Render Boxes and Translucent Fills
    for layer_type, result_obj in layers_to_render:
        if result_obj is None or result_obj.boxes is None or len(result_obj.boxes) == 0:
            continue

        colors = COLOR_PALETTE[layer_type]
        boxes = result_obj.boxes

        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Bound clamp
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Draw translucent fill (especially for voids)
            if layer_type == "void" and highlight_voids:
                overlay_draw.rectangle([x1, y1, x2, y2], fill=colors["fill"])
            
            # Draw boundary
            overlay_draw.rectangle([x1, y1, x2, y2], outline=colors["border"] + (255,), width=box_thickness)

            # Label text
            if layer_type == "void":
                label_text = f"VOID {conf * 100:.0f}%" if show_confidence else "EMPTY VOID"
            elif layer_type == "sku":
                label_text = f"SKU #{cls_id} {conf * 100:.0f}%" if show_confidence else f"SKU #{cls_id}"
            else:
                label_text = f"P-{cls_id} {conf * 100:.0f}%" if show_confidence else f"Product {cls_id}"

            # Calculate label background size
            try:
                bbox_text = font.getbbox(label_text)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            except Exception:
                text_w, text_h = 70, 14

            pad_x, pad_y = 6, 3
            badge_h = text_h + 2 * pad_y
            badge_w = text_w + 2 * pad_x

            # Determine badge position (above box or inside top if near image edge)
            if y1 - badge_h >= 0:
                badge_y1 = y1 - badge_h
                badge_y2 = y1
            else:
                badge_y1 = y1
                badge_y2 = y1 + badge_h

            badge_x1 = x1
            badge_x2 = min(width, x1 + badge_w)

            # Draw badge background
            text_draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=colors["tag"] + (240,))
            text_draw.text((badge_x1 + pad_x, badge_y1 + pad_y), label_text, fill=colors["text"], font=font)

    # Composite layers: base + translucent fills/borders + badges
    combined = Image.alpha_composite(base_rgba, overlay)
    final_image = Image.alpha_composite(combined, text_img)
    return final_image.convert("RGB")
