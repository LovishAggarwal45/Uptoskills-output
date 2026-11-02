"""Local ResNet-18 inference following the T229 notebook pipeline."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
import csv
import json
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    ML_IMPORT_ERROR = None
except Exception as exc:
    torch = nn = models = transforms = None
    ML_IMPORT_ERROR = str(exc)
try:
    from shapely import wkt
except Exception:
    wkt = None

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
COLORS = {"no-damage": (43, 210, 126), "minor-damage": (244, 196, 48), "major-damage": (255, 139, 61), "destroyed": (239, 91, 91)}

class InferenceEngine:
    def __init__(self, root: Path):
        self.root, self.model, self.idx_to_class, self.error = root, None, {}, None
        self.device = "cpu"
        self.checkpoint_path = self._find_checkpoint()
        self._load()

    def _find_checkpoint(self):
        for path in (self.root / "checkpoints" / "resnet18_context_best.pth", self.root / "checkpoints" / "resnet18_damage_best.pth"):
            if path.is_file(): return path
        return None

    def _load(self):
        if not self.checkpoint_path:
            self.error = "No ResNet-18 checkpoint found in checkpoints/."; return
        if ML_IMPORT_ERROR:
            self.error = f"PyTorch/torchvision unavailable: {ML_IMPORT_ERROR}"; return
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            mapping = checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None
            mapping = mapping or {name: index for index, name in enumerate(CLASSES)}
            self.idx_to_class = {int(index): name for name, index in mapping.items()}
            model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, len(self.idx_to_class))
            model.load_state_dict(state); model.eval(); self.model = model
        except Exception as exc:
            self.error = f"Checkpoint could not be loaded: {exc}"

    @property
    def available(self): return self.model is not None

    def status(self):
        return {"available": self.available, "model": "ResNet-18 context classifier" if self.available else None,
                "checkpoint": self.checkpoint_path.name if self.checkpoint_path else None, "device": self.device,
                "message": "Model ready for inference." if self.available else self.error,
                "classes": [self.idx_to_class.get(i, name) for i, name in enumerate(CLASSES)]}

    @staticmethod
    def _features(annotation: dict[str, Any], width: int, height: int):
        if not isinstance(annotation, dict) or not isinstance(annotation.get("features"), dict): raise ValueError("Annotation JSON must contain a features object.")
        raw = annotation["features"].get("xy")
        if not isinstance(raw, list): raise ValueError("Annotation JSON must contain features.xy as a list.")
        if wkt is None: raise RuntimeError("Install shapely to parse xBD WKT building polygons.")
        result = []
        for index, feature in enumerate(raw, 1):
            try:
                polygon = wkt.loads(feature["wkt"]); minx, miny, maxx, maxy = polygon.bounds
                if polygon.is_empty or not polygon.is_valid or maxx <= 0 or maxy <= 0 or minx >= width or miny >= height: continue
                props = feature.get("properties", {}); uid = str(props.get("uid", f"building-{index:04d}")); result.append((uid, polygon, props))
            except Exception: continue
        return result

    def analyze(self, image_path: Path, annotation: dict[str, Any], output_path: Path):
        if not self.available: raise RuntimeError(self.error or "Model unavailable.")
        image = Image.open(image_path).convert("RGB"); features = self._features(annotation, image.width, image.height)
        if not features: raise ValueError("No valid in-image building polygons were found in features.xy.")
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        rows, overlay = [], image.copy(); draw = ImageDraw.Draw(overlay)
        for uid, polygon, props in features:
            minx, miny, maxx, maxy = polygon.bounds; left, top = max(0, int(minx)-48), max(0, int(miny)-48); right, bottom = min(image.width, int(maxx)+48), min(image.height, int(maxy)+48)
            if right <= left or bottom <= top: continue
            tensor = transform(image.crop((left, top, right, bottom))).unsqueeze(0)
            with torch.no_grad(): probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            confidence, index = probabilities.max(dim=0); label = self.idx_to_class.get(int(index), "unknown")
            coords = [(int(x), int(y)) for x, y in polygon.exterior.coords]; draw.line(coords + [coords[0]], fill=COLORS.get(label, (255,255,255)), width=3)
            rows.append({"building_id": uid, "predicted_class": label, "confidence": round(float(confidence), 4), "ground_truth": props.get("subtype")})
        overlay.save(output_path, format="PNG")
        return {"buildings_analyzed": len(rows), "counts": dict(Counter(row["predicted_class"] for row in rows)), "predictions": rows}

def load_annotation(path: Path):
    """Load xBD JSON or a polygon CSV into the common features.xy shape."""
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                polygon = row.get("pixel_geometry_wkt", "").strip()
                if polygon:
                    rows.append({"wkt": polygon, "properties": {"uid": row.get("building_uid", f"building-{index:04d}"), "subtype": row.get("ground_truth_damage") or row.get("ground_truth_subtype")}})
        if not rows: raise ValueError("CSV must contain a pixel_geometry_wkt column with building polygons.")
        return {"features": {"xy": rows}}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise ValueError(f"Annotation JSON is malformed: {exc.msg}.") from exc
