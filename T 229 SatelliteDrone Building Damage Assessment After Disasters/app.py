from __future__ import annotations
import csv, os, uuid
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from model_inference import InferenceEngine, load_annotation

ROOT = Path(__file__).resolve().parent; UPLOADS = ROOT / "uploads"; OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app = Flask(__name__); app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
engine = InferenceEngine(ROOT)

@app.get("/")
def index(): return render_template("index.html", model=engine.status())

@app.get("/api/health")
def health(): return jsonify({"application": "ready", "model": engine.status()})

@app.post("/api/analyze")
def analyze():
    image_file, annotation_file = request.files.get("image"), request.files.get("annotation")
    if not image_file or not image_file.filename: return jsonify({"error": "Choose a satellite or drone image first."}), 400
    if not annotation_file or not annotation_file.filename: return jsonify({"error": "A matching xBD annotation JSON or building polygon CSV is required for building-level analysis."}), 400
    if not image_file.filename.lower().endswith((".png", ".jpg", ".jpeg")): return jsonify({"error": "Images must be PNG or JPEG files."}), 400
    if not annotation_file.filename.lower().endswith((".json", ".csv")): return jsonify({"error": "Footprints must be a JSON or CSV file."}), 400
    token = uuid.uuid4().hex; image_path = UPLOADS / f"{token}.image"; annotation_path = UPLOADS / f"{token}{Path(annotation_file.filename).suffix.lower()}"; output_path = OUTPUTS / f"{token}.png"
    try:
        image_file.save(image_path); annotation_file.save(annotation_path)
        with Image.open(image_path) as image: image.verify()
        result = engine.analyze(image_path, load_annotation(annotation_path), output_path); result["overlay_url"] = f"/api/outputs/{output_path.name}"; result["csv_url"] = f"/api/results/{token}.csv"
        with (OUTPUTS / f"{token}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["building_id", "predicted_class", "confidence", "ground_truth"]); writer.writeheader(); writer.writerows(result["predictions"])
        return jsonify(result)
    except (ValueError, RuntimeError) as exc: return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Analysis failed"); return jsonify({"error": f"Analysis failed safely: {exc}"}), 500

@app.get("/api/outputs/<name>")
def output_image(name):
    path = OUTPUTS / Path(name).name; return send_file(path) if path.is_file() else ("Not found", 404)

@app.get("/api/results/<name>")
def output_csv(name):
    path = OUTPUTS / Path(name).name
    return send_file(path, as_attachment=True, download_name="disaster_scope_predictions.csv") if path.is_file() else ("Not found", 404)

if __name__ == "__main__": app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", "5000")), debug=False)
