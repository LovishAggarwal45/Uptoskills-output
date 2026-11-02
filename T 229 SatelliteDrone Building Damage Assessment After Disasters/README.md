# DisasterScope AI

Local Flask frontend for the T229 satellite/drone building damage assessment prototype.

## Features

- Responsive dark dashboard with drag-and-drop image upload.
- Validates xBD JSON and also accepts the Florence submission CSV using its `pixel_geometry_wkt` building polygons.
- Uses the notebook’s ResNet-18 context pipeline when `checkpoints/resnet18_context_best.pth` exists: 48 px context, 128×128 RGB crop, ImageNet normalization, CPU-safe inference.
- Returns actual per-building class/confidence, counts, CSV, and a polygon overlay.
- Starts honestly without a checkpoint; it never fabricates predictions or retrains at startup.

The experimental Siamese U-Net branch is not exposed as an operational feature. Predictions require human review.

## Windows PowerShell

```powershell
cd "D:\K.C engineering\Internship\SatelliteDrone Building Damage Assessment After Disasters"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and stop with `Ctrl+C`.

For genuine inference, install the compatible ML stack with `pip install -r requirements-ml.txt`, place the real checkpoint at `checkpoints/resnet18_context_best.pth`, and provide a post-disaster PNG/JPEG with its matching xBD label JSON containing `features.xy`. Keep checkpoints, datasets, reports, uploads, and outputs local and out of Git.

## Submission

Include `app.py`, `model_inference.py`, `templates/`, `static/`, `requirements.txt`, `README.md`, `.gitignore`, and tests. Preserve the original notebook. Do not commit weights, datasets, generated artifacts, credentials, or archives.
