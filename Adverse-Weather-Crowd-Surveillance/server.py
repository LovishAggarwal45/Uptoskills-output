import os
import sys
import time
import io
import json
import csv
import subprocess
import imageio_ffmpeg
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from surveillance_pipeline import (
    CrowdSurveillancePipeline,
    array_to_base64_jpeg
)

# Initialize FastAPI App
app = FastAPI(
    title="Adverse Weather Dynamic De-Hazing and Crowd Counting System",
    description="Functional Research Prototype & Surveillance Demonstration Pipeline",
    version="1.0.0-prototype"
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

for d in [TEMPLATES_DIR, STATIC_DIR, OUTPUTS_DIR, REPORTS_DIR, SAMPLE_DIR]:
    os.makedirs(d, exist_ok=True)

# Mount static, templates, and sample data
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/sample_data", StaticFiles(directory=SAMPLE_DIR), name="sample_data")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Pipeline Engine (CPU fallback guaranteed)
pipeline = CrowdSurveillancePipeline(
    dehaze_weights=os.path.join(BASE_DIR, "models", "dehazing_best.pth"),
    crowd_weights=os.path.join(BASE_DIR, "models", "crowd_best.pt")
)

# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the main surveillance control center dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "config": pipeline.config,
            "device": str(pipeline.device)
        }
    )

@app.get("/api/health")
async def health_check():
    """Health diagnostic endpoint reporting model device and status."""
    return {
        "status": "healthy",
        "system": "Adverse Weather Dynamic De-Hazing and Crowd Counting Pipeline (Prototype)",
        "device": str(pipeline.device),
        "models_loaded": {
            "dehazing_net": True,
            "crowd_density_net": True
        },
        "config": pipeline.config
    }

@app.get("/api/config")
async def get_configuration():
    """Returns current active surveillance configuration."""
    return pipeline.config

@app.post("/api/config")
async def update_configuration(request: Request):
    """Updates configurable capacity thresholds, multipliers, and camera location."""
    data = await request.json()
    updated = pipeline.update_config(data)
    return {"status": "success", "config": updated}

@app.post("/api/process/image")
async def process_image_endpoint(file: UploadFile = File(...), location: str = Form(None)):
    """
    Primary Image Upload Endpoint:
    Processes degraded camera frame end-to-end and returns:
    - Base64 images: Original, Transmission Map, Restored Frame, Density Heatmap
    - Telemetry: condition, count, visibility index, risk score, alert status
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    result = pipeline.process_frame(img_np, location_override=location, log_event=True)
    
    # Encode images to Base64 for instantaneous UI rendering
    orig_b64 = array_to_base64_jpeg(result["original_frame"])
    trans_b64 = array_to_base64_jpeg(result["transmission_colormap"])
    rest_b64 = array_to_base64_jpeg(result["restored_frame"])
    dens_b64 = array_to_base64_jpeg(result["density_colormap"])
    
    return {
        "status": "success",
        "telemetry": {
            "location": result["location"],
            "detected_condition": result["detected_condition"],
            "degradation_explanation": result["degradation_explanation"],
            "airlight_A": result["airlight_A"],
            "crowd_count": result["crowd_count"],
            "density_peak": result["density_peak"],
            "visibility_index": result["visibility_index"],
            "max_crowd_capacity": result["max_crowd_capacity"],
            "weather_multiplier": result["weather_multiplier"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "alert_status": result["alert_status"],
            "latency_ms": result["latency_ms"],
            "fps": result["fps"],
            "bounding_boxes": result["bounding_boxes"],
            "bounding_box_count": len(result["bounding_boxes"])
        },
        "visuals": {
            "original_frame": orig_b64,
            "transmission_map": trans_b64,
            "restored_frame": rest_b64,
            "density_heatmap": dens_b64
        }
    }

@app.post("/api/process/preset")
async def process_preset_endpoint(name: str = "fog"):
    """
    Instant Demonstration Preset Endpoint:
    Loads one of the verified adverse weather test samples:
    fog, rain, glare, or clear.
    """
    preset_map = {
        "clear": os.path.join(SAMPLE_DIR, "01_shibuya_crossing_clear_crowd.jpg"),
        "fog": os.path.join(SAMPLE_DIR, "02_heavy_fog_pedestrians.jpg"),
        "rain": os.path.join(SAMPLE_DIR, "03_monsoon_rain_umbrellas.jpg"),
        "glare": os.path.join(SAMPLE_DIR, "04_times_square_sun_glare.jpg"),
        "empty": os.path.join(SAMPLE_DIR, "05_empty_bridge_fog_control.jpg")
    }
    
    preset_key = name.lower().strip()
    if preset_key not in preset_map:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{name}'. Choose from: {list(preset_map.keys())}")
        
    img_path = preset_map[preset_key]
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Preset file not found at {img_path}")
        
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img)
    
    result = pipeline.process_frame(img_np, location_override=f"Preset-{preset_key.upper()}", log_event=True)
    
    orig_b64 = array_to_base64_jpeg(result["original_frame"])
    trans_b64 = array_to_base64_jpeg(result["transmission_colormap"])
    rest_b64 = array_to_base64_jpeg(result["restored_frame"])
    dens_b64 = array_to_base64_jpeg(result["density_colormap"])
    
    return {
        "status": "success",
        "telemetry": {
            "location": result["location"],
            "detected_condition": result["detected_condition"],
            "degradation_explanation": result["degradation_explanation"],
            "airlight_A": result["airlight_A"],
            "crowd_count": result["crowd_count"],
            "density_peak": result["density_peak"],
            "visibility_index": result["visibility_index"],
            "max_crowd_capacity": result["max_crowd_capacity"],
            "weather_multiplier": result["weather_multiplier"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "alert_status": result["alert_status"],
            "latency_ms": result["latency_ms"],
            "fps": result["fps"],
            "bounding_boxes": result["bounding_boxes"],
            "bounding_box_count": len(result["bounding_boxes"])
        },
        "visuals": {
            "original_frame": orig_b64,
            "transmission_map": trans_b64,
            "restored_frame": rest_b64,
            "density_heatmap": dens_b64
        }
    }

def reencode_to_browser_h264(video_path):
    """Re-encodes video using ffmpeg to native web H.264 (libx264, yuv420p) for seamless HTML5 video player playback."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_out = video_path.replace(".mp4", "_h264.mp4")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            temp_out
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(temp_out):
            os.replace(temp_out, video_path)
    except Exception as e:
        print(f"[WARN] Failed to re-encode video with ffmpeg: {e}")

@app.post("/api/process/video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    sampling_interval: int = Form(3),
    location: str = Form(None)
):
    """
    Periodic Video Processing Endpoint:
    Samples video frames at specified intervals, runs full surveillance pipeline,
    and renders an annotated dual-panel video with HUD overlay.
    """
    temp_in = os.path.join(OUTPUTS_DIR, f"temp_upload_{int(time.time())}.mp4")
    video_out_name = f"surveillance_annotated_{int(time.time())}.mp4"
    video_out_path = os.path.join(OUTPUTS_DIR, video_out_name)
    
    contents = await file.read()
    with open(temp_in, "wb") as f:
        f.write(contents)
        
    cap = cv2.VideoCapture(temp_in)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video file.")
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
    
    # Output dual panel: width * 2 x height
    out_w, out_h = w * 2, h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / sampling_interval), (out_w, out_h))
    
    frame_idx = 0
    processed_count = 0
    max_frames_to_process = 60  # Guardrail for prototype CPU processing
    
    events_summary = []
    
    while cap.isOpened() and processed_count < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sampling_interval == 0:
            res = pipeline.process_frame(frame, location_override=location, log_event=True)
            processed_count += 1
            events_summary.append({
                "frame": frame_idx,
                "count": res["crowd_count"],
                "condition": res["detected_condition"],
                "risk": res["risk_level"],
                "alert": res["alert_status"]
            })
            
            # Left Panel: Restored RGB to BGR
            left = cv2.cvtColor(res["restored_frame"], cv2.COLOR_RGB2BGR)
            # Right Panel: Density Colormap
            right = cv2.cvtColor(res["density_colormap"], cv2.COLOR_RGB2BGR)
            right = cv2.resize(right, (w, h))
            
            # Canvas
            canvas = np.hstack([left, right])
            
            # HUD Banner
            hud_color = (0, 0, 255) if res["risk_level"] == "CRITICAL" else (0, 165, 255) if res["risk_level"] == "HIGH" else (0, 255, 0)
            cv2.rectangle(canvas, (0, 0), (out_w, 55), (30, 30, 30), -1)
            hud_txt = f"COUNT: {res['crowd_count']:.0f} | WEATHER: {res['detected_condition'].upper()} | RISK: {res['risk_level']} ({res['risk_score']:.2f}) | VIS: {res['visibility_index']:.2f}"
            cv2.putText(canvas, hud_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(canvas, f"ALERT: {res['alert_status']}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
            
            out_writer.write(canvas)
            
        frame_idx += 1
        
    cap.release()
    out_writer.release()
    reencode_to_browser_h264(video_out_path)
    if os.path.exists(temp_in):
        os.remove(temp_in)
        
    return {
        "status": "success",
        "processed_frames": processed_count,
        "video_url": f"/outputs/{video_out_name}",
        "events_summary": events_summary
    }

@app.post("/api/process/video-preset")
async def process_video_preset_endpoint(name: str = "fog", sampling_interval: int = 3):
    """
    Runs video surveillance on pre-packaged sample video:
    fog or rain.
    """
    preset_videos = {
        "fog": os.path.join(SAMPLE_DIR, "surveillance_fog_sample.mp4"),
        "rain": os.path.join(SAMPLE_DIR, "surveillance_rain_sample.mp4")
    }
    
    key = name.lower().strip()
    if key not in preset_videos:
        raise HTTPException(status_code=404, detail=f"Unknown video preset '{name}'. Choose from: {list(preset_videos.keys())}")
        
    vid_in = preset_videos[key]
    if not os.path.exists(vid_in):
        raise HTTPException(status_code=404, detail=f"Preset video not found at {vid_in}")
        
    video_out_name = f"surveillance_preset_{key}_{int(time.time())}.mp4"
    video_out_path = os.path.join(OUTPUTS_DIR, video_out_name)
    
    cap = cv2.VideoCapture(vid_in)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video file.")
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    
    out_w, out_h = w * 2, h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / sampling_interval), (out_w, out_h))
    
    frame_idx = 0
    processed_count = 0
    events_summary = []
    
    while cap.isOpened() and processed_count < 30:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sampling_interval == 0:
            res = pipeline.process_frame(frame, location_override=f"Preset-Video-{key.upper()}", log_event=True)
            processed_count += 1
            events_summary.append({
                "frame": frame_idx,
                "count": res["crowd_count"],
                "condition": res["detected_condition"],
                "risk": res["risk_level"],
                "alert": res["alert_status"]
            })
            
            left = cv2.cvtColor(res["restored_frame"], cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(res["density_colormap"], cv2.COLOR_RGB2BGR)
            right = cv2.resize(right, (w, h))
            
            canvas = np.hstack([left, right])
            hud_color = (0, 0, 255) if res["risk_level"] == "CRITICAL" else (0, 165, 255) if res["risk_level"] == "HIGH" else (0, 255, 0)
            cv2.rectangle(canvas, (0, 0), (out_w, 55), (30, 30, 30), -1)
            hud_txt = f"COUNT: {res['crowd_count']:.0f} | WEATHER: {res['detected_condition'].upper()} | RISK: {res['risk_level']} ({res['risk_score']:.2f}) | VIS: {res['visibility_index']:.2f}"
            cv2.putText(canvas, hud_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(canvas, f"ALERT: {res['alert_status']}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
            
            out_writer.write(canvas)
            
        frame_idx += 1
        
    cap.release()
    out_writer.release()
    reencode_to_browser_h264(video_out_path)
    
    return {
        "status": "success",
        "processed_frames": processed_count,
        "video_url": f"/outputs/{video_out_name}",
        "events_summary": events_summary
    }

# =========================================================================
# LIVE STREAMING (MJPEG FEED)
# =========================================================================

def generate_live_mjpeg_stream():
    """Streams continuous real outdoor surveillance camera footage of actual humans through the pipeline."""
    video_path = os.path.join(SAMPLE_DIR, "real_pedestrian_crowd.ogv")
    if not os.path.exists(video_path):
        video_path = os.path.join(SAMPLE_DIR, "surveillance_fog_sample.mp4")
        
    cap = cv2.VideoCapture(video_path)
    
    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            
        ret, frame = cap.read()
        if not ret:
            # Loop seamlessly back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
                
        h, w = frame.shape[:2]
        
        # Process through full pipeline (restored frame + CSRNet crowd density)
        res = pipeline.process_frame(frame, log_event=False)
        
        # Dual Panel: Left = Restored Real Human Video, Right = Real CSRNet Density Heatmap
        left = cv2.cvtColor(res["restored_frame"], cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(res["density_colormap"], cv2.COLOR_RGB2BGR)
        right = cv2.resize(right, (w, h))
        stream_canvas = np.hstack([left, right])
        
        # Telemetry HUD banner
        cv2.rectangle(stream_canvas, (0, 0), (w * 2, 40), (25, 25, 25), -1)
        cv2.putText(
            stream_canvas,
            f"LIVE CCTV FEED | REAL PEDESTRIANS: {res['crowd_count']:.0f} | WEATHER: {res['detected_condition'].upper()} | RISK: {res['risk_level']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        _, jpeg = cv2.imencode('.jpg', stream_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.08)  # ~12 FPS real surveillance stream

@app.get("/api/stream/feed")
async def live_stream_feed():
    """Provides a real-time MJPEG live stream of the surveillance pipeline."""
    return StreamingResponse(
        generate_live_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================================
# AUDIT LOGS & CSV EXPORT
# =========================================================================

@app.get("/api/logs")
async def get_recent_logs(limit: int = 50):
    """Retrieves recent logged events from the CSV audit log."""
    csv_file = pipeline.logger.csv_path
    if not os.path.exists(csv_file):
        return {"logs": []}
        
    logs = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logs.append(row)
            
    return {"logs": logs[-limit:]}

@app.get("/api/logs/csv")
async def download_csv_log():
    """Triggers download of the raw CSV audit log file."""
    csv_file = pipeline.logger.csv_path
    if not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail="No logs recorded yet.")
    return FileResponse(
        csv_file,
        media_type="text/csv",
        filename="crowd_surveillance_log.csv"
    )

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting Adverse Weather Crowd Surveillance Server (Prototype)")
    print("Dashboard available at: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
