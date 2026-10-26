"""
Web Application Server for Wrong-Way Driving & Lane Discipline Detection
Provides a modern local web dashboard with real-time video streaming,
dynamic session-specific violation logging, live HTML audit report with
in-browser clip player modal, and video-specific CSV export.
"""

import csv
import io
import json
import os
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

from config import DEFAULT_LANES_720P, LaneDefinition, SystemConfig
from clip_recorder import ViolationClipRecorder
from csv_logger import CSVLogger
from detector_tracker import VehicleDetectorTracker
from lane_manager import LaneManager
from trajectory_estimator import TrajectoryEstimator
from violation_detector import ViolationDetector
from visualizer import Visualizer

app = Flask(__name__)
UPLOAD_DIR = "uploads"
CLIPS_DIR = os.path.join("output", "violation_clips")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

class StreamState:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.restart_requested = False
        self.current_frame_bytes: Optional[bytes] = None
        self.lock = threading.Lock()
        self.active_vehicles = 0
        self.wrong_way_count = 0
        self.straddle_count = 0
        self.fps = 0.0
        # Start completely fresh — no stale preloaded sample records!
        self.violations: List[Dict[str, str]] = []
        self.video_source = "sample_traffic.mp4"
        self.confidence = 0.40
        self.ww_persistence = 15
        self.straddle_persistence = 25
        self.thread: Optional[threading.Thread] = None

# Global clip recorder placeholder
clip_recorder: Optional[ViolationClipRecorder] = None

state = StreamState()

def processing_worker():
    global clip_recorder

    global state
    print(f"[*] Starting video processing thread on '{state.video_source}'...")
    cap = cv2.VideoCapture(state.video_source)
    if not cap.isOpened():
        print(f"[!] Failed to open {state.video_source}")
        state.is_running = False
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Scale lanes if video is 1280x720 or similar
    sx, sy = width / 1280.0, height / 720.0
    lanes = [
        LaneDefinition(
            lane_id=ld.lane_id,
            name=ld.name,
            polygon=[(int(x * sx), int(y * sy)) for x, y in ld.polygon],
            expected_heading_deg=ld.expected_heading_deg,
            allowed_angle_deviation_deg=ld.allowed_angle_deviation_deg,
        )
        for ld in DEFAULT_LANES_720P
    ]

    config = SystemConfig()
    detector = VehicleDetectorTracker(
        model_path=config.model_name,
        confidence_threshold=state.confidence,
        target_classes=config.vehicle_classes
    )
    lane_mgr = LaneManager(lanes=lanes, frame_width=width, frame_height=height, auto_learn_flow=True)
    traj_estimator = TrajectoryEstimator(
        max_history=config.trajectory_history_length,
        direction_window=config.direction_window_frames,
        min_displacement=config.min_displacement_px
    )
    violation_detector = ViolationDetector(
        wrong_way_persistence_frames=state.ww_persistence,
        straddle_persistence_frames=state.straddle_persistence,
        min_total_displacement_px=config.min_total_displacement_px,
        fps=fps
    )
    
    # Initialize clip recorder with video FPS
    global clip_recorder
    clip_recorder = ViolationClipRecorder(CLIPS_DIR, fps=fps)
    # Save a clean session-specific CSV log
    safe_video_name = os.path.splitext(os.path.basename(state.video_source))[0]
    session_csv_path = os.path.join("output", f"violations_{safe_video_name}.csv")
    csv_logger = CSVLogger(session_csv_path)
    visualizer = Visualizer(lane_manager=lane_mgr)

    frame_idx = 0
    t0 = time.time()

    while state.is_running:
        with state.lock:
            if state.restart_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                state.restart_requested = False
            paused = state.is_paused

        if paused:
            time.sleep(0.04)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            # Loop video continuously
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        if clip_recorder:
            clip_recorder.buffer_frame(frame_idx, frame)

        tracked_vehicles = detector.process_frame(frame)
        active_ids = [v.track_id for v in tracked_vehicles]

        annotated = frame.copy()
        annotated = visualizer.draw_lanes(annotated)
        current_alerts = []

        for vehicle in tracked_vehicles:
            heading_deg, velocity, speed, is_moving = traj_estimator.update_vehicle(
                track_id=vehicle.track_id,
                frame_idx=frame_idx,
                centroid=vehicle.centroid
            )

            primary_lane_id, overlaps = lane_mgr.assign_vehicle_to_lane(vehicle.bbox)
            primary_lane = lane_mgr.get_lane(primary_lane_id) if primary_lane_id else None
            lane_name = primary_lane.name if primary_lane else "Roadway"

            is_straddling, _ = lane_mgr.check_straddling(
                overlaps, min_ratio=config.straddle_overlap_min_ratio
            )

            is_wrong_way_instant = False
            deviation_deg = 0.0
            traj = traj_estimator.get_trajectory(vehicle.track_id)
            tot_disp = traj.total_displacement if traj else 0.0

            if heading_deg is not None and is_moving and tot_disp >= 25.0:
                is_wrong_way_instant, deviation_deg = lane_mgr.is_wrong_way(
                    lane_id=primary_lane_id,
                    vehicle_heading_deg=heading_deg,
                    centroid=vehicle.centroid,
                    threshold_deg=config.wrong_way_angle_threshold
                )
                if not is_wrong_way_instant and not is_straddling:
                    lane_mgr.update_learned_flow(
                        lane_id=primary_lane_id,
                        heading_deg=heading_deg,
                        confidence=vehicle.confidence,
                        centroid=vehicle.centroid,
                        vehicle_id=vehicle.track_id
                    )

            new_violations, hud_alerts = violation_detector.update_vehicle(
                vehicle_id=vehicle.track_id,
                vehicle_type=vehicle.class_name,
                frame_idx=frame_idx,
                primary_lane_id=primary_lane_id,
                lane_name=lane_name,
                is_moving=is_moving,
                heading_deg=heading_deg,
                is_wrong_way_instant=is_wrong_way_instant,
                deviation_deg=deviation_deg,
                is_straddling_instant=is_straddling,
                confidence=vehicle.confidence,
                displacement_px=tot_disp
            )
            current_alerts.extend(hud_alerts)

            for ev in new_violations:
                csv_logger.log_violation(ev)
                # Trigger clip recording if available
                if clip_recorder:
                    clip_recorder.trigger_violation_clip(ev)
                # Store only the basename for UI
                clip_fname = os.path.basename(ev.clip_path) if getattr(ev, "clip_path", None) else ""
                record = {
                    "violation_id": ev.violation_id,
                    "vehicle_id": str(ev.vehicle_id),
                    "vehicle_type": ev.vehicle_type,
                    "violation_type": ev.violation_type.value,
                    "lane": ev.lane_name,
                    "timestamp": ev.start_timestamp,
                    "confidence": f"{ev.confidence:.2f}",
                    "heading_deg": f"{ev.heading_deg:.1f}",
                    "deviation_deg": f"{ev.deviation_deg:.1f}",
                    "duration_seconds": f"{ev.duration_seconds:.2f}",
                    "clip_path": clip_fname
                }
                with state.lock:
                    state.violations.append(record)
                    if "Wrong" in ev.violation_type.value:
                        state.wrong_way_count += 1
                    else:
                        state.straddle_count += 1

            is_confirmed_ww, is_confirmed_st = violation_detector.get_vehicle_violation_status(vehicle.track_id)
            trail = traj.get_trail() if traj else []

            visualizer.draw_vehicle(
                frame=annotated,
                vehicle=vehicle,
                lane_name=lane_name,
                heading_deg=heading_deg,
                is_moving=is_moving,
                is_wrong_way=is_confirmed_ww,
                is_straddling=is_confirmed_st,
                trail=trail
            )

        traj_estimator.prune_stale_tracks(active_ids)
        violation_detector.prune_stale_vehicles(active_ids, frame_idx)

        visualizer.draw_hud(
            frame=annotated,
            frame_idx=frame_idx,
            fps=fps,
            active_vehicles_count=len(tracked_vehicles),
            wrong_way_count=state.wrong_way_count,
            straddle_count=state.straddle_count,
            active_alerts=list(set(current_alerts))
        )

        # Update telemetry
        elapsed = time.time() - t0
        with state.lock:
            state.active_vehicles = len(tracked_vehicles)
            state.fps = frame_idx / max(0.001, elapsed)

        # Encode JPEG for stream
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with state.lock:
            state.current_frame_bytes = jpeg.tobytes()

        time.sleep(0.01)

    cap.release()
    print("[*] Video processing stopped.")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        with state.lock:
            frame_data = state.current_frame_bytes
        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    with state.lock:
        return jsonify({
            "is_running": state.is_running,
            "is_paused": state.is_paused,
            "active_vehicles": state.active_vehicles,
            "wrong_way_count": state.wrong_way_count,
            "straddle_count": state.straddle_count,
            "fps": state.fps,
            "video_source": os.path.basename(state.video_source),
            "violations": state.violations[-100:]
        })

@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.get_json(silent=True) or {}
    new_source = data.get('video', 'sample_traffic.mp4')

    with state.lock:
        state.video_source = new_source
        state.confidence = float(data.get('confidence', 0.40))
        state.ww_persistence = int(data.get('wrong_way_persistence', 15))
        state.straddle_persistence = int(data.get('straddle_persistence', 25))
        state.is_paused = False
        state.restart_requested = False
        
        # RESET counters and violations cleanly for the new video
        state.violations = []
        state.wrong_way_count = 0
        state.straddle_count = 0
        state.active_vehicles = 0
        state.fps = 0.0
        state.current_frame_bytes = None

    if not state.is_running:
        state.is_running = True
        state.thread = threading.Thread(target=processing_worker, daemon=True)
        state.thread.start()

    return jsonify({"status": "started", "video": state.video_source})

@app.route('/api/pause', methods=['POST'])
def api_pause():
    data = request.get_json(silent=True) or {}
    with state.lock:
        if 'pause' in data:
            state.is_paused = bool(data['pause'])
        else:
            state.is_paused = not state.is_paused
        paused = state.is_paused
    return jsonify({"status": "ok", "is_paused": paused})

@app.route('/api/restart', methods=['POST'])
def api_restart():
    with state.lock:
        state.restart_requested = True
        state.is_paused = False
    return jsonify({"status": "ok", "message": "restarted"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    with state.lock:
        state.is_running = False
        state.is_paused = False
        state.restart_requested = False
    return jsonify({"status": "stopped"})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    filename = f"upload_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    return jsonify({"status": "uploaded", "filepath": filepath, "filename": filename})

# Serve clips with both route conventions
@app.route('/clips/<path:filename>')
@app.route('/output/violation_clips/<path:filename>')
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename, mimetype="video/mp4")

@app.route('/download/csv')
def download_csv():
    """Generates and serves a clean CSV containing ONLY the current video session's violations."""
    with state.lock:
        records = list(state.violations)
        video_name = os.path.basename(state.video_source)

    output_io = io.StringIO()
    fieldnames = [
        "violation_id", "vehicle_id", "vehicle_type", "violation_type",
        "lane", "timestamp", "confidence", "heading_deg", "deviation_deg",
        "duration_seconds"
    ]
    writer = csv.DictWriter(output_io, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        writer.writerow({
            "violation_id": r.get("violation_id", ""),
            "vehicle_id": r.get("vehicle_id", ""),
            "vehicle_type": r.get("vehicle_type", ""),
            "violation_type": r.get("violation_type", ""),
            "lane": r.get("lane", ""),
            "timestamp": r.get("timestamp", ""),
            "confidence": r.get("confidence", ""),
            "heading_deg": r.get("heading_deg", ""),
            "deviation_deg": r.get("deviation_deg", ""),
            "duration_seconds": r.get("duration_seconds", "0.00")
        })

    csv_bytes = output_io.getvalue().encode("utf-8")
    safe_name = os.path.splitext(video_name)[0]
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=violations_{safe_name}.csv"}
    )

@app.route('/report')
def serve_report():
    """Dynamically generates the audit report with working in-browser clip player modal."""
    with state.lock:
        records = list(state.violations)
        video_name = os.path.basename(state.video_source)

    total_violations = len(records)
    types_counter = Counter(r.get("violation_type", "") for r in records)
    classes_counter = Counter(r.get("vehicle_type", "") for r in records)
    lanes_counter = Counter(r.get("lane", "") for r in records)

    confs = [float(r.get("confidence", 0)) for r in records if r.get("confidence")]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    table_rows = []
    if not records:
        table_rows.append('<tr><td colspan="9" style="text-align: center; color: #94a3b8; padding: 30px;">No violations detected yet in the current video.</td></tr>')
    else:
        for r in records:
            vtype_badge = "badge-danger" if "Wrong-Way" in r.get("violation_type", "") else "badge-warning"
            
            clip_btn = f"<button class='btn-clip' onclick=\"openClipModal('{r.get('clip_path')}')\">View Clip</button>" if r.get('clip_path') else "<span style='color:#64748b;'>No clip</span>"
            table_rows.append(f"""
            <tr>
                <td><strong>{r.get('violation_id', '')}</strong></td>
                <td>#{r.get('vehicle_id', '')}</td>
                <td><span class='badge-vehicle'>{r.get('vehicle_type', '').capitalize()}</span></td>
                <td><span class='{vtype_badge}'>{r.get('violation_type', '')}</span></td>
                <td>{r.get('lane', '')}</td>
                <td><code>{r.get('timestamp', '')}</code></td>
                <td>{float(r.get('confidence', 0)):.1%}</td>
                <td>{r.get('heading_deg', '0')}°</td>
                <td>{r.get('deviation_deg', '0')}°</td>
                <td>{clip_btn}</td>
            </tr>
            """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Traffic Violation Audit Report - {video_name}</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --border: #1f2937;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.6rem;
            color: var(--accent);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .kpi-title {{
            color: var(--text-dim);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}
        th {{
            background: #1e293b;
            padding: 14px 16px;
            color: var(--text-dim);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover {{
            background: rgba(255,255,255,0.02);
        }}
        .badge-danger {{
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            padding: 4px 8px;
            border-radius: 14px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-warning {{
            background: rgba(245, 158, 11, 0.2);
            color: #fcd34d;
            padding: 4px 8px;
            border-radius: 14px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-vehicle {{
            background: rgba(56, 189, 248, 0.15);
            color: #7dd3fc;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
        }}
        .btn-clip {{
            background: #2563eb;
            color: #fff;
            padding: 6px 12px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .btn-clip:hover {{
            background: #1d4ed8;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        .modal.active {{
            display: flex;
        }}
        .modal-content {{
            background: #111827;
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 24px;
            width: 90%;
            max-width: 760px;
            position: relative;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }}
        .modal-close {{
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 1.8rem;
            cursor: pointer;
        }}
        video {{
            width: 100%;
            border-radius: 10px;
            background: #000;
            max-height: 480px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🚨 Live Violation Audit Report</h1>
            <div style="color: var(--text-dim); margin-top: 4px; font-size: 0.85rem;">Active Video Source: <strong>{video_name}</strong></div>
        </div>
        <div>
            <a href="/download/csv" style="background:#374151; color:#fff; padding:8px 14px; border-radius:6px; text-decoration:none; font-size:0.85rem; font-weight:600;">📥 Download Session CSV</a>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Violations</div>
            <div class="kpi-value" style="color: var(--accent);">{total_violations}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Wrong-Way Driving</div>
            <div class="kpi-value" style="color: var(--danger);">{types_counter.get('Wrong-Way Driving', 0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Lane Discipline (Straddling)</div>
            <div class="kpi-value" style="color: var(--warning);">{types_counter.get('Lane Discipline Violation (Straddling)', 0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg Confidence</div>
            <div class="kpi-value" style="color: var(--success);">{avg_conf:.1%}</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Violation ID</th>
                    <th>Vehicle</th>
                    <th>Type</th>
                    <th>Infraction</th>
                    <th>Lane</th>
                    <th>Timestamp</th>
                    <th>Confidence</th>
                    <th>Heading</th>
                    <th>Deviation</th>
                    <th>Clip</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>

    <!-- Modal for viewing violation clip -->
    <div id="clipModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle" style="margin:0; font-size:1.1rem; color:var(--accent);">Violation Video Clip</h3>
                <button class="modal-close" onclick="closeClipModal()">&times;</button>
            </div>
            <video id="modalVideo" controls autoplay playsinline style="width:100%; border-radius:8px; background:#000; max-height:480px;"></video>
        </div>
    </div>

    <script>
        function openClipModal(clipPath) {{
            if (!clipPath) return;
            const modal = document.getElementById('clipModal');
            const video = document.getElementById('modalVideo');
            video.src = '/clips/' + clipPath;
            modal.classList.add('active');
            video.play().catch(e => console.log('Autoplay blocked:', e));
        }}

        function closeClipModal() {{
            const modal = document.getElementById('clipModal');
            const video = document.getElementById('modalVideo');
            video.pause();
            video.src = '';
            modal.classList.remove('active');
        }}

        window.onclick = function(event) {{
            const modal = document.getElementById('clipModal');
            if (event.target === modal) {{
                closeClipModal();
            }}
        }};
    </script>
</body>
</html>"""
    return html

if __name__ == '__main__':
    print("=" * 60)
    print("  AI TRAFFIC MONITORING - LOCAL WEB APPLICATION")
    print("  Access locally at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
