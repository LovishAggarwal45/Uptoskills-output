import os
import sys
import io
import csv
import json
from fastapi.testclient import TestClient
from PIL import Image

# Import server app
from server import app, pipeline, BASE_DIR

client = TestClient(app)

def test_01_health_and_model_loading():
    """Verify system health, device detection, and loaded model checkpoints."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["models_loaded"]["dehazing_net"] is True
    assert data["models_loaded"]["crowd_density_net"] is True
    assert "config" in data
    print(f"\n[PASS] Health Check passed. Device: {data['device']}")

def test_02_configuration_crud():
    """Verify configuration reading and updating."""
    # Get current config
    res_get = client.get("/api/config")
    assert res_get.status_code == 200
    cfg = res_get.json()
    assert "max_crowd_capacity" in cfg
    assert "weather_multipliers" in cfg

    # Update config
    update_payload = {
        "location": "Automated-Test-Gate-5",
        "max_crowd_capacity": 85.0,
        "sampling_interval": 4,
        "weather_multipliers": {
            "fog": 1.35
        }
    }
    res_post = client.post("/api/config", json=update_payload)
    assert res_post.status_code == 200
    res_data = res_post.json()
    assert res_data["status"] == "success"
    assert res_data["config"]["location"] == "Automated-Test-Gate-5"
    assert res_data["config"]["max_crowd_capacity"] == 85.0
    assert res_data["config"]["sampling_interval"] == 4
    assert res_data["config"]["weather_multipliers"]["fog"] == 1.35
    print("\n[PASS] Dynamic Configuration Updates verified.")

def test_03_image_upload_processing():
    """Verify single frame image processing through the complete surveillance pipeline."""
    sample_fog_path = os.path.join(BASE_DIR, "test_outputs", "Sample_2_Heavy_Fog_original.png")
    assert os.path.exists(sample_fog_path), f"Test sample not found at {sample_fog_path}"

    with open(sample_fog_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/process/image",
        files={"file": ("test_fog.png", file_bytes, "image/png")},
        data={"location": "Test-Zone-A"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    t = data["telemetry"]
    assert t["location"] == "Test-Zone-A"
    assert t["detected_condition"] == "fog"
    assert "Atmospheric Fog" in t["degradation_explanation"]
    assert len(t["airlight_A"]) == 3
    assert t["crowd_count"] > 0
    assert 0.05 <= t["visibility_index"] <= 1.0
    assert 0.0 <= t["risk_score"] <= 1.0
    assert t["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert t["alert_status"] != ""
    assert t["latency_ms"] > 0
    assert t["fps"] > 0

    v = data["visuals"]
    for k in ["original_frame", "transmission_map", "restored_frame", "density_heatmap"]:
        assert k in v
        assert v[k].startswith("data:image/jpeg;base64,")

    print(f"\n[PASS] Image Upload Endpoint verified: Count={t['crowd_count']}, Vis={t['visibility_index']}, Risk={t['risk_level']} ({t['risk_score']})")

def test_04_preset_processing_all_conditions():
    """Verify all 4 weather condition presets (Fog, Rain, Glare, Clear)."""
    presets = {
        "fog": "fog",
        "rain": "rain",
        "glare": "glare",
        "clear": "clear"
    }
    for preset_name, expected_cond in presets.items():
        res = client.post(f"/api/process/preset?name={preset_name}")
        assert res.status_code == 200, f"Preset {preset_name} failed: {res.text}"
        data = res.json()
        assert data["status"] == "success"
        cond = data["telemetry"]["detected_condition"]
        assert cond == expected_cond, f"Expected {expected_cond} for preset {preset_name}, got {cond}"
        assert data["telemetry"]["crowd_count"] > 0
        print(f"  Preset '{preset_name}': detected={cond}, count={data['telemetry']['crowd_count']:.1f}, latency={data['telemetry']['latency_ms']} ms")
    print("[PASS] All 4 Weather Condition Presets verified.")

def test_05_dynamic_risk_calculation():
    """Verify deterministic risk calculation under varying capacity thresholds."""
    # Test high risk: small capacity (count is ~5.0 for fog preset)
    client.post("/api/config", json={"max_crowd_capacity": 4.0, "weather_multipliers": {"fog": 1.25}})
    res = client.post("/api/process/preset?name=fog")
    assert res.status_code == 200
    t = res.json()["telemetry"]
    # With count = 5.0 and cap=4.0, risk score should clip to 1.0 and level should be CRITICAL
    assert t["risk_score"] == 1.0
    assert t["risk_level"] == "CRITICAL"
    assert t["alert_status"] == "CAPACITY THRESHOLD EXCEEDED"

    # Test low risk: large capacity
    client.post("/api/config", json={"max_crowd_capacity": 500.0, "weather_multipliers": {"fog": 1.0}})
    res_low = client.post("/api/process/preset?name=fog")
    t_low = res_low.json()["telemetry"]
    assert t_low["risk_level"] in ["LOW", "MEDIUM"]
    print(f"\n[PASS] Dynamic Risk Logic verified: High cap risk={t_low['risk_score']} vs Low cap risk={t['risk_score']}")

def test_06_csv_event_logging_and_export():
    """Verify that surveillance events are appended to CSV with the exact schema."""
    # Retrieve logs via API
    res = client.get("/api/logs?limit=10")
    assert res.status_code == 200
    logs = res.json()["logs"]
    assert len(logs) > 0, "Expected at least 1 log entry in audit file"

    last_log = logs[-1]
    expected_fields = ["timestamp", "location", "detected_condition", "crowd_count", "visibility_index", "risk_score", "alert_status"]
    for field in expected_fields:
        assert field in last_log, f"Missing required column '{field}' in log"

    # Verify CSV file download endpoint
    res_csv = client.get("/api/logs/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    csv_content = res_csv.text
    lines = [line.strip() for line in csv_content.strip().split("\n") if line.strip()]
    assert len(lines) >= 2  # Header + at least one row
    header = lines[0].split(",")
    assert header == expected_fields
    print(f"\n[PASS] CSV Audit Logger and Exporter verified with {len(lines) - 1} records.")

def test_07_dashboard_html_render():
    """Verify that the dashboard HTML loads and contains key DOM elements."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Adverse Weather Dynamic De-Hazing & Crowd Counting Pipeline" in html
    assert "Quad-Stream" in html or "quad-grid" in html
    assert "metric-count" in html
    assert "CSRNet" in html
    print("\n[PASS] Dashboard HTML Template rendering verified.")

def test_08_video_periodic_processing():
    """Verify periodic video sampling and dual-panel annotated MP4 rendering."""
    import tempfile
    import cv2
    import numpy as np

    temp_vid = os.path.join(tempfile.gettempdir(), "test_surveillance_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(temp_vid, fourcc, 5.0, (320, 240))
    for f in range(6):
        # Synthetic frame with simulated pedestrians
        fr = np.full((240, 320, 3), 150 + f * 5, dtype=np.uint8)
        cv2.circle(fr, (100 + f * 10, 120), 12, (50, 50, 50), -1)
        vw.write(fr)
    vw.release()

    with open(temp_vid, "rb") as f:
        vid_bytes = f.read()
    if os.path.exists(temp_vid):
        os.remove(temp_vid)

    response = client.post(
        "/api/process/video",
        files={"file": ("test_vid.mp4", vid_bytes, "video/mp4")},
        data={"sampling_interval": "2", "location": "Video-Test-Concourse"}
    )
    assert response.status_code == 200, f"Video processing failed: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert data["processed_frames"] >= 2
    assert "video_url" in data
    assert len(data["events_summary"]) >= 2
    
    # Check that output video file exists on disk
    out_rel = data["video_url"].lstrip("/")
    out_full = os.path.join(BASE_DIR, out_rel)
    assert os.path.exists(out_full), f"Output video was not saved at {out_full}"
    assert os.path.getsize(out_full) > 0
    print(f"\n[PASS] Video Periodic Processing verified: {data['processed_frames']} frames processed -> {data['video_url']}")

def test_09_video_preset_processing():
    """Verify pre-packaged video preset surveillance execution."""
    res = client.post("/api/process/video-preset?name=fog&sampling_interval=5")
    assert res.status_code == 200, f"Video preset failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert data["processed_frames"] >= 2
    assert "video_url" in data
    assert os.path.exists(os.path.join(BASE_DIR, data["video_url"].lstrip("/")))
    print(f"\n[PASS] Pre-Packaged Video Preset verified: {data['processed_frames']} frames processed -> {data['video_url']}")

def test_10_empty_bridge_zero_detections():
    """Verify that an empty bridge with no people yields 0 count and 0 bounding boxes."""
    res = client.post("/api/process/preset?name=empty")
    assert res.status_code == 200, f"Empty preset failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    t = data["telemetry"]
    assert t["crowd_count"] == 0.0, f"Expected 0.0 count on empty bridge, got {t['crowd_count']}"
    assert t["risk_level"] == "LOW"
    assert t["alert_status"] == "NORMAL CAPACITY"
    assert len(t.get("bounding_boxes", [])) == 0, f"Expected 0 bounding boxes on empty bridge, got {len(t.get('bounding_boxes', []))}"
    print(f"\n[PASS] Empty Control Scene verified: Count = {t['crowd_count']}, Boxes = {len(t.get('bounding_boxes', []))}")

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING BACKEND TEST SUITE")
    print("=" * 70)
    test_01_health_and_model_loading()
    test_02_configuration_crud()
    test_03_image_upload_processing()
    test_04_preset_processing_all_conditions()
    test_05_dynamic_risk_calculation()
    test_06_csv_event_logging_and_export()
    test_07_dashboard_html_render()
    test_08_video_periodic_processing()
    test_09_video_preset_processing()
    test_10_empty_bridge_zero_detections()
    print("=" * 70)
    print("ALL BACKEND TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
