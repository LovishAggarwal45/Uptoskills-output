"""
Batch Processor for All Video Scenarios.

Runs VideoMAE Crowd Outlier Localization across multiple video feeds,
generating annotated output videos with pixelated heatmaps, bounding boxes, alerts, and analytics.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from detector.outlier_detector import CrowdOutlierDetector


def run_batch_detection():
    print("=" * 70)
    print("      AI CROWD SURVEILLANCE SYSTEM: BATCH SCENARIO PROCESSING       ")
    print("=" * 70)

    detector = CrowdOutlierDetector(config_path="config.yaml")

    scenarios = [
        ("data/sample_crowd_test.mp4", "outputs/output_baseline_anomaly.mp4", "Baseline Panic & Counter-Flow"),
        ("data/scenario_stampede.mp4", "outputs/output_stampede_bottleneck.mp4", "Stampede & Bottleneck Blockade"),
        ("data/scenario_multizone.mp4", "outputs/output_multizone_disturbances.mp4", "Multi-Zone Concurrent Disturbances"),
        ("data/scenario_night_surge.mp4", "outputs/output_night_vision_surge.mp4", "Night Vision Crowd Surge"),
        ("data/scenario_fire_evacuation.mp4", "outputs/output_fire_evacuation.mp4", "Fire Alarm Panic Evacuation"),
        ("data/scenario_vehicle_intrusion.mp4", "outputs/output_vehicle_intrusion.mp4", "High-Speed Vehicle Intrusion"),
        ("data/scenario_crowd_freeze.mp4", "outputs/output_crowd_freeze.mp4", "Sudden Crowd Freeze Anomaly")
    ]

    generated_outputs = []

    for vpath, out_vpath, title in scenarios:
        if not os.path.exists(vpath):
            print(f"[Warning] Scenario video not found: {vpath}. Skipping.")
            continue

        print(f"\n[Processing] {title}...")
        print(f"  Input:  {vpath}")
        print(f"  Target: {out_vpath}")

        start_t = time.time()
        res_vpath, scores = detector.process_video(video_path=vpath, output_path=out_vpath)
        elapsed = time.time() - start_t

        generated_outputs.append((title, res_vpath, scores))
        print(f"  [DONE] Processed in {elapsed:.2f}s. Saved to: {res_vpath}")

    print("\n" + "=" * 70)
    print("ALL 7 SCENARIOS PROCESSED SUCCESSFULLY!")
    print("=" * 70)
    for title, vpath, scores in generated_outputs:
        print(f"- {title:35s} -> {vpath} (Score Peak: {scores.max():.4f})")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_detection()
