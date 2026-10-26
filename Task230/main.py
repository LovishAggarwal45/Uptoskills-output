"""
Main Pipeline Runner — Wrong-Way Driving & Lane Discipline Violation Detection
Coordinates:
  1. Video Input & Frame Extraction
  2. Flow Auto-Calibration Phase  (first N frames → learns dominant headings per lane)
  3. Vehicle Detection & Tracking (YOLOv8n + ByteTrack)
  4. Trajectory & Direction Estimation
  5. Lane Assignment & Learned Flow Direction Map
  6. Violation Logic with Persistence Thresholds
  7. Outputs: Annotated Video | Violation Clips with Alerts | CSV Log
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DEFAULT_LANES_720P, LaneDefinition, SystemConfig
from clip_recorder import ViolationClipRecorder
from csv_logger import CSVLogger
from detector_tracker import VehicleDetectorTracker
from lane_manager import LaneManager
from trajectory_estimator import TrajectoryEstimator
from violation_detector import ViolationDetector
from visualizer import Visualizer


# ─────────────────────────────────────────────────────────────
# Helper: Circular mean of heading angles
# ─────────────────────────────────────────────────────────────
def _circular_mean(angles_deg: List[float]) -> float:
    if not angles_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


# ─────────────────────────────────────────────────────────────
# Phase 1: Auto-Calibration — learn real dominant flow per lane
# ─────────────────────────────────────────────────────────────
def auto_calibrate_flow(
    cap: cv2.VideoCapture,
    detector: VehicleDetectorTracker,
    lane_mgr: LaneManager,
    config: SystemConfig,
    calibration_frames: int = 120
):
    """
    Scans the first `calibration_frames` frames. For each tracked vehicle whose
    trajectory covers at least 8 frames, computes the heading and pushes it into
    the learned flow map for the lane it belongs to.
    Rewinds the capture to frame 0 when done.
    """
    print(f"[*] Auto-calibrating lane flow directions from first {calibration_frames} frames...")
    local_tracks: Dict[int, List[Tuple[int, float, float]]] = {}  # id -> [(fidx, cx, cy)]

    for fidx in range(1, calibration_frames + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        vehicles = detector.process_frame(frame)
        for v in vehicles:
            if v.track_id not in local_tracks:
                local_tracks[v.track_id] = []
            local_tracks[v.track_id].append((fidx, v.centroid[0], v.centroid[1]))

    # Compute heading per track and accumulate into lane manager
    calibrated_count = 0
    for tid, pts in local_tracks.items():
        if len(pts) < 8:
            continue
        # Use first and last 3-point averages for robustness
        early = pts[:3]
        late = pts[-3:]
        x1 = sum(p[1] for p in early) / len(early)
        y1 = sum(p[2] for p in early) / len(early)
        x2 = sum(p[1] for p in late) / len(late)
        y2 = sum(p[2] for p in late) / len(late)
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < config.min_displacement_px * 3:
            continue

        heading = math.degrees(math.atan2(dy, dx)) % 360.0
        avg_x = (x1 + x2) / 2.0
        avg_y = (y1 + y2) / 2.0
        mid_bbox = (avg_x - 40, avg_y - 20, avg_x + 40, avg_y + 20)
        primary_lane_id, _ = lane_mgr.assign_vehicle_to_lane(mid_bbox)
        lane_mgr.update_learned_flow(
            lane_id=primary_lane_id,
            heading_deg=heading,
            confidence=1.0,
            centroid=(avg_x, avg_y),
            vehicle_id=tid
        )
        calibrated_count += 1

    # Print calibration results
    print(f"[*] Calibration used {calibrated_count} tracks.")
    for lane_id in lane_mgr.get_lane_ids():
        learned = lane_mgr.get_flow_heading(lane_id)
        lane = lane_mgr.get_lane(lane_id)
        print(f"    Lane {lane_id} ({lane.name if lane else '?'}): learned flow = {learned:.1f}°")

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[*] Calibration complete. Rewinding to frame 0.")


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline(
    video_source: str,
    output_video_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    clips_dir: Optional[str] = None,
    display: bool = False,
    max_frames: Optional[int] = None,
    skip_calibration: bool = False,
):
    print("=" * 70)
    print("  AI TRAFFIC MONITORING: WRONG-WAY & LANE DISCIPLINE DETECTOR  ")
    print("=" * 70)

    config = SystemConfig()
    paths = config.get_output_paths()
    output_video_path = output_video_path or paths["annotated_video"]
    output_csv_path = output_csv_path or paths["csv_path"]
    clips_dir = clips_dir or paths["clips_dir"]

    print(f"[*] Input Video Source : {video_source}")
    print(f"[*] Annotated Output   : {output_video_path}")
    print(f"[*] Violations CSV     : {output_csv_path}")
    print(f"[*] Violation Clips    : {clips_dir}")

    # ── 1. Open Video ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[!] Error: Cannot open '{video_source}'.", file=sys.stderr)
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Video: {width}x{height}  FPS={fps:.1f}  Frames={total_frames}")

    # ── Scale default lane polygons to actual resolution ───────
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

    # ── Init modules ───────────────────────────────────────────
    print("[*] Loading YOLOv8 detector …")
    detector = VehicleDetectorTracker(
        model_path=config.model_name,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        target_classes=config.vehicle_classes,
    )

    lane_mgr       = LaneManager(lanes=lanes, frame_width=width, frame_height=height, auto_learn_flow=True)
    traj_estimator = TrajectoryEstimator(
        max_history=config.trajectory_history_length,
        direction_window=config.direction_window_frames,
        min_displacement=config.min_displacement_px,
    )
    viol_detector  = ViolationDetector(
        wrong_way_persistence_frames=config.wrong_way_persistence_frames,
        straddle_persistence_frames=config.straddle_persistence_frames,
        min_total_displacement_px=config.min_total_displacement_px,
        fps=fps,
    )
    clip_recorder  = ViolationClipRecorder(
        output_dir=clips_dir,
        fps=fps,
        pre_roll_frames=config.pre_roll_frames,
        post_roll_frames=config.post_roll_frames,
    )
    csv_logger     = CSVLogger(output_csv_path)
    visualizer     = Visualizer(lane_manager=lane_mgr)

    # ── 2. Auto-Calibration Phase ──────────────────────────────
    if not skip_calibration and isinstance(video_source, str) and os.path.isfile(video_source):
        auto_calibrate_flow(cap, detector, lane_mgr, config, calibration_frames=120)
    else:
        print("[*] Skipping calibration (live stream or disabled).")

    # ── Output writer ──────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx      = 0
    start_time     = time.time()
    total_ww       = 0
    total_st       = 0

    print("[*] Starting main detection pass …")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # ── 3. Detect & Track ─────────────────────────────
            tracked_vehicles = detector.process_frame(frame)
            active_ids       = [v.track_id for v in tracked_vehicles]

            annotated = frame.copy()
            annotated = visualizer.draw_lanes(annotated)

            current_alerts: List[str] = []

            for vehicle in tracked_vehicles:
                # ── 4. Trajectory & Direction ──────────────────
                heading_deg, velocity, speed, is_moving = traj_estimator.update_vehicle(
                    track_id=vehicle.track_id,
                    frame_idx=frame_idx,
                    centroid=vehicle.centroid,
                )

                # ── 5. Lane Assignment ─────────────────────────
                primary_lane_id, overlaps = lane_mgr.assign_vehicle_to_lane(vehicle.bbox)
                primary_lane = lane_mgr.get_lane(primary_lane_id) if primary_lane_id else None
                lane_name    = primary_lane.name if primary_lane else "Off-Lane"

                is_straddling, _ = lane_mgr.check_straddling(
                    overlaps, min_ratio=config.straddle_overlap_min_ratio
                )

                # ── 6. Wrong-way Check ────────────────────────
                is_wrong_way_instant = False
                deviation_deg        = 0.0
                traj = traj_estimator.get_trajectory(vehicle.track_id)
                tot_disp = traj.total_displacement if traj else 0.0

                if heading_deg is not None and is_moving and tot_disp >= 25.0:
                    is_wrong_way_instant, deviation_deg = lane_mgr.is_wrong_way(
                        lane_id=primary_lane_id,
                        vehicle_heading_deg=heading_deg,
                        centroid=vehicle.centroid,
                        threshold_deg=config.wrong_way_angle_threshold,
                    )
                    if not is_wrong_way_instant and not is_straddling:
                        lane_mgr.update_learned_flow(
                            lane_id=primary_lane_id,
                            heading_deg=heading_deg,
                            confidence=vehicle.confidence,
                            centroid=vehicle.centroid,
                            vehicle_id=vehicle.track_id
                        )

                # ── 7. Violation Logic & Persistence ──────────
                new_violations, hud_alerts = viol_detector.update_vehicle(
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
                    displacement_px=tot_disp,
                )

                current_alerts.extend(hud_alerts)

                for ev in new_violations:
                    print(f"  [!] {ev.violation_type.value} | Veh#{ev.vehicle_id}"
                          f" | {ev.lane_name} | t={ev.start_timestamp}"
                          f" | conf={ev.confidence:.2f}")
                    clip_recorder.trigger_violation_clip(ev)
                    csv_logger.log_violation(ev)
                    if "Wrong-Way" in ev.violation_type.value:
                        total_ww += 1
                    else:
                        total_st += 1

                # ── Draw per-vehicle annotation ────────────────
                is_confirmed_ww, is_confirmed_st = viol_detector.get_vehicle_violation_status(
                    vehicle.track_id
                )
                traj = traj_estimator.get_trajectory(vehicle.track_id)
                trail = traj.get_trail() if traj else []

                visualizer.draw_vehicle(
                    frame=annotated,
                    vehicle=vehicle,
                    lane_name=lane_name,
                    heading_deg=heading_deg,
                    is_moving=is_moving,
                    is_wrong_way=is_confirmed_ww,
                    is_straddling=is_confirmed_st,
                    trail=trail,
                )

            # ── Housekeeping ───────────────────────────────────
            traj_estimator.prune_stale_tracks(active_ids)
            viol_detector.prune_stale_vehicles(active_ids, frame_idx)

            # ── HUD ───────────────────────────────────────────
            visualizer.draw_hud(
                frame=annotated,
                frame_idx=frame_idx,
                fps=fps,
                active_vehicles_count=len(tracked_vehicles),
                wrong_way_count=total_ww,
                straddle_count=total_st,
                active_alerts=list(set(current_alerts)),
            )

            # ── Clip Buffer & Write ───────────────────────────
            clip_recorder.buffer_frame(frame_idx, annotated)
            writer.write(annotated)

            if display:
                cv2.imshow("Traffic Monitor [q=quit]", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Stopped by user.")
                    break

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed  = time.time() - start_time
                proc_fps = frame_idx / max(0.001, elapsed)
                pct      = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[*] Frame {frame_idx}/{total_frames} ({pct:.0f}%)"
                      f"  Speed={proc_fps:.1f} FPS"
                      f"  Violations: WW={total_ww}  Straddle={total_st}")

    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
        viol_detector.close_all_active(frame_idx)
        clip_recorder.finalize_all()

    elapsed = time.time() - start_time
    print("=" * 70)
    print("  PROCESSING COMPLETE  ")
    print("=" * 70)
    print(f"  Frames processed        : {frame_idx}")
    print(f"  Total time              : {elapsed:.1f}s  ({frame_idx/max(0.01,elapsed):.1f} FPS)")
    print(f"  Wrong-Way violations    : {total_ww}")
    print(f"  Lane-Straddle violations: {total_st}")
    print(f"  Annotated video         : {output_video_path}")
    print(f"  Violations CSV          : {output_csv_path}")
    print(f"  Violation clips dir     : {clips_dir}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Wrong-Way Driving & Lane Discipline Violation Detection"
    )
    parser.add_argument("--video",          default="sample_traffic.mp4",
                        help="Input video / stream URL / webcam index")
    parser.add_argument("--output-video",   default=None)
    parser.add_argument("--output-csv",     default=None)
    parser.add_argument("--clips-dir",      default=None)
    parser.add_argument("--display",        action="store_true",
                        help="Show live window")
    parser.add_argument("--max-frames",     type=int, default=None)
    parser.add_argument("--skip-calib",     action="store_true",
                        help="Skip auto-calibration phase")
    args = parser.parse_args()

    # Auto-generate synthetic video if requested source doesn't exist
    if not os.path.exists(args.video) and args.video == "sample_traffic.mp4":
        print(f"[*] '{args.video}' not found — generating synthetic demo video …")
        from generate_sample_video import generate_sample_traffic_video
        generate_sample_traffic_video(args.video, duration_seconds=20, fps=25)

    run_pipeline(
        video_source=args.video,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        clips_dir=args.clips_dir,
        display=args.display,
        max_frames=args.max_frames,
        skip_calibration=args.skip_calib,
    )


if __name__ == "__main__":
    main()
