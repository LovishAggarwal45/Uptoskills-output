"""
Trajectory & Direction Estimation Module
Maintains vehicle movement histories, applies trajectory smoothing,
and computes displacement vectors, velocities, and heading angles.
"""

from collections import deque
import math
from typing import Dict, List, Optional, Tuple

class VehicleTrajectory:
    def __init__(self, track_id: int, max_history: int = 30):
        self.track_id = track_id
        self.max_history = max_history
        # Raw positions: list of (frame_idx, (x, y))
        self.positions: deque = deque(maxlen=max_history)
        self.smoothed_positions: deque = deque(maxlen=max_history)
        self.last_heading_deg: Optional[float] = None
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_speed_px_per_frame: float = 0.0
        self.is_moving: bool = False
        self.frames_since_update: int = 0
        self.initial_point: Optional[Tuple[float, float]] = None
        self.total_displacement: float = 0.0

    def update(self, frame_idx: int, point: Tuple[float, float], smoothing_window: int = 5):
        if self.initial_point is None:
            self.initial_point = point
        self.positions.append((frame_idx, point))
        self.frames_since_update = 0

        # Compute total net displacement from initial spawn point
        self.total_displacement = math.hypot(point[0] - self.initial_point[0], point[1] - self.initial_point[1])

        # Moving average smoothing
        pts = [p for _, p in self.positions]
        k = min(len(pts), smoothing_window)
        smooth_x = sum(p[0] for p in pts[-k:]) / k
        smooth_y = sum(p[1] for p in pts[-k:]) / k
        self.smoothed_positions.append((smooth_x, smooth_y))

    def estimate_direction(
        self, window: int = 10, min_displacement: float = 6.0
    ) -> Tuple[Optional[float], Tuple[float, float], float, bool]:
        """
        Estimates the heading angle and displacement vector over the recent window.
        :param window: Number of recent positions to consider.
        :param min_displacement: Minimum px displacement needed to consider the vehicle in motion.
        :return: (heading_deg, (dx, dy), speed_px, is_moving)
                 heading_deg is in range [0, 360), where:
                 0 = East (Right), 90 = South (Down), 180 = West (Left), 270 = North (Up)
        """
        if len(self.smoothed_positions) < 3:
            return self.last_heading_deg, self.last_velocity, self.last_speed_px_per_frame, False

        recent = list(self.smoothed_positions)[-min(len(self.smoothed_positions), window):]
        start_pt = recent[0]
        end_pt = recent[-1]

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        dist = math.hypot(dx, dy)

        if dist >= min_displacement:
            # Heading in degrees [0, 360)
            heading_rad = math.atan2(dy, dx)
            heading_deg = math.degrees(heading_rad) % 360.0
            
            self.last_heading_deg = heading_deg
            self.last_velocity = (dx / len(recent), dy / len(recent))
            self.last_speed_px_per_frame = dist / len(recent)
            self.is_moving = True
            return heading_deg, self.last_velocity, self.last_speed_px_per_frame, True
        else:
            self.is_moving = False
            return self.last_heading_deg, (0.0, 0.0), 0.0, False

    def get_trail(self) -> List[Tuple[int, int]]:
        """Returns smoothed positions as integer tuples for rendering trails."""
        return [(int(x), int(y)) for x, y in self.smoothed_positions]


class TrajectoryEstimator:
    def __init__(self, max_history: int = 30, direction_window: int = 10, min_displacement: float = 6.0):
        self.max_history = max_history
        self.direction_window = direction_window
        self.min_displacement = min_displacement
        self.trajectories: Dict[int, VehicleTrajectory] = {}

    def update_vehicle(
        self, track_id: int, frame_idx: int, centroid: Tuple[float, float]
    ) -> Tuple[Optional[float], Tuple[float, float], float, bool]:
        """
        Updates the trajectory for the given vehicle and estimates its direction.
        :return: (heading_deg, (dx, dy), speed_px, is_moving)
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = VehicleTrajectory(track_id, max_history=self.max_history)

        traj = self.trajectories[track_id]
        traj.update(frame_idx, centroid)
        return traj.estimate_direction(window=self.direction_window, min_displacement=self.min_displacement)

    def get_trajectory(self, track_id: int) -> Optional[VehicleTrajectory]:
        return self.trajectories.get(track_id)

    def prune_stale_tracks(self, active_track_ids: List[int], max_idle_frames: int = 30):
        """Removes trajectories for vehicles that have left the scene."""
        active_set = set(active_track_ids)
        stale_ids = []
        for tid, traj in self.trajectories.items():
            if tid not in active_set:
                traj.frames_since_update += 1
                if traj.frames_since_update > max_idle_frames:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self.trajectories[tid]
