"""
Violation Logic & Persistence Threshold Module
Implements wrong-way detection and lane straddling detection with
strict temporal persistence, minimum displacement validation, and
single-alert enforcement per vehicle to prevent false alarms and duplicate counts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

class ViolationType(str, Enum):
    WRONG_WAY = "Wrong-Way Driving"
    LANE_STRADDLING = "Lane Discipline Violation (Straddling)"

@dataclass
class ViolationEvent:
    violation_id: str
    vehicle_id: int
    vehicle_type: str
    violation_type: ViolationType
    lane_id: Optional[int]
    lane_name: str
    start_frame: int
    start_timestamp: str
    confidence: float
    heading_deg: float
    deviation_deg: float
    end_frame: Optional[int] = None
    duration_frames: int = 0
    duration_seconds: float = 0.0
    clip_path: str = ""
    is_active: bool = True

class VehicleViolationState:
    def __init__(self, vehicle_id: int):
        self.vehicle_id = vehicle_id
        
        # Wrong way state
        self.wrong_way_counter: int = 0
        self.wrong_way_confirmed: bool = False
        self.wrong_way_alerted: bool = False
        self.wrong_way_event: Optional[ViolationEvent] = None
        self.wrong_way_total_displacement: float = 0.0
        
        # Lane straddling state
        self.straddle_counter: int = 0
        self.straddle_confirmed: bool = False
        self.straddle_alerted: bool = False
        self.straddle_event: Optional[ViolationEvent] = None

class ViolationDetector:
    def __init__(
        self,
        wrong_way_persistence_frames: int = 15,
        straddle_persistence_frames: int = 25,
        min_total_displacement_px: float = 30.0,
        fps: float = 30.0
    ):
        self.wrong_way_persistence = wrong_way_persistence_frames
        self.straddle_persistence = straddle_persistence_frames
        self.min_total_displacement = min_total_displacement_px
        self.fps = fps if fps > 0 else 30.0
        
        self.states: Dict[int, VehicleViolationState] = {}
        self.active_violations: List[ViolationEvent] = []
        self.completed_violations: List[ViolationEvent] = []
        self.violation_counter: int = 1

    def _get_timestamp_str(self, frame_idx: int) -> str:
        total_seconds = frame_idx / self.fps
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        millis = int((total_seconds - int(total_seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

    def update_vehicle(
        self,
        vehicle_id: int,
        vehicle_type: str,
        frame_idx: int,
        primary_lane_id: Optional[int],
        lane_name: str,
        is_moving: bool,
        heading_deg: Optional[float],
        is_wrong_way_instant: bool,
        deviation_deg: float,
        is_straddling_instant: bool,
        confidence: float,
        displacement_px: float = 0.0
    ) -> Tuple[List[ViolationEvent], List[str]]:
        """
        Updates violation status for a vehicle in current frame.
        Guarantees that each vehicle ID can only trigger an alert ONCE.
        """
        if vehicle_id not in self.states:
            self.states[vehicle_id] = VehicleViolationState(vehicle_id)

        state = self.states[vehicle_id]
        newly_triggered: List[ViolationEvent] = []
        hud_alerts: List[str] = []

        # ----------------- 1. WRONG-WAY DRIVING LOGIC -----------------
        if is_moving and is_wrong_way_instant:
            state.wrong_way_counter += 1
            state.wrong_way_total_displacement += displacement_px
            
            # Check persistence threshold AND minimum real distance traveled
            if (state.wrong_way_counter >= self.wrong_way_persistence and
                state.wrong_way_total_displacement >= self.min_total_displacement):
                
                state.wrong_way_confirmed = True
                
                # ENFORCE SINGLE-ALERT RULE: Only fire alert once per vehicle ID!
                if not state.wrong_way_alerted:
                    event_id = f"VIO-{self.violation_counter:04d}"
                    self.violation_counter += 1
                    event = ViolationEvent(
                        violation_id=event_id,
                        vehicle_id=vehicle_id,
                        vehicle_type=vehicle_type,
                        violation_type=ViolationType.WRONG_WAY,
                        lane_id=primary_lane_id,
                        lane_name=lane_name,
                        start_frame=frame_idx,
                        start_timestamp=self._get_timestamp_str(frame_idx),
                        confidence=confidence,
                        heading_deg=heading_deg if heading_deg is not None else 0.0,
                        deviation_deg=deviation_deg,
                        is_active=True
                    )
                    state.wrong_way_alerted = True
                    state.wrong_way_event = event
                    self.active_violations.append(event)
                    newly_triggered.append(event)
        else:
            # Decay counter slowly if not actively violating, but NEVER reset alerted flag!
            state.wrong_way_counter = max(0, state.wrong_way_counter - 1)

        # ----------------- 2. LANE STRADDLING LOGIC -----------------
        if is_straddling_instant and is_moving:
            state.straddle_counter += 1
            
            if state.straddle_counter >= self.straddle_persistence:
                state.straddle_confirmed = True
                
                # ENFORCE SINGLE-ALERT RULE
                if not state.straddle_alerted:
                    event_id = f"VIO-{self.violation_counter:04d}"
                    self.violation_counter += 1
                    event = ViolationEvent(
                        violation_id=event_id,
                        vehicle_id=vehicle_id,
                        vehicle_type=vehicle_type,
                        violation_type=ViolationType.LANE_STRADDLING,
                        lane_id=primary_lane_id,
                        lane_name=lane_name,
                        start_frame=frame_idx,
                        start_timestamp=self._get_timestamp_str(frame_idx),
                        confidence=confidence,
                        heading_deg=heading_deg if heading_deg is not None else 0.0,
                        deviation_deg=deviation_deg,
                        is_active=True
                    )
                    state.straddle_alerted = True
                    state.straddle_event = event
                    self.active_violations.append(event)
                    newly_triggered.append(event)
        else:
            state.straddle_counter = max(0, state.straddle_counter - 1)

        if state.wrong_way_confirmed:
            hud_alerts.append(f"WRONG WAY: Vehicle #{vehicle_id}")
        if state.straddle_confirmed:
            hud_alerts.append(f"LANE STRADDLE: Vehicle #{vehicle_id}")

        return newly_triggered, hud_alerts

    def _close_violation(self, event: ViolationEvent, frame_idx: int):
        event.is_active = False
        event.end_frame = frame_idx
        event.duration_frames = max(1, frame_idx - event.start_frame)
        event.duration_seconds = event.duration_frames / self.fps
        if event in self.active_violations:
            self.active_violations.remove(event)
        if event not in self.completed_violations:
            self.completed_violations.append(event)

    def close_all_active(self, final_frame_idx: int):
        """Closes all ongoing active violations at the end of the video stream."""
        for event in list(self.active_violations):
            self._close_violation(event, final_frame_idx)

    def get_vehicle_violation_status(self, vehicle_id: int) -> Tuple[bool, bool]:
        """Returns (is_confirmed_wrong_way, is_confirmed_straddling)."""
        if vehicle_id not in self.states:
            return False, False
        st = self.states[vehicle_id]
        return st.wrong_way_confirmed, st.straddle_confirmed

    def prune_stale_vehicles(self, active_track_ids: List[int], current_frame: int):
        active_set = set(active_track_ids)
        for vid in list(self.states.keys()):
            if vid not in active_set:
                st = self.states[vid]
                if st.wrong_way_event and st.wrong_way_event.is_active:
                    self._close_violation(st.wrong_way_event, current_frame)
                if st.straddle_event and st.straddle_event.is_active:
                    self._close_violation(st.straddle_event, current_frame)
                del self.states[vid]
