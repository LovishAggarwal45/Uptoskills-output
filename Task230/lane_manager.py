"""
Adaptive Spatial Flow Field & Lane Manager Module
Learns dominant traffic-flow directions across spatial grid cells to support
ANY custom camera angle, road layout, or uploaded video without hardcoding.
Filters false positives and supports both explicit lane polygons and grid flow fields.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, box, Point
from config import LaneDefinition

class SpatialFlowField:
    """
    Divides the frame into an adaptive grid of cells.
    Learns dominant traffic direction in each cell from passing vehicles.
    """
    def __init__(self, frame_width: int = 1280, frame_height: int = 720, cell_size: int = 80):
        self.width = frame_width
        self.height = frame_height
        self.cell_size = cell_size
        self.cols = max(1, math.ceil(frame_width / cell_size))
        self.rows = max(1, math.ceil(frame_height / cell_size))
        
        # Grid storage: (row, col) -> list of (dx, dy, vehicle_id)
        self.grid_vectors: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
        # Cached consensus angles: (row, col) -> (angle_deg, confidence)
        self.grid_consensus: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        c = int(np.clip(x // self.cell_size, 0, self.cols - 1))
        r = int(np.clip(y // self.cell_size, 0, self.rows - 1))
        return r, c

    def update_flow(self, x: float, y: float, heading_deg: float, vehicle_id: int):
        """Records a vehicle's motion vector in its corresponding grid cell."""
        r, c = self._get_cell_coords(x, y)
        cell = (r, c)
        if cell not in self.grid_vectors:
            self.grid_vectors[cell] = []

        # Avoid spamming multiple identical points from the same vehicle in the same cell
        existing = [v for v in self.grid_vectors[cell] if v[2] == vehicle_id]
        if len(existing) >= 2:
            return

        rad = math.radians(heading_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)
        self.grid_vectors[cell].append((dx, dy, vehicle_id))

        # Keep rolling buffer of recent 50 observations per cell
        if len(self.grid_vectors[cell]) > 50:
            self.grid_vectors[cell].pop(0)

        # Invalidate cached consensus
        self._recompute_cell(r, c)

    def _recompute_cell(self, r: int, c: int):
        cell = (r, c)
        vectors = self.grid_vectors.get(cell, [])
        if len(vectors) < 2:
            return

        # Vector mean
        sum_dx = sum(v[0] for v in vectors)
        sum_dy = sum(v[1] for v in vectors)
        mag = math.hypot(sum_dx, sum_dy)
        if mag > 0.01:
            mean_angle = math.degrees(math.atan2(sum_dy, sum_dx)) % 360.0
            consistency = mag / len(vectors)
            conf = min(1.0, (len(vectors) / 5.0) * consistency)
            self.grid_consensus[cell] = (mean_angle, conf)

    def get_flow_at(self, x: float, y: float) -> Tuple[Optional[float], float]:
        """
        Returns (consensus_heading_deg, confidence) at (x, y).
        If cell has no direct data, checks 3x3 neighboring cells.
        """
        r, c = self._get_cell_coords(x, y)
        cell = (r, c)
        if cell in self.grid_consensus and self.grid_consensus[cell][1] >= 0.35:
            return self.grid_consensus[cell]

        # Check neighbor interpolation
        neighbor_angles = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                ncell = (nr, nc)
                if ncell in self.grid_consensus and self.grid_consensus[ncell][1] >= 0.4:
                    neighbor_angles.append(self.grid_consensus[ncell][0])

        if neighbor_angles:
            sin_s = sum(math.sin(math.radians(a)) for a in neighbor_angles)
            cos_s = sum(math.cos(math.radians(a)) for a in neighbor_angles)
            mean_a = math.degrees(math.atan2(sin_s, cos_s)) % 360.0
            return mean_a, 0.4

        return None, 0.0


class LaneManager:
    def __init__(
        self,
        lanes: Optional[List[LaneDefinition]] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        auto_learn_flow: bool = True
    ):
        self.auto_learn_flow = auto_learn_flow
        self.flow_field = SpatialFlowField(frame_width, frame_height)
        
        # Explicit lanes if defined
        self.lanes: Dict[int, LaneDefinition] = {}
        self.polygons: Dict[int, Polygon] = {}
        if lanes:
            for lane in lanes:
                self.lanes[lane.lane_id] = lane
                self.polygons[lane.lane_id] = Polygon(lane.polygon)

        # Learned flow heading angle (degrees) per explicit lane
        self.learned_headings: Dict[int, float] = {
            lane_id: lane.expected_heading_deg for lane_id, lane in self.lanes.items()
        }

    def get_lane_ids(self) -> List[int]:
        return list(self.lanes.keys())

    def get_lane(self, lane_id: int) -> Optional[LaneDefinition]:
        return self.lanes.get(lane_id)

    def assign_vehicle_to_lane(self, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[int], Dict[int, float]]:
        """
        Assigns vehicle to primary lane and calculates overlap ratio with all lanes.
        """
        if not self.polygons:
            return None, {}

        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        footprint_y1 = y2 - height * 0.40
        vehicle_box = box(x1, footprint_y1, x2, y2)
        vehicle_area = vehicle_box.area

        overlaps: Dict[int, float] = {}
        max_overlap_ratio = 0.0
        primary_lane_id: Optional[int] = None

        for lane_id, poly in self.polygons.items():
            if not poly.intersects(vehicle_box):
                continue
            inter_area = poly.intersection(vehicle_box).area
            ratio = inter_area / vehicle_area
            if ratio > 0.05:
                overlaps[lane_id] = ratio
                if ratio > max_overlap_ratio:
                    max_overlap_ratio = ratio
                    primary_lane_id = lane_id

        if primary_lane_id is None:
            footprint_point = Point((x1 + x2) / 2.0, y2)
            for lane_id, poly in self.polygons.items():
                if poly.contains(footprint_point):
                    primary_lane_id = lane_id
                    overlaps[lane_id] = 1.0
                    break

        return primary_lane_id, overlaps

    def check_straddling(
        self, overlaps: Dict[int, float], min_ratio: float = 0.20
    ) -> Tuple[bool, List[int]]:
        """
        Determines whether vehicle is straddling two or more lanes.
        """
        straddled = [lane_id for lane_id, ratio in overlaps.items() if ratio >= min_ratio]
        if len(straddled) >= 2:
            return True, straddled
        return False, []

    def update_learned_flow(
        self,
        lane_id: Optional[int],
        heading_deg: float,
        confidence: float = 1.0,
        centroid: Optional[Tuple[float, float]] = None,
        vehicle_id: int = 0
    ):
        """
        Updates both the spatial flow field and lane flow map.
        """
        if centroid is not None:
            self.flow_field.update_flow(centroid[0], centroid[1], heading_deg, vehicle_id)

        if lane_id is not None and lane_id in self.lanes and self.auto_learn_flow:
            current = self.learned_headings.get(lane_id, self.lanes[lane_id].expected_heading_deg)
            rad_curr = math.radians(current)
            rad_new = math.radians(heading_deg)
            dx = math.cos(rad_curr) * 0.95 + math.cos(rad_new) * 0.05
            dy = math.sin(rad_curr) * 0.95 + math.sin(rad_new) * 0.05
            self.learned_headings[lane_id] = math.degrees(math.atan2(dy, dx)) % 360.0

    def get_flow_heading(self, lane_id: Optional[int] = None, centroid: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Returns expected heading from spatial flow field or explicit lane."""
        if centroid is not None:
            field_heading, field_conf = self.flow_field.get_flow_at(centroid[0], centroid[1])
            if field_heading is not None and field_conf >= 0.35:
                return field_heading

        if lane_id is not None and lane_id in self.learned_headings:
            return self.learned_headings[lane_id]
        if lane_id is not None and lane_id in self.lanes:
            return self.lanes[lane_id].expected_heading_deg
        return None

    @staticmethod
    def calculate_angle_deviation(heading1_deg: float, heading2_deg: float) -> float:
        """
        Calculates acute angle difference between two headings [0, 180] deg.
        """
        return abs((heading1_deg - heading2_deg + 180.0) % 360.0 - 180.0)

    def is_wrong_way(
        self,
        lane_id: Optional[int],
        vehicle_heading_deg: float,
        centroid: Optional[Tuple[float, float]] = None,
        threshold_deg: float = 125.0
    ) -> Tuple[bool, float]:
        """
        Checks if vehicle heading opposes the local consensus flow.
        Requires sufficient flow confidence before raising a violation.
        """
        flow_heading = self.get_flow_heading(lane_id, centroid)
        if flow_heading is None:
            return False, 0.0

        deviation = self.calculate_angle_deviation(vehicle_heading_deg, flow_heading)
        is_wrong = deviation >= threshold_deg
        return is_wrong, deviation
