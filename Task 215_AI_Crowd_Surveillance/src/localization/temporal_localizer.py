"""
Temporal Localization Engine for Long CCTV Surveillance Videos.
Implements 1D Gaussian temporal smoothing, hysteresis thresholding,
event boundary grouping, gap merging, and minimum duration filtering.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Any, Tuple, Optional

def format_timestamp(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.ms string."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"

class TemporalLocalizer:
    def __init__(
        self,
        threshold: float = 0.60,
        temporal_smoothing_kernel: int = 5,
        min_duration_seconds: float = 1.0,
        merge_gap_seconds: float = 2.0,
        classes: Optional[List[str]] = None,
        anomaly_class_ids: Optional[List[int]] = None
    ):
        self.threshold = threshold
        self.smoothing_kernel = temporal_smoothing_kernel
        self.min_duration = min_duration_seconds
        self.merge_gap = merge_gap_seconds
        self.classes = classes or ["Normal", "Fighting", "Panic_Dispersal"]
        self.anomaly_class_ids = anomaly_class_ids if anomaly_class_ids is not None else [1, 2]

    def smooth_scores(self, scores: np.ndarray) -> np.ndarray:
        """Applies 1D Gaussian temporal smoothing over segment anomaly scores."""
        if len(scores) <= 2 or self.smoothing_kernel <= 1:
            return scores
        sigma = max(0.6, self.smoothing_kernel / 4.0)
        return gaussian_filter1d(scores, sigma=sigma, mode="nearest")

    def localize_events(
        self,
        segment_scores: np.ndarray,
        segment_action_probs: np.ndarray,
        timestamps: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Localizes temporal events from segment anomaly scores and action distributions.
        Returns a list of structured event records.
        """
        if len(segment_scores) == 0:
            return []

        # 1. Temporal Smoothing
        smoothed = self.smooth_scores(segment_scores)

        # 2. Thresholding: identify binary candidate segments
        candidates = (smoothed >= self.threshold)

        # 3. Form contiguous intervals
        raw_intervals = []
        in_interval = False
        start_idx = 0

        for i, is_pos in enumerate(candidates):
            if is_pos and not in_interval:
                in_interval = True
                start_idx = i
            elif not is_pos and in_interval:
                in_interval = False
                raw_intervals.append((start_idx, i - 1))
        if in_interval:
            raw_intervals.append((start_idx, len(candidates) - 1))

        if not raw_intervals:
            return []

        # 4. Merge intervals separated by less than merge_gap_seconds
        merged_intervals = []
        curr_start_idx, curr_end_idx = raw_intervals[0]

        for next_start, next_end in raw_intervals[1:]:
            prev_end_time = timestamps[curr_end_idx][1]
            next_start_time = timestamps[next_start][0]

            if (next_start_time - prev_end_time) <= self.merge_gap:
                curr_end_idx = next_end
            else:
                merged_intervals.append((curr_start_idx, curr_end_idx))
                curr_start_idx, curr_end_idx = next_start, next_end
        merged_intervals.append((curr_start_idx, curr_end_idx))

        # 5. Filter by min_duration & resolve action class and confidence
        events = []
        event_id = 1

        for s_idx, e_idx in merged_intervals:
            t_start = timestamps[s_idx][0]
            t_end = timestamps[e_idx][1]
            duration = t_end - t_start

            if duration < self.min_duration:
                continue

            interval_scores = smoothed[s_idx:e_idx + 1]
            peak_conf = float(np.max(interval_scores))
            mean_conf = float(np.mean(interval_scores))

            interval_probs = segment_action_probs[s_idx:e_idx + 1]
            mean_probs = np.mean(interval_probs, axis=0)
            
            anom_probs = mean_probs.copy()
            anom_probs[0] = -1.0 # exclude Normal from winning anomalous event
            best_class_id = int(np.argmax(anom_probs))
            predicted_action = self.classes[best_class_id]

            events.append({
                "event_id": event_id,
                "action": predicted_action,
                "class_id": best_class_id,
                "start_time": round(float(t_start), 2),
                "end_time": round(float(t_end), 2),
                "duration": round(float(duration), 2),
                "start_time_str": format_timestamp(t_start),
                "end_time_str": format_timestamp(t_end),
                "confidence": round(mean_conf, 4),
                "peak_confidence": round(peak_conf, 4),
                "start_segment": s_idx,
                "end_segment": e_idx
            })
            event_id += 1

        return events
