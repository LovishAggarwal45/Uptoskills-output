"""
Avenue Surveillance Ground-Truth Evaluation Engine.

Evaluates predictions on the CUHK Avenue benchmark using the actual pixel-level
ground-truth masks (testing_label_mask/*.mat).

Computes:
1. Frame-level metrics:
   - Confusion matrix (TP, FP, TN, FN)
   - Accuracy, Precision, Recall, F1
   - ROC-AUC and PR-AUC (Average Precision) from continuous anomaly scores
2. Spatial metrics:
   - Intersection over Union (IoU) per abnormal frame
   - Mean IoU (mIoU) over abnormal frames
   - Spatial Precision and Recall
   - Explicitly reports unavailable when model does not provide spatial predictions
3. Temporal metrics:
   - Extraction of ground-truth abnormal intervals
   - Extraction of predicted abnormal intervals
   - Temporal overlap (intersection) and temporal union
   - Temporal IoU (tIoU)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support
)


def load_avenue_ground_truth(gt_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads an Avenue ground-truth MAT file (e.g. '1_label.mat').

    Returns a dict with:
      - 'frame_count': int
      - 'frame_labels': np.ndarray of shape [N] (1 = abnormal, 0 = normal)
      - 'binary_masks': list of np.ndarray of shape [H, W] (uint8 binary)
      - 'abnormal_frame_indices': list of int
      - 'normal_frame_indices': list of int
    """
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth MAT file not found: {path}")

    mat = loadmat(str(path))
    if "volLabel" not in mat:
        raise KeyError(f"Expected key 'volLabel' in MAT file: {path}")

    vol_label = mat["volLabel"]
    if len(vol_label.shape) != 2 or vol_label.shape[0] != 1:
        raise ValueError(f"Unexpected volLabel shape {vol_label.shape} in {path}")

    num_frames = vol_label.shape[1]
    binary_masks = []
    frame_labels = np.zeros(num_frames, dtype=np.int32)

    for i in range(num_frames):
        mask_raw = vol_label[0, i]
        mask_binary = (mask_raw > 0).astype(np.uint8)
        binary_masks.append(mask_binary)
        if np.count_nonzero(mask_binary) > 0:
            frame_labels[i] = 1

    abnormal_indices = np.where(frame_labels == 1)[0].tolist()
    normal_indices = np.where(frame_labels == 0)[0].tolist()

    return {
        "gt_path": str(path.resolve()).replace("\\", "/"),
        "frame_count": num_frames,
        "frame_labels": frame_labels,
        "binary_masks": binary_masks,
        "abnormal_frame_indices": abnormal_indices,
        "normal_frame_indices": normal_indices,
        "abnormal_frame_count": len(abnormal_indices),
        "normal_frame_count": len(normal_indices)
    }


def extract_temporal_intervals(
    binary_sequence: np.ndarray,
    fps: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Converts a 1D binary sequence of frame labels into contiguous abnormal intervals.
    Returns list of interval dictionaries with frame bounds and timestamps.
    """
    intervals = []
    in_interval = False
    start_frame = 0

    N = len(binary_sequence)
    for i in range(N):
        is_pos = (binary_sequence[i] > 0)
        if is_pos and not in_interval:
            in_interval = True
            start_frame = i
        elif not is_pos and in_interval:
            in_interval = False
            end_frame = i - 1
            intervals.append({
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "frame_span": int(end_frame - start_frame + 1),
                "start_time": round(float(start_frame / fps), 3),
                "end_time": round(float((end_frame + 1) / fps), 3),
                "duration_seconds": round(float((end_frame - start_frame + 1) / fps), 3)
            })

    if in_interval:
        end_frame = N - 1
        intervals.append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "frame_span": int(end_frame - start_frame + 1),
            "start_time": round(float(start_frame / fps), 3),
            "end_time": round(float((end_frame + 1) / fps), 3),
            "duration_seconds": round(float((end_frame - start_frame + 1) / fps), 3)
        })

    return intervals


def compute_temporal_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    fps: float = 25.0
) -> Dict[str, Any]:
    """
    Computes temporal intervals, overlap, and temporal Intersection over Union (tIoU).
    """
    gt_intervals = extract_temporal_intervals(gt_labels, fps=fps)
    pred_intervals = extract_temporal_intervals(pred_labels, fps=fps)

    # Frame-wise intersection and union
    intersection_frames = int(np.sum((gt_labels == 1) & (pred_labels == 1)))
    union_frames = int(np.sum((gt_labels == 1) | (pred_labels == 1)))

    if union_frames > 0:
        temporal_iou = float(intersection_frames / union_frames)
    else:
        # Both ground truth and predictions have 0 abnormal frames -> perfect alignment
        temporal_iou = 1.0 if len(gt_intervals) == 0 and len(pred_intervals) == 0 else 0.0

    # Detailed interval overlaps
    interval_overlaps = []
    for g_idx, g in enumerate(gt_intervals):
        g_start, g_end = g["start_frame"], g["end_frame"]
        matched_preds = []
        for p_idx, p in enumerate(pred_intervals):
            p_start, p_end = p["start_frame"], p["end_frame"]
            inter_start = max(g_start, p_start)
            inter_end = min(g_end, p_end)
            if inter_end >= inter_start:
                inter_len = inter_end - inter_start + 1
                union_len = (g_end - g_start + 1) + (p_end - p_start + 1) - inter_len
                tiou = inter_len / union_len if union_len > 0 else 0.0
                matched_preds.append({
                    "pred_interval_idx": p_idx,
                    "overlap_frames": int(inter_len),
                    "tiou": round(float(tiou), 4)
                })

        best_tiou = max([m["tiou"] for m in matched_preds], default=0.0)
        interval_overlaps.append({
            "gt_interval_idx": g_idx,
            "gt_bounds": (g["start_frame"], g["end_frame"]),
            "matched_predictions": matched_preds,
            "best_tiou": round(float(best_tiou), 4)
        })

    return {
        "gt_intervals": gt_intervals,
        "pred_intervals": pred_intervals,
        "gt_interval_count": len(gt_intervals),
        "pred_interval_count": len(pred_intervals),
        "temporal_overlap_frames": intersection_frames,
        "temporal_union_frames": union_frames,
        "temporal_iou": round(temporal_iou, 4),
        "interval_overlaps": interval_overlaps
    }


def compute_frame_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes frame-level classification metrics: TP, FP, TN, FN, accuracy,
    precision, recall, F1, and continuous ROC-AUC and PR-AUC if scores are available.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f"Length mismatch: gt_labels ({len(gt_labels)}) vs pred_labels ({len(pred_labels)})"
        )

    tp = int(np.sum((gt_labels == 1) & (pred_labels == 1)))
    fp = int(np.sum((gt_labels == 0) & (pred_labels == 1)))
    tn = int(np.sum((gt_labels == 0) & (pred_labels == 0)))
    fn = int(np.sum((gt_labels == 1) & (pred_labels == 0)))

    total_frames = len(gt_labels)
    accuracy = float((tp + tn) / total_frames) if total_frames > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "total_frames": total_frames,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    # Continuous ranking metrics (ROC-AUC, PR-AUC)
    if pred_scores is not None:
        if len(pred_scores) != total_frames:
            raise ValueError("pred_scores length does not match gt_labels length.")

        unique_gt = np.unique(gt_labels)
        if len(unique_gt) >= 2:
            try:
                roc_auc = float(roc_auc_score(gt_labels, pred_scores))
                pr_auc = float(average_precision_score(gt_labels, pred_scores))
                metrics["roc_auc"] = round(roc_auc, 4)
                metrics["pr_auc"] = round(pr_auc, 4)
            except Exception as e:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
                metrics["auc_note"] = f"Calculation error: {e}"
        else:
            # Video contains only normal or only abnormal frames
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["auc_note"] = f"Only one class ({unique_gt[0]}) present in ground truth"
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["auc_note"] = "Continuous anomaly scores not provided"

    return metrics


def compute_spatial_metrics(
    gt_masks: List[np.ndarray],
    pred_masks: Optional[List[np.ndarray]],
    abnormal_frame_indices: List[int],
    expected_shape: Tuple[int, int] = (360, 640)
) -> Dict[str, Any]:
    """
    Computes spatial localization metrics against ground-truth pixel masks for abnormal frames.

    For each abnormal frame:
      - IoU = intersection(gt_mask, pred_mask) / union(gt_mask, pred_mask)
      - precision = intersection / sum(pred_mask)
      - recall = intersection / sum(gt_mask)
    """
    if pred_masks is None or len(pred_masks) == 0:
        return {
            "spatial_metrics_available": False,
            "reason": "Model predictions did not include spatial masks or bounding boxes.",
            "per_frame_iou": None,
            "mean_iou": None,
            "spatial_precision": None,
            "spatial_recall": None
        }

    if len(abnormal_frame_indices) == 0:
        return {
            "spatial_metrics_available": True,
            "reason": "Video contains 0 abnormal ground-truth frames.",
            "per_frame_iou": [],
            "mean_iou": 1.0,
            "spatial_precision": 1.0,
            "spatial_recall": 1.0
        }

    frame_ious = []
    frame_precisions = []
    frame_recalls = []

    for f_idx in abnormal_frame_indices:
        gt_m = (gt_masks[f_idx] > 0).astype(np.uint8)

        if f_idx < len(pred_masks) and pred_masks[f_idx] is not None:
            pred_m = (pred_masks[f_idx] > 0).astype(np.uint8)
            # Resize if prediction mask does not match GT resolution
            if pred_m.shape != gt_m.shape:
                import cv2
                pred_m = cv2.resize(pred_m, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            pred_m = np.zeros_like(gt_m)

        inter = int(np.sum((gt_m == 1) & (pred_m == 1)))
        union = int(np.sum((gt_m == 1) | (pred_m == 1)))
        gt_sum = int(np.sum(gt_m))
        pred_sum = int(np.sum(pred_m))

        iou = inter / union if union > 0 else 0.0
        prec = inter / pred_sum if pred_sum > 0 else 0.0
        rec = inter / gt_sum if gt_sum > 0 else 0.0

        frame_ious.append(iou)
        frame_precisions.append(prec)
        frame_recalls.append(rec)

    mean_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
    mean_prec = float(np.mean(frame_precisions)) if frame_precisions else 0.0
    mean_rec = float(np.mean(frame_recalls)) if frame_recalls else 0.0

    return {
        "spatial_metrics_available": True,
        "evaluated_abnormal_frames": len(abnormal_frame_indices),
        "mean_iou": round(mean_iou, 4),
        "spatial_precision": round(mean_prec, 4),
        "spatial_recall": round(mean_rec, 4),
        "per_frame_iou": [round(x, 4) for x in frame_ious]
    }


def align_clip_scores_to_frames(
    clip_scores: np.ndarray,
    clip_native_ranges: List[Tuple[int, int]],
    total_native_frames: int
) -> np.ndarray:
    """
    Maps clip-level anomaly scores to frame-level scores across all native frames.
    For frames covered by multiple overlapping clips, computes the average score.
    For uncovered border frames, fills using the nearest available score.
    """
    accum_scores = np.zeros(total_native_frames, dtype=np.float32)
    counts = np.zeros(total_native_frames, dtype=np.int32)

    for score, (s_idx, e_idx) in zip(clip_scores, clip_native_ranges):
        start = max(0, min(total_native_frames - 1, int(s_idx)))
        end = max(0, min(total_native_frames, int(e_idx) + 1))
        accum_scores[start:end] += float(score)
        counts[start:end] += 1

    valid_mask = (counts > 0)
    frame_scores = np.zeros(total_native_frames, dtype=np.float32)
    frame_scores[valid_mask] = accum_scores[valid_mask] / counts[valid_mask]

    # Interpolate / extrapolate uncovered frames if any
    if not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            frame_scores = np.interp(
                np.arange(total_native_frames),
                valid_indices,
                frame_scores[valid_indices]
            ).astype(np.float32)

    return frame_scores


class AvenueEvaluator:
    """
    Evaluation Engine for the CUHK Avenue benchmark.
    Handles per-video and dataset-wide evaluation against pixel-level ground truth.
    """

    def __init__(
        self,
        ground_truth_root: Optional[Union[str, Path]] = None,
        default_threshold: float = 0.50
    ):
        self.ground_truth_root = Path(ground_truth_root) if ground_truth_root else None
        self.default_threshold = default_threshold
        self._gt_cache: Dict[str, Dict[str, Any]] = {}

    def get_ground_truth(self, video_id: str, gt_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves ground-truth data for a video, caching in memory.
        """
        key = str(video_id).zfill(2)
        if key in self._gt_cache:
            return self._gt_cache[key]

        target_path: Optional[Path] = None
        if gt_path and Path(gt_path).exists():
            target_path = Path(gt_path)
        elif self.ground_truth_root:
            vid_num = int(key)
            candidates = [
                self.ground_truth_root / f"{vid_num}_label.mat",
                self.ground_truth_root / f"{vid_num:02d}_label.mat"
            ]
            for cand in candidates:
                if cand.exists():
                    target_path = cand
                    break

        if target_path is None or not target_path.exists():
            raise FileNotFoundError(
                f"Ground-truth MAT file could not be found for video '{video_id}' "
                f"(gt_path='{gt_path}', root='{self.ground_truth_root}')."
            )

        gt_data = load_avenue_ground_truth(target_path)
        self._gt_cache[key] = gt_data
        return gt_data

    def evaluate_video(
        self,
        video_id: str,
        pred_frame_scores: Optional[np.ndarray] = None,
        pred_frame_labels: Optional[np.ndarray] = None,
        pred_spatial_masks: Optional[List[np.ndarray]] = None,
        gt_path: Optional[str] = None,
        threshold: Optional[float] = None,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Evaluates predictions for a single test video.
        """
        gt_data = self.get_ground_truth(video_id, gt_path=gt_path)
        gt_labels = gt_data["frame_labels"]
        total_frames = gt_data["frame_count"]

        thresh = self.default_threshold if threshold is None else threshold

        # Resolve binary predictions
        if pred_frame_labels is None:
            if pred_frame_scores is not None:
                pred_frame_labels = (pred_frame_scores >= thresh).astype(np.int32)
            else:
                raise ValueError(
                    f"Either pred_frame_labels or pred_frame_scores must be provided for video '{video_id}'."
                )

        if len(pred_frame_labels) != total_frames:
            raise ValueError(
                f"Frame count mismatch for video '{video_id}': "
                f"ground truth has {total_frames} frames, but predictions have {len(pred_frame_labels)} frames."
            )

        # 1. Frame-Level Metrics
        frame_metrics = compute_frame_metrics(
            gt_labels=gt_labels,
            pred_labels=pred_frame_labels,
            pred_scores=pred_frame_scores
        )

        # 2. Temporal Metrics
        temporal_metrics = compute_temporal_metrics(
            gt_labels=gt_labels,
            pred_labels=pred_frame_labels,
            fps=fps
        )

        # 3. Spatial Metrics
        spatial_metrics = compute_spatial_metrics(
            gt_masks=gt_data["binary_masks"],
            pred_masks=pred_spatial_masks,
            abnormal_frame_indices=gt_data["abnormal_frame_indices"]
        )

        return {
            "video_id": str(video_id).zfill(2),
            "frame_count": total_frames,
            "fps": fps,
            "threshold_used": thresh,
            "gt_abnormal_frames": gt_data["abnormal_frame_count"],
            "gt_normal_frames": gt_data["normal_frame_count"],
            "pred_abnormal_frames": int(np.sum(pred_frame_labels == 1)),
            "pred_normal_frames": int(np.sum(pred_frame_labels == 0)),
            "frame_metrics": frame_metrics,
            "temporal_metrics": temporal_metrics,
            "spatial_metrics": spatial_metrics,
            "gt_path": gt_data["gt_path"]
        }

    def evaluate_test_set(
        self,
        video_predictions: List[Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a complete set of test video predictions.

        Each element in video_predictions should contain:
          - 'video_id': str (e.g. '01')
          - 'pred_frame_scores': np.ndarray of shape [N] (optional if pred_frame_labels given)
          - 'pred_frame_labels': np.ndarray of shape [N] (optional if pred_frame_scores given)
          - 'pred_spatial_masks': optional list of [H, W] masks
          - 'ground_truth_path': optional str
          - 'fps': optional float (defaults to 25.0)

        Returns complete report with global, video-level, and per-frame tables.
        """
        thresh = self.default_threshold if threshold is None else threshold
        video_reports = []

        all_gt_frames = []
        all_pred_frames = []
        all_pred_scores = []
        has_scores = True

        frame_records = []
        video_records = []

        for item in video_predictions:
            vid_id = str(item["video_id"]).zfill(2)
            gt_path = item.get("ground_truth_path")
            fps = float(item.get("fps", 25.0))

            scores = item.get("pred_frame_scores")
            labels = item.get("pred_frame_labels")
            spatial = item.get("pred_spatial_masks")

            v_report = self.evaluate_video(
                video_id=vid_id,
                pred_frame_scores=scores,
                pred_frame_labels=labels,
                pred_spatial_masks=spatial,
                gt_path=gt_path,
                threshold=thresh,
                fps=fps
            )
            video_reports.append(v_report)

            # Collect for global metrics
            gt_labels = self.get_ground_truth(vid_id, gt_path)["frame_labels"]
            pred_labels = (scores >= thresh).astype(np.int32) if labels is None else labels
            all_gt_frames.extend(gt_labels.tolist())
            all_pred_frames.extend(pred_labels.tolist())

            if scores is not None and has_scores:
                all_pred_scores.extend(scores.tolist())
            else:
                has_scores = False

            # Build per-frame records
            for f_i in range(len(gt_labels)):
                f_gt = int(gt_labels[f_i])
                f_pred = int(pred_labels[f_i])
                f_score = float(scores[f_i]) if scores is not None else None

                if f_gt == 1 and f_pred == 1:
                    cls_type = "TP"
                elif f_gt == 0 and f_pred == 1:
                    cls_type = "FP"
                elif f_gt == 0 and f_pred == 0:
                    cls_type = "TN"
                else:
                    cls_type = "FN"

                frame_records.append({
                    "video_id": vid_id,
                    "frame_index": f_i,
                    "timestamp_seconds": round(f_i / fps, 3),
                    "ground_truth": f_gt,
                    "predicted_label": f_pred,
                    "predicted_score": round(f_score, 4) if f_score is not None else "",
                    "classification": cls_type,
                    "is_correct": (f_gt == f_pred)
                })

            # Build per-video summary record
            fm = v_report["frame_metrics"]
            tm = v_report["temporal_metrics"]
            sm = v_report["spatial_metrics"]

            video_records.append({
                "video_id": vid_id,
                "frame_count": v_report["frame_count"],
                "gt_abnormal_frames": v_report["gt_abnormal_frames"],
                "pred_abnormal_frames": v_report["pred_abnormal_frames"],
                "tp": fm["tp"],
                "fp": fm["fp"],
                "tn": fm["tn"],
                "fn": fm["fn"],
                "accuracy": fm["accuracy"],
                "precision": fm["precision"],
                "recall": fm["recall"],
                "f1": fm["f1"],
                "roc_auc": fm["roc_auc"] if fm["roc_auc"] is not None else "",
                "pr_auc": fm["pr_auc"] if fm["pr_auc"] is not None else "",
                "temporal_iou": tm["temporal_iou"],
                "gt_intervals": tm["gt_interval_count"],
                "pred_intervals": tm["pred_interval_count"],
                "spatial_miou": sm["mean_iou"] if sm["spatial_metrics_available"] else ""
            })

        # Global dataset-wide metrics
        all_gt_arr = np.array(all_gt_frames, dtype=np.int32)
        all_pred_arr = np.array(all_pred_frames, dtype=np.int32)
        all_scores_arr = np.array(all_pred_scores, dtype=np.float32) if has_scores else None

        global_frame_metrics = compute_frame_metrics(
            gt_labels=all_gt_arr,
            pred_labels=all_pred_arr,
            pred_scores=all_scores_arr
        )

        # Global temporal metrics across entire test set
        global_inter_frames = int(np.sum((all_gt_arr == 1) & (all_pred_arr == 1)))
        global_union_frames = int(np.sum((all_gt_arr == 1) | (all_pred_arr == 1)))
        global_temporal_iou = float(global_inter_frames / global_union_frames) if global_union_frames > 0 else 0.0

        # Global spatial summary
        spatial_available = any(v["spatial_metrics"]["spatial_metrics_available"] for v in video_reports)
        if spatial_available:
            miou_list = [v["spatial_metrics"]["mean_iou"] for v in video_reports if v["spatial_metrics"]["mean_iou"] is not None]
            global_miou = float(np.mean(miou_list)) if miou_list else None
        else:
            global_miou = None

        dataset_summary = {
            "dataset": "CUHK Avenue",
            "evaluated_videos": len(video_predictions),
            "threshold_used": thresh,
            "total_test_frames": int(len(all_gt_arr)),
            "total_gt_abnormal_frames": int(np.sum(all_gt_arr == 1)),
            "total_gt_normal_frames": int(np.sum(all_gt_arr == 0)),
            "total_pred_abnormal_frames": int(np.sum(all_pred_arr == 1)),
            "total_pred_normal_frames": int(np.sum(all_pred_arr == 0)),
            "global_frame_metrics": global_frame_metrics,
            "global_temporal_metrics": {
                "temporal_overlap_frames": global_inter_frames,
                "temporal_union_frames": global_union_frames,
                "global_temporal_iou": round(global_temporal_iou, 4)
            },
            "global_spatial_metrics": {
                "spatial_metrics_available": spatial_available,
                "mean_iou": round(global_miou, 4) if global_miou is not None else None,
                "note": "Spatial metrics calculated over abnormal ground-truth frames." if spatial_available else "Spatial predictions unavailable."
            },
            "per_video_reports": video_reports
        }

        frame_metrics_df = pd.DataFrame(frame_records)
        video_metrics_df = pd.DataFrame(video_records)

        return {
            "summary": dataset_summary,
            "frame_metrics_df": frame_metrics_df,
            "video_metrics_df": video_metrics_df
        }
