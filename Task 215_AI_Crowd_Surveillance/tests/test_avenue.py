"""
Unit tests for CUHK Avenue dataset components:
- Avenue metadata & format helpers
- AvenueDataset (PyTorch Dataset)
- AvenueEvaluator (frame-level, temporal, spatial metrics)
"""

import unittest
import numpy as np
import torch
import tempfile
import csv
from pathlib import Path
from typing import Dict, Any

from src.evaluation.avenue_evaluator import (
    extract_temporal_intervals,
    compute_temporal_metrics,
    compute_frame_metrics,
    compute_spatial_metrics,
    align_clip_scores_to_frames,
    AvenueEvaluator
)
from scripts.prepare_avenue_dataset import find_matching_gt_file


class TestAvenueEvaluator(unittest.TestCase):

    def test_extract_temporal_intervals(self):
        # Array with two abnormal intervals: frames 2..4 (len 3) and 8..9 (len 2)
        seq = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int32)
        intervals = extract_temporal_intervals(seq, fps=25.0)

        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]["start_frame"], 2)
        self.assertEqual(intervals[0]["end_frame"], 4)
        self.assertEqual(intervals[0]["frame_span"], 3)
        self.assertAlmostEqual(intervals[0]["start_time"], 2 / 25.0)

        self.assertEqual(intervals[1]["start_frame"], 8)
        self.assertEqual(intervals[1]["end_frame"], 9)
        self.assertEqual(intervals[1]["frame_span"], 2)

    def test_compute_temporal_metrics(self):
        gt = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.int32)
        # Prediction overlaps on frames 3..4, misses 2 and 5, false alarm on 6
        pred = np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=np.int32)

        res = compute_temporal_metrics(gt, pred, fps=25.0)
        # GT abnormal = 4 frames (2,3,4,5), Pred abnormal = 3 frames (3,4,6)
        # Intersection = frames 3,4 (2 frames)
        # Union = frames 2,3,4,5,6 (5 frames)
        self.assertEqual(res["temporal_overlap_frames"], 2)
        self.assertEqual(res["temporal_union_frames"], 5)
        self.assertAlmostEqual(res["temporal_iou"], 2.0 / 5.0, places=3)
        self.assertEqual(res["gt_interval_count"], 1)

    def test_compute_frame_metrics(self):
        gt = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
        pred = np.array([1, 1, 0, 1, 0, 0], dtype=np.int32)
        scores = np.array([0.9, 0.8, 0.3, 0.7, 0.1, 0.2], dtype=np.float32)

        metrics = compute_frame_metrics(gt, pred, pred_scores=scores)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 2)

        # Accuracy = (2 + 2) / 6 = 4/6 = 0.6667
        self.assertAlmostEqual(metrics["accuracy"], 4.0 / 6.0, places=3)
        # Precision = 2 / 3
        self.assertAlmostEqual(metrics["precision"], 2.0 / 3.0, places=3)
        # Recall = 2 / 3
        self.assertAlmostEqual(metrics["recall"], 2.0 / 3.0, places=3)
        # F1 = 2 / 3
        self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0, places=3)

        self.assertIsNotNone(metrics["roc_auc"])
        self.assertGreater(metrics["roc_auc"], 0.5)

    def test_compute_spatial_metrics_unavailable(self):
        res = compute_spatial_metrics([], None, [0, 1])
        self.assertFalse(res["spatial_metrics_available"])
        self.assertIsNone(res["mean_iou"])

    def test_compute_spatial_metrics_calculation(self):
        # Create dummy 10x10 masks
        gt_mask = np.zeros((10, 10), dtype=np.uint8)
        gt_mask[2:6, 2:6] = 1 # 4x4 = 16 pixels

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        pred_mask[2:6, 4:8] = 1 # 4x4 = 16 pixels, overlaps on 2:6, 4:6 (4x2 = 8 pixels)

        gt_masks = [gt_mask]
        pred_masks = [pred_mask]
        abnormal_indices = [0]

        res = compute_spatial_metrics(gt_masks, pred_masks, abnormal_indices, expected_shape=(10, 10))
        self.assertTrue(res["spatial_metrics_available"])
        # Intersection = 8, Union = 16 + 16 - 8 = 24. IoU = 8/24 = 1/3 ~ 0.3333
        self.assertAlmostEqual(res["mean_iou"], 8.0 / 24.0, places=3)
        # Precision = 8 / 16 = 0.5
        self.assertAlmostEqual(res["spatial_precision"], 0.5, places=3)
        # Recall = 8 / 16 = 0.5
        self.assertAlmostEqual(res["spatial_recall"], 0.5, places=3)

    def test_align_clip_scores_to_frames(self):
        # 3 clips of 10 native frames each, overlapping
        clip_scores = np.array([0.2, 0.8, 0.4], dtype=np.float32)
        clip_ranges = [(0, 9), (5, 14), (10, 19)]
        total_frames = 20

        frame_scores = align_clip_scores_to_frames(clip_scores, clip_ranges, total_frames)
        self.assertEqual(len(frame_scores), total_frames)
        # Frame 0..4 covered only by clip 0 -> score 0.2
        self.assertAlmostEqual(frame_scores[0], 0.2, places=3)
        # Frame 7 covered by clip 0 (0.2) and clip 1 (0.8) -> average 0.5
        self.assertAlmostEqual(frame_scores[7], 0.5, places=3)
        # Frame 12 covered by clip 1 (0.8) and clip 2 (0.4) -> average 0.6
        self.assertAlmostEqual(frame_scores[12], 0.6, places=3)


class TestAvenueDatasetNoLeakage(unittest.TestCase):

    def test_metadata_matching_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "1_label.mat").touch()
            (tmp_path / "10_label.mat").touch()

            self.assertIsNotNone(find_matching_gt_file(tmp_path, "01"))
            self.assertIsNotNone(find_matching_gt_file(tmp_path, "1"))
            self.assertIsNotNone(find_matching_gt_file(tmp_path, "10"))
            self.assertIsNone(find_matching_gt_file(tmp_path, "99"))


if __name__ == "__main__":
    unittest.main()
