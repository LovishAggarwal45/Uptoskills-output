"""
Unit tests for Avenue One-Class Anomaly Detection (Experiment G):
1. Normal prototype creation
2. Feature normalization
3. Cosine-distance anomaly scoring
4. Threshold percentile calculation
5. Checkpoint save/load
6. Score smoothing
7. Temporal post-processing (gap merging & min duration)
8. No test-label leakage
9. Correct output dimensions
10. Integration with AvenueEvaluator
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from src.models.avenue_oneclass import (
    AvenueOneClassDetector,
    l2_normalize,
    compute_score_statistics
)
from src.evaluation.avenue_evaluator import (
    AvenueEvaluator,
    compute_frame_metrics,
    compute_temporal_metrics
)


class TestAvenueOneClass(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.feature_dim = 64
        # Generate 100 normal training feature vectors clustered around a base vector
        base_vec = np.random.randn(self.feature_dim)
        base_vec /= np.linalg.norm(base_vec)
        self.train_features = base_vec + 0.1 * np.random.randn(100, self.feature_dim)

        # Generate 20 normal validation feature vectors
        self.val_features = base_vec + 0.12 * np.random.randn(20, self.feature_dim)

    # 1. Normal prototype creation
    def test_01_normal_prototype_creation(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        stats = detector.fit(self.train_features)

        self.assertIsNotNone(detector.centroid)
        self.assertEqual(detector.centroid.shape, (self.feature_dim,))
        # Centroid must be unit normalized
        centroid_norm = np.linalg.norm(detector.centroid)
        self.assertAlmostEqual(centroid_norm, 1.0, places=5)
        self.assertIn("mean", stats)
        self.assertIn("p99", stats)

    # 2. Feature normalization
    def test_02_feature_normalization(self):
        feats = np.random.randn(15, self.feature_dim) * 50.0
        norm_feats = l2_normalize(feats)
        norms = np.linalg.norm(norm_feats, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=5)

        # Test zero vector handling without NaNs
        zero_vec = np.zeros((2, self.feature_dim))
        norm_zero = l2_normalize(zero_vec)
        self.assertFalse(np.isnan(norm_zero).any())

    # 3. Cosine-distance anomaly scoring
    def test_03_cosine_distance_anomaly_scoring(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.fit(self.train_features)

        # Feature exactly equal to centroid should have distance ~ 0.0
        c_score = detector.predict_score(detector.centroid.reshape(1, -1))[0]
        self.assertAlmostEqual(c_score, 0.0, places=4)

        # Orthogonal feature should have distance ~ 1.0
        ortho = np.random.randn(self.feature_dim)
        ortho -= np.dot(ortho, detector.centroid) * detector.centroid
        ortho /= np.linalg.norm(ortho)
        ortho_score = detector.predict_score(ortho.reshape(1, -1))[0]
        self.assertAlmostEqual(ortho_score, 1.0, places=4)

        # Opposite feature should have distance ~ 2.0
        opp = -detector.centroid
        opp_score = detector.predict_score(opp.reshape(1, -1))[0]
        self.assertAlmostEqual(opp_score, 2.0, places=4)

    # 4. Threshold percentile calculation
    def test_04_threshold_percentile_calculation(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.fit(self.train_features)

        val_scores = detector.predict_score(self.val_features)
        thresh = detector.calibrate_threshold(val_scores, percentile=95.0)

        expected_thresh = float(np.percentile(val_scores, 95.0))
        self.assertAlmostEqual(thresh, expected_thresh, places=5)
        self.assertEqual(detector.threshold, thresh)
        self.assertEqual(detector.calibration_percentile, 95.0)

    # 5. Checkpoint save/load
    def test_05_checkpoint_save_and_load(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.fit(self.train_features)
        val_scores = detector.predict_score(self.val_features)
        detector.calibrate_threshold(val_scores, percentile=99.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_detector.pt"
            detector.save(ckpt_path, extra_metadata={"experiment": "test"})

            loaded_detector = AvenueOneClassDetector()
            meta = loaded_detector.load(ckpt_path)

            self.assertEqual(meta.get("experiment"), "test")
            self.assertEqual(loaded_detector.feature_dim, self.feature_dim)
            self.assertAlmostEqual(loaded_detector.threshold, detector.threshold, places=5)
            self.assertTrue(np.allclose(loaded_detector.centroid, detector.centroid))

    # 6. Score smoothing
    def test_06_score_smoothing(self):
        # Array with a single spike
        scores = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32)
        smoothed = AvenueOneClassDetector.smooth_scores(scores, window_size=3)

        self.assertEqual(len(smoothed), len(scores))
        # The peak at index 2 should be dampened by surrounding values
        self.assertLess(smoothed[2], 0.9)
        self.assertGreater(smoothed[1], 0.1)

    # 7. Temporal post-processing
    def test_07_temporal_postprocessing(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.threshold = 0.5

        # 50 frames at 25 fps:
        # Frames 10-11: 2-frame spike (0.08s < 0.5s min duration) -> should be discarded
        # Frames 20-35: 16-frame abnormal span (0.64s > 0.5s) -> should survive
        # Frames 38-48: 11-frame abnormal span, separated from 35 by 2 frames (0.08s < 0.5s gap) -> should merge!
        scores = np.zeros(50, dtype=np.float32)
        scores[10:12] = 0.8
        scores[20:36] = 0.8
        scores[38:49] = 0.8

        smoothed, binary_preds, intervals = detector.temporal_postprocess(
            frame_scores=scores,
            threshold=0.5,
            smoothing_window=1,
            min_anomaly_duration_seconds=0.5,
            merge_gap_seconds=0.2, # 5 frames
            fps=25.0
        )

        self.assertEqual(len(binary_preds), 50)
        # 2-frame spike should be filtered out
        self.assertEqual(binary_preds[10], 0)
        self.assertEqual(binary_preds[11], 0)

        # Merged span: 20 to 48 should be active
        self.assertEqual(binary_preds[20], 1)
        self.assertEqual(binary_preds[36], 1) # bridge frame was merged
        self.assertEqual(binary_preds[37], 1) # bridge frame was merged
        self.assertEqual(binary_preds[48], 1)
        self.assertEqual(len(intervals), 1)

    # 8. No test-label leakage
    def test_08_no_test_label_leakage(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        # fit only accepts training features
        fit_params = detector.fit.__code__.co_varnames
        self.assertNotIn("test_labels", fit_params)
        self.assertNotIn("ground_truth", fit_params)

        # calibrate_threshold only accepts validation_scores
        calib_params = detector.calibrate_threshold.__code__.co_varnames
        self.assertNotIn("test_labels", calib_params)
        self.assertNotIn("ground_truth", calib_params)

    # 9. Correct output dimensions
    def test_09_correct_output_dimensions(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.fit(self.train_features)
        detector.threshold = 0.4

        N = 350
        test_feats = np.random.randn(N, self.feature_dim)
        scores = detector.predict_score(test_feats)
        self.assertEqual(scores.shape, (N,))

        smoothed, preds, intervals = detector.temporal_postprocess(scores, fps=25.0)
        self.assertEqual(smoothed.shape, (N,))
        self.assertEqual(preds.shape, (N,))
        self.assertEqual(preds.dtype, np.int32)

    # 10. Integration with AvenueEvaluator
    def test_10_integration_with_avenue_evaluator(self):
        detector = AvenueOneClassDetector(feature_dim=self.feature_dim)
        detector.fit(self.train_features)
        detector.threshold = 0.45

        # Synthetic test video with 100 frames
        gt_labels = np.zeros(100, dtype=np.int32)
        gt_labels[30:60] = 1 # abnormal span

        # Model scores
        pred_scores = np.ones(100, dtype=np.float32) * 0.2
        pred_scores[28:58] = 0.7 # overlapping abnormal span

        smoothed, binary_preds, _ = detector.temporal_postprocess(
            frame_scores=pred_scores,
            threshold=0.45,
            fps=25.0
        )

        frame_metrics = compute_frame_metrics(gt_labels, binary_preds, pred_scores=smoothed)
        self.assertIn("accuracy", frame_metrics)
        self.assertIn("precision", frame_metrics)
        self.assertIn("recall", frame_metrics)
        self.assertIn("f1", frame_metrics)
        self.assertIn("roc_auc", frame_metrics)
        self.assertGreater(frame_metrics["roc_auc"], 0.7)

        temp_metrics = compute_temporal_metrics(gt_labels, binary_preds, fps=25.0)
        self.assertGreater(temp_metrics["temporal_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
