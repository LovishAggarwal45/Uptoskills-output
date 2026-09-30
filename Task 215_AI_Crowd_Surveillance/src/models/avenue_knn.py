"""
Avenue Feature-Memory / kNN One-Class Anomaly Detector.

Experiment H:
Normal-only anomaly detection using a memory bank of normal training
feature vectors.

Method:
1. Extract fused RGB + optical-flow features from normal Avenue training videos.
2. L2-normalize the feature vectors.
3. Store the normalized normal features as a memory bank.
4. For a new feature, calculate cosine distance to normal memory-bank features.
5. Select the K nearest normal features.
6. Use the mean K-nearest-neighbor distance as the anomaly score.

Higher score = more different from learned normal behavior.

Integrity:
- Training uses only normal Avenue training videos.
- Threshold calibration uses only normal validation videos.
- Test ground-truth masks are never used for score generation or threshold selection.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch


def l2_normalize(
    features: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    L2-normalize feature vectors along the final dimension.

    Input:
        [N, D] or [D]

    Output:
        Same shape as input.
    """
    features = np.asarray(features, dtype=np.float32)

    norm = np.linalg.norm(
        features,
        axis=-1,
        keepdims=True
    )

    norm = np.maximum(norm, eps)

    return features / norm


def compute_score_statistics(
    scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute descriptive statistics for anomaly scores.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if len(scores) == 0:
        return {}

    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "p50": float(np.percentile(scores, 50.0)),
        "p75": float(np.percentile(scores, 75.0)),
        "p90": float(np.percentile(scores, 90.0)),
        "p95": float(np.percentile(scores, 95.0)),
        "p99": float(np.percentile(scores, 99.0)),
    }


class AvenueKNNDetector:
    """
    Normal-feature memory-bank anomaly detector.

    The detector stores normal training features and scores a new feature
    according to its distance from the K nearest normal examples.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        k_neighbors: int = 5,
        threshold: Optional[float] = None
    ):
        if k_neighbors < 1:
            raise ValueError(
                f"k_neighbors must be >= 1, got {k_neighbors}"
            )

        self.feature_dim = int(feature_dim)
        self.k_neighbors = int(k_neighbors)
        self.threshold = threshold

        self.memory_bank: Optional[np.ndarray] = None

        self.normal_training_stats: Dict[str, float] = {}
        self.normal_validation_stats: Dict[str, float] = {}

        self.calibration_percentile: Optional[float] = None

    def fit(
        self,
        training_features: np.ndarray,
        eps: float = 1e-8
    ) -> Dict[str, float]:
        """
        Build the normal feature memory bank.

        training_features:
            [N, D]

        Returns:
            Statistics of training nearest-neighbor scores.
        """
        training_features = np.asarray(
            training_features,
            dtype=np.float32
        )

        if training_features.ndim != 2:
            raise ValueError(
                "Expected training features with shape [N, D], "
                f"got {training_features.shape}"
            )

        if len(training_features) == 0:
            raise ValueError(
                "Cannot fit kNN detector with zero training features."
            )

        self.feature_dim = int(training_features.shape[1])

        # Normalize all normal training features.
        normalized = l2_normalize(
            training_features,
            eps=eps
        )

        # Store the normal memory bank.
        self.memory_bank = normalized.astype(
            np.float32,
            copy=True
        )

        # Effective K cannot exceed the number of training samples.
        effective_k = min(
            self.k_neighbors,
            len(self.memory_bank)
        )

        self.k_neighbors = int(effective_k)

        # Compute leave-one-out training scores.
        #
        # We exclude each feature itself. Otherwise its nearest distance
        # would always be zero and training statistics would be misleading.
        similarity_matrix = np.matmul(
            self.memory_bank,
            self.memory_bank.T
        )

        np.clip(
            similarity_matrix,
            -1.0,
            1.0,
            out=similarity_matrix
        )

        distance_matrix = 1.0 - similarity_matrix

        # Exclude self-neighbor.
        np.fill_diagonal(
            distance_matrix,
            np.inf
        )

        if len(self.memory_bank) == 1:
            # With only one training sample there is no valid
            # leave-one-out neighbor.
            training_scores = np.zeros(
                len(self.memory_bank),
                dtype=np.float32
            )
        else:
            effective_k_train = min(
                self.k_neighbors,
                len(self.memory_bank) - 1
            )

            nearest_distances = np.partition(
                distance_matrix,
                kth=effective_k_train - 1,
                axis=1
            )[:, :effective_k_train]

            training_scores = np.mean(
                nearest_distances,
                axis=1
            ).astype(np.float32)

        self.normal_training_stats = compute_score_statistics(
            training_scores
        )

        return self.normal_training_stats

    def predict_score(
        self,
        features: np.ndarray,
        eps: float = 1e-8,
        query_chunk_size: int = 256
    ) -> np.ndarray:
        """
        Compute kNN anomaly scores.

        Each query feature is compared against the normal memory bank.

        Score:
            mean distance to K nearest normal training features.

        Higher score = more anomalous.

        Input:
            [N, D] or [D]

        Output:
            [N]
        """
        if self.memory_bank is None:
            raise RuntimeError(
                "Detector must be fitted before predicting scores."
            )

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError(
                "Expected features with shape [N, D], "
                f"got {features.shape}"
            )

        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected "
                f"{self.feature_dim}, got {features.shape[1]}"
            )

        normalized_queries = l2_normalize(
            features,
            eps=eps
        )

        memory = self.memory_bank

        effective_k = min(
            self.k_neighbors,
            len(memory)
        )

        all_scores = []

        # Process queries in chunks to avoid creating a very large
        # query-by-memory matrix.
        for start in range(
            0,
            len(normalized_queries),
            query_chunk_size
        ):
            query_chunk = normalized_queries[
                start:start + query_chunk_size
            ]

            # Cosine similarity because both vectors are L2-normalized.
            similarities = np.matmul(
                query_chunk,
                memory.T
            )

            np.clip(
                similarities,
                -1.0,
                1.0,
                out=similarities
            )

            distances = 1.0 - similarities

            nearest_distances = np.partition(
                distances,
                kth=effective_k - 1,
                axis=1
            )[:, :effective_k]

            scores = np.mean(
                nearest_distances,
                axis=1
            )

            all_scores.append(
                scores.astype(np.float32)
            )

        return np.concatenate(
            all_scores,
            axis=0
        )

    def calibrate_threshold(
        self,
        validation_scores: np.ndarray,
        percentile: float = 99.0
    ) -> float:
        """
        Calibrate threshold using ONLY normal validation scores.
        """
        validation_scores = np.asarray(
            validation_scores,
            dtype=np.float32
        ).reshape(-1)

        if len(validation_scores) == 0:
            raise ValueError(
                "Cannot calibrate threshold with empty validation scores."
            )

        if not (0.0 < percentile <= 100.0):
            raise ValueError(
                f"Percentile must be in (0, 100], got {percentile}"
            )

        self.calibration_percentile = float(percentile)

        self.threshold = float(
            np.percentile(
                validation_scores,
                percentile
            )
        )

        self.normal_validation_stats = compute_score_statistics(
            validation_scores
        )

        return self.threshold

    @staticmethod
    def smooth_scores(
        scores: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Moving-average smoothing.

        Kept intentionally simple here. Temporal post-processing remains
        in the existing Avenue evaluation pipeline.
        """
        scores = np.asarray(
            scores,
            dtype=np.float32
        )

        if len(scores) <= 1 or window_size <= 1:
            return scores.copy()

        from scipy.ndimage import uniform_filter1d

        w = min(
            int(window_size),
            len(scores)
        )

        return uniform_filter1d(
            scores,
            size=w,
            mode="nearest"
        ).astype(np.float32)

    def save(
        self,
        checkpoint_path: Union[str, Path],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save detector state and normal feature memory bank.
        """
        if self.memory_bank is None:
            raise RuntimeError(
                "Cannot save an unfitted detector."
            )

        path = Path(checkpoint_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        payload = {
            "memory_bank": torch.from_numpy(
                self.memory_bank
            ),
            "feature_dim": self.feature_dim,
            "k_neighbors": self.k_neighbors,
            "threshold": self.threshold,
            "calibration_percentile": self.calibration_percentile,
            "normal_training_stats": self.normal_training_stats,
            "normal_validation_stats": self.normal_validation_stats,
            "extra_metadata": extra_metadata or {}
        }

        torch.save(
            payload,
            str(path)
        )

    def load(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load detector state.
        """
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}"
            )

        payload = torch.load(
            str(path),
            map_location="cpu"
        )

        memory = payload.get("memory_bank")

        if memory is None:
            raise ValueError(
                "Checkpoint does not contain a memory_bank."
            )

        if isinstance(memory, torch.Tensor):
            self.memory_bank = memory.numpy().astype(
                np.float32
            )
        else:
            self.memory_bank = np.asarray(
                memory,
                dtype=np.float32
            )

        self.feature_dim = int(
            payload.get(
                "feature_dim",
                self.memory_bank.shape[1]
            )
        )

        self.k_neighbors = int(
            payload.get(
                "k_neighbors",
                5
            )
        )

        self.k_neighbors = min(
            self.k_neighbors,
            len(self.memory_bank)
        )

        self.threshold = payload.get(
            "threshold"
        )

        self.calibration_percentile = payload.get(
            "calibration_percentile"
        )

        self.normal_training_stats = payload.get(
            "normal_training_stats",
            {}
        )

        self.normal_validation_stats = payload.get(
            "normal_validation_stats",
            {}
        )

        return payload.get(
            "extra_metadata",
            {}
        )