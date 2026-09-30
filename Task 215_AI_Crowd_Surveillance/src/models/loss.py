"""
Loss Functions for Weakly Supervised Action Localization and Anomaly Detection.

Implements:
1. Video-level action classification loss
2. Video-level anomaly loss
3. MIL ranking loss between anomalous and normal videos
4. Temporal smoothness regularization
5. Sparsity regularization on anomalous videos

The loss is designed for weak supervision, where training data provides
video-level labels rather than frame-level action annotations.
"""

import torch
import torch.nn as nn
from typing import Dict


class WSSTALLoss(nn.Module):
    def __init__(
    self,
    smoothness_weight: float = 0.0001,
    sparsity_weight: float = 0.0001,
    ranking_margin: float = 1.0,
    classification_weight: float = 2.0,
    anomaly_weight: float = 1.0,
    ranking_weight: float = 0.5
):
        super().__init__()

        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.ranking_margin = ranking_margin
        self.classification_weight = classification_weight
        self.anomaly_weight = anomaly_weight
        self.ranking_weight = ranking_weight

        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_labels: torch.Tensor,
        target_is_anomaly: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        outputs:
            Dictionary returned by TwoStreamWSSTALNet.

        target_labels:
            Tensor of shape [B].
            Class indices:
                0 = Normal
                1 = Fighting
                2 = Panic_Dispersal

        target_is_anomaly:
            Tensor of shape [B].
            0.0 = Normal
            1.0 = Anomalous

        Returns
        -------
        Dictionary containing the total loss and individual components.
        """

        video_anom_score = outputs["video_anomaly_score"]       # [B]
        video_cls_logits = outputs["video_action_logits"]       # [B, C]
        segment_scores = outputs["segment_anomaly_scores"]      # [B, S]

        # ---------------------------------------------------------
        # 1. Video-Level Anomaly Classification
        # ---------------------------------------------------------
        #
        # Teaches the model to distinguish normal from anomalous
        # videos at the video level.
        #
        l_anom = self.bce_loss(
            video_anom_score,
            target_is_anomaly
        )

        # ---------------------------------------------------------
        # 2. Video-Level Action Classification
        # ---------------------------------------------------------
        #
        # Teaches the model to distinguish:
        #   Normal
        #   Fighting
        #   Panic_Dispersal
        #
        l_cls = self.ce_loss(
            video_cls_logits,
            target_labels
        )

        # ---------------------------------------------------------
        # 3. MIL Ranking Loss
        # ---------------------------------------------------------
        #
        # Anomalous videos should have higher maximum segment
        # anomaly scores than normal videos.
        #
        anom_mask = target_is_anomaly > 0.5
        norm_mask = target_is_anomaly <= 0.5

        if anom_mask.any() and norm_mask.any():

            max_anom = (
                segment_scores[anom_mask]
                .max(dim=1)[0]
                .mean()
            )

            max_norm = (
                segment_scores[norm_mask]
                .max(dim=1)[0]
                .mean()
            )

            l_rank = torch.clamp(
                self.ranking_margin
                - max_anom
                + max_norm,
                min=0.0
            )

        else:
            # This can happen when a batch contains only anomalous
            # or only normal videos.
            l_rank = torch.zeros(
                (),
                device=segment_scores.device
            )

        # ---------------------------------------------------------
        # 4. Temporal Smoothness Regularization
        # ---------------------------------------------------------
        #
        # Encourages neighboring temporal segments to have
        # reasonably smooth anomaly scores.
        #
        if segment_scores.size(1) > 1:
            diff = (
                segment_scores[:, 1:]
                - segment_scores[:, :-1]
            )

            l_smooth = torch.mean(
                torch.sum(diff ** 2, dim=1)
            )
        else:
            l_smooth = torch.zeros(
                (),
                device=segment_scores.device
            )

        # ---------------------------------------------------------
        # 5. Temporal Sparsity Regularization
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        # Apply sparsity only to anomalous videos.
        #
        # We do NOT want to force Normal videos to have an
        # additional anomaly-score penalty here because their
        # anomaly target is already handled by BCE loss.
        #
        if anom_mask.any():
            anomalous_segment_scores = segment_scores[anom_mask]

            l_sparse = torch.mean(
                torch.sum(anomalous_segment_scores, dim=1)
            )
        else:
            l_sparse = torch.zeros(
                (),
                device=segment_scores.device
            )

        # ---------------------------------------------------------
        # 6. Total Loss
        # ---------------------------------------------------------
        #
        # Main objectives:
        #   classification + anomaly detection + MIL ranking
        #
        # Small regularization terms:
        #   temporal smoothness + sparsity
        #
        total_loss = (
    (self.classification_weight * l_cls)
    + (self.anomaly_weight * l_anom)
    + (self.ranking_weight * l_rank)
    + (self.smoothness_weight * l_smooth)
    + (self.sparsity_weight * l_sparse)
)

        return {
            "total_loss": total_loss,
            "loss_cls": l_cls,
            "loss_anom": l_anom,
            "loss_rank": l_rank,
            "loss_smooth": l_smooth,
            "loss_sparse": l_sparse
        }