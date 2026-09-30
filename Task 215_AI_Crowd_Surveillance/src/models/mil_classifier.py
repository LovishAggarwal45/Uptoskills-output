"""
Weakly Supervised Spatio-Temporal Action Localization Network.

Architecture:
    RGB + Optical Flow
        ↓
    Two-Stream Feature Extraction
        ↓
    Feature Fusion
        ↓
    Temporal Transformer
        ↓
    ├── Segment Anomaly Scores
    ├── Segment Action Classification
    └── Temporal Attention / MIL Pooling
        ↓
    Video-Level Predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.features.feature_extractor import TwoStreamFeatureExtractor
from src.models.temporal_encoder import TemporalTransformerEncoder


class TwoStreamWSSTALNet(nn.Module):
    """
    Weakly supervised two-stream spatio-temporal action localization model.

    The model receives RGB clips and optical-flow clips and produces:
        - segment-level anomaly scores
        - segment-level action probabilities
        - video-level anomaly score
        - video-level action logits
        - temporal attention weights
        - temporal encoded features
    """

    def __init__(
        self,
        num_classes: int = 3,
        rgb_feature_dim: int = 512,
        flow_feature_dim: int = 128,
        fusion_dim: int = 256,
        encoder_layers: int = 2,
        nheads: int = 4,
        dropout: float = 0.15,
        mil_top_k: int = 3,
        pretrained: bool = True,
        freeze_rgb_backbone: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mil_top_k = mil_top_k

        # ---------------------------------------------------------
        # Two-stream feature extractor
        # ---------------------------------------------------------
        self.feature_extractor = TwoStreamFeatureExtractor(
            rgb_feature_dim=rgb_feature_dim,
            flow_feature_dim=flow_feature_dim,
            fusion_dim=fusion_dim,
            pretrained=pretrained,
            freeze_rgb_backbone=freeze_rgb_backbone,
        )

        # ---------------------------------------------------------
        # Temporal Transformer
        # ---------------------------------------------------------
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=fusion_dim,
            nhead=nheads,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Segment-level anomaly detector
        # ---------------------------------------------------------
        self.anomaly_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ---------------------------------------------------------
        # Segment-level action classifier
        # ---------------------------------------------------------
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ---------------------------------------------------------
        # Temporal attention used for MIL aggregation
        # ---------------------------------------------------------
        self.attention_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        rgb_clips: torch.Tensor,
        flow_clips: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            rgb_clips:
                RGB input tensor.

            flow_clips:
                Optical-flow input tensor.

        Returns:
            Dictionary containing segment-level and video-level outputs.
        """

        # ---------------------------------------------------------
        # Extract two-stream fused features
        # ---------------------------------------------------------
        fused = self.feature_extractor(
            rgb_clips,
            flow_clips,
        )

        # ---------------------------------------------------------
        # Temporal modeling
        # ---------------------------------------------------------
        encoded = self.temporal_encoder(fused)

        # ---------------------------------------------------------
        # Segment-level anomaly scores
        # ---------------------------------------------------------
        segment_anomaly = self.anomaly_head(
            encoded
        ).squeeze(-1)

        # ---------------------------------------------------------
        # Segment-level action classification
        # ---------------------------------------------------------
        segment_action_logits = self.action_head(
            encoded
        )

        segment_action_probs = F.softmax(
            segment_action_logits,
            dim=-1,
        )

        # ---------------------------------------------------------
        # Temporal attention
        # ---------------------------------------------------------
        raw_attn = self.attention_head(
            encoded
        ).squeeze(-1)

        attn_weights = F.softmax(
            raw_attn,
            dim=-1,
        )

        # ---------------------------------------------------------
        # Multiple Instance Learning (MIL)
        #
        # Select the top-K anomaly segments and average them
        # to obtain the video-level anomaly score.
        # ---------------------------------------------------------
        sequence_length = encoded.size(1)

        k = min(
            self.mil_top_k,
            sequence_length,
        )

        topk_scores, _ = torch.topk(
            segment_anomaly,
            k=k,
            dim=1,
        )

        video_anomaly_score = topk_scores.mean(
            dim=1
        )

        # ---------------------------------------------------------
        # Attention-weighted video-level action classification
        # ---------------------------------------------------------
        video_action_logits = torch.sum(
            attn_weights.unsqueeze(-1)
            * segment_action_logits,
            dim=1,
        )

        # ---------------------------------------------------------
        # Return all outputs required by the training loss,
        # validation code, inference pipeline and diagnostics.
        # ---------------------------------------------------------
        return {
            "segment_anomaly_scores": segment_anomaly,
            "segment_action_probs": segment_action_probs,
            "segment_action_logits": segment_action_logits,
            "video_anomaly_score": video_anomaly_score,
            "video_action_logits": video_action_logits,
            "attention_weights": attn_weights,
            "features": encoded,
        }