"""
Two-Stream Feature Extractor for Surveillance Action Recognition.

Combines:
1. ImageNet-pretrained ResNet18 RGB appearance features
2. Multi-frame RGB temporal pooling
3. Dense Optical Flow motion features
4. Cross-modal feature fusion

The RGB branch samples multiple frames from each temporal segment
instead of relying on a single center frame.

The RGB backbone can optionally be frozen when working with small
training datasets. This helps preserve the pretrained ImageNet
representation and reduces overfitting.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class MotionEncoder(nn.Module):
    """
    Lightweight convolutional encoder for optical-flow displacement maps.

    Input:
        [B*T, 2, H_flow, W_flow]

    Output:
        [B*T, feature_dim]
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = 128
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                feature_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.net(x)


class TwoStreamFeatureExtractor(nn.Module):
    """
    Extracts and fuses RGB appearance and optical-flow motion features.

    RGB:
        Multiple sampled RGB frames
        → ImageNet-pretrained ResNet18
        → temporal mean pooling
        → 512 dimensions

    Optical Flow:
        Temporal mean flow
        → MotionEncoder
        → 128 dimensions

    Fusion:
        512 + 128
        → 256 dimensions

    The RGB backbone can be frozen to reduce overfitting on
    very small datasets.
    """

    def __init__(
        self,
        rgb_feature_dim: int = 512,
        flow_feature_dim: int = 128,
        fusion_dim: int = 256,
        pretrained: bool = True,
        freeze_rgb_backbone: bool = False
    ):
        super().__init__()

        # ---------------------------------------------------------
        # 1. RGB BACKBONE
        # ---------------------------------------------------------

        if pretrained:
            print(
                "Loading ImageNet-pretrained ResNet18 weights..."
            )

            weights = models.ResNet18_Weights.DEFAULT

        else:
            print(
                "Using randomly initialized ResNet18..."
            )

            weights = None

        base_resnet = models.resnet18(
            weights=weights
        )

        # Remove the final classification layer.
        #
        # ResNet18 output remains 512-dimensional.
        self.rgb_backbone = nn.Sequential(
            *list(base_resnet.children())[:-1]
        )

        self.rgb_dim = rgb_feature_dim

        # Number of RGB frames sampled from each temporal segment.
        self.num_rgb_samples = 4

        # ---------------------------------------------------------
        # OPTIONAL RGB BACKBONE FREEZING
        # ---------------------------------------------------------
        #
        # When enabled, ImageNet-pretrained RGB features are kept
        # fixed during training.
        #
        # This is useful for very small datasets because the
        # backbone contains many trainable parameters.
        # ---------------------------------------------------------

        self.freeze_rgb_backbone = freeze_rgb_backbone

        if self.freeze_rgb_backbone:

            print(
                "Freezing ImageNet-pretrained RGB backbone..."
            )

            for parameter in self.rgb_backbone.parameters():

                parameter.requires_grad = False

        # ---------------------------------------------------------
        # 2. OPTICAL FLOW BACKBONE
        # ---------------------------------------------------------

        self.flow_backbone = MotionEncoder(
            in_channels=2,
            feature_dim=flow_feature_dim
        )

        self.flow_dim = flow_feature_dim

        # ---------------------------------------------------------
        # 3. CROSS-MODAL FEATURE FUSION
        # ---------------------------------------------------------

        total_in = (
            self.rgb_dim +
            self.flow_dim
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                total_in,
                fusion_dim
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.15
            )
        )

        self.fusion_dim = fusion_dim

    def _sample_frame_indices(
        self,
        num_frames: int,
        device: torch.device
    ) -> torch.Tensor:

        """
        Select representative RGB frames across the whole clip.

        For a normal 16-frame clip this produces approximately:
            [0, 5, 10, 15]

        Using linspace also makes the method robust if the actual
        clip contains a different number of frames.
        """

        if num_frames <= 1:

            return torch.zeros(
                1,
                dtype=torch.long,
                device=device
            )

        num_samples = min(
            self.num_rgb_samples,
            num_frames
        )

        indices = torch.linspace(
            0,
            num_frames - 1,
            steps=num_samples,
            device=device
        ).round().long()

        return indices

    def forward(
        self,
        rgb_clips: torch.Tensor,
        flow_clips: torch.Tensor
    ) -> torch.Tensor:

        """
        Inputs
        ------

        rgb_clips:
            [B, num_segments, T, 3, H, W]

        flow_clips:
            [B, num_segments, T, 2, H_flow, W_flow]

        Returns
        -------

        fused_features:
            [B, num_segments, fusion_dim]
        """

        B, S, T, C_rgb, H_rgb, W_rgb = (
            rgb_clips.shape
        )

        _, _, _, C_flow, H_flow, W_flow = (
            flow_clips.shape
        )

        # ---------------------------------------------------------
        # RGB TEMPORAL REPRESENTATION
        # ---------------------------------------------------------

        frame_indices = self._sample_frame_indices(
            T,
            rgb_clips.device
        )

        sampled_rgb = rgb_clips[
            :,
            :,
            frame_indices
        ]

        K = sampled_rgb.shape[2]

        # Flatten B, S and K so ResNet processes all sampled
        # frames independently.
        rgb_frames = sampled_rgb.reshape(
            B * S * K,
            C_rgb,
            H_rgb,
            W_rgb
        )

        # ---------------------------------------------------------
        # RGB FEATURE EXTRACTION
        # ---------------------------------------------------------
        #
        # If the backbone is frozen, gradients are not needed.
        # This reduces memory usage and prevents updates to the
        # pretrained ResNet weights.
        # ---------------------------------------------------------

        if self.freeze_rgb_backbone:

            with torch.no_grad():

                rgb_frame_features = (
                    self.rgb_backbone(
                        rgb_frames
                    ).flatten(1)
                )

        else:

            rgb_frame_features = (
                self.rgb_backbone(
                    rgb_frames
                ).flatten(1)
            )

        # Restore temporal sample dimension.
        rgb_frame_features = (
            rgb_frame_features.reshape(
                B * S,
                K,
                self.rgb_dim
            )
        )

        # Aggregate appearance information across sampled frames.
        rgb_feat = rgb_frame_features.mean(
            dim=1
        )

        # ---------------------------------------------------------
        # OPTICAL FLOW REPRESENTATION
        # ---------------------------------------------------------
        #
        # Keep the existing motion representation unchanged.
        # Flow is averaged over the temporal clip.
        # ---------------------------------------------------------

        flow_mean = flow_clips.mean(
            dim=2
        ).reshape(
            B * S,
            C_flow,
            H_flow,
            W_flow
        )

        flow_feat = self.flow_backbone(
            flow_mean
        )

        # ---------------------------------------------------------
        # CROSS-MODAL FUSION
        # ---------------------------------------------------------

        combined = torch.cat(
            [
                rgb_feat,
                flow_feat
            ],
            dim=1
        )

        fused = self.fusion(
            combined
        )

        return fused.view(
            B,
            S,
            self.fusion_dim
        )