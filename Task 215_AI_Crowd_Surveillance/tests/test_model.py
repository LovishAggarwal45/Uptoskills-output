"""
Unit tests for model forward pass, temporal attention, and MIL loss.
"""

import unittest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.models.loss import WSSTALLoss

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = TwoStreamWSSTALNet(
            num_classes=3,
            rgb_feature_dim=512,
            flow_feature_dim=128,
            fusion_dim=128,
            encoder_layers=1,
            nheads=2
        )
        self.criterion = WSSTALLoss()

    def test_forward_and_loss(self):
        # B=2, S=4, T=4, H=112, W=112
        rgb = torch.randn(2, 4, 4, 3, 112, 112)
        flow = torch.randn(2, 4, 4, 2, 56, 56)

        out = self.model(rgb, flow)
        self.assertEqual(out["segment_anomaly_scores"].shape, (2, 4))
        self.assertEqual(out["segment_action_probs"].shape, (2, 4, 3))
        self.assertEqual(out["video_anomaly_score"].shape, (2,))
        self.assertEqual(out["video_action_logits"].shape, (2, 3))

        target_cls = torch.tensor([1, 0], dtype=torch.long)
        target_anom = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss_dict = self.criterion(out, target_cls, target_anom)
        self.assertIn("total_loss", loss_dict)
        self.assertGreater(loss_dict["total_loss"].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
