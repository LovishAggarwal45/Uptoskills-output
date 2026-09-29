"""
Unit & System Integration Tests for VideoMAE Crowd Outlier Localization System.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import numpy as np
from models.videomae import VideoMAE, PatchEmbed3D, get_3d_sincos_pos_embed
from dataset.video_dataset import VideoClipDataset
from dataset.generate_sample_video import SyntheticCrowdGenerator
from trainer import train_videomae
from detector.outlier_detector import CrowdOutlierDetector


def test_pos_embedding():
    pos_emb = get_3d_sincos_pos_embed(embed_dim=128, grid_size_t=8, grid_size_h=8, grid_size_w=8)
    assert pos_emb.shape == (512, 128), f"Unexpected Pos Embedding shape: {pos_emb.shape}"


def test_videomae_model_forward():
    model = VideoMAE(
        img_size=(128, 128),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=128,
        encoder_depth=2,
        decoder_depth=2
    )
    dummy_video = torch.randn(2, 3, 16, 128, 128)
    pred, mask, target = model(dummy_video)

    assert pred.shape == (2, 512, 1536), f"Unexpected pred shape: {pred.shape}"
    assert mask.shape == (2, 512), f"Unexpected mask shape: {mask.shape}"
    assert target.shape == (2, 512, 1536), f"Unexpected target shape: {target.shape}"

    loss = model.compute_loss(dummy_video)
    assert loss.item() > 0, "Loss should be positive float"

    amap, rec, pmse = model.compute_spatiotemporal_anomaly_map(dummy_video, num_mask_passes=1)
    assert amap.shape == (2, 8, 8, 8), f"Unexpected anomaly map shape: {amap.shape}"


def test_end_to_end_pipeline():
    # 1. Generate small test videos
    gen = SyntheticCrowdGenerator(width=160, height=120, num_pedestrians=20)
    normal_vpath = "outputs/test_normal.mp4"
    anomaly_vpath = "outputs/test_anomaly.mp4"
    gen.generate_normal_video(normal_vpath, total_frames=40)
    gen.generate_anomaly_video(anomaly_vpath, total_frames=60)

    assert os.path.exists(normal_vpath)
    assert os.path.exists(anomaly_vpath)

    # 2. Dataset loading
    dataset = VideoClipDataset(video_paths=[normal_vpath], num_frames=16, stride=4, image_size=(128, 128))
    assert len(dataset) > 0, "Dataset should have at least 1 clip"

    # 3. Model training for 1 epoch
    ckpt_path = train_videomae(config_path="config.yaml", video_paths=[normal_vpath], epochs=1)
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"

    # 4. Outlier detection & annotation
    detector = CrowdOutlierDetector(config_path="config.yaml")
    out_vpath, scores = detector.process_video(video_path=anomaly_vpath, output_path="outputs/test_annotated.mp4")

    assert os.path.exists(out_vpath)
    assert os.path.exists("outputs/anomaly_events.csv")
    assert os.path.exists("outputs/anomaly_timeline.png")
    assert os.path.exists("outputs/spatial_heatmap_summary.png")
    print("\n[TEST PASSED] All end-to-end integration tests completed successfully!")


if __name__ == "__main__":
    test_pos_embedding()
    test_videomae_model_forward()
    test_end_to_end_pipeline()
