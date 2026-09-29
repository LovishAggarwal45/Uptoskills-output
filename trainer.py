"""
Trainer Module for Self-Supervised VideoMAE on Normal Crowd Dynamics.

Trains VideoMAE model exclusively on normal video clips using random tubelet masking
and pixel-level spatio-temporal reconstruction loss.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
from torch.utils.data import DataLoader
from models.videomae import VideoMAE
from dataset.video_dataset import VideoClipDataset


def train_videomae(config_path="config.yaml", video_paths=None, epochs=None):
    """
    Train VideoMAE self-supervised model on normal crowd video files.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["system"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Trainer] CUDA not available, falling back to CPU.")
        device = "cpu"

    if video_paths is None:
        video_paths = ["data/sample_crowd_normal.mp4"]

    if epochs is None:
        epochs = cfg["training"]["epochs"]

    print(f"[Trainer] Loading dataset from: {video_paths}")
    dataset = VideoClipDataset(
        video_paths=video_paths,
        num_frames=cfg["data"]["num_frames"],
        stride=cfg["data"]["stride"],
        image_size=tuple(cfg["data"]["image_size"])
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=False
    )

    print(f"[Trainer] Dataset size: {len(dataset)} clips across {len(video_paths)} videos.")

    # Initialize VideoMAE model
    model = VideoMAE(
        img_size=tuple(cfg["data"]["image_size"]),
        patch_size=cfg["data"]["patch_size"],
        num_frames=cfg["data"]["num_frames"],
        tubelet_size=cfg["data"]["tubelet_size"],
        in_chans=cfg["data"]["channels"],
        embed_dim=cfg["model"]["embed_dim"],
        encoder_depth=cfg["model"]["encoder_depth"],
        encoder_heads=cfg["model"]["encoder_heads"],
        decoder_embed_dim=cfg["model"]["decoder_embed_dim"],
        decoder_depth=cfg["model"]["decoder_depth"],
        decoder_heads=cfg["model"]["decoder_heads"],
        masking_ratio=cfg["model"]["masking_ratio"],
        mlp_ratio=cfg["model"]["mlp_ratio"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_path = cfg["training"]["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"[Trainer] Starting VideoMAE Normality Training for {epochs} epochs on {device}...")
    model.train()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch_clips in dataloader:
            batch_clips = batch_clips.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(batch_clips)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction Loss (MSE): {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": cfg
            }, checkpoint_path)

    print(f"[Trainer] Training complete! Best Loss: {best_loss:.6f}. Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_videomae()
