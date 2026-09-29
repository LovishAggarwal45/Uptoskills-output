"""
Self-Supervised Video Masked Autoencoder (VideoMAE) for Crowd Outlier Localization.

Extracts 3D spatio-temporal video tube patches (tubelets), applies spatio-temporal
positional encodings, masks a high percentage of tubelets during self-supervised learning,
and measures reconstruction MSE to localize crowd anomalies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_3d_sincos_pos_embed(embed_dim, grid_size_t, grid_size_h, grid_size_w):
    """
    Generate 3D sine-cosine positional embedding for spatio-temporal tubelets.
    
    Args:
        embed_dim (int): Embedding dimension (must be even).
        grid_size_t (int): Temporal grid size (T / tubelet_size).
        grid_size_h (int): Spatial height grid size (H / patch_size).
        grid_size_w (int): Spatial width grid size (W / patch_size).
        
    Returns:
        torch.Tensor: Positional embeddings of shape (grid_size_t * grid_size_h * grid_size_w, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."
    
    # Assign even channel dimensions to T, H, W so they sum precisely to embed_dim
    dim_t = (embed_dim // 3) // 2 * 2
    dim_h = (embed_dim // 3) // 2 * 2
    dim_w = embed_dim - dim_t - dim_h

    # Temporal coordinates
    grid_t = torch.arange(grid_size_t, dtype=torch.float32)
    # Spatial coordinates
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)

    # Meshgrid across (T, H, W)
    grid_t, grid_h, grid_w = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')

    def get_1d_sincos_pos_embed_from_grid(dim, pos):
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

    emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid_t)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid_w)

    pos_embed = torch.cat([emb_t, emb_h, emb_w], dim=1)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """
    3D Spatio-Temporal Patch Embedding (Tubelet Embedding).
    Converts (B, C, T, H, W) video tensor into 3D tubelet tokens.
    """
    def __init__(self, img_size=(128, 128), patch_size=16, num_frames=16, tubelet_size=2, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        
        self.grid_t = num_frames // tubelet_size
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_patches = self.grid_t * self.grid_h * self.grid_w

        # 3D convolution for projection: kernel_size = (tubelet_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, grid_t, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Multi-Head Self-Attention + MLP Transformer Layer."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VideoMAE(nn.Module):
    """
    Self-Supervised Video Masked Autoencoder for Crowd Outlier Localization.
    """
    def __init__(
        self,
        img_size=(128, 128),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=128,
        encoder_depth=4,
        encoder_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_heads=4,
        masking_ratio=0.75,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.masking_ratio = masking_ratio
        
        # Patch & Tubelet embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_pixels = in_chans * tubelet_size * patch_size * patch_size

        # Positional Encodings (Encoder & Decoder)
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim, self.patch_embed.grid_t, self.patch_embed.grid_h, self.patch_embed.grid_w
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, num_patches, embed_dim)

        dec_pos_embed = get_3d_sincos_pos_embed(
            decoder_embed_dim, self.patch_embed.grid_t, self.patch_embed.grid_h, self.patch_embed.grid_w
        )
        self.register_buffer("decoder_pos_embed", dec_pos_embed.unsqueeze(0))  # (1, num_patches, decoder_embed_dim)

        # ViT Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, mlp_ratio) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Encoder-to-Decoder projection
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask Token for Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # ViT Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Final pixel reconstruction projection
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_pixels)

    def patchify(self, videos):
        """
        Convert (B, C, T, H, W) video to patch vector sequence (B, num_patches, patch_pixels).
        """
        B, C, T, H, W = videos.shape
        p = self.patch_size
        t = self.tubelet_size
        
        grid_t = T // t
        grid_h = H // p
        grid_w = W // p

        # Reshape to (B, grid_t, t, C, grid_h, p, grid_w, p)
        x = videos.reshape(B, C, grid_t, t, grid_h, p, grid_w, p)
        x = x.permute(0, 2, 4, 6, 3, 1, 5, 7)  # (B, grid_t, grid_h, grid_w, t, C, p, p)
        x = x.reshape(B, grid_t * grid_h * grid_w, t * C * p * p)
        return x

    def unpatchify(self, patches):
        """
        Convert (B, num_patches, patch_pixels) back to (B, C, T, H, W) video tensor.
        """
        B = patches.shape[0]
        p = self.patch_size
        t = self.tubelet_size
        grid_t = self.patch_embed.grid_t
        grid_h = self.patch_embed.grid_h
        grid_w = self.patch_embed.grid_w
        C = self.in_chans

        x = patches.reshape(B, grid_t, grid_h, grid_w, t, C, p, p)
        x = x.permute(0, 5, 1, 4, 2, 6, 3, 7)  # (B, C, grid_t, t, grid_h, p, grid_w, p)
        x = x.reshape(B, C, grid_t * t, grid_h * p, grid_w * p)
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform random tubelet masking per sample.
        
        Args:
            x: Patch tokens of shape (B, N, D)
            mask_ratio: Float masking ratio
            
        Returns:
            x_masked: Kept visible tokens (B, N_keep, D)
            mask: Binary mask (B, N), 0 is kept, 1 is masked
            ids_restore: Re-ordering indices to restore full sequence (B, N)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # Noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending: small values keep, large values remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle mask to match original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Embed video and pass ONLY unmasked tubelets through ViT encoder.
        """
        # Embed patches
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add 3D positional encoding
        x = x + self.pos_embed

        # Apply random masking
        if mask_ratio > 0:
            x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x_masked = x
            mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)

        # Pass through ViT Encoder blocks
        for block in self.encoder_blocks:
            x_masked = block(x_masked)
        x_masked = self.encoder_norm(x_masked)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x_visible, ids_restore):
        """
        Pass visible encoder tokens + learnable [MASK] tokens through ViT decoder.
        """
        # Project encoder embeddings to decoder dimension
        x_visible = self.encoder_to_decoder(x_visible)

        # Append mask tokens for missing positions
        B, N_keep, D_dec = x_visible.shape
        N_full = self.num_patches
        
        mask_tokens = self.mask_token.repeat(B, N_full - N_keep, 1)
        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        
        # Restore original spatial-temporal tubelet sequence order
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec)
        )

        # Add decoder 3D positional embedding
        x_full = x_full + self.decoder_pos_embed

        # Pass through decoder Transformer blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict target pixel values for each patch
        pred = self.decoder_pred(x_full)  # (B, N_full, patch_pixels)
        return pred

    def forward(self, videos, mask_ratio=None):
        """
        Forward pass for training/evaluation.
        
        Returns:
            pred: Patch pixel predictions (B, N, patch_pixels)
            mask: Binary mask of shape (B, N), where 1 = masked patch
            target: Ground truth patch pixel values (B, N, patch_pixels)
        """
        if mask_ratio is None:
            mask_ratio = self.masking_ratio

        target = self.patchify(videos)
        x_visible, mask, ids_restore = self.forward_encoder(videos, mask_ratio)
        pred = self.forward_decoder(x_visible, ids_restore)
        return pred, mask, target

    def compute_loss(self, videos, mask_ratio=None):
        """
        Computes mean squared error (MSE) loss on MASKED patches.
        """
        pred, mask, target = self.forward(videos, mask_ratio)

        # Calculate element-wise MSE loss per patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) - mean loss over patch pixels

        # Measure loss only on masked tubelet patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def compute_spatiotemporal_anomaly_map(self, videos, num_mask_passes=3):
        """
        Inference routine for outlier localization using VideoMAE masked tubelet reconstruction.
        Runs multiple masking passes (masking ratio = 0.75) to evaluate patch reconstruction MSE
        under self-supervised VideoMAE tubelet masking.
        
        Returns:
            anomaly_map_3d: (B, grid_t, grid_h, grid_w) tensor of patch reconstruction MSE scores
            reconstructed_video: (B, C, T, H, W) reconstructed video tensor
            patch_mse_final: (B, N) tensor of per-patch MSE scores
        """
        self.eval()
        with torch.no_grad():
            target = self.patchify(videos)
            B, N, _ = target.shape

            accumulated_mse = torch.zeros((B, N), device=videos.device)
            mask_counts = torch.zeros((B, N), device=videos.device)

            for _ in range(num_mask_passes):
                pred, mask, _ = self.forward(videos, mask_ratio=self.masking_ratio)
                patch_mse = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
                
                # Accumulate MSE on masked patches
                accumulated_mse += patch_mse * mask
                mask_counts += mask

            # Fallback for any unmasked patches across passes
            unmasked = (mask_counts == 0)
            if unmasked.any():
                pred_unmasked, _, _ = self.forward(videos, mask_ratio=0.0)
                patch_mse_unmasked = ((pred_unmasked - target) ** 2).mean(dim=-1)
                accumulated_mse[unmasked] = patch_mse_unmasked[unmasked]
                mask_counts[unmasked] = 1.0

            patch_mse_final = accumulated_mse / mask_counts

            grid_t = self.patch_embed.grid_t
            grid_h = self.patch_embed.grid_h
            grid_w = self.patch_embed.grid_w

            anomaly_map_3d = patch_mse_final.reshape(B, grid_t, grid_h, grid_w)
            reconstructed_video = self.unpatchify(pred)

            return anomaly_map_3d, reconstructed_video, patch_mse_final
