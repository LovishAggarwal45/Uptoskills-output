"""
Temporal Encoder for Weakly Supervised Action Localization.
Applies Multi-Head Self-Attention / Transformer layers over temporal segment features
to model long-range dependencies and temporal context.
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.15):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, S, d_model]
        Output: [B, S, d_model]
        """
        x_pos = self.pos_encoder(x)
        out = self.transformer(x_pos)
        return self.layer_norm(out)
