from typing_extensions import Self

import torch
from torch import nn

from data.data_module import Dim
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
)


class ControlPredictor(nn.Module):
    def __init__(
        self: Self,
        *,
        embed_dim: int,
        hidden_mult: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.control_to_agent_attn = DynamicLatentQueryAttentionBlock(
            sequence_length=Dim.A,
            latent_query_length=Dim.Cd**2,
            input_dim=embed_dim,
            hidden_dim=hidden_mult * embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.regression = nn.Linear(embed_dim, 1)

    def forward(self: Self, scene_embedding: torch.Tensor) -> torch.Tensor:
        """
        scene_embedding[B, A, T, E]
        """
        B = scene_embedding.shape[0]
        # x[B, T, A, E]
        x = scene_embedding.transpose(1, 2)
        # x[B*T, A, E]
        x = x.reshape(B * Dim.T, Dim.A, -1)
        # x[B*T, Cd^2, E]
        x = self.control_to_agent_attn(x)
        # x[B*T, Cd^2]
        x = self.regression(x)
        return x.view(B, Dim.T, Dim.Cd**2)
