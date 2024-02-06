from typing_extensions import Self

import torch
from torch import nn

from data.dimensions import Dim
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
    TransformerConfig,
)


class ControlPredictor(nn.Module):
    def __init__(
        self: Self,
        *,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.control_to_agent_attn = DynamicLatentQueryAttentionBlock(
            sequence_length=Dim.A,
            latent_query_length=Dim.Cd**2,
            config=config,
        )
        self.regression = nn.Linear(config.embed_dim, 1)

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
