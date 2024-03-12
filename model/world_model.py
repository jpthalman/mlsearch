from typing_extensions import Self

import torch
from torch import nn

from data.config import Dim
from model.transformer_block import (
    SelfAttentionBlock,
    TransformerBlock,
    TransformerConfig,
)


class EncoderBlock(nn.Module):
    def __init__(self: Self, config: TransformerConfig) -> None:
        super().__init__()
        self.E = config.embed_dim
        self.self_attn = SelfAttentionBlock(config=config)
        self.cross_attn = TransformerBlock(config=config)

    def forward(
        self: Self,
        scene_embedding: torch.Tensor,
        ego_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        scene_embedding[B, T, A, E]
        ego_embedding[B, T, E]
        """
        B = scene_embedding.shape[0]
        T = scene_embedding.shape[1]

        scene_embedding = scene_embedding.view(B*T, Dim.A, self.E)
        ego_embedding = ego_embedding.unsqueeze(2).view(B*T, 1, self.E)

        scene_embedding = self.cross_attn(scene_embedding, ego_embedding)
        scene_embedding = self.self_attn(scene_embedding)
        return scene_embedding.view(B, T, Dim.A, self.E)


class WorldModel(nn.Module):
    def __init__(
        self: Self,
        *,
        num_layers: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_layers > 0

        self.E = config.embed_dim
        self.encoders = nn.ModuleList([
            EncoderBlock(config=config) for _ in range(num_layers)
        ])
        self.state_encoder = nn.Linear(Dim.S, self.E)

    def forward(
        self: Self,
        scene_embedding: torch.Tensor,
        next_ego_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        scene_embedding[B, A, T, E]
        next_ego_state[B, T, S]
        """
        # ego_embedding[B, T, E]
        ego_embedding = self.state_encoder(next_ego_state)
        # scene_embedding[B, T, A, E]
        scene_embedding = scene_embedding.transpose(1, 2).contiguous()

        for encoder in self.encoders:
            scene_embedding = encoder(scene_embedding, ego_embedding)

        # scene_embedding[B, A, T, E]
        return scene_embedding.transpose(1, 2).contiguous()
