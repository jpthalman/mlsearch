import math
from typing_extensions import Self

import torch
from torch import nn

from data.data_module import Dim
from model.transformer_block import (
    SelfAttentionBlock,
    LatentQueryAttentionBlock,
    DynamicLatentQueryAttentionBlock,
)


class SceneEncoder(nn.Module):
    def __init__(
        self: Self,
        *,
        num_layers: int,
        embed_dim: int,
        hidden_mult: int,
        latent_query_ratio: float,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        assert num_layers > 1
        assert embed_dim > 0
        assert hidden_mult > 0
        assert 0.0 < latent_query_ratio < 1.0

        # TODO: Try RFF + Embedding projection
        self.agent_history_proj = nn.Linear(Dim.S, embed_dim)
        self.agent_interactions_proj = nn.Linear(Dim.S, embed_dim)
        self.roadgraph_proj = nn.Linear(Dim.Rd, embed_dim)

        layers = []

        L = Dim.T * (1 + Dim.Ai + Dim.R)
        lq_length = math.ceil(latent_query_ratio * L)
        lq_layer = LatentQueryAttentionBlock(
            latent_query_length=lq_length,
            input_dim=embed_dim,
            hidden_dim=hidden_mult * embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        layers.append(lq_layer)

        for _ in range(num_layers - 2):
            layers.append(SelfAttentionBlock(
                input_dim=embed_dim,
                hidden_dim=hidden_mult * embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            ))

        dlq_layer = DynamicLatentQueryAttentionBlock(
            sequence_length=lq_length,
            latent_query_length=Dim.T,
            input_dim=embed_dim,
            hidden_dim=hidden_mult * embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        layers.append(dlq_layer)
        self.encoders = nn.Sequential(*layers)

    def forward(
        self: Self,
        *,
        agent_history: torch.Tensor,
        agent_interactions: torch.Tensor,
        agent_mask: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_interactions[B, A, T, Ai, S]
        agent_mask[B, A, T]
        roadgraph[B, A, 1, R, Rd]
        """
        # x[B, A, T, 1+Ai+R, D]
        x = self._early_fusion(agent_history, agent_interactions, roadgraph)
        B, A, T, S, D = x.shape
        # x[B*A, T*(1+Ai*R), D]
        x = x.view(B*A, T*S, D)
        out = self.encoders(x)
        return out.view(B, A, Dim.T, -1)

    def _early_fusion(
        self: Self,
        agent_history: torch.Tensor,
        agent_interactions: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_interactions[B, A, T, Ai, S]
        roadgraph[B, A, 1, R, Rd]
        """
        return torch.cat(
            [
                self.agent_history_proj(agent_history).relu_(),
                self.agent_interactions_proj(agent_interactions).relu_(),
                self.roadgraph_proj(roadgraph).relu_().repeat(1, 1, Dim.T, 1, 1),
            ],
            dim=-2,
        )
