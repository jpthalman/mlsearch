import math
from typing_extensions import Self

import torch
from torch import nn

from data.dimensions import Dim
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
    LatentQueryAttentionBlock,
    SelfAttentionBlock,
    TransformerConfig,
)


class SceneEncoder(nn.Module):
    def __init__(
        self: Self,
        *,
        num_layers: int,
        latent_query_ratio: float,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_layers > 0
        assert 0.0 < latent_query_ratio < 1.0

        self.E = config.embed_dim

        # TODO: Try RFF + Embedding projection
        self.agent_history_proj = nn.Linear(Dim.S, self.E)
        self.agent_interactions_proj = nn.Linear(Dim.S, self.E)
        self.roadgraph_proj = nn.Linear(Dim.Rd, self.E)

        layers = []
        if num_layers == 1:
            layers.append(LatentQueryAttentionBlock(
                latent_query_length=Dim.T,
                config=config,
            ))
        else:
            L = Dim.T * (1 + Dim.Ai) + Dim.R
            lq_length = math.ceil(latent_query_ratio * L)
            layers.append(LatentQueryAttentionBlock(
                latent_query_length=lq_length,
                config=config,
            ))

            for _ in range(num_layers - 2):
                layers.append(SelfAttentionBlock(config=config))

            layers.append(DynamicLatentQueryAttentionBlock(
                sequence_length=lq_length,
                latent_query_length=Dim.T,
                config=config,
            ))

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
        B = agent_history.shape[0]
        # x[B*A, T*(1+Ai+R/10), D]
        x = self._early_fusion(agent_history, agent_interactions, roadgraph)
        out = self.encoders(x)
        return out.view(B, Dim.A, Dim.T, self.E)

    def _early_fusion(
        self: Self,
        agent_history: torch.Tensor,
        agent_interactions: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_interactions[B, A, T, Ai, S]
        roadgraph[B, 1, 1, R, Rd]
        """
        B = agent_history.shape[0]

        agent_embed = self.agent_history_proj(agent_history)
        agent_embed = agent_embed.reshape(B * Dim.A, Dim.T * 1, self.E)

        interact_embed = self.agent_interactions_proj(agent_interactions)
        interact_embed = interact_embed.reshape(B * Dim.A, Dim.T * Dim.Ai, self.E)

        roadgraph_embed = self.roadgraph_proj(roadgraph)
        roadgraph_embed = roadgraph_embed.view(B * Dim.A, Dim.R, self.E)

        return torch.cat([agent_embed, interact_embed, roadgraph_embed], dim=1)
