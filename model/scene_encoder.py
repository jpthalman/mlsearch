import math
from typing_extensions import Self

import torch
from torch import nn

from data.config import Dim, POS_SCALE, VEL_SCALE
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
    LatentQueryAttentionBlock,
    SelfAttentionBlock,
    TransformerBlock,
    TransformerConfig,
)


class EncoderBlock(nn.Module):
    def __init__(self: Self, config: TransformerConfig) -> None:
        super().__init__()
        self.temporal_self_attn = SelfAttentionBlock(config=config)
        self.map_cross_attn = TransformerBlock(config=config)
        self.social_self_attn = SelfAttentionBlock(config=config)

    def forward(
        self: Self,
        agent_embed: torch.Tensor,
        roadgraph_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_embed[B, A, T, E]
        roadgraph_embed[B, R, E]
        """
        B = agent_embed.shape[0]
        T = agent_embed.shape[2]
        E = agent_embed.shape[3]

        causal_mask = torch.ones([T, T], device=agent_embed.device).bool()
        causal_mask = torch.tril(causal_mask).logical_not()
        causal_mask = causal_mask.unsqueeze(0).expand(B * Dim.A, -1, -1)

        # Causal self-attention over time
        agent_embed = agent_embed.view(B * Dim.A, T, E)
        agent_embed = self.temporal_self_attn(agent_embed, mask=causal_mask)
        agent_embed = agent_embed.view(B, Dim.A, T, E)

        # Cross-attention with map
        agent_embed = agent_embed.transpose(1, 2).reshape(B * T, Dim.A, E)
        roadgraph_embed = roadgraph_embed.repeat(T, 1, 1)
        agent_embed = self.map_cross_attn(agent_embed, roadgraph_embed)

        # Social self-attention over agents
        agent_embed = self.social_self_attn(agent_embed)
        agent_embed = agent_embed.view(B, T, Dim.A, E).transpose(1, 2)
        return agent_embed.contiguous()


class RoadgraphEncoder(nn.Module):
    def __init__(
        self: Self,
        num_blocks: int,
        downsample_frac: float,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_blocks > 0
        assert 0 < downsample_frac < 1

        self.E = config.embed_dim
        self.proj = nn.Linear(Dim.Rd, self.E)
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(config=config) for _ in range(num_blocks - 1)
        ])
        self.lq_block = LatentQueryAttentionBlock(
            latent_query_length=math.ceil(downsample_frac * Dim.R),
            config=config,
        )

    def forward(self: Self, roadgraph: torch.Tensor) -> torch.Tensor:
        embed = self.proj(roadgraph)
        embed = self.blocks(embed)
        return self.lq_block(embed)


class SceneEncoder(nn.Module):
    def __init__(
        self: Self,
        *,
        num_agent_blocks: int,
        num_roadgraph_blocks: int,
        roadgraph_downsample_frac: float,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_agent_blocks > 0

        self.E = config.embed_dim
        self.roadgraph_encoder = RoadgraphEncoder(
            num_roadgraph_blocks,
            roadgraph_downsample_frac,
            config,
        )
        self.agent_proj = nn.Linear(Dim.S, self.E)
        self.agent_blocks = nn.ModuleList([
            EncoderBlock(config) for _ in range(num_agent_blocks)
        ])

    def forward(
        self: Self,
        agent_history: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, S]
        roadgraph[B, R, Rd]
        """
        agent_embed = self.agent_proj(agent_history)
        roadgraph_embed = self.roadgraph_encoder(roadgraph)
        for block in self.agent_blocks:
            agent_embed = block(agent_embed, roadgraph_embed)
        return agent_embed
