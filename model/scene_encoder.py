import math
from typing_extensions import Self

import torch
from torch import nn

from data.config import Dim
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
    LatentQueryAttentionBlock,
    SelfAttentionBlock,
    TransformerBlock,
    TransformerConfig,
)


class MixerBlock(nn.Module):
    def __init__(
        self: Self,
        num_layers: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_layers > 0

        self.cross_attn = TransformerBlock(config=config)
        self.self_attn = nn.ModuleList([
            SelfAttentionBlock(config=config) for _ in range(num_layers - 1)
        ])

    def forward(
        self: Self,
        agent_embed: torch.Tensor,
        roadgraph_embed: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.cross_attn(agent_embed, roadgraph_embed, cross_attn_mask)
        for block in self.self_attn:
            x = block(x, self_attn_mask)
        return x


class SceneEncoder(nn.Module):
    def __init__(
        self: Self,
        *,
        num_blocks: int,
        num_self_attention_layers: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        assert num_blocks > 0
        assert num_self_attention_layers > 0

        self.E = config.embed_dim

        # TODO: Try RFF + Embedding projection
        self.agent_history_proj = nn.Linear(Dim.S, self.E)
        self.agent_interactions_proj = nn.Linear(Dim.S, self.E)
        self.roadgraph_proj = nn.Linear(Dim.Rd, self.E)

        self.blocks = nn.ModuleList([
            MixerBlock(num_self_attention_layers, config)
            for _ in range(num_blocks)
        ])
        self.output_layer = DynamicLatentQueryAttentionBlock(
            sequence_length=Dim.T*(Dim.Ai+1),
            latent_query_length=Dim.T,
            config=config,
        )

    def forward(
        self: Self,
        *,
        agent_history: torch.Tensor,
        agent_history_mask: torch.Tensor,
        agent_interactions: torch.Tensor,
        agent_interactions_mask: torch.Tensor,
        roadgraph: torch.Tensor,
        roadgraph_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_history_mask[B, A, T]
        agent_interactions[B, A, T, Ai, S]
        agent_interactions_mask[B, A, T, Ai]
        roadgraph[B, R, Rd]
        roadgraph_mask[B, R]
        """
        B = agent_history.shape[0]

        # agent_embed[B*A, T*(Ai+1), E]
        # agent_mask[B*A, T*(Ai+1)]
        # self_attn_mask[B*A, T*(Ai+1), T*(Ai+1)]
        agent_embed, agent_mask, self_attn_mask = self._embed_agents(
            agent_history,
            agent_history_mask,
            agent_interactions,
            agent_interactions_mask,
        )

        agent_mask = agent_mask.logical_not().unsqueeze(-1)
        roadgraph_mask = roadgraph_mask.logical_not().unsqueeze(-2)
        roadgraph_mask = roadgraph_mask.repeat(Dim.A, 1, 1)
        cross_attn_mask = (agent_mask * roadgraph_mask).logical_not()

        # roadgraph_embed[B, R, E]
        roadgraph_embed = self.roadgraph_proj(roadgraph)
        # roadgraph_embed[B*A, R, E]
        roadgraph_embed = roadgraph_embed.repeat(Dim.A, 1, 1)

        for block in self.blocks:
            agent_embed = block(
                agent_embed,
                roadgraph_embed,
                self_attn_mask,
                cross_attn_mask,
            )

        # out[B*A, T, E]
        out = self.output_layer(agent_embed)
        return out.view(B, Dim.A, Dim.T, self.E)

    def _embed_agents(
        self: Self,
        agent_history: torch.Tensor,
        agent_history_mask: torch.Tensor,
        agent_interactions: torch.Tensor,
        agent_interactions_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_history_mask[B, A, T]
        agent_interactions[B, A, T, Ai, S]
        agent_interactions_mask[B, A, T, Ai]
        """
        B = agent_history.shape[0]

        # agent_mask[B,A,T,Ai+1]
        agent_mask = torch.cat(
            [agent_history_mask.unsqueeze(-1), agent_interactions_mask],
            dim=3,
        )
        # agent_mask[B*A,T*(Ai+1)]
        agent_mask = agent_mask.view(B * Dim.A, Dim.T * (Dim.Ai + 1))

        # Computes the outer product of the last dimension with a logical_or
        # instead of multiplication. So a mask like [0,0,1,1] would go to:
        # [0 0 1 1]
        # [0 0 1 1]
        # [1 1 1 1]
        # [1 1 1 1]
        outer_agent_mask = agent_mask.logical_not()
        outer_agent_mask = (
            outer_agent_mask.unsqueeze(-1) * outer_agent_mask.unsqueeze(-2)
        )
        # outer_agent_mask[B*A, T*(Ai+1), T*(Ai+1)]
        outer_agent_mask = outer_agent_mask.logical_not()

        # Computes a block-causal mask with shape:
        # [0 0 1 1 1 1]
        # [0 0 1 1 1 1]
        # [0 0 0 0 1 1]
        # [0 0 0 0 1 1]
        # [0 0 0 0 0 0]
        # [0 0 0 0 0 0]
        # Since we are combining the spatial and temporal dimension, there are
        # N elements per time step, so we need to expand the causal mask to
        # match that.
        causal_mask = torch.ones([Dim.T, Dim.T], device=agent_mask.device).bool()
        causal_mask = torch.tril(causal_mask).logical_not()
        causal_mask = torch.repeat_interleave(causal_mask, Dim.Ai + 1, dim=0)
        causal_mask = torch.repeat_interleave(causal_mask, Dim.Ai + 1, dim=1)
        # outer_agent_mask[B,A,T*(Ai+1),T*(Ai+1)]
        causal_mask = causal_mask.unsqueeze(0).expand(B * Dim.A, -1, -1)

        # Combine the outer product mask and the block causal mask to get the
        # final output causal mask. This mask will only attend to pair-wise
        # spatio-temporal information that exists in the present or the past.
        causal_agent_mask = outer_agent_mask.logical_or(causal_mask)

        history_embed = self.agent_history_proj(agent_history)
        history_embed = history_embed.reshape(B * Dim.A, Dim.T * 1, self.E)

        interact_embed = self.agent_interactions_proj(agent_interactions)
        interact_embed = interact_embed.reshape(B * Dim.A, Dim.T * Dim.Ai, self.E)

        agent_embed = torch.cat([history_embed, interact_embed], dim=1)
        return agent_embed, agent_mask, causal_agent_mask
