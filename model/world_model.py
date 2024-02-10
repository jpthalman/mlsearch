from typing_extensions import Self

import torch
from torch import nn

from data.dimensions import Dim
from model.dense_block import DenseBlock
from model.transformer_block import (
    DynamicLatentQueryAttentionBlock,
    SelfAttentionBlock,
    TransformerConfig,
)


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
            SelfAttentionBlock(config=config) for _ in range(num_layers)
        ])
        self.control_encoder = DenseBlock(
            input_dim=Dim.C,
            hidden_dim=self.E,
            output_dim=self.E,
            dropout=config.dropout,
        )

    def forward(
        self: Self,
        scene_embedding: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """
        scene_embedding[B, A, T, E]
        controls[B, T-1, C]
        """
        B = scene_embedding.shape[0]

        # x[B, (A+1)*T, E]
        controls = torch.cat([controls, controls[:, -1:, :]], dim=1)
        x = torch.cat(
            [
                scene_embedding.view(B, Dim.A * Dim.T, self.E),
                self.control_encoder(controls),
            ],
            dim=1,
        )

        # Computes a block-causal mask with shape:
        # [0 0 1 1 1 1]
        # [0 0 1 1 1 1]
        # [0 0 0 0 1 1]
        # [0 0 0 0 1 1]
        # [0 0 0 0 0 0]
        # [0 0 0 0 0 0]
        # Since we are combining the agent and temporal dimension, there are
        # A elements per time step, so we need to expand the causal mask to
        # match that.
        causal_mask = torch.ones([Dim.T, Dim.T], device=controls.device).bool()
        causal_mask = torch.tril(causal_mask).logical_not()
        causal_mask = torch.repeat_interleave(causal_mask, Dim.A + 1, dim=0)
        causal_mask = torch.repeat_interleave(causal_mask, Dim.A + 1, dim=1)
        # causal_mask[B, (A+1)*T, (A+1)*T]
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

        # x[B, (A+1)*T, E]
        for encoder in self.encoders:
            x = encoder(x, causal_mask)

        # x[B, A*T, E]
        x = x[:, :(Dim.A * Dim.T), :]
        return x.view(B, Dim.A, Dim.T, self.E)
