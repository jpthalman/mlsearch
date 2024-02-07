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
        self.encoders = nn.Sequential(*[
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
        controls[B, T, C]
        """
        B = scene_embedding.shape[0]
        # x[B,T*A+(T-1),E]
        x = torch.cat(
            [
                scene_embedding.view(B, Dim.A * Dim.T, self.E),
                self.control_encoder(controls),
            ],
            dim=1,
        )
        # x[B,A*T+(T-1),E]
        out = self.encoders(x)
        # x[B,A*T,E]
        out = out[:, :(Dim.T * Dim.A), :]
        return out.view(B, Dim.A, Dim.T, self.E)
