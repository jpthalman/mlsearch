from typing_extensions import Self

import torch
from torch import nn

from data.data_module import Dim


class ControlPredictor(nn.Module):
    def __init__(
        self: Self,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.naive = nn.Linear(embed_dim, Dim.Cd**2)

    def forward(self: Self, scene_embedding: torch.Tensor) -> torch.Tensor:
        """
        scene_embedding[B, A, T, E]
        """
        return self.naive(scene_embedding)

