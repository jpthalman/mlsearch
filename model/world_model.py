from typing_extensions import Self

import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self.naive = nn.Linear(128, 128)

    def forward(
        self: Self,
        scene_embedding: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """
        scene_embedding[B, A, T, E]
        controls[B, A, T, C]
        """
        return self.naive(scene_embedding)
