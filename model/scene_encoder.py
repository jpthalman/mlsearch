from typing_extensions import Self

import torch
from torch import nn


class SceneEncoder(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self.naive = nn.Linear(42, 128)

    def forward(
        self: Self,
        agent_history: torch.Tensor,
        agent_interactions: torch.Tensor,
        agent_mask: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_history[B, A, T, 1, S]
        agent_interactions[B, A, T, N, S]
        agent_mask[B, A, T]
        roadgraph[B, A, 1, N, E]
        """
        return self.naive(agent_history[:, :, :, 0, :])
