from typing_extensions import Self

import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(
        self: Self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.w0 = nn.Linear(input_dim, hidden_dim)
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.w0(x) * nn.functional.silu(self.w1(x)))
        return self.dropout(x)
