from typing_extensions import Self

import attrs
import torch
from torch import nn

from model.dense_block import DenseBlock


@attrs.frozen(kw_only=True)
class TransformerConfig:
    embed_dim: int
    hidden_dim: int
    num_heads: int
    dropout: float


class TransformerBlock(nn.Module):
    def __init__(
        self: Self,
        *,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.dense = DenseBlock(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embed_dim,
            dropout=config.dropout,
        )
        self.attn_norm = nn.LayerNorm(config.embed_dim)
        self.dense_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x[B, N, D]
        y[B, M, D]
        """
        if mask is not None:
            mask = torch.repeat_interleave(mask, self.config.num_heads, dim=0)
            mask = mask.to(x.dtype)
            mask.masked_fill_(mask == 1.0, -(2**15))
        # z[B, N, D]
        z = self.attn(x, y, y, attn_mask=mask, need_weights=False)[0]
        z = self.attn_norm(x + z)
        return self.dense_norm(z + self.dense(z))


class SelfAttentionBlock(nn.Module):
    def __init__(
        self: Self,
        *,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(config=config)

    def forward(
        self: Self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x[B, N, D]
        """
        return self.transformer(x, x, mask)


class LatentQueryAttentionBlock(nn.Module):
    def __init__(
        self: Self,
        *,
        latent_query_length: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(config=config)
        self.latent_query = nn.Parameter(
            torch.empty([1, latent_query_length, config.embed_dim])
        )

        # Initialize the weights of the latent query
        gen = torch.Generator()
        gen.manual_seed(42)
        k = latent_query_length**-0.5
        self.latent_query.data.uniform_(-k, k, generator=gen)

    def forward(
        self: Self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x[B, N, D]
        """
        # q[B, L, D]
        q = self.latent_query.repeat(x.shape[0], 1, 1)
        # out[B, L, D]
        return self.transformer(q, x, mask)


class DynamicLatentQueryAttentionBlock(nn.Module):
    def __init__(
        self: Self,
        *,
        sequence_length: int,
        latent_query_length: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(config=config)
        self.dense = DenseBlock(
            input_dim=sequence_length,
            hidden_dim=latent_query_length,
            output_dim=latent_query_length,
            dropout=config.dropout,
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x[B, N, D]
        """
        # q[B, L, D]
        q = self.dense(x.transpose(1, 2)).transpose(1, 2)
        # out[B, L, D]
        return self.transformer(q, x, mask)
