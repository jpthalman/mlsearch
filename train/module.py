import math
from typing import Dict, Tuple

from typing_extensions import Self

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from data import controls as control_utils
from data.config import Dim
from model.control_predictor import ControlPredictor
from model.scene_encoder import SceneEncoder
from model.transformer_block import TransformerConfig
from model.world_model import WorldModel


Dt = 5


def compute_control_error(ego_history, controls):
    """
    ego_history[B, T, S]
    controls[B, T, C]
    """
    B = ego_history.shape[0]

    states = ego_history[:, :-Dt, :].reshape(B * (Dim.T - Dt), Dim.S)
    controls = controls[:, :-Dt, :].reshape(B * (Dim.T - Dt), Dim.C)
    next_states = control_utils.integrate(states, controls)

    pos_gt = ego_history[:, Dt:, :2].reshape(B * (Dim.T - Dt), 2)
    pos_pred = next_states[:, :2]
    rel_gt = control_utils.relative_positions(states[:, :4], pos_gt)
    rel_pred = control_utils.relative_positions(states[:, :4], pos_pred)
    err = rel_pred - rel_gt
    std = rel_gt.abs().detach().std(dim=0)
    return err.abs() / (std + 1e-2)


class MLSearchModule(pl.LightningModule):
    # Model params
    AGENT_ENCODER_BLOCKS = 2
    ROADGRAPH_ENCODER_BLOCKS = 4
    ROADGRAPH_DOWNSAMPLE_FRAC = 0.25
    WORLD_MODEL_LAYERS = 2
    EMBEDDING_DIM = 128
    HIDDEN_MULTIPLIER = 2**0.5
    NUM_HEADS = 2
    DROPOUT = 0.01

    # Optimizer params
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Loss params
    EMBEDDING_LOSS_SCALE = 0.05
    CONTROL_WARMUP_FRAC = 0.5

    def __init__(self: Self) -> None:
        super().__init__()
        config = TransformerConfig(
            embed_dim=self.EMBEDDING_DIM,
            hidden_dim=math.ceil(self.EMBEDDING_DIM * self.HIDDEN_MULTIPLIER),
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
        )

        self.scene_encoder = SceneEncoder(
            num_agent_blocks=self.AGENT_ENCODER_BLOCKS,
            num_roadgraph_blocks=self.ROADGRAPH_ENCODER_BLOCKS,
            roadgraph_downsample_frac=self.ROADGRAPH_DOWNSAMPLE_FRAC,
            config=config,
        )
        self.control_predictor = ControlPredictor(config=config)
        self.world_model = WorldModel(
            num_layers=self.WORLD_MODEL_LAYERS,
            config=config,
        )

        self.example_input_array = dict(
            agent_history=torch.zeros([1, Dim.A, Dim.T, Dim.S]),
            roadgraph=torch.zeros([1, Dim.R, Dim.Rd]),
        )

    def forward(
        self: Self,
        agent_history: torch.Tensor,
        roadgraph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.scene_encoder(
            agent_history=agent_history,
            roadgraph=roadgraph,
        )
        controls = self.control_predictor(embedding)
        next_ego_state = control_utils.integrate(
            agent_history[:, 0, :, :].view(-1, Dim.S),
            controls.view(-1, Dim.C),
        )
        next_ego_state = next_ego_state.view(-1, Dim.T, Dim.S)
        embedding = self.world_model(
            scene_embedding=embedding,
            next_ego_state=next_ego_state,
        )
        return controls, embedding

    def _step(
        self: Self,
        context: str,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        B = batch["agent_history"].shape[0]

        embed_true = self.scene_encoder(
            agent_history=batch["agent_history"],
            roadgraph=batch["roadgraph"],
        )
        embed_pred = self.world_model(
            scene_embedding=embed_true,
            next_ego_state=batch["agent_history"][:, 0, :, :],
        )
        controls = self.control_predictor(embed_true)

        embed_pred = embed_pred[:, :, :-Dt, :]
        embed_present = embed_true[:, :, :-Dt, :]
        embed_future = embed_true[:, :, Dt:, :]

        scale = 0.1 * embed_true.detach().norm(dim=-1).mean() + 1e-3
        embed_pred_error = (embed_pred - embed_future) / scale
        embed_pred_error = embed_pred_error.norm(dim=-1)
        embed_similarity = (embed_present - embed_future) / scale
        embed_similarity = (-embed_similarity.norm(dim=-1)).exp()
        embed_loss = embed_pred_error.mean() + embed_similarity.mean()
        embed_loss *= self.EMBEDDING_LOSS_SCALE
        if self.global_step < self._control_warmup_steps():
            embed_loss = embed_loss.detach()

        control_error = compute_control_error(
            batch["agent_history"][:, 0, :, :],
            controls,
        )
        control_loss = control_error.abs().mean()

        loss = control_loss + embed_loss

        training = (context == "train")
        log_kwargs = dict(
            prog_bar=training,
            on_step=training,
            on_epoch=not training,
            batch_size=B,
        )

        self.log(
            f"{context}/loss",
            loss,
            **log_kwargs,
        )
        self.log(
            f"{context}/control_loss",
            control_loss,
            **log_kwargs,
        )
        self.log(
            f"{context}/embed_sim",
            embed_similarity.mean(),
            **log_kwargs,
        )
        self.log(
            f"{context}/embed_pred",
            embed_pred_error.mean(),
            **log_kwargs,
        )
        self.log(
            f"{context}/metric/long_err",
            control_error[:, 0].abs().mean(),
            **log_kwargs,
        )
        self.log(
            f"{context}/metric/lat_err",
            control_error[:, 1].abs().mean(),
            **log_kwargs,
        )
        return loss

    def training_step(
        self: Self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step("train", batch, batch_idx)

    def validation_step(
        self: Self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step("val", batch, batch_idx)

    def configure_optimizers(self: Self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            fused=True,
        )
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.trainer.max_epochs,
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval="epoch",
                frequency=1,
            ),
        )

    def _total_steps(self: Self) -> int:
        return self.trainer.estimated_stepping_batches

    def _control_warmup_steps(self: Self) -> int:
        return math.ceil(self.CONTROL_WARMUP_FRAC * self._total_steps())
