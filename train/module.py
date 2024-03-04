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
    controls = controls.reshape(B * (Dim.T - Dt), Dim.C)
    pred_states = control_utils.integrate(states, controls)
    gt_states = ego_history[:, Dt:, :].reshape(B * (Dim.T - Dt), Dim.S)

    rel_pred = control_utils.relative_positions(states, pred_states)
    rel_gt = control_utils.relative_positions(states, gt_states)
    return rel_pred - rel_gt


class MLSearchModule(pl.LightningModule):
    # Model params
    ENCODER_BLOCKS = 2
    WORLD_MODEL_LAYERS = 2
    EMBEDDING_DIM = 128
    HIDDEN_MULTIPLIER = 2**0.5
    NUM_HEADS = 2
    DROPOUT = 0.01

    # Optimizer params
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    def __init__(self: Self) -> None:
        super().__init__()
        config = TransformerConfig(
            embed_dim=self.EMBEDDING_DIM,
            hidden_dim=math.ceil(self.EMBEDDING_DIM * self.HIDDEN_MULTIPLIER),
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
        )

        self.scene_encoder = SceneEncoder(
            num_blocks=self.ENCODER_BLOCKS,
            config=config,
        )
        self.control_predictor = ControlPredictor(config=config)
        self.world_model = WorldModel(
            num_layers=self.WORLD_MODEL_LAYERS,
            config=config,
        )
        self.embedding_loss = torch.nn.CosineEmbeddingLoss()

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
        embed_input = embed_true[:, :, :-Dt, :]
        embed_pred = self.world_model(
            scene_embedding=embed_input,
            next_ego_state=batch["agent_history"][:, 0, Dt:, :],
        )
        controls = self.control_predictor(embed_input)

        D = B * Dim.A * (Dim.T - Dt)
        embed_error = self.embedding_loss(
            embed_true[:, :, Dt:, :].reshape(D, self.EMBEDDING_DIM).detach(),
            embed_pred.reshape(D, self.EMBEDDING_DIM),
            target=torch.ones(D).to(self.device),
        )
        control_error = compute_control_error(
            batch["agent_history"][:, 0, :, :],
            controls,
        )

        control_loss = control_error.square().sum(dim=1).abs().sqrt().mean()
        embed_loss = embed_error.mean()
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
            f"{context}/embedding_loss",
            embed_loss,
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
