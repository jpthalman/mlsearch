import math
from typing import Dict, Tuple

from typing_extensions import Self

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from data import controls as control_utils
from data import config
from data.config import Dim
from model.control_predictor import ControlPredictor
from model.scene_encoder import SceneEncoder
from model.transformer_block import TransformerConfig
from model.world_model import WorldModel


def compute_control_error(ego_history, controls):
    """
    ego_history[B, T, S]
    controls[B, T, C]
    """
    B = ego_history.shape[0]
    ego_history = ego_history.clone()
    controls = controls.clone()

    ego_history[:, :, (0, 1)] *= config.POS_SCALE
    ego_history[:, :, 4] *= config.VEL_SCALE

    states = ego_history[:, :-1, :].reshape(B * (Dim.T - 1), Dim.S)
    x0 = states[:, 0]
    y0 = states[:, 1]
    cyaw = states[:, 2]
    syaw = states[:, 3]
    v0 = states[:, 4]

    controls = controls[:, :-1, :].reshape(B * (Dim.T - 1), Dim.C)
    accel = (2 * controls[:, 0] - 1) * control_utils.MAX_ACCEL
    curv = (2 * controls[:, 1] - 1) * control_utils.MAX_CURV

    v1 = v0 + accel
    dl = 0.5 * (v0 + v1)
    x1, y1 = control_utils._advance(x0, y0, cyaw, syaw, curv, dl)

    x1_gt = ego_history[:, 1:, 0].reshape(B * (Dim.T - 1))
    y1_gt = ego_history[:, 1:, 1].reshape(B * (Dim.T - 1))
    dx = x1 - x1_gt
    dy = y1 - y1_gt
    dist_err = torch.sqrt(dx**2 + dy**2 + 1e-5)
    return dist_err


class MLSearchModule(pl.LightningModule):
    # Model params
    ENCODER_BLOCKS = 2
    ENCODER_SELF_ATTN_LAYERS = 2
    WORLD_MODEL_LAYERS = 2
    EMBEDDING_DIM = 128
    HIDDEN_MULTIPLIER = 2**0.5
    NUM_HEADS = 2
    DROPOUT = 0.1

    # Optimizer params
    LEARNING_RATE = 1e-3
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
            num_self_attention_layers=self.ENCODER_SELF_ATTN_LAYERS,
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
            agent_history_mask=torch.zeros([1, Dim.A, Dim.T]),
            agent_interactions=torch.zeros([1, Dim.A, Dim.T, Dim.Ai, Dim.S]),
            agent_interactions_mask=torch.zeros([1, Dim.A, Dim.T, Dim.Ai]),
            roadgraph=torch.zeros([1, Dim.R, Dim.Rd]),
            roadgraph_mask=torch.zeros([1, Dim.R]),
            controls=torch.zeros([1, Dim.T, Dim.C]),
        )

    def forward(
        self: Self,
        agent_history: torch.Tensor,
        agent_history_mask: torch.Tensor,
        agent_interactions: torch.Tensor,
        agent_interactions_mask: torch.Tensor,
        roadgraph: torch.Tensor,
        roadgraph_mask: torch.Tensor,
        controls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scene_embedding = self.scene_encoder(
            agent_history=agent_history,
            agent_history_mask=agent_history_mask,
            agent_interactions=agent_interactions,
            agent_interactions_mask=agent_interactions_mask,
            roadgraph=roadgraph,
            roadgraph_mask=roadgraph_mask,
        )
        next_embedding = self.world_model(
            scene_embedding=scene_embedding,
            controls=controls,
        )
        pred_control_dist = self.control_predictor(scene_embedding)
        return dict(
            scene_embedding=scene_embedding,
            next_embedding=next_embedding,
            pred_control_dist=pred_control_dist,
        )

    def _step(
        self: Self,
        context: str,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        B = batch["ground_truth_controls"].shape[0]
        D = B * Dim.A * (Dim.T - 1)

        out = self.forward(
            agent_history=batch["agent_history"],
            agent_history_mask=batch["agent_history_mask"],
            agent_interactions=batch["agent_interactions"],
            agent_interactions_mask=batch["agent_interactions_mask"],
            roadgraph=batch["roadgraph"],
            roadgraph_mask=batch["roadgraph_mask"],
            controls=batch["ground_truth_controls"],
        )

        embedding_loss = self.embedding_loss(
            out["scene_embedding"][:, :, 1:, :].reshape(D, self.EMBEDDING_DIM),
            out["next_embedding"][:, :, :-1, :].reshape(D, self.EMBEDDING_DIM),
            target=torch.ones(D).to(self.device),
        )

        control_error = compute_control_error(
            batch["agent_history"][:, 0, :, :],
            out["pred_control_dist"],
        )
        # control_error = (out["pred_control_dist"] - batch["ground_truth_controls"])**2

        embedding_loss = embedding_loss.mean()
        control_loss = control_error.mean()
        loss = embedding_loss + control_loss

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
            f"{context}/loss_embd",
            embedding_loss,
            **log_kwargs,
        )
        self.log(
            f"{context}/loss_ctrl",
            control_loss,
            **log_kwargs,
        )

        return control_loss

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
