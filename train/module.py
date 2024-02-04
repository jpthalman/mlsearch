from typing import Dict

from typing_extensions import Self

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from data.data_module import Dim
from model.control_predictor import ControlPredictor
from model.scene_encoder import SceneEncoder
from model.world_model import WorldModel


class MLSearchModule(pl.LightningModule):
    # Model params
    NUM_ENCODER_LAYERS = 4
    EMBEDDING_DIM = 128
    HIDDEN_MULTIPLIER = 2
    LQ_RATIO = 0.25
    NUM_HEADS = 8
    DROPOUT = 0.1

    # Optimizer params
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    def __init__(self: Self) -> None:
        super().__init__()
        self.scene_encoder = SceneEncoder(
            num_layers=self.NUM_ENCODER_LAYERS,
            embed_dim=self.EMBEDDING_DIM,
            hidden_mult=self.HIDDEN_MULTIPLIER,
            latent_query_ratio=self.LQ_RATIO,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
        )
        self.control_predictor = ControlPredictor(self.EMBEDDING_DIM)
        self.world_model = WorldModel(self.EMBEDDING_DIM)

        self.embedding_loss = torch.nn.CosineEmbeddingLoss()
        self.control_loss = torch.nn.CrossEntropyLoss()

    def _step(
        self: Self,
        context: str,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        B = batch["ground_truth_control"].shape[0]
        D = B * Dim.A * (Dim.T - 1)

        scene_embedding = self.scene_encoder(
            agent_history=batch["agent_history"],
            agent_interactions=batch["agent_interactions"],
            agent_mask=batch["agent_mask"],
            roadgraph=batch["roadgraph"],
        )
        next_embedding = self.world_model(
            scene_embedding=scene_embedding,
            controls=batch["ground_truth_control"],
        )

        E = scene_embedding.shape[-1]
        embedding_loss = self.embedding_loss(
            scene_embedding[:, :, 1:, :].reshape(D, E),
            next_embedding[:, :, :-1, :].reshape(D, E),
            target=torch.ones(D).to(scene_embedding.device),
        )

        # TODO
        pred_control_dist = self.control_predictor(scene_embedding)
        control_loss = self.control_loss(
            pred_control_dist[:, :, :-1, :].reshape(D, Dim.Cd**2),
            batch["ground_truth_control_dist"].view(D, Dim.Cd**2),
        )

        scene_embedding = scene_embedding.mean()
        self.log(
            "train/loss_embedding",
            scene_embedding,
            prog_bar=True,
            batch_size=B,
        )

        control_loss = control_loss.mean()
        self.log(
            "train/loss_control",
            control_loss,
            prog_bar=True,
            batch_size=B,
        )

        loss = embedding_loss + control_loss
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=B,
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
