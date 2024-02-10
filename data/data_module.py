import os
from pathlib import Path
from typing import Dict
from typing_extensions import Self

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.dimensions import Dim


class AV2DataModule(pl.LightningDataModule):
    def __init__(self: Self, *, batch_size: int) -> None:
        super().__init__()
        self._batch_size = batch_size

    def train_dataloader(self: Self) -> DataLoader:
        return DataLoader(
            AV2Dataset("train"),
            batch_size=self._batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

    def val_dataloader(self: Self) -> DataLoader:
        return DataLoader(
            AV2Dataset("val"),
            batch_size=self._batch_size,
            num_workers=os.cpu_count(),
        )


class AV2Dataset(Dataset[Dict[str, torch.Tensor]]):
    ROOT = Path("/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2")

    def __init__(self: Self, name: str) -> None:
        self._paths = []
        root = self.ROOT / name
        print(f"Collecting {name} scenario info...")
        for path in root.iterdir():
            self._paths.append(dict(
                scenario_name=path.name,
                scenario_path=path / f"scenario_{path.name}.parquet",
                map_path=path / f"log_map_archive_{path.name}.json",
            ))

    def __getitem__(self: Self, idx: int) -> Dict[str, torch.Tensor]:
        info = self._paths[idx]
        # TODO: Populate these tensors
        controls = torch.rand([Dim.T, Dim.C])
        return dict(
            scenario_name=info["scenario_name"],
            agent_history=torch.zeros([Dim.A, Dim.T, 1, Dim.S]),
            agent_history_mask=torch.zeros([Dim.A, Dim.T]),
            agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S]),
            agent_interactions_mask=torch.zeros([Dim.A, Dim.T, Dim.Ai]),
            roadgraph=torch.zeros([Dim.R, Dim.Rd]),
            roadgraph_mask=torch.zeros([Dim.R]),
            ground_truth_controls=controls,
        )

    def __len__(self: Self) -> int:
        return len(self._paths)
