import enum
import os
from pathlib import Path
from typing import Dict
from typing_extensions import Self

import pytorch_lightning as pl
import torch
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet
)
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import DataLoader, Dataset


class Dim(enum.IntEnum):
    # Max agents
    A = 32
    # Time dimension size
    T = 11
    # Agent state size
    S = 42
    # Max agent interactions
    Ai = 8
    # Number of roadgraph features per agent
    R = 32
    # Dimension of roadgraph features
    Rd = 32
    # Dimension of the controls that can be applied to the vehicle
    C = 2
    # Discretization of each control dimension
    Cd = 16


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
        for path in root.iterdir():
            if not path.is_dir():
                continue
            self._paths.append(dict(
                scenario_name=path.name,
                scenario_path=path / f"scenario_{path.name}.parquet",
                map_path=path / f"log_map_archive_{path.name}.json",
            ))

    def __getitem__(self: Self, idx: int) -> Dict[str, torch.Tensor]:
        info = self._paths[idx]
        scenario = load_argoverse_scenario_parquet(info["scenario_path"])
        static_map = ArgoverseStaticMap.from_json(info["map_path"])
        # TODO: Populate these tensors
        return dict(
            scenario_name=info["scenario_name"],
            agent_history=torch.zeros([Dim.A, Dim.T, 1, Dim.S]),
            agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.Ai]),
            agent_mask=torch.zeros([Dim.A, Dim.T]),
            roadgraph=torch.zeros([Dim.A, 1, Dim.R, Dim.Rd]),
            ground_truth_control=torch.zeros([Dim.A, Dim.T - 1, Dim.C]),
            ground_truth_control_dist=torch.zeros([Dim.A, Dim.T - 1, Dim.Cd**2]),
        )

    def __len__(self: Self) -> int:
        return len(self._paths)
