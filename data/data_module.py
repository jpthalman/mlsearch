import os
from pathlib import Path
from typing import Dict
from typing_extensions import Self

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.config import Dim, POS_SCALE, VEL_SCALE


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
            self._paths.append(path)

    """
    Returns a tensors dictionary of the following form:
    dict(
            scenario_name= str,
            agent_history= [Dim.A, Dim.T, 1, Dim.S],
            agent_history_mask= [Dim.A, Dim.T],
            agent_interactions= [Dim.A, Dim.T, Dim.Ai, Dim.S],
            agent_interactions_mask= [Dim.A, Dim.T, Dim.Ai],
            roadgraph= [Dim.R, Dim.Rd],
            roadgraph_mask= [Dim.R],
            ground_truth_controls= [Dim.T, Dim.C],
        )
    """
    def __getitem__(self: Self, idx: int) -> Dict[str, torch.Tensor]:
        path = self._paths[idx]

        history = torch.load(path / "agent_history.pt")
        history = history[:64, ::2, :]
        history[:, :, 0] /= POS_SCALE
        history[:, :, 1] /= POS_SCALE
        history[:, :, 4] /= VEL_SCALE

        roadgraph = torch.load(path / "roadgraph.pt")
        roadgraph[:, :4] /= POS_SCALE

        return dict(
            scenario_name=path.stem,
            agent_history=history,
            roadgraph=roadgraph,
        )

    def __len__(self: Self) -> int:
        return len(self._paths)


def main():
    test_tensors_path = Path("/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/data.pt")
    tensors_dict = torch.load(test_tensors_path)
    for k, v in tensors_dict.items():
        print(k)
        if type(v) == str:
            print(v)
        else:
            print(v.shape)

if __name__ == "__main__":
    main()