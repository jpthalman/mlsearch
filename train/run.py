"""
Run with `python -m train.run` from the project root
"""

import argparse
import getpass
from pathlib import Path

import comet_ml
import torch
import pytorch_lightning as pl

from data.config import EXPERIMENT_ROOT
from data.data_module import AV2DataModule
from train.module import MLSearchModule


API_KEY_WARNING = """
=== COMET API KEY NOT FOUND ===
Create an account on https://www.comet.com/, go to Account Setting -> API Keys -> Copy
and place it in the API_KEYS dict in train/run.py. The username should be your username
on the computer you'll be training on.
===============================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--for-real", action="store_true")

    args = parser.parse_args()
    if args.for_real and args.name == "debug":
        raise RuntimeError(
            "Give your experiement a name when running with the full dataset"
        )
    return args


def configure_logger(args: argparse.Namespace) -> pl.loggers.CometLogger | None:
    API_KEYS = dict(
        jthalman="QpbkK2X7dvOnkqJHqvuRwBLKL",
        sqian="8Qd7fTGpsowvw3Cpv96EXq69b",
    )
    key = API_KEYS.get(getpass.getuser(), None)
    if key is None:
        raise RuntimeError(API_KEY_WARNING)
    if not args.for_real:
        return
    return pl.loggers.CometLogger(
        workspace="jpthalman",
        project_name="mlsearch",
        api_key=key,
        experiment_name=args.name,
    )


def main() -> None:
    args = parse_args()

    max_epochs = 10 if args.for_real else 1
    max_steps = -1 if args.for_real else 1000
    output_root = EXPERIMENT_ROOT / args.name
    output_root.mkdir(exist_ok=True, parents=True)

    torch.set_float32_matmul_precision("medium")

    model = MLSearchModule()
    print(pl.utilities.model_summary.ModelSummary(model, max_depth=-1))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        precision="16-mixed",
        logger=configure_logger(args),
        default_root_dir=output_root,
        enable_model_summary=False,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.ModelCheckpoint(
                dirpath=output_root,
                filename="last",
                enable_version_counter=False,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=output_root,
                monitor="val/loss",
                filename="best_val_loss",
                enable_version_counter=False,
            ),
        ],
    )
    trainer.fit(model, datamodule=AV2DataModule(batch_size=30))


if __name__ == "__main__":
    main()
