"""
Run with `python -m train.run` from the project root
"""

import argparse
import getpass
from pathlib import Path

import pytorch_lightning as pl

from data.data_module import AV2DataModule
from train.module import MLSearchModule


OUTPUT_ROOT = Path("/tmp/mlsearch")
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
    )
    key = API_KEYS.get(getpass.getuser(), None)
    if key is None:
        raise RuntimeError(API_KEY_WARNING)
    if not args.for_real:
        return
    return pl.loggers.CometLogger(
        workspace_name="jpthalman",
        project_name="mlsearch",
        api_key=key,
        experiment_name=args.name,
    )


def main() -> None:
    args = parse_args()

    max_epochs = 10 if args.for_real else 1
    max_steps = None if args.for_real else 1000
    output_root = OUTPUT_ROOT / args.name
    output_root.mkdir(exist_ok=True, parents=True)

    model = MLSearchModule()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        precision=32,
        logger=configure_logger(args),
        enable_checkpointing=args.for_real,
        default_root_dir=output_root,
    )
    trainer.fit(model, datamodule=AV2DataModule(batch_size=16))


if __name__ == "__main__":
    main()
