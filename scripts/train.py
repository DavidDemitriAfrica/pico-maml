#!/usr/bin/env python3
"""
A minimal script to train the Pico language model. In practice, you should just use the
`poetry run train` command to run the training pipeline. Doing so will call on this script.
"""

import click
from pathlib import Path
from src.training.trainer import Trainer
import torch
import sys

sys.setrecursionlimit(10000)


@click.command()
@click.option(
    "--config_path",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the training configuration file",
)
def main(config_path: Path) -> None:
    """Train the Pico language model using the specified configuration."""
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(config_path=str(config_path))
    trainer.train()


if __name__ == "__main__":
    main()
