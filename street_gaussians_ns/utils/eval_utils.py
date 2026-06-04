"""Checkpoint loading helpers for PyTorch >=2.6 (weights_only default)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Union

import torch

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import eval_utils as _ns_eval_utils
from nerfstudio.utils.rich_utils import CONSOLE


def load_checkpoint_state(path: Path, map_location: Union[str, torch.device] = "cpu") -> dict:
    """Load a nerfstudio .ckpt file (pickle), compatible with PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Tuple[Path, int]:
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = load_checkpoint_state(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step


_ns_eval_utils.eval_load_checkpoint = eval_load_checkpoint

eval_setup = _ns_eval_utils.eval_setup
