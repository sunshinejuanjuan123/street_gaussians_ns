"""Street Gaussians pipeline with optional eval image export."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, Tuple

import torch
import torchvision.utils as vutils

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class SgnPipelineConfig(VanillaPipelineConfig):
    """Pipeline config with optional eval PNG export."""

    _target: Type = field(default_factory=lambda: SgnPipeline)


class SgnPipeline(VanillaPipeline):
    """Vanilla pipeline that can save eval renders as PNG files on disk."""

    config: SgnPipelineConfig
    _eval_image_output_root: Optional[Path] = None

    def _should_save_eval_images(self) -> bool:
        return os.environ.get("SAVE_EVAL_IMAGES", "").strip() == "1"

    def _save_eval_images(self, images_dict: Dict[str, torch.Tensor], step: int, suffix: str = "") -> None:
        if not self._should_save_eval_images():
            return
        root = self._eval_image_output_root
        if root is None:
            return
        out_dir = root / f"step_{step:06d}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image in images_dict.items():
            if image.ndim == 3:
                vutils.save_image(image.permute(2, 0, 1).cpu(), out_dir / f"{image_name}.png")
            else:
                vutils.save_image(image.cpu(), out_dir / f"{image_name}.png")

    def get_eval_image_metrics_and_images(self, step: int):
        metrics_dict, images_dict = super().get_eval_image_metrics_and_images(step=step)
        self._save_eval_images(images_dict, step)
        return metrics_dict, images_dict

    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        if output_path is None and self._should_save_eval_images() and self._eval_image_output_root is not None:
            output_path = self._eval_image_output_root / f"step_{step:06d}_all"
        return super().get_average_eval_image_metrics(
            step=step, output_path=output_path, get_std=get_std
        )
