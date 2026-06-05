# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union
import enum

import cv2
import numpy as np
import torch
from PIL import Image


class SemanticType(enum.IntEnum):
    DEFAULT = 0
    GROUND = 1
    SKY = 2

def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    image = np.array(pil_image, dtype="int64")
    if len(image.shape) == 3:
        image = image[:, :, 0]
    # TODO(zz): fix magic number.
    semantics = np.zeros_like(image)
    semantics[(image == 7) + (image == 8) + (image == 13) + (image == 14) + (image == 23) + (image == 24)] = SemanticType.GROUND.value
    semantics[image == 27] = SemanticType.SKY.value

    semantics = torch.from_numpy(semantics).unsqueeze(-1)
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float = 1,
    interpolation: int = cv2.INTER_NEAREST,
    depth_type=None
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1] (float32).
    """
    depth, _ = get_depth_and_valid_from_path(
        filepath=filepath,
        height=height,
        width=width,
        depth_scale_factor=scale_factor,
        interpolation=interpolation,
        depth_type=depth_type,
    )
    return depth


def get_depth_valid_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float = 1,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Load depth validity mask from npz (key: valid) or infer from depth > 0."""
    _, valid = get_depth_and_valid_from_path(
        filepath=filepath,
        height=height,
        width=width,
        depth_scale_factor=scale_factor,
        interpolation=interpolation,
    )
    return valid


def get_depth_and_valid_from_path(
    filepath: Path,
    height: int,
    width: int,
    depth_scale_factor: float = 1,
    interpolation: int = cv2.INTER_NEAREST,
    depth_type=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load depth and validity in one file read. Returns float32 depth and bool valid mask."""
    if filepath.suffix == ".npz":
        with np.load(filepath) as data:
            if "depth" in data:
                depth = data["depth"].astype(np.float32) * depth_scale_factor
            else:
                depth = data["arr_0"].astype(np.float32) * depth_scale_factor
            if "valid" in data:
                valid = data["valid"].astype(np.float32)
            else:
                valid = (depth > 0).astype(np.float32)
        depth = cv2.resize(depth, (width, height), interpolation=interpolation)
        valid = cv2.resize(valid, (width, height), interpolation=interpolation)
        return torch.from_numpy(depth).unsqueeze(-1), torch.from_numpy(valid > 0).unsqueeze(-1).bool()

    if filepath.suffix == ".npy":
        depth = np.load(filepath).astype(np.float32) * depth_scale_factor
        depth = cv2.resize(depth, (width, height), interpolation=interpolation)
    elif depth_type == "2x8bit" or filepath.suffix == ".png":
        depth_img = cv2.imread(str(filepath.absolute()))
        depth = depth_img[:, :, 0] + (depth_img[:, :, 1] * 256)
        depth = depth.astype(np.float32) * depth_scale_factor * 0.01
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        depth = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) * depth_scale_factor
        depth = cv2.resize(depth, (width, height), interpolation=interpolation)
    depth_tensor = torch.from_numpy(depth).unsqueeze(-1)
    return depth_tensor, depth_tensor > 0

