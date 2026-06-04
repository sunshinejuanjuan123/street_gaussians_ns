"""Compatibility wrappers for gsplat 0.1.x project/rasterize APIs on gsplat >= 1.0."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)


def _viewmat_to_4x4(viewmat: Tensor) -> Tensor:
    if viewmat.shape == (4, 4):
        return viewmat
    if viewmat.shape == (3, 4):
        viewmat4 = torch.eye(4, device=viewmat.device, dtype=viewmat.dtype)
        viewmat4[:3, :] = viewmat
        return viewmat4
    raise ValueError(f"Unsupported viewmat shape: {viewmat.shape}")


def _intrinsics_from_fx_fy_cx_cy(
    fx: float, fy: float, cx: float, cy: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def project_gaussians(
    means3d: Tensor,
    scales: Tensor,
    glob_scale: float,
    quats: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, None, Tensor, None]:
    """Legacy gsplat projection API backed by ``fully_fused_projection``."""
    device = means3d.device
    dtype = means3d.dtype
    viewmats = _viewmat_to_4x4(viewmat).unsqueeze(0).unsqueeze(0)
    ks = _intrinsics_from_fx_fy_cx_cy(fx, fy, cx, cy, device, dtype).unsqueeze(0).unsqueeze(0)
    scales = scales * glob_scale

    radii, means2d, depths, conics, _ = fully_fused_projection(
        means3d.unsqueeze(0),
        None,
        quats.unsqueeze(0),
        scales.unsqueeze(0),
        viewmats,
        ks,
        img_width,
        img_height,
        near_plane=clip_thresh,
        packed=False,
    )

    xys = means2d[0, 0]
    depths = depths[0, 0]
    conics = conics[0, 0]
    radii_2d = radii[0, 0]
    # Legacy splatfacto expects per-Gaussian scalar radii shape [N], not [N, 2].
    radii_legacy = radii_2d.max(dim=-1).values

    tile_width = math.ceil(img_width / float(block_width))
    tile_height = math.ceil(img_height / float(block_width))
    num_tiles_hit, _, _ = isect_tiles(
        xys.unsqueeze(0),
        radii_2d.unsqueeze(0),
        depths.unsqueeze(0),
        block_width,
        tile_width,
        tile_height,
        packed=False,
    )
    num_tiles_hit = num_tiles_hit[0]

    return xys, depths, radii_legacy, conics, None, num_tiles_hit, None


def rasterize_gaussians(
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    conics: Tensor,
    num_tiles_hit: Tensor,
    colors: Tensor,
    opacity: Tensor,
    img_height: int,
    img_width: int,
    block_width: int,
    background: Optional[Tensor] = None,
    return_alpha: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Legacy gsplat rasterization API backed by ``rasterize_to_pixels``."""
    del num_tiles_hit  # recomputed via isect_tiles for gsplat >= 1.0

    if depths.dim() == 2 and depths.shape[-1] == 1:
        depths = depths.squeeze(-1)
    if radii.dim() == 1:
        radii_2d = radii.unsqueeze(-1).expand(-1, 2)
    elif radii.dim() == 2 and radii.shape[-1] == 1:
        radii_2d = radii.expand(-1, 2)
    else:
        radii_2d = radii
    if opacity.dim() == 2 and opacity.shape[-1] == 1:
        opacities = opacity.squeeze(-1)
    else:
        opacities = opacity

    tile_width = math.ceil(img_width / float(block_width))
    tile_height = math.ceil(img_height / float(block_width))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        xys.unsqueeze(0),
        radii_2d.unsqueeze(0),
        depths.unsqueeze(0),
        block_width,
        tile_width,
        tile_height,
        packed=False,
    )
    del tiles_per_gauss
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    backgrounds = None
    if background is not None:
        backgrounds = background.view(1, -1)

    render_colors, render_alphas = rasterize_to_pixels(
        xys.unsqueeze(0),
        conics.unsqueeze(0),
        colors.unsqueeze(0),
        opacities.unsqueeze(0),
        img_width,
        img_height,
        block_width,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=False,
    )

    rgb = render_colors[0]
    alpha = render_alphas[0]
    if return_alpha:
        return rgb, alpha
    return rgb
