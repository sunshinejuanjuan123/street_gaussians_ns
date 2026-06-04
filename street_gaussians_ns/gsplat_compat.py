"""Compatibility wrappers for gsplat 0.1.x project/rasterize APIs on gsplat >= 1.0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

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


def default_ftheta_coeffs() -> Any:
    """Default F-Theta distortion parameters (NVIDIA example coefficients)."""
    from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

    return FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=(0.0, 8.43e-3, 2.32e-6, -5.05e-8, 6.14e-10, -1.74e-12),
        angle_to_pixeldist_poly=(0.0, 118.43, -2.56, 6.32, -10.42, 3.67),
        max_angle=1000.0,
        linear_cde=(9.997e-1, 1.87e-5, 1.77e-5),
    )


def load_ftheta_coeffs_from_json(path: Union[str, Path]) -> Any:
    """Load FThetaCameraDistortionParameters from a JSON file."""
    from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ref = data.get("reference_poly", "ANGLE_TO_PIXELDIST")
    if isinstance(ref, str):
        ref = FThetaPolynomialType[ref]
    elif isinstance(ref, int):
        ref = FThetaPolynomialType(ref)

    def _tuple6(key: str) -> Tuple[float, float, float, float, float, float]:
        vals = data[key]
        vals = list(vals) + [0.0] * (6 - len(vals))
        return tuple(float(v) for v in vals[:6])  # type: ignore[return-value]

    return FThetaCameraDistortionParameters(
        reference_poly=ref,
        pixeldist_to_angle_poly=_tuple6("pixeldist_to_angle_poly"),
        angle_to_pixeldist_poly=_tuple6("angle_to_pixeldist_poly"),
        max_angle=float(data.get("max_angle", 1000.0)),
        linear_cde=tuple(float(v) for v in data["linear_cde"]),  # type: ignore[assignment]
    )


def _radial_coeffs_from_distortion(
    distortion_params: Optional[Tensor], device: torch.device, dtype: torch.dtype
) -> Tensor:
    """OpenCV fisheye k1-k4 from nerfstudio distortion_params [k1,k2,k3,k4,p1,p2,...]."""
    if distortion_params is None:
        coeffs = torch.zeros(4, device=device, dtype=dtype)
    else:
        flat = distortion_params.reshape(-1)
        coeffs = flat[:4].to(device=device, dtype=dtype)
    return coeffs.unsqueeze(0)  # [1, 4]


def _camera_space_depth(means: Tensor, viewmat: Tensor, near_plane: float) -> Tensor:
    """Per-Gaussian depth (camera z) for 3DGUT depth pass."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam = means @ R.T + t
    return cam[:, 2].clamp(min=near_plane)


def rasterize_gaussians_3dgut(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    camera_model: Literal["fisheye", "ftheta"] = "fisheye",
    distortion_params: Optional[Tensor] = None,
    ftheta_coeffs: Optional[Any] = None,
    background: Optional[Tensor] = None,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    render_mode: Literal["RGB", "D", "RGB+D"] = "RGB",
    tile_size: int = 16,
    near_plane: float = 0.01,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """3DGUT rasterization via gsplat ``rasterization`` (UT + eval3d, packed=False).

    ``with_eval3d`` requires exactly 3 color channels; depth uses a separate RGB pass.

    Returns:
        rgb: [H, W, 3]
        alpha: [H, W, 1]
        depth: [H, W, 1] if render_mode includes depth, else None
    """
    from gsplat.rendering import rasterization

    device = means.device
    dtype = means.dtype
    viewmats = _viewmat_to_4x4(viewmat).unsqueeze(0)
    ks = _intrinsics_from_fx_fy_cx_cy(fx, fy, cx, cy, device, dtype).unsqueeze(0)

    if opacities.dim() == 2 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)

    radial_coeffs = None
    ftheta = None
    if camera_model == "fisheye":
        radial_coeffs = _radial_coeffs_from_distortion(distortion_params, device, dtype)
    elif camera_model == "ftheta":
        ftheta = ftheta_coeffs if ftheta_coeffs is not None else default_ftheta_coeffs()

    backgrounds = None
    if background is not None:
        backgrounds = background.view(1, -1)

    want_depth = render_mode in ("D", "RGB+D", "RGB+ED")

    def _rasterize(color_buf: Tensor, bg: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        bgs = bg.view(1, -1) if bg is not None else None
        render_colors, render_alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=color_buf,
            viewmats=viewmats,
            Ks=ks,
            width=img_width,
            height=img_height,
            near_plane=near_plane,
            packed=False,
            tile_size=tile_size,
            backgrounds=bgs,
            render_mode="RGB",
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            ftheta_coeffs=ftheta,
            with_ut=True,
            with_eval3d=True,
        )
        out = render_colors[0]
        a = render_alphas[0]
        if a.ndim == 2:
            a = a.unsqueeze(-1)
        return out, a

    rgb, alpha = _rasterize(colors, background)

    depth = None
    if want_depth:
        depth_vals = _camera_space_depth(means, viewmat, near_plane)
        depth_colors = depth_vals.unsqueeze(-1).expand(-1, 3)
        depth_rgb, _ = _rasterize(depth_colors, torch.zeros(3, device=device, dtype=dtype))
        depth = depth_rgb[..., :1]

    if render_mode == "D":
        return depth if depth is not None else rgb[..., :1], alpha, depth
    return rgb, alpha, depth
