"""Smoke test for lazy depth loading in FullImageDatamanager."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataparser import ColmapDataParserConfig


def _assert_no_depth_in_cache(cache: list[dict], label: str) -> None:
    for idx, item in enumerate(cache):
        assert "depth" not in item, f"{label}[{idx}] should not cache depth"
        assert "depth_valid" not in item, f"{label}[{idx}] should not cache depth_valid"


def _assert_depth_batch(data: dict, image_shape: tuple[int, int]) -> None:
    assert "depth" in data, "batch missing depth"
    assert "depth_valid" in data, "batch missing depth_valid"
    assert data["depth"].dtype == torch.float32, f"depth dtype={data['depth'].dtype}, expected float32"
    assert data["depth_valid"].dtype == torch.bool, f"depth_valid dtype={data['depth_valid'].dtype}"
    assert tuple(data["depth"].shape[:2]) == image_shape, (
        f"depth shape {data['depth'].shape[:2]} != image shape {image_shape}"
    )
    assert data["depth_valid"].float().mean().item() > 0, "depth_valid is all false"


def main() -> None:
    data_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/mnt/iag/yuanweizhong/datasets/pvb_delivery_11v_86125_uniscene/"
        "2026_03_07_14_11_11_dlp_pilotGtParser/"
        "2026_03_07_14_11_11_dlp_pilotGtParser_unknown_1772863892910819000_1772863912910819000/"
        "3dgs_format"
    )
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    config = FullImageDatamanagerConfig(
        dataparser=ColmapDataParserConfig(
            data=data_root,
            colmap_path=Path("colmap/sparse/0"),
            depths_path=Path("depths"),
            segments_path=Path("segs"),
            filter_camera_id=[1],
            load_3D_points=False,
            undistort=True,
        ),
        cache_images="cpu",
        cache_images_type="uint8",
    )
    dm = FullImageDatamanager(config=config, device="cpu", test_mode="val")
    train_cache = dm.cached_train
    _assert_no_depth_in_cache(train_cache, "cached_train")

    camera, batch = dm.next_train(step=0)
    image_shape = tuple(batch["image"].shape[:2])
    _assert_depth_batch(batch, image_shape)

    eager = dm.train_dataset.get_data(0, image_type="uint8", load_depth=True)
    lazy_depth = batch["depth"].cpu()
    lazy_valid = batch["depth_valid"].cpu()
    depth_ctx = dm._train_depth_undistort_ctx[0] if dm._train_depth_undistort_ctx else None
    if depth_ctx is None and "depth" in eager:
        assert torch.allclose(lazy_depth, eager["depth"], atol=1e-4, rtol=1e-4), "lazy/eager depth mismatch"
        assert torch.equal(lazy_valid, eager["depth_valid"]), "lazy/eager depth_valid mismatch"
    else:
        print("Skip eager/lazy value comparison (undistorted depth path)")

    _, eval_batch = dm.next_eval(step=0)
    _assert_depth_batch(eval_batch, tuple(eval_batch["image"].shape[:2]))

    print(
        f"OK: lazy depth smoke test passed on {data_root} "
        f"(train={len(train_cache)} frames, depth_valid_ratio={batch['depth_valid'].float().mean():.4f})"
    )


if __name__ == "__main__":
    main()
