"""Project LiDAR point clouds to camera views and save sparse depth maps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

_sfm_tools_root = Path(__file__).resolve().parents[3] / "sfm_tools" / "sfm_tools"
if _sfm_tools_root.exists():
    sys.path.insert(0, str(_sfm_tools_root))

from feature_extract_match.model.read_write_model import read_model


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def build_pose_lookup(uniscene: dict) -> dict[int, np.ndarray]:
    pose_info: dict[int, np.ndarray] = {}
    for ego_info in uniscene["ego_status"]:
        timestamp = int(round(ego_info["timestamp"], 3) * 1000)
        quat = ego_info["ego_orientation"]
        trsl = ego_info["ego_position"]
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        pose[:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])
        pose_info[timestamp] = pose
    return pose_info


def load_lidar_points(lidar_abs_path: str, max_points: int = 50000) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(lidar_abs_path)
    points = np.asarray(pcd.points)
    nan_rows = np.isnan(points).any(axis=1)
    points = points[~nan_rows]
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
    return points


def project_pinhole(points_world: np.ndarray, w2c: np.ndarray, K: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    homogeneous = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    cam = (w2c @ homogeneous.T).T[:, :3]
    z = cam[:, 2]
    valid_z = z > 0.1
    cam = cam[valid_z]
    z = z[valid_z]
    uv_h = (K @ cam.T).T
    u = np.round(uv_h[:, 0] / uv_h[:, 2]).astype(np.int32)
    v = np.round(uv_h[:, 1] / uv_h[:, 2]).astype(np.int32)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[in_img], v[in_img], z[in_img]
    for ui, vi, zi in zip(u, v, z):
        if zi < depth_map[vi, ui]:
            depth_map[vi, ui] = zi
    valid = depth_map < np.inf
    depth_map[~valid] = 0.0
    return depth_map, valid


def project_fisheye(points_world: np.ndarray, w2c: np.ndarray, K: np.ndarray, dist: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    homogeneous = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    cam = (w2c @ homogeneous.T).T[:, :3].astype(np.float64)
    z = cam[:, 2]
    valid_z = z > 0.1
    cam = cam[valid_z]
    z = z[valid_z]
    if cam.shape[0] == 0:
        valid = depth_map < np.inf
        depth_map[~valid] = 0.0
        return depth_map, valid
    rvec, _ = cv2.Rodrigues(np.eye(3))
    tvec = np.zeros(3)
    dist_coeffs = dist[:4].reshape(1, 4)
    uv, _ = cv2.fisheye.projectPoints(cam.reshape(-1, 1, 3), rvec, tvec, K[:3, :3], dist_coeffs)
    uv = uv.reshape(-1, 2)
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[in_img], v[in_img], z[in_img]
    for ui, vi, zi in zip(u, v, z):
        if zi < depth_map[vi, ui]:
            depth_map[vi, ui] = zi
    valid = depth_map < np.inf
    depth_map[~valid] = 0.0
    return depth_map, valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LiDAR depth maps for each camera image.")
    parser.add_argument("--data_root", required=True, help="PVB dataset root (contains plannerGt/)")
    parser.add_argument("--gs_data_root", required=True, help="3dgs_format root")
    parser.add_argument("--colmap_subdir", default="colmap/sparse/0", help="Colmap sparse model path relative to gs_data_root")
    parser.add_argument("--output_subdir", default="depths", help="Output depth directory relative to gs_data_root")
    parser.add_argument("--max_points", type=int, default=50000, help="Max LiDAR points per frame")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    gs_data_root = Path(args.gs_data_root)
    sparse_dir = gs_data_root / args.colmap_subdir
    output_root = gs_data_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    uniscene = json.load(open(data_root / "plannerGt/unisceneproto.json", "r"))
    pose_lookup = build_pose_lookup(uniscene)
    cameras, images, _ = read_model(sparse_dir, ext=".bin")

    generated = 0
    skipped = 0
    for sensor_info in tqdm(uniscene["sensor_frames"], desc="lidar depth"):
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        if timestamp not in pose_lookup:
            skipped += 1
            continue
        if not sensor_info.get("lidar_data"):
            skipped += 1
            continue

        lidar2enu = pose_lookup[timestamp]
        lidar_abs_path = data_root / sensor_info["lidar_data"][0]["file_path"]
        if not lidar_abs_path.exists():
            skipped += 1
            continue

        points = load_lidar_points(str(lidar_abs_path), max_points=args.max_points)
        if points.shape[0] == 0:
            skipped += 1
            continue

        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        points_enu = (lidar2enu @ homogeneous.T).T[:, :3]

        for image_id, image in images.items():
            cam_name, image_name = image.name.split("/")
            image_timestamp, _ = os.path.splitext(image_name)
            if image_timestamp != str(timestamp):
                continue

            cam = cameras[image.camera_id]
            h, w = cam.height, cam.width
            K = np.eye(3)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cam.params[0], cam.params[1], cam.params[2], cam.params[3]

            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(image.qvec)
            w2c[:3, 3] = image.tvec

            if cam.model in ("OPENCV", "PINHOLE", "SIMPLE_PINHOLE"):
                depth_map, valid = project_pinhole(points_enu, w2c, K, h, w)
            elif cam.model in ("OPENCV_FISHEYE", "FISHEYE"):
                dist = np.array(cam.params[4:8] if len(cam.params) >= 8 else [0, 0, 0, 0])
                depth_map, valid = project_fisheye(points_enu, w2c, K, dist, h, w)
            else:
                depth_map, valid = project_pinhole(points_enu, w2c, K, h, w)

            out_path = output_root / image.name
            out_path = out_path.with_suffix(".npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, depth=depth_map, valid=valid)
            generated += 1

    print(f"Generated {generated} depth maps to {output_root} (skipped {skipped} lidar frames)")


if __name__ == "__main__":
    main()
