"""
Street Gaussians configuration file.
"""
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from street_gaussians_ns.sgn_pipeline import SgnPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataparser import ColmapDataParserConfig
from street_gaussians_ns.data.utils.bbox_optimizers import BBoxOptimizerConfig
from street_gaussians_ns.sgn_splatfacto import SplatfactoModelConfig
from street_gaussians_ns.sgn_splatfacto_scene_graph import SplatfactoSceneGraphModelConfig

import os

env_value = os.getenv("STREET_GAUSSIANS_MAX_ITERATIONS", "").strip()

try:
    STREET_GAUSSIANS_MAX_ITERATIONS = int(env_value) if env_value else 70000
except ValueError:
    print(f"Warning: Invalid STREET_GAUSSIANS_MAX_ITERATIONS='{env_value}', using default value 70000.")
    STREET_GAUSSIANS_MAX_ITERATIONS = 70000

print(f"3DGS: STREET_GAUSSIANS_MAX_ITERATIONS={STREET_GAUSSIANS_MAX_ITERATIONS}")

# 少量迭代时让 eval 在训练过程中实际触发（step 范围是 [0, max_num_iterations)）
if STREET_GAUSSIANS_MAX_ITERATIONS <= 100:
    _eval_every = max(2, STREET_GAUSSIANS_MAX_ITERATIONS // 3)
    _steps_per_eval_image = _eval_every
    _steps_per_eval_batch = _eval_every
    _steps_per_save = STREET_GAUSSIANS_MAX_ITERATIONS
    _steps_per_eval_all_images = max(_eval_every, STREET_GAUSSIANS_MAX_ITERATIONS - 1)
else:
    _steps_per_eval_image = 500
    _steps_per_eval_batch = 500
    _steps_per_save = 2000
    _steps_per_eval_all_images = 70000

street_gaussians_ns_method = MethodSpecification(
    config=TrainerConfig(
        method_name="street-gaussians-ns",
        steps_per_eval_image=_steps_per_eval_image,
        steps_per_eval_batch=_steps_per_eval_batch,
        steps_per_save=_steps_per_save,
        steps_per_eval_all_images=_steps_per_eval_all_images,
        max_num_iterations=STREET_GAUSSIANS_MAX_ITERATIONS,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, 'semantic': 10},
        pipeline=SgnPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(
                    load_3D_points=True,
                    max_2D_matches_per_3D_point=0,
                    undistort=True,
                    colmap_path=Path("colmap/sparse/0"),
                    segments_path=Path("segs"),
                    depths_path=Path("depths"),
                    load_dynamic_annotations=True,
                ),
            ),
            model=SplatfactoSceneGraphModelConfig(
                # TODO simplify this, warper model use background_model directly
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                bbox_optimizer=BBoxOptimizerConfig(mode="simple"),
                use_sky_sphere=True,
                sh_degree=3,
                background_model=SplatfactoModelConfig(
                    cull_alpha_thresh=0.02,
                    cull_scale_thresh=0.2,
                    # densify_grad_thresh=0.0001,
                    warmup_length=500,
                    refine_every=100,
                    reset_alpha_every=30,
                    stop_split_at=50000,
                    fourier_features_dim=1,
                    depth_loss_mult=0.05,
                    depth_loss_start_step=500,
                    output_depth_during_training=True,
                ),
                object_model_template=SplatfactoModelConfig(
                    cull_alpha_thresh=0.005,
                    cull_scale_thresh=0.2,
                    densify_grad_thresh=0.0002,
                    warmup_length=500,
                    refine_every=100,
                    reset_alpha_every=30,
                    stop_split_at=50000,
                    fourier_features_dim=5,
                    num_random=10000,
                )
            ),
        ),
        optimizers={
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000, warmup_steps=1000, lr_pre_warmup=0),
            },
            "sky_sphere": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-6, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=70000),
            },
            "bbox_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-6, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=70000),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-7,
                    max_steps=70000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.001 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_legacy+tensorboard",
    ),
    description="Base config for Street Gaussians",
)
