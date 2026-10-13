"""DR02 Pro AMP environment on rough terrain."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.mdp import terrain_levels_vel

from ... import mdp
from .env_cfg import AMPObservationsCfg, AMPSceneCfg, DR02AMPFlatEnvCfg


def _terrain_height_grid(cfg: patterns.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Start yaw-aligned rays above the robot without shifting the scan grid."""
    ray_starts, ray_directions = patterns.grid_pattern(cfg, device)
    ray_starts[:, 2] = 2.0
    return ray_starts, ray_directions


@configclass
class DR02AMPRoughSceneCfg(AMPSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            func=_terrain_height_grid,
            resolution=0.1,
            size=(1.6, 1.0),
            ordering="yx",
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class DR02AMPRoughObservationsCfg(AMPObservationsCfg):
    @configclass
    class CriticCfg(AMPObservationsCfg.CriticCfg):
        terrain_heights = ObsTerm(
            func=mdp.critic_terrain_heights,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class DR02AMPRoughCurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_vel)


@configclass
class DR02AMPRoughEnvCfg(DR02AMPFlatEnvCfg):
    """AMP task with rough terrain, critic height sensing, and terrain curriculum."""

    scene: DR02AMPRoughSceneCfg = DR02AMPRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DR02AMPRoughObservationsCfg = DR02AMPRoughObservationsCfg()
    curriculum: DR02AMPRoughCurriculumCfg = DR02AMPRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG.copy().replace(curriculum=True),
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=1.0,
            ),
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.events.reset_reference.params["xy_range"] = (-1.0, 1.0)
        self.events.reset_reference.params["terrain_height_sensor"] = "height_scanner"
