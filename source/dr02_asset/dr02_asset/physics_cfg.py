"""Physics backend presets shared by DR02 AMP and tracking tasks."""

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class DR02PhysicsCfg(PresetCfg):
    """Select Newton by default, with the two supported PhysX runtimes available."""

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx: OvPhysxCfg = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp
