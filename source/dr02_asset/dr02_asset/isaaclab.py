"""Isaac Lab articulation configuration for the DR02 Pro model."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

_DESCRIPTION = Path(__file__).parent / "description"
_URDF = _DESCRIPTION / "urdf" / "dr02_pro.urdf"

DR02_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_URDF),
        fix_base=False,
        run_asset_transformer=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            "^(?!.*(?:_elbow_joint|_shoulder_z_joint)$).*$": 0.0,
            "left_shoulder_z_joint": 0.765,
            "right_shoulder_z_joint": -0.765,
            ".*_elbow_joint": 1.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": DCMotorCfg(
            joint_names_expr=["waist_[xyz]_joint"],
            effort_limit=105.0,
            saturation_effort=105.0,
            velocity_limit=17.38,
            stiffness=150.0,
            damping=3.0,
            friction=0.0,
        ),
        "arms": DCMotorCfg(
            joint_names_expr=[".*_shoulder_[xyz]_joint", ".*_elbow_joint"],
            effort_limit=105.0,
            saturation_effort=105.0,
            velocity_limit=17.38,
            stiffness=100.0,
            damping=2.5,
            armature={
                ".*_shoulder_y_joint": 0.6664750278,
                ".*_shoulder_x_joint": 0.5299741328,
                ".*_shoulder_z_joint": 0.5262607038,
                ".*_elbow_joint": 0.1438293383,
            },
            friction=0.0,
        ),
        "wrists": DCMotorCfg(
            joint_names_expr=[".*_wrist_[xyz]_joint"],
            effort_limit=50.0,
            saturation_effort=50.0,
            velocity_limit=23.76,
            stiffness=100.0,
            damping=2.5,
            armature={
                ".*_wrist_z_joint": 0.1398061365,
                ".*_wrist_y_joint": 0.0234856559,
                ".*_wrist_x_joint": 0.0175240059,
            },
            friction=0.0,
        ),
        "hip_pitch_roll": DCMotorCfg(
            joint_names_expr=[".*_hip_[xy]_joint"],
            effort_limit=330.0,
            saturation_effort=330.0,
            velocity_limit=18.0,
            stiffness=500.0,
            damping=8.5,
            friction=0.0,
        ),
        "knees": DCMotorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=330.0,
            saturation_effort=330.0,
            velocity_limit=18.0,
            stiffness=500.0,
            damping=8.5,
            friction=0.0,
        ),
        "hip_yaw": DCMotorCfg(
            joint_names_expr=[".*_hip_z_joint"],
            effort_limit=105.0,
            saturation_effort=105.0,
            velocity_limit=17.38,
            stiffness=180.0,
            damping=4.0,
            friction=0.0,
        ),
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=[".*_ankle_y_joint"],
            effort_limit=105.0,
            saturation_effort=105.0,
            velocity_limit=17.38,
            stiffness=100.0,
            damping=2.5,
            friction=0.0,
        ),
        "ankle_roll": DCMotorCfg(
            joint_names_expr=[".*_ankle_x_joint"],
            effort_limit=35.0,
            saturation_effort=35.0,
            velocity_limit=20.76,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
