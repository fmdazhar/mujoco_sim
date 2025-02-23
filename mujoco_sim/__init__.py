from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register, WrapperSpec

register(
    id="ur5ePegInHoleGymEnv_easy-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="XYZGripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:XYZGripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "connector_pose",
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 0,
            },
            "UR5E_CONFIG": {
                "tcp_xyz_randomize": True,
                "mocap_orient": True,
            },
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_medium-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="XYZGripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:XYZGripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 0,
            },
            "UR5E_CONFIG": {
                "port_xy_randomize": True,  # Randomize port xy placement
                "port_z_randomize": False,  # Randomize port z placement
                "tcp_xyz_randomize": True,
                "mocap_orient": True,
            },
            "CONTROLLER_CONFIG": {
                "trans_damping_ratio": 0.996,  # Damping ratio for translational control
                "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
                "error_tolerance_pos": 0.001,  # Position error tolerance
                "error_tolerance_ori": 0.001,  # Orientation error tolerance
                "max_pos_error": 0.01,  # Maximum position error
                "max_ori_error": 0.03,  # Maximum orientation error
                "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
                "inertia_compensation": False,  # Whether to compensate for inertia
                "pos_gains": (100, 100, 100),  # Proportional gains for position control
                # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
                "max_angvel": 4,  # Maximum angular velocity
                "integration_dt": 0.2,  # Integration time step for controller
                "gravity_compensation": True,  # Whether to compensate for gravity
            }
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_medium-v1",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="XYZGripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:XYZGripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="StackObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:StackObsWrapper",
            kwargs={"k": 4},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 0,
            },
            "UR5E_CONFIG": {
                "port_xy_randomize": True,  # Randomize port xy placement
                "port_z_randomize": False,  # Randomize port z placement
                "tcp_xyz_randomize": True,
                "mocap_orient": True,
            },
            "CONTROLLER_CONFIG": {
                "trans_damping_ratio": 0.996,  # Damping ratio for translational control
                "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
                "error_tolerance_pos": 0.001,  # Position error tolerance
                "error_tolerance_ori": 0.001,  # Orientation error tolerance
                "max_pos_error": 0.01,  # Maximum position error
                "max_ori_error": 0.03,  # Maximum orientation error
                "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
                "inertia_compensation": False,  # Whether to compensate for inertia
                "pos_gains": (100, 100, 100),  # Proportional gains for position control
                # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
                "max_angvel": 4,  # Maximum angular velocity
                "integration_dt": 0.2,  # Integration time step for controller
                "gravity_compensation": True,  # Whether to compensate for gravity
            }
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_hard-v0",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="XYZQzGripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:XYZQzGripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 1,
            },
            "UR5E_CONFIG": {
                "port_xy_randomize": True,  # Randomize port xy placement
                "port_z_randomize": False,  # Randomize port z placement
                "port_orientation_randomize": True,  # Randomize port placement
                "max_port_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 5,  # Maximum deviation in degrees around z-axis
                },
                "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
                "mocap_orient": False,
                "max_mocap_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 0,  # Maximum deviation in degrees around z-axis
                },
            },
            "CONTROLLER_CONFIG":{
                "trans_damping_ratio": 0.996,  # Damping ratio for translational control
                "rot_damping_ratio": 0.6,  # Damping ratio for rotational control
                "error_tolerance_pos": 0.001,  # Position error tolerance
                "error_tolerance_ori": 0.001,  # Orientation error tolerance
                "max_pos_error": 0.01,  # Maximum position error
                "max_ori_error": 10,  # Maximum orientation error
                "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
                "inertia_compensation": False,  # Whether to compensate for inertia
                "pos_gains": (500, 500, 500),  # Proportional gains for position control
                "ori_gains": (1, 1, 1),  # Proportional gains for orientation control
                "max_angvel": 50,  # Maximum angular velocity
                "integration_dt": 0.2,  # Integration time step for controller
                "gravity_compensation": True,  # Whether to compensate for gravity
            }
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_hard-v1",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="XYZQzGripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:XYZQzGripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="StackObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:StackObsWrapper",
            kwargs={"k": 4},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 1,
            },
            "UR5E_CONFIG": {
                "port_xy_randomize": True,  # Randomize port xy placement
                "port_z_randomize": False,  # Randomize port z placement
                "port_orientation_randomize": True,  # Randomize port placement
                "max_port_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 5,  # Maximum deviation in degrees around z-axis
                },
                "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
                "mocap_orient": False,
                "max_mocap_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 0,  # Maximum deviation in degrees around z-axis
                },
            },
            "CONTROLLER_CONFIG":{
                "trans_damping_ratio": 0.996,  # Damping ratio for translational control
                "rot_damping_ratio": 0.996,  # Damping ratio for rotational control
                "error_tolerance_pos": 0.001,  # Position error tolerance
                "error_tolerance_ori": 0.001,  # Orientation error tolerance
                "max_pos_error": 100,  # Maximum position error
                "max_ori_error": 90,  # Maximum orientation error
                "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
                "inertia_compensation": False,  # Whether to compensate for inertia
                "pos_gains": (100, 100, 100),  # Proportional gains for position control
                "ori_gains": (5, 5, 5),  # Proportional gains for orientation control
                "max_angvel": 5000,  # Maximum angular velocity
                "integration_dt": 0.2,  # Integration time step for controller
                "gravity_compensation": True,  # Whether to compensate for gravity
            }
        },
    },
)

register(
    id="ur5ePegInHoleGymEnv_very_hard-v1",
    entry_point="mujoco_sim.envs:ur5ePegInHoleGymEnv",
    additional_wrappers=(
        WrapperSpec(
            name="GripperCloseEnv",
            entry_point="mujoco_sim.envs.wrappers:GripperCloseEnv",  # Replace with actual module path
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="SpacemouseIntervention",
            entry_point="mujoco_sim.envs.wrappers:SpacemouseIntervention",
            kwargs={},  # Add any necessary kwargs for this wrapper
        ),
        WrapperSpec(
            name="CustomObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:CustomObsWrapper",  # Replace with actual module path
            kwargs={
                "keys_to_keep": [
                    "controller_pose",
                    "ur5e/tcp_pose",
                    "ur5e/tcp_vel",
                    "ur5e/wrist_force",
                    "ur5e/wrist_torque",
                ],
            },
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="StackObsWrapper",
            entry_point="mujoco_sim.envs.wrappers:StackObsWrapper",
            kwargs={"k": 4},  # Add any necessary kwargs for this wrapper if needed
        ),
        WrapperSpec(
            name="FlattenObservation",
            entry_point="gymnasium.wrappers:FlattenObservation",
            kwargs={},  # Add any necessary kwargs for this wrapper if needed
        ),
    ),
    kwargs={
        "config": {
            "ENV_CONFIG": {
                "version": 2,
            },
            "UR5E_CONFIG": {
                "port_xy_randomize": True,  # Randomize port xy placement
                "port_z_randomize": False,  # Randomize port z placement
                "port_orientation_randomize": True,  # Randomize port placement
                "max_port_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 5,  # Maximum deviation in degrees around z-axis
                },
                "tcp_xyz_randomize": True,  # Randomize tcp xyz placement
                "mocap_orient": False,
                "max_mocap_orient_randomize": {
                    "x": 0,  # Maximum deviation in degrees around x-axis
                    "y": 0,  # Maximum deviation in degrees around y-axis
                    "z": 0,  # Maximum deviation in degrees around z-axis
                },
            },
            "CONTROLLER_CONFIG":{
                "trans_damping_ratio": 0.996,  # Damping ratio for translational control
                "rot_damping_ratio": 0.996,  # Damping ratio for rotational control
                "error_tolerance_pos": 0.001,  # Position error tolerance
                "error_tolerance_ori": 0.001,  # Orientation error tolerance
                "max_pos_error": 100,  # Maximum position error
                "max_ori_error": 90,  # Maximum orientation error
                "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
                "inertia_compensation": False,  # Whether to compensate for inertia
                "pos_gains": (100, 100, 100),  # Proportional gains for position control
                "ori_gains": (10, 10, 10),  # Proportional gains for orientation control
                "max_angvel": 5000,  # Maximum angular velocity
                "integration_dt": 0.2,  # Integration time step for controller
                "gravity_compensation": True,  # Whether to compensate for gravity
            }
        },
    },
)
