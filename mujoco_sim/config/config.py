import numpy as np
import mujoco

class PegEnvConfig:
    """Set the configuration for FrankaEnv."""

    def __init__(self, **kwargs):
        # General Environment Configuration
        self.ENV_CONFIG = {
            # "action_scale": np.array(
            #     [0.0012, 0.015, 1]
            "action_scale": np.array(
                [1, 1, 1]
            ),  # Scaling factors for position, orientation, and gripper control
            "control_dt": 0.02,  # Time step for controller updates
            "physics_dt": 0.002,  # Time step for physics simulation
            "time_limit": 5.0,  # Time limit for each episode
            "seed": 0,  # Random seed
            "camera_ids" : [1, 2, 3, 4],  # Camera IDs for rendering"
            "version": 0,    # BUG --> pose-orientation-obs are velocities --> demos for "medium" made with velocities (no impact here but for "hard")
            "enable_force_feedback": False,  # Enable force feedback
            "enable_force_visualization": True,  # Enable visualization
            "enable_slider_controller": False,  # Enable slider controller
        }

        self.DEFAULT_CAM_CONFIG = {
        'type': mujoco.mjtCamera.mjCAMERA_FREE,  # Camera type
        'fixedcamid': 0,                            # ID of the fixed camera
        'lookat': np.array([-0.13658866, -0.07953108, -0.12050411]),        # Point the camera looks at
        'distance': 0.7681,                            # Distance from the lookat point
        'azimuth': -170.1003,                           # Horizontal angle
        'elevation': -19.9855,                         # Vertical angle
        }

        # UR5e Robot Configuration
        self.UR5E_CONFIG = {
            "home_position": np.array(
                [
                    -1.90726387,
                    -1.56481928,
                    2.34983533,
                    -2.35537841,
                    -1.57080242,
                    1.23432895,
                ]
            ),
            "reset_position": np.array(
                [
                    -1.97248201,
                    -1.80736221,
                    2.08230528,
                    -1.84573939,
                    -1.57079633,
                    1.16911064,
                ]
            ),  # Reset joint angles
            "default_cartesian_bounds": np.array(
                [[0.2, -0.3, 0.0], [0.6, 0.3, 0.05]]
            ),  # Workspace boundaries in Cartesian space
            "restrict_cartesian_bounds": True,  # Whether to restrict the end effector to the Cartesian bounds
            "default_port_pos": np.array([0.4, 0.0, 0.0]),  # Default port position
            "port_sampling_bounds": np.array(
                [[0.3925, -0.0075, 0], [0.4075, 0.0075, 0]]
            ),  # Sampling range for port placement
            "port_xy_randomize": False,  # Randomize port xy placement
            "port_z_randomize": False,  # Randomize port z placement
            "port_orientation_randomize": False,  # Randomize port placement
            "max_port_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },
            "tcp_xyz_randomize": False,  # Randomize tcp xyz placement
            "mocap_orient": True,  # Orient the tcp to the port
            "max_mocap_orient_randomize": {
                "x": 0,  # Maximum deviation in degrees around x-axis
                "y": 0,  # Maximum deviation in degrees around y-axis
                "z": 0,  # Maximum deviation in degrees around z-axis
            },
            "randomization_bounds": np.array(
                [[-0.005, -0.005, 0.06], [0.005, 0.005, 0.06]]
            ),  # Randomization bounds for port positions xyz
            "reset_tolerance": 0.0005,
        }

        # Controller Configuration
        self.CONTROLLER_CONFIG = {
            "trans_damping_ratio": 0.996,  # Damping ratio for translational control
            "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
            "error_tolerance_pos": 0.001,  # Position error tolerance
            "error_tolerance_ori": 0.001,  # Orientation error tolerance
            "trans_clip_min": np.array([-0.01, -0.01, -0.01]),  # Translational negative clipping limits
            "trans_clip_max": np.array([0.01, 0.01, 0.01]),  # Translational positive clipping limits
            "rot_clip_min": np.array([-0.03, -0.03, -0.03]),  # Rotational negative clipping limits
            "rot_clip_max": np.array([0.03, 0.03, 0.03]),  # Rotational positive clipping
            "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
            "inertia_compensation": False,  # Whether to compensate for inertia
            "pos_gains": (100, 100, 100),  # Proportional gains for position control
            # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
            "max_angvel": 4,  # Maximum angular velocity
            "integration_dt": 0.2,  # Integration time step for controller
            "gravity_compensation": True,  # Whether to compensate for gravity
        }


        # Rendering Configuration
        self.RENDERING_CONFIG = {
            "width": 640,  # Rendering width
            "height": 480,  # Rendering height
        }

        # Reward Shaping
        self.REWARD_CONFIG = {
            "reward_shaping": False,  # Use dense reward shaping
            "dense_reward_weights": {
                "box_target": 1.0,  # Weight for reaching target position
            },
            "sparse_reward_weights": 0,  # Reward for completing the task
            "task_complete_tolerance": 0.0025,  # Distance threshold for task completion
        }

        # Update configurations with provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr_value = getattr(self, key)
                if isinstance(attr_value, dict) and isinstance(value, dict):
                    attr_value.update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
