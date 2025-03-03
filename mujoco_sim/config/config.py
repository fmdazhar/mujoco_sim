import numpy as np
import mujoco

class PegEnvConfig:
    """Configuration settings for the UR5e Peg-In-Hole Gym Environment."""

    def __init__(self, **kwargs):
        # ----------------------------- #
        # General Environment Settings  #
        # ----------------------------- #
        self.ENV_CONFIG = {
            "action_scale": np.array([1, 1, 1]),  # Scaling factors for position, orientation, and gripper control
            "control_dt": 0.02,  # Controller update time step (s)
            "physics_dt": 0.002,  # Physics simulation time step (s)
            "_max_episode_steps": 1000.0,  # Maximum episode steps
            "seed": 0,  # Random seed for reproducibility
            "camera_ids": [],  # Camera IDs used for rendering
            "version": 0,  # Environment version (affects observation space and behavior)
            "enable_force_feedback": False,  # Enable force feedback visualization
            "enable_force_visualization": True,  # Enable visualization of contact forces
            "enable_slider_controller": False,  # Enable interactive slider-based controller tuning
        }

        # ---------------------------- #
        # Camera Configuration         #
        # ---------------------------- #
        self.DEFAULT_CAM_CONFIG = {
            'type': mujoco.mjtCamera.mjCAMERA_FREE,  # Camera type (free, fixed, etc.)
            'fixedcamid': 0,  # ID of the fixed camera
            'lookat': np.array([-0.1366, -0.0795, -0.1205]),  # Camera focus point in the scene
            'distance': 0.7681,  # Distance from focus point
            'azimuth': -170.1,  # Horizontal viewing angle (degrees)
            'elevation': -19.99,  # Vertical viewing angle (degrees)
        }

        # ---------------------------- #
        # UR5e Robot Configuration     #
        # ---------------------------- #
        self.UR5E_CONFIG = {
            "home_position": np.array([-1.907, -1.565, 2.350, -2.355, -1.571, 1.234]),  # Joint angles for home position
            "default_cartesian_bounds": np.array([[0.2, -0.3, 0.0], [0.6, 0.3, 0.05]]),  # XYZ workspace limits
            "restrict_cartesian_bounds": True,  # Constrain motion to defined Cartesian boundaries
            "default_port_pos": np.array([0.4, 0.0, 0.0]),  # Default position of the insertion port
            "port_sampling_bounds": np.array([[0.3925, -0.0075, 0], [0.4075, 0.0075, 0]]),  # Randomized port placement bounds
            "port_xy_randomize": False,  # Enable randomization of the port’s XY position
            "port_z_randomize": False,  # Enable randomization of the port’s Z position
            "port_orientation_randomize": False,  # Randomize port rotation
            "max_port_orient_randomize": {"x": 0, "y": 0, "z": 0},  # Max allowed orientation deviation (degrees)
            "tcp_xyz_randomize": False,  # Randomize tool center point (TCP) position
            "mocap_orient": True,  # Set mocap orientation to align TCP with port
            "max_mocap_orient_randomize": {"x": 0, "y": 0, "z": 0},  # Max allowed mocap orientation deviation (degrees)
            "randomization_bounds": np.array([[-0.005, -0.005, 0.06], [0.005, 0.005, 0.06]]),  # Bounds for randomization
            "reset_tolerance": 0.0005,  # Tolerance threshold for successful reset
        }

        # ---------------------------- #
        # Controller Configuration     #
        # ---------------------------- #
        self.CONTROLLER_CONFIG = {
            "trans_damping_ratio": 0.996,  # Translational damping ratio (affects stability)
            "rot_damping_ratio": 0.286,  # Rotational damping ratio
            "error_tolerance_pos": 0.001,  # Position error tolerance
            "error_tolerance_ori": 0.001,  # Orientation error tolerance
            "trans_clip_min": np.array([-0.01, -0.01, -0.01]),  # Min translational velocity limits
            "trans_clip_max": np.array([0.01, 0.01, 0.01]),  # Max translational velocity limits
            "rot_clip_min": np.array([-0.03, -0.03, -0.03]),  # Min rotational velocity limits
            "rot_clip_max": np.array([0.03, 0.03, 0.03]),  # Max rotational velocity limits
            "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
            "inertia_compensation": False,  # Enable compensation for robot inertia for "dynamics" method
            "pos_gains": (100, 100, 100),  # Proportional gain values for position control
            "max_angvel": 4,  # Maximum angular velocity
            "integration_dt": 0.2,  # Integration time step for controller
            "gravity_compensation": True,  # Enable gravity compensation
        }

        # ---------------------------- #
        # Rendering Configuration      #
        # ---------------------------- #
        self.RENDERING_CONFIG = {
            "width": 640,  # Image width
            "height": 480,  # Image height
        }

        # ---------------------------- #
        # Reward Function Configuration#
        # ---------------------------- #
        self.REWARD_CONFIG = {
            "reward_shaping": False,  # Use dense reward signals for learning
            "dense_reward_weights": {
                "box_target": 1.0,  # Reward weight for reaching the target
            },
            "sparse_reward_weights": 0,  # Sparse Reward given upon task completion
            "task_complete_tolerance": 0.0025,  # Distance threshold for considering task complete
        }

        # Update configurations with provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr_value = getattr(self, key)
                if isinstance(attr_value, dict) and isinstance(value, dict):
                    attr_value.update(value)  # Merge dictionaries
                else:
                    setattr(self, key, value)  # Overwrite attributes
            else:
                setattr(self, key, value)  # Add new attributes dynamically
