from pathlib import Path
from typing import Any, Dict, Tuple, Literal
import mujoco
import numpy as np
import gymnasium
from gymnasium import spaces
from mujoco_sim.mujoco_rendering import MujocoRenderer
from mujoco_sim.controllers import Controller
from mujoco_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from mujoco_sim.config import PegEnvConfig
from mujoco_sim.utils.slider_controller import SliderController
from mujoco_sim.utils.force_feedback import ForceFeedback
from scipy.spatial.transform import Rotation
import glfw
from OpenGL.GL import *

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "ur5e_demos.xml"


class ur5ePegInHoleGymEnv(MujocoGymEnv):
    """
    UR5e peg-in-hole environment using Mujoco.
    
    This environment implements a simulation of the UR5e robot performing a peg-in-hole task.
    It provides methods for resetting the environment, stepping through the simulation,
    and computing observations and rewards.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
    }

    def __init__(
        self, render_mode: Literal["rgb_array", "human"] = "rgb_array", config: Any = None
    ):
        """
        Initialize the UR5e peg-in-hole environment.

        Parameters:
            render_mode (Literal["rgb_array", "human"]): The mode to render the environment.
            config (Optional[Any]): Configuration dictionary or object for the environment.
        """
        if config is None:
            config = PegEnvConfig()
        else:
            config = PegEnvConfig(**config)

        self.render_width = config.RENDERING_CONFIG["width"]
        self.render_height = config.RENDERING_CONFIG["height"]
        self.camera_id = config.ENV_CONFIG["camera_ids"]
        render_spec = GymRenderingSpec(
            height=config.RENDERING_CONFIG["height"],
            width=config.RENDERING_CONFIG["width"],
        )

        super().__init__(
            xml_path=_XML_PATH,
            control_dt=config.ENV_CONFIG["control_dt"],
            physics_dt=config.ENV_CONFIG["physics_dt"],
            time_limit=config.ENV_CONFIG["time_limit"],
            seed=config.ENV_CONFIG["seed"],
            render_spec=render_spec,
        )

        self._action_scale = config.ENV_CONFIG["action_scale"]
        self.env_version = config.ENV_CONFIG["version"],
        self.env_version = self.env_version[0]
        self.enable_force_feedback = config.ENV_CONFIG.get("enable_force_feedback", False)
        self.enable_slider_controller = config.ENV_CONFIG.get("enable_slider_controller", False)
        self.enable_force_visualization = config.ENV_CONFIG.get("enable_force_visualization", False)

        self.render_mode = render_mode
        self.default_cam_config = config.DEFAULT_CAM_CONFIG

        self.ur5e_home = config.UR5E_CONFIG["home_position"]
        self.ur5e_reset = config.UR5E_CONFIG["reset_position"]
        self.cartesian_bounds = config.UR5E_CONFIG["default_cartesian_bounds"]
        self.restrict_cartesian_bounds = config.UR5E_CONFIG["restrict_cartesian_bounds"]
        self.default_port_pos = config.UR5E_CONFIG["default_port_pos"]
        self.port_sampling_bounds = config.UR5E_CONFIG["port_sampling_bounds"]
        self.tcp_xyz_randomize = config.UR5E_CONFIG["tcp_xyz_randomize"]
        self.port_xy_randomize = config.UR5E_CONFIG["port_xy_randomize"]
        self.port_z_randomize = config.UR5E_CONFIG["port_z_randomize"]
        self.port_orientation_randomize = config.UR5E_CONFIG["port_orientation_randomize"]
        self.max_port_orient = config.UR5E_CONFIG["max_port_orient_randomize"]
        self.mocap_orient = config.UR5E_CONFIG["mocap_orient"]
        self.max_mocap_orient = config.UR5E_CONFIG["max_mocap_orient_randomize"]
        self.randomization_bounds = config.UR5E_CONFIG["randomization_bounds"]
        self.reset_tolerance = config.UR5E_CONFIG["reset_tolerance"]

        self.reward_config = config.REWARD_CONFIG
        self.sparse_reward = 0.0
        self.dense_reward = 0.0

        self.external_viewer = None

        self._ur5e_dof_ids = np.asarray(
            [
                self._model.joint("shoulder_pan_joint").id,
                self._model.joint("shoulder_lift_joint").id,
                self._model.joint("elbow_joint").id,
                self._model.joint("wrist_1_joint").id,
                self._model.joint("wrist_2_joint").id,
                self._model.joint("wrist_3_joint").id,
            ]
        )

        self._hande_dof_ids = np.asarray(
            [
                self._model.joint("hande_left_finger_joint").id,
                self._model.joint("hande_right_finger_joint").id,
            ]
        )

        self._ur5e_ctrl_ids = np.asarray(
            [
                self._model.actuator("shoulder_pan").id,
                self._model.actuator("shoulder_lift").id,
                self._model.actuator("elbow").id,
                self._model.actuator("wrist_1").id,
                self._model.actuator("wrist_2").id,
                self._model.actuator("wrist_3").id,
            ]
        )

        self._gripper_ctrl_id = self._model.actuator("hande_fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._port_id = self._model.body("port_adapter").id
        self._port1_id = self._model.body("port1").id
        self._port_site = self._model.site("port_top").id
        self._port_site_id = self._model.site("port_top").id
        self._hande_right_id = self._model.body("hande_right_finger").id
        self._hande_left_id = self._model.body("hande_left_finger").id
        self._attatchment_id = self._model.site("attachment_site").id
        self._connector_top_id = self._model.site("connector_top").id
        self._connector_bottom_id = self._model.site("connector_bottom").id

        self._floor_geom = self._model.geom("floor").id
        self._left_finger_geom = self._model.geom("left_pad1").id
        self._right_finger_geom = self._model.geom("right_pad1").id
        self._hand_geom = self._model.geom("hande_base").id
        self._connector_head_geom = self._model.geom("connector_head").id

        self._cartesian_bounds_geom_id = self._model.geom("cartesian_bounds").id
        lower_bounds, upper_bounds = self.cartesian_bounds
        center = (upper_bounds + lower_bounds) / 2.0
        size = (upper_bounds - lower_bounds) / 2.0
        self._model.geom_size[self._cartesian_bounds_geom_id] = size
        self._model.geom_pos[self._cartesian_bounds_geom_id] = center

        self.quat_err = np.zeros(4)
        self.quat_conj = np.zeros(4)
        self.ori_err = np.zeros(3)

        self.controller = Controller(
            model=self._model,
            data=self._data,
            site_id=self._pinch_site_id,
            dof_ids=self._ur5e_dof_ids,
            config=config.CONTROLLER_CONFIG,
        )

        if self.render_mode == "human":
            if self.enable_force_feedback:
                self.force_feedback = ForceFeedback()
            if self.enable_slider_controller:
                self.slider_controller = SliderController(self.controller)

        obs_bound = 1e6
        self.observation_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Dict(
                    {
                        "controller_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "ur5e/tcp_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "ur5e/tcp_vel": spaces.Box(
                            -obs_bound, obs_bound, shape=(3,), dtype=np.float32
                        ),
                        "ur5e/joint_pos": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "ur5e/joint_vel": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "ur5e/wrist_force": spaces.Box(
                            -obs_bound, obs_bound, shape=(3,), dtype=np.float32
                        ),
                        "ur5e/wrist_torque": spaces.Box(
                            -obs_bound, obs_bound, shape=(3,), dtype=np.float32
                        ),
                        "connector_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                        "port_pose": spaces.Box(
                            -obs_bound, obs_bound, shape=(6,), dtype=np.float32
                        ),
                    }
                ),
            }
        )
        # self.action_space = gymnasium.spaces.Box(
        #     low=np.asarray(
        #         [
        #             -0.01 / self._action_scale[0],
        #             -0.01 / self._action_scale[0],
        #             -0.01 / self._action_scale[0],
        #             -0.01 / self._action_scale[1],
        #             -0.01 / self._action_scale[1],
        #             -0.01 / self._action_scale[1],
        #             -1.0 / self._action_scale[2],
        #         ]
        #     ),
        #     high=np.asarray(
        #         [
        #             0.01 / self._action_scale[0],
        #             0.01 / self._action_scale[0],
        #             0.01 / self._action_scale[0],
        #             0.01 / self._action_scale[1],
        #             0.01 / self._action_scale[1],
        #             0.01 / self._action_scale[1],
        #             1.0 / self._action_scale[2],
        #         ]
        #     ),
        #     dtype=np.float32,
        # )

        self.action_space = gymnasium.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # glfw init
        glfw.init()
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
            offscreen_width=render_spec.width,
            offscreen_height=render_spec.height,
            window_width=width,
            window_height=height,
            default_cam_config=self.default_cam_config,
            extra_views_camera_ids = self.camera_id,
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.control_dt))

    def reset(self, seed: Any = None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Parameters:
            seed: Optional seed for randomization.
            **kwargs: Additional arguments.

        Returns:
            A tuple containing the observation dictionary and an empty info dictionary.
        """
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._ur5e_dof_ids] = self.ur5e_home
        self._data.qvel[self._ur5e_dof_ids] = 0
        mujoco.mj_forward(self._model, self._data)

        port_xy = self.default_port_pos[:2]
        port_z = self.default_port_pos[2]

        if self.port_xy_randomize:
            port_xy = np.random.uniform(
                self.port_sampling_bounds[0][:2], self.port_sampling_bounds[1][:2]
            )
        self._model.body_pos[self._port_id][:2] = port_xy

        if self.port_z_randomize:
            port_z = np.random.uniform(
                self.port_sampling_bounds[0][2], self.port_sampling_bounds[1][2]
            )
        self._model.body_pos[self._port_id][2] = port_z

        if self.port_orientation_randomize:
            max_angle_rad_x = np.deg2rad(self.max_port_orient["x"])
            max_angle_rad_y = np.deg2rad(self.max_port_orient["y"])
            max_angle_rad_z = np.deg2rad(self.max_port_orient["z"])
            random_angle_x = np.random.uniform(-max_angle_rad_x, max_angle_rad_x)
            random_angle_y = np.random.uniform(-max_angle_rad_y, max_angle_rad_y)
            random_angle_z = np.random.uniform(-max_angle_rad_z, max_angle_rad_z)
            random_angles = np.array([random_angle_x, random_angle_y, random_angle_z])
            quat_des = Rotation.from_euler('xyz', random_angles, degrees=False).as_quat()
            quat_des = np.roll(quat_des, shift=1)
            self._model.body_quat[self._port_id] = quat_des

        mujoco.mj_forward(self._model, self._data)

        plate_pos = self._data.geom("plate").xpos
        half_width, half_height, half_depth = self._model.geom("plate").size
        local_vertices = np.array(
            [
                [half_width, half_height, half_depth],
                [half_width, half_height, -half_depth],
                [half_width, -half_height, half_depth],
                [half_width, -half_height, -half_depth],
                [-half_width, half_height, half_depth],
                [-half_width, half_height, -half_depth],
                [-half_width, -half_height, half_depth],
                [-half_width, -half_height, -half_depth],
            ]
        )
        rotation_matrix = self.data.xmat[self._port_id].reshape(3, 3)
        rotated_vertices = local_vertices @ rotation_matrix.T + plate_pos
        z_coords = rotated_vertices[:, 2]
        z_min = np.min(z_coords)
        z_offset = -z_min if z_min < 0.0 else 0.0
        self._model.body_pos[self._port_id][2] += z_offset
        mujoco.mj_forward(self._model, self._data)

        if self.restrict_cartesian_bounds:
            self._model.geom_size[self._cartesian_bounds_geom_id][:2] = (
                self._model.geom("plate").size[:2] * np.array([0.5, 0.6])
            ) * np.array([0.75, 0.75])
            lower_bound = self._data.sensor("port_bottom_pos").data[2] - np.array([0.01])
            upper_bound = self._data.sensor("port_bottom_pos").data[2] + np.array([0.05])
            self._model.geom_size[self._cartesian_bounds_geom_id][2] = (upper_bound - lower_bound) / 2.0
            self._model.geom_pos[self._cartesian_bounds_geom_id] = self._data.geom("plate").xpos
            self._model.geom_pos[self._cartesian_bounds_geom_id] += np.array([0, 0, 0.05] @ rotation_matrix.T)
            self._model.geom_quat[self._cartesian_bounds_geom_id] = self._model.body_quat[self._port_id]

        port_xyz = self.data.site_xpos[self._port_site_id]

        if self.tcp_xyz_randomize:
            random_xyz_local = np.random.uniform(*self.randomization_bounds)
            random_xyz_global = random_xyz_local @ rotation_matrix.T + port_xyz
            if self.env_version == 2:
                self._data.mocap_pos[0] = np.array(
                    [random_xyz_global[0] + 0.01, random_xyz_global[1] + 0.01, random_xyz_global[2]]
                )
            else:
                self._data.mocap_pos[0] = np.array(
                    [random_xyz_global[0], random_xyz_global[1], random_xyz_global[2]]
                )
        else:
            self._data.mocap_pos[0] = np.array(
                [port_xyz[0], port_xyz[1], port_xyz[2] + 0.05]
            )

        if self.mocap_orient:
            quat_des = np.zeros(4)
            mujoco.mju_mat2Quat(quat_des, self.data.site_xmat[self._port_site_id])
            self._data.mocap_quat[0] = quat_des
        else:
            ori = self._data.mocap_quat[0].copy()
            max_angle_rad_x = np.deg2rad(self.max_mocap_orient["x"])
            max_angle_rad_y = np.deg2rad(self.max_mocap_orient["y"])
            max_angle_rad_z = np.deg2rad(self.max_mocap_orient["z"])
            random_angle_x = np.random.uniform(-max_angle_rad_x, max_angle_rad_x)
            random_angle_y = np.random.uniform(-max_angle_rad_y, max_angle_rad_y)
            random_angle_z = np.random.uniform(-max_angle_rad_z, max_angle_rad_z)
            if self.env_version == 2:
                random_angle_x -= np.pi/45
                random_angle_z -= np.pi/10
            random_angles = np.array([random_angle_x, random_angle_y, random_angle_z - np.pi/2])
            new_ori = Rotation.from_euler('xyz', random_angles, degrees=False).as_quat()
            new_ori = np.roll(new_ori, shift=1)
            quat_des = np.zeros(4)
            mujoco.mju_mulQuat(quat_des, new_ori, ori)
            self._data.mocap_quat[0] = quat_des

        mujoco.mj_forward(self._model, self._data)

        # Reset the arm to the port position.  
        step_count = 0
        while step_count < 2000:
            ctrl, _ = self.controller.control(
                pos=self._data.mocap_pos[0].copy(),
                ori=self._data.mocap_quat[0].copy(),
            )
            # self._data.qpos[self._ur5e_dof_ids] = ctrl
            self._data.ctrl[self._ur5e_ctrl_ids] = ctrl
            mujoco.mj_step(self._model, self._data)
            if np.linalg.norm(self.controller.error) <= 0.0001:
                break  # Goal reached
            step_count += 1

        obs = self._compute_observation()
        return obs, {}

    def get_state(self) -> np.ndarray:
        """
        Get the current simulation state.

        Returns:
            A numpy array concatenating time, joint positions, and joint velocities.
        """
        return np.concatenate(
            [[self.data.time], np.copy(self.data.qpos), np.copy(self.data.qvel)], axis=0
        )

    def step(
        self, action: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a simulation step using the provided action.

        Parameters:
            action (np.ndarray): Action array containing deltas for position, orientation, and grasp.

        Returns:
            A tuple containing:
              - observation (dict[str, np.ndarray])
              - reward (float)
              - terminated (bool)
              - truncated (bool)
              - info (dict[str, Any])
        """
        delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, grasp = action
        TCP_site_pos = np.array(self._data.site_xpos[self._pinch_site_id])
        Bottom_site_pos = np.array(self._data.site_xpos[self._model.site("bottom_corner").id])
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([delta_x, delta_y, delta_z]) * self._action_scale[0]

        cartesian_pos = self._data.geom("cartesian_bounds").xpos
        cartesian_half_extents = self._model.geom("cartesian_bounds").size
        obb_rotation = self.data.xmat[self._port_id].reshape(3, 3)
        if self.env_version == 2:
            r_c_global = TCP_site_pos - Bottom_site_pos
            r_c_local = obb_rotation.T @ r_c_global
            nori = np.asarray([delta_qx, delta_qy, delta_qz], dtype=np.float32) * self._action_scale[1]
            nori_mat = Rotation.from_euler('xyz', nori, degrees=False).as_matrix()
            r_c_local_new = nori_mat @ r_c_local
            dpos_add = r_c_local_new - r_c_local
            dpos += dpos_add

        local_pos = obb_rotation.T @ (pos + dpos - cartesian_pos)
        clipped_local_pos = np.clip(local_pos, -cartesian_half_extents, cartesian_half_extents)
        new_pos = obb_rotation @ clipped_local_pos + cartesian_pos
        self._data.mocap_pos[0] = new_pos

        ori = self._data.mocap_quat[0].copy()
        if any([delta_qx, delta_qy, delta_qz]):
            nori = np.asarray([delta_qx, delta_qy, delta_qz], dtype=np.float32) * self._action_scale[1]
            new_ori = Rotation.from_euler('xyz', nori, degrees=False).as_quat()
            new_ori = np.roll(new_ori, shift=1)
            quat_des = np.zeros(4)
            mujoco.mju_mulQuat(quat_des, new_ori, ori)
            self._data.mocap_quat[0] = quat_des

        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[2]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            ctrl, _ = self.controller.control(
                pos=self._data.mocap_pos[0].copy(),
                ori=self._data.mocap_quat[0].copy(),
            )
            self._data.ctrl[self._ur5e_ctrl_ids] = ctrl
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew, task_complete = self._compute_reward()
        terminated = task_complete
        truncated = self.time_limit_exceeded()
        return obs, rew, terminated, truncated, {"is_success": task_complete}
    
    def render(self, cam_ids: list = None) -> np.ndarray:
        """
        Render frames from specified camera IDs.

        Parameters:
            cam_ids (list, optional): List of camera IDs to render from. 
                                    If None, defaults to self.camera_id.

        Returns:
            np.ndarray: List of rendered frames.
        """
        if cam_ids is None:
            cam_ids = [0]  # Use default camera IDs if none provided

        rendered_frames = []
        for cam_id in cam_ids:
            self._viewer.camera_id = cam_id  # Set the camera based on cam_id
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array")
            )

        return rendered_frames


    def _get_contact_info(self, geom1_id: int, geom2_id: int) -> Tuple[bool, float, np.ndarray]:
        """
        Get contact information between two geometries.

        Parameters:
            geom1_id (int): ID of the first geometry.
            geom2_id (int): ID of the second geometry.

        Returns:
            A tuple of (contact_found, distance, normal) where contact_found is True if a contact
            exists, distance is the contact distance, and normal is the contact normal vector.
        """
        for contact in self._data.contact[: self._data.ncon]:
            if {contact.geom1, contact.geom2} == {geom1_id, geom2_id}:
                distance = contact.dist
                normal = contact.frame[:3]
                return True, distance, normal
        return False, float("inf"), np.zeros(3)

    def _compute_observation(self) -> dict:
        """
        Compute and return the current observation.

        Returns:
            A dictionary containing the state observation.
        """
        obs: Dict[str, Any] = {"state": {}}

        controller_pos = self._data.mocap_pos[0].copy().astype(np.float32)
        controller_ori_quat = self._data.mocap_quat[0].copy().astype(np.float32)
        controller_ori_euler = np.zeros(3)
        if self.env_version == 0:
            mujoco.mju_quat2Vel(controller_ori_euler, controller_ori_quat, 1.0)
        else:
            controller_ori_scipy_quat = [controller_ori_quat[1], controller_ori_quat[2], controller_ori_quat[3], controller_ori_quat[0]]
            r = Rotation.from_quat(controller_ori_scipy_quat)
            controller_ori_euler = r.as_euler('xyz', degrees=False)

        obs["state"]["controller_pose"] = np.concatenate(
            (controller_pos, controller_ori_euler)
        ).astype(np.float32)

        tcp_pos = self._data.sensor("hande/pinch_pos").data.astype(np.float32)
        tcp_ori_quat = self._data.sensor("hande/pinch_quat").data.astype(np.float32)
        tcp_ori_euler = np.zeros(3)
        if self.env_version == 0:
            mujoco.mju_quat2Vel(tcp_ori_euler, tcp_ori_quat, 1.0)
        else:
            tcp_ori_scipy_quat = [tcp_ori_quat[1], tcp_ori_quat[2], tcp_ori_quat[3], tcp_ori_quat[0]]
            r = Rotation.from_quat(tcp_ori_scipy_quat)
            tcp_ori_euler = r.as_euler('xyz', degrees=False)
        obs["state"]["ur5e/tcp_pose"] = np.concatenate((tcp_pos, tcp_ori_euler)).astype(np.float32)

        tcp_vel = self._data.sensor("hande/pinch_vel").data
        obs["state"]["ur5e/tcp_vel"] = tcp_vel.astype(np.float32)

        joint_pos = np.stack(
            [self._data.sensor(f"ur5e/joint{i}_pos").data for i in range(1, 7)]
        ).ravel()
        obs["state"]["ur5e/joint_pos"] = joint_pos.astype(np.float32)

        joint_vel = np.stack(
            [self._data.sensor(f"ur5e/joint{i}_vel").data for i in range(1, 7)]
        ).ravel()
        obs["state"]["ur5e/joint_vel"] = joint_vel.astype(np.float32)

        bodyid = self._model.site_bodyid[self._attatchment_id]
        total_mass = self._model.body_subtreemass[bodyid]
        force_gravity = self._model.opt.gravity * total_mass
        attachment_force = self._data.sensor("ur5e/wrist_force").data
        attachment_force = np.dot(self._data.site_xmat[self._attatchment_id].reshape(3, 3), attachment_force)
        attachment_torque = self._data.sensor("ur5e/wrist_torque").data
        attachment_torque = np.dot(self._data.site_xmat[self._attatchment_id].reshape(3, 3), attachment_torque)
        wrist_force = attachment_force + force_gravity
        obs["state"]["ur5e/wrist_force"] = wrist_force.astype(np.float32)

        site_id = self._model.site("bottom_pinch").id
        body_id = self._model.site_bodyid[site_id]
        site_pos_world = np.array(self._data.site_xpos[site_id])
        torque_site_pos_world = np.zeros(3)

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1_body = self._model.geom_bodyid[contact.geom1]
            geom2_body = self._model.geom_bodyid[contact.geom2]
            if body_id not in [geom1_body, geom2_body]:
                continue
            contact_point_world = np.array(contact.pos)
            r_i = contact_point_world - site_pos_world
            efc_address = contact.efc_address
            if efc_address != -1:
                contact_force_contact_frame = self._data.efc_force[efc_address:efc_address+3]
                contact_frame = np.array(contact.frame).reshape(3, 3)
                contact_force_world = np.dot(contact_frame, contact_force_contact_frame)
                torque_i = np.cross(r_i, contact_force_world)
                torque_site_pos_world += torque_i

        obs["state"]["ur5e/wrist_torque"] = torque_site_pos_world.astype(np.float32)

        def visualize_contact_forces(model, data, renderer, scale: float = 0.01) -> None:
            """
            Visualize contact forces in the Mujoco environment.

            Parameters:
                model: The Mujoco model.
                data: The Mujoco data.
                renderer: The MujocoRenderer instance.
                scale (float): Scaling factor for the force vector.
            """
            for i in range(data.ncon):
                contact = data.contact[i]
                efc_address = contact.efc_address
                if efc_address == -1:
                    continue
                contact_point = np.array(contact.pos)
                contact_force_contact_frame = data.efc_force[efc_address:efc_address + 3]
                contact_frame = np.array(contact.frame).reshape(3, 3)
                contact_force_world = np.dot(contact_frame, contact_force_contact_frame)
                force_vector = scale * contact_force_world
                force_magnitude = np.linalg.norm(force_vector)
                if force_magnitude > 0:
                    force_direction = force_vector / force_magnitude
                    z_axis = np.array([0, 0, 1])
                    rotation_axis = np.cross(z_axis, force_direction)
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        angle = np.arccos(np.dot(z_axis, force_direction))
                        quat = np.zeros(4)
                        mujoco.mju_axisAngle2Quat(quat, rotation_axis, angle)
                        rotation_matrix = np.zeros(9)
                        mujoco.mju_quat2Mat(rotation_matrix, quat)
                    else:
                        rotation_matrix = np.eye(3).flatten()
                    renderer.add_marker(
                        pos=contact_point,
                        mat=rotation_matrix,
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.001, 0.001, force_magnitude],
                        rgba=[1, 0, 0, 1],
                        label="",
                    )

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)
            if self.enable_force_feedback:
                self.force_feedback.root.update()
                self.force_feedback.update_xy_pos(tcp_pos[0], tcp_pos[1])
                self.force_feedback.update_z_pos(tcp_pos[2])
                self.force_feedback.update_force_rectangles(wrist_force, 100)
                self.force_feedback.update_torque_rectangles(torque_site_pos_world, 0.2)
            if self.enable_slider_controller:
                self.slider_controller.root.update_idletasks()
                self.slider_controller.root.update()
            if self.enable_force_visualization:
                visualize_contact_forces(self._model, self._data, self._viewer.viewer)

        connector_pos = self._data.sensor("connector_head_pos").data.astype(np.float32)
        connector_ori_quat = self._data.sensor("connector_head_quat").data.astype(np.float32)
        connector_ori_euler = np.zeros(3)
        if self.env_version == 0:
            mujoco.mju_quat2Vel(connector_ori_euler, connector_ori_quat, 1.0)
        else:
            scipy_connector_ori_quat = [connector_ori_quat[1], connector_ori_quat[2], connector_ori_quat[3], connector_ori_quat[0]]
            r = Rotation.from_quat(scipy_connector_ori_quat)
            connector_ori_euler = r.as_euler('xyz', degrees=False)
        connector_pos = np.array([4.0300e-01, 1.0000e-03, 6.5031e-02])
        obs["state"]["connector_pose"] = np.concatenate((connector_pos, connector_ori_euler)).astype(np.float32)

        port_pos = self._data.sensor("port_bottom_pos").data.astype(np.float32)
        port_ori_quat = self._data.sensor("port_bottom_quat").data.astype(np.float32)
        port_ori_euler = np.zeros(3)
        if self.env_version == 0:
            mujoco.mju_quat2Vel(port_ori_euler, port_ori_quat, 1.0)
        else:
            scipy_port_ori_quat = [port_ori_quat[1], port_ori_quat[2], port_ori_quat[3], port_ori_quat[0]]
            r = Rotation.from_quat(scipy_port_ori_quat)
            port_ori_euler = r.as_euler('xyz', degrees=False)
        obs["state"]["port_pose"] = np.concatenate((port_pos, port_ori_euler)).astype(np.float32)
        return obs

    def _compute_reward(self) -> Tuple[float, bool]:
        """
        Compute and return the reward and task completion flag.

        Returns:
            A tuple containing:
              - reward (float): The computed reward.
              - task_complete (bool): True if the task is completed, False otherwise.
        """
        sensor_data = self._data.sensor
        connector_head_pos = sensor_data("connector_head_pos").data
        connector_head_ori = sensor_data("connector_head_quat").data
        tcp_pos = sensor_data("hande/pinch_pos").data
        connector_bottom_pos = sensor_data("connector_bottom_pos").data
        port_bottom_pos = sensor_data("port_bottom_pos").data
        port_bottom_quat = sensor_data("port_bottom_quat").data
        distance = np.linalg.norm(connector_bottom_pos - port_bottom_pos)
        z_distance = abs(connector_bottom_pos[2] - port_bottom_pos[2])

        mujoco.mju_negQuat(self.quat_conj, connector_head_ori)
        mujoco.mju_mulQuat(self.quat_err, port_bottom_quat, self.quat_conj)
        if self.env_version == 0:
            mujoco.mju_quat2Vel(self.ori_err, self.quat_err, 1.0)
        else:
            scipy_quat_err = [self.quat_err[1], self.quat_err[2], self.quat_err[3], self.quat_err[0]]
            r = Rotation.from_quat(scipy_quat_err)
            self.ori_err = r.as_euler('xyz', degrees=False)
        distance += 0 * np.linalg.norm(self.ori_err)
        task_complete = distance < self.reward_config["task_complete_tolerance"]

        dense_weights = self.reward_config["dense_reward_weights"]
        reward_components = {
            "box_target": lambda: max(1 - distance, 0),
        }
        self.dense_reward = sum(
            dense_weights[component] * reward_components[component]()
            for component in dense_weights
            if component in reward_components
        )
        self.sparse_reward = self.reward_config["sparse_reward_weights"] if task_complete else -1
        if not self.reward_config["reward_shaping"]:
            return self.sparse_reward, task_complete
        return self.dense_reward, task_complete


if __name__ == "__main__":
    env = ur5ePegInHoleGymEnv()
    env.reset()
    for i in range(1000):
        env.step(np.random.uniform(-1, 1, 7))
        env.render()
    env.close()
