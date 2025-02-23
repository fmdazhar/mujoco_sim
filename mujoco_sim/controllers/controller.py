from typing import Optional, Tuple, Union
import mujoco
import numpy as np

class Controller:
    def __init__(
        self,
        model,
        data,
        site_id,
        dof_ids: np.ndarray,
        config: dict,
    ):

        self.model = model
        self.data = data
        self.site_id = site_id
        self.integration_dt = config.get("integration_dt", model.opt.timestep)
        self.dof_ids = dof_ids
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        # Set parameters from the config dictionary
        self.trans_damping_ratio = config.get("trans_damping_ratio", 0.996)
        self.rot_damping_ratio = config.get("rot_damping_ratio", 0.286)
        self.error_tolerance_pos = config.get("error_tolerance_pos", 0.001)
        self.error_tolerance_ori = config.get("error_tolerance_ori", 0.001)
        self.method = config.get("method", "dynamics")
        self.inertia_compensation = config.get("inertia_compensation", False)
        self.pos_gains = config.get("pos_gains", (100, 100, 100))
        self.ori_gains = config.get(
            "ori_gains", tuple(gain * 1 / 8 for gain in self.pos_gains)
        )
        self.pos_kd = config.get("pos_kd", None)
        self.ori_kd = config.get("ori_kd", None)
        self.max_angvel = config.get("max_angvel", 4)
        self.gravity_compensation = config.get("gravity_compensation", True)
        self.trans_clip_min = config.get("trans_clip_min", np.array([-0.01, -0.01, -0.01]))
        self.trans_clip_max = config.get("trans_clip_max", np.array([0.01,  0.01,  0.01]))
        self.rot_clip_min = config.get("rot_clip_min", np.array([-0.05, -0.05, -0.05]))
        self.rot_clip_max = config.get("rot_clip_max", np.array([ 0.05,  0.05,  0.05]))

        # Preallocate memory for commonly used variables
        self.quat = np.zeros(4)
        self.quat_des = np.zeros(4)
        self.quat_conj = np.zeros(4)
        self.quat_err = np.zeros(4)
        self.ori_err = np.zeros(3)
        self.x_err_norm = 0.0
        self.ori_err_norm = 0.0
        self.J_v = np.zeros((3, model.nv), dtype=np.float64)
        self.J_w = np.zeros((3, model.nv), dtype=np.float64)
        self.M = np.zeros((model.nv, model.nv), dtype=np.float64)
        self.error = np.zeros(6)

    def set_parameters(
        self,
        trans_damping_ratio: Optional[float] = None,
        rot_damping_ratio: Optional[float] = None,
        error_tolerance_pos: Optional[float] = None,
        error_tolerance_ori: Optional[float] = None,
        trans_clip_min: Optional[np.ndarray] = None,
        trans_clip_max: Optional[np.ndarray] = None,
        rot_clip_min: Optional[np.ndarray] = None,
        rot_clip_max: Optional[np.ndarray] = None,
        max_angvel: Optional[float] = None,
        pos_gains: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        pos_kd: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        ori_kd: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        method: Optional[str] = None,
        inertia_compensation: Optional[bool] = None,
    ):
        if trans_damping_ratio is not None:
            self.trans_damping_ratio = trans_damping_ratio
        if rot_damping_ratio is not None:
            self.rot_damping_ratio = rot_damping_ratio
        if error_tolerance_pos is not None:
            self.error_tolerance_pos = error_tolerance_pos
        if error_tolerance_ori is not None:
            self.error_tolerance_ori = error_tolerance_ori
        if trans_clip_min is not None:
            self.trans_clip_min = np.asarray(trans_clip_min)
        if trans_clip_max is not None:
            self.trans_clip_max = np.asarray(trans_clip_max)
        if rot_clip_min is not None:
            self.rot_clip_min = np.asarray(rot_clip_min)
        if rot_clip_max is not None:
            self.rot_clip_max = np.asarray(rot_clip_max)
        if max_angvel is not None:
            self.max_angvel = max_angvel
        if pos_gains is not None:
            self.pos_gains = pos_gains
        if ori_gains is not None:
            self.ori_gains = ori_gains
        if pos_kd is not None:
            self.pos_kd = pos_kd
        if ori_kd is not None:
            self.ori_kd = ori_kd
        if method is not None:
            if method in ["dynamics", "pinv", "svd", "trans", "dls"]:
                self.method = method
            else:
                raise ValueError(
                    "Method must be one of 'dynamics', 'pinv', 'svd', 'trans', 'dls'"
                )
        if inertia_compensation is not None:
            self.inertia_compensation = inertia_compensation

    def compute_gains(
        self, gains, kd_values, method: str, damping_ratio: float
    ) -> np.ndarray:

        if method == "dynamics":
            kp = np.asarray(gains)
            if kd_values is None:
                kd = damping_ratio * 2 * np.sqrt(kp)
            else:
                kd = np.asarray(kd_values)

        else:
            kp = np.asarray(gains) / self.integration_dt
            if kd_values is None:
                kd = 0 * kp * self.integration_dt
            else:
                kd = np.asarray(kd_values)

        return np.stack([kp, kd], axis=-1)

    def control(
        self, pos: Optional[np.ndarray] = None, ori: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Desired position and orientation
        x_des = self.data.site_xpos[self.site_id] if pos is None else np.asarray(pos)
        if ori is None:
            mujoco.mju_mat2Quat(self.quat_des, self.data.site_xmat[self.site_id])
        else:
            self.quat_des[:] = np.asarray(ori)

        kp_kv_pos = self.compute_gains(
            self.pos_gains, self.pos_kd, self.method, self.trans_damping_ratio
        )
        kp_kv_ori = self.compute_gains(
            self.ori_gains, self.ori_kd, self.method, self.rot_damping_ratio
        )

        q = self.data.qpos[self.dof_ids]
        dq = self.data.qvel[self.dof_ids]

        mujoco.mj_jacSite(self.model, self.data, self.J_v, self.J_w, self.site_id)
        J_v = self.J_v[:, self.dof_ids]
        J_w = self.J_w[:, self.dof_ids]
        J = np.concatenate([J_v, J_w], axis=0)

        # Position Control
        x_err = self.data.site_xpos[self.site_id] - x_des
        dx_err = J_v @ dq

        self.x_err_norm = np.linalg.norm(x_err)
        x_err = np.clip(x_err, self.trans_clip_min, self.trans_clip_max)

        # Check positional error tolerance
        if self.x_err_norm < self.error_tolerance_pos:
            # If the error is within tolerance, set it to zero
            x_err = np.zeros_like(x_err)

        x_err *= -kp_kv_pos[:, 0]
        dx_err *= -kp_kv_pos[:, 1]

        ddx = x_err + dx_err

        # Orientation Control
        mujoco.mju_mat2Quat(self.quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.quat_conj, self.quat_des)
        mujoco.mju_mulQuat(self.quat_err, self.quat, self.quat_conj)
        mujoco.mju_quat2Vel(self.ori_err, self.quat_err, 1.0)
        w_err = J_w @ dq

        self.ori_err_norm = np.linalg.norm(self.ori_err)
        self.ori_err = np.clip(self.ori_err, self.rot_clip_min, self.rot_clip_max)

        # Check orientation error tolerance
        if self.ori_err_norm < self.error_tolerance_ori:
            # If the error is within tolerance, set it to zero
            self.ori_err = np.zeros_like(self.ori_err)

        self.ori_err *= -kp_kv_ori[:, 0]
        w_err *= -kp_kv_ori[:, 1]

        dw = self.ori_err + w_err

        self.error = np.concatenate([ddx, dw], axis=0)

        if self.method == "dynamics":
            if self.inertia_compensation:
                mujoco.mj_fullM(self.model, self.M, self.data.qM)
                M = self.M[self.dof_ids, :][:, self.dof_ids]
                M_inv = np.linalg.inv(M)
                ddq = M_inv @ J.T @ self.error
            else:
                ddq = J.T @ self.error  # Skip M_inv if inertia compensation is off

            dq += ddq * self.integration_dt

        elif self.method == "pinv":
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ self.error
            q += dq
        elif self.method == "svd":
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            S_inv = np.zeros_like(S)
            S_inv[S > 1e-5] = 1.0 / S[S > 1e-5]
            J_pinv = Vt.T @ np.diag(S_inv) @ U.T
            dq = J_pinv @ self.error
            q += dq
        elif self.method == "trans":
            dq = J.T @ self.error
            q += dq
        else:
            damping = 1e-4
            lambda_I = damping * np.eye(J.shape[0])
            dq = J.T @ np.linalg.inv(J @ J.T + lambda_I) @ self.error

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max
        q += dq * self.integration_dt

        q_min = self.model.actuator_ctrlrange[:6, 0]
        q_max = self.model.actuator_ctrlrange[:6, 1]
        np.clip(q, q_min, q_max, out=q)

        if self.gravity_compensation:
            self.data.qfrc_applied[:] = 0.0
            jac = np.empty((3, self.model.nv))
            subtreeid = 1
            total_mass = self.model.body_subtreemass[subtreeid]
            mujoco.mj_jacSubtreeCom(self.model, self.data, jac, subtreeid)
            self.data.qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac

        return q, dq
