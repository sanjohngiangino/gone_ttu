"""Minimal MuJoCo T2 helpers for smoke / fine-tune."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from g1_to_t2.joints import T2_JOINT_NAMES


@dataclass
class T2MujocoEnv:
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_qposadr: np.ndarray
    joint_dofadr: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    @classmethod
    def load(
        cls,
        xml_path: str | Path,
        kp: float = 180.0,
        kd: float = 4.0,
    ) -> "T2MujocoEnv":
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        qposadr = np.zeros(31, dtype=np.int32)
        dofadr = np.zeros(31, dtype=np.int32)
        for i, name in enumerate(T2_JOINT_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found in MJCF: {name}")
            qposadr[i] = model.jnt_qposadr[jid]
            dofadr[i] = model.jnt_dofadr[jid]
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != i:
                # still ok if order matches by name; verify actuator exists
                if aid < 0:
                    raise RuntimeError(f"Actuator not found: {name}")
        return cls(
            model=model,
            data=data,
            joint_qposadr=qposadr,
            joint_dofadr=dofadr,
            kp=np.full(31, kp, dtype=np.float64),
            kd=np.full(31, kd, dtype=np.float64),
        )

    def reset(
        self,
        q_t2: np.ndarray | None = None,
        height: float = 0.95,
        place_feet_on_ground: bool = True,
        foot_clearance: float = 0.002,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # freejoint: pos + quat
        self.data.qpos[0:3] = np.array([0.0, 0.0, height])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        if q_t2 is None:
            q_t2 = np.zeros(31)
        self.data.qpos[self.joint_qposadr] = q_t2
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if place_feet_on_ground:
            self._place_feet_on_ground(foot_clearance)

    def _place_feet_on_ground(self, clearance: float = 0.002) -> None:
        zs = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                zs.append(float(self.data.xpos[bid][2]))
        if zs:
            self.data.qpos[2] -= min(zs) - clearance
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def settle(self, target_q: np.ndarray, steps: int = 300) -> None:
        """Hold PD pose briefly so contacts settle before policy starts."""
        for _ in range(steps):
            self.step_pd(target_q, n_substeps=1)

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self.joint_qposadr].copy()

    def get_qd(self) -> np.ndarray:
        return self.data.qvel[self.joint_dofadr].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    def get_projected_gravity(self) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if bid < 0:
            bid = 1
        R = self.data.xmat[bid].reshape(3, 3)
        g_world = np.array([0.0, 0.0, -1.0])
        return R.T @ g_world

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=np.float64).copy()
        for i, name in enumerate(T2_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.model.jnt_range[jid]
            if self.model.jnt_limited[jid]:
                out[i] = np.clip(out[i], lo, hi)
        return out

    def apply_pd(self, target_q: np.ndarray) -> None:
        target_q = self.clip_q(target_q)
        q = self.get_q()
        qd = self.get_qd()
        tau = self.kp * (target_q - q) - self.kd * qd
        # respect motor ctrlrange
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(tau, lo, hi)

    def step_pd(self, target_q: np.ndarray, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            self.apply_pd(target_q)
            mujoco.mj_step(self.model, self.data)

    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def upright(self) -> float:
        # z-component of base up vector
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if bid < 0:
            bid = 1
        return float(self.data.xmat[bid].reshape(3, 3)[2, 2])
