"""Per-frame T2 arm IK: match G1 wrist direction + elbow crease angle."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES

_ROOT = Path(__file__).resolve().parents[1]
_G1_XML = _ROOT / "refs" / "G1_29dof" / "scene.xml"
_T2_XML = _ROOT / "refs" / "T2_31dof" / "scene.xml"

_ARM = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
    ],
}


def _jadr(model: mujoco.MjModel, names: list[str]) -> dict[str, int]:
    return {
        n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in names
    }


def _bid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _crease(sh: np.ndarray, el: np.ndarray, wr: np.ndarray) -> float:
    u = el - sh
    f = wr - el
    nu = float(np.linalg.norm(u)) + 1e-9
    nf = float(np.linalg.norm(f)) + 1e-9
    c = float(np.clip((u / nu) @ (f / nf), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


class ArmIKRetargeter:
    def __init__(self) -> None:
        self.m1 = mujoco.MjModel.from_xml_path(str(_G1_XML))
        self.m2 = mujoco.MjModel.from_xml_path(str(_T2_XML))
        self.d1 = mujoco.MjData(self.m1)
        self.d2 = mujoco.MjData(self.m2)
        self.g1a = _jadr(self.m1, G1_JOINT_NAMES)
        self.t2a = _jadr(self.m2, T2_JOINT_NAMES)
        self.t2i = {n: i for i, n in enumerate(T2_JOINT_NAMES)}

        self.g1_sh = {
            "left": _bid(self.m1, "left_shoulder_roll_link"),
            "right": _bid(self.m1, "right_shoulder_roll_link"),
        }
        self.g1_el = {
            "left": _bid(self.m1, "left_elbow_link"),
            "right": _bid(self.m1, "right_elbow_link"),
        }
        self.g1_wr = {
            "left": _bid(self.m1, "left_wrist_yaw_link"),
            "right": _bid(self.m1, "right_wrist_yaw_link"),
        }
        self.t2_sh = {
            "left": _bid(self.m2, "left_shoulder_roll_link"),
            "right": _bid(self.m2, "right_shoulder_roll_link"),
        }
        self.t2_el = {
            "left": _bid(self.m2, "left_elbow_yaw_link"),
            "right": _bid(self.m2, "right_elbow_yaw_link"),
        }
        self.t2_wr = {
            "left": _bid(self.m2, "left_wrist_yaw_link"),
            "right": _bid(self.m2, "right_wrist_yaw_link"),
        }

        q = np.zeros(len(T2_JOINT_NAMES))
        q[self.t2i["left_shoulder_roll_joint"]] = -1.3
        q[self.t2i["right_shoulder_roll_joint"]] = 1.3
        self._setq(
            self.m2, self.d2, self.t2a, np.zeros(3), np.array([0, 0, 0, 1.0]), T2_JOINT_NAMES, q
        )
        self.t2_wlen = {
            s: float(
                np.linalg.norm(self.d2.xpos[self.t2_wr[s]] - self.d2.xpos[self.t2_sh[s]])
            )
            for s in ("left", "right")
        }

        self._lims = {}
        for side, names in _ARM.items():
            lo, hi = [], []
            for n in names:
                jid = mujoco.mj_name2id(self.m2, mujoco.mjtObj.mjOBJ_JOINT, n)
                lo.append(float(self.m2.jnt_range[jid, 0]))
                hi.append(float(self.m2.jnt_range[jid, 1]))
            self._lims[side] = (np.asarray(lo), np.asarray(hi))

    @staticmethod
    def _setq(model, data, adr, bp, bq_xyzw, names, q) -> None:
        data.qpos[:] = 0.0
        data.qpos[0:3] = bp
        x, y, z, w = bq_xyzw
        data.qpos[3:7] = [w, x, y, z]
        for n, v in zip(names, q):
            data.qpos[adr[n]] = v
        mujoco.mj_forward(model, data)

    def _err(
        self,
        side: str,
        bp: np.ndarray,
        bq: np.ndarray,
        q: np.ndarray,
        dir_w: np.ndarray,
        crease_deg: float,
    ) -> np.ndarray:
        self._setq(self.m2, self.d2, self.t2a, bp, bq, T2_JOINT_NAMES, q)
        sh = self.d2.xpos[self.t2_sh[side]]
        el = self.d2.xpos[self.t2_el[side]]
        wr = self.d2.xpos[self.t2_wr[side]]
        tgt_w = sh + dir_w * self.t2_wlen[side]
        # wrist position error (3) + crease error mapped to 1D scaled like meters
        c = _crease(sh, el, wr)
        # 1 deg ~ 0.004 m weight → 40 deg mismatch ~ 0.16 m equivalent
        return np.concatenate([wr - tgt_w, np.array([(c - crease_deg) * 0.006])])

    def _ik_side(
        self,
        side: str,
        q: np.ndarray,
        q_prev: np.ndarray,
        bp: np.ndarray,
        bq: np.ndarray,
        dir_w: np.ndarray,
        crease_deg: float,
        iters: int = 50,
        w_vel: float = 0.12,
    ) -> np.ndarray:
        names = _ARM[side]
        idx = [self.t2i[n] for n in names]
        lo, hi = self._lims[side]
        q_ref = np.array([q_prev[i] for i in idx], dtype=np.float64)
        for _ in range(iters):
            pos_err = self._err(side, bp, bq, q, dir_w, crease_deg)
            q_arm = np.array([q[i] for i in idx], dtype=np.float64)
            vel_err = w_vel * (q_arm - q_ref)
            err = np.concatenate([pos_err, vel_err])
            if float(np.linalg.norm(pos_err)) < 5e-4:
                break
            J = np.zeros((err.shape[0], len(names)))
            eps = 1e-4
            for k, ii in enumerate(idx):
                q2 = q.copy()
                q2[ii] += eps
                e2 = np.concatenate(
                    [
                        self._err(side, bp, bq, q2, dir_w, crease_deg),
                        w_vel
                        * (
                            np.array([q2[j] for j in idx], dtype=np.float64) - q_ref
                        ),
                    ]
                )
                J[:, k] = (e2 - err) / eps
            dq = -J.T @ np.linalg.solve(J @ J.T + 3e-2 * np.eye(err.shape[0]), err)
            # Keep frame-to-frame steps small to avoid shoulder flicker.
            dq = np.clip(dq, -0.12, 0.12)
            for k, ii in enumerate(idx):
                q[ii] = float(np.clip(q[ii] + dq[k], lo[k], hi[k]))
        # Hard clamp again (L/R elbow_yaw have opposite ranges).
        for k, ii in enumerate(idx):
            q[ii] = float(np.clip(q[ii], lo[k], hi[k]))
        return q

    def retarget_sequence(
        self,
        q_g1: np.ndarray,
        q_t2_seed: np.ndarray,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        *,
        iters: int = 45,
        smooth_sigma: float = 2.8,
    ) -> np.ndarray:
        q_g1 = np.asarray(q_g1, dtype=np.float64)
        out = np.asarray(q_t2_seed, dtype=np.float64).copy()
        T = len(q_g1)
        prev = out[0].copy()
        prev[self.t2i["left_shoulder_roll_joint"]] = -1.0
        prev[self.t2i["right_shoulder_roll_joint"]] = 1.0
        # Elbow yaw ranges are one-sided (L≤0, R≥0); seed inside the range.
        prev[self.t2i["left_elbow_yaw_joint"]] = -0.4
        prev[self.t2i["right_elbow_yaw_joint"]] = 0.4
        prev[self.t2i["left_elbow_pitch_joint"]] = -0.3
        prev[self.t2i["right_elbow_pitch_joint"]] = 0.3

        for t in range(T):
            bp = base_pos[t]
            bq = base_quat_xyzw[t]
            self._setq(self.m1, self.d1, self.g1a, bp, bq, G1_JOINT_NAMES, q_g1[t])
            q = prev.copy()
            for n in T2_JOINT_NAMES:
                if n not in _ARM["left"] and n not in _ARM["right"]:
                    q[self.t2i[n]] = out[t, self.t2i[n]]
            for side in ("left", "right"):
                sh = self.d1.xpos[self.g1_sh[side]]
                el = self.d1.xpos[self.g1_el[side]]
                wr = self.d1.xpos[self.g1_wr[side]]
                dw = wr - sh
                nw = float(np.linalg.norm(dw)) + 1e-9
                crease = _crease(sh, el, wr)
                q = self._ik_side(
                    side, q, prev, bp, bq, dw / nw, crease, iters=iters
                )
            # Cap total frame delta vs previous (kills residual flicker).
            for side in ("left", "right"):
                for n in _ARM[side]:
                    i = self.t2i[n]
                    lo, hi = self._lims[side]
                    j = _ARM[side].index(n)
                    q[i] = float(
                        np.clip(
                            np.clip(q[i], prev[i] - 0.20, prev[i] + 0.20),
                            lo[j],
                            hi[j],
                        )
                    )
            out[t] = q
            prev = q
            if (t + 1) % 50 == 0 or t == T - 1:
                print(f"    arm-IK {t+1}/{T}", flush=True, file=sys.stderr)

        # Temporal smooth on arm DoFs only.
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            gaussian_filter1d = None
        if gaussian_filter1d is not None and smooth_sigma > 0:
            for side in ("left", "right"):
                lo, hi = self._lims[side]
                for j, n in enumerate(_ARM[side]):
                    i = self.t2i[n]
                    sm = gaussian_filter1d(out[:, i], sigma=smooth_sigma, mode="nearest")
                    out[:, i] = np.clip(sm, lo[j], hi[j])
        return out
