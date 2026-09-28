"""Shared G1|T2 AMP motion playback helpers (no GLFW / display)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_to_t2.adapter import JointAdapter
from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION_DIR = ROOT / "data" / "datasets" / "goalkeeper"
DEFAULT_G1_XML = ROOT / "refs" / "G1_29dof" / "scene.xml"
DEFAULT_T2_XML = ROOT / "refs" / "T2_31dof" / "scene.xml"
DEFAULT_MAP = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"

AMP_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

MOTION_FILES = [
    "lefthand.pt",
    "leftjump.pt",
    "leftstep.pt",
    "righthand.pt",
    "rightjump.pt",
    "rightstep.pt",
]


def amp21_to_g1_29(q21: np.ndarray) -> np.ndarray:
    q21 = np.asarray(q21, dtype=np.float64)
    out = np.zeros(q21.shape[:-1] + (29,), dtype=np.float64)
    amp_index = {n: i for i, n in enumerate(AMP_JOINT_NAMES)}
    for gi, name in enumerate(G1_JOINT_NAMES):
        if name in amp_index:
            out[..., gi] = q21[..., amp_index[name]]
    return out


def _joint_qpos_addrs(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not in model: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int64)


def _foot_geom_ids(model: mujoco.MjModel, body_names: tuple[str, ...]) -> list[int]:
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names
    }
    return [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in body_ids]


def _mesh_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]) -> float:
    return min(
        float(data.geom_xpos[g, 2]) - float(model.geom_rbound[g]) for g in geom_ids
    )


class RobotPlayer:
    def __init__(
        self,
        xml: Path,
        joint_names: list[str],
        foot_bodies: tuple[str, ...],
        label: str,
        width: int,
        height: int,
    ) -> None:
        self.label = label
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = _joint_qpos_addrs(self.model, joint_names)
        self.foot_geoms = _foot_geom_ids(self.model, foot_bodies)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.lookat[:] = np.array([0.0, 0.0, 0.55])
        self.cam.distance = 2.7
        self.cam.azimuth = 140.0
        self.cam.elevation = -15.0

    def set_pose(
        self,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        q: np.ndarray,
        z_shift: float = 0.0,
    ) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = base_pos
        self.data.qpos[2] += z_shift
        x, y, z, w = base_quat_xyzw
        self.data.qpos[3:7] = [w, x, y, z]
        self.data.qpos[self.qpos_addrs] = q
        mujoco.mj_forward(self.model, self.data)

    def foot_bottom(self) -> float:
        return _mesh_bottom(self.model, self.data, self.foot_geoms)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()
        # Release EGL context so a second Renderer can make_current (RDP/EGL).
        ctx = getattr(self.renderer, "_gl_context", None)
        if ctx is not None and hasattr(ctx, "_display"):
            try:
                from OpenGL import EGL

                EGL.eglMakeCurrent(
                    ctx._display,
                    EGL.EGL_NO_SURFACE,
                    EGL.EGL_NO_SURFACE,
                    EGL.EGL_NO_CONTEXT,
                )
            except Exception:
                pass
        return img


class MotionBank:
    def __init__(
        self,
        motion_dir: Path,
        adapter: JointAdapter,
        margin: float = 0.03,
        *,
        arm_ik: bool = True,
    ):
        self.adapter = adapter
        self.margin = margin
        self.names: list[str] = []
        self.clips: list[dict] = []
        ik = None
        cache_dir = ROOT / "refs" / "motion_previews" / "arm_ik_cache"
        if arm_ik:
            from g1_to_t2.arm_ik import ArmIKRetargeter

            print("Arm IK retargeter loading…", flush=True, file=__import__("sys").stderr)
            ik = ArmIKRetargeter()
            cache_dir.mkdir(parents=True, exist_ok=True)

        for fname in MOTION_FILES:
            path = motion_dir / fname
            if not path.exists():
                continue
            raw = torch.load(path, map_location="cpu", weights_only=False)
            q21 = raw["joint_position"].numpy().astype(np.float64)
            q_g1 = amp21_to_g1_29(q21)
            q_t2 = adapter.unpack_q_g1_to_t2(q_g1)
            base_pos = raw["base_position"].numpy().astype(np.float64)
            base_quat = raw["base_pose"].numpy().astype(np.float64)
            if ik is not None:
                cache = cache_dir / f"{path.stem}_q_t2.npy"
                # Invalidate if seed map changed: store hash of first/last seed frame.
                seed_sig = (
                    b"arm_ik_v5_smooth|"
                    + np.concatenate([q_t2[0], q_t2[-1]]).tobytes()
                )
                sig_path = cache_dir / f"{path.stem}_sig.bin"
                use_cache = (
                    cache.exists()
                    and sig_path.exists()
                    and sig_path.read_bytes() == seed_sig
                )
                if use_cache:
                    print(
                        f"  arm-IK cache hit: {path.stem}",
                        flush=True,
                        file=__import__("sys").stderr,
                    )
                    q_t2 = np.load(cache)
                else:
                    print(
                        f"  arm-IK solving: {path.stem} ({len(q_g1)} fr)…",
                        flush=True,
                        file=__import__("sys").stderr,
                    )
                    q_t2 = ik.retarget_sequence(q_g1, q_t2, base_pos, base_quat)
                    np.save(cache, q_t2)
                    sig_path.write_bytes(seed_sig)
            self.names.append(path.stem)
            self.clips.append(
                {
                    "q_g1": q_g1,
                    "q_t2": q_t2,
                    "base_pos": base_pos,
                    "base_quat": base_quat,
                    "T": len(q_g1),
                }
            )
        if not self.clips:
            raise SystemExit(f"No motion .pt files in {motion_dir}")

    def __len__(self) -> int:
        return len(self.clips)

    def compute_z_shift(self, player: RobotPlayer, clip: dict, which: str) -> float:
        q_key = "q_g1" if which == "g1" else "q_t2"
        bottoms = []
        for t in range(clip["T"]):
            player.set_pose(clip["base_pos"][t], clip["base_quat"][t], clip[q_key][t], 0.0)
            bottoms.append(player.foot_bottom())
        return float(self.margin - float(np.min(bottoms)))


__all__ = [
    "AMP_JOINT_NAMES",
    "DEFAULT_G1_XML",
    "DEFAULT_MAP",
    "DEFAULT_MOTION_DIR",
    "DEFAULT_T2_XML",
    "G1_JOINT_NAMES",
    "MOTION_FILES",
    "MotionBank",
    "RobotPlayer",
    "ROOT",
    "T2_JOINT_NAMES",
    "amp21_to_g1_29",
]
