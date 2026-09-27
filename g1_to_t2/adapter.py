"""Pack / unpack joint vectors between G1 (29) and T2 (31)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES


@dataclass(frozen=True)
class MapEntry:
    g1: str
    t2: str
    sign: float
    scale: float
    status: str


@dataclass(frozen=True)
class HoldEntry:
    t2: str
    hold_value: float
    status: str


class JointAdapter:
    """Name-based remapper between G1 and T2 joint vectors."""

    def __init__(
        self,
        entries: list[MapEntry],
        holds: list[HoldEntry],
        g1_names: list[str] | None = None,
        t2_names: list[str] | None = None,
    ) -> None:
        self.g1_names = list(g1_names or G1_JOINT_NAMES)
        self.t2_names = list(t2_names or T2_JOINT_NAMES)
        self.g1_index = {n: i for i, n in enumerate(self.g1_names)}
        self.t2_index = {n: i for i, n in enumerate(self.t2_names)}
        self.entries = entries
        self.holds = holds
        self._validate()

        self._g1_idx = np.array([self.g1_index[e.g1] for e in entries], dtype=np.int64)
        self._t2_idx = np.array([self.t2_index[e.t2] for e in entries], dtype=np.int64)
        self._factors = np.array([e.sign * e.scale for e in entries], dtype=np.float64)
        self._hold_t2_idx = np.array([self.t2_index[h.t2] for h in holds], dtype=np.int64)
        self._hold_vals = np.array([h.hold_value for h in holds], dtype=np.float64)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JointAdapter":
        data = yaml.safe_load(Path(path).read_text())
        entries = [
            MapEntry(
                g1=e["g1"],
                t2=e["t2"],
                sign=float(e.get("sign", 1.0)),
                scale=float(e.get("scale", 1.0)),
                status=str(e.get("status", "exact")),
            )
            for e in data["entries"]
        ]
        holds = [
            HoldEntry(
                t2=h["t2"],
                hold_value=float(h.get("hold_value", 0.0)),
                status=str(h.get("status", "drop")),
            )
            for h in data.get("t2_hold", [])
        ]
        return cls(entries=entries, holds=holds)

    def _validate(self) -> None:
        if len(self.entries) != len(self.g1_names):
            raise ValueError(
                f"Expected {len(self.g1_names)} map entries (one per G1 joint), "
                f"got {len(self.entries)}"
            )
        mapped_g1 = {e.g1 for e in self.entries}
        mapped_t2 = {e.t2 for e in self.entries}
        hold_t2 = {h.t2 for h in self.holds}

        missing_g1 = set(self.g1_names) - mapped_g1
        if missing_g1:
            raise ValueError(f"G1 joints missing from map: {sorted(missing_g1)}")

        extra_g1 = mapped_g1 - set(self.g1_names)
        if extra_g1:
            raise ValueError(f"Unknown G1 joints in map: {sorted(extra_g1)}")

        covered_t2 = mapped_t2 | hold_t2
        missing_t2 = set(self.t2_names) - covered_t2
        if missing_t2:
            raise ValueError(f"T2 joints missing from map/hold: {sorted(missing_t2)}")

        overlap = mapped_t2 & hold_t2
        if overlap:
            raise ValueError(f"T2 joints both mapped and held: {sorted(overlap)}")

        if len(mapped_t2) != len(self.entries):
            raise ValueError("Duplicate T2 targets in map entries")

    @property
    def g1_dof(self) -> int:
        return len(self.g1_names)

    @property
    def t2_dof(self) -> int:
        return len(self.t2_names)

    def pack_q_t2_to_g1(self, q_t2: np.ndarray) -> np.ndarray:
        """Map T2 joint positions/velocities/actions → G1 order."""
        q_t2 = np.asarray(q_t2, dtype=np.float64)
        if q_t2.shape[-1] != self.t2_dof:
            raise ValueError(f"Expected last dim {self.t2_dof}, got {q_t2.shape}")
        out = np.zeros(q_t2.shape[:-1] + (self.g1_dof,), dtype=np.float64)
        vals = q_t2[..., self._t2_idx] * self._factors
        out[..., self._g1_idx] = vals
        return out

    def unpack_q_g1_to_t2(self, q_g1: np.ndarray) -> np.ndarray:
        """Map G1 joint vector → T2 order; held joints set to hold_value."""
        q_g1 = np.asarray(q_g1, dtype=np.float64)
        if q_g1.shape[-1] != self.g1_dof:
            raise ValueError(f"Expected last dim {self.g1_dof}, got {q_g1.shape}")
        out = np.zeros(q_g1.shape[:-1] + (self.t2_dof,), dtype=np.float64)
        vals = q_g1[..., self._g1_idx] * self._factors
        out[..., self._t2_idx] = vals
        if self._hold_t2_idx.size:
            out[..., self._hold_t2_idx] = self._hold_vals
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "g1_dof": self.g1_dof,
            "t2_dof": self.t2_dof,
            "exact": sum(1 for e in self.entries if e.status == "exact"),
            "approx": sum(1 for e in self.entries if e.status == "approx"),
            "held": len(self.holds),
        }
