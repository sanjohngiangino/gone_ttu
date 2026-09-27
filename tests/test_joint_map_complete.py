from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP_YAML = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"


def test_joint_counts():
    from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES

    assert len(G1_JOINT_NAMES) == 29
    assert len(T2_JOINT_NAMES) == 31
    assert len(set(G1_JOINT_NAMES)) == 29
    assert len(set(T2_JOINT_NAMES)) == 31


def test_map_covers_all_joints():
    from g1_to_t2.adapter import JointAdapter
    from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES

    adapter = JointAdapter.from_yaml(MAP_YAML)
    assert adapter.g1_dof == 29
    assert adapter.t2_dof == 31
    assert set(e.g1 for e in adapter.entries) == set(G1_JOINT_NAMES)
    covered = set(e.t2 for e in adapter.entries) | set(h.t2 for h in adapter.holds)
    assert covered == set(T2_JOINT_NAMES)
    summary = adapter.summary()
    assert summary["exact"] + summary["approx"] == 29
    assert summary["held"] == 2


def test_roundtrip_mapped_joints():
    from g1_to_t2.adapter import JointAdapter

    adapter = JointAdapter.from_yaml(MAP_YAML)
    rng = np.random.default_rng(0)
    q_g1 = rng.normal(size=29)
    q_t2 = adapter.unpack_q_g1_to_t2(q_g1)
    assert q_t2.shape == (31,)
    # head held at 0
    assert q_t2[0] == 0.0 and q_t2[1] == 0.0
    q_g1_back = adapter.pack_q_t2_to_g1(q_t2)
    np.testing.assert_allclose(q_g1_back, q_g1, rtol=1e-6, atol=1e-6)


def test_pack_shape_batch():
    from g1_to_t2.adapter import JointAdapter

    adapter = JointAdapter.from_yaml(MAP_YAML)
    q_t2 = np.zeros((4, 31))
    q_g1 = adapter.pack_q_t2_to_g1(q_t2)
    assert q_g1.shape == (4, 29)


def test_bad_dim_raises():
    from g1_to_t2.adapter import JointAdapter

    adapter = JointAdapter.from_yaml(MAP_YAML)
    with pytest.raises(ValueError):
        adapter.pack_q_t2_to_g1(np.zeros(29))
    with pytest.raises(ValueError):
        adapter.unpack_q_g1_to_t2(np.zeros(31))
