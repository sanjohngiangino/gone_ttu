from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAP_YAML = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"
POLICY_YAML = ROOT / "configs" / "policies" / "goalkeeper_g1.yaml"
CKPT = ROOT / "refs" / "goalkeeper.pt"


def test_checkpoint_shapes():
    from g1_to_t2.loader import inspect_checkpoint

    if not CKPT.exists():
        import pytest

        pytest.skip("refs/goalkeeper.pt not present")
    info = inspect_checkpoint(CKPT)
    assert info.num_actions == 29
    assert info.has_history_encoder
    assert info.history_in_dim == 960
    assert info.actor_in_dim == 119


def test_load_and_act_inference():
    from g1_to_t2.loader import load_rsl_actor

    if not CKPT.exists():
        import pytest

        pytest.skip("refs/goalkeeper.pt not present")
    actor, info = load_rsl_actor(CKPT, device="cpu")
    assert info.num_actions == 29
    hist = torch.zeros(1, 960)
    out = actor.act_inference(hist)
    assert out.shape == (1, 29)


def test_wrapper_t2_to_action():
    from g1_to_t2.policy import G1PolicyWrapper

    if not CKPT.exists():
        import pytest

        pytest.skip("refs/goalkeeper.pt not present")
    wrapper = G1PolicyWrapper.from_checkpoint(
        CKPT, MAP_YAML, POLICY_YAML, device="cpu"
    )
    q = np.zeros(31)
    qd = np.zeros(31)
    result = wrapper.act(q_t2=q, qd_t2=qd)
    assert result["action_g1"].shape == (29,)
    assert result["action_t2"].shape == (31,)
    assert result["target_q_t2"].shape == (31,)
    # head locked
    assert result["action_t2"][0] == 0.0
    assert result["action_t2"][1] == 0.0
