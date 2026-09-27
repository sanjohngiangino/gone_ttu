# g1-to-t2

Transfer **Unitree G1** joint-PD policies (rsl_rl / Goalkeeper-style) onto **Booster T2** via joint remapping, then fine-tune for standing stability.

## What this is

1. **Adapter** — pack/unpack obs & actions between G1 (29 DoF) and T2 (31 DoF)
2. **Loader** — load `goalkeeper.pt`-style checkpoints (`actor`, `std`, history/ball/region encoders)
3. **Wrapper** — T2 low-state → G1 history obs → actor → T2 PD targets
4. **Smoke** — MuJoCo T2 rollouts
5. **Fine-tune** — BC distill from G1 teacher + short PPO-style stability training → native T2 31-DoF checkpoint

Arm topology is **approximate** (G1 `shoulder_yaw` → T2 `elbow_yaw`). Zero-shot skill transfer is a smoke test, not a finished product.

## Setup

```bash
cd spqr_zoff
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sim]"
```

Assets expected:

- `refs/goalkeeper.pt` — G1 policy
- `refs/T2_31dof/` — MJCF + meshes from [booster_assets](https://github.com/BoosterRobotics/booster_assets)

## Goalkeeper su mjlab (G1 prima)

Come [SoccerLab](https://github.com/Renforce-Dynamics/soccerLab) (Isaac → mjlab),
portiamo il portiere su mjlab **prima** del transfer T2.

- Doc: [`docs/MJLAB_GOALKEEPER_PORT.md`](docs/MJLAB_GOALKEEPER_PORT.md)
- Task: `SPQR-Mjlab-Goalkeeper-G1`
- Package: `source/goalkeeper_tasks_mjlab/`

```bash
pip install mjlab   # Linux + NVIDIA consigliato
pip install -e source/goalkeeper_tasks_mjlab
python scripts/factoryMjlab/play.py SPQR-Mjlab-Goalkeeper-G1 --agent zero
python scripts/factoryMjlab/train.py SPQR-Mjlab-Goalkeeper-G1 --num_envs 64 --max_iterations 50
```

I pesi Isaac `goalkeeper.pt` non vanno caricati zero-shot qui: si ri-allena nell’env mjlab.

---

## Side-by-side viewer (G1 | T2)

```bash
# Live window (G1 left, T2 right) — press q to quit
python scripts/view_side_by_side.py --mode window --seconds 10

# Save MP4
python scripts/view_side_by_side.py --mode video --seconds 8 --out refs/g1_t2_compare.mp4

# Two native MuJoCo windows
python scripts/view_side_by_side.py --mode dual --seconds 10
```

Left = **G1** with native Goalkeeper weights. Right = **T2** with the same checkpoint through the remapper.


```bash
# Inspect a G1 checkpoint
python scripts/dump_ckpt_shapes.py refs/goalkeeper.pt

# Calibrate native T2 stand (PD hold; must report stable=True)
python scripts/calibrate_signs.py

# Smoke: hold stand pose
python scripts/smoke_mujoco_t2.py --seconds 3 --hold-default --require-stand

# Smoke: remapped Goalkeeper on T2 (often falls zero-shot — expected)
python scripts/smoke_mujoco_t2.py --seconds 3

# Fine-tune native T2 stability policy (stand hold)
python scripts/finetune_stability.py

pytest -q
```

Zero-shot Goalkeeper→T2 is a **shape/runtime smoke test**, not a working keeper.
Stability comes from `finetune_stability.py` → `refs/t2_stability.pt`.

## Add any G1 joint-PD policy (checklist)

1. **Dump shapes** — `python scripts/dump_ckpt_shapes.py path/to/policy.pt`
   - Need `num_actions == 29` and preferably `std` + `actor.*`
2. **Confirm obs layout** — copy `configs/policies/goalkeeper_g1.yaml` → `configs/policies/<name>.yaml`
   - Match one-step dim: base ang vel / gravity / extras / `q` / `qd` / last action
   - Set `history_length`, `action_scale`, `g1_default_joint_angles`
3. **Joint map** — reuse `configs/maps/g1_29_to_t2_31.yaml` unless your G1 joint **order** differs (then edit map or `g1_to_t2/joints.py`)
4. **Signs** — if a joint flips, `python scripts/calibrate_signs.py --flip <t2_joint_name> --write-map configs/maps/g1_29_to_t2_31.yaml`
5. **Smoke** — `python scripts/smoke_mujoco_t2.py --ckpt path/to/policy.pt --policy configs/policies/<name>.yaml`
6. **Fine-tune** — `python scripts/finetune_stability.py --ckpt path/to/policy.pt --policy configs/policies/<name>.yaml`

Policies that are **not** joint-target PD (torque, latent, wrong DoF) are out of scope for v1.

## Layout

```
g1_to_t2/           # Python package
configs/maps/       # G1↔T2 joint map
configs/policies/   # per-policy obs/action YAML
scripts/            # dump / calibrate / smoke / finetune
tests/
refs/               # checkpoints + T2 MJCF
```

## Notes

- Head joints on T2 are held at 0.
- After fine-tune, prefer the native `refs/t2_stability.pt` (31-DoF) for deploy — do not keep remapping forever.
- Hardware deploy should reuse the same `JointAdapter` / obs packing once sim signs are trusted.
