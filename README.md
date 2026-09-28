# gone_ttu

Transfer **Unitree G1** joint-PD policies (rsl_rl / Goalkeeper-style) onto **Booster T2** via joint remapping, then fine-tune for standing stability.

Repo: [sanjohngiangino/gone_ttu](https://github.com/sanjohngiangino/gone_ttu)

## What this is

1. **Adapter** — pack/unpack obs & actions between G1 (29 DoF) and T2 (31 DoF)
2. **Arm IK retarget** — motion clips: seed map + per-frame IK (wrist direction + elbow crease)
3. **Loader** — load `goalkeeper.pt`-style checkpoints (`actor`, `std`, history/ball/region encoders)
4. **Wrapper** — T2 low-state → G1 history obs → actor → T2 PD targets
5. **Smoke / fine-tune** — MuJoCo T2 rollouts + short stability training

## Setup

```bash
cd gone_ttu
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sim]"
# viewer also needs: glfw, PyOpenGL, opencv-python, scipy
```

Assets expected:

- `refs/goalkeeper.pt` — G1 policy
- `refs/T2_31dof/` — MJCF + meshes from [booster_assets](https://github.com/BoosterRobotics/booster_assets)
- `refs/G1_29dof/` — G1 MJCF for dual viewer
- `data/datasets/goalkeeper/*.pt` — AMP region clips (21-DoF → G1 → T2)

## G1 → T2 motion retarget (viewer)

Seed joint map: `configs/maps/g1_29_to_t2_31.yaml`  
Arm refinement: `g1_to_t2/arm_ik.py` (cached under `refs/motion_previews/arm_ik_cache/`, gitignored)

```bash
# Native dual window on DISPLAY (default :1)
# Keys: 1–6 motion, Space pause, [ ] prev/next, LMB orbit, wheel zoom, C reset cam, Esc quit
scripts/factoryMjlab/run_retarget_gui.sh

# Or explicitly:
python -u scripts/factoryMjlab/view_retarget_motions.py --start 0

# Optional browser fallback
python -u scripts/factoryMjlab/view_retarget_web.py   # http://127.0.0.1:8090/
```

Details: [`TRANSFER_G1_TO_T2.md`](TRANSFER_G1_TO_T2.md)

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

## Policy transfer tools

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
2. **Confirm obs layout** — copy `configs/policies/goalkeeper_g1.yaml` → `configs/policies/<name>.yaml`
3. **Joint map** — reuse `configs/maps/g1_29_to_t2_31.yaml` unless G1 joint **order** differs
4. **Signs** — `python scripts/calibrate_signs.py --flip <t2_joint_name> --write-map configs/maps/g1_29_to_t2_31.yaml`
5. **Smoke** — `python scripts/smoke_mujoco_t2.py --ckpt path/to/policy.pt --policy configs/policies/<name>.yaml`
6. **Fine-tune** — `python scripts/finetune_stability.py --ckpt ... --policy ...`

## Layout

```
g1_to_t2/                 # adapter, joints, arm_ik
configs/maps/             # G1↔T2 joint map
configs/policies/         # per-policy obs/action YAML
scripts/factoryMjlab/     # retarget GUI + mjlab train/play
scripts/                  # dump / calibrate / smoke / finetune
source/goalkeeper_tasks_mjlab/
data/datasets/goalkeeper/ # AMP .pt clips
refs/                     # ckpts + G1/T2 MJCF
```

## Notes

- Head joints on T2 are held at 0 in the seed map.
- Arm IK cache regenerates on first GUI launch if missing.
- After fine-tune, prefer native `refs/t2_stability.pt` (31-DoF) for deploy.
