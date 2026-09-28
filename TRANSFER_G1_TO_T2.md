# G1 → T2 transfer (Goalkeeper motions + policy)

**Obiettivo:** portare motion / policy Unitree G1 (29 DoF) sul Booster T2 (31 DoF).  
**Checkpoint policy:** `refs/goalkeeper.pt`  
**Modelli:** `refs/G1_29dof/`, `refs/T2_31dof/` ([booster_assets](https://github.com/BoosterRobotics/booster_assets))  
**Clip AMP:** `data/datasets/goalkeeper/*.pt`

---

## Verdetto

| Path | Stato |
|------|--------|
| Load diretto `goalkeeper.pt` su T2 | **No** (29 vs 31 DoF, ordine, braccia) |
| Seed joint map + **arm IK** su clip AMP | **Sì** — viewer nativo OK |
| Zero-shot policy Goalkeeper su T2 | Solo smoke; aspetta cadute |
| Fine-tune stabilità T2 | `scripts/finetune_stability.py` |

---

## Morpho

| Parte | G1 | T2 | Note |
|-------|----|----|------|
| Gambe | 6+6 | 6+6 (`*_knee_pitch`) | Name-matched |
| Waist | yaw, roll, pitch | pitch, roll, yaw | Nomi ok, ordine diverso in MJCF |
| Braccia | sh P/R/**Y** → elbow → wrist | sh P/R → **elbow P/Y** → wrist | Topologia diversa |
| Testa | — | `aa_head_yaw`, `head_pitch` | Hold a 0 |

```
G1:  sh_pitch → sh_roll → sh_yaw → elbow → wrist…
T2:  sh_pitch → sh_roll → elbow_pitch → elbow_yaw → wrist…
```

Con le braccia T2 “appese” (`shoulder_roll ≈ ±1.3`), l’asse di `elbow_pitch` è quasi allineato al braccio (twist). La **piega visibile del gomito** è `elbow_yaw`.

---

## Seed map (`configs/maps/g1_29_to_t2_31.yaml`)

`q_t2 = sign * scale * q_g1 + offset` (`g1_to_t2/adapter.py`).

| G1 | T2 | sign | offset |
|----|----|------|--------|
| legs / waist / wrists | stessi nomi (knee→knee_pitch) | +1 | 0 |
| `*_shoulder_pitch` | stesso | +1 | 0 |
| `left_shoulder_roll` | stesso | +1 | **−1.30** |
| `right_shoulder_roll` | stesso | +1 | **+1.30** |
| `*_shoulder_yaw` | `*_elbow_pitch` | **−1** | 0 |
| `*_elbow` | `*_elbow_yaw` | **−1** | 0 |
| — | head yaw/pitch | hold 0 | — |

Il seed da solo non basta sulle braccia (lunghezze e assi diversi).

---

## Arm IK (`g1_to_t2/arm_ik.py`)

Dopo l’unpack del seed, per ogni frame:

1. Direzione G1 spalla→polso (e crease gomito) nel frame mondo
2. IK sui 4 DoF braccio T2 (`shoulder_pitch/roll`, `elbow_pitch/yaw`) con lunghezza T2
3. Warm-start frame precedente + penalità velocità + clamp Δq + smooth Gaussiano

Cache: `refs/motion_previews/arm_ik_cache/` (gitignored).  
Invalida cambiando il prefisso `arm_ik_v*` in `MotionBank`.

---

## Viewer

```bash
scripts/factoryMjlab/run_retarget_gui.sh          # DISPLAY=:1, parte da motion 2
python -u scripts/factoryMjlab/view_retarget_motions.py --start 0
python -u scripts/factoryMjlab/view_retarget_web.py     # :8090 fallback
```

| Input | Azione |
|-------|--------|
| `1`–`6` / `[` `]` | Cambia clip |
| Space | Pause |
| LMB / wheel / RMB | Orbit / zoom / pan |
| `C` | Reset camera |
| Esc / `Q` | Esci |

EGL render in subprocess; GLFW solo per blit (evita `GLX BadAccess` con MuJoCo EGL).

---

## Policy remapping (runtime)

```
obs_T2 (31)
  → pack layout G1 (29)     # JointAdapter
  → history / actor G1
  → action_G1 (29)
  → unpack T2 (31)          # head=0; braccia seed (IK opzionale offline)
  → PD T2
```

I pesi rete restano 29-DoF. Per deploy lungo termine preferire un ckpt nativo T2 dopo fine-tune.

---

## Cosa non fare

- Caricare `goalkeeper.pt` grezzo sul low-level T2
- Hardware al primo tentativo senza smoke sim
- Aspettarsi parate zero-shot fedeli: le clip AMP retargetate sono il riferimento motion; la policy va ri-allenata / fine-tunata
