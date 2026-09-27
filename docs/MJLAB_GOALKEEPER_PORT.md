# Goalkeeper → mjlab port (G1 first)

Stesso approccio di [SoccerLab `docs/mjlab_port.md`](https://github.com/Renforce-Dynamics/soccerLab/blob/master/docs/mjlab_port.md):
**Isaac (Gym/Lab) → mjlab**, robot **G1 29-DoF**, poi solo dopo T2.

SoccerLab ha portato kick/dribble. Noi portiamo **Humanoid Goalkeeper**
([InternRobotics](https://github.com/InternRobotics/Humanoid-Goalkeeper)).

---

## Perché prima mjlab (non il nostro viewer MuJoCo grezzo)

| Path | Cosa fa |
|------|---------|
| `play.py` IsaacGym ufficiale | Portiere “vero” con il loro stack |
| Nostro `view_side_by_side.py` | Smoke transfer — collassa (atteso) |
| **Questo port mjlab** | Env Goalkeeper su MuJoCo-Warp API Isaac-like, train/play serio |

I pesi `goalkeeper.pt` (IsaacGym) **non** si caricano zero-shot in mjlab.
Si riporta l’**env + AMP + reward**; si ri-allena (o si tenta warm-start dopo allineamento obs).

---

## Dipendenze (come SoccerLab)

```bash
# Linux + NVIDIA GPU consigliato (mujoco-warp)
pip install mjlab          # tipicamente >= 1.4, torch>=2.7, mujoco~=3.8
# AMP runner (stesso ecosistema beyondAMP di SoccerLab)
# pip install -e beyondAMP/...  se serve AMP

pip install -e source/goalkeeper_tasks_mjlab
```

macOS: si può sviluppare cfg/obs/reward; **train GPU** su macchina CUDA.

---

## Task registrato

| Task ID | Robot | Algo target |
|---------|-------|-------------|
| `SPQR-Mjlab-Goalkeeper-G1` | Unitree G1 29-DoF (mjlab asset zoo) | PPO + AMP region-conditioned |

Layout (parallelo a SoccerLab `soccer_tasks_mjlab`):

```
source/goalkeeper_tasks_mjlab/
  goalkeeper_tasks_mjlab/
    __init__.py
    g1/
      __init__.py              # register_mjlab_task
      env_cfg.py
      agents/ppo_cfg.py
      mdp/
        observations.py
        rewards.py
        events.py
        terminations.py
data/assets/ball/soccer_ball.xml
data/datasets/goalkeeper/      # motion .pt da Humanoid-Goalkeeper (symlink/copy)
scripts/factoryMjlab/
  train.py
  play.py
```

---

## Mapping IsaacGym Goalkeeper → mjlab

### Asset

- Robot: `mjlab.asset_zoo.robots.get_g1_robot_cfg()` (stessi 29 joint names)
- Palla: `data/assets/ball/soccer_ball.xml` (FIFA-ish)
- Motion AMP: `resources/datasets/goalkeeper/*.pt` della repo ufficiale

### Observation (actor one-step = 96, history T=10)

Da `G129Cfg`:

```
ang_vel(3) + gravity(3) + ball_pos(3) + dof_pos(29) + dof_vel(29) + last_action(29)
= 96  →  stacked/history 960
```

Critic privilegiato: + root_lin_vel, region, target, ball_vel, hand poses, reach (come paper).

In mjlab: gruppo obs `"actor"` (non `"policy"` — vedi cheat sheet SoccerLab).

### Action

- 29 joint position targets, `action_scale=0.25`, PD come config G1 Goalkeeper

### Rewards (pesi da `G129Cfg.rewards.scales`)

| Term | Scale | Note port |
|------|------:|-----------|
| eereach | 10 | EE (mano regione) → target |
| success | 5 | |
| stopball | 100 | |
| stayonline / noretreat | -2 | |
| successland, feetorientation, … | vari | feet |
| postorientation, postangvel, … | vari | post-task |
| ang_vel_xy, dof_acc, smoothness, limits | reg | |

AMP: `amp_coef=0.4`, obs dof×2, **region-conditioned** discriminators (6 regioni).

### Eventi

- Reset robot + lancio palla verso regione `R ∈ {0..5}` (ranges height/width in config)
- Episode 3 s, post-task stability

### API cheat sheet (da SoccerLab)

| Isaac | mjlab |
|-------|-------|
| `body_pos_w` | `body_link_pos_w` |
| `root_pos_w` | `root_link_pos_w` |
| `@configclass` | `@dataclass(kw_only=True)` |
| `ManagerBasedRLEnvCfg` | `ManagerBasedRlEnvCfg` |
| `gym.register` | `register_mjlab_task` |
| obs group `policy` | obs group `actor` |

---

## Ordine di lavoro

1. **Scaffold** task + ball asset + doc (questo step)
2. **Smoke env** `play --agent zero` (robot + palla, no policy)
3. **Reward/obs parity** vs Isaac (unit test dimensioni 96 / privileged)
4. **AMP** motion buffer da `.pt` Goalkeeper + discriminator per regione
5. **Train** smoke 50–100 iter su GPU
6. **Play** checkpoint mjlab
7. **Dopo**: stesso task su T2 / transfer (g1_to_t2)

---

## Comandi target

```bash
# Lista task
python -c "from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Goalkeeper' in t])"

# Zero agent (sanity scena)
python scripts/factoryMjlab/play.py SPQR-Mjlab-Goalkeeper-G1 --agent zero

# Train smoke
python scripts/factoryMjlab/train.py SPQR-Mjlab-Goalkeeper-G1 --num_envs 64 --max_iterations 50
```

---

## Cosa non fare adesso

- Caricare `refs/goalkeeper.pt` Isaac in mjlab e aspettarsi parate
- Port T2 prima che G1 mjlab stia in piedi sul task
- Dipendere dal viewer MuJoCo grezzo di `g1_to_t2` per validare il portiere
