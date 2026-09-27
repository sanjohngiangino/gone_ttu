# Esperimento: pesi G1 Goalkeeper → Booster T2

**Obiettivo:** provare `goalkeeper.pt` (Unitree G1, 29 DoF) sul T2 (31 DoF).  
**Checkpoint locale:** `refs/goalkeeper.pt`  
**URDF T2:** `refs/T2_31dof.urdf` (da [booster_assets](https://github.com/BoosterRobotics/booster_assets))

---

## Verdetto corto

**Load diretto dei pesi: no.**  
**Prova con adapter di remapping: sì, solo in sim, aspettati comportamento fragile.**

Motivi:

1. **DoF diversi:** G1 policy = **29** azioni; T2 = **31** (+ testa)
2. **Ordine joint diverso**
3. **Cinematica braccia diversa** (non solo “simili”)
4. Policy allenata per **parate con le mani**, non piedi

---

## Cosa c’è nel checkpoint

```
goalkeeper.pt
  model_state_dict
    std                         (29,)          ← action dim G1
    history_encoder.*           in: 960        ← 10 × 96 obs/step
    ball_estimator.*            in: 960
    region_estimator.*
    actor.0.weight              (512, 119)     ← obs ridotta + features
    ...
  optimizer_state_dict
  iter / infos
```

Dalla config G1 Goalkeeper:

- `num_actions = 29`, `num_dofs = 29`
- `num_actor_history = 10`
- `num_one_step_observations = 6 + 3 + 29*2 + 29 = 96`
- action = target joint PD, `action_scale = 0.25`

---

## Confronto morpho

| Parte | G1 (Goalkeeper) | T2 (31 DoF) | Match? |
|-------|-----------------|-------------|--------|
| Gambe | 6+6 | 6+6 (`*_knee_pitch`) | Circa sì (nomi/sign da verificare) |
| Waist | yaw, roll, pitch | pitch, roll, yaw | Sì, **ordine diverso** |
| Braccia | shoulder P/R/**Y**, elbow, wrist R/P/Y | shoulder P/R, **elbow P/Y**, wrist P/Y/R | **No** — topologia diversa |
| Testa | — | `aa_head_yaw`, `head_pitch` | Extra T2 → tenere a 0 |

### Braccia: il vero ostacolo

```
G1:  sh_pitch → sh_roll → sh_yaw → elbow → wr_roll → wr_pitch → wr_yaw
T2:  sh_pitch → sh_roll → elbow_pitch → elbow_yaw → wr_pitch → wr_yaw → wr_roll
```

Non esiste un mapping 1:1 di `shoulder_yaw` / `elbow` G1 sugli assi T2.  
Qualsiasi remapping è un’approssimazione.

---

## Mapping proposto (esperimento)

Solo per **smoke test in sim**. Segni (±) e default angle da calibrare sul T2 reale.

| Indice G1 | Joint G1 | Joint T2 (candidato) | Note |
|-----------|----------|----------------------|------|
| 0–5 | left leg * | left_hip_* / left_knee_pitch / left_ankle_* | |
| 6–11 | right leg * | right_* analoghi | |
| 12 | waist_yaw | waist_yaw_joint | |
| 13 | waist_roll | waist_roll_joint | |
| 14 | waist_pitch | waist_pitch_joint | |
| 15–16 | L sh pitch/roll | left_shoulder_pitch/roll | |
| 17 | L sh yaw | **???** → 0 o mix su elbow_yaw | mismatch |
| 18 | L elbow | left_elbow_pitch | approx |
| 19–21 | L wrist R/P/Y | left_wrist_roll/pitch/yaw | ordine diverso |
| 22–28 | right arm | speculare | stessi mismatch |
| — | — | aa_head_yaw, head_pitch | **forzati a 0** |

Flusso runtime:

```
obs_T2 (31 DoF)
  → pack in layout G1 (29)   # drop head; remap arm approx
  → history stack (10)
  → actor G1 (goalkeeper.pt)
  → action_G1 (29)
  → unpack su T2 (31)        # head=0; arm approx
  → PD T2 (gains T2, non G1)
```

I pesi della rete **non si ricompilano**: resta MLP 29-DoF.  
Cambia solo il **wrapper** obs/action.

---

## Cosa NON fare

- Caricare `goalkeeper.pt` direttamente nel deploy T2 (shape mismatch / joint order sbagliato)
- Mettere su **hardware** al primo tentativo (policy braccia aggressive + kinematics sbagliata)
- Aspettarsi parate a piedi: questi pesi sono **hand-keeper**

---

## Piano di prova consigliato

1. **Sim T2** (MuJoCo/Isaac con `T2_31dof` da booster_assets)
2. Implementare wrapper remapping (tabella sopra)
3. Ball obs in frame robot come nel paper (locale)
4. PD gains **T2**, non copiare stiffness G1 alla cieca
5. Metriche smoke: non cade a riposo? reagisce alla palla? braccia/gambe plausibili?
6. Se interessante → fine-tune su T2 (congelare backbone o train corto), non transfer zero-shot “buono”

---

## Prerequisiti che mi servono da te

- [ ] Stack deploy T2 che usate (booster_deploy / custom / Isaac?)
- [ ] Ordine joint ufficiale del low-level T2 (non solo URDF)
- [ ] Conferma: prova in **sim** o volete proprio hardware subito?
- [ ] Segni joint noti (Booster a volte inverte roll sx/dx rispetto a Unitree)

---

## File già scaricati in questo repo

- `refs/goalkeeper.pt` — policy G1
- `refs/T2_31dof.urdf` — modello T2 pubblico
