# SPQR Portiere — RL su Booster T2

**Stato:** brief di fattibilità (bozza)  
**Piattaforma target:** Booster T2  
**Proxy di sviluppo:** Unitree G1 (morpho e literature più affini)

---

## 1. Obiettivo

Policy RL end-to-end per portiere umanoide:

- intercettare palle in volo / a terra verso la porta
- motion naturalistiche (style da video/MoCap)
- deploy hardware-feasible sul T2
- **senza focus sulle mani** → end-effector = **piedi** (+ corpo / ginocchia)

---

## 2. Perché G1 (e non solo T1)

Il T2 è morfologicamente vicino al G1. Nella literature humanoid ball sports conviene quindi partire da portieri RL già dimostrati su G1, non da pipeline T1-specific (dribbling/shooting Booster).

Lista di riferimento:

- [awesome-humanoid-ball-sports](https://github.com/SUZ-tsinghua/awesome-humanoid-ball-sports)

---

## 3. Paper di riferimento

### A — Task portiere (primario)

**Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints**  
- Paper: https://arxiv.org/abs/2510.18002  
- Project: https://humanoid-goalkeeper.github.io/Goalkeeper/  
- Code: https://github.com/InternRobotics/Humanoid-Goalkeeper  
- Robot: Unitree G1 · Real robot ✅  

Cosa prende SPQR:

| Componente paper | Uso SPQR |
|------------------|----------|
| Video → GVHMR → retarget G1 | Stesso pipeline, retarget su T2 |
| Position-conditioned AMP | Prior diversi per regione di arrivo palla |
| PPO single-stage | Stesso |
| Task reward su EE vs landing | **Adattato: piedi invece di mani** |
| Critic privilegiato (palla, regione, EE) | Critic con **piede sx / dx** |
| Post-task stability | Tenere |

Cosa **non** prendere as-is:

- reward e critic basati su `p_hand_left / p_hand_right`
- punch/catch con le mani (RoboCup / SPQR: parata a piedi/corpo)

### B — Ricetta training multi-skill (secondario)

**SkillX: Unified Multi-Skill Policy Learning for Humanoid Soccer**  
- Paper: https://arxiv.org/html/2609.06718v1  
- Project: https://yzc0731.github.io/SkillX/  
- Robot: Noetix E1 · Real robot ✅  

Cosa prende SPQR:

| Componente SkillX | Uso SPQR |
|-------------------|----------|
| Skill-specific adversarial motion priors | Una `D_i` per regione / skill di parata |
| Skill-specific critics + PPO | Value heads per skill/regione |
| Object-aware temporal encoder | Utile se palla noisy / history corta |
| MoCap → retarget → Isaac | Stesso flusso dati |

SkillX è su dribble/trap/shoot, **non** sul portiere; serve come blueprint di training, non come task.

---

## 4. Pipeline dati (già dimostrata)

```
Video umani (parate)
    → stima motion (es. GVHMR / MoCap)
    → retarget morpho T2 (via G1 come ponte se serve)
    → buffer motion per regione / skill
    → discriminatori AMP (style reward)
```

Note dagli appunti:

> algoritmo adv motion prior — già funziona, a partire da video — mockup — dati mockup — reward

Verdetto: **sì, il pezzo dati→AMP è maturo** in entrambi i paper. Il rischio non è “AMP da video”, è l’adattamento task piedi + asset T2.

---

## 5. Design SPQR (da appunti)

### 5.1 Actor / Critic / Algo

- **Algo:** PPO
- **Actor:** history di propriocezione + palla in frame robot → target joint (PD basso livello)
- **Critic:** privilegiato — regione, velocità palla, **piede sx / dx**, distanza EE–target
- **Non ci interessano le mani** come EE di task

### 5.2 Motion prior

- AMP **condizionato sulla palla** (regione di landing / skill)
- Feature discriminator tipiche: joint, key bodies (**ginocchio**, bacino), root vel, **foot contact**
- Style reward da `D_i` attivo (o blend soft tra skill, stile SkillX)

### 5.3 Reward (da ricalibrare guardando i paper)

Ispirazione Humanoid Goalkeeper §III-A2, riscritta sui piedi:

1. **Task / keep** — avvicinare piede (sx|dx) scelto dalla regione al target (landing lontano, palla vicino)
2. **Whole-body modulation** — shift laterale / squat / jump in base alla regione
3. **AMP style** — resemblance alle demo umane della regione
4. **Post-task stability** — bilanciamento dopo la parata
5. **Hardware feasibility** — torque, contact, cadute (come in appendix paper)

Da fare: tabella reward numerica (pesi) dopo lettura appendix Goalkeeper + primi run sim.

### 5.4 Observation space (bozza)

| Osservazione | Actor | Critic |
|--------------|:-----:|:------:|
| Posizione palla (frame locale) | ✓ | ✓ |
| Base ang. vel / gravity | ✓ | ✓ |
| Joint pos / vel | ✓ | ✓ |
| Last action | ✓ | ✓ |
| History (T frame) | ✓ | ✓ |
| Base lin. vel | — | ✓ |
| Regione / landing | — | ✓ |
| Velocità palla | — | ✓ |
| Piede sx / dx pos | — | ✓ |
| Reach distance piede–target | — | ✓ |
| Mani | — | **no** |

---

## 6. Mappa regioni → skill (proposta iniziale)

Allineata al paper Goalkeeper (regioni porta), semplificata per piedi:

| Regione | EE preferito | Motion prior atteso |
|---------|--------------|---------------------|
| Basso sinistra | Piede sx | Laterale + blocco basso |
| Basso destra | Piede dx | Speculare |
| Centro basso | Piede più vicino / corpo | Blocco / clear |
| Alto / volo | Corpo + ginocchio / jump | Da validare: T2 può coprire? |

Se in RoboCup la palla è soprattutto a terra, si può partire con **3 regioni basse** e aggiungere salto dopo.

---

## 7. Transfer pesi G1 → T2 (esperimento)

Vedi dettaglio in [`TRANSFER_G1_TO_T2.md`](TRANSFER_G1_TO_T2.md).

**Prima però:** port Goalkeeper su **mjlab** (G1), stile [SoccerLab](https://github.com/Renforce-Dynamics/soccerLab) — doc [`docs/MJLAB_GOALKEEPER_PORT.md`](docs/MJLAB_GOALKEEPER_PORT.md).

- Checkpoint Goalkeeper: **29 DoF**, hand-keeper
- T2 pubblico: **31 DoF** (`booster_assets/robots/T2/T2_31dof`)
- Load diretto: **no** (dim + ordine + topologia braccia diverse)
- Prova utile: wrapper remapping in **sim**, testa a 0; non hardware al primo colpo

---

## 8. Fattibilità

### 8.1 Cosa è già dimostrato (alto confidenza)

| Pezzo | Evidenza | Fattibile? |
|-------|----------|------------|
| AMP da video/MoCap | Goalkeeper + SkillX | ✅ |
| PPO + task reward + AMP single-stage | Goalkeeper su G1 real | ✅ |
| Conditioning su posizione palla / regione | Goalkeeper | ✅ |
| Multi-critic / multi-AMP | SkillX | ✅ (opzionale v1) |
| Sim Isaac + PD joint targets | Entrambi | ✅ |

### 8.2 Cosa è adattamento SPQR (medio rischio)

| Pezzo | Rischio | Note |
|-------|---------|------|
| **Mani → piedi** | Medio | Cambia reward, critic, motion library; non copy-paste del code Goalkeeper |
| **Retarget G1 → T2** | Medio | Serve URDF/asset T2 + retarget (PHC o tool Booster); DoF/limits diversi |
| **Copertura “ops space”** | Medio-alto | Range di parata del T2 vs G1; validare reach piedi e stabilità laterale |
| **Palla a terra vs volo** | Basso-medio | A terra è più vicino a soccer skills esistenti; volo è il hard case del paper |

### 8.3 Cosa può bloccare

1. **Asset / sim T2** non pronti in Isaac (URDF, actuator model, PD gains)
2. **Motion library** di parate a piedi insufficiente o male retargetata
3. **Sim-to-real** palla (massa, bounce, delay visione) senza MoCap iniziale
4. **Regole RoboCup** (contatto, fall, hands) che vincolano reward/safety oltre il paper

### 8.4 Verdetto

**Sì, è fattibile** come linea di ricerca/engineering, perché:

1. esiste già un portiere RL G1 end-to-end con code pubblico;
2. AMP video→robot è collaudato;
3. T2≈G1 riduce il salto morpho rispetto a ripartire da zero.

**Non è un clone overnight:** il delta principale è **task a piedi** + **porting asset T2**. Stima realistica di un MVP sim (parata bassa sx/dx): ordine di settimane se asset e MoCap/video sono disponibili; mesi per hardware robusto.

---

## 9. Piano MVP (dopo ok su questo .md)

1. **Lock scope:** solo regioni basse, EE = piedi, no mani  
2. **Clone** Humanoid-Goalkeeper, far girare su G1 in sim (baseline)  
3. **Sostituire** reward/critic `hand` → `foot_L/R`  
4. **Raccolta motion** video parate a piedi → retarget  
5. **Port asset** G1 → T2 (o train su G1 e transfer se T2 non ancora in sim)  
6. **Train** PPO + AMP region-conditioned  
7. **Eval sim** success rate per regione + cadute  
8. **Deploy** MoCap palla → poi visione onboard  

---

## 10. Domande aperte (da chiudere prima di code serio)

- [ ] Palla principalmente a terra o anche in volo?
- [ ] Quante regioni porta v1?
- [ ] Abbiamo già URDF/sim Booster T2 in Isaac?
- [ ] Video/MoCap di parate disponibili, o da registrare?
- [ ] Constraint RoboCup su uso braccia / fall recovery?
- [ ] Train diretto su T2 o prima G1 poi transfer?

---

## 11. Link rapidi

| Risorsa | URL |
|---------|-----|
| Awesome list | https://github.com/SUZ-tsinghua/awesome-humanoid-ball-sports |
| Humanoid Goalkeeper paper | https://arxiv.org/abs/2510.18002 |
| Humanoid Goalkeeper code | https://github.com/InternRobotics/Humanoid-Goalkeeper |
| SkillX paper | https://arxiv.org/html/2609.06718v1 |
| SkillX project | https://yzc0731.github.io/SkillX/ |

---

*Bozza da appunti SPQR — RL Portiere — T2 Booster Robot.*
