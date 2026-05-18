# CAIE — Cognitive Adaptation in the Intelligence Era

This repository contains the simulation code and figures for the paper:

**"Cognitive Adaptation in the Intelligence Era (CAIE): A Dynamical Systems Framework for Education Reform in the Age of Artificial Intelligence"**

Author: Chung Ming Chen  
Submitted to: Frontiers in Education

---

## Repository Contents

```
math/
  caie_experiment.py       # Main ODE simulation (RK4 solver, 3 education scenarios)
  caie_sensitivity.py      # Sensitivity analysis (±20% parameter perturbation)
caie_fig5_sensitivity_tornado.png  # Sensitivity tornado chart (Figure in paper)
```

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scipy

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

---

## Running the Simulation

**Main simulation** (reproduces cognitive trajectory predictions, Hypotheses 1–4):
```bash
python math/caie_experiment.py
```

**Sensitivity analysis** (reproduces parameter robustness results, Section 4.5):
```bash
python math/caie_sensitivity.py
```

---

## Model Overview

The CAIE model defines a 7-dimensional cognitive-affective state vector S(t) whose evolution is governed by:

```
dS/dt = E_eff(S, RPE) - R_tech(t) - D_res(S) - D_cum(t)
```

Where:
- `E_eff` — effective education function modulated by dopamine reward prediction error (RPE)
- `R_tech` — sigmoid-shaped technology pressure (steam, internet, AI epochs)
- `D_res` — three-component cognitive resistance (neuroplasticity decay, bandwidth constraint, habitual inertia)
- `D_cum` — cumulative cognitive burden from prior technological epochs

Three scenarios are simulated: Traditional Education, CAIE (no RPE), and CAIE (with RPE).

---

## Key Predictions

| Hypothesis | Prediction |
|---|---|
| H1 | Traditional curricula become insufficient after AI inflection point (~2023) |
| H2 | CAIE+RPE design maintains positive adaptation across all 7 dimensions |
| H3 | Optimal concurrent AI tools: n* ≈ 3 (untrained), 5–8 (trained) |
| H4 | Critical intervention window: τ* ≈ 1.25 years post-inflection |

---

## License

MIT License. See LICENSE for details.
