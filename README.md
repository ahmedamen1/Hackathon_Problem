# Energy AI Hackathon 2026 — Solship

Residential energy optimization: load forecasting + MPC battery dispatch controller for an Italian site with rooftop PV and lithium battery.

---

## What's in this repo

| File | Description |
|---|---|
| `SolshipHack1&2.ipynb` | Main notebook — full pipeline from EDA to results |
| `generate_datasets.py` | Generates synthetic training (2024) and test (2025) datasets |
| `dataset_2024.csv` | Synthetic 2024 training data (35 136 rows, 15-min resolution) |
| `dataset_2025.csv` | Synthetic 2025 test data with injected battery corruption window |
| `Solship_Participant_Brief_v4 (1).pdf` | Official hackathon problem statement |

---

## Pipeline overview

```
dataset_2024.csv  →  Preprocessing  →  Feature Engineering  →  LightGBM (train)
                                                                      ↓
dataset_2025.csv  →  Preprocessing  →  Feature Engineering  →  LightGBM (forecast)
                                                                      ↓
                                                              MPC Optimizer (CVXPY)
                                                                      ↓
                                                              Results & Plots
```

---

## Notebook blocks

| Block | What it does |
|---|---|
| 1 — Preprocessing | Cleans data, enforces physical bounds (load ≥ 0, PV ≥ 0) |
| 2 — EDA | Year overview, seasonal profiles, day-of-week heatmap, autocorrelation, SoC reconstruction + **corruption window detection** on 2025 |
| 3 — Feature Engineering | 30+ features: cyclic sin/cos encoding of hour/DOW/month, lag features (15 min → 1 week), rolling stats, same-slot-yesterday/lastweek, Italian tariff band |
| 4 — Train/Val Split | Temporal split (no leakage): first 80% of 2024 trains, last 20% validates |
| 5 — LightGBM Model | Gradient boosting with early stopping; feature importance plot |
| 6 — Forecast 2025 | Apply trained model to 2025 (causal: 2024 tail prepended for lag warmup) |
| 7 — Baselines | **Baseline A** (historical on-site controller) and **Baseline B** (zero-intelligence, battery off) |
| 8 — MPC Optimizer | CVXPY LP per timestep, CLARABEL solver, H=96 (24 h) look-ahead |
| 9 — Run MPC | Full 2025 dispatch with forecasted load |
| 10 — Oracle Gap | Re-run MPC with perfect load — quantifies cost of forecast error |
| 11 — Extension 1 | Horizon sensitivity: H = 4 / 24 / 96 steps (1 h / 6 h / 24 h) |
| 12 — March Week 3 Plot | **Mandatory** 5-panel dispatch plot: load, PV, battery, grid, SoC |
| 13 — Results Table | Full scorecard: bills, savings vs A/B, oracle gap, extension table |
| 14 — Day 2 Dataset | Stub cell: drop `dataset_day2.csv` and re-run to get generalization NRMSE |

---

## System specs (from brief)

| Parameter | Value |
|---|---|
| Solar PV capacity | 9 kWp |
| Battery capacity | 16 kWh |
| Battery max power | 8 kW (charge & discharge) |
| Battery round-trip efficiency | 90% (√0.90 ≈ 0.9487 per direction) |
| Grid connection limit | 6 kW |
| Initial SoC | 50% |
| Time resolution | 15 min (Δt = 0.25 h) |

Italian ToU tariff: F1 = €0.2540/kWh, F2 = €0.2682/kWh, F3 = €0.2440/kWh.

---

## Install dependencies

```bash
pip install pandas numpy matplotlib lightgbm cvxpy holidays scikit-learn
```

---

## How to run

1. Open `SolshipHack1&2.ipynb` in Jupyter or VS Code
2. Run all cells top to bottom
3. MPC blocks (9, 10, 11) are the slow ones — each runs ~35 000 LP solves
4. On Day 2, place the released dataset as `dataset_day2.csv` and run Block 14

---

## Scoring (brief summary)

| Criterion | Points |
|---|---|
| Controller savings vs Baseline A | 35 |
| Forecasting NRMSE on 2025 | 25 |
| Generalization NRMSE on Day 2 dataset | 25 |
| Reasoning & presentation clarity | 15 |
| **Extension 1** (horizon sensitivity ≥ 3 H values) | +5 bonus |
