# Energy AI Hackathon 2026 — Solship

Residential energy optimization for an Italian site with rooftop PV and lithium battery.
**Task:** Day-ahead load forecasting + MPC battery dispatch controller to minimize electricity bill.

---

## System Specs

| Parameter | Value |
|---|---|
| Solar PV | 9 kWp |
| Battery capacity | 16 kWh |
| Battery max power | 8 kW (charge & discharge) |
| Round-trip efficiency | 90% (√0.90 ≈ 0.9487 per direction) |
| Grid connection limit | 6 kW |
| Initial SoC | 50% |
| Time resolution | 15 min (Δt = 0.25 h) |

Italian ToU tariff: F1 = €0.2540/kWh (weekday 08-19h), F2 = €0.2682/kWh (shoulders), F3 = €0.2440/kWh (night/weekend).

---

## Approaches & Trials

### Forecasting

#### Trial 1 — Single LightGBM (baseline)
- 32 features including lag_1, lag_2, lag_4
- **Problem:** val curve flatlined at round 10-45 — lag_1 leakage (model learns "next ≈ current")
- Val RMSE: 0.79 kW (artificially low due to leakage)

#### Trial 2 — LightGBM day-ahead (27 features, no short lags)
- Removed lag_1, lag_2, lag_4, roll_mean_4, roll_std_4
- Kept only day-ahead safe features: lag_96 (yesterday), lag_672 (last week), rolling 96/672 windows
- Re-ran Optuna (50 trials) on the correct 27-feature set
- **Best params:** lr=0.0317, num_leaves=125, max_depth=4, min_child_samples=188
- **Result:** Best round 147, Train RMSE 0.82, Val RMSE 1.09, NRMSE 15.48% (max-min)

#### Trial 3 — Ensemble (LightGBM + XGBoost + CatBoost + RF + Ridge)
- Simple average + Nelder-Mead optimal weighted combination
- **Result:** Combined NRMSE 14.81% (Apr 15.57%, Sep 14.05%)

#### Trial 4 — Rolling day-ahead forecast (day-by-day)
- Each day forecasted using real observed lags from previous day
- **Result:** Combined NRMSE 14.80% — identical to one-shot (lags already use real data)

#### Trial 5 — Hybrid ARIMA + LSTM
- ARIMA(2,0,2) fitted on 2024 training series
- **Problem:** ARIMA std = 0.000 kW — collapses to flat mean for long-horizon forecasts
- ARIMA alone NRMSE: ~33% — useless beyond ~100 steps out-of-sample
- LSTM on residuals: minor improvement but ARIMA base too weak

#### Trial 6 — Two-Model LightGBM (Day + Night)
- Night model (22:00–06:00): stronger regularisation (num_leaves=31, reg_lambda=5.0)
- Day model (06:00–22:00): standard Optuna-tuned params
- Motivation: nighttime load is near-constant ~0.2-0.4 kW, separate model prevents overfitting flat signal
- **Result:** See lgb_two_model_forecasts.csv

#### Trial 7 — Transformer (direct day-ahead)
- 7-day context window (672 slots), 12 input features, 96-step direct output
- Positional embeddings, 3 encoder layers, hidden=128, nhead=8
- **Problem:** CUDA kernel incompatibility on Kaggle T4 (cudaErrorNoKernelImageForDevice)
- Workaround: CPU training — too slow (15-30 min/epoch)

### MPC Controller

#### Approach — CVXPY rolling-horizon LP
- Solver: CLARABEL
- Variables: p_charge, p_discharge, p_grid, SoC per timestep
- Constraints: SoC 5-95%, battery power ≤ 8 kW, grid ≤ 6 kW, energy balance
- Objective: minimize import cost − export revenue over horizon H

#### Extension 1 — Horizon sensitivity
| H (steps) | H (hours) | Bill (EUR) | Savings vs Baseline A |
|---|---|---|---|
| 4 | 1h | higher than A | negative |
| 24 | 6h | moderate | ~5% |
| 96 | 24h | best | ~11.2% |

**Conclusion:** H=96 (24h look-ahead) is optimal. H=4 is myopic and makes decisions that increase cost.

---

## Pipeline

```
Raw CSV → Clean (DST fix, ffill gaps) → Feature Engineering (27 features)
    → LightGBM (Optuna-tuned, day-ahead safe)
    → Ensemble (LightGBM + XGB + CatBoost + RF + Ridge)
    → MPC Optimizer (CVXPY CLARABEL, H=96)
    → Results & Plots
```

---

## Files

| File | Description |
|---|---|
| `energy-ai-hackathon-2026-solship.ipynb` | Main notebook — full pipeline |
| `lgb_two_model_forecasts.csv` | Day/night model predictions vs actuals |
| `April.jpeg` | April 2025 forecast plot |
| `september.jpeg` | September 2025 forecast plot |
| `ENERGY_Hackathon_DataSet(Sheet1).csv` | Raw dataset (semicolon-separated, European decimal) |

---

## Results

| Metric | Value |
|---|---|
| Forecast NRMSE (Apr, max-min) | 15.48% |
| Forecast NRMSE (Sep, max-min) | ~14.05% |
| Combined NRMSE | ~14.81% |
| MPC savings vs Baseline A | ~11.2% |
| MPC savings vs Baseline B | varies |

---

## Install

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Upload `ENERGY_Hackathon_DataSet(Sheet1).csv` to Kaggle input
2. Open `energy-ai-hackathon-2026-solship.ipynb`
3. Run all cells top to bottom
4. MPC cells are slow (~35k LP solves) — allow 10-15 min
