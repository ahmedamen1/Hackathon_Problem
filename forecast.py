"""
Energy AI Hackathon 2026 — Load Forecasting Pipeline
Predicts load_kw only for April & September 2025
Runs locally, iterates to best results via Optuna
"""
import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays as hol
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, 'ENERGY_Hackathon_DataSet(Sheet1).csv')
OUTDIR = os.path.join(BASE, 'forecast_output')
os.makedirs(OUTDIR, exist_ok=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))

def nrmse_range(y, p):
    return rmse(y, p) / (y.max() - y.min())

def nrmse_mean(y, p):
    return rmse(y, p) / y.mean()

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Load & Clean
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("BLOCK 1 — Loading data")
print("=" * 60)

raw = pd.read_csv(DATA, sep=';', decimal=',')
raw['timestamp'] = pd.to_datetime(raw['timestamp'])
raw = raw.sort_values('timestamp').reset_index(drop=True)

# Keep only columns needed for load forecasting
df = raw.rename(columns={
    'load_p':               'load_kw',
    'pv_p':                 'pv_kw',
    'Selling_price_eur_kwh':'sell_price',
})
# Only keep what we need — no battery/grid columns
df = df[['timestamp', 'load_kw', 'pv_kw', 'sell_price']].copy()

# Fix DST duplicates
before = len(df)
df = df.drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate timestamps")

# Complete 15-min grid
full_index = pd.date_range(start=df['timestamp'].min(),
                           end=df['timestamp'].max(), freq='15min')
df = df.set_index('timestamp').reindex(full_index).ffill().reset_index()
df = df.rename(columns={'index': 'timestamp'})
df['sell_price'] = df['sell_price'].ffill()

# Buy price
italy_hols = hol.Italy(years=[2024, 2025])
def get_buy_price(ts):
    h, d = ts.hour, ts.dayofweek
    if ts.date() in italy_hols or d == 6: return 0.2440
    if d <= 4 and 8 <= h < 19:           return 0.2540
    if (d <= 4 and (7 <= h < 8 or 19 <= h < 23)) or (d == 5 and 7 <= h < 23):
        return 0.2682
    return 0.2440

df['buy_price'] = df['timestamp'].map(get_buy_price)

train_df = df[df['timestamp'].dt.year == 2024].reset_index(drop=True)
test_df  = df[df['timestamp'].dt.year == 2025].reset_index(drop=True)

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"Train load — mean: {train_df['load_kw'].mean():.3f}  max: {train_df['load_kw'].max():.3f}")
print(f"Test  load — mean: {test_df['load_kw'].mean():.3f}  max: {test_df['load_kw'].max():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 2 — Feature Engineering")
print("=" * 60)

def make_features(df_in, slot_stats=None):
    d  = df_in.copy()
    ts = d['timestamp']

    d['hour']       = ts.dt.hour
    d['minute']     = ts.dt.minute
    d['dow']        = ts.dt.dayofweek
    d['month']      = ts.dt.month
    d['day']        = ts.dt.day
    d['slot']       = d['hour'] * 4 + d['minute'] // 15
    d['is_weekend'] = (d['dow'] >= 5).astype(int)
    d['is_holiday'] = ts.apply(lambda x: int(x.date() in italy_hols))

    d['hour_sin']  = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos']  = np.cos(2 * np.pi * d['hour'] / 24)
    d['dow_sin']   = np.sin(2 * np.pi * d['dow'] / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['dow'] / 7)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['slot_sin']  = np.sin(2 * np.pi * d['slot'] / 96)
    d['slot_cos']  = np.cos(2 * np.pi * d['slot'] / 96)

    def tou(row):
        h, dw = row['hour'], row['dow']
        if row['is_holiday'] or dw == 6:                         return 2
        if dw <= 4 and 8 <= h < 19:                              return 0
        if (dw <= 4 and (7 <= h < 8 or 19 <= h < 23)) or \
           (dw == 5 and 7 <= h < 23):                            return 1
        return 2
    d['tou_band'] = d.apply(tou, axis=1)

    lk = d['load_kw']
    d['lag_96']              = lk.shift(96)
    d['lag_192']             = lk.shift(192)
    d['lag_288']             = lk.shift(288)
    d['lag_672']             = lk.shift(672)
    d['lag_1344']            = lk.shift(1344)
    d['lag_year']            = lk.shift(35040)
    d['same_slot_year_m1d']  = lk.shift(35040 - 96)
    d['same_slot_year_p1d']  = lk.shift(35040 + 96)
    d['same_slot_year_avg']  = (lk.shift(35040-96) + lk.shift(35040) + lk.shift(35040+96)) / 3

    d['roll_mean_96']  = lk.shift(96).rolling(96).mean()
    d['roll_std_96']   = lk.shift(96).rolling(96).std()
    d['roll_mean_192'] = lk.shift(96).rolling(192).mean()
    d['roll_std_192']  = lk.shift(96).rolling(192).std()
    d['roll_mean_672'] = lk.shift(96).rolling(672).mean()
    d['roll_std_672']  = lk.shift(96).rolling(672).std()

    d['same_slot_yesterday'] = lk.shift(96)
    d['same_slot_last_week'] = lk.shift(672)
    d['same_slot_2weeks']    = lk.shift(672 * 2)
    d['same_slot_3weeks']    = lk.shift(672 * 3)
    d['same_slot_4weeks']    = lk.shift(672 * 4)
    d['same_slot_week_avg']  = (lk.shift(672) + lk.shift(672*2) +
                                lk.shift(672*3) + lk.shift(672*4)) / 4

    pk = d['pv_kw']
    d['pv_lag_96']             = pk.shift(96)
    d['pv_roll_mean_96']       = pk.shift(96).rolling(96).mean()
    d['pv_lag_672']            = pk.shift(672)
    d['pv_lag_192']            = pk.shift(192)
    d['pv_lag_288']            = pk.shift(288)
    d['pv_lag_1344']           = pk.shift(1344)
    d['pv_roll_mean_672']      = pk.shift(96).rolling(672).mean()
    d['pv_same_slot_week_avg'] = (pk.shift(672) + pk.shift(672*2) + pk.shift(672*3)) / 3

    # Solar-load interaction features
    d['is_solar_peak']    = ((d['hour'] >= 9) & (d['hour'] < 16)).astype(int)
    d['hour_x_month']     = d['hour'] * d['month']
    d['pv_x_hour_sin']    = pk.shift(96) * d['hour_sin']
    d['load_pv_ratio_96'] = lk.shift(96) / (pk.shift(96) + 0.1)

    # Slot stats from training only
    if slot_stats is None:
        valid       = d.dropna(subset=['load_kw'])
        sdm         = valid.groupby(['slot', 'dow'])['load_kw'].mean()
        slot_stats  = {'slot_dow': sdm}

    d['slot_dow_mean'] = (
        d.set_index(['slot', 'dow']).index.map(slot_stats['slot_dow'])
    )

    return d, slot_stats

FEATURES = [
    'hour', 'minute', 'dow', 'month', 'day', 'slot',
    'is_weekend', 'is_holiday',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'slot_sin', 'slot_cos',
    'tou_band',
    'lag_96', 'lag_192', 'lag_288', 'lag_672', 'lag_1344',
    'lag_year', 'same_slot_year_m1d', 'same_slot_year_p1d', 'same_slot_year_avg',
    'roll_mean_96',  'roll_std_96',
    'roll_mean_192', 'roll_std_192',
    'roll_mean_672', 'roll_std_672',
    'same_slot_yesterday', 'same_slot_last_week',
    'same_slot_2weeks', 'same_slot_3weeks', 'same_slot_4weeks',
    'same_slot_week_avg',
    'slot_dow_mean',
    'pv_lag_96', 'pv_lag_192', 'pv_lag_288', 'pv_lag_672', 'pv_lag_1344',
    'pv_roll_mean_96', 'pv_roll_mean_672', 'pv_same_slot_week_avg',
    'is_solar_peak', 'hour_x_month', 'pv_x_hour_sin', 'load_pv_ratio_96',
]

print("Building train features...")
train_f, slot_stats = make_features(train_df)

WARMUP     = 35040 + 192
combined   = pd.concat([train_df.tail(WARMUP), test_df], ignore_index=True)
print("Building test features...")
combined_f, _ = make_features(combined, slot_stats=slot_stats)
test_f = combined_f.iloc[WARMUP:].reset_index(drop=True)

REQUIRED   = ['lag_96', 'lag_672', 'roll_mean_672', 'load_kw']
train_clean = train_f.dropna(subset=REQUIRED).reset_index(drop=True)
split       = int(len(train_clean) * 0.8)

X_train = train_clean[FEATURES].iloc[:split]
y_train = train_clean['load_kw'].iloc[:split]
X_val   = train_clean[FEATURES].iloc[split:]
y_val   = train_clean['load_kw'].iloc[split:]

# Recency weights: exponential decay, half-life = 6 months (17520 slots)
_n = len(X_train)
_half_life = 2880 * 30   # ~6 months
w_train = np.exp(np.linspace(-np.log(2) * _n / _half_life, 0, _n))
w_train = w_train / w_train.mean()   # keep mean=1 so LGB scale is unchanged

apr_2025 = test_f[test_f['timestamp'].dt.month == 4].reset_index(drop=True)
sep_2025 = test_f[test_f['timestamp'].dt.month == 9].reset_index(drop=True)

y_true_apr = apr_2025['load_kw'].values
y_true_sep = sep_2025['load_kw'].values

print(f"Features   : {len(FEATURES)}")
print(f"X_train    : {X_train.shape}  X_val: {X_val.shape}")
print(f"Apr 2025   : {len(apr_2025)} rows | Sep 2025: {len(sep_2025)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Baseline LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 3 — Baseline LightGBM")
print("=" * 60)

base_params = {
    'objective':         'regression',
    'metric':            'rmse',
    'verbosity':         -1,
    'learning_rate':     0.02,
    'num_leaves':        31,
    'max_depth':         5,
    'min_child_samples': 100,
    'subsample':         0.7,
    'subsample_freq':    1,
    'colsample_bytree':  0.7,
    'reg_alpha':         0.5,
    'reg_lambda':        2.0,
    'n_jobs':            -1,
    'seed':              42,
}

ds_tr = lgb.Dataset(X_train, label=y_train, weight=w_train)
ds_vl = lgb.Dataset(X_val,   label=y_val, reference=ds_tr)

evals = {}
model_base = lgb.train(
    base_params, ds_tr, num_boost_round=5000,
    valid_sets=[ds_tr, ds_vl], valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(300, verbose=False),
               lgb.log_evaluation(200),
               lgb.record_evaluation(evals)],
)

tr_rmse  = evals['train']['rmse'][model_base.best_iteration - 1]
val_rmse = evals['val']['rmse'][model_base.best_iteration - 1]
print(f"Best round : {model_base.best_iteration}")
print(f"Train RMSE : {tr_rmse:.4f} kW  Val RMSE: {val_rmse:.4f} kW  Gap: {val_rmse-tr_rmse:.4f}")

p_apr = np.clip(model_base.predict(apr_2025[FEATURES].fillna(0)), 0, None)
p_sep = np.clip(model_base.predict(sep_2025[FEATURES].fillna(0)), 0, None)
print(f"Baseline — Apr NRMSE/range: {nrmse_range(y_true_apr,p_apr)*100:.2f}%  "
      f"Sep: {nrmse_range(y_true_sep,p_sep)*100:.2f}%  "
      f"Combined: {0.5*(nrmse_range(y_true_apr,p_apr)+nrmse_range(y_true_sep,p_sep))*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — Optuna Tuning
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 4 — Optuna Tuning (100 trials)")
print("=" * 60)

def objective(trial):
    params = {
        'objective':           'regression',
        'metric':              'rmse',
        'verbosity':           -1,
        'n_jobs':              -1,
        'seed':                42,
        'feature_pre_filter':  False,
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':          trial.suggest_int('num_leaves', 16, 128),
        'max_depth':           trial.suggest_int('max_depth', 3, 8),
        'min_child_samples':   trial.suggest_int('min_child_samples', 20, 300),
        'subsample':           trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':    trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':           trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda':          trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    _ds_tr = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    _ds_vl = lgb.Dataset(X_val,   label=y_val,   reference=_ds_tr, free_raw_data=False)
    m = lgb.train(
        params, _ds_tr, num_boost_round=2000,
        valid_sets=[_ds_vl], valid_names=['val'],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    return m.best_score['val']['rmse']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_params
print(f"\nBest val RMSE: {study.best_value:.4f} kW")
print(f"Best params: {best}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — Final Model with Best Params
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 5 — Final Model")
print("=" * 60)

final_params = {
    'objective':          'regression',
    'metric':             'rmse',
    'verbosity':          -1,
    'n_jobs':             -1,
    'seed':               42,
    'feature_pre_filter': False,
    **best,
}

ds_tr_f = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
ds_vl_f = lgb.Dataset(X_val,   label=y_val,   reference=ds_tr_f, free_raw_data=False)

evals_final = {}
model_final = lgb.train(
    final_params, ds_tr_f, num_boost_round=5000,
    valid_sets=[ds_tr_f, ds_vl_f], valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(300, verbose=False),
               lgb.log_evaluation(100),
               lgb.record_evaluation(evals_final)],
)

tr_f  = evals_final['train']['rmse'][model_final.best_iteration - 1]
vl_f  = evals_final['val']['rmse'][model_final.best_iteration - 1]
print(f"Best round : {model_final.best_iteration}")
print(f"Train RMSE : {tr_f:.4f}  Val RMSE: {vl_f:.4f}  Gap: {vl_f-tr_f:.4f}")

# Loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(evals_final['train']['rmse'], label='Train', color='steelblue', lw=1.2)
ax.plot(evals_final['val']['rmse'],   label='Val',   color='tomato',    lw=1.2)
ax.axvline(model_final.best_iteration-1, color='green', ls='--',
           label=f'Best {model_final.best_iteration}')
ax.set_title(f'LightGBM — Train {tr_f:.4f}  Val {vl_f:.4f}  Gap {vl_f-tr_f:.4f}')
ax.set_xlabel('Round'); ax.set_ylabel('RMSE (kW)'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'loss_curve.png'), dpi=110)
plt.close()
print("Saved loss_curve.png")

# Feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, imp_type in zip(axes, ['gain', 'split']):
    imp = pd.Series(model_final.feature_importance(imp_type),
                    index=FEATURES).sort_values(ascending=True).tail(20)
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Feature importance ({imp_type}) — top 20')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'feature_importance.png'), dpi=110)
plt.close()
print("Saved feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 6 — Evaluate on Apr & Sep 2025
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 6 — Evaluation")
print("=" * 60)

p_apr_f = np.clip(model_final.predict(apr_2025[FEATURES].fillna(0)), 0, None)
p_sep_f = np.clip(model_final.predict(sep_2025[FEATURES].fillna(0)), 0, None)

def print_metrics(label, y, p):
    print(f"\n{label}")
    print(f"  RMSE           : {rmse(y,p):.4f} kW")
    print(f"  MAE            : {mean_absolute_error(y,p):.4f} kW")
    print(f"  NRMSE/(max-min): {nrmse_range(y,p)*100:.2f}%")
    print(f"  NRMSE/mean     : {nrmse_mean(y,p)*100:.2f}%")

print_metrics("April 2025",     y_true_apr, p_apr_f)
print_metrics("September 2025", y_true_sep, p_sep_f)

comb_range = 0.5*(nrmse_range(y_true_apr,p_apr_f)+nrmse_range(y_true_sep,p_sep_f))*100
comb_mean  = 0.5*(nrmse_mean(y_true_apr,p_apr_f)+nrmse_mean(y_true_sep,p_sep_f))*100
print(f"\nCombined NRMSE/(max-min): {comb_range:.2f}%")
print(f"Combined NRMSE/mean     : {comb_mean:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 7 — Save CSV + Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 7 — Saving outputs")
print("=" * 60)

out = pd.concat([
    pd.DataFrame({
        'timestamp':    apr_2025['timestamp'],
        'actual_kw':    y_true_apr,
        'forecast_kw':  p_apr_f,
        'error_kw':     p_apr_f - y_true_apr,
        'abs_error_kw': np.abs(p_apr_f - y_true_apr),
        'loss_mse':     (p_apr_f - y_true_apr)**2,
        'month':        'April',
    }),
    pd.DataFrame({
        'timestamp':    sep_2025['timestamp'],
        'actual_kw':    y_true_sep,
        'forecast_kw':  p_sep_f,
        'error_kw':     p_sep_f - y_true_sep,
        'abs_error_kw': np.abs(p_sep_f - y_true_sep),
        'loss_mse':     (p_sep_f - y_true_sep)**2,
        'month':        'September',
    }),
], ignore_index=True)

csv_path = os.path.join(OUTDIR, 'predictions.csv')
out.to_csv(csv_path, index=False)
print(f"Saved predictions.csv — {len(out)} rows")

# Forecast plots
for month_name, feat_df, y_true, y_pred, color in [
    ('April 2025',     apr_2025, y_true_apr, p_apr_f, 'steelblue'),
    ('September 2025', sep_2025, y_true_sep, p_sep_f, 'darkorange'),
]:
    ts = feat_df['timestamp'].values
    fig, axes = plt.subplots(2, 1, figsize=(20, 9),
                             gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(ts, y_true, color=color,    lw=0.9, alpha=0.9, label='Actual')
    axes[0].plot(ts, y_pred, color='tomato', lw=0.9, alpha=0.9, ls='--', label='Forecast')
    axes[0].fill_between(ts, y_true, y_pred, alpha=0.12, color='tomato')
    for d in pd.date_range(ts[0], ts[-1], freq='W-MON'):
        axes[0].axvline(d, color='grey', lw=0.5, ls='--', alpha=0.5)
    axes[0].set_title(f'{month_name} — Actual vs Forecast', fontsize=13)
    axes[0].set_ylabel('Load (kW)'); axes[0].legend()
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    residual = y_pred - y_true
    axes[1].fill_between(ts, residual, 0, where=(residual >= 0),
                         alpha=0.6, color='tomato',    label='Over')
    axes[1].fill_between(ts, residual, 0, where=(residual <  0),
                         alpha=0.6, color='steelblue', label='Under')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('Error (kW)')
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[1].legend()

    plt.suptitle(
        f'LightGBM — {month_name}  |  RMSE: {rmse(y_true,y_pred):.3f} kW  '
        f'|  NRMSE/(max-min): {nrmse_range(y_true,y_pred)*100:.2f}%',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fname = month_name.replace(' ', '_').lower() + '_forecast.png'
    plt.savefig(os.path.join(OUTDIR, fname), dpi=110, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 8 — Extra LightGBM with different seed + Random Forest ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLOCK 8 — Extra LGB seeds + RF Ensemble")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

Xtr_fill = X_train.fillna(0).values
Xvl_fill = X_val.fillna(0).values
X_apr_fill = apr_2025[FEATURES].fillna(0).values
X_sep_fill = sep_2025[FEATURES].fillna(0).values

# Train 2 more LGB models with different seeds for diversity
extra_lgb_preds = []
for seed in [7, 123]:
    ep = {**final_params, 'seed': seed}
    _ds = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    _dv = lgb.Dataset(X_val,   label=y_val,   reference=_ds, free_raw_data=False)
    em = lgb.train(
        ep, _ds, num_boost_round=5000,
        valid_sets=[_dv],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
    )
    extra_lgb_preds.append({
        'apr': np.clip(em.predict(X_apr_fill), 0, None),
        'sep': np.clip(em.predict(X_sep_fill), 0, None),
        'val': np.clip(em.predict(Xvl_fill), 0, None),
    })
    print(f"LGB seed={seed} best round: {em.best_iteration}")

# Random Forest (fast, sklearn)
print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=20,
    max_features=0.5, n_jobs=-1, random_state=42
)
rf.fit(Xtr_fill, y_train.values)
p_apr_rf  = np.clip(rf.predict(X_apr_fill), 0, None)
p_sep_rf  = np.clip(rf.predict(X_sep_fill), 0, None)
pv_rf     = np.clip(rf.predict(Xvl_fill), 0, None)
print_metrics("April 2025 (RF)",     y_true_apr, p_apr_rf)
print_metrics("September 2025 (RF)", y_true_sep, p_sep_rf)
rf_range = 0.5*(nrmse_range(y_true_apr,p_apr_rf)+nrmse_range(y_true_sep,p_sep_rf))*100
print(f"RF Combined NRMSE/(max-min): {rf_range:.2f}%")

# Extra Trees (diversifier)
print("Training Extra Trees...")
et = ExtraTreesRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=20,
    max_features=0.5, n_jobs=-1, random_state=42
)
et.fit(Xtr_fill, y_train.values)
p_apr_et  = np.clip(et.predict(X_apr_fill), 0, None)
p_sep_et  = np.clip(et.predict(X_sep_fill), 0, None)
pv_et     = np.clip(et.predict(Xvl_fill), 0, None)
print_metrics("April 2025 (ET)",     y_true_apr, p_apr_et)
print_metrics("September 2025 (ET)", y_true_sep, p_sep_et)
et_range = 0.5*(nrmse_range(y_true_apr,p_apr_et)+nrmse_range(y_true_sep,p_sep_et))*100
print(f"ET Combined NRMSE/(max-min): {et_range:.2f}%")

# ── Weighted ensemble ─────────────────────────────────────────────────────────
def ensemble_rmse(w, preds_list, y):
    w = np.abs(np.array(w)); w /= w.sum()
    p = sum(wi * pi for wi, pi in zip(w, preds_list))
    return rmse(y, np.clip(p, 0, None))

pv_lgb   = np.clip(model_final.predict(Xvl_fill), 0, None)
pv_lgb7  = extra_lgb_preds[0]['val']
pv_lgb123= extra_lgb_preds[1]['val']

all_pv   = [pv_lgb, pv_lgb7, pv_lgb123, pv_rf, pv_et]
all_apr  = [p_apr_f, extra_lgb_preds[0]['apr'], extra_lgb_preds[1]['apr'], p_apr_rf, p_apr_et]
all_sep  = [p_sep_f, extra_lgb_preds[0]['sep'], extra_lgb_preds[1]['sep'], p_sep_rf, p_sep_et]
n_models = len(all_pv)

from scipy.optimize import minimize as spmin
res = spmin(
    lambda w: ensemble_rmse(w, all_pv, y_val.values),
    [1/n_models]*n_models, method='Nelder-Mead',
    options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7}
)
w_opt = np.abs(res.x); w_opt /= w_opt.sum()
names = ['LGB42', 'LGB7', 'LGB123', 'RF', 'ET']
print("\nOptimal weights:")
for n, w in zip(names, w_opt): print(f"  {n}: {w:.3f}")

def blend(w, apr_list, sep_list):
    w = np.abs(np.array(w, dtype=float)); w /= w.sum()
    return (
        np.clip(sum(wi*pi for wi,pi in zip(w,apr_list)), 0, None),
        np.clip(sum(wi*pi for wi,pi in zip(w,sep_list)), 0, None),
    )

# Simple equal average
p_apr_avg, p_sep_avg = blend([1]*n_models, all_apr, all_sep)
print_metrics("April 2025 (equal avg)",     y_true_apr, p_apr_avg)
print_metrics("September 2025 (equal avg)", y_true_sep, p_sep_avg)
avg_range = 0.5*(nrmse_range(y_true_apr,p_apr_avg)+nrmse_range(y_true_sep,p_sep_avg))*100
print(f"Equal avg Combined NRMSE/(max-min): {avg_range:.2f}%")

# Optimal weighted
p_apr_ens, p_sep_ens = blend(w_opt, all_apr, all_sep)
print_metrics("April 2025 (weighted ens)",     y_true_apr, p_apr_ens)
print_metrics("September 2025 (weighted ens)", y_true_sep, p_sep_ens)
ens_range = 0.5*(nrmse_range(y_true_apr,p_apr_ens)+nrmse_range(y_true_sep,p_sep_ens))*100
print(f"Weighted ens Combined NRMSE/(max-min): {ens_range:.2f}%")

# Pick best
all_results = [
    ('LightGBM (Optuna)', p_apr_f,   p_sep_f,   comb_range),
    ('RF',                p_apr_rf,  p_sep_rf,  rf_range),
    ('ET',                p_apr_et,  p_sep_et,  et_range),
    ('Equal avg',         p_apr_avg, p_sep_avg, avg_range),
    ('Weighted ens',      p_apr_ens, p_sep_ens, ens_range),
]
best_result = min(all_results, key=lambda x: x[3])
print(f"\n*** Best: {best_result[0]}  Combined NRMSE/(max-min): {best_result[3]:.2f}% ***")
p_apr_best, p_sep_best = best_result[1], best_result[2]

# ══════════════════════════════════════════════════════════════════════════════
# Re-save CSV and plots with best predictions
# ══════════════════════════════════════════════════════════════════════════════
out_best = pd.concat([
    pd.DataFrame({
        'timestamp':    apr_2025['timestamp'].values,
        'actual_kw':    y_true_apr,
        'forecast_kw':  p_apr_best,
        'error_kw':     p_apr_best - y_true_apr,
        'abs_error_kw': np.abs(p_apr_best - y_true_apr),
        'loss_mse':     (p_apr_best - y_true_apr)**2,
        'model':        best_result[0],
        'month':        'April',
    }),
    pd.DataFrame({
        'timestamp':    sep_2025['timestamp'].values,
        'actual_kw':    y_true_sep,
        'forecast_kw':  p_sep_best,
        'error_kw':     p_sep_best - y_true_sep,
        'abs_error_kw': np.abs(p_sep_best - y_true_sep),
        'loss_mse':     (p_sep_best - y_true_sep)**2,
        'model':        best_result[0],
        'month':        'September',
    }),
], ignore_index=True)
out_best.to_csv(os.path.join(OUTDIR, 'predictions_best.csv'), index=False)
print(f"\nSaved predictions_best.csv ({best_result[0]})")

for month_name, feat_df, y_true, y_pred, color in [
    ('April 2025',     apr_2025, y_true_apr, p_apr_best, 'steelblue'),
    ('September 2025', sep_2025, y_true_sep, p_sep_best, 'darkorange'),
]:
    ts = feat_df['timestamp'].values
    fig, axes = plt.subplots(2, 1, figsize=(20, 9),
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(ts, y_true, color=color,    lw=0.9, alpha=0.9, label='Actual')
    axes[0].plot(ts, y_pred, color='tomato', lw=0.9, alpha=0.9, ls='--', label='Forecast')
    axes[0].fill_between(ts, y_true, y_pred, alpha=0.12, color='tomato')
    for d in pd.date_range(ts[0], ts[-1], freq='W-MON'):
        axes[0].axvline(d, color='grey', lw=0.5, ls='--', alpha=0.5)
    axes[0].set_title(f'{month_name} — Actual vs Forecast ({best_result[0]})', fontsize=13)
    axes[0].set_ylabel('Load (kW)'); axes[0].legend()
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    residual = y_pred - y_true
    axes[1].fill_between(ts, residual, 0, where=(residual >= 0), alpha=0.6, color='tomato',    label='Over')
    axes[1].fill_between(ts, residual, 0, where=(residual <  0), alpha=0.6, color='steelblue', label='Under')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('Error (kW)')
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[1].legend()
    plt.suptitle(
        f'Best Model ({best_result[0]}) — {month_name}  |  RMSE: {rmse(y_true,y_pred):.3f} kW  '
        f'|  NRMSE/(max-min): {nrmse_range(y_true,y_pred)*100:.2f}%',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fname = 'best_' + month_name.replace(' ', '_').lower() + '_forecast.png'
    plt.savefig(os.path.join(OUTDIR, fname), dpi=110, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

print("\n" + "=" * 60)
print("DONE")
print(f"Best model: {best_result[0]}")
print(f"Combined NRMSE/(max-min): {best_result[3]:.2f}%")
print(f"Outputs in: {OUTDIR}")
print("=" * 60)
