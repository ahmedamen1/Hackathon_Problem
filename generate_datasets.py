"""
Synthetic dataset generator for Solship Energy AI Hackathon 2026.
Produces dataset_2024.csv (training) and dataset_2025.csv (test)
matching the exact brief schema:
  timestamp, load_kw, pv_kw, buy_price, sell_price, p_battery_kw
"""

import pandas as pd
import numpy as np
import holidays

np.random.seed(42)

DELTA_T = 0.25
BATTERY_CAPACITY_KWH = 16.0
BATTERY_MAX_POWER_KW = 8.0
BATTERY_EFFICIENCY = np.sqrt(0.90)
GRID_LIMIT_KW = 6.0
PV_CAPACITY_KWP = 9.0

italy_hols_2024 = holidays.Italy(years=[2024])
italy_hols_2025 = holidays.Italy(years=[2025])


# ─────────────────────────────────────────────
# BUY PRICE (Italian ToU tariff)
# ─────────────────────────────────────────────
def get_buy_price(ts, hols):
    h = ts.hour
    d = ts.dayofweek  # 0=Mon … 6=Sun
    is_hol = ts.date() in hols
    if is_hol or d == 6:          # F3: holidays + Sundays
        return 0.2440
    if d <= 4 and 8 <= h < 19:    # F1: Mon-Fri peak
        return 0.2540
    if (d <= 4 and (7 <= h < 8 or 19 <= h < 23)) or (d == 5 and 7 <= h < 23):
        return 0.2682             # F2
    return 0.2440                 # F3: night


# ─────────────────────────────────────────────
# SELL PRICE (market-based, noisy fraction of buy)
# ─────────────────────────────────────────────
def get_sell_price(ts, buy_price, rng_day):
    h = ts.hour
    # Sell price peaks midday when PV is abundant (market suppression)
    midday_factor = 0.85 if 10 <= h < 16 else 1.0
    base = buy_price * rng_day.uniform(0.38, 0.52) * midday_factor
    return round(max(0.04, base), 4)


# ─────────────────────────────────────────────
# PV GENERATION PROFILE
# ─────────────────────────────────────────────
def pv_profile(ts, rng_day):
    doy = ts.timetuple().tm_yday
    h = ts.hour + ts.minute / 60.0
    decl = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    lat = 41.9  # Rome latitude
    ha = 15.0 * (h - 12.0)
    cos_z = (np.sin(np.radians(lat)) * np.sin(np.radians(decl)) +
             np.cos(np.radians(lat)) * np.cos(np.radians(decl)) * np.cos(np.radians(ha)))
    cos_z = max(0.0, cos_z)
    # Seasonal cloud cover: clearer in summer
    month = ts.month
    clear_sky = 0.78 + 0.18 * np.sin(np.radians((month - 3) * 30))
    cloud_noise = rng_day.uniform(0.65, 1.0)
    pv = PV_CAPACITY_KWP * cos_z * clear_sky * cloud_noise
    pv += rng_day.normal(0, 0.04 * pv + 0.001)
    return max(0.0, round(pv, 3))


# ─────────────────────────────────────────────
# RESIDENTIAL LOAD PROFILE
# ─────────────────────────────────────────────
def load_profile(ts, hols, rng_step):
    h = ts.hour + ts.minute / 60.0
    d = ts.dayofweek
    m = ts.month
    is_wknd = (ts.date() in hols) or d >= 5

    # Base load
    base = 1.1

    # Daily shape
    if 0 <= h < 6:
        base += 0.0
    elif 6 <= h < 8:
        base += 1.2 * ((h - 6) / 2)
    elif 8 <= h < 9:
        base += 1.5
    elif 9 <= h < 12:
        base += 0.6
    elif 12 <= h < 14:
        base += 1.4 * np.sin(np.radians((h - 12) / 2 * 180))
    elif 14 <= h < 17:
        base += 0.4
    elif 17 <= h < 20:
        base += 2.8 * np.sin(np.radians((h - 17) / 3 * 180))
    elif 20 <= h < 22:
        base += 1.8
    elif 22 <= h < 24:
        base += 0.8 * (1 - (h - 22) / 2)

    # Weekend boost
    if is_wknd:
        if 10 <= h < 14:
            base += 0.7
        if 19 <= h < 22:
            base += 0.5

    # Seasonal: heating (winter) + cooling (summer)
    seasonal = (0.4 * np.cos(np.radians((m - 1) * 30)) +      # winter peak
                0.25 * np.cos(np.radians((m - 7) * 30)))       # summer AC
    base += seasonal

    base += rng_step.normal(0, 0.12)
    return max(0.05, round(base, 3))


# ─────────────────────────────────────────────
# RULE-BASED BATTERY CONTROLLER (baseline A)
# ─────────────────────────────────────────────
def simulate_battery(timestamps, loads, pvs, buy_prices):
    soc = 0.5
    p_battery = []
    for i, ts in enumerate(timestamps):
        load = loads[i]
        pv = pvs[i]
        bp = buy_prices[i]
        net = load - pv  # positive = need from grid/battery

        # Strategy: charge during cheap F3, discharge during expensive F1
        if bp == 0.2540 and soc > 0.15:          # F1 peak → discharge
            want_discharge = min(net, BATTERY_MAX_POWER_KW)
            want_discharge = max(0, want_discharge)
            max_from_bat = soc * BATTERY_CAPACITY_KWH * BATTERY_EFFICIENCY / DELTA_T
            p = min(want_discharge, max_from_bat, BATTERY_MAX_POWER_KW)
        elif bp <= 0.2440 and soc < 0.88:          # F3 cheap → charge
            headroom = (0.88 - soc) * BATTERY_CAPACITY_KWH
            p = -min(BATTERY_MAX_POWER_KW, headroom / (DELTA_T * BATTERY_EFFICIENCY))
        else:
            p = 0.0

        # Update SoC
        if p < 0:  # charging
            soc += abs(p) * DELTA_T * BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        else:       # discharging
            soc -= p * DELTA_T / BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
        soc = np.clip(soc, 0.0, 1.0)
        p_battery.append(round(p, 4))
    return p_battery


# ─────────────────────────────────────────────
# GENERATE ONE YEAR OF DATA
# ─────────────────────────────────────────────
def generate_year(year, hols, corrupt_battery_window=None):
    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:45",
        freq="15min"
    )
    n = len(timestamps)

    loads = []
    pvs = []
    buy_prices = []
    sell_prices = []

    prev_doy = -1
    rng_day = np.random.RandomState(year * 1000)
    rng_step = np.random.RandomState(year * 999)

    for ts in timestamps:
        doy = ts.timetuple().tm_yday
        if doy != prev_doy:
            rng_day = np.random.RandomState(year * 1000 + doy)
            prev_doy = doy

        pv = pv_profile(ts, rng_day)
        load = load_profile(ts, hols, rng_step)
        bp = get_buy_price(ts, hols)
        sp = get_sell_price(ts, bp, rng_day)

        pvs.append(pv)
        loads.append(load)
        buy_prices.append(bp)
        sell_prices.append(sp)

    # Simulate battery controller
    p_battery = simulate_battery(timestamps, loads, pvs, buy_prices)

    # Inject corruption in 2025 (as the brief mandates)
    if corrupt_battery_window is not None:
        start_corrupt, end_corrupt = corrupt_battery_window
        mask = (timestamps >= start_corrupt) & (timestamps <= end_corrupt)
        idx = np.where(mask)[0]
        # Random sign-flips and scale errors to make SoC inconsistent
        rng_corrupt = np.random.RandomState(2025)
        for i in idx:
            if rng_corrupt.rand() < 0.4:
                p_battery[i] = round(rng_corrupt.uniform(-8, 8), 3)
            elif rng_corrupt.rand() < 0.3:
                p_battery[i] = -p_battery[i]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_kw": loads,
        "pv_kw": pvs,
        "buy_price": buy_prices,
        "sell_price": sell_prices,
        "p_battery_kw": p_battery,
    })
    return df


print("Generating 2024 dataset (35040 rows)...")
df_2024 = generate_year(2024, italy_hols_2024)
print(f"  Shape: {df_2024.shape}")
print(f"  Load range: {df_2024['load_kw'].min():.2f} – {df_2024['load_kw'].max():.2f} kW")
print(f"  PV range:   {df_2024['pv_kw'].min():.2f} – {df_2024['pv_kw'].max():.2f} kW")

print("\nGenerating 2025 dataset (35040 rows, with battery corruption window)...")
# Corrupt roughly weeks 6-9 of 2025 (mid-Feb to mid-Mar)
corrupt_start = pd.Timestamp("2025-02-10")
corrupt_end   = pd.Timestamp("2025-03-08")
df_2025 = generate_year(2025, italy_hols_2025, corrupt_battery_window=(corrupt_start, corrupt_end))
print(f"  Shape: {df_2025.shape}")
print(f"  Load range: {df_2025['load_kw'].min():.2f} – {df_2025['load_kw'].max():.2f} kW")
print(f"  PV range:   {df_2025['pv_kw'].min():.2f} – {df_2025['pv_kw'].max():.2f} kW")

df_2024.to_csv("d:/Solship/dataset_2024.csv", index=False)
df_2025.to_csv("d:/Solship/dataset_2025.csv", index=False)

print("\nSaved dataset_2024.csv and dataset_2025.csv")
print("\nColumn check 2024:", list(df_2024.columns))
print("Column check 2025:", list(df_2025.columns))
print("\nHead 2024:")
print(df_2024.head(3).to_string())
print("\nHead 2025:")
print(df_2025.head(3).to_string())
