"""
AEGIS Component 1 — Continuous Virtual Well Testing
Synthetic data generator (Version 2.0 - Physics-Informed)

WHAT THIS SIMULATES
--------------------
Renaissance's DCS continuously reads surface variables (WHP, WHT, DP) hourly.
Separately, a routine production well test (test separator) occurs once every 24 hours.

Upgrades in v2.0:
- Physics-grounded WHP: WHP is derived from Reservoir Pressure, IPR, and VLP.
- Hydrostatic penalty: As water cut rises, the mixture gets heavier, increasing tubing
  pressure drop and lowering WHP natively.
- Decoupled measurement noise: The test separator adds independent noise to oil, water,
  and gas measurements. Water cut is mathematically derived from these noisy volumes.
- Target leaks removed: gor_sensor_scf_bbl and water_cut_sensor_pct are completely axed.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# 1. Define the well population
WELLS = [
    # well_id,        flow_station,   legacy, base_oil_bpd, base_GOR, base_water_cut
    ("SOKU-W01",     "Soku-FS",       True,   1800, 620, 0.52),
    ("SOKU-W02",     "Soku-FS",       True,   1200, 580, 0.61),
    ("BELEMA-W01",   "Belema-FS",     True,   2200, 700, 0.44),
    ("BELEMA-W02",   "Belema-FS",     True,   1500, 650, 0.58),
    ("BELEMA-W03",   "Belema-FS",     False,  2600, 480, 0.18),
    ("OTUMARA-W01",  "Otumara-FS",    False,  3000, 510, 0.15),
    ("OTUMARA-W02",  "Otumara-FS",    False,  2100, 530, 0.22),
    ("UGHELLI-W01",  "Ughelli-FS",    False,  2800, 460, 0.20),
    ("UGHELLI-W02",  "Ughelli-FS",    True,   1400, 610, 0.55),
    ("NORKPO-W01",   "Norkpo-FS",     False,  1900, 540, 0.30),
]

N_HOURS = 60 * 24          # 60 days
TEST_INTERVAL_H = 24       # Physical test every 24h
TEST_START_HOUR = 6        # Tests at 06:00

records = []

for well_id, flow_station, is_legacy, base_oil, base_gor, base_wc in WELLS:
    
    # Reservoir parameters
    decline_rate = RNG.uniform(0.0006, 0.0015) if not is_legacy else RNG.uniform(0.0015, 0.0030)
    wc_drift_per_day = RNG.uniform(0.0005, 0.0015) if is_legacy else RNG.uniform(0.0001, 0.0004)
    initial_reservoir_pressure = RNG.uniform(3200, 3800)
    productivity_index = (base_oil / 1000.0) * RNG.uniform(0.9, 1.3)
    well_depth_ft = RNG.uniform(5500, 7500)

    # Choke changes over 60 days
    n_choke_events = RNG.integers(2, 6)
    choke_event_hours = np.sort(RNG.choice(np.arange(24, N_HOURS - 24), size=n_choke_events, replace=False))
    current_choke = RNG.integers(28, 48)
    choke_series = np.full(N_HOURS, current_choke, dtype=float)
    
    for h in range(N_HOURS):
        if h in choke_event_hours:
            current_choke = np.clip(current_choke + RNG.integers(-6, 7), 16, 64)
        choke_series[h] = current_choke

    for h in range(N_HOURS):
        day = h // 24
        hour_of_day = h % 24
        timestamp = pd.Timestamp("2026-06-01") + pd.Timedelta(hours=h)

        # --- TRUE RESERVOIR STATE (Hidden from model) ---
        decline_factor = np.exp(-decline_rate * day)
        current_res_pressure = initial_reservoir_pressure * decline_factor
        
        true_water_cut = np.clip(base_wc + (wc_drift_per_day * day) + RNG.normal(0, 0.005), 0.02, 0.95)
        true_gor = base_gor * (1 + RNG.normal(0, 0.02))
        
        true_oil_rate = max(50.0, base_oil * decline_factor * (1 + RNG.normal(0, 0.02)))
        true_gas_rate = true_oil_rate * true_gor / 1000.0
        true_water_rate = true_oil_rate * true_water_cut / (1 - true_water_cut)
        true_liquid_rate = true_oil_rate + true_water_rate

        choke_64th = choke_series[h]

        # --- PHYSICS INJECTION: IPR & VLP PROXY ---
        # 1. Flowing Bottom-Hole Pressure (IPR)
        pwf = current_res_pressure - (true_liquid_rate / productivity_index)
        
        # 2. Hydrostatic Tubing Drop (VLP)
        # As WC rises, mixture SG rises, increasing pressure drop in tubing
        sg_oil, sg_water = 0.85, 1.05
        sg_mixture = (true_water_cut * sg_water) + ((1 - true_water_cut) * sg_oil)
        hydrostatic_drop = 0.433 * well_depth_ft * sg_mixture
        
        # 3. Surface Pressures
        base_whp = pwf - hydrostatic_drop
        choke_backpressure = (64.0 / choke_64th)**1.5 * RNG.uniform(40, 60)
        
        wellhead_pressure_psi = max(100.0, base_whp + choke_backpressure + RNG.normal(0, 8))
        flowline_pressure_psi = max(40.0, wellhead_pressure_psi * 0.45 + RNG.normal(0, 5))
        differential_pressure_psi = wellhead_pressure_psi - flowline_pressure_psi
        wellhead_temperature_F = 158 + 3 * np.sin(2 * np.pi * hour_of_day / 24) + RNG.normal(0, 1.0)
        vibration_index = max(0.0, RNG.normal(2.0, 0.4))

        # --- CONSTRUCT ROW ---
        row = {
            "well_id": well_id,
            "flow_station": flow_station,
            "timestamp": timestamp,
            "choke_size_64th": choke_64th,
            "wellhead_pressure_psi": round(wellhead_pressure_psi, 1),
            "flowline_pressure_psi": round(flowline_pressure_psi, 1),
            "differential_pressure_psi": round(differential_pressure_psi, 1),
            "wellhead_temperature_F": round(wellhead_temperature_F, 1),
            "vibration_index": round(vibration_index, 3),
            "is_legacy_reactivated_well": is_legacy,
            
            # Ground truth testing columns (Empty by default)
            "is_production_test_hour": False,
            "test_oil_bpd": np.nan,
            "test_gas_mscfd": np.nan,
            "test_water_bpd": np.nan,
            "test_liquid_bpd": np.nan,
            "test_water_cut_pct": np.nan,
        }

        # --- ROUTINE PRODUCTION TEST (Once per 24h) ---
        if hour_of_day == TEST_START_HOUR:
            # Independent error distributions per phase
            meas_oil = max(0.0, true_oil_rate * RNG.normal(1.0, 0.025))
            meas_gas = max(0.0, true_gas_rate * RNG.normal(1.0, 0.040))
            meas_water = max(0.0, true_water_rate * RNG.normal(1.0, 0.035))
            meas_liquid = meas_oil + meas_water
            meas_wc_pct = (meas_water / meas_liquid * 100.0) if meas_liquid > 0 else 0.0

            row["is_production_test_hour"] = True
            row["test_oil_bpd"] = round(meas_oil, 1)
            row["test_gas_mscfd"] = round(meas_gas, 1)
            row["test_water_bpd"] = round(meas_water, 1)
            row["test_liquid_bpd"] = round(meas_liquid, 1)
            row["test_water_cut_pct"] = round(meas_wc_pct, 2)

        records.append(row)

df = pd.DataFrame(records)
df.to_csv("aegis_data_v2.csv", index=False)

print(f"aegis_data_v2.csv written: {len(df):,} rows.")
print(f"Total production tests simulated: {df['is_production_test_hour'].sum()}")