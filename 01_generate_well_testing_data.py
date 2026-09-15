"""
AEGIS Component 1 — Continuous Virtual Well Testing
Synthetic data generator

WHAT THIS SIMULATES
--------------------
Renaissance's Delta-V DCS continuously reads wellhead pressure, temperature,
differential pressure and other proxies every hour. Separately, a physical
Maximum Efficient Rate (MER) test — routing the well through a test separator
— happens only once every 24 hours and gives the "ground truth" oil/gas/water
split for that day.

The virtual well-testing model is trained to predict the MER result from the
continuous DCS signals, so production can be estimated hourly instead of only
once a day.

This script generates:
  - Hourly continuous DCS sensor readings for every well (the model's INPUT
    features), available at every timestamp.
  - Daily MER test-separator ground-truth readings (the model's TRAINING
    TARGET), available only once every 24 hours, exactly as it is in reality.

OUTPUT
------
well_testing_data.csv  — one row per well per hour, with MER columns populated
                          only on test hours and left blank (NaN) otherwise.
                          This blank-on-non-test-hours structure is intentional:
                          it is exactly the gap the virtual testing model exists
                          to fill.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Define the well population
# ---------------------------------------------------------------------------
# A mix of "legacy reactivated" wells (higher, rising water cut — consistent
# with assets like Soku/Belema described in the operational research) and
# "standard" wells with more stable production.

WELLS = [
    # well_id,        flow_station,   legacy_reactivated, base_oil_bpd, base_GOR_scf_bbl, base_water_cut
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

N_HOURS = 60 * 24          # 60 days of hourly data
MER_TEST_INTERVAL_H = 24   # one physical test per well per 24 hours
MER_TEST_START_HOUR = 6    # tests run starting at 06:00 each day

records = []

for well_id, flow_station, is_legacy, base_oil, base_gor, base_wc in WELLS:

    # Per-well random "personality" so wells aren't identical
    decline_rate = RNG.uniform(0.0006, 0.0015) if not is_legacy else RNG.uniform(0.0015, 0.0030)
    wc_drift_per_day = RNG.uniform(0.0005, 0.0015) if is_legacy else RNG.uniform(0.0000, 0.0004)
    noise_scale = RNG.uniform(0.03, 0.06)

    # Choke changes: a handful of operator interventions over 60 days
    n_choke_events = RNG.integers(2, 6)
    choke_event_hours = np.sort(RNG.choice(np.arange(24, N_HOURS - 24), size=n_choke_events, replace=False))
    choke_base = RNG.integers(28, 48)   # 64ths of an inch
    choke_series = np.full(N_HOURS, choke_base, dtype=float)
    current_choke = choke_base
    for h in range(N_HOURS):
        if h in choke_event_hours:
            current_choke = np.clip(current_choke + RNG.integers(-6, 7), 16, 64)
        choke_series[h] = current_choke

    for h in range(N_HOURS):
        day = h // 24
        hour_of_day = h % 24
        timestamp = pd.Timestamp("2026-06-01") + pd.Timedelta(hours=h)

        # --- true underlying production (not directly observed by DCS) ---
        decline_factor = np.exp(-decline_rate * day)
        true_water_cut = np.clip(base_wc + wc_drift_per_day * day + RNG.normal(0, 0.01), 0.02, 0.93)
        true_oil_rate_bpd = max(50.0, base_oil * decline_factor * (1 + RNG.normal(0, noise_scale)))
        true_gor = base_gor * (1 + RNG.normal(0, 0.04))
        true_gas_rate_mscfd = true_oil_rate_bpd * true_gor / 1000.0
        true_water_rate_bpd = true_oil_rate_bpd * true_water_cut / (1 - true_water_cut)
        true_liquid_rate_bpd = true_oil_rate_bpd + true_water_rate_bpd

        choke_64th = choke_series[h]
        choke_frac = choke_64th / 64.0

        # --- DCS-visible continuous sensor proxies (features) ---
        # Differential pressure across the choke/orifice ~ roughly proportional
        # to the square of liquid rate, scaled down by choke opening, plus noise.
        differential_pressure_inH2O = (
            1.0e-5 * (true_liquid_rate_bpd ** 2) / (choke_frac ** 1.5 + 0.05)
            + RNG.normal(0, 8)
        )
        differential_pressure_inH2O = max(5.0, differential_pressure_inH2O)

        # Wellhead pressure drops slightly as liquid rate rises (drawdown proxy)
        wellhead_pressure_psi = max(
            150.0,
            1450 - 0.09 * true_liquid_rate_bpd + RNG.normal(0, 15)
        )

        # Flowline pressure downstream of choke
        flowline_pressure_psi = max(40.0, wellhead_pressure_psi * (0.55 + 0.25 * choke_frac) + RNG.normal(0, 8))

        # Wellhead temperature — mild diurnal swamp-ambient variation + noise
        wellhead_temperature_F = 158 + 3 * np.sin(2 * np.pi * hour_of_day / 24) + RNG.normal(0, 1.5)

        # Vibration index — mostly noise; will later correlate with instrument
        # health issues for a subset of wells (handled in script 02)
        vibration_index = max(0.0, RNG.normal(2.0, 0.4))

        # Gas-oil-ratio sensor estimate (noisier, cheaper proxy than lab test)
        gor_sensor_scf_bbl = max(50.0, true_gor + RNG.normal(0, 25))

        # Water-cut sensor estimate (e.g. a capacitance probe) — noisier and
        # slightly biased low, which is realistic and part of why the physical
        # MER test remains the ground truth.
        water_cut_sensor_pct = np.clip((true_water_cut - RNG.normal(0.015, 0.01)) * 100, 0, 100)

        row = {
            "well_id": well_id,
            "flow_station": flow_station,
            "timestamp": timestamp,
            "hour_index": h,
            "choke_size_64th": choke_64th,
            "wellhead_pressure_psi": round(wellhead_pressure_psi, 1),
            "flowline_pressure_psi": round(flowline_pressure_psi, 1),
            "wellhead_temperature_F": round(wellhead_temperature_F, 1),
            "differential_pressure_inH2O": round(differential_pressure_inH2O, 1),
            "vibration_index": round(vibration_index, 3),
            "gor_sensor_scf_bbl": round(gor_sensor_scf_bbl, 1),
            "water_cut_sensor_pct": round(water_cut_sensor_pct, 2),
            "is_legacy_reactivated_well": is_legacy,
            # --- MER test-separator ground truth: populated ONLY on test hours ---
            "is_mer_test_hour": False,
            "mer_oil_bbl_24hr": np.nan,
            "mer_gas_mscf_24hr": np.nan,
            "mer_water_bbl_24hr": np.nan,
            "mer_water_cut_pct": np.nan,
        }

        # Physical test happens once every 24h, at MER_TEST_START_HOUR local hour
        if hour_of_day == MER_TEST_START_HOUR:
            # Measurement noise on the physical test itself (~2-4%), representing
            # real test-separator measurement uncertainty — this is why even the
            # "ground truth" isn't perfect, but it's the best available reference.
            meas_noise = RNG.normal(1.0, 0.02)
            row["is_mer_test_hour"] = True
            row["mer_oil_bbl_24hr"] = round(true_oil_rate_bpd * meas_noise, 1)
            row["mer_gas_mscf_24hr"] = round(true_gas_rate_mscfd * meas_noise, 1)
            row["mer_water_bbl_24hr"] = round(true_water_rate_bpd * meas_noise, 1)
            row["mer_water_cut_pct"] = round(true_water_cut * 100, 2)

        records.append(row)

df = pd.DataFrame(records)
df.to_csv("well_testing_data.csv", index=False)

print(f"well_testing_data.csv written: {len(df):,} rows, {df['well_id'].nunique()} wells, "
      f"{df['is_mer_test_hour'].sum()} MER test readings "
      f"({df['is_mer_test_hour'].sum() / df['well_id'].nunique()} tests/well)")
