"""
AEGIS Component 2a — Predictive Instrument Health
Synthetic data generator + anomaly/drift detection logic

WHAT THIS SIMULATES
--------------------
Field instruments (pressure transmitters, temperature transmitters,
differential-pressure transmitters, level transmitters) report a standard
4-20 mA analog signal back to the DCS marshalling cabinet. In a healthy
instrument this signal sits at a stable baseline with small noise. As an
instrument degrades — corroded terminals, fouled sensing element, failing
electronics, all common in the humid, saline Niger Delta environment — the
signal drifts, either slowly (gradual degradation) or suddenly (a hard fault).

This script generates:
  1. instrument_health_data.csv — hourly 4-20mA readings per instrument, with
     a subset of instruments deliberately given a drift-to-failure or a
     sudden-fault pattern partway through the 60-day window, and a ground
     truth "actual_status" column (healthy / degrading / failed) so the
     detection logic below can be scored against reality.
  2. The actual detection functions (rolling z-score + trend-slope drift
     detection) that the dashboard would run against this data.
  3. A demonstration of the Continuous Learning reconciliation loop described
     in the AEGIS master document (Part 2.5): comparing yesterday's flagged
     predictions to the ground truth, tagging why misses happened, and
     showing a simple threshold retrain in response.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

N_HOURS = 60 * 24

# ---------------------------------------------------------------------------
# 1. Define the instrument population
# ---------------------------------------------------------------------------
# Tied to the same wells/flow stations as script 01, plus a few flow-station-
# level instruments (e.g. LACT-adjacent metering, separator level).

INSTRUMENTS = [
    # instrument_id,         type,                 location,        baseline_mA, corrosion_index(0-1)
    ("PT-SOKU-W01-01",   "pressure_transmitter",    "SOKU-W01",      12.0, 0.7),
    ("TT-SOKU-W01-01",   "temperature_transmitter", "SOKU-W01",      10.5, 0.5),
    ("PT-SOKU-W02-01",   "pressure_transmitter",    "SOKU-W02",      11.5, 0.8),
    ("DPT-BELEMA-W01-01","dp_transmitter",          "BELEMA-W01",    13.0, 0.6),
    ("PT-BELEMA-W01-01", "pressure_transmitter",    "BELEMA-W01",    12.2, 0.55),
    ("TT-BELEMA-W02-01", "temperature_transmitter", "BELEMA-W02",    10.8, 0.65),
    ("LT-BELEMA-FS-01",  "level_transmitter",       "Belema-FS",     11.0, 0.4),
    ("PT-BELEMA-W03-01", "pressure_transmitter",    "BELEMA-W03",    12.5, 0.3),
    ("DPT-OTUMARA-W01-01","dp_transmitter",         "OTUMARA-W01",   13.2, 0.35),
    ("PT-OTUMARA-W01-01", "pressure_transmitter",   "OTUMARA-W01",   12.0, 0.3),
    ("TT-OTUMARA-W02-01", "temperature_transmitter","OTUMARA-W02",   10.6, 0.25),
    ("LT-OTUMARA-FS-01",  "level_transmitter",      "Otumara-FS",    11.2, 0.2),
    ("PT-UGHELLI-W01-01", "pressure_transmitter",   "UGHELLI-W01",   12.1, 0.4),
    ("DPT-UGHELLI-W02-01","dp_transmitter",         "UGHELLI-W02",   13.1, 0.6),
    ("TT-UGHELLI-W02-01", "temperature_transmitter","UGHELLI-W02",   10.7, 0.5),
    ("PT-NORKPO-W01-01",  "pressure_transmitter",   "NORKPO-W01",    12.0, 0.45),
    ("LT-NORKPO-FS-01",   "level_transmitter",      "Norkpo-FS",     11.1, 0.3),
    ("DPT-SOKU-FS-01",    "dp_transmitter",         "Soku-FS",       13.0, 0.75),
]

# Signal-noise scale differs slightly by instrument type (realistic: DP
# transmitters are noisier than temperature transmitters)
TYPE_NOISE = {
    "pressure_transmitter": 0.06,
    "temperature_transmitter": 0.03,
    "dp_transmitter": 0.09,
    "level_transmitter": 0.05,
}

# Failure-threshold: how far from baseline (in mA) counts as "failed"
FAILURE_DEVIATION_MA = 2.0

records = []
fault_log = []  # for README documentation of which instruments were seeded with faults

for inst_id, inst_type, location, baseline_ma, corrosion_idx in INSTRUMENTS:

    noise_sd = TYPE_NOISE[inst_type] * (1 + corrosion_idx * 0.3)

    # Probability and style of an injected fault scales with corrosion exposure
    will_fault = RNG.random() < (0.25 + 0.35 * corrosion_idx)
    fault_style = None
    fault_start_hour = None
    drift_rate_per_hour = 0.0

    if will_fault:
        fault_style = RNG.choice(["slow_drift", "sudden_fault"], p=[0.7, 0.3])
        fault_start_hour = int(RNG.integers(int(N_HOURS * 0.3), int(N_HOURS * 0.8)))
        if fault_style == "slow_drift":
            direction = RNG.choice([-1, 1])
            drift_rate_per_hour = direction * RNG.uniform(0.004, 0.015)
        fault_log.append({
            "instrument_id": inst_id, "fault_style": fault_style,
            "fault_start_hour": fault_start_hour,
            "drift_rate_per_hour": round(drift_rate_per_hour, 5)
        })

    signal = baseline_ma
    for h in range(N_HOURS):
        timestamp = pd.Timestamp("2026-06-01") + pd.Timedelta(hours=h)
        actual_status = "healthy"

        if will_fault and h >= fault_start_hour:
            hours_since_fault = h - fault_start_hour
            if fault_style == "slow_drift":
                signal = baseline_ma + drift_rate_per_hour * hours_since_fault
                deviation = abs(signal - baseline_ma)
                if deviation >= FAILURE_DEVIATION_MA:
                    actual_status = "failed"
                elif deviation >= FAILURE_DEVIATION_MA * 0.4:
                    actual_status = "degrading"
            elif fault_style == "sudden_fault":
                # Sudden faults jump immediately and stay faulted (e.g. a short
                # or open circuit), which is a different, easier-to-catch
                # pattern than slow drift.
                signal = baseline_ma + RNG.choice([-1, 1]) * RNG.uniform(3.0, 6.0)
                actual_status = "failed"
        else:
            signal = baseline_ma

        reading_ma = signal + RNG.normal(0, noise_sd)
        reading_ma = np.clip(reading_ma, 3.0, 21.0)

        records.append({
            "instrument_id": inst_id,
            "instrument_type": inst_type,
            "location": location,
            "corrosion_exposure_index": corrosion_idx,
            "timestamp": timestamp,
            "hour_index": h,
            "signal_ma": round(reading_ma, 4),
            "baseline_ma": baseline_ma,
            "actual_status": actual_status,   # ground truth, used only for
                                               # validating the detection logic
                                               # and for the reconciliation demo
        })

df = pd.DataFrame(records)
df.to_csv("instrument_health_data.csv", index=False)
pd.DataFrame(fault_log).to_csv("instrument_health_fault_log.csv", index=False)

print(f"instrument_health_data.csv written: {len(df):,} rows, {df['instrument_id'].nunique()} instruments")
print(f"{len(fault_log)} instruments seeded with a fault pattern (see instrument_health_fault_log.csv)")
print(df["actual_status"].value_counts())
