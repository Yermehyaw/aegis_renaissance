"""
AEGIS Component 2b — Pipeline Segment Risk Scoring
Synthetic data generator

WHAT THIS SIMULATES
--------------------
Renaissance's pipeline network (based on the operational research: the Trans
Forcados Pipeline, Trans-Niger Pipeline, and connecting flowlines) is broken
into discrete monitored segments. Each segment has static structural
attributes (age, DCVG survey defect count, cathodic protection reading,
historical incidents, clamp count, throughput, whether it has been
geo-verified under the Phase 0 legacy-asset reconciliation effort) that feed
the weighted risk-scoring formula in reconciliation_logic.py.

This script generates a MONTHLY panel (segment x month) over 24 months so the
dashboard can show risk trending over time, and includes a synthetic
"actual_incident_occurred" ground-truth flag for each segment-month so the
risk score's predictive validity can be backtested and shown in the pitch
(e.g. "segments scored high-risk had an incident rate N times higher than
segments scored low-risk").

OUTPUT
------
pipeline_segments.csv — one row per segment per month
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(21)

N_MONTHS = 24
START = pd.Timestamp("2025-01-01")

# ---------------------------------------------------------------------------
# 1. Define pipeline segments across the two major corridors + connectors
# ---------------------------------------------------------------------------
# corridor governance types tie directly into the community layer (script 04)
# so the two datasets can be joined by `corridor` for a combined risk view.

CORRIDORS = [
    # corridor_name, governance_type,     terrain,     n_segments, base_throughput_bpd
    ("TFP",          "urhobo_isoko_cdc",  "swamp",     14, 220000),
    ("TNP",          "ogoni_traditional", "upland",    16, 180000),
    ("Riverine-Connectors", "ijaw_wari",  "riverine",  12,  60000),
]

records = []
segment_meta = []

seg_counter = 1
for corridor, governance, terrain, n_segments, base_throughput in CORRIDORS:
    for s in range(n_segments):
        segment_id = f"SEG-{seg_counter:03d}"
        seg_counter += 1

        install_year = RNG.integers(1965, 2018)  # consistent with 1960s-vintage
                                                    # TFP infrastructure noted
                                                    # in the operational research
        clamp_count = max(0, int(RNG.poisson(3) + (2026 - install_year) / 15))
        geo_verified = RNG.random() > 0.35  # ~35% start unverified — the
                                             # Phase 0 reconciliation backlog
        throughput_bpd = max(5000, base_throughput / n_segments * RNG.uniform(0.6, 1.4))
        distance_to_community_km = round(RNG.uniform(0.1, 8.0), 2)

        # Baseline incident propensity: older, more clamped, less-verified,
        # closer-to-community segments carry structurally higher risk. This
        # baseline propensity is what the risk model is trying to recover
        # from the observable features below.
        age = 2026 - install_year
        true_propensity = (
            0.015
            + 0.010 * (age / 60)
            + 0.020 * min(clamp_count / 15, 1.0)
            + (0.03 if not geo_verified else 0)
            + 0.015 * max(0, (3.0 - distance_to_community_km) / 3.0)
        )
        true_propensity = np.clip(true_propensity, 0.01, 0.25)

        segment_meta.append({
            "segment_id": segment_id, "corridor": corridor, "governance_type": governance,
            "terrain": terrain, "install_year": install_year, "clamp_count": clamp_count,
            "geo_verified": geo_verified, "throughput_bpd": round(throughput_bpd),
            "distance_to_community_km": distance_to_community_km,
            "true_incident_propensity_monthly": round(true_propensity, 4),
        })

        cumulative_incidents = 0
        cp_reading = RNG.uniform(-1100, -750)  # mV; more negative = better
                                                 # cathodic protection in this
                                                 # synthetic convention

        for m in range(N_MONTHS):
            period = START + pd.DateOffset(months=m)
            age_years = 2026 - install_year

            # DCVG survey defect count drifts slowly upward between surveys,
            # with periodic surveys (every ~6 months) partially resetting it
            # after remediation work.
            if m % 6 == 0:
                dcvg_defects = max(0, int(RNG.poisson(1) + clamp_count * 0.3))
            else:
                dcvg_defects = max(0, int(dcvg_defects + RNG.poisson(0.4)))

            # Cathodic protection reading drifts mildly month to month
            cp_reading += RNG.normal(0, 15)
            cp_reading = float(np.clip(cp_reading, -1200, -600))

            # Historical incident count (trailing 3-year window, so it moves
            # slowly and reflects accumulated history, not just this month)
            incident_this_month = RNG.random() < true_propensity
            if incident_this_month:
                cumulative_incidents += 1

            historical_incident_count_3yr = cumulative_incidents

            records.append({
                "segment_id": segment_id,
                "corridor": corridor,
                "governance_type": governance,
                "terrain": terrain,
                "period": period,
                "month_index": m,
                "age_years": age_years,
                "dcvg_defect_count": dcvg_defects,
                "cathodic_protection_reading_mv": round(cp_reading, 1),
                "historical_incident_count_3yr": historical_incident_count_3yr,
                "throughput_bpd": round(throughput_bpd),
                "clamp_count": clamp_count,
                "geo_verified": geo_verified,
                "distance_to_community_km": distance_to_community_km,
                "actual_incident_occurred_this_month": incident_this_month,
            })

df = pd.DataFrame(records)
df.to_csv("pipeline_segments.csv", index=False)
pd.DataFrame(segment_meta).to_csv("pipeline_segments_static_meta.csv", index=False)

print(f"pipeline_segments.csv written: {len(df):,} rows, "
      f"{df['segment_id'].nunique()} segments x {N_MONTHS} months")
print(f"Total incidents across all segment-months: {df['actual_incident_occurred_this_month'].sum()}")
