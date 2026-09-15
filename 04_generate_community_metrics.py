"""
AEGIS Component 4 — Community-Risk-Aware Forecasting & Custodianship
Metrics definition + synthetic dashboard data generator

WHAT THIS SIMULATES
--------------------
Unlike the mechanical and pipeline-integrity layers, the community layer is
not primarily a prediction problem — it is a TRACKING and TRUST-BUILDING
problem. This script defines the full metrics taxonomy (documented in
COMMUNITY_METRICS_DICTIONARY.md, generated at the bottom of this script) and
produces a monthly synthetic panel across nine communities spanning the three
governance corridors identified in the demographic research:

  - Ijaw / Wari riverine corridor   : Belema, Soku, Otumara
  - Ogoni / traditional-ruler corridor : Bomu, Norkpo, K-Dere
  - Urhobo-Isoko / CDC corridor     : Ughelli, Otu-Jeremi, Uzere

The synthetic data is built to show a REALISTIC PROGRAM EFFECT: metrics are
generated with a "custodianship program launch" at month 12 of 24, after
which trust, participation, and early-warning metrics improve gradually
while incident and shutdown-risk metrics decline — so the dashboard can
visibly demonstrate the value of the custodianship model over time, exactly
as it would need to for the Value Reinvestment Framework's evidence base.

OUTPUT
------
community_metrics.csv               — monthly panel, all communities
COMMUNITY_METRICS_DICTIONARY.md     — full definition of every tracked metric
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(99)

N_MONTHS = 24
PROGRAM_LAUNCH_MONTH = 12  # Pipeline Guardian / custodianship program begins
START = pd.Timestamp("2025-01-01")

COMMUNITIES = [
    # community, corridor,          governance_channel,        baseline_trust(0-100), baseline_incident_rate
    ("Belema",     "Ijaw-Riverine",  "Amanyanabo/Wari",          38, 0.18),
    ("Soku",       "Ijaw-Riverine",  "Amanyanabo/Wari",          42, 0.15),
    ("Otumara",    "Ijaw-Riverine",  "Amanyanabo/Wari",          30, 0.22),  # live HCDT
                                                                              # board dispute
                                                                              # per research —
                                                                              # lowest baseline trust
    ("Bomu",       "Ogoni-Upland",   "Gberemene/Traditional",    45, 0.12),
    ("Norkpo",     "Ogoni-Upland",   "Gberemene/Traditional",    47, 0.10),
    ("K-Dere",     "Ogoni-Upland",   "Gberemene/Traditional",    44, 0.11),
    ("Ughelli",    "Urhobo-Isoko",   "CDC/Umunna",               55, 0.08),
    ("Otu-Jeremi", "Urhobo-Isoko",   "CDC/Umunna",               52, 0.09),
    ("Uzere",      "Urhobo-Isoko",   "CDC/Umunna",               50, 0.09),
]

records = []

for community, corridor, governance, base_trust, base_incident_rate in COMMUNITIES:

    trust = base_trust
    open_grievances = RNG.integers(2, 6)
    guardians_enrolled = 0
    guardians_active = 0

    for m in range(N_MONTHS):
        period = START + pd.DateOffset(months=m)
        program_active = m >= PROGRAM_LAUNCH_MONTH
        months_since_launch = max(0, m - PROGRAM_LAUNCH_MONTH)

        # --- A. Trust & relationship health --------------------------------
        # Trust drifts slowly; the custodianship program adds a positive
        # trend after launch, with diminishing early gains (sqrt shape) and
        # noise so it doesn't look artificially smooth.
        if program_active:
            trust += RNG.normal(0.9, 0.6) * (1 / np.sqrt(months_since_launch + 1) + 0.3)
        else:
            trust += RNG.normal(0.0, 0.5)
        trust = float(np.clip(trust, 5, 95))

        # Grievances: opened at a steady low rate, resolved faster once the
        # program (with its liaison structure) is active.
        new_grievances = RNG.poisson(1.2)
        resolution_rate = 0.35 if not program_active else 0.65
        resolved = int(open_grievances * resolution_rate)
        open_grievances = max(0, open_grievances + new_grievances - resolved)

        # Otota / traditional-channel formal engagement visits per month
        formal_engagements = RNG.poisson(1.0 if not program_active else 2.4)

        # Sentiment proxy (-1 to 1), derived loosely from trust level + noise
        sentiment_score = np.clip((trust - 50) / 50 + RNG.normal(0, 0.15), -1, 1)

        # --- B. Economic participation ("shareholder" metrics) -------------
        # HCDT disbursement: on-time rate improves once transparency/liaison
        # improvements (bundled with the custodianship program) land.
        hcdt_disbursement_on_time = RNG.random() < (0.45 if not program_active else 0.80)
        hcdt_disbursement_amount_usd = round(RNG.uniform(8_000, 40_000), 0)
        disbursement_transparency_score = round(
            np.clip((55 if not program_active else 82) + RNG.normal(0, 8), 0, 100), 1
        )

        if program_active:
            # Guardian enrollment ramps up over the first several months post-launch
            target_enrolled = min(15, 3 + months_since_launch)
            guardians_enrolled = max(guardians_enrolled, int(target_enrolled))
            attrition = RNG.random() < 0.05
            guardians_active = max(0, guardians_active + (1 if guardians_active < guardians_enrolled else 0)
                                    - (1 if attrition else 0))
        guardian_retention_rate = (guardians_active / guardians_enrolled) if guardians_enrolled else np.nan

        youth_leader_engaged = program_active and RNG.random() < min(0.85, 0.2 + months_since_launch * 0.05)

        # --- C. Security & incident correlation -----------------------------
        # Incident rate declines post-program, reflecting the custodianship
        # thesis; community-reported EARLY WARNINGS rise sharply post-program
        # -- this is the key leading indicator that the community is becoming
        # an active sensor rather than a passive risk.
        incident_rate_this_month = base_incident_rate * (1.0 if not program_active
                                                           else max(0.25, 1 - 0.05 * months_since_launch))
        incident_occurred = RNG.random() < incident_rate_this_month

        community_reported_early_warnings = RNG.poisson(0.3 if not program_active else (0.8 + 0.15 * months_since_launch))
        externally_detected_incidents = RNG.poisson(base_incident_rate * 3 * (1 if not program_active else 0.7))
        total_detected = community_reported_early_warnings + externally_detected_incidents
        pct_community_sourced_detection = round(
            (community_reported_early_warnings / total_detected * 100) if total_detected > 0 else 0, 1
        )

        avg_response_time_hours = round(
            max(1.5, RNG.normal(18 if not program_active else 9, 3)), 1
        )

        # --- D. Risk / shutdown indicators -----------------------------------
        active_ultimatums = 1 if (community == "Otumara" and m in range(18, 22)) else 0
        shutdown_days_this_month = 0
        if incident_occurred:
            shutdown_days_this_month = int(RNG.integers(1, 6)) if not program_active else int(RNG.integers(0, 3))
        if active_ultimatums:
            shutdown_days_this_month += int(RNG.integers(0, 2))

        # --- E. Composite custodianship index ---------------------------------
        # A single 0-100 trackable score combining trust, participation and
        # incident correlation, for at-a-glance dashboard use.
        custodianship_index = round(np.clip(
            0.35 * trust
            + 0.20 * (disbursement_transparency_score)
            + 0.20 * (guardian_retention_rate * 100 if not np.isnan(guardian_retention_rate) else 0)
            + 0.15 * (pct_community_sourced_detection)
            + 0.10 * (100 - min(100, shutdown_days_this_month * 15))
            , 0, 100
        ), 1)

        records.append({
            "community": community,
            "corridor": corridor,
            "governance_channel": governance,
            "period": period,
            "month_index": m,
            "program_active": program_active,
            # A. Trust & relationship health
            "trust_score_0_100": round(trust, 1),
            "open_grievances": open_grievances,
            "grievances_resolved_this_month": resolved,
            "formal_engagements_this_month": formal_engagements,
            "sentiment_score": round(sentiment_score, 3),
            # B. Economic participation
            "hcdt_disbursement_on_time": hcdt_disbursement_on_time,
            "hcdt_disbursement_amount_usd": hcdt_disbursement_amount_usd,
            "disbursement_transparency_score": disbursement_transparency_score,
            "pipeline_guardians_enrolled": guardians_enrolled,
            "pipeline_guardians_active": guardians_active,
            "guardian_retention_rate": round(guardian_retention_rate, 3) if not np.isnan(guardian_retention_rate) else np.nan,
            "youth_leader_program_engaged": youth_leader_engaged,
            # C. Security & incident correlation
            "incident_occurred_this_month": incident_occurred,
            "community_reported_early_warnings": community_reported_early_warnings,
            "externally_detected_incidents": externally_detected_incidents,
            "pct_incidents_community_sourced": pct_community_sourced_detection,
            "avg_incident_response_time_hours": avg_response_time_hours,
            # D. Risk / shutdown indicators
            "active_ultimatum_flag": bool(active_ultimatums),
            "shutdown_days_this_month": shutdown_days_this_month,
            # E. Composite
            "custodianship_index_0_100": custodianship_index,
        })

df = pd.DataFrame(records)
df.to_csv("community_metrics.csv", index=False)

print(f"community_metrics.csv written: {len(df):,} rows, {df['community'].nunique()} communities x {N_MONTHS} months")
print()
print("Custodianship index, before vs after program launch (mean by corridor):")
print(df.groupby(["corridor", "program_active"], observed=True)["custodianship_index_0_100"].mean().round(1))


# ---------------------------------------------------------------------------
# Metrics dictionary — documents every tracked metric for the dashboard team
# ---------------------------------------------------------------------------
METRICS_DOC = """# AEGIS Community Layer — Trackable Metrics Dictionary

This is the full set of metrics tracked for the community custodianship
layer, grouped by category. Every metric below is a column in
`community_metrics.csv` (monthly panel, one row per community per month).

## A. Trust & Relationship Health

| Metric | Definition | Why it matters |
|---|---|---|
| `trust_score_0_100` | Composite proxy for community sentiment toward Renaissance, 0-100 | The single most fundamental leading indicator — most other metrics are downstream of trust |
| `open_grievances` | Count of unresolved formal or informal grievances at month end | A rising count is an early warning of relationship deterioration, often before any incident occurs |
| `grievances_resolved_this_month` | Count of grievances closed in the month | Tracks whether the resolution process itself is functioning, not just whether grievances exist |
| `formal_engagements_this_month` | Count of formal engagement events through proper protocol (e.g. Otota-mediated visits) | Distinguishes genuine relationship investment from ad hoc contact |
| `sentiment_score` | -1 (very negative) to +1 (very positive), derived from liaison log tone | A finer-grained, faster-moving companion to the trust score |

## B. Economic Participation ("Shareholder" Metrics)

| Metric | Definition | Why it matters |
|---|---|---|
| `hcdt_disbursement_on_time` | Whether the month's Host Community Development Trust disbursement was made on schedule | Late disbursement is a specifically documented trigger for community frustration nationally |
| `hcdt_disbursement_amount_usd` | Disbursement amount for the period | Tracks the raw economic flow underpinning the relationship |
| `disbursement_transparency_score` | 0-100 proxy for how visible/auditable the disbursement process is to the community | Directly addresses the "elite capture" and opacity criticism documented in PIA/HCDT implementation research |
| `pipeline_guardians_enrolled` | Cumulative count of community members formally enrolled in the Pipeline Guardian custodian program | Direct measure of the "custodian, not stakeholder" model's uptake |
| `pipeline_guardians_active` | Count currently active (not attrited) | Distinguishes real engagement from nominal enrollment |
| `guardian_retention_rate` | active / enrolled | A low rate signals the program isn't sticking — an operational red flag |
| `youth_leader_program_engaged` | Whether Youth Leaders in this community are actively engaged with the program this month | The single highest-leverage indicator per the demographic research — Youth Leaders control both sabotage and legitimate surveillance mobilization |

## C. Security & Incident Correlation

| Metric | Definition | Why it matters |
|---|---|---|
| `incident_occurred_this_month` | Whether a theft/vandalism/sabotage incident occurred in this community's corridor segment | The outcome the whole layer is ultimately trying to reduce |
| `community_reported_early_warnings` | Count of threats/suspicious activity reported BY the community before an incident occurred | The core "custodian becoming an asset" leading indicator — this should rise over time as the program matures |
| `externally_detected_incidents` | Count of incidents detected by non-community sources (Tantita drones/satellite, DCVG surveys, etc.) | A companion metric — comparing this to community-sourced detection shows the community's growing relative contribution |
| `pct_incidents_community_sourced` | community_reported / (community_reported + externally_detected), as a percentage | The single clearest number for showing the community becoming "the greatest asset to pipeline security" over time |
| `avg_incident_response_time_hours` | Average hours from incident report to response team arrival | Should fall as community-embedded reporting shortens the detection-to-response chain |

## D. Risk / Shutdown Indicators

| Metric | Definition | Why it matters |
|---|---|---|
| `active_ultimatum_flag` | Whether the community currently has an active grievance ultimatum against operations (e.g. the Otumara-style HCDT board dispute) | A direct, near-term production-risk signal that AEGIS's Component 4 forecasting consumes |
| `shutdown_days_this_month` | Days of production shutdown attributable to community action this month | The hard cost metric — ties this layer directly to Challenge 2's production-efficiency objective |

## E. Composite Metric

| Metric | Definition | Why it matters |
|---|---|---|
| `custodianship_index_0_100` | Weighted composite of trust (35%), disbursement transparency (20%), guardian retention (20%), community-sourced detection share (15%), and shutdown-day inverse (10%) | The single number surfaced most prominently on the dashboard's community-risk panel — designed to move visibly in response to real program actions, not just drift randomly |

## How these metrics feed the wider AEGIS dashboard

- `active_ultimatum_flag`, `shutdown_days_this_month`, and `custodianship_index_0_100`
  feed directly into the unified alert feed described in the AEGIS master
  document (Part Two), tagged as "community" risk — distinct from
  "mechanical" (well-testing/instrument health) and "integrity" (pipeline
  risk score) alerts, so planners can tell the three apart at a glance.
- The `corridor` field joins directly to `pipeline_segments.csv`'s
  `corridor` field (via the shared TFP / TNP / Riverine-Connectors naming),
  allowing a combined view of physical pipeline risk and community
  relationship risk for the same geography.
- The pre/post `program_active` split in this synthetic data is exactly the
  kind of before/after evidence the Value Reinvestment Framework (Part
  Three of the master document) needs to justify redirecting verified
  savings into an expanded custodianship program.
"""

with open("COMMUNITY_METRICS_DICTIONARY.md", "w") as f:
    f.write(METRICS_DOC)

print("\nCOMMUNITY_METRICS_DICTIONARY.md written.")
