# AEGIS Community Layer — Trackable Metrics Dictionary

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
