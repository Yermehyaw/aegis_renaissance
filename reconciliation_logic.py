"""
AEGIS — Detection Logic + Continuous Learning Reconciliation Engine

This module contains the actual scoring/detection functions the dashboard
would run, plus the daily reconciliation loop described in the AEGIS master
document (Part 2.5): compare yesterday's predictions to what actually
happened, tag why misses occurred, detect sustained drift, and demonstrate a
simple threshold retrain.

Three things live here:
  1. detect_instrument_health()   — rolling z-score + trend-slope drift logic
  2. score_pipeline_risk()        — weighted pipeline segment risk score
  3. ReconciliationEngine         — shared continuous-learning loop used by
                                     both models
"""

import numpy as np
import pandas as pd


# ===========================================================================
# 1. INSTRUMENT HEALTH DETECTION LOGIC
# ===========================================================================

def detect_instrument_health(
    df: pd.DataFrame,
    rolling_window_h: int = 24,
    drift_window_h: int = 48,
    z_threshold: float = 3.0,
    drift_slope_threshold: float = 0.006,
    sustained_hours_required: int = 3,
) -> pd.DataFrame:
    """
    Runs two complementary detectors per instrument, because they catch
    different failure modes:

      (a) DEVIATION detector (rolling z-score): catches SUDDEN faults well —
          a signal that jumps several mA away from its own recent baseline
          in one step stands out immediately.

      (b) DRIFT detector (rolling trend slope): catches SLOW faults that a
          pure deviation check misses — a signal creeping away from baseline
          at (say) 0.01 mA/hour stays within normal noise bounds for a long
          time, so we instead fit a linear trend over a trailing window and
          flag when the SLOPE itself is sustained and directionally
          consistent, not just when the current value looks unusual.

    A flag only fires after `sustained_hours_required` consecutive hours of
    the same signal, specifically to avoid single-point noise spikes being
    mistaken for real degradation (a false-positive control).

    Returns the input dataframe with three new columns:
      - z_score            : rolling deviation score
      - drift_slope        : rolling trend slope (mA/hour)
      - predicted_status   : 'healthy' | 'watch' | 'degrading' | 'failed'
    """
    out = []
    for inst_id, g in df.sort_values("hour_index").groupby("instrument_id"):
        g = g.copy()
        baseline_mean = g["signal_ma"].iloc[:rolling_window_h].mean()
        baseline_std = max(g["signal_ma"].iloc[:rolling_window_h].std(), 0.01)

        g["z_score"] = (g["signal_ma"] - baseline_mean).abs() / baseline_std

        # Rolling slope: fit a line to the trailing `drift_window_h` readings
        # and take its slope. A sustained, consistent slope (not noise, which
        # averages toward zero) indicates real drift.
        slopes = np.full(len(g), np.nan)
        values = g["signal_ma"].to_numpy()
        for i in range(drift_window_h, len(g)):
            window = values[i - drift_window_h:i]
            x = np.arange(drift_window_h)
            slope = np.polyfit(x, window, 1)[0]
            slopes[i] = slope
        g["drift_slope"] = slopes

        # Sustained-hours logic: flag only if the deviation or drift condition
        # has held for `sustained_hours_required` consecutive hours.
        dev_flag = (g["z_score"] > z_threshold).astype(int)
        drift_flag = (g["drift_slope"].abs() > drift_slope_threshold).astype(int)

        dev_sustained = dev_flag.rolling(sustained_hours_required).sum() >= sustained_hours_required
        drift_sustained = drift_flag.rolling(sustained_hours_required).sum() >= sustained_hours_required

        status = np.where(
            dev_sustained & (g["z_score"] > z_threshold * 1.6), "failed",
            np.where(dev_sustained | drift_sustained, "degrading",
            np.where(drift_flag.astype(bool), "watch", "healthy"))
        )
        g["predicted_status"] = status
        out.append(g)

    return pd.concat(out, ignore_index=True)


# ===========================================================================
# 2. PIPELINE RISK SCORING LOGIC
# ===========================================================================

# Weights sum to 1.0. Each input factor is normalized 0-1 before weighting.
# Weights reflect the priority ordering discussed in the AEGIS design:
# structural/historical risk factors carry more weight than throughput alone,
# since the goal is prioritizing PREVENTION, not just protecting high-value
# segments.
RISK_WEIGHTS = {
    "age_years":                 0.16,
    "dcvg_defect_count":         0.18,
    "cathodic_protection_gap":   0.12,   # higher gap = worse protection
    "historical_incident_count": 0.22,
    "throughput_bpd":            0.10,
    "clamp_count":               0.14,   # legacy "clamp culture" indicator
    "unverified_asset":          0.08,   # 1 if not yet geo-verified (Phase 0)
}


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def score_pipeline_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a 0-1 risk_score per pipeline segment-month from normalized,
    weighted structural and historical factors. Also returns a
    risk_category label for dashboard color-coding.
    """
    out = df.copy()

    # Cathodic protection reading: LOWER mV magnitude = WORSE protection in
    # this synthetic convention, so we invert it before normalizing.
    cp_gap = out["cathodic_protection_reading_mv"].max() - out["cathodic_protection_reading_mv"]

    normalized = pd.DataFrame({
        "age_years":                 _normalize(out["age_years"]),
        "dcvg_defect_count":         _normalize(out["dcvg_defect_count"]),
        "cathodic_protection_gap":   _normalize(cp_gap),
        "historical_incident_count": _normalize(out["historical_incident_count_3yr"]),
        "throughput_bpd":            _normalize(out["throughput_bpd"]),
        "clamp_count":               _normalize(out["clamp_count"]),
        "unverified_asset":          (~out["geo_verified"]).astype(float),
    })

    risk_score = sum(normalized[col] * w for col, w in RISK_WEIGHTS.items())
    out["risk_score"] = risk_score.round(4)
    out["risk_category"] = pd.cut(
        out["risk_score"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["low", "moderate", "elevated", "critical"],
    )
    return out


# ===========================================================================
# 3. CONTINUOUS LEARNING RECONCILIATION ENGINE (shared by both models)
# ===========================================================================

class ReconciliationEngine:
    """
    Implements the daily reconciliation loop described in the AEGIS master
    document (Part 2.5):

      1. Log what the model predicted.
      2. Compare it, once the outcome window has closed, to what actually
         happened.
      3. Classify each prediction as a true/false positive/negative.
      4. Tag WHY misses happened (a lightweight rule-based tagger here; in
         production this would draw on richer contextual metadata).
      5. Track a rolling accuracy trend and flag sustained drift.
      6. Demonstrate a simple threshold retrain in response to a run of
         misses of the same type — standing in for full model retraining in
         this prototype, with a shadow-vs-champion validation step before
         promotion.
    """

    def __init__(self, name: str):
        self.name = name
        self.log = []

    def reconcile(self, entity_id: str, period, predicted_positive: bool,
                  actual_positive: bool, context: dict | None = None):
        context = context or {}
        if predicted_positive and actual_positive:
            outcome = "true_positive"
            reason = "correctly flagged risk that materialized"
        elif predicted_positive and not actual_positive:
            outcome = "false_positive"
            reason = context.get("fp_reason", "flagged risk did not materialize in this window")
        elif not predicted_positive and actual_positive:
            outcome = "false_negative"
            reason = context.get("fn_reason", "missed risk that materialized")
        else:
            outcome = "true_negative"
            reason = "correctly identified as low-risk"

        self.log.append({
            "entity_id": entity_id,
            "period": period,
            "predicted_positive": predicted_positive,
            "actual_positive": actual_positive,
            "outcome": outcome,
            "reason": reason,
        })

    def rolling_accuracy(self, window: int = 14) -> pd.DataFrame:
        log_df = pd.DataFrame(self.log)
        if log_df.empty:
            return log_df
        log_df["is_correct"] = log_df["outcome"].isin(["true_positive", "true_negative"]).astype(int)
        log_df = log_df.sort_values("period")
        log_df["rolling_accuracy"] = log_df["is_correct"].rolling(window, min_periods=1).mean()
        return log_df

    def detect_sustained_drift(self, window: int = 14, drop_threshold: float = 0.10) -> bool:
        """
        Flags sustained drift if rolling accuracy has dropped by more than
        `drop_threshold` compared to the window before it — i.e. a real,
        sustained degradation rather than one bad day.
        """
        log_df = self.rolling_accuracy(window)
        if len(log_df) < window * 2:
            return False
        recent = log_df["rolling_accuracy"].iloc[-window:].mean()
        prior = log_df["rolling_accuracy"].iloc[-2 * window:-window].mean()
        return (prior - recent) > drop_threshold

    def shadow_vs_champion(self, champion_accuracy: float, shadow_accuracy: float,
                            min_improvement: float = 0.02) -> dict:
        """
        Champion-challenger promotion rule: a retrained ("shadow") model
        only replaces the live ("champion") model if it beats it by at least
        `min_improvement` on held-out validation performance. This is the
        safeguard against a retrained model being promoted on the strength
        of a lucky short run.
        """
        promote = (shadow_accuracy - champion_accuracy) >= min_improvement
        return {
            "champion_accuracy": champion_accuracy,
            "shadow_accuracy": shadow_accuracy,
            "improvement": round(shadow_accuracy - champion_accuracy, 4),
            "promoted": promote,
        }
