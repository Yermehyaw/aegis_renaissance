"""
AEGIS Command Deck
Renaissance Innovation Week 2026 — Challenge 2

A unified operations dashboard fusing production (well testing), pipeline
integrity, and community-relationship risk into one prioritized decision
layer, with a live continuous-learning reconciliation panel.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from reconciliation_logic import (
    detect_instrument_health,
    score_pipeline_risk,
    ReconciliationEngine,
)

# ===========================================================================
# PAGE CONFIG + DESIGN SYSTEM (CSS)
# ===========================================================================

st.set_page_config(
    page_title="AEGIS Command Deck",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "canvas": "#0D1917",
    "panel": "#142623",
    "panel_raised": "#1B322E",
    "text": "#EDEDE4",
    "text_muted": "#8FA39C",
    "normal": "#4C9A73",
    "watch": "#D9A441",
    "elevated": "#C97A3D",
    "critical": "#B33F30",
    "signal": "#2FB8A6",
    "grid": "#24413C",
}

STATUS_COLOR = {
    "healthy": COLORS["normal"], "low": COLORS["normal"],
    "watch": COLORS["watch"], "moderate": COLORS["watch"],
    "degrading": COLORS["elevated"], "elevated": COLORS["elevated"],
    "failed": COLORS["critical"], "critical": COLORS["critical"],
}

LAYER_COLOR = {
    "mechanical": "#5B8FD9",
    "integrity": "#C97A3D",
    "community": "#A67CC2",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.stApp {{
    background-color: {COLORS['canvas']};
}}

#MainMenu, footer, header {{visibility: hidden;}}

.block-container {{
    padding-top: 1.2rem;
    max-width: 1280px;
}}

.aegis-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid {COLORS['grid']};
    padding-bottom: 14px;
    margin-bottom: 6px;
}}
.aegis-wordmark {{
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: {COLORS['text']};
}}
.aegis-pulse {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {COLORS['signal']};
    margin-right: 8px;
    box-shadow: 0 0 6px {COLORS['signal']};
}}
.aegis-subtitle {{
    color: {COLORS['text_muted']};
    font-size: 0.85rem;
}}
.aegis-clock {{
    font-family: 'IBM Plex Mono', monospace;
    color: {COLORS['text_muted']};
    font-size: 0.85rem;
}}

.kpi-row {{
    display: flex;
    border-bottom: 1px solid {COLORS['grid']};
    padding: 16px 0 18px 0;
    margin-bottom: 8px;
}}
.kpi-block {{
    flex: 1;
    padding-right: 28px;
    border-right: 1px solid {COLORS['grid']};
    margin-right: 28px;
}}
.kpi-block:last-child {{ border-right: none; }}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: {COLORS['text']};
}}
.kpi-label {{
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    margin-top: 2px;
}}
.kpi-caption {{
    font-size: 0.72rem;
    color: {COLORS['text_muted']};
    opacity: 0.75;
    margin-top: 1px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 1px solid {COLORS['grid']};
}}
.stTabs [data-baseweb="tab"] {{
    height: 38px;
    background-color: transparent;
    border-radius: 0px;
    color: {COLORS['text_muted']};
    font-size: 0.86rem;
    padding: 0 14px;
}}
.stTabs [aria-selected="true"] {{
    color: {COLORS['text']} !important;
    border-bottom: 2px solid {COLORS['signal']} !important;
    background-color: transparent;
}}

.alert-row {{
    display: flex;
    align-items: center;
    border-bottom: 1px solid {COLORS['grid']};
    padding: 9px 4px;
    font-size: 0.86rem;
}}
.alert-layerbar {{
    width: 4px;
    align-self: stretch;
    margin-right: 12px;
    border-radius: 1px;
}}
.alert-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    flex-shrink: 0;
}}
.alert-entity {{
    font-family: 'IBM Plex Mono', monospace;
    color: {COLORS['text']};
    width: 170px;
    padding-right: 14px;
    flex-shrink: 0;
}}
.alert-desc {{
    color: {COLORS['text']};
    flex-grow: 1;
}}
.alert-meta {{
    font-family: 'IBM Plex Mono', monospace;
    color: {COLORS['text_muted']};
    font-size: 0.78rem;
    text-align: right;
    width: 130px;
    flex-shrink: 0;
}}
.alert-layerlabel {{
    font-size: 0.68rem;
    color: {COLORS['text_muted']};
    width: 84px;
    flex-shrink: 0;
}}

.section-label {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {COLORS['text']};
    margin: 4px 0 2px 0;
}}
.section-caption {{
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    margin-bottom: 10px;
}}

.stSelectbox label, .stSlider label, .stRadio label {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.82rem !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def plotly_base_layout(fig, height=340):
    fig.update_layout(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="IBM Plex Sans", color=COLORS["text_muted"], size=12),
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


# ===========================================================================
# DATA LOADING
# ===========================================================================

@st.cache_data
def load_data():
    well = pd.read_csv("data/well_testing_data.csv", parse_dates=["timestamp"])
    inst = pd.read_csv("data/instrument_health_data.csv", parse_dates=["timestamp"])
    pipe = pd.read_csv("data/pipeline_segments.csv", parse_dates=["period"])
    comm = pd.read_csv("data/community_metrics.csv", parse_dates=["period"])
    return well, inst, pipe, comm


@st.cache_data
def compute_instrument_health(inst_df):
    return detect_instrument_health(inst_df)


@st.cache_data
def compute_pipeline_risk(pipe_df):
    return score_pipeline_risk(pipe_df)


@st.cache_data
def fit_virtual_well_model(well_df, well_id):
    """Per-well linear regression: continuous DCS features -> MER oil_bbl and
    water cut.

    Trained on a CHRONOLOGICAL split of this well's historical MER test
    readings (the earlier ~70% of tests only) and evaluated on the later,
    held-out ~30% — points the model never saw during training. shuffle=False
    in train_test_split is what enforces the chronological split rather than
    a random one: a random split would let the model "see the future"
    relative to some of its test points, which isn't how it would actually
    be used in the field (always predicting forward from what's already
    known). The reported error is therefore a genuine forward-looking
    accuracy estimate, not in-sample fit quality.
    """
    g = well_df[well_df["well_id"] == well_id].sort_values("hour_index").reset_index(drop=True)
    features = ["wellhead_pressure_psi", "flowline_pressure_psi",
                "wellhead_temperature_F", "differential_pressure_inH2O", "choke_size_64th"]

    mer_rows = g.dropna(subset=["mer_oil_bbl_24hr"]).sort_values("hour_index")
    train_idx, test_idx = train_test_split(mer_rows.index, test_size=0.3, shuffle=False)
    train, test = mer_rows.loc[train_idx], mer_rows.loc[test_idx]

    oil_model = LinearRegression().fit(train[features], train["mer_oil_bbl_24hr"])
    wc_model = LinearRegression().fit(train[features], train["mer_water_cut_pct"])

    # Held-out evaluation — computed ONLY on the test split, never on
    # training rows.
    test_oil_pred = oil_model.predict(test[features])
    holdout_mae = float(np.mean(np.abs(test["mer_oil_bbl_24hr"] - test_oil_pred)))
    holdout_mape = float(
        np.mean(np.abs(test["mer_oil_bbl_24hr"] - test_oil_pred) / test["mer_oil_bbl_24hr"]) * 100
    )

    # Raw hourly estimate carries real sensor-level noise, as it would in the
    # field. A production decision layer would smooth this over a short
    # trailing window before treating it as a stable "virtual test" reading —
    # the same way a control-room operator would not react to a single noisy
    # instantaneous sample. An 8-hour centered rolling mean is applied for
    # exactly that reason, not for cosmetic effect.
    g["virtual_oil_bbl_est_raw"] = oil_model.predict(g[features])
    g["virtual_oil_bbl_est"] = g["virtual_oil_bbl_est_raw"].rolling(8, center=True, min_periods=1).mean()

    g["virtual_water_cut_est_raw"] = np.clip(wc_model.predict(g[features]), 0, 100)
    g["virtual_water_cut_est"] = g["virtual_water_cut_est_raw"].rolling(8, center=True, min_periods=1).mean()

    # Tag which MER points were used for training vs held out, so the chart
    # can show this honestly instead of implying every point was a fair test.
    # Initialized as an object-dtype column (via None, not np.nan) so string
    # labels can be assigned into it without a dtype conflict.
    g["mer_split"] = pd.Series([None] * len(g), dtype="object")
    g.loc[train.index, "mer_split"] = "train"
    g.loc[test.index, "mer_split"] = "test"

    return g, holdout_mae, holdout_mape


# ===========================================================================
# CONTINUOUS LEARNING RECONCILIATION (Component 3)
# ===========================================================================

def tag_pipeline_miss(row, outcome):
    if outcome == "false_negative":
        if not row["geo_verified"]:
            return "missed — asset unverified under Phase 0 reconciliation, risk factors incomplete"
        if row["clamp_count"] >= 8:
            return "missed — legacy clamp-heavy segment, consider raising clamp weighting"
        return "missed — incident occurred despite moderate risk profile, review contributing factors"
    if outcome == "false_positive":
        if row["historical_incident_count_3yr"] >= 3 and row["dcvg_defect_count"] < 2:
            return "false alarm — elevated mainly by accumulated history, no new structural defects"
        return "false alarm — flagged elevated, no incident materialized this period"
    return ""


@st.cache_data
def run_pipeline_reconciliation(pipe_scored, threshold):
    eng = ReconciliationEngine("pipeline_risk")
    df = pipe_scored.sort_values(["segment_id", "month_index"]).copy()
    df["prior_risk_score"] = df.groupby("segment_id")["risk_score"].shift(1)
    df = df.dropna(subset=["prior_risk_score"])
    for _, row in df.iterrows():
        predicted_positive = row["prior_risk_score"] > threshold
        actual_positive = bool(row["actual_incident_occurred_this_month"])
        ctx = {}
        if predicted_positive != actual_positive:
            ctx = {"fn_reason": tag_pipeline_miss(row, "false_negative"),
                   "fp_reason": tag_pipeline_miss(row, "false_positive")}
        eng.reconcile(row["segment_id"], row["period"], predicted_positive, actual_positive, ctx)
    return eng


def tag_instrument_miss(row, outcome):
    if outcome == "false_negative":
        return "missed — drift too gradual to cross threshold within the day"
    if outcome == "false_positive":
        return "false alarm — transient noise spike, not sustained degradation"
    return ""


@st.cache_data
def run_instrument_reconciliation(inst_scored):
    eng = ReconciliationEngine("instrument_health")
    df = inst_scored.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby(["instrument_id", "date"]).agg(
        predicted_positive=("predicted_status", lambda s: (s != "healthy").any()),
        actual_positive=("actual_status", lambda s: (s != "healthy").any()),
        period=("timestamp", "first"),
    ).reset_index()
    daily = daily.sort_values(["instrument_id", "period"])
    for _, row in daily.iterrows():
        ctx = {}
        if row["predicted_positive"] != row["actual_positive"]:
            ctx = {"fn_reason": tag_instrument_miss(row, "false_negative"),
                   "fp_reason": tag_instrument_miss(row, "false_positive")}
        eng.reconcile(row["instrument_id"], row["period"], row["predicted_positive"], row["actual_positive"], ctx)
    return eng


# ===========================================================================
# LOAD + PRECOMPUTE
# ===========================================================================

well_df, inst_df, pipe_df, comm_df = load_data()
inst_scored = compute_instrument_health(inst_df)
pipe_scored = compute_pipeline_risk(pipe_df)

latest_month = pipe_scored["month_index"].max()
latest_pipe = pipe_scored[pipe_scored["month_index"] == latest_month]
latest_inst_hour = inst_scored["hour_index"].max()
latest_inst = inst_scored[inst_scored["hour_index"] == latest_inst_hour]
latest_comm_month = comm_df["month_index"].max()
latest_comm = comm_df[comm_df["month_index"] == latest_comm_month]

# ===========================================================================
# HEADER
# ===========================================================================

st.markdown(f"""
<div class="aegis-header">
    <div>
        <span class="aegis-wordmark"><span class="aegis-pulse"></span>AEGIS</span>
        <div class="aegis-subtitle">Asset, Evacuation &amp; Guardianship Intelligence System — Command Deck</div>
    </div>
    <div class="aegis-clock">RENAISSANCE JV · NIGER DELTA OPERATIONS THEATRE</div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# KPI STRIP
# ===========================================================================

n_mech_alerts = (latest_inst["predicted_status"] != "healthy").sum()
n_integrity_alerts = latest_pipe["risk_category"].isin(["elevated", "critical"]).sum()
n_comm_alerts = (
    latest_comm["active_ultimatum_flag"].sum()
    + (latest_comm["custodianship_index_0_100"] < 45).sum()
)
total_alerts = int(n_mech_alerts + n_integrity_alerts + n_comm_alerts)

champion_eng = run_pipeline_reconciliation(pipe_scored, threshold=0.5)
acc_df = champion_eng.rolling_accuracy(window=12)
current_accuracy = acc_df["rolling_accuracy"].iloc[-1] if not acc_df.empty else float("nan")

avg_custodianship = latest_comm["custodianship_index_0_100"].mean()

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-block">
        <div class="kpi-value">~2,400–4,800 bbl/d</div>
        <div class="kpi-label">Recovery potential (modeled)</div>
        <div class="kpi-caption">1–2% downtime reduction, 240k bpd baseline</div>
    </div>
    <div class="kpi-block">
        <div class="kpi-value">{total_alerts}</div>
        <div class="kpi-label">Active alerts, all layers</div>
        <div class="kpi-caption">{n_mech_alerts} mechanical · {n_integrity_alerts} integrity · {n_comm_alerts} community</div>
    </div>
    <div class="kpi-block">
        <div class="kpi-value">{current_accuracy:.1%}</div>
        <div class="kpi-label">Pipeline risk model accuracy</div>
        <div class="kpi-caption">trailing 12-period rolling window</div>
    </div>
    <div class="kpi-block">
        <div class="kpi-value">{avg_custodianship:.1f} / 100</div>
        <div class="kpi-label">Community custodianship index</div>
        <div class="kpi-caption">fleet average, latest period</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# TABS
# ===========================================================================

tab_overview, tab_well, tab_inst, tab_pipe, tab_comm, tab_learn = st.tabs([
    "Overview", "Well Testing", "Instrument Health", "Pipeline Risk", "Community", "Continuous Learning"
])

# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-label">Unified alert feed</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Mechanical, pipeline integrity, and community risk — '
        'one ranked list instead of three disconnected pictures.</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for _, r in latest_inst[latest_inst["predicted_status"] != "healthy"].iterrows():
        sev_rank = {"failed": 0, "degrading": 1, "watch": 2}.get(r["predicted_status"], 3)
        rows.append(dict(layer="mechanical", severity=r["predicted_status"], sev_rank=sev_rank,
                          entity=r["instrument_id"],
                          desc=f"{r['instrument_type'].replace('_',' ')} at {r['location']} — sustained deviation detected",
                          meta=f"z={r['z_score']:.1f}"))
    for _, r in latest_pipe[latest_pipe["risk_category"].isin(["elevated", "critical"])].iterrows():
        sev_rank = {"critical": 0, "elevated": 1}.get(r["risk_category"], 3)
        rows.append(dict(layer="integrity", severity=r["risk_category"], sev_rank=sev_rank,
                          entity=r["segment_id"],
                          desc=f"{r['corridor']} corridor — {r['dcvg_defect_count']} DCVG defects, {r['clamp_count']} legacy clamps",
                          meta=f"risk={r['risk_score']:.2f}"))
    for _, r in latest_comm.iterrows():
        if r["active_ultimatum_flag"]:
            rows.append(dict(layer="community", severity="critical", sev_rank=0,
                              entity=r["community"],
                              desc="Active HCDT board / grievance ultimatum in effect",
                              meta=f"idx={r['custodianship_index_0_100']:.0f}"))
        elif r["custodianship_index_0_100"] < 45:
            rows.append(dict(layer="community", severity="elevated", sev_rank=1,
                              entity=r["community"],
                              desc="Custodianship index below threshold — relationship risk rising",
                              meta=f"idx={r['custodianship_index_0_100']:.0f}"))

    alert_df = pd.DataFrame(rows).sort_values(["sev_rank", "entity"]) if rows else pd.DataFrame()

    if alert_df.empty:
        st.markdown('<div class="section-caption">No active alerts across any layer this period.</div>', unsafe_allow_html=True)
    else:
        html_rows = []
        for _, r in alert_df.iterrows():
            html_rows.append(f"""
            <div class="alert-row">
                <div class="alert-layerbar" style="background:{LAYER_COLOR[r['layer']]}"></div>
                <div class="alert-layerlabel">{r['layer']}</div>
                <span class="alert-dot" style="background:{STATUS_COLOR.get(r['severity'], COLORS['text_muted'])}"></span>
                <div class="alert-entity">{r['entity']}</div>
                <div class="alert-desc">{r['desc']}</div>
                <div class="alert-meta">{r['meta']}</div>
            </div>""")
        st.markdown("".join(html_rows), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Alerts by layer</div>', unsafe_allow_html=True)
        if not alert_df.empty:
            counts = alert_df["layer"].value_counts()
            fig = go.Figure(go.Bar(
                x=counts.values, y=counts.index, orientation="h",
                marker_color=[LAYER_COLOR[l] for l in counts.index],
            ))
            st.plotly_chart(plotly_base_layout(fig, height=220), width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-label">Custodianship index by corridor</div>', unsafe_allow_html=True)
        corridor_avg = latest_comm.groupby("corridor")["custodianship_index_0_100"].mean().sort_values()
        fig = go.Figure(go.Bar(
            x=corridor_avg.values, y=corridor_avg.index, orientation="h",
            marker_color=COLORS["signal"],
        ))
        st.plotly_chart(plotly_base_layout(fig, height=220), width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
with tab_well:
    st.markdown('<div class="section-label">Continuous virtual well testing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Estimating oil rate and water cut between physical 24-hour '
        'MER test-separator cycles, from continuous DCS signals.</div>', unsafe_allow_html=True,
    )
    well_ids = sorted(well_df["well_id"].unique())
    sel_well = st.selectbox("Well", well_ids, index=0)
    g, holdout_mae, holdout_mape = fit_virtual_well_model(well_df, sel_well)

    mer_train = g[g["mer_split"] == "train"]
    mer_test = g[g["mer_split"] == "test"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["virtual_oil_bbl_est"], mode="lines",
                              name="Virtual estimate (hourly)", line=dict(color=COLORS["signal"], width=1.5)))
    fig.add_trace(go.Scatter(x=mer_train["timestamp"], y=mer_train["mer_oil_bbl_24hr"], mode="markers",
                              name="MER test — used to train",
                              marker=dict(color=COLORS["text_muted"], size=6, symbol="diamond")))
    fig.add_trace(go.Scatter(x=mer_test["timestamp"], y=mer_test["mer_oil_bbl_24hr"], mode="markers",
                              name="MER test — held out (scored)",
                              marker=dict(color=COLORS["watch"], size=8, symbol="diamond")))
    fig.update_layout(yaxis_title="Oil rate (bbl/24h)")
    st.plotly_chart(plotly_base_layout(fig, height=360), width='stretch', config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=g["timestamp"], y=g["virtual_water_cut_est"], mode="lines",
                                   name="Virtual water cut estimate", line=dict(color="#5B8FD9", width=1.5)))
        fig2.add_trace(go.Scatter(x=mer_test["timestamp"], y=mer_test["mer_water_cut_pct"], mode="markers",
                                   name="MER water cut — held out",
                                   marker=dict(color=COLORS["watch"], size=7, symbol="diamond")))
        fig2.update_layout(yaxis_title="Water cut (%)")
        st.markdown('<div class="section-label">Water cut trend</div>', unsafe_allow_html=True)
        st.plotly_chart(plotly_base_layout(fig2, height=260), width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-label">Held-out accuracy (never seen in training)</div>', unsafe_allow_html=True)
        st.metric("Mean absolute error", f"{holdout_mae:,.0f} bbl/24h")
        st.metric("Mean absolute % error", f"{holdout_mape:.1f}%")
        st.markdown(
            f'<div class="section-caption">Trained on the earliest {len(mer_train)} MER tests for this '
            f'well; scored only on the {len(mer_test)} later tests the model never saw during training — '
            'a genuine forward-looking accuracy estimate, computed with a chronological '
            '(non-shuffled) train/test split via scikit-learn.</div>', unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
with tab_inst:
    st.markdown('<div class="section-label">Predictive instrument health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Rolling deviation + trend-slope drift detection on existing '
        '4–20 mA signal streams — catching failures before a technician would notice manually.</div>',
        unsafe_allow_html=True,
    )
    inst_ids = sorted(inst_scored["instrument_id"].unique())
    sel_inst = st.selectbox("Instrument", inst_ids, index=0)
    gi = inst_scored[inst_scored["instrument_id"] == sel_inst].sort_values("hour_index")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gi["timestamp"], y=gi["signal_ma"], mode="lines",
                              name="Signal (mA)", line=dict(color=COLORS["text_muted"], width=1)))
    for status, color in [("watch", COLORS["watch"]), ("degrading", COLORS["elevated"]), ("failed", COLORS["critical"])]:
        sub = gi[gi["predicted_status"] == status]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["signal_ma"], mode="markers",
                                      name=f"Predicted: {status}", marker=dict(color=color, size=5)))
    fig.update_layout(yaxis_title="Signal (mA)")
    st.plotly_chart(plotly_base_layout(fig, height=360), width='stretch', config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    result_full = inst_scored.copy()
    result_full["actual_positive"] = result_full["actual_status"] != "healthy"
    result_full["predicted_positive"] = result_full["predicted_status"] != "healthy"
    tp = ((result_full.actual_positive) & (result_full.predicted_positive)).sum()
    fn = ((result_full.actual_positive) & (~result_full.predicted_positive)).sum()
    fp = ((~result_full.actual_positive) & (result_full.predicted_positive)).sum()
    tn = ((~result_full.actual_positive) & (~result_full.predicted_positive)).sum()
    with c1:
        st.metric("Fleet-wide recall", f"{tp/(tp+fn):.1%}", help="Share of real issues caught")
    with c2:
        st.metric("False positive rate", f"{fp/(fp+tn):.1%}")
    with c3:
        st.metric("Instruments monitored", f"{inst_scored['instrument_id'].nunique()}")

# ---------------------------------------------------------------------------
with tab_pipe:
    st.markdown('<div class="section-label">Pipeline segment risk scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Weighted score from age, DCVG survey defects, cathodic '
        'protection, incident history, throughput, legacy clamp count, and Phase 0 verification status.</div>',
        unsafe_allow_html=True,
    )

    corridors = ["All"] + sorted(pipe_scored["corridor"].unique())
    sel_corridor = st.selectbox("Corridor", corridors, index=0)
    view = latest_pipe if sel_corridor == "All" else latest_pipe[latest_pipe["corridor"] == sel_corridor]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-label">Highest-risk segments, current period</div>', unsafe_allow_html=True)
        top = view.sort_values("risk_score", ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=top["risk_score"], y=top["segment_id"], orientation="h",
            marker_color=[STATUS_COLOR.get(c, COLORS["text_muted"]) for c in top["risk_category"]],
        ))
        fig.update_layout(xaxis_title="Risk score", yaxis=dict(autorange="reversed"))
        st.plotly_chart(plotly_base_layout(fig, height=340), width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-label">Model validation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Actual incident rate by risk category.</div>', unsafe_allow_html=True)
        val = pipe_scored.groupby("risk_category", observed=True)["actual_incident_occurred_this_month"].mean() * 100
        fig = go.Figure(go.Bar(
            x=val.index.astype(str), y=val.values,
            marker_color=[STATUS_COLOR.get(c, COLORS["text_muted"]) for c in val.index],
        ))
        fig.update_layout(yaxis_title="Actual incident rate (%)")
        st.plotly_chart(plotly_base_layout(fig, height=300), width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
with tab_comm:
    st.markdown('<div class="section-label">Community custodianship layer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Trust, economic participation, and incident-detection share — '
        'tracking the community becoming an asset, not just a stakeholder.</div>', unsafe_allow_html=True,
    )

    communities = sorted(comm_df["community"].unique())
    sel_comm = st.multiselect("Communities", communities, default=communities[:3])
    plot_df = comm_df[comm_df["community"].isin(sel_comm)] if sel_comm else comm_df

    fig = go.Figure()
    palette = ["#2FB8A6", "#D9A441", "#5B8FD9", "#C97A3D", "#A67CC2", "#4C9A73", "#EDEDE4", "#8FA39C", "#B33F30"]
    for i, c in enumerate(plot_df["community"].unique()):
        sub = plot_df[plot_df["community"] == c]
        fig.add_trace(go.Scatter(x=sub["period"], y=sub["custodianship_index_0_100"], mode="lines",
                                  name=c, line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_vline(x=comm_df[comm_df["program_active"]]["period"].min(), line_dash="dot",
                  line_color=COLORS["text_muted"], annotation_text="Custodianship program launch")
    fig.update_layout(yaxis_title="Custodianship index (0-100)")
    st.plotly_chart(plotly_base_layout(fig, height=340), width='stretch', config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Community-sourced detection share</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Rising share = the community becoming an active sensor.</div>', unsafe_allow_html=True)
        trend = comm_df.groupby("month_index")["pct_incidents_community_sourced"].mean()
        fig = go.Figure(go.Scatter(x=trend.index, y=trend.values, mode="lines", fill="tozeroy",
                                    line=dict(color=COLORS["signal"], width=2)))
        fig.update_layout(xaxis_title="Month", yaxis_title="% community-sourced")
        st.plotly_chart(plotly_base_layout(fig, height=260), width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-label">Latest period, by community</div>', unsafe_allow_html=True)
        show_cols = ["community", "corridor", "custodianship_index_0_100", "pipeline_guardians_active",
                     "open_grievances", "shutdown_days_this_month"]
        st.dataframe(latest_comm[show_cols].sort_values("custodianship_index_0_100"),
                     hide_index=True, width='stretch', height=260)

# ---------------------------------------------------------------------------
with tab_learn:
    st.markdown('<div class="section-label">Continuous learning — prediction vs. outcome</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Every prediction is logged and reconciled against what actually '
        'happened. Sustained misses are tagged with a likely reason and trigger a retrain — validated in '
        'shadow mode before it replaces the live model.</div>', unsafe_allow_html=True,
    )

    model_choice = st.radio("Model", ["Pipeline risk scoring", "Instrument health detection"], horizontal=True)

    if model_choice == "Pipeline risk scoring":
        threshold = st.slider("Current live (champion) risk threshold", 0.2, 0.8, 0.5, 0.05)
        champion = run_pipeline_reconciliation(pipe_scored, threshold)
        champ_acc = champion.rolling_accuracy(window=12)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div class="section-label">Accuracy by month (fleet-wide average)</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-caption">Each point averages that month\'s prediction-vs-outcome '
                'results across all 42 segments — the underlying log is per-segment-per-month.</div>',
                unsafe_allow_html=True,
            )
            monthly = champ_acc.copy()
            monthly["month"] = pd.to_datetime(monthly["period"]).dt.to_period("M")
            monthly_acc = monthly.groupby("month")["is_correct"].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_acc.index.astype(str), y=monthly_acc.values,
                                      mode="lines+markers", name="Champion model",
                                      line=dict(color=COLORS["signal"], width=2), marker=dict(size=5)))
            fig.update_layout(yaxis_title="Accuracy", yaxis_range=[0, 1.05])
            st.plotly_chart(plotly_base_layout(fig, height=280), width='stretch', config={"displayModeBar": False})
        with c2:
            st.markdown('<div class="section-label">Recent misses, tagged</div>', unsafe_allow_html=True)
            misses = pd.DataFrame(champion.log)
            misses = misses[misses["outcome"].isin(["false_positive", "false_negative"])].tail(6)
            for _, m in misses.iloc[::-1].iterrows():
                dotcolor = COLORS["critical"] if m["outcome"] == "false_negative" else COLORS["watch"]
                st.markdown(
                    f'<div style="font-size:0.78rem; padding:5px 0; border-bottom:1px solid {COLORS["grid"]}">'
                    f'<span class="alert-dot" style="background:{dotcolor}"></span>'
                    f'<b>{m["entity_id"]}</b> — {m["reason"]}</div>', unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Retrain — shadow vs. champion</div>', unsafe_allow_html=True)
        shadow_threshold = st.slider("Candidate (shadow) threshold to test", 0.2, 0.8, max(0.2, threshold - 0.1), 0.05)
        if st.button("Run shadow validation"):
            shadow = run_pipeline_reconciliation(pipe_scored, shadow_threshold)
            shadow_acc_val = shadow.rolling_accuracy(window=12)["rolling_accuracy"].iloc[-1]
            champ_acc_val = champ_acc["rolling_accuracy"].iloc[-1]
            result = champion.shadow_vs_champion(champ_acc_val, shadow_acc_val)
            if result["promoted"]:
                st.success(
                    f"Shadow model promoted: accuracy improved from {result['champion_accuracy']:.1%} "
                    f"to {result['shadow_accuracy']:.1%} ({result['improvement']:+.1%}). New threshold: {shadow_threshold}"
                )
            else:
                st.warning(
                    f"Shadow model NOT promoted: accuracy changed from {result['champion_accuracy']:.1%} "
                    f"to {result['shadow_accuracy']:.1%} ({result['improvement']:+.1%}) — below the required improvement margin."
                )

    else:
        inst_eng = run_instrument_reconciliation(inst_scored)
        acc = inst_eng.rolling_accuracy(window=10)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div class="section-label">Accuracy by week, fleet-wide average</div>', unsafe_allow_html=True)
            weekly = acc.copy()
            weekly["week"] = pd.to_datetime(weekly["period"]).dt.to_period("W")
            weekly_acc = weekly.groupby("week")["is_correct"].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly_acc.index.astype(str), y=weekly_acc.values,
                                      mode="lines+markers", line=dict(color=COLORS["signal"], width=2), marker=dict(size=5)))
            fig.update_layout(yaxis_title="Accuracy", yaxis_range=[0, 1.05])
            st.plotly_chart(plotly_base_layout(fig, height=280), width='stretch', config={"displayModeBar": False})
        with c2:
            st.markdown('<div class="section-label">Recent misses, tagged</div>', unsafe_allow_html=True)
            misses = pd.DataFrame(inst_eng.log)
            misses = misses[misses["outcome"].isin(["false_positive", "false_negative"])].tail(6)
            for _, m in misses.iloc[::-1].iterrows():
                dotcolor = COLORS["critical"] if m["outcome"] == "false_negative" else COLORS["watch"]
                st.markdown(
                    f'<div style="font-size:0.78rem; padding:5px 0; border-bottom:1px solid {COLORS["grid"]}">'
                    f'<span class="alert-dot" style="background:{dotcolor}"></span>'
                    f'<b>{m["entity_id"]}</b> — {m["reason"]}</div>', unsafe_allow_html=True,
                )
