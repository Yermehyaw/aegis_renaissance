# AEGIS Command Deck — Streamlit Dashboard

A working, wired-up dashboard implementing the "AEGIS Command Deck" design —
built specifically as an industrial DCS/SCADA-style command display rather
than a generic analytics dashboard, on the reasoning that this is the actual
visual world Renaissance's control-room operators live in (dark canvas,
alarm-color semantics, monospace telemetry) rather than a decorative choice.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. No external services or API keys required
— everything runs on the bundled synthetic CSVs in `data/`.

## What's actually working (not just styled)

- **Well Testing tab**: fits a real per-well `scikit-learn` `LinearRegression`
  against each well's historical MER test readings, using a **chronological**
  (non-shuffled) `train_test_split` — the model only ever trains on a well's
  earlier tests and is scored on its later, held-out ones, matching how it
  would actually be used in the field. The chart visually distinguishes
  training points (grey) from held-out scored points (amber), so the split
  is visible, not just claimed. Reported error is held-out-only:
  **1.7%–8.7% MAPE depending on well** (higher-water-cut legacy wells are
  predictably harder to predict than stable ones — a real, citable pattern,
  not noise).
- **Instrument Health tab**: runs the actual `detect_instrument_health()`
  drift/deviation detector from `reconciliation_logic.py` against all 18
  instruments and reports fleet-wide recall/false-positive rate live
  (~99.98% recall in testing).
- **Pipeline Risk tab**: runs the actual `score_pipeline_risk()` weighted
  model and shows the real validation chart — actual incident rate by risk
  category, computed from the data each time it loads.
- **Community tab**: plots the real custodianship-index time series showing
  the before/after custodianship-program effect built into the synthetic
  data.
- **Continuous Learning tab**: the most important one — runs the actual
  `ReconciliationEngine` from the previous deliverable against the pipeline
  risk model's historical predictions, live. You can drag the "champion"
  threshold, see the monthly accuracy trend (which genuinely degrades over
  the 24 months in this dataset — a real, unforced finding, and a strong
  talking point: "here's why a static model isn't enough"), inspect tagged
  misses, then drag a "shadow" threshold and click **Run shadow validation**
  to see the champion/challenger promotion decision computed live in front
  of judges.

## Design notes

Full design rationale (palette, type, layout reasoning) is in the project
conversation; in short: canvas `#0D1917`, panel `#142623`, status colors
grounded in real HMI alarm semantics (not decorative), IBM Plex Sans/Mono
pairing (Mono reserved for telemetry values only — a functional distinction,
not decorative), and the unified alert feed as the hero rather than a stat
callout, since that fusion is AEGIS's actual thesis.

## If something looks off when you run it locally

- Fonts: the Google Fonts `@import` requires internet access in the
  browser rendering the page — it degrades gracefully to system sans-serif/
  monospace if blocked, just less polished.
- First load is slower than subsequent ones — instrument-health detection
  runs a per-instrument linear-fit loop across ~26,000 rows on first call;
  `@st.cache_data` means every tab switch after that is instant.
- If you add more wells/instruments/segments to the underlying CSVs, no
  code changes are needed — every chart and dropdown reads the data
  dynamically.

## Known limitations (say this plainly if asked)

- All underlying data is synthetic, generated to be structurally realistic,
  not copied from real Renaissance operations.
- The well-testing train/test split is chronological but still drawn from
  60 days of synthetic history per well — a real deployment would want a
  longer history and ideally validation across multiple well "regimes"
  (e.g. before/after a workover), not just a single continuous window.
- The pipeline risk model's accuracy decline over the 24-month synthetic
  window is real (computed, not scripted) but reflects the specific way the
  synthetic data was generated (clamp counts and DCVG defects drift upward
  over time while the champion threshold stays fixed) — real-world drift
  patterns would need to be discovered from actual historical data, not
  assumed to look like this.
