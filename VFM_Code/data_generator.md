# AEGIS V2 Synthetic Data Generator

## Code-Level Technical Annotation

### 1. Purpose and Scope

The AEGIS V2 synthetic data generator creates a 60-day hourly production dataset for 10 simulated oil-producing wells.

The generator attempts to represent a simplified production environment in which:

* reservoir conditions evolve over time;
* wells have different production characteristics;
* water cut changes with time;
* gas production varies with oil production;
* wellbore pressure is related to production through a simplified productivity-index relationship;
* fluid density changes with water cut;
* hydrostatic pressure loss changes with fluid composition;
* choke size influences simulated surface backpressure;
* surface measurements contain random measurement noise;
* periodic production tests provide noisy reference measurements.

The generated dataset is subsequently used to train the AEGIS Virtual Flow Meter.

It is important to distinguish three layers within the code:

```text
                    SIMULATION
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Hidden/latent state        Observable signals
          │                         │
          │                         ├── WHP
          │                         ├── WHT
          │                         ├── DP
          │                         ├── Choke
          │                         └── Vibration
          │
          └── True oil/gas/water
                       │
                       ▼
               Physical test
                       │
                       ▼
             Noisy reference data
```

The ML model subsequently sees the observable data and learns from the periodic test measurements.

---

# 2. Imports and Random Number Generator

### Code

```python
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
```

### What it does

Two Python libraries are imported:

* `numpy` for numerical calculations and random-number generation;
* `pandas` for constructing the time-series dataset and exporting it to CSV.

The random-number generator is initialised with the seed `42`.

### Why the seed matters

A random generator is used throughout the simulation for:

* well parameters;
* reservoir pressure;
* water-cut variation;
* production variation;
* choke events;
* measurement noise;
* temperature noise;
* vibration.

Without a fixed seed, every execution would produce a different dataset.

With:

```python
RNG = np.random.default_rng(42)
```

the same code produces the same synthetic dataset each time.

This is important for reproducibility.

The generator therefore behaves deterministically **given the same code and seed**, despite using stochastic processes internally.

---

# 3. Well Population

### Code

```python
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
```

### What it does

The list establishes the simulated field population.

Each tuple contains:

```text
well ID
flow station
legacy/reactivated status
base oil rate
base GOR
base water cut
```

For example:

```python
("SOKU-W01", "Soku-FS", True, 1800, 620, 0.52)
```

means the simulated well:

* is identified as `SOKU-W01`;
* belongs to `Soku-FS`;
* is classified as a legacy/reactivated well;
* has a nominal oil production of 1,800 bpd;
* has a nominal GOR of 620 scf/bbl;
* starts with a nominal water cut of 52%.

### Engineering purpose

The population deliberately contains wells with different production and fluid characteristics.

For example:

```text
Legacy wells
→ generally higher water cut
→ generally higher decline/drift assumptions

Standard wells
→ generally lower water cut
→ generally lower decline/drift assumptions
```

This introduces heterogeneity into the simulated field.

Without this variation, the ML problem would be much easier because every well would behave similarly.

### Important qualification

The names and values are **synthetic**.

They should not be presented as actual production values from Soku, Belema, Otumara, Ughelli or Norkpo assets.

---

# 4. Simulation Configuration

### Code

```python
N_HOURS = 60 * 24
TEST_INTERVAL_H = 24
TEST_START_HOUR = 6

records = []
```

### `N_HOURS`

```python
N_HOURS = 60 * 24
```

creates:

$$
60\times24=1,440
$$

hourly observations per well.

With 10 wells:

$$
1,440\times10=14,400
$$

total rows.

### `TEST_INTERVAL_H`

```python
TEST_INTERVAL_H = 24
```

is intended to represent a physical test every 24 hours.

However, there is an important implementation detail:

**The current code does not actually use `TEST_INTERVAL_H` to determine when tests occur.**

The test is triggered later by:

```python
if hour_of_day == TEST_START_HOUR:
```

Therefore, the actual test frequency is controlled by `TEST_START_HOUR`, not by `TEST_INTERVAL_H`.

For the current value of 24 hours, the intended and actual behaviour coincide: one test occurs every day at 06:00.

However, changing:

```python
TEST_INTERVAL_H = 48
```

would **not** automatically produce tests every 48 hours.

This should be corrected in a future version.

---

# 5. Iterating Through Each Well

### Code

```python
for well_id, flow_station, is_legacy, base_oil, base_gor, base_wc in WELLS:
```

### What it does

The generator processes each simulated well independently.

For every well, the previously defined properties are unpacked into variables.

This is important because each well receives its own:

* decline rate;
* water-cut behaviour;
* initial reservoir pressure;
* productivity index;
* depth;
* choke history.

Consequently, the simulated wells do not share one universal reservoir state.

---

# 6. Reservoir Parameters

### Code

```python
decline_rate = (
    RNG.uniform(0.0006, 0.0015)
    if not is_legacy
    else RNG.uniform(0.0015, 0.0030)
)

wc_drift_per_day = (
    RNG.uniform(0.0005, 0.0015)
    if is_legacy
    else RNG.uniform(0.0001, 0.0004)
)

initial_reservoir_pressure = RNG.uniform(3200, 3800)

productivity_index = (
    (base_oil / 1000.0)
    * RNG.uniform(0.9, 1.3)
)

well_depth_ft = RNG.uniform(5500, 7500)
```

This is one of the most important sections of the generator.

It establishes the hidden characteristics of each well.

---

# 7. Decline Rate

### Code

```python
decline_rate = (
    RNG.uniform(0.0006, 0.0015)
    if not is_legacy
    else RNG.uniform(0.0015, 0.0030)
)
```

The generator assigns each well a random decline coefficient.

Standard wells receive:

$$
0.0006\leq D\leq0.0015
$$

Legacy/reactivated wells receive:

$$
0.0015\leq D\leq0.0030
$$

### Engineering interpretation

The generator assumes legacy/reactivated wells experience a greater decline tendency than standard wells.

This is a synthetic modelling assumption designed to differentiate the populations.

It should therefore be described as:

> **A synthetic asset-condition assumption, not an empirically established Renaissance decline relationship.**

---

# 8. Exponential Decline

Later in the code, the decline rate is converted into a time-dependent decline factor:

```python
decline_factor = np.exp(-decline_rate * day)
```

Mathematically:

$$
D_f(t)=e^{-Dt}
$$

where:

* \(D_f\) = decline factor;
* \(D\) = well-specific decline coefficient;
* \(t\) = elapsed days.

This factor is subsequently applied to both reservoir pressure and oil production.

Thus:

$$
P_r(t)=P_{r,0}e^{-Dt}
$$

and:

$$
Q_o(t)=Q_{o,0}e^{-Dt}\times\text{noise}
$$

This is an exponential decline mechanism.

It is **not an Arps decline model**.

That distinction should be preserved in technical documentation.

---

# 9. Water-Cut Drift

### Code

```python
wc_drift_per_day = (
    RNG.uniform(0.0005, 0.0015)
    if is_legacy
    else RNG.uniform(0.0001, 0.0004)
)
```

The generator allows water cut to evolve over time.

Legacy wells receive a higher drift range than standard wells.

This represents an assumption that older/reactivated wells are more susceptible to increasing water production.

Again, this is a synthetic assumption rather than a field-calibrated conclusion.

---

# 10. Initial Reservoir Pressure

### Code

```python
initial_reservoir_pressure = RNG.uniform(3200, 3800)
```

Each well begins with a random reservoir pressure between:

$$
3,200\leq P_{r,0}\leq3,800\text{ psi}
$$

This creates different reservoir-pressure starting conditions between wells.

Because this variable is not exported to the final ML feature set, it functions as a **latent reservoir state**.

---

# 11. Productivity Index

### Code

```python
productivity_index = (
    (base_oil / 1000.0)
    * RNG.uniform(0.9, 1.3)
)
```

The productivity index is made proportional to the well's nominal oil rate.

For example, a nominal 2,000-bpd well gets:

$$
2.0\times U(0.9,1.3)
$$

where \(U(0.9,1.3)\) is a random value between 0.9 and 1.3.

### Engineering meaning

The productivity index represents how effectively pressure drawdown is converted into production.

The conventional linear PI relationship is:

$$
J=\frac{q}{P_r-P_{wf}}
$$

where:

* \(J\) = productivity index;
* \(q\) = production rate;
* \(P_r\) = reservoir pressure;
* \(P_{wf}\) = flowing bottom-hole pressure.

### Important modelling caveat

The way PI is currently generated means that **higher-base-rate wells are automatically assigned higher PI**.

That is a convenient synthetic relationship, but it is not necessarily how PI behaves in a real field.

For a stronger future generator, PI should ideally be independently parameterised or derived from reservoir/well characteristics rather than directly scaled from base oil rate.

---

# 12. Well Depth

### Code

```python
well_depth_ft = RNG.uniform(5500, 7500)
```

Each well receives a random depth between:

$$
5,500\text{ ft}
$$

and:

$$
7,500\text{ ft}
$$

Depth is subsequently used in the hydrostatic pressure calculation.

This gives each well a different hydrostatic pressure penalty.

---

# 13. Choke Schedule

### Code

```python
n_choke_events = RNG.integers(2, 6)
```

Each well receives between 2 and 5 choke events during the 60-day simulation.

The generator then chooses random hours at which these events occur:

```python
choke_event_hours = np.sort(
    RNG.choice(
        np.arange(24, N_HOURS - 24),
        size=n_choke_events,
        replace=False
    )
)
```

This prevents the first and last 24 hours from being selected.

---

# 14. Initial Choke

### Code

```python
current_choke = RNG.integers(28, 48)
```

The initial choke size is randomly selected between 28/64 and 47/64 inch, depending on NumPy's integer upper-bound behaviour.

The value is stored as:

```python
choke_size_64th
```

This is a size-based representation rather than a percentage-opening representation.

That distinction is appropriate because physical choke sizes are commonly represented in fractions of an inch.

---

# 15. Choke Event Behaviour

### Code

```python
for h in range(N_HOURS):
    if h in choke_event_hours:
        current_choke = np.clip(
            current_choke + RNG.integers(-6, 7),
            16,
            64
        )

    choke_series[h] = current_choke
```

At each randomly selected choke event, the choke size changes by a random amount between approximately -6 and +6 64ths.

The result is restricted to:

$$
16/64\leq choke\leq64/64
$$

through:

```python
np.clip(..., 16, 64)
```

### Engineering purpose

This introduces operational changes into the time series.

The model therefore encounters periods where the operating configuration changes rather than seeing a completely static choke.

---

# 16. Hourly Simulation Loop

### Code

```python
for h in range(N_HOURS):
```

The generator now creates each hourly observation.

For every hour:

```python
day = h // 24
hour_of_day = h % 24
```

converts the absolute hour index into:

* elapsed simulation day;
* hour within the day.

---

# 17. Timestamp

### Code

```python
timestamp = (
    pd.Timestamp("2026-06-01")
    + pd.Timedelta(hours=h)
)
```

The simulated field begins at:

**1 June 2026**

and advances one hour per row.

The timestamp is important for later feature engineering because the VFM model calculates historical changes and rolling values.

---

# 18. TRUE RESERVOIR STATE

The code explicitly identifies the next section as:

```python
# --- TRUE RESERVOIR STATE (Hidden from model) ---
```

This is a very important conceptual boundary.

The generator has information that the ML model does not.

That information includes:

* reservoir pressure;
* true water cut;
* true GOR;
* true oil rate;
* true gas rate;
* true water rate;
* true liquid rate.

These are used to generate the observable surface signals and test measurements.

The model later attempts to infer production without directly receiving these latent variables.

---

# 19. Reservoir Pressure Evolution

### Code

```python
decline_factor = np.exp(-decline_rate * day)

current_res_pressure = (
    initial_reservoir_pressure * decline_factor
)
```

Mathematically:

$$
P_r(t)
=
P_{r,0}e^{-Dt}
$$

At day zero:

$$
P_r(0)=P_{r,0}
$$

As time increases:

$$
P_r(t)<P_{r,0}
$$

### Engineering interpretation

The reservoir is treated as undergoing depletion.

This introduces a slowly changing reservoir state rather than assuming constant reservoir pressure throughout the 60-day period.

---

# 20. True Water Cut

### Code

```python
true_water_cut = np.clip(
    base_wc
    + (wc_drift_per_day * day)
    + RNG.normal(0, 0.005),
    0.02,
    0.95
)
```

This is one of the most useful pieces of the V2 physics.

The underlying water cut is calculated as:

$$
WC(t)
=
WC_0
+
D_{WC}t
+
\epsilon
$$

where:

* \(WC_0\) = base water cut;
* \(D_{WC}\) = daily water-cut drift;
* \(\epsilon\) = random variation.

The random variation has standard deviation:

$$
\sigma=0.005
$$

or roughly 0.5 percentage points when interpreted as a fraction.

The final value is constrained between:

$$
2\%
$$

and:

$$
95\%
$$

through `np.clip`.

### Why this matters

Water cut is no longer treated as permanently fixed.

As water cut changes, it subsequently changes:

* water production;
* total liquid production;
* mixture density;
* hydrostatic pressure loss;
* therefore surface pressure behaviour.

This creates a chain of physical dependency.

---

# 21. True GOR

### Code

```python
true_gor = base_gor * (
    1 + RNG.normal(0, 0.02)
)
```

The well's base GOR is perturbed with approximately 2% random variation.

Mathematically:

$$
GOR(t)
=
GOR_0(1+\epsilon)
$$

where:

$$
\epsilon\sim N(0,0.02)
$$

### Engineering interpretation

Gas-oil ratio is allowed to fluctuate rather than remaining perfectly constant.

However, the current implementation does **not** make GOR explicitly evolve with reservoir pressure or depletion.

Therefore, this is a stochastic GOR variation rather than a full reservoir PVT/GOR model.

---

# 22. True Oil Rate

### Code

```python
true_oil_rate = max(
    50.0,
    base_oil
    * decline_factor
    * (1 + RNG.normal(0, 0.02))
)
```

The underlying oil rate is:

$$
Q_o(t)
=
Q_{o,0}
e^{-Dt}
(1+\epsilon)
$$

with approximately 2% random variation.

The rate is prevented from falling below:

$$
50\text{ bpd}
$$

using `max()`.

### Engineering interpretation

Oil production declines over time while experiencing stochastic fluctuations.

The 50-bpd floor prevents unrealistic negative or near-zero values caused by the random component.

---

# 23. True Gas Rate

### Code

```python
true_gas_rate = (
    true_oil_rate
    * true_gor
    / 1000.0
)
```

The gas rate is calculated from oil rate and GOR.

Because GOR is in scf/bbl and gas rate is desired in Mscf/d:

$$
Q_g
=
\frac{Q_o\times GOR}{1000}
$$

For example, if:

$$
Q_o=2,000\text{ bpd}
$$

and:

$$
GOR=500\text{ scf/bbl}
$$

then:

$$
Q_g
=
\frac{2,000\times500}{1000}
=
1,000\text{ Mscf/d}
$$

This is a direct and physically interpretable phase relationship.

---

# 24. True Water Rate

### Code

```python
true_water_rate = (
    true_oil_rate
    * true_water_cut
    / (1 - true_water_cut)
)
```

This follows directly from the definition of water cut.

Starting with:

$$
WC=\frac{Q_w}{Q_o+Q_w}
$$

rearranging gives:

$$
Q_w
=
\frac{Q_oWC}{1-WC}
$$

which is exactly what the code implements.

For example, at:

$$
Q_o=2,000\text{ bpd}
$$

and:

$$
WC=0.50
$$

the water rate becomes:

$$
Q_w=
\frac{2000(0.50)}{1-0.50}
=
2000\text{ bpd}
$$

---

# 25. True Liquid Rate

### Code

```python
true_liquid_rate = (
    true_oil_rate
    + true_water_rate
)
```

The total liquid rate is:

$$
Q_L=Q_o+Q_w
$$

This is the latent production quantity that the VFM liquid model ultimately attempts to estimate.

---

# 26. Choke State at the Current Hour

### Code

```python
choke_64th = choke_series[h]
```

The previously generated choke schedule is now applied to the current hourly observation.

This means the simulated well can change operating condition at selected times.

---

# 27. Physics Injection: IPR and VLP Proxy

The code labels the next section:

```python
# --- PHYSICS INJECTION: IPR & VLP PROXY ---
```

This is the central physics-informed section.

It consists of three stages:

```text
True liquid production
        ↓
Simplified PI/IPR relationship
        ↓
Pwf
        ↓
Hydrostatic pressure loss
        ↓
Surface pressure
```

---

# 28. Flowing Bottom-Hole Pressure

### Code

```python
pwf = (
    current_res_pressure
    - (true_liquid_rate / productivity_index)
)
```

This implements:

$$
P_{wf}
=
P_r-\frac{Q_L}{J}
$$

which follows the linear PI relationship:

$$
Q_L=J(P_r-P_{wf})
$$

Rearranging:

$$
P_{wf}=P_r-\frac{Q_L}{J}
$$

### Engineering interpretation

Higher production at the same productivity index requires greater pressure drawdown.

Likewise, a higher productivity index allows the well to produce the same rate with less drawdown.

This is a reasonable simplified inflow relationship.

---

# 29. Critical Limitation of the Current IPR Implementation

There is an important subtlety here.

The code does **not solve production from the IPR**.

Instead, it first generates:

```python
true_liquid_rate
```

and then calculates:

```python
pwf
```

from that rate.

Therefore, the direction is:

```text
True production
      ↓
PI relationship
      ↓
Pwf
```

rather than:

```text
Reservoir pressure
       +
IPR
       +
Pwf/outflow constraints
       ↓
Solved production
```

This means the IPR is being used as a **physics-consistency relationship**, rather than as the engine that determines the production rate.

This is perfectly reasonable for the current PoC, but it should be described accurately.

---

# 30. Fluid Specific Gravity

### Code

```python
sg_oil, sg_water = 0.85, 1.05
```

The generator assumes:

$$
SG_o=0.85
$$

for oil and:

$$
SG_w=1.05
$$

for water.

These are simplified constant specific gravities.

---

# 31. Mixture Specific Gravity

### Code

```python
sg_mixture = (
    (true_water_cut * sg_water)
    + ((1 - true_water_cut) * sg_oil)
)
```

This is a weighted approximation:

$$
SG_{mix}
=
WC(SG_w)
+
(1-WC)(SG_o)
$$

As water cut increases, mixture specific gravity increases.

For example, at:

$$
WC=0.20
$$

$$
SG_{mix}
=
0.20(1.05)+0.80(0.85)
=
0.89
$$

At:

$$
WC=0.80
$$

$$
SG_{mix}
=
0.80(1.05)+0.20(0.85)
=
1.01
$$

Therefore, a higher-water-cut well produces a heavier simulated liquid mixture.

---

# 32. Hydrostatic Tubing Pressure Drop

### Code

```python
hydrostatic_drop = (
    0.433
    * well_depth_ft
    * sg_mixture
)
```

The equation is:

$$
\Delta P_h
=
0.433\times H\times SG_{mix}
$$

where:

* \(\Delta P_h\) = hydrostatic pressure gradient;
* \(H\) = well depth in feet;
* \(SG_{mix}\) = mixture specific gravity.

The coefficient 0.433 psi/ft corresponds approximately to the pressure gradient of water with unit specific gravity.

### Engineering meaning

The deeper the well:

$$
H\uparrow
\Rightarrow
\Delta P_h\uparrow
$$

The heavier the fluid:

$$
SG_{mix}\uparrow
\Rightarrow
\Delta P_h\uparrow
$$

Because water is assumed denser than oil:

$$
WC\uparrow
\Rightarrow
SG_{mix}\uparrow
\Rightarrow
\Delta P_h\uparrow
$$

This is the main mechanism by which **changing water cut affects the simulated surface pressure**.

---

# 33. Important VLP Qualification

The code comments call this:

```python
# 2. Hydrostatic Tubing Drop (VLP)
```

Technically, this should be described as a:

> **Simplified hydrostatic/VLP proxy**

rather than a complete VLP model.

A full VLP model would involve substantially more physics, including:

* multiphase flow regimes;
* frictional pressure losses;
* gas-liquid slip;
* PVT properties;
* acceleration effects;
* tubing geometry;
* temperature effects;
* pressure-dependent fluid properties.

V2 only captures the hydrostatic component through an approximate mixture density.

---

# 34. Base Wellhead Pressure

### Code

```python
base_whp = pwf - hydrostatic_drop
```

Conceptually:

$$
P_{WHP,base}
=
P_{wf}-\Delta P_h
$$

The generator therefore attempts to move from bottom-hole pressure toward surface pressure by subtracting the hydrostatic pressure requirement.

The conceptual chain becomes:

```text
Reservoir pressure
       ↓
PI relationship
       ↓
Pwf
       ↓
hydrostatic loss
       ↓
base WHP
```

---

# 35. Choke Backpressure

### Code

```python
choke_backpressure = (
    (64.0 / choke_64th)**1.5
    * RNG.uniform(40, 60)
)
```

This creates a synthetic relationship between choke size and backpressure.

The central term is:

$$
\left(\frac{64}{C}\right)^{1.5}
$$

where \(C\) is choke size in 64ths.

Therefore:

* smaller choke → larger backpressure;
* larger choke → smaller backpressure.

For example:

$$
C=32
$$

gives:

$$
(64/32)^{1.5}=2^{1.5}\approx2.83
$$

whereas:

$$
C=64
$$

gives:

$$
(64/64)^{1.5}=1
$$

The result is then multiplied by a random factor between 40 and 60 psi.

### Engineering purpose

This gives choke changes an observable effect on surface pressure.

A choke restriction therefore produces a pressure response in the simulated DCS data.

---

# 36. Critical Limitation: Choke Does Not Yet Control True Production

This is one of the most important findings from reading the actual code.

The generator calculates:

```python
true_liquid_rate
```

**before** calculating the choke effect.

The choke then influences:

```python
choke_backpressure
```

which affects WHP.

It does not feed back into:

```python
true_oil_rate
true_water_rate
true_liquid_rate
```

Therefore, the current causal structure is:

```text
Production ───────────────→ WHP
                               ↑
Choke ─→ Backpressure ─────────┘
```

rather than:

```text
Reservoir state
      +
Choke restriction
      ↓
Production rate
      ↓
Pressure response
```

This is a genuine limitation of V2.

It does not invalidate the Friday PoC, but it should be fixed in a more rigorous generator.

---

# 37. Wellhead Pressure

### Code

```python
wellhead_pressure_psi = max(
    100.0,
    base_whp
    + choke_backpressure
    + RNG.normal(0, 8)
)
```

The simulated WHP consists of:

$$
P_{WHP}
=
P_{wf}
-
\Delta P_h
+
P_{choke}
+
\epsilon
$$

with:

$$
\epsilon\sim N(0,8)
$$

and a minimum allowed WHP of:

$$
100\text{ psi}
$$

### Engineering interpretation

The surface pressure is intended to respond to:

* reservoir pressure;
* production/drawdown;
* fluid mixture density;
* well depth;
* choke restriction;
* measurement noise.

This is the principal synthetic surface-pressure signal used by the VFM.

---

# 38. A Further Issue Worth Flagging

The calculation can produce a negative or very low `base_whp`, particularly because the hydrostatic pressure calculation is quite large for a 5,500–7,500-ft well.

The subsequent:

```python
max(100.0, ...)
```

prevents negative WHP.

This means some simulated conditions may become **artificially pinned at 100 psi**.

That should be checked in the dataset.

A future generator should preferably avoid needing a hard pressure floor as frequently by using a more physically coupled wellbore-pressure model.

This is not a reason to throw away V2, but it is a reason to inspect the WHP distribution before calling the generator physically realistic.

---

# 39. Flowline Pressure

### Code

```python
flowline_pressure_psi = max(
    40.0,
    wellhead_pressure_psi * 0.45
    + RNG.normal(0, 5)
)
```

The flowline pressure is approximated as:

$$
P_{FL}
=
0.45P_{WHP}+\epsilon
$$

where:

$$
\epsilon\sim N(0,5)
$$

and a minimum of 40 psi is imposed.

### Engineering interpretation

The generator assumes that downstream flowline pressure is some fraction of wellhead pressure.

This is a simplified representation of downstream pressure loss/backpressure.

It is not a full pipeline hydraulic model.

---

# 40. Differential Pressure

### Code

```python
differential_pressure_psi = (
    wellhead_pressure_psi
    - flowline_pressure_psi
)
```

The equation is straightforward:

$$
\Delta P
=
P_{WHP}-P_{FL}
$$

This creates another observable pressure feature.

Because both pressures are linked, changes in WHP propagate into the differential-pressure signal.

---

# 41. Wellhead Temperature

### Code

```python
wellhead_temperature_F = (
    158
    + 3 * np.sin(
        2 * np.pi * hour_of_day / 24
    )
    + RNG.normal(0, 1.0)
)
```

This generates a baseline temperature of:

$$
158^\circ F
$$

with a sinusoidal daily fluctuation of amplitude:

$$
3^\circ F
$$

plus random noise with standard deviation:

$$
1^\circ F
$$

The sinusoidal term is:

$$
3\sin\left(\frac{2\pi h}{24}\right)
$$

### Engineering interpretation

This provides a plausible continuously changing temperature measurement.

### Limitation

The current temperature model is **not strongly coupled to production or fluid thermodynamics**.

It is essentially a synthetic sensor signal with temporal structure.

Therefore, it should not be presented as a full thermal model of the wellbore.

---

# 42. Vibration Index

### Code

```python
vibration_index = max(
    0.0,
    RNG.normal(2.0, 0.4)
)
```

Vibration is generated around a mean of approximately 2.0 with standard deviation 0.4.

Negative values are prevented.

### Why it exists

Vibration is part of the broader AEGIS architecture.

It is more relevant to:

> **Instrument Health**

than to the VFM itself.

The current VFM training pipeline explicitly excludes `vibration_index`.

This preserves separation between the AEGIS components.

---

# 43. Constructing the Observation Row

### Code

```python
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

    "is_production_test_hour": False,
    "test_oil_bpd": np.nan,
    "test_gas_mscfd": np.nan,
    "test_water_bpd": np.nan,
    "test_liquid_bpd": np.nan,
    "test_water_cut_pct": np.nan,
}
```

This creates the observable dataset row.

The distinction between the two groups of variables is important.

### Continuous operational variables

```text
wellhead_pressure
flowline_pressure
differential_pressure
wellhead_temperature
vibration
choke
```

These are available every hour.

### Production-test variables

```text
test_oil
test_gas
test_water
test_liquid
test_water_cut
```

These remain empty except during a production-test event.

This distinction mirrors the VFM problem:

> Continuous signals are available frequently, while direct production measurements are only available periodically.

---

# 44. Why the Test Columns Begin as NaN

### Code

```python
"test_oil_bpd": np.nan,
"test_gas_mscfd": np.nan,
"test_water_bpd": np.nan,
"test_liquid_bpd": np.nan,
"test_water_cut_pct": np.nan,
```

`NaN` means no physical test measurement exists at that hour.

This is preferable to filling every hour with a repeated production-test value because it preserves the difference between:

```text
continuous DCS observation
```

and:

```text
periodic physical measurement
```

---

# 45. Routine Production Test

### Code

```python
if hour_of_day == TEST_START_HOUR:
```

Since:

```python
TEST_START_HOUR = 6
```

a production test occurs at 06:00 every simulated day.

This produces:

$$
60\text{ tests per well}
$$

and therefore:

$$
60\times10=600
$$

simulated production-test observations across the entire field.

This is consistent with the 10-well × 60-day structure.

---

# 46. Independent Oil Measurement Error

### Code

```python
meas_oil = max(
    0.0,
    true_oil_rate
    * RNG.normal(1.0, 0.025)
)
```

The physical-test oil measurement is:

$$
Q_{oil,test}
=
Q_{oil,true}
\times
N(1,0.025)
$$

The standard deviation corresponds to approximately 2.5% relative measurement noise.

The `max(0.0, ...)` prevents negative measurements.

---

# 47. Independent Gas Measurement Error

### Code

```python
meas_gas = max(
    0.0,
    true_gas_rate
    * RNG.normal(1.0, 0.040)
)
```

The gas measurement receives approximately 4% relative noise:

$$
Q_{gas,test}
=
Q_{gas,true}
\times
N(1,0.040)
$$

This is independent from the oil and water measurement errors.

---

# 48. Independent Water Measurement Error

### Code

```python
meas_water = max(
    0.0,
    true_water_rate
    * RNG.normal(1.0, 0.035)
)
```

Water receives approximately 3.5% relative measurement noise:

$$
Q_{water,test}
=
Q_{water,true}
\times
N(1,0.035)
$$

Again, the measurement error is independently generated.

---

# 49. Why Independent Measurement Noise Matters

The three phase measurements are not simply copied from the latent values.

Instead:

```text
True oil
   ↓
measurement error
   ↓
Measured oil

True water
   ↓
different measurement error
   ↓
Measured water

True gas
   ↓
different measurement error
   ↓
Measured gas
```

This is important because real measurements are imperfect.

It also prevents the physical-test columns from being exact replicas of the hidden truth.

---

# 50. Test Liquid Rate

### Code

```python
meas_liquid = meas_oil + meas_water
```

The measured liquid rate is:

$$
Q_{L,test}
=
Q_{oil,test}+Q_{water,test}
$$

This is internally consistent with the phase measurements.

---

# 51. Test Water Cut

### Code

```python
meas_wc_pct = (
    meas_water / meas_liquid * 100.0
    if meas_liquid > 0
    else 0.0
)
```

Water cut is calculated from the **measured** oil and water quantities:

$$
WC_{test}
=
\frac{Q_{water,test}}
{Q_{oil,test}+Q_{water,test}}
\times100
$$

This is an important modelling decision.

The generator does not independently add noise to water cut.

Instead, water cut emerges mathematically from the noisy phase measurements.

This is more internally consistent.

---

# 52. Recording the Physical Test

### Code

```python
row["is_production_test_hour"] = True
row["test_oil_bpd"] = round(meas_oil, 1)
row["test_gas_mscfd"] = round(meas_gas, 1)
row["test_water_bpd"] = round(meas_water, 1)
row["test_liquid_bpd"] = round(meas_liquid, 1)
row["test_water_cut_pct"] = round(meas_wc_pct, 2)
```

The previously empty test columns are populated.

The resulting row now contains both:

* continuous operational measurements;
* a physical-test reference measurement.

This is the key supervised-learning observation.

---

# 53. Adding the Row

### Code

```python
records.append(row)
```

Every hourly observation is appended to the master list.

After all wells and all hours have been processed, the list contains:

$$
14,400
$$

rows.

---

# 54. DataFrame Construction

### Code

```python
df = pd.DataFrame(records)
```

The list of dictionaries becomes a pandas DataFrame.

This creates the final structured dataset.

---

# 55. CSV Export

### Code

```python
df.to_csv(
    "aegis_data_v2.csv",
    index=False
)
```

The complete dataset is exported as:

`aegis_data_v2.csv`

The index is not included because the DataFrame index has no engineering meaning.

---

# 56. Final Statistics

### Code

```python
print(
    f"aegis_data_v2.csv written: {len(df):,} rows."
)

print(
    f"Total production tests simulated: "
    f"{df['is_production_test_hour'].sum()}"
)
```

The first statement confirms the dataset size.

The second counts the rows identified as production-test observations.

For the current configuration, the expected values are approximately:

$$
14,400\text{ total rows}
$$

and:

$$
600\text{ production-test rows}
$$

because:

$$
10\text{ wells}\times60\text{ daily tests}=600
$$

---

# 57. Complete Causal Structure of V2

The code can now be understood as a chain rather than a collection of equations.

```text
                    WELL DEFINITION
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        Base oil      Base GOR    Base WC
             │           │           │
             └───────────┼───────────┘
                         ▼
                Well-specific parameters
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Reservoir     PI         Depth
          decline
              │
              ▼
       Current reservoir pressure
              │
              ├─────────────────┐
              ▼                 ▼
       True oil rate       True water cut
              │                 │
              ▼                 ▼
         True GOR          True water rate
              │                 │
              ▼                 │
        True gas rate           │
              │                 │
              └────────┬────────┘
                       ▼
                True liquid rate
                       │
                       ▼
                 Simplified PI
                       │
                       ▼
                      Pwf
                       │
              ┌────────┴─────────┐
              ▼                  ▼
        Fluid mixture        Choke state
              │                  │
              ▼                  ▼
        Mixture SG          Backpressure
              │                  │
              └────────┬─────────┘
                       ▼
                Surface pressure
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           WHP        DP        WHT
                       │
                       ▼
                  DCS dataset

At test time:

True oil ──noise──→ Test oil
True water ─noise─→ Test water
True gas ──noise──→ Test gas
                      │
                ┌─────┴─────┐
                ▼           ▼
          Test liquid    Test WC
```

---

# 58. What the ML Model Actually Sees

The generator contains considerably more information than is given to the VFM.

### Hidden from the model

```text
initial_reservoir_pressure
current_res_pressure
decline_rate
productivity_index
well_depth_ft
true_water_cut
true_gor
true_oil_rate
true_gas_rate
true_water_rate
true_liquid_rate
pwf
```

### Potentially observable/model-eligible

```text
well_id
choke_size_64th
wellhead_pressure_psi
flowline_pressure_psi
differential_pressure_psi
wellhead_temperature_F
is_legacy_reactivated_well
```

### Used as calibration/reference targets

```text
test_liquid_bpd
test_gas_mscfd
```

### Reserved for another AEGIS component

```text
vibration_index
```

This separation is fundamental.

The VFM is not being given the answer.

It is being given surface measurements from which it attempts to infer the answer.

---

# 59. Relationship to the VFM Training Pipeline

The generator produces:

```text
aegis_data_v2.csv
```

The VFM training pipeline subsequently performs feature engineering.

For example:

$$
\Delta WHP_{1h}
=
WHP_t-WHP_{t-1}
$$

and:

$$
\Delta WHP_{24h}
=
WHP_t-WHP_{t-24}
$$

These features provide the model with temporal information that is not represented by a single raw pressure value.

The final architecture therefore becomes:

```text
V2 Generator
     ↓
Synthetic DCS
     ↓
Temporal feature engineering
     ↓
Physical-test rows
     ↓
Chronological train/validation/test split
     ↓
XGBoost
     ├── Total liquid
     └── Gas
```

---

# 60. Why the Generator Is "Physics-Informed"

The term "physics-informed" is justified because the data are not generated entirely independently.

There are explicit engineering relationships such as:

$$
Q_L=Q_o+Q_w
$$

$$
WC=\frac{Q_w}{Q_o+Q_w}
$$

$$
Q_w=\frac{Q_oWC}{1-WC}
$$

$$
Q_g=\frac{Q_oGOR}{1000}
$$

$$
P_{wf}=P_r-\frac{Q_L}{J}
$$

$$
SG_{mix}=WC\,SG_w+(1-WC)SG_o
$$

$$
\Delta P_h=0.433HSG_{mix}
$$

These relationships impose structure on the synthetic environment.

However, V2 should **not** be described as:

* a full reservoir simulator;
* a full multiphase flow simulator;
* a full VLP simulator;
* a PINN;
* a history-matched field model.

It is more accurately:

> **A synthetic time-series generator incorporating simplified petroleum-engineering relationships.**

---

# 61. The Most Important Methodological Limitation

The strongest limitation in the current generator is the direction of causality.

Production is generated first:

```text
true_oil_rate
true_water_rate
true_gas_rate
true_liquid_rate
```

and surface pressure is subsequently generated from those quantities.

That is useful for constructing a controlled dataset, but it means the generator is not solving the entire physical system.

In particular:

```text
Choke
  ↓
Backpressure
  ↓
WHP
```

but not:

```text
Choke
  ↓
Backpressure
  ↓
Production rate
```

Likewise:

```text
Production
  ↓
Pwf
```

rather than:

```text
Reservoir + IPR + VLP + choke
  ↓
Production
```

This distinction should be explicitly acknowledged.

---

# 62. What V2 Successfully Adds Compared With V1

The V2 generator represents a meaningful methodological improvement.

### V1

The structure was closer to:

```text
Generate production
       ↓
Generate surface signals
       ↓
Add test noise
```

### V2

The structure becomes:

```text
Well characteristics
       ↓
Reservoir state
       ↓
Production state
       ↓
Fluid composition
       ↓
Pressure relationship
       ↓
Hydrostatic effect
       ↓
Choke/backpressure
       ↓
Surface signals
       ↓
Noisy physical test
```

The latter has substantially more internal structure.

---

# 63. What Should Be Improved in V2.1/V3

The next version should not primarily focus on making the XGBoost model more sophisticated.

The generator itself should become more physically coupled.

The highest-value improvements are:

### 1. Couple choke to production

Instead of generating production independently of choke:

$$
q=f(P_r,IPR,VLP,choke)
$$

should determine the production state.

### 2. Improve the IPR

A future version could use a nonlinear IPR where appropriate rather than the current linear PI relationship.

### 3. Improve VLP

Replace the hydrostatic-only proxy with a more complete simplified tubing outflow model.

### 4. Improve PVT

Introduce pressure-dependent:

* oil properties;
* gas properties;
* formation volume factors;
* solution GOR;
* viscosity.

### 5. Improve temperature coupling

WHT could respond to production and fluid conditions rather than being predominantly diurnal.

### 6. Make test interval genuinely configurable

The test scheduler should actually use:

```python
TEST_INTERVAL_H
```

### 7. Introduce realistic sensor behaviour

Future versions could include:

* sensor drift;
* missing values;
* spikes;
* stuck sensors;
* calibration offsets.

This would also help the Instrument Health component.

### 8. Improve stochastic production behaviour

Instead of independent hourly noise:

$$
q_t=q_{trend}(t)(1+\epsilon_t)
$$

the generator could introduce temporally correlated disturbances.

---

# 64. The Generator's Appropriate Claim

The strongest defensible description of the current generator is:

> **AEGIS V2 is a physics-informed synthetic production environment that generates heterogeneous well-level time-series data by combining simplified reservoir depletion, productivity-index-based pressure relationships, evolving water cut, gas-oil behaviour, fluid-mixture hydrostatics, choke-related surface backpressure and noisy periodic production-test measurements.**

The generator demonstrates that the VFM pipeline can operate in a controlled environment.

It does **not** demonstrate that the same performance will necessarily be achieved on Renaissance field data.

---

# 65. Final Engineering Interpretation

The V2 generator essentially creates an artificial field in which the following relationship exists:

$$
\boxed{
\text{Well Characteristics}
\rightarrow
\text{Reservoir State}
\rightarrow
\text{Phase Production}
\rightarrow
\text{Wellbore Pressure}
\rightarrow
\text{Surface Signals}
}
$$

with periodic observations:

$$
\boxed{
\text{True Production}
\rightarrow
\text{Measurement Noise}
\rightarrow
\text{Production Test}
}
$$

The VFM then reverses part of this information flow:

$$
\boxed{
\text{Surface Signals}
\rightarrow
\text{Machine Learning}
\rightarrow
\widehat{Q_L},\widehat{Q_g}
}
$$

and the liquid estimate can subsequently be decomposed using the latest physical-test water cut:

$$
\hat Q_w=\hat Q_L WC
$$

$$
\hat Q_o=\hat Q_L(1-WC)
$$

The fundamental purpose of the generator is therefore to create a **controlled inverse problem**:

> The generator knows the underlying production state; the VFM only sees the observable surface information and must learn to estimate the production state from it.

That is the core technical idea behind AEGIS Component 1.
