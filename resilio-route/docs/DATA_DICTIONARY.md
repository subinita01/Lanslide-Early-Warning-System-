# Data Dictionary

Every field used in Resilio-Route, with units, valid ranges, and notes.

---

## Raw Sensor Fields

| Field | Type | Unit | Valid range | Source |
|-------|------|------|-------------|--------|
| `node_id` | string | — | `NM-01` … `NM-99` | Node config |
| `timestamp` | datetime (UTC) | — | 2024-01-01 → now | Node RTC or gateway assign |
| `tilt_deg` | float | degrees | 0.0 – 5.0 | MEMS accelerometer |
| `moisture_d1` | float | % saturation | 0 – 100 | Capacitive probe, depth 10 cm |
| `moisture_d2` | float | % saturation | 0 – 100 | Capacitive probe, depth 20 cm |
| `moisture_d3` | float | % saturation | 0 – 100 | Capacitive probe, depth 30 cm |
| `moisture_d4` | float | % saturation | 0 – 100 | Capacitive probe, depth 40 cm |
| `moisture_d5` | float | % saturation | 0 – 100 | Capacitive probe, depth 60 cm |
| `moisture_d6` | float | % saturation | 0 – 100 | Capacitive probe, depth 80 cm |
| `pore_pressure` | float | kPa | 0 – 100 | Piezometer |
| `rainfall_1hr` | float | mm/hr | 0 – 300 | Tipping bucket gauge |
| `battery_pct` | int | % | 0 – 100 | Battery voltage divider |
| `rssi_dbm` | int | dBm | -130 – 0 | LoRa radio RSSI |

---

## Engineered Features

These are computed in `ml/feature_engineering.py` from the raw fields above.

| Feature | Formula | Unit | Physical meaning |
|---------|---------|------|-----------------|
| `tilt_rate` | `(T_t - T_{t-1}) / 5` | °/min | Slope acceleration |
| `tilt_rolling_mean_3hr` | `mean(tilt, t-35..t)` | degrees | Smoothed trend |
| `tilt_variance_1hr` | `var(tilt, t-11..t)` | degrees² | Instability signal |
| `saturation_ratio` | `mean(d1..d6) / 100` | 0–1 | Overall soil fullness |
| `moisture_gradient` | `d1 - d6` | % | Top-to-bottom water flow |
| `pore_spike` | `P_t - P_{t-6}` | kPa | Sudden pressure rise (30 min) |
| `rainfall_6hr` | `sum(rain, t-5..t)` | mm | Cumulative rainfall |
| `rainfall_24hr` | `sum(rain, t-23..t)` | mm | Antecedent rainfall |
| `rain_tilt_product` | `rainfall_1hr × tilt_deg` | mm/hr·° | Joint danger signal |

---

## Labels

| Value | Integer | Factor of Safety | Physical meaning |
|-------|---------|-----------------|-----------------|
| `GREEN` | 0 | FS ≥ 1.3 | Slope is stable |
| `YELLOW` | 1 | 1.0 ≤ FS < 1.3 | Marginal — monitor closely |
| `RED` | 2 | FS < 1.0 | Failure zone — imminent danger |

Label sources (recorded in `labeled_events.label_source`):

| Source | Description |
|--------|-------------|
| `physics` | Factor of Safety computed from pore pressure + soil params |
| `historical` | Retroactively assigned from NDMA / NHIDCL event records |
| `expert` | Manual label from geotechnical expert (active learning) |
| `active` | Selected by uncertainty sampling, confirmed by expert |

---

## Navigator Output

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `GREEN` / `YELLOW` / `RED` |
| `label_int` | int | 0 / 1 / 2 |
| `p_green` | float 0–1 | Ensemble probability of GREEN |
| `p_yellow` | float 0–1 | Ensemble probability of YELLOW |
| `p_red` | float 0–1 | Ensemble probability of RED |
| `confidence` | float 0–1 | `max(p_green, p_yellow, p_red)` |
| `rf_proba` | list[float] | RF sub-model probabilities |
| `cnn_proba` | list[float] | CNN sub-model probabilities |

---

## Node Registry Fields

| Field | Unit | Description |
|-------|------|-------------|
| `slope_angle_deg` | degrees | Hillside angle from DEM + survey |
| `soil_cohesion` | kPa | From geotechnical borehole (update per site) |
| `friction_angle` | degrees | Internal friction from lab test |
| `deployed_at` | datetime UTC | When node was first activated |

---

## Factor of Safety Model

```
FS = (c' + (γz·cos²β - u)·tan φ') / (γz·sinβ·cosβ)

Where:
  c'  = soil cohesion (kPa)         — from site survey, default 8.0
  γ   = unit weight of soil (kN/m³) — from site survey, default 19.0
  z   = depth of failure plane (m)  — from site survey, default 1.5
  β   = slope angle (degrees)       — from DEM / tilt baseline
  u   = pore water pressure (kPa)   — measured live by piezometer
  φ'  = friction angle (degrees)    — from site survey, default 28.0

FS < 1.0  → slope will fail     → RED
FS < 1.3  → marginal stability  → YELLOW
FS ≥ 1.3  → stable              → GREEN
```

Sensitivity: pore pressure `u` is the most dynamic input (direct sensor).
Soil params `c'`, `γ`, `z`, `φ'` are site constants set once after survey.
