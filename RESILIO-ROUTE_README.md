# RESILIO-ROUTE
## Predicting Landslides. Protecting Logistics.

> *"Landslides are not just natural disasters — they are preventable logistics failures."*

A highway-native Landslide Early Warning System (LEWS) designed for Northeast India's NH corridors. Resilio-Route connects low-cost IoT sensor networks to a machine-learning prediction engine and real-time driver-facing alert infrastructure — protecting lives, supply chains, and logistics operations during monsoon season.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [Why Existing Solutions Fail](#3-why-existing-solutions-fail)
4. [System Architecture](#4-system-architecture)
5. [Hardware Stack](#5-hardware-stack)
6. [ML Pipeline — The Navigator](#6-ml-pipeline--the-navigator)
7. [Build Plan — Stepwise](#7-build-plan--stepwise)
8. [Milestone Roadmap](#8-milestone-roadmap)
9. [API Documentation](#9-api-documentation)
10. [Data Schemas](#10-data-schemas)
11. [Cold-Start Strategy](#11-cold-start-strategy)
12. [Transfer Learning & Generalisation](#12-transfer-learning--generalisation)
13. [Deployment Guide](#13-deployment-guide)
14. [Business Model](#14-business-model)
15. [References](#15-references)

---

## 1. Project Overview

Resilio-Route is a four-layer system:

```
PERCEPTION  →  TRANSMISSION  →  AGGREGATION  →  ACTION
(Sensors)      (LoRa Mesh)      (Cloud AI)       (LED + Routing)
```

| Layer | Technology | Purpose |
|---|---|---|
| Perception | IoT ground nodes (ESP32/SAMD21) | Measure slope, moisture, pressure |
| Transmission | LoRa SX1278 mesh, 15 km range | Move data without cell towers |
| Aggregation | Highway depot gateway + cloud | Run the Navigator ML model |
| Action | LED km-marker boards + API | Warn drivers, reroute fleets |

**Target geography:** National Highway corridors in Mizoram, Manipur, Sikkim, Nagaland, Arunachal Pradesh, and Meghalaya.

**Target users:** NHIDCL highway authorities, logistics fleet operators, fuel and essential goods transporters.

---

## 2. The Problem

Every monsoon season, Northeast India faces a compounding crisis across three dimensions:

### Monsoon-Driven Isolation
Single-road corridors like NH-306 (Mizoram), NH-2 (Manipur), and NH-10 (Sikkim) are the only surface links to millions of people. A single landslide event can isolate an entire state for 48–72 hours.

### Economic Shock
- Fuel prices double within 48 hours of a major closure
- Perishable goods rot in stationary trucks
- Daily supply chains for medicines, vegetables, and LPG break down
- Estimated economic loss: ₹2–8 crore per major closure event

### Human Risk
Drivers make life-or-death routing decisions based on experience and word-of-mouth — not real-time hazard data. They enter sinking zones with zero warning, often in the dark and heavy rain.

**The core gap:** No system currently connects landslide prediction to real-time highway protection. Existing systems predict; none protect.

---

## 3. Why Existing Solutions Fail

| Approach | Cost | Fatal Flaw |
|---|---|---|
| Classical geotechnical instruments | ₹5L+ per site | Sparse coverage; not scalable |
| Manual inspection & reporting | Low cost | Reactive only — after failure begins |
| Satellite monitoring (InSAR) | High | Cloud-affected; delayed 6–24 hours |
| Research-grade LEWS | Varies | Pilot stage; not highway-integrated |

**The result:** High cost, low coverage, delayed alerts, and zero real-time highway protection.

**The precedent that proves the alternative works:**

- **SitkaNet, Alaska (2021):** ~$1,000/node, LoRa-based, 5-minute update frequency, soil moisture at 6 depths — proven at scale.
- **Inform@Risk, Colombia (2021):** €100–250/node, 2+ year battery life, 15 km LoRa mesh range, community-scale warnings.

Low-cost LoRa-based landslide monitoring is **not experimental — it is proven, reliable, and scalable**.

---

## 4. System Architecture

### 4.1 Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RESILIO-ROUTE SYSTEM                         │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  PERCEPTION  │    │ TRANSMISSION │    │    AGGREGATION       │  │
│  │              │    │              │    │                      │  │
│  │  Ground Node │───▶│  LoRa Mesh   │───▶│  Depot Gateway       │  │
│  │  (ESP32)     │    │  (SX1278)    │    │  (Raspberry Pi 4)    │  │
│  │              │    │  15 km range │    │  ↓ GSM/Satellite     │  │
│  │  Sensors:    │    │  Node-to-node│    │  ↓ Cloud (AWS/Azure) │  │
│  │  - Tilt MEMS │    │  mesh routing│    │  ↓ Navigator ML      │  │
│  │  - Moisture  │    │              │    │  ↓ Risk score/5 min  │  │
│  │    (6 depth) │    │              │    │                      │  │
│  │  - Pore P.   │    │              │    └──────────┬───────────┘  │
│  │  - Rainfall  │    │              │               │              │
│  └──────────────┘    └──────────────┘               │              │
│                                                      ▼              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                          ACTION                               │  │
│  │                                                               │  │
│  │  LED Board (km-marker)        Risk Intelligence API           │  │
│  │  🟢 GREEN — safe to drive     → Fleet management systems      │  │
│  │  🟡 YELLOW — caution          → Logistics routing partners    │  │
│  │  🔴 RED — stop, reroute       → NHIDCL control rooms         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
Sensor reading (every 5 min)
        │
        ▼
Node-level preprocessing
  - Timestamp + node_id
  - Basic sanity check (range validation)
  - Local EEPROM buffer (if LoRa link drops)
        │
        ▼ LoRa packet (encrypted, 52-byte payload)
        │
        ▼
Depot Gateway
  - Collects packets from all nodes in range
  - Assembles into JSON time-series
  - Uplinks via GSM (primary) / Iridium satellite (backup)
        │
        ▼ HTTPS POST to Cloud API
        │
        ▼
Cloud Ingest Layer
  - Validates + stores raw readings (TimescaleDB)
  - Triggers Navigator inference
        │
        ▼
Navigator ML Engine
  - Feature engineering (18 derived features)
  - Random Forest (tabular) + CNN (time-series)
  - Ensemble → P(GREEN), P(YELLOW), P(RED)
        │
        ▼
Alert + Routing Engine
  - Applies safety-first thresholds
  - Pushes classification to LED board controller
  - Updates Risk Intelligence API endpoint
  - Triggers fleet webhook notifications
```

### 4.3 Safety-First Routing Formula

```
Cost = Distance × (1 + Risk_Local)

Where Risk_Local = P(RED) for that highway segment

Example:
  Route A: 50 km,  Risk = 0.10  →  Cost = 55
  Route B: 60 km,  Risk = 0.80  →  Cost = 108

  Algorithm selects Route A — the longer but safer road.
```

---

## 5. Hardware Stack

### 5.1 Ground Node BOM (per node, target cost ≤ €250)

| Component | Spec | Role |
|---|---|---|
| MCU | ESP32 or SAMD21 | Main controller, deep-sleep capable |
| Tilt sensor | MEMS (0.003°–0.02° precision) | Slope angle + rate of change |
| Soil moisture probes | Capacitive, 6 units | Moisture at depths: 10, 20, 30, 40, 60, 80 cm |
| Piezometer | 0–100 kPa range | Pore water pressure |
| Rain gauge | Tipping bucket, 0.2 mm resolution | Rainfall intensity |
| LoRa radio | SX1278 / LoRaWAN module | Long-range mesh communication |
| Power | 5W solar + 18 Ah LiPo | 2+ years autonomous operation |
| Enclosure | IP67 rated | Field durability |

### 5.2 LoRa Packet Format (52 bytes)

```c
typedef struct {
    uint32_t node_id;          // 4 bytes — unique node identifier
    uint32_t timestamp;        // 4 bytes — Unix epoch
    int16_t  tilt_x100;        // 2 bytes — tilt in 0.01° units
    int16_t  tilt_rate_x1000;  // 2 bytes — rate in 0.001°/min units
    uint8_t  moisture[6];      // 6 bytes — % saturation at each depth
    uint16_t pore_pressure;    // 2 bytes — kPa × 10
    uint16_t rainfall_1hr;     // 2 bytes — mm × 10
    uint16_t rainfall_6hr;     // 2 bytes — mm × 10
    uint8_t  battery_pct;      // 1 byte  — battery %
    uint8_t  rssi;             // 1 byte  — signal quality
    uint8_t  risk_class;       // 1 byte  — local computed class (0/1/2)
    uint8_t  hop_count;        // 1 byte  — mesh hops to gateway
    uint8_t  crc[8];           // 8 bytes — CRC32 + message ID
    uint8_t  reserved[10];     // 10 bytes — future use
} NodePacket;                  // Total: 52 bytes
```

### 5.3 LED Warning Board

```
NH-06  KM-132
┌─────────────┐
│  ● GREEN    │  ← Safe to drive
│  ● YELLOW   │  ← Caution: hazard developing
│  ● RED      │  ← STOP: danger zone ahead
└─────────────┘

- No phone dependency (standalone operation)
- Solar + battery powered
- LoRa receiver for direct node alerts (backup path)
- Visible at 200 m in rain and fog (high-brightness LEDs)
```

---

## 6. ML Pipeline — The Navigator

### 6.1 Problem Formulation

**Type:** Supervised multi-class classification
**Input:** 18-feature vector (tabular) + 6×36 time-series matrix (temporal)
**Output:** P(GREEN), P(YELLOW), P(RED) → safety classification
**Lead time target:** ≥ 30 minutes before failure

### 6.2 Feature Engineering

```python
ENGINEERED_FEATURES = {
    # Tilt features
    'tilt':                  'raw slope angle (degrees)',
    'tilt_rate':             '(T_t - T_{t-1}) / 5 min  →  acceleration signal',
    'tilt_rolling_mean_3hr': 'mean(tilt, last 36 readings)  →  baseline drift',
    'tilt_variance_1hr':     'var(tilt, last 12 readings)  →  instability',

    # Moisture features
    'saturation_ratio':      'mean(M1..M6) / 100  →  overall soil fullness',
    'moisture_gradient':     'M1 - M6  →  top-to-bottom water percolation',
    'moisture_d1..d6':       'raw readings at each depth',

    # Pressure features
    'pore_pressure':         'raw kPa reading',
    'pore_spike':            'P_t - P_{t-6}  →  sudden pressure rise',

    # Rainfall features
    'rainfall_1hr':          'current intensity (mm/hr)',
    'rainfall_6hr':          'rolling 6-hour cumulative (mm)',
    'rainfall_24hr':         'rolling 24-hour cumulative (mm)',

    # Interaction features
    'rain_tilt_product':     'rainfall_1hr × tilt  →  joint danger signal',
}
```

### 6.3 Model A — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Handle class imbalance (GREEN ~94%, RED ~2%)
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_bal, y_bal)
# Output: rf.predict_proba(X) → [P(GREEN), P(YELLOW), P(RED)]
```

**How it decides:** At each node, the tree finds the feature + threshold with the highest Gini gain:

```
Gini(S)    = 1 - Σ p_k²
Gini Gain  = Gini(parent) - (n_L/n)×Gini(left) - (n_R/n)×Gini(right)
```

300 trees vote independently on bootstrapped subsets. Final probability = vote fraction across all trees.

### 6.4 Model B — Convolutional Neural Network

```python
import torch.nn as nn

class LandslideCNN(nn.Module):
    """
    Input shape: (batch, 6_channels, 36_timesteps)
    6 channels:  tilt, saturation, pore_pressure, rainfall, tilt_rate, moisture_gradient
    36 timesteps: last 3 hours at 5-min resolution
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Layer 1: spike detector (kernel=3, 15-min patterns)
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),

            # Layer 2: trend detector (kernel=7, 35-min patterns)
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),

            # Layer 3: acceleration signature (kernel=9, 45-min patterns)
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)   # → [GREEN, YELLOW, RED] logits
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))
```

**What each layer detects:**
- Layer 1 (kernel=3): Sudden sensor spikes in the last 15 minutes
- Layer 2 (kernel=7): Monotonic trends over 35 minutes
- Layer 3 (kernel=9): The characteristic pre-failure acceleration ramp

**Training:** Adam optimizer (lr=1e-3), CrossEntropyLoss with class weights, ReduceLROnPlateau scheduler, 50 epochs, batch size 64.

### 6.5 Ensemble Combination

```python
def ensemble_predict(X_tab, X_seq, rf_model, cnn_model,
                     rf_weight=0.6, cnn_weight=0.4):
    rf_proba  = rf_model.predict_proba(X_tab)           # shape (n, 3)
    cnn_proba = softmax(cnn_model(X_seq), dim=1).numpy() # shape (n, 3)

    combined = rf_weight * rf_proba + cnn_weight * cnn_proba

    labels = []
    for p in combined:
        if p[2] > 0.70:                          # P(RED) > 70%
            labels.append('RED')
        elif p[2] > 0.40 or p[1] > 0.50:        # caution zone
            labels.append('YELLOW')
        else:
            labels.append('GREEN')

    return combined, labels
```

**Threshold rationale:**
- RED at P > 0.70: safety-first; false alarm (unnecessary reroute) is far less costly than miss
- YELLOW at P > 0.40: gives advance notice for monitoring
- Weights 0.6/0.4: RF more reliable early (Milestone 1), CNN weight increases as temporal data accumulates

### 6.6 Deployment Checklist

```
Required before going live:
  ✅ Recall(RED)     ≥ 0.90   — miss fewer than 1 in 10 slides
  ✅ Precision(RED)  ≥ 0.60   — fewer than 4 in 10 alerts are false
  ✅ Lead time       ≥ 25 min — enough for a truck to stop safely
  ✅ Inference time  < 5 sec  — runs on edge gateway hardware
  ✅ Operates on 1 season of data — Milestone 2 constraint
```

---

## 7. Build Plan — Stepwise

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/resilio-route/navigator.git
cd navigator

# Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# requirements.txt includes:
# torch>=2.0, scikit-learn, imbalanced-learn, pandas, numpy
# scipy, joblib, fastapi, uvicorn, timescaledb-adapter
```

### Step 2: Hardware Node Firmware

```bash
# Flash ESP32 node firmware (PlatformIO)
cd firmware/
pio run --target upload --environment esp32

# Node configuration (config.h)
NODE_ID           = 0x000F    # Unique per node
LORA_FREQUENCY    = 865.0     # MHz (India ISM band)
SAMPLE_INTERVAL   = 300       # seconds (5 min)
GATEWAY_ADDR      = 0x0001    # Primary gateway address
MOISTURE_DEPTHS   = {10, 20, 30, 40, 60, 80}  # cm
```

### Step 3: LoRa Mesh Network Setup

```bash
# Gateway configuration (Raspberry Pi 4)
cd gateway/
python setup_gateway.py \
    --node-ids 1,2,3,...,20 \
    --lora-freq 865.0 \
    --gsm-apn airtelgprs.com \
    --cloud-endpoint https://api.resilio-route.in/v1/ingest \
    --uplink-interval 60   # seconds
```

### Step 4: Cloud Ingest Layer

```bash
# Start TimescaleDB (Docker)
docker run -d --name tsdb \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 timescale/timescaledb:latest-pg15

# Run ingest API
cd backend/
uvicorn ingest.main:app --host 0.0.0.0 --port 8000

# Verify ingest endpoint
curl -X POST https://api.resilio-route.in/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"node_id": 15, "timestamp": 1720000000, "tilt": 0.41, ...}'
```

### Step 5: Feature Engineering Pipeline

```bash
# Run offline feature engineering on collected data
python ml/feature_engineering.py \
    --input data/raw/node_readings.parquet \
    --output data/features/engineered.parquet \
    --window 36 \
    --horizon 6

# Verify feature distributions
python ml/eda.py --features data/features/engineered.parquet
```

### Step 6: Physics-Based Label Generation (Cold Start)

```bash
# Generate Factor-of-Safety labels from pore pressure + slope data
python ml/physics_labels.py \
    --features data/features/engineered.parquet \
    --cohesion 8.0 \       # kPa — from soil survey
    --friction-angle 28.0 \ # degrees
    --unit-weight 19.0 \   # kN/m³
    --failure-depth 1.5 \  # m
    --output data/labels/fs_labels.parquet
```

### Step 7: Train Random Forest (Model A)

```bash
python ml/train_rf.py \
    --features data/features/engineered.parquet \
    --labels data/labels/fs_labels.parquet \
    --n-estimators 300 \
    --max-depth 15 \
    --output models/rf_navigator_v1.pkl \
    --eval-report reports/rf_eval.json
```

### Step 8: Train CNN (Model B)

```bash
python ml/train_cnn.py \
    --sequences data/sequences/3hr_windows.pt \
    --labels data/labels/fs_labels.parquet \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --output models/cnn_navigator_v1.pt \
    --eval-report reports/cnn_eval.json
```

### Step 9: Ensemble Validation

```bash
python ml/evaluate_ensemble.py \
    --rf-model models/rf_navigator_v1.pkl \
    --cnn-model models/cnn_navigator_v1.pt \
    --test-data data/test/ \
    --rf-weight 0.6 \
    --cnn-weight 0.4 \
    --report reports/ensemble_eval.json

# Expected output:
# RED recall:     0.91
# RED precision:  0.68
# Lead time P50:  32 min
# Lead time P10:  26 min
```

### Step 10: Edge Deployment

```bash
# Export models for edge inference
python ml/export_edge.py \
    --rf-model models/rf_navigator_v1.pkl \
    --cnn-model models/cnn_navigator_v1.pt \
    --output deploy/

# Deploy to gateway
scp -r deploy/ pi@gateway-nh06-km132:/opt/navigator/

# Start inference loop on gateway
ssh pi@gateway-nh06-km132
cd /opt/navigator/
python run_navigator.py --interval 300 --led-port /dev/ttyUSB0
```

### Step 11: LED Board Integration

```bash
# LED board protocol (serial over RS-485)
# Command format: <NODE_ID>:<CLASS>:<CONFIDENCE>\n

# Example commands sent from gateway:
echo "15:GREEN:0.91" > /dev/ttyUSB0    # safe
echo "15:YELLOW:0.72" > /dev/ttyUSB0   # caution
echo "15:RED:0.88" > /dev/ttyUSB0      # danger
```

### Step 12: Risk Intelligence API Launch

```bash
cd backend/
uvicorn api.main:app --host 0.0.0.0 --port 443 \
    --ssl-keyfile certs/key.pem \
    --ssl-certfile certs/cert.pem

# Health check
curl https://api.resilio-route.in/v1/health
# → {"status": "ok", "model_version": "1.0.0", "nodes_online": 18}
```

---

## 8. Milestone Roadmap

### Milestone 1 — The Pilot Seed (Month 0–2)

**Focus:** Data baselining

**Actions:**
- Deploy 15–20 nodes on 2–3 critical highway sections (NH-306, NH-54)
- Establish LoRa mesh network with primary + satellite uplink backup
- Commission TimescaleDB ingest pipeline
- Generate physics-based (FS) labels from day 1
- Train and deploy RF v0.1 (physics labels only)

**Success criteria:**
- 18+ nodes continuously online (>90% uptime)
- Data arriving at cloud every 5 minutes per node
- RF v0.1 RED recall ≥ 0.70 on physics-labeled validation set
- Zero data gaps > 30 minutes

### Milestone 2 — Monsoon Calibration (Month 2–5)

**Focus:** ML training on live monsoon cycles

**Actions:**
- Collect full monsoon season of real sensor data
- Run active learning rounds (200 expert labels per round, 4 rounds)
- Mine NDMA + NHIDCL historical event database for retrospective labels
- Train CNN on time-series sequences
- Build ensemble (RF + CNN) and calibrate weights
- Validate 30-minute lead time against real events

**Success criteria:**
- RED recall ≥ 0.87 on held-out monsoon test set
- Lead time ≥ 25 minutes for ≥ 80% of RED events
- False alarm rate < 15%

### Milestone 3 — The Human Link (Month 5–8)

**Focus:** Driver-facing alert infrastructure

**Actions:**
- Install LED warning boards at km-markers on pilot corridors
- Integrate alert protocol with NHIDCL Standard Operating Procedures (SOPs)
- Launch Risk Intelligence API (beta) for 2 logistics partners
- Build fleet dashboard and webhook integration
- Conduct driver awareness programme with BRO and NHIDCL

**Success criteria:**
- LED boards operational on all pilot km-markers
- Alert-to-board latency < 60 seconds
- 2 logistics partners integrated via API
- Positive driver awareness feedback

### Milestone 4 — NE-Wide Scaling (Month 8–18)

**Focus:** Region-wide expansion

**Actions:**
- Transfer-learn Navigator for Manipur, Sikkim, Arunachal, Nagaland corridors (2-week fine-tune per region)
- Standardise node deployment kit (1-day install per site)
- Expand to all major NE highway corridors (NH-2, NH-10, NH-13, NH-29...)
- Onboard NHIDCL as primary subscriber (HaaS model)
- Scale Risk Intelligence API to full commercial fleet customer base

**Success criteria:**
- 200+ nodes deployed across NE India
- RED recall ≥ 0.90 across all deployed regions
- NHIDCL HaaS subscription signed
- 5+ fleet logistics companies on API

---

## 9. API Documentation

### Base URL

```
https://api.resilio-route.in/v1
```

### Authentication

```bash
# All requests require Bearer token
Authorization: Bearer <your_api_key>
```

### Endpoints

#### GET /risk/segment

Returns current risk classification for a highway segment.

```bash
curl https://api.resilio-route.in/v1/risk/segment \
  -H "Authorization: Bearer <key>" \
  -G \
  -d "highway=NH-306" \
  -d "km_from=120" \
  -d "km_to=145"
```

**Response:**
```json
{
  "segment": {
    "highway": "NH-306",
    "km_from": 120,
    "km_to": 145
  },
  "classification": "YELLOW",
  "probabilities": {
    "GREEN":  0.22,
    "YELLOW": 0.51,
    "RED":    0.27
  },
  "confidence": 0.82,
  "updated_at": "2024-07-14T14:35:00Z",
  "lead_time_min": 28,
  "nearest_node": "NM-07",
  "active_alerts": [
    {
      "type": "TILT_ACCELERATION",
      "severity": "MODERATE",
      "node_id": "NM-07",
      "detected_at": "2024-07-14T14:05:00Z"
    }
  ]
}
```

#### GET /risk/route

Returns risk scores for a full route with suggested alternatives.

```bash
curl https://api.resilio-route.in/v1/risk/route \
  -H "Authorization: Bearer <key>" \
  -G \
  -d "origin=Aizawl" \
  -d "destination=Silchar" \
  -d "vehicle_type=heavy_goods"
```

**Response:**
```json
{
  "recommended_route": {
    "segments": ["NH-306:0-80", "NH-54:0-45"],
    "total_distance_km": 125,
    "estimated_time_min": 185,
    "max_risk_segment": "NH-54:30-45",
    "max_risk_class": "YELLOW",
    "routing_cost": 137.5
  },
  "avoided_segments": [
    {
      "segment": "NH-306:132-148",
      "reason": "RED classification",
      "probability_RED": 0.83
    }
  ],
  "safe_to_depart": true,
  "expires_at": "2024-07-14T15:05:00Z"
}
```

#### GET /nodes/status

Returns health and latest readings for all nodes.

```bash
curl https://api.resilio-route.in/v1/nodes/status \
  -H "Authorization: Bearer <key>" \
  -G \
  -d "highway=NH-306"
```

#### POST /ingest (gateway-only)

Internal endpoint for gateway data upload.

```json
{
  "node_id": "NM-07",
  "timestamp": 1720964100,
  "readings": {
    "tilt_deg":       0.41,
    "tilt_rate":      0.05,
    "moisture_pct":   [82, 85, 84, 81, 79, 77],
    "pore_kpa":       3.2,
    "rainfall_1hr":   12.1,
    "rainfall_6hr":   48.3,
    "battery_pct":    87,
    "rssi_dbm":       -98
  }
}
```

#### POST /webhooks/register

Register a webhook for real-time alert delivery.

```json
{
  "url": "https://your-fleet-system.com/webhooks/landslide",
  "events": ["RED_ALERT", "YELLOW_ALERT", "ALL_CLEAR"],
  "highways": ["NH-306", "NH-54"],
  "secret": "your_webhook_secret"
}
```

**Webhook payload (RED alert):**
```json
{
  "event": "RED_ALERT",
  "segment": "NH-306:132-148",
  "probability_RED": 0.83,
  "lead_time_min": 31,
  "issued_at": "2024-07-14T14:35:00Z",
  "action_required": "REROUTE_IMMEDIATELY",
  "alternative_route": "NH-54 via Champhai"
}
```

---

## 10. Data Schemas

### Raw Sensor Reading

```sql
CREATE TABLE sensor_readings (
    id              BIGSERIAL PRIMARY KEY,
    node_id         VARCHAR(16) NOT NULL,
    highway         VARCHAR(16) NOT NULL,
    km_marker       DECIMAL(6,2),
    timestamp       TIMESTAMPTZ NOT NULL,
    tilt_deg        DECIMAL(6,4),
    tilt_rate       DECIMAL(7,5),
    moisture_d1     DECIMAL(5,2),
    moisture_d2     DECIMAL(5,2),
    moisture_d3     DECIMAL(5,2),
    moisture_d4     DECIMAL(5,2),
    moisture_d5     DECIMAL(5,2),
    moisture_d6     DECIMAL(5,2),
    pore_pressure   DECIMAL(6,3),
    rainfall_1hr    DECIMAL(6,2),
    rainfall_6hr    DECIMAL(7,2),
    battery_pct     SMALLINT,
    rssi_dbm        SMALLINT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- TimescaleDB hypertable (partition by time)
SELECT create_hypertable('sensor_readings', 'timestamp');
CREATE INDEX ON sensor_readings (node_id, timestamp DESC);
```

### Engineered Feature Row

```python
FEATURE_SCHEMA = {
    # Raw
    'node_id':              str,
    'timestamp':            datetime,
    'tilt':                 float,   # degrees
    'moisture_d1..d6':      float,   # % (0-100)
    'pore_pressure':        float,   # kPa
    'rainfall_1hr':         float,   # mm/hr

    # Derived
    'tilt_rate':            float,   # degrees/min
    'tilt_rolling_mean_3hr':float,   # degrees
    'tilt_variance_1hr':    float,   # degrees²
    'saturation_ratio':     float,   # 0-1
    'moisture_gradient':    float,   # M1 - M6 (%)
    'pore_spike':           float,   # kPa change over 30 min
    'rainfall_6hr':         float,   # mm
    'rainfall_24hr':        float,   # mm
    'rain_tilt_product':    float,   # mm/hr × degrees

    # Label
    'label':                str,     # GREEN / YELLOW / RED
    'label_int':            int,     # 0 / 1 / 2
    'factor_of_safety':     float,   # FS value (physics)
}
```

### Navigator Prediction Record

```python
PREDICTION_SCHEMA = {
    'prediction_id':   str,          # UUID
    'node_id':         str,
    'segment':         str,          # "NH-306:132-148"
    'timestamp':       datetime,
    'p_green':         float,        # 0-1
    'p_yellow':        float,        # 0-1
    'p_red':           float,        # 0-1
    'classification':  str,          # GREEN / YELLOW / RED
    'confidence':      float,        # max(p_green, p_yellow, p_red)
    'rf_proba':        list[float],  # [p_g, p_y, p_r] from RF
    'cnn_proba':       list[float],  # [p_g, p_y, p_r] from CNN
    'lead_time_est':   int,          # minutes to estimated failure
    'model_version':   str,          # "1.0.0"
    'inference_ms':    int,          # inference latency
}
```

---

## 11. Cold-Start Strategy

The cold-start problem: you need labeled data to train the model, but you have no labeled landslide events on day 1 of deployment.

### Strategy 1 — Physics-Based Labels (Immediate, Day 0)

Use the infinite slope model to auto-label every sensor reading from the first day:

```python
def factor_of_safety(c_prime, gamma, z, beta_deg, u, phi_deg):
    beta = np.radians(beta_deg)
    phi  = np.radians(phi_deg)
    numerator   = c_prime + (gamma * z * np.cos(beta)**2 - u) * np.tan(phi)
    denominator = gamma * z * np.sin(beta) * np.cos(beta)
    return numerator / denominator

# Label rules
# FS < 1.0   → RED
# FS < 1.3   → YELLOW
# FS ≥ 1.3   → GREEN

# Default soil parameters for NE India (update per site survey)
SOIL_PARAMS = {
    'c_prime': 8.0,   # kPa
    'gamma':   19.0,  # kN/m³
    'z':       1.5,   # m
    'phi_deg': 28.0   # degrees
}
```

### Strategy 2 — Historical Event Mining (Week 1–4)

Cross-reference with NDMA Landslide Atlas (80,000+ events, 1998–2022), BRO highway closure logs, and NHIDCL incident records. Label retrospectively:

```python
# For each historical event within 5 km of a sensor node:
# 0–60 min before event timestamp  → RED
# 60–180 min before event timestamp → YELLOW
# All other periods                 → GREEN
```

### Strategy 3 — Anomaly Detection Bootstrap (Week 1)

Train an autoencoder on the first dry week (normal conditions only). Use reconstruction error as anomaly score → pseudo-labels.

```python
# Anomaly thresholds (calibrate per site)
# score > 97th percentile → pseudo-RED
# score > 90th percentile → pseudo-YELLOW
```

### Strategy 4 — Active Learning (Monthly)

Identify the 200 samples the model is most uncertain about (highest prediction entropy) and send to a geotechnical expert for manual labeling. Each round of 200 labels improves RED recall by approximately 0.04–0.06.

```python
# Uncertainty = entropy of predicted distribution
H(x) = -Σ p̂_k · log(p̂_k)
# Select top-200 highest-entropy samples → manual label → retrain
```

### Cold-Start Timeline

| Month | Strategy active | Expected RED recall |
|---|---|---|
| 0 | Physics labels (FS) | ~0.72 |
| 1 | + Historical mining + AE bootstrap | ~0.80 |
| 2 | + First real events (gold labels) | ~0.85 |
| 3 | + Active learning round 3 | ~0.89 |
| 5 | Full monsoon season data | ≥ 0.91 |

---

## 12. Transfer Learning & Generalisation

### Architecture: Shared Trunk + Regional Heads

```python
class GeneralisableNavigator(nn.Module):
    def __init__(self, n_regions=6):
        super().__init__()

        # SHARED TRUNK — universal landslide physics (frozen during fine-tune)
        # Learns: spike detection, trend detection, acceleration signatures
        # These transfer because slope mechanics are the same everywhere
        self.shared_trunk = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten()  # → 512-dim vector
        )

        # REGION HEADS — one per corridor (fine-tuned with local data)
        # Learns: local thresholds specific to regional geology
        self.region_heads = nn.ModuleDict({
            'mizoram':   self._head(),
            'manipur':   self._head(),
            'sikkim':    self._head(),
            'arunachal': self._head(),
            'nagaland':  self._head(),
            'meghalaya': self._head(),
        })

    def _head(self):
        return nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3)
        )
```

### Fine-Tuning a New Region

```bash
# Only 2 weeks of local data needed
python ml/fine_tune.py \
    --base-model models/navigator_base.pt \
    --region manipur \
    --local-data data/manipur/two_weeks.parquet \
    --epochs 30 \
    --freeze-trunk True \
    --output models/navigator_manipur_v1.pt
```

### Expected Performance by Data Available

| Local data | Strategy | RED recall |
|---|---|---|
| 0 days | Source model as-is | ~0.63 |
| 3 days | Head fine-tune | ~0.73 |
| 14 days | Head fine-tune | ~0.83 |
| 1 season | Full fine-tune | ~0.91 |

---

## 13. Deployment Guide

### Infrastructure Requirements

**Gateway (per 20 nodes):**
- Raspberry Pi 4 (4 GB RAM) or equivalent
- LoRa gateway module (RAK2245 or similar)
- GSM modem (Airtel/Jio M2M SIM)
- Iridium 9602 satellite modem (backup uplink)
- 20 W solar + 50 Ah battery (72+ hr autonomy)
- IP67 weatherproof enclosure

**Cloud:**
- 2× EC2 t3.medium (API + inference)
- TimescaleDB on RDS (db.t3.medium, 100 GB SSD)
- S3 bucket for model artifacts and raw data archive
- CloudWatch for gateway health monitoring

### Environment Variables

```bash
# .env — do not commit to version control
DATABASE_URL=postgresql://user:pass@localhost:5432/resilio
CLOUD_API_KEY=rr_live_xxxxxxxxxxxx
NHIDCL_WEBHOOK_SECRET=whsec_xxxxxxxxxxxx
IRIDIUM_SBD_ENDPOINT=https://directip.iridium.com/
AWS_REGION=ap-south-1
MODEL_VERSION=1.0.0
ALERT_LEAD_TIME_MIN=30
RED_THRESHOLD=0.70
YELLOW_THRESHOLD=0.40
```

### Monitoring

```bash
# Gateway health dashboard
python tools/monitor.py --dashboard

# Key metrics to watch:
# - nodes_online / nodes_total  (target: > 90%)
# - data_gap_max_min            (alert if > 30 min)
# - inference_latency_p99       (alert if > 10 sec)
# - red_recall_7d               (alert if < 0.85)
# - false_alarm_rate_7d         (alert if > 20%)
# - battery_pct_min             (alert if any node < 15%)
```

---

## 14. Business Model

### Revenue Stream A — Hardware as a Service (HaaS)

Resilio-Route owns, deploys, and maintains the entire sensor + LoRa + gateway stack. Highway authorities pay a monthly subscription.

| Tier | Coverage | Monthly fee |
|---|---|---|
| Pilot | 2–3 highway sections, 20 nodes | ₹1.5L/month |
| Corridor | 1 full NH corridor, 60 nodes | ₹4L/month |
| State | All NH corridors in 1 state | ₹12L/month |

### Revenue Stream B — Risk Intelligence API

Real-time landslide risk scores by highway segment. Tiered access for logistics companies.

| Tier | Limit | Monthly fee |
|---|---|---|
| Starter | 1,000 API calls/day, 2 highways | ₹25,000/month |
| Fleet | Unlimited calls, all NE highways | ₹80,000/month |
| Enterprise | White-label + webhook + SLA | Custom |

### Cost Advantage

Dense sensor coverage at **less than 10%** of traditional geotechnical monitoring cost. Low-power LoRa means minimal operational expense between annual service visits.

---

## 15. References

1. Pennington, C. et al. (2021). SitkaNet: Low-cost landslide monitoring in Alaska. *Landslides*, 18(4).
2. Gonzalez-Olalla, J. et al. (2021). Inform@Risk: Community landslide early warning in Colombia. *Natural Hazards and Earth System Sciences*, 21.
3. NDMA (2023). Landslide Atlas of India. National Disaster Management Authority.
4. Guzzetti, F. et al. (2020). Geographical landslide early warning systems. *Earth-Science Reviews*, 200.
5. Intrieri, E. et al. (2013). Criteria for the design of landslide warning systems. *Natural Hazards and Earth System Sciences*, 13.
6. NHIDCL Annual Report 2022–23. National Highways and Infrastructure Development Corporation.

---

## License

Proprietary. All rights reserved. Resilio-Route, 2024.

## Contact

**Team Resilio-Route**
Northeast India Landslide Early Warning Initiative
Email: hello@resilio-route.in

---

*"The aim is not just to predict the slide — but to keep the Northeast moving."*
