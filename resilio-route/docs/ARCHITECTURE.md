# System Architecture

## Overview

Resilio-Route is a four-layer edge-to-cloud system. Data originates at soil
level, travels through a LoRa mesh to a highway depot gateway, moves to the
cloud for ML inference, and terminates as a physical LED alert and a routing
API call — all within 30 seconds of the sensor reading.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RESILIO-ROUTE SYSTEM                                │
│                                                                         │
│  LAYER 1: PERCEPTION  (field, every 5 minutes)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Ground Node │  │  Ground Node │  │  Ground Node │  ...20 nodes     │
│  │  ESP32+MEMS  │  │  ESP32+MEMS  │  │  ESP32+MEMS  │                  │
│  │  - Tilt      │  │  - Tilt      │  │  - Tilt      │                  │
│  │  - Moisture  │  │  - Moisture  │  │  - Moisture  │                  │
│  │  - Pore P.   │  │  - Pore P.   │  │  - Pore P.   │                  │
│  │  - Rainfall  │  │  - Rainfall  │  │  - Rainfall  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│  LAYER 2: TRANSMISSION  (LoRa mesh, 15 km range)                        │
│         │                 │                 │                           │
│         └────────┬────────┘                 │                           │
│                  └──────────────────────────┘                           │
│                             │  52-byte LoRa packets                     │
│                             ▼                                           │
│  LAYER 3: AGGREGATION  (depot gateway, Raspberry Pi 4)                  │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  gateway/run_navigator.py                                │           │
│  │  ┌────────────────┐    ┌──────────────────────────────┐  │           │
│  │  │ LoRa Receiver  │───▶│ Edge Inference (Navigator)   │  │           │
│  │  │ gateway/       │    │ ml/navigator.py              │  │           │
│  │  │ lora_parser.py │    │ RF pkl + CNN pt              │  │           │
│  │  └────────────────┘    └──────────────┬───────────────┘  │           │
│  │                                       │                  │           │
│  │  ┌──────────────────────────────────  │ ─────────────┐   │           │
│  │  │  LOCAL EEPROM BUFFER             │ │              │   │           │
│  │  │  (stores 24 hrs if uplink drops) │ │              │   │           │
│  │  └──────────────────────────────────┘ │              │   │           │
│  └────────────────────────────────────── │ ─────────────┘   │           │
│                           GSM │ Satellite▼                              │
│                               │  HTTPS POST                             │
│  LAYER 4: CLOUD  (AWS / Azure)                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  backend/ingest/main.py  (port 8001)                     │           │
│  │         │                                                │           │
│  │         ▼                                                │           │
│  │  TimescaleDB  ──▶  Feature engineering  ──▶  Navigator   │           │
│  │                    ml/feature_            ml/navigator.py│           │
│  │                    engineering.py                        │           │
│  │         │                                                │           │
│  │         ▼                                                │           │
│  │  backend/api/main.py  (port 8000)                        │           │
│  └──────────────────────────────────────────────────────────┘           │
│                               │                                         │
│  LAYER 5: ACTION                                                         │
│  ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │  LED Board       │   │  Fleet API      │   │  NHIDCL Control Room │  │
│  │  km-marker signs │   │  Webhook alerts │   │  Dashboard + SOPs    │  │
│  │  🟢/🟡/🔴        │   │  JSON push      │   │                      │  │
│  └──────────────────┘   └─────────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Step by Step

### Every 5 minutes (one inference cycle)

```
1. ESP32 node wakes from deep sleep
2. Reads: tilt (MEMS), moisture ×6 (ADC), pore pressure (ADC), rainfall (ISR counter)
3. Packs 52-byte LoRa packet, computes local FS estimate
4. Transmits on 865 MHz LoRa, enters deep sleep

5. Gateway LoRa radio receives packet
6. gateway/lora_parser.py decodes binary → Python dict
7. gateway/run_navigator.py:
   a. Pulls last WINDOW_STEPS (36) readings from local SQLite buffer
   b. Calls ml/navigator.py Navigator.predict()
   c. Navigator runs RF (tabular snapshot) + CNN (3-hr window)
   d. Weighted ensemble → P(GREEN), P(YELLOW), P(RED)
   e. Applies safety-first thresholds → classification

8. Gateway writes LED command to /dev/ttyUSB0 (RS-485 serial)
   LED board at km-marker updates immediately

9. Gateway POSTs to backend/ingest/main.py:
   - Raw reading → TimescaleDB sensor_readings
   - Prediction  → TimescaleDB predictions

10. backend/api/main.py serves:
    - GET /v1/risk/segment → logistics partners
    - POST (webhook) → fleet management systems
    - GET /v1/nodes/status → NHIDCL dashboard
```

---

## Navigator ML Pipeline

```
Raw sensors (5-min reading)
        │
        ▼ ml/feature_engineering.py
18 engineered features
  [tilt, tilt_rate, tilt_rolling_mean_3hr, tilt_variance_1hr,
   saturation_ratio, moisture_gradient, moisture_d1..d6,
   pore_pressure, pore_spike,
   rainfall_1hr, rainfall_6hr, rainfall_24hr,
   rain_tilt_product]
        │
        ├──▶  Random Forest (300 trees)
        │     Input: 18-feature tabular snapshot
        │     Output: P_RF(GREEN, YELLOW, RED)
        │     Handles: missing data, noisy sensors, class imbalance
        │
        └──▶  CNN (3-layer temporal)
              Input: 6 channels × 36 timesteps (3-hr window)
              Layer 1 (k=3): spike detector     → 15-min patterns
              Layer 2 (k=7): trend detector     → 35-min patterns
              Layer 3 (k=9): acceleration ramp  → 45-min patterns
              Output: P_CNN(GREEN, YELLOW, RED)
                    │
                    ▼
        Weighted ensemble: 0.6×RF + 0.4×CNN
                    │
                    ▼
        Safety-first thresholds:
          P(RED) ≥ 0.70  →  🔴 RED    (stop, reroute)
          P(RED) ≥ 0.40  →  🟡 YELLOW (caution)
          else           →  🟢 GREEN  (safe)
                    │
                    ▼
        Routing formula: Cost = Distance × (1 + Risk_Local)
```

---

## Generalisation Architecture

```
Base model (trained on Mizoram data)
  Shared trunk:    [Conv1 → Conv2 → Conv3 → Pool → Flatten]
                   Learns universal slope physics
                   FROZEN during regional fine-tuning

  Regional heads:  mizoram  | manipur | sikkim | arunachal | ...
                   Small MLP per region (512 → 64 → 3)
                   Re-trained with 14 days of local data

Fine-tune workflow:
  14 days local data
        │
        ▼ ml/transfer_learning.py
  Freeze trunk, train head only
        │
        ▼
  RED recall ≈ 0.83 on new region
  (vs 0.63 without any fine-tuning)
```

---

## Cold-Start Strategy

When deployed to a new highway (day 0, no labeled slides):

| Month | Strategy | Labels available | RED recall |
|-------|----------|-----------------|-----------|
| 0 | Physics (FS model) | 85,000+ | ~0.72 |
| 1 | + NDMA historical mining | +12,000 | ~0.80 |
| 1 | + Anomaly detection (AE) | +50,000 (weak) | ~0.82 |
| 1 | + Active learning round 1 | +200 (expert) | ~0.84 |
| 2 | + First real monsoon events | +gold labels | ~0.87 |
| 3 | + Active learning rounds 2-3 | +400 more | ~0.90 |
| 5 | Full monsoon season | Complete | ≥ 0.91 |

---

## Deployment Topology

```
Highway NH-306 (example)

KM 100  NM-01 ──┐
KM 108  NM-02 ──┤
KM 115  NM-03 ──┤
KM 122  NM-04 ──┤── LoRa mesh ──▶ Gateway @ KM-130 depot
KM 128  NM-05 ──┤                 (Raspberry Pi 4)
KM 132  NM-06 ──┤                 GSM uplink primary
KM 140  NM-07 ──┤                 Iridium SBD backup
KM 147  NM-08 ──┘

             ┌── LED board NH-06 KM-129 (before danger zone)
             ├── LED board NH-06 KM-140
             └── LED board NH-06 KM-150
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Sensor MCU | ESP32 / SAMD21 | Low power, deep sleep, ADC |
| Radio | LoRa SX1278 | 15 km range, no cell tower needed |
| Gateway compute | Raspberry Pi 4 | Edge inference, 4 GB RAM |
| Primary uplink | GSM (Airtel/Jio M2M) | Widely available NE India |
| Backup uplink | Iridium SBD | Works anywhere, no cell coverage |
| Time-series DB | TimescaleDB (PostgreSQL) | Hypertable partitioning, SQL |
| ML — tabular | scikit-learn RandomForest | Fast, interpretable, handles NaN |
| ML — temporal | PyTorch CNN | Learns pre-failure patterns |
| API framework | FastAPI | Async, auto-docs, Pydantic |
| Edge serialisation | TorchScript | No Python needed on Raspberry Pi |
| Config | pydantic-settings | Type-safe .env loading |

---

## API Surface

```
Public (logistics partners, NHIDCL):
  GET  /v1/health
  GET  /v1/risk/segment    ?highway= &km_from= &km_to=
  GET  /v1/risk/route      ?origin= &destination= &vehicle_type=
  GET  /v1/nodes/status    ?highway=
  POST /v1/webhooks/register

Internal (gateway → cloud):
  POST /v1/ingest           (single reading)
  POST /v1/ingest/batch     (buffered batch after uplink recovery)
```
