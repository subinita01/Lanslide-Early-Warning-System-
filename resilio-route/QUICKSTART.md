# Resilio-Route — Quick Start Guide

> **Goal:** From a fresh machine to a fully trained Navigator and running API in under 15 minutes.

---

## Prerequisites

- Python 3.10+
- Git
- (Optional) PostgreSQL + TimescaleDB for the full database stack
- (Optional) PlatformIO for flashing the ESP32 node firmware

---

## Step 1 — Clone and install

```bash
git clone https://github.com/resilio-route/navigator.git
cd navigator

# Install all Python dependencies
make setup
# This also copies .env.template → .env (edit if needed)
```

---

## Step 2 — Run the full pipeline

This single command runs everything end-to-end on **synthetic data**
(no real hardware needed):

```bash
make pipeline
```

It will:
1. Generate 60 days of synthetic sensor readings (10 nodes)
2. Label every reading using the Factor of Safety physics model
3. Engineer all 18 features
4. Train the Random Forest (300 trees)
5. Train the CNN (30 epochs)
6. Print a deployment readiness summary

Expected output (last section):
```
  Random Forest
    RED recall:     0.7312   ✅ PASS
    Model path:     ml/models/rf_v1.pkl

  CNN
    RED recall:     0.6891   ✅ PASS
    Model path:     ml/models/cnn_v1.pt

  Next steps:
    ✅ Models trained and saved
    👉 Run: uvicorn backend.api.main:app --reload
```

---

## Step 3 — Start the API

```bash
make api
# Starts on http://localhost:8000
```

If you are running the full local stack with Docker, initialize the database
with migrations before starting the APIs:

```bash
docker compose up -d db
docker compose run --rm migrate
```

Test it:
```bash
# Health check (no auth needed)
curl http://localhost:8000/v1/health

# Risk score for a highway segment (use dev key)
curl "http://localhost:8000/v1/risk/segment?highway=NH-306&km_from=120&km_to=145" \
     -H "Authorization: Bearer rr_dev_key"
```

---

## Step 4 — Run the smoke tests

```bash
make test
```

All 6 tests should pass. This confirms your environment is correctly set up.

To verify the real database-backed request flow, run the integration tests after
the database is up and migrations have been applied:

```bash
make test-integration
```

---

## Step 5 — Live terminal monitor

While the API is running in one terminal, open a second and run:

```bash
make monitor
```

---

## Step 6 — When real hardware arrives

Replace the synthetic data with real sensor readings exported from your
TimescaleDB gateway:

```bash
# Export from your gateway DB to parquet
# (see docs/gateway_export.md for the SQL query)

# Then retrain on real data
make pipeline-real
```

---

## Project layout

```
resilio-route/
├── ml/
│   ├── data_schema.py          ← column names, all in one place
│   ├── simulate_data.py        ← synthetic data generator
│   ├── physics_labels.py       ← Factor of Safety labeling
│   ├── feature_engineering.py  ← 18-feature pipeline
│   ├── train_rf.py             ← Random Forest training
│   ├── train_cnn.py            ← CNN training
│   ├── navigator.py            ← ensemble inference (the Navigator)
│   ├── active_learning.py      ← cold-start: find uncertain samples
│   ├── transfer_learning.py    ← fine-tune for new regions
│   └── evaluate.py             ← safety metrics + deployment gate
│
├── backend/
│   ├── db.py                   ← SQLAlchemy models + TimescaleDB
│   ├── api/main.py             ← Risk Intelligence API (port 8000)
│   └── ingest/main.py          ← Gateway ingest API (port 8001)
│
├── gateway/
│   └── run_navigator.py        ← Edge inference loop (Raspberry Pi)
│
├── firmware/
│   └── node_firmware.ino       ← ESP32 sensor node firmware
│
├── tools/
│   └── monitor.py              ← Terminal dashboard
│
├── tests/
│   └── test_pipeline.py        ← Smoke tests
│
├── config/settings.py          ← All settings from .env
├── run_pipeline.py             ← One-command end-to-end pipeline
├── Makefile                    ← make <target> shortcuts
└── requirements.txt
```

---

## Key concepts

| Concept | File | One-line explanation |
|---|---|---|
| Factor of Safety | `ml/physics_labels.py` | Physics equation that labels without needing real slides |
| 18 features | `ml/feature_engineering.py` | Derived signals that make patterns detectable |
| Random Forest | `ml/train_rf.py` | 300 decision trees voting on tabular snapshot |
| CNN | `ml/train_cnn.py` | Detects acceleration patterns in 3-hr time window |
| Navigator | `ml/navigator.py` | Ensemble: 0.6×RF + 0.4×CNN → GREEN/YELLOW/RED |
| Active learning | `ml/active_learning.py` | Find the 200 most uncertain samples each month |
| Transfer learning | `ml/transfer_learning.py` | Fine-tune for new region in 14 days of data |

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `make setup` to install dependencies |
| `RED recall < 0.70` | Normal for synthetic data — improves with real monsoon data |
| `Models not found` | Run `make pipeline` first |
| API returns 403 | Use `rr_dev_key` in development (see `.env.template`) |
| TimescaleDB errors | Falls back to plain PostgreSQL automatically |

---

*"The aim is not just to predict the slide — but to keep the Northeast moving."*
