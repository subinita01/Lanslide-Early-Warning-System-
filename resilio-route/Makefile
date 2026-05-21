# ═══════════════════════════════════════════════════════════════════════════
#  Resilio-Route — Makefile
#  Single interface for every common task.
#
#  Usage:
#    make setup          — install dependencies + copy .env
#    make pipeline       — run full ML pipeline (simulate → train → evaluate)
#    make api            — start the Risk Intelligence API (port 8000)
#    make ingest         — start the Ingest API (port 8001)
#    make test           — run all smoke tests
#    make monitor        — live terminal dashboard
#    make active         — run one active learning round
#    make evaluate       — evaluate trained models
#    make clean          — remove generated data and models
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: all setup pipeline api ingest migrate test test-integration monitor active evaluate \
        simulate label features train-rf train-cnn clean help

PYTHON    = python
UVICORN   = uvicorn
RF_MODEL  = ml/models/rf_v1.pkl
CNN_MODEL = ml/models/cnn_v1.pt
FEATURES  = ml/data/features.parquet
RAW       = ml/data/raw.parquet
LABELED   = ml/data/labeled.parquet


# ── Default ───────────────────────────────────────────────────────────────
all: help

help:
	@echo ""
	@echo "  Resilio-Route — available make targets"
	@echo "  ──────────────────────────────────────"
	@echo "  make setup       Install deps + copy .env template"
	@echo "  make pipeline    Full ML pipeline end-to-end (quick start)"
	@echo "  make api         Start Risk Intelligence API on :8000"
	@echo "  make ingest      Start Ingest API on :8001"
	@echo "  make migrate     Apply database migrations"
	@echo "  make test        Run smoke tests"
	@echo "  make test-integration Run DB-backed API integration tests"
	@echo "  make monitor     Live terminal dashboard (needs API running)"
	@echo "  make simulate    Generate synthetic sensor data only"
	@echo "  make label       Run physics labeling (Factor of Safety)"
	@echo "  make features    Run feature engineering"
	@echo "  make train-rf    Train Random Forest only"
	@echo "  make train-cnn   Train CNN only"
	@echo "  make evaluate    Full evaluation with safety metrics"
	@echo "  make active      One active learning round (200 samples)
	@echo "  make deploy-check Run pre-deployment checklist"
	@echo "  make export      Export DB to parquet (DAYS=30)"
	@echo "  make lora-test   Test LoRa packet parser"
	@echo "  make docker-up   Start full stack via Docker Compose"
	@echo "  make clean       Remove generated data and model files"
	@echo ""


# ── Setup ─────────────────────────────────────────────────────────────────
setup:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Copying .env template..."
	@[ -f .env ] || cp .env.template .env
	@echo ""
	@echo "  ✅ Setup complete."
	@echo "  Edit .env if needed, then run: make pipeline"
	@echo ""

# ── Full pipeline ──────────────────────────────────────────────────────────
pipeline:
	$(PYTHON) run_pipeline.py

pipeline-real:
	$(PYTHON) run_pipeline.py --real


# ── Individual steps ──────────────────────────────────────────────────────
simulate:
	$(PYTHON) ml/simulate_data.py \
	    --nodes 10 --days 60 \
	    --output $(RAW)

label: $(RAW)
	$(PYTHON) ml/physics_labels.py \
	    --input  $(RAW) \
	    --output $(LABELED)

features: $(LABELED)
	$(PYTHON) -c "\
	import pandas as pd; \
	from ml.feature_engineering import engineer_features; \
	df = pd.read_parquet('$(LABELED)'); \
	featured = engineer_features(df); \
	featured.to_parquet('$(FEATURES)', index=False); \
	print(f'Features saved: {len(featured):,} rows')"

train-rf: $(FEATURES)
	$(PYTHON) ml/train_rf.py \
	    --features $(FEATURES) \
	    --output   $(RF_MODEL)

train-cnn: $(FEATURES)
	$(PYTHON) ml/train_cnn.py \
	    --features $(FEATURES) \
	    --output   $(CNN_MODEL) \
	    --epochs   30

evaluate: $(RF_MODEL) $(CNN_MODEL) $(FEATURES)
	$(PYTHON) ml/evaluate.py \
	    --features  $(FEATURES) \
	    --rf-model  $(RF_MODEL) \
	    --cnn-model $(CNN_MODEL)


# ── Active learning ────────────────────────────────────────────────────────
active: $(RF_MODEL) $(FEATURES)
	@echo "Selecting 200 uncertain samples for expert labeling..."
	$(PYTHON) ml/active_learning.py select \
	    --features  $(FEATURES) \
	    --rf-model  $(RF_MODEL) \
	    --output    ml/data/uncertain_batch.parquet \
	    --n-samples 200
	@echo ""
	@echo "  CSV exported to: ml/data/uncertain_batch.csv"
	@echo "  → Send to geotechnical expert to fill 'expert_label' column"
	@echo "  → Then run: make import-labels"
	@echo ""

import-labels:
	$(PYTHON) ml/active_learning.py import \
	    --features    $(FEATURES) \
	    --labeled-csv ml/data/uncertain_batch.csv \
	    --output      ml/data/features_expert.parquet
	@echo "  → Retrain with: make train-rf FEATURES=ml/data/features_expert.parquet"


# ── Services ──────────────────────────────────────────────────────────────
api:
	$(UVICORN) backend.api.main:app \
	    --host 0.0.0.0 --port 8000 --reload

ingest:
	$(UVICORN) backend.ingest.main:app \
	    --host 0.0.0.0 --port 8001 --reload

db-setup:
	$(PYTHON) backend/db.py

migrate:
	alembic upgrade head


# ── Tools ─────────────────────────────────────────────────────────────────
monitor:
	$(PYTHON) tools/monitor.py --interval 30

monitor-once:
	$(PYTHON) tools/monitor.py --once


# ── Tests ─────────────────────────────────────────────────────────────────
test:
	$(PYTHON) tests/test_pipeline.py

test-verbose:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-integration:
	$(PYTHON) -m pytest tests/test_api_integration.py -v --tb=short


# ── Transfer learning ──────────────────────────────────────────────────────
# Example: make transfer REGION=manipur LOCAL_DATA=ml/data/manipur.parquet
transfer:
	$(PYTHON) ml/transfer_learning.py \
	    --base-model $(CNN_MODEL) \
	    --local-data $(LOCAL_DATA) \
	    --region     $(REGION) \
	    --epochs     30


# ── Gateway ───────────────────────────────────────────────────────────────
gateway:
	$(PYTHON) gateway/run_navigator.py \
	    --rf-model  $(RF_MODEL) \
	    --cnn-model $(CNN_MODEL) \
	    --nodes     NM-01,NM-02,NM-03 \
	    --interval  300


# ── Pre-deployment checklist ──────────────────────────────────────────────
deploy-check: $(RF_MODEL) $(CNN_MODEL)
	$(PYTHON) tools/deploy_check.py \
	    --rf-model  $(RF_MODEL) \
	    --cnn-model $(CNN_MODEL) \
	    --api-url   http://localhost:8000 \
	    --api-key   rr_dev_key


# ── Export DB for retraining ──────────────────────────────────────────────
# Example: make export DAYS=90
export:
	$(PYTHON) tools/export_db.py \
	    --days   $(or $(DAYS),30) \
	    --output ml/data/raw_export.parquet


# ── Test LoRa parser ──────────────────────────────────────────────────────
lora-test:
	$(PYTHON) gateway/lora_parser.py


# ── Docker ────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api


# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	@echo "Removing generated files..."
	rm -f ml/data/*.parquet ml/data/*.csv
	rm -f ml/models/*.pkl ml/models/*.pt
	rm -f ml/reports/*.json
	@echo "Clean complete."

clean-all: clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
