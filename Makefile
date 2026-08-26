# Project Janus - one entry point for the whole pipeline.
#
# `make all` takes a clean clone to a working demo. Every step is reproducible from a
# seed and none of it needs an API key, a network connection after the reference-data
# fetch, or a GPU.

UV ?= uv
SEED ?= 42

.PHONY: help all setup data reference train evaluate loao arena arena-fast fidelity check-ui check-clone serve ui test test-all lint clean status results docx

help:
	@echo "Project Janus"
	@echo ""
	@echo "  make all        full pipeline: data -> train -> evaluate -> reports"
	@echo "  make setup      create the venv and install dependencies"
	@echo "  make data       generate the synthetic payment world"
	@echo "  make reference  download public datasets used for fidelity measurement"
	@echo "  make fidelity   measure how closely the synthetic data matches real data"
	@echo "  make train      train and persist the defence"
	@echo "  make evaluate   detection efficacy on a temporal holdout"
	@echo "  make loao       leave-one-attack-out (slow: trains one model per family)"
	@echo "  make arena      run the Red/Blue adversarial loop (~1h, reported numbers)"
	@echo "  make arena-fast same loop at demo scale (~15 min)"
	@echo "  make serve      start the API"
	@echo "  make ui         start the web console (needs make serve in another shell)"
	@echo "  make results    regenerate RESULTS.md from reports/"
	@echo "  make docx       regenerate the challenge walkthrough document"
	@echo "  make test       run the test suite (fast subset)"
	@echo "  make check-ui   smoke-test every console view for render errors"
	@echo "  make check-clone verify a fresh clone builds and runs from scratch"
	@echo "  make status     show what has been built so far"
	@echo ""
	@echo "  Note: 'make loao' is not part of 'make all' - it trains one model per"
	@echo "  attack family and takes ~30 minutes. Run it separately."

setup:
	$(UV) sync --extra dev
	cd web && npm install --no-audit --no-fund

data:
	$(UV) run janus generate run --seed $(SEED)

reference:
	$(UV) run python scripts/fetch_reference_data.py

fidelity: reference
	$(UV) run janus generate fidelity

train: data
	$(UV) run janus defend train

evaluate: data
	$(UV) run janus defend evaluate

loao:
	$(UV) run janus defend loao --customers 6000 --days 45

arena:
	$(UV) run janus arena run --profile full

arena-fast:
	$(UV) run janus arena run --profile fast

results:
	$(UV) run python scripts/build_results.py

docx:
	$(UV) run python scripts/build_walkthrough.py

all: data train evaluate fidelity arena-fast results docx
	@$(UV) run janus status

serve:
	$(UV) run janus serve

ui:
	cd web && npm run dev

test:
	$(UV) run pytest -m "not slow"

check-clone:
	./scripts/check_clean_clone.sh

check-ui:
	@echo "Requires 'make serve' and 'make ui' running in other shells."
	./scripts/check_ui.sh

test-all:
	$(UV) run pytest

lint:
	$(UV) run ruff check janus tests scripts

status:
	@$(UV) run janus status

clean:
	rm -rf data/synthetic models reports/*.json reports/*.csv
