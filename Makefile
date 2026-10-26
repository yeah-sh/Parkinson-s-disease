# Targets mirror the `pdeeg` commands (src/pdeeg/cli.py); every step is skipped while current.
#
#   make all                  the whole pipeline
#   make stats FORCE=1        run a step again even if it is current
#   make features N_JOBS=8    with this many processes
#   make train-final          the final models (models/), after classify and cascade
#   make predict FILE=x.bdf [PAIRED=y.bdf]   a research demo, not a diagnostic tool
#   make test | lint | format | docker-build | docker-run | app
#
# Plain `make` prints this list rather than starting a pipeline that can take hours.

PYTHON ?= python
PDEEG ?= pdeeg
CONFIG_DIR ?=
FORCE ?=
N_JOBS ?=
IMAGE ?= pdeeg
FILE ?=
PAIRED ?=

GLOBAL := $(if $(CONFIG_DIR),--config-dir "$(CONFIG_DIR)")
FORCE_FLAG := $(if $(FORCE),--force)
JOBS_FLAG := $(if $(N_JOBS),--n-jobs $(N_JOBS))
PREDICT_ARGS = "$(FILE)" $(if $(PAIRED),--paired "$(PAIRED)") $(JOBS_FLAG)

PARALLEL_STEPS := preprocess features stats classify cascade psd-features mfdfa-features \
	mfdfa-curves

.DEFAULT_GOAL := help
.PHONY: help install install-app all download recordings $(PARALLEL_STEPS) report \
	mfdfa-scaling config train-final predict check-file test lint format docker-build \
	docker-run docker-train-final docker-predict app

help:
	@echo "Pipeline:  all download recordings preprocess features stats classify cascade report"
	@echo "Parts:     psd-features mfdfa-features mfdfa-curves mfdfa-scaling config"
	@echo "Models:    train-final, predict FILE=recording.bdf [PAIRED=other.bdf]"
	@echo "Options:   FORCE=1 (run even if current), N_JOBS=n, CONFIG_DIR=path"
	@echo "Project:   install install-app test lint format docker-build docker-run app"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-app:
	$(PYTHON) -m pip install -e ".[dev,app]"

all:
	$(PDEEG) $(GLOBAL) all $(FORCE_FLAG)

download recordings:
	$(PDEEG) $(GLOBAL) $@ $(FORCE_FLAG)

$(PARALLEL_STEPS):
	$(PDEEG) $(GLOBAL) $@ $(FORCE_FLAG) $(JOBS_FLAG)

mfdfa-scaling:
	$(PDEEG) $(GLOBAL) $@ $(JOBS_FLAG)

report config train-final:
	$(PDEEG) $(GLOBAL) $@

check-file:
	@test -n "$(FILE)" || { echo "usage: make $(MAKECMDGOALS) FILE=recording.bdf [PAIRED=other.bdf]"; exit 2; }

predict: check-file
	$(PDEEG) $(GLOBAL) predict $(PREDICT_ARGS)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -v "$(CURDIR)/data:/app/data" $(IMAGE) all

docker-train-final:
	docker run --rm -v "$(CURDIR)/data:/app/data" $(IMAGE) train-final

# FILE (and PAIRED) are paths inside the container: put the recording under data/ and pass
# FILE=data/recording.bdf. Models and predictions go to data/models and data/predictions.
docker-predict: check-file
	docker run --rm -v "$(CURDIR)/data:/app/data" $(IMAGE) predict $(PREDICT_ARGS)

app:  # uploads stay on this machine: the page is served on localhost only
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.address localhost
