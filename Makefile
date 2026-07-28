# Augury — master polyglot build workflow.
#
# Recipes are POSIX sh, so on Windows run this from Git Bash (or any shell
# where `sh` resolves to a POSIX shell), not cmd.exe or PowerShell.

SHELL := /bin/sh

ROOT        := $(CURDIR)
# --env-file is explicit: with `-f docker/...`, Compose would otherwise look for
# an env file next to the compose file rather than at the repo root.
COMPOSE     := docker compose --env-file $(CURDIR)/.env -f docker/docker-compose.yml
PY          ?= python
VENV        := apps/augury-signal/.venv
VENV_PY     := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
VENV_PY     := $(VENV)/Scripts/python
endif

# Toolchains installed per-user are not on PATH in a fresh shell. Override any
# of these on the command line, e.g. `make test-java MVN=/usr/bin/mvn`.
# See docs/BUILD.md for where each one lives on this machine.
MVN         ?= mvn
RSCRIPT     ?= Rscript
CARGO       ?= cargo

# Loaded by recipes that need DB credentials; see .env.example.
ENV_FILE    := $(ROOT)/.env

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: env
env: ## Create .env from the template if it does not exist
	@test -f $(ENV_FILE) || (cp .env.example $(ENV_FILE) && echo "created .env from .env.example")

# ---------------------------------------------------------------------------
# Infrastructure (Stage 1)
# ---------------------------------------------------------------------------

.PHONY: up
up: ## Start TimescaleDB + Redis
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop containers (volumes preserved)
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop containers AND delete their volumes (destroys all stored data)
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail container logs
	$(COMPOSE) logs -f

.PHONY: db-migrate
db-migrate: ## Apply SQL migrations in docker/migrations, in order
	@sh docker/migrate.sh

.PHONY: psql
psql: ## Open a psql shell against the running database
	$(COMPOSE) exec timescaledb psql -U augury -d augury

# ---------------------------------------------------------------------------
# Python — augury-signal (Stages 2, 4, 7)
# ---------------------------------------------------------------------------

.PHONY: py-setup
py-setup: ## Create the augury-signal virtualenv and install dependencies
	cd apps/augury-signal && uv venv --python 3.12 && uv pip install -e ".[dev]"

.PHONY: signal
signal: ## Run one signal refresh cycle for all configured markets
	cd apps/augury-signal && $(ROOT)/$(VENV_PY) -m augury_signal.cli refresh

.PHONY: slice
slice: ## End-to-end vertical slice for one market: make slice MARKET=<ticker>
	cd apps/augury-signal && $(ROOT)/$(VENV_PY) -m augury_signal.cli slice --market $(MARKET)

.PHONY: dagster
dagster: ## Launch the Dagster UI
	cd apps/augury-signal && $(ROOT)/$(VENV_PY) -m dagster dev -f augury_signal/pipeline/orchestrator.py

.PHONY: watcher
watcher: ## Run the market resolution watcher once
	cd apps/augury-signal && $(ROOT)/$(VENV_PY) -m augury_signal.cli watch-resolutions

# ---------------------------------------------------------------------------
# Rust — augury-ingest (Stage 3)
# ---------------------------------------------------------------------------

.PHONY: ingest
ingest: ## Run the Rust ingestion service (fixture mode unless AUGURY_X_LIVE=1)
	cd apps/augury-ingest && $(CARGO) run --release

.PHONY: ingest-build
ingest-build: ## Build the Rust ingestion service
	cd apps/augury-ingest && $(CARGO) build --release

# ---------------------------------------------------------------------------
# C++ — augury-engine (Stage 5)
# ---------------------------------------------------------------------------

.PHONY: engine-build
engine-build: ## Configure and build the C++ LMSR engine (Release)
	cmake -S apps/augury-engine -B apps/augury-engine/build -DCMAKE_BUILD_TYPE=Release
	cmake --build apps/augury-engine/build --config Release --parallel

.PHONY: engine
engine: engine-build ## Run the LMSR engine daemon
	./apps/augury-engine/build/bin/augury_engine

.PHONY: bench
bench: engine-build ## Run the C++ benchmark suite
	./apps/augury-engine/build/bin/engine_benchmarks

# ---------------------------------------------------------------------------
# Java — augury-api (Stage 6)
# ---------------------------------------------------------------------------

.PHONY: api
api: ## Run the Spring Boot API
	cd apps/augury-api && $(MVN) spring-boot:run

.PHONY: api-build
api-build: ## Package the Spring Boot API
	cd apps/augury-api && $(MVN) -q package

# ---------------------------------------------------------------------------
# R — augury-analytics (Stage 7)
# ---------------------------------------------------------------------------

.PHONY: analytics
analytics: ## Render the lead-lag / Granger causality Quarto report
	cd apps/augury-analytics && $(RSCRIPT) -e "quarto::quarto_render('reports/lead_lag_analysis.qmd')"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test-py
test-py: ## pytest — augury-signal
	cd apps/augury-signal && $(ROOT)/$(VENV_PY) -m pytest -q

.PHONY: test-rust
test-rust: ## cargo test — augury-ingest
	cd apps/augury-ingest && $(CARGO) test

.PHONY: test-cpp
test-cpp: ## ctest (Catch2) — augury-engine
	cmake -S apps/augury-engine -B apps/augury-engine/build -DCMAKE_BUILD_TYPE=Debug
	cmake --build apps/augury-engine/build --parallel
	ctest --test-dir apps/augury-engine/build --output-on-failure

.PHONY: test-java
test-java: ## mvn test — augury-api
	cd apps/augury-api && $(MVN) -B test

.PHONY: test-r
test-r: ## testthat — augury-analytics
	cd apps/augury-analytics && $(RSCRIPT) -e "testthat::test_dir('tests/testthat')"

.PHONY: test-all
test-all: test-py test-rust test-cpp test-java test-r ## Run every language's test suite
