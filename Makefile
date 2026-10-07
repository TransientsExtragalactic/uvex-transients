.DEFAULT_GOAL := help
.PHONY: help install lint format test coverage docs docs-clean clean run sync-scheduler release

RUN_CONFIG ?= configs/full_run.yaml
RUN_OUT_DIR ?= results/dev/

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package editable, with dev + test extras.
	pip install -e ".[dev,test]"

lint: ## Check code style with ruff (no fixes applied).
	ruff check .

format: ## Auto-format the codebase with ruff.
	ruff format .

test: ## Run the test suite.
	pytest

coverage: ## Run the test suite with branch coverage.
	pytest --cov

docs: ## Build the Sphinx documentation (HTML).
	$(MAKE) -C docs html

docs-clean: ## Remove built documentation output.
	$(MAKE) -C docs clean

clean: ## Remove build artifacts (see clean.sh).
	./clean.sh

run: ## Run the full-population pipeline (RUN_CONFIG -> RUN_OUT_DIR; both overridable).
	uvex-transients run $(RUN_CONFIG) --out-dir $(RUN_OUT_DIR) --overwrite

sync-scheduler: ## Pin the repo to the latest uvex-scheduler release (see scripts/sync_scheduler.py).
	python scripts/sync_scheduler.py

release: ## Tag and push a release (VERSION=v0.2.1alpha); the tag triggers the build_and_release workflow.
	@test -n "$(VERSION)" || { echo "usage: make release VERSION=vX.Y.Z[alpha]"; exit 1; }
	@test "$$(git branch --show-current)" = main || { echo "release from main (currently on $$(git branch --show-current))"; exit 1; }
	@git diff --quiet HEAD || { echo "working tree is not clean"; exit 1; }
	git pull --ff-only
	git tag -a $(VERSION) -m "$(VERSION)"
	git push origin $(VERSION)
