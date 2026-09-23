.DEFAULT_GOAL := help
.PHONY: help install lint format test coverage docs docs-clean clean run notebooks

RUN_CONFIG ?= configs/full_run.yaml
RUN_OUT_DIR ?= results/dev/
NOTEBOOKS := $(wildcard notebooks/*.ipynb)

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

notebooks: ## Execute every notebook in notebooks/ in place (requires the dev+docs extras).
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $(NOTEBOOKS)
