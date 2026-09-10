#!/usr/bin/env bash
# Remove build artifacts. Does NOT touch .venv or Python source files.

find . -path './.venv' -prune -o -name "*.so" -exec rm -v {} \;

# setuptools / pip build artifacts
rm -rvf build dist uvex_transients.egg-info
