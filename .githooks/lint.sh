#!/usr/bin/env bash
set -euo pipefail

PKG=${PKG:-stlbench}
TESTS=${TESTS:-tests}

echo "==> ruff check"
poetry run ruff check "$PKG" "$TESTS"

echo "==> ruff format (check)"
poetry run ruff format --check "$PKG" "$TESTS"

echo "==> isort (check)"
poetry run isort --check-only --diff "$PKG" "$TESTS"

echo "==> black (check)"
poetry run black --check --diff "$PKG" "$TESTS"

echo "==> mypy"
poetry run mypy "$PKG" "$TESTS"
