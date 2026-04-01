.PHONY: lint format test setup-env setup_env debug build clear publish

SHELL := /bin/bash
PKG := stlbench
TESTS := tests
PY ?= python3

lint:
	PKG=$(PKG) TESTS=$(TESTS) bash .githooks/lint.sh

format:
	poetry run isort $(PKG) $(TESTS)
	poetry run black $(PKG) $(TESTS)
	poetry run ruff check --fix $(PKG) $(TESTS)
	poetry run ruff format $(PKG) $(TESTS)

test:
	poetry run pytest

setup-env setup_env:
	poetry env use "$(PY)"
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry install --with dev

debug:
	PYTHONDONTWRITEBYTECODE=1 poetry run stlbench --help

build:
	poetry build

publish: build
	poetry run twine upload dist/*

clear:
	find $(PKG) $(TESTS) -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; true
