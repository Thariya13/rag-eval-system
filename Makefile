.PHONY: install lint test

install:
	uv sync --extra rerank --extra eval --extra dev

lint:
	uv run ruff check src tests

test:
	uv run pytest -q