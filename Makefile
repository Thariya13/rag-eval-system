.PHONY: install lint test data up down ingest

EU_AI_ACT_URL = https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689

up:
	docker compose up -d qdrant

down:
	docker compose down

ingest:
	uv run ragsys ingest --chunking structure

install:
	uv sync --extra rerank --extra eval --extra dev

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

data:
	mkdir -p data/raw
	curl -fL -A "Mozilla/5.0" "$(EU_AI_ACT_URL)" -o data/raw/eu_ai_act_2024_1689.pdf