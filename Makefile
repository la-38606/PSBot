.PHONY: sync check test doctor showdown-up showdown-down showdown-logs

sync:
	uv sync --extra ml

check:
	uv run ruff check .
	uv run mypy src
	uv run pytest

test:
	uv run pytest

doctor:
	uv run psbot doctor

showdown-up:
	docker compose up --build -d showdown

showdown-down:
	docker compose down

showdown-logs:
	docker compose logs -f showdown
