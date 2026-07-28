# SHOWDOWN_COMMIT must match infra/showdown/Dockerfile (compose.yaml) and psbot.constants.
SHOWDOWN_COMMIT := f2fe71c8754edf140b2ca95fb3f41222404f2936
SHOWDOWN_DIR ?= $(HOME)/pokemon-showdown
NODE_BIN ?= /opt/homebrew/opt/node@22/bin

.PHONY: sync check test doctor smoke showdown-up showdown-down showdown-logs showdown-native-setup showdown-native

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

# iCloud-synced folders (Desktop/Documents) mark .venv contents hidden, and
# Python >= 3.11.9 silently skips hidden .pth files, breaking `import psbot`.
# Run this whenever the venv mysteriously stops finding the package.
repair-venv:
	chflags -R nohidden .venv

smoke:
	uv run psbot smoke

showdown-up:
	docker compose up --build -d showdown

showdown-down:
	docker compose down

showdown-logs:
	docker compose logs -f showdown

# No-Docker alternative: run the simulator as a native Node checkout pinned to
# the same commit as the Docker image. One-time setup, then showdown-native.
showdown-native-setup:
	test -d $(SHOWDOWN_DIR) || git clone https://github.com/smogon/pokemon-showdown.git $(SHOWDOWN_DIR)
	cd $(SHOWDOWN_DIR) && git fetch --quiet origin && git checkout --quiet $(SHOWDOWN_COMMIT)
	cd $(SHOWDOWN_DIR) && PATH="$(NODE_BIN):$$PATH" npm ci
	test -f $(SHOWDOWN_DIR)/config/config.js || cp $(SHOWDOWN_DIR)/config/config-example.js $(SHOWDOWN_DIR)/config/config.js

showdown-native:
	cd $(SHOWDOWN_DIR) && PATH="$(NODE_BIN):$$PATH" node pokemon-showdown start --no-security
