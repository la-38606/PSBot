# PSBot

PSBot is a month-long learning project to build and evaluate a machine-learning agent for Pokémon Showdown Gen 9 Random Battles.

The project progresses from heuristic and classical supervised-learning baselines to behavioral cloning and PPO self-play, with uncertainty-aware features, matched-team evaluation, and a target peak ladder rating of 1500.

See the [month-one project plan](PROJECT_PLAN.md) for the architecture, technology stack, milestones, evaluation gates, and scaling strategy.

## Day-one quickstart

```bash
uv sync --extra ml
docker compose up --build -d showdown
uv run psbot doctor
uv run psbot smoke   # one random-vs-random battle against the local server
uv run pytest
```

The simulator is pinned in Docker, generated data and models remain outside Git, and all public commands are available through `uv run psbot --help`.

Persist smoke and tournament summaries as JSON when needed:

```bash
uv run psbot smoke --output logs/smoke/latest.json
uv run psbot evaluate tournament --a max-base-power --b random --n 100 \
  --output logs/tournaments/max-vs-random.json
```

### Without Docker

On machines without Docker, run the simulator as a native Node checkout pinned to the same commit as the Docker image (requires `brew install node@22`):

```bash
make showdown-native-setup   # one-time: clone, pin, npm ci (~/pokemon-showdown)
make showdown-native         # start the server on port 8000 (foreground)
```

`psbot doctor` and `psbot smoke` work identically against either server.
