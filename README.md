# PSBot

PSBot is a month-long learning project to build and evaluate a machine-learning agent for Pokémon Showdown Gen 9 Random Battles.

The project progresses from heuristic and classical supervised-learning baselines to behavioral cloning and PPO self-play, with uncertainty-aware features, matched-team evaluation, and a target peak ladder rating of 1500.

See the [month-one project plan](PROJECT_PLAN.md) for the architecture, technology stack, milestones, evaluation gates, and scaling strategy.

## Day-one quickstart

```bash
uv sync --extra ml
docker compose up --build -d showdown
uv run psbot doctor
uv run pytest
```

The simulator is pinned in Docker, generated data and models remain outside Git, and all public commands are available through `uv run psbot --help`.
