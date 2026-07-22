# PSBot Month-One Project Plan

## Summary and goals

This document is the authoritative roadmap for building a Gen 9 Random Battle agent. It is written for two contributors and is intended to be suitable for a public portfolio.

- **Primary goal:** A locked model reaches a peak Pokémon Showdown rating of 1500 within 300 uninterrupted ladder games.
- **Milestones:** Reach 1400, then 1500; report peak and final ratings, W/L record, GXE/Glicko, the full rating curve, and the checkpoint hash.
- **Local quality gate:** Outperform `SimpleHeuristicsPlayer` over 1,000 held-out battles with the lower bound of a 95% win-rate confidence interval above 50%.
- **Learning goal:** Progress through heuristic baselines, classical supervised learning, behavioral cloning, and PPO self-play while documenting experiments and failures.
- **Month-one exclusions:** Other formats, team building, LLM agents, a user interface, exhaustive minimax search, and custom recurrent PPO.

### Strategy decision: ML spine, search-informed teacher

The project keeps behavioral cloning and PPO because learning Python and ML is a primary goal. The reviewed search-and-inference proposal contributes several valuable components without replacing that goal:

- Bayesian random-set inference becomes an observation feature and a source of heuristic confidence, not a separate product architecture.
- A bounded damage/type-aware search policy becomes a strong teacher, fallback, and evaluation baseline.
- Seat-swapped, matched-team evaluation reduces Random Battle matchup variance; sequential tests are used for model ablations, while Wilson intervals remain the fixed-baseline reporting standard.
- Raw protocol logs, immutable experiment manifests, reconnect behavior, timer safety, and pre-committed cut lines become release requirements.
- `foul-play` is a pinned external yardstick and prior-art reference only. Its GPL-3.0 implementation is not copied into PSBot.

The proposal's unsupported novelty claims, full-time-solo schedule, removal of ML, and replacement of the 1400-1500 ladder goal are rejected. Public ladder use is conditional on current server policy and must stop if it degrades human experience or an administrator requests it.

## Technology and repository design

- **Runtime:** Python 3.11 managed by [`uv`](https://docs.astral.sh/uv/), with exact dependencies committed in `uv.lock`.
- **Simulator:** Official [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server in Docker, pinned to one Git commit for reproducibility.
- **Environment:** [`poke-env`](https://github.com/hsahovic/poke-env) `SinglesEnv` configured for `gen9randombattle`. Its official example supports parallel PPO and legal-action masking. See the [`poke-env` RL example](https://poke-env.readthedocs.io/en/latest/examples/reinforcement_learning.html).
- **ML/RL:** PyTorch, Stable-Baselines3 PPO, Gymnasium, scikit-learn, NumPy, Polars, and PyArrow.
- **Visualization:** Matplotlib and Seaborn.
- **Interfaces and configuration:** Typer CLI, YAML configuration, Pydantic settings, HTTPX, and Tenacity.
- **Experiment infrastructure:** Weights & Biases for metrics; private Hugging Face datasets and models during development; selected artifacts may be published after review.
- **Quality:** pytest, pytest-asyncio, Ruff, mypy, pre-commit, and GitHub Actions.
- **Deployment:** Docker Compose locally and the same image on RunPod; target cloud spend is $50.
- **Search reference:** [`foul-play`](https://github.com/pmariglia/foul-play) is used only as a documented comparison point; PSBot remains an original `poke-env` codebase.

### Repository organization

```text
src/psbot/
  agents/        # Random, heuristic, BC/PPO, and hybrid players
  battle/        # poke-env environment and Showdown integration
  features/      # Observation, history, and action encoding
  inference/     # Random-set posterior features and uncertainty tracking
  data/          # Replay acquisition, reconstruction, and datasets
  training/      # Classical baselines, BC, PPO, and checkpoints
  evaluation/    # Tournaments, statistics, and ladder runner
  cli.py
configs/
  data/
  bc/
  ppo/
  evaluation/
infra/showdown/
tests/
docs/
```

Large datasets, checkpoints, `wandb/`, `tmp/`, and the local course PDF remain outside Git. Raw artifacts are cached locally and synchronized through Hugging Face.

### Public CLI contract

- `psbot doctor`: Validate Python, Showdown, dependencies, credentials, and the model device.
- `psbot collect selfplay`: Generate local trajectories.
- `psbot replays fetch`: Download and cache replay JSON and input logs.
- `psbot replays reconstruct`: Produce first-person trajectories.
- `psbot train classical|bc|ppo`: Run versioned experiments.
- `psbot evaluate tournament`: Evaluate checkpoints against frozen opponents.
- `psbot ladder run`: Run the locked ladder checkpoint and record ratings.

## Data, model, and training design

### Data contract

Use one schema version across human and self-play episodes.

- **Episode metadata:** Battle ID, source, format, timestamp, player ratings, winner, simulator commit, seed when available, and schema version.
- **Transition:** Player point of view, turn, observation, chosen action, 26-element legal-action mask, reward, terminal flag, and outcome.
- **Raw record:** Append-only protocol stream, timestamps, decision latency, hashed player identifiers, simulator/data hashes, and reconnect events.
- **Run manifest:** Bot Git SHA, `uv.lock` hash, simulator SHA, configuration hash, seeds, feature-schema version, model hash, and opponent versions.
- Preserve raw compressed replay and input-log files and write processed episodes as partitioned Parquet.
- Split by complete battle, never by turn; deduplicate by battle ID.
- Strip usernames from processed and public artifacts and never include unrevealed opponent information.

Pokémon Showdown exposes JSON replays, input logs for autogenerated-team formats, format search, and pagination through its [official replay API](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md).

### Observation and action contract

- Use `SinglesEnv`'s fixed 26-action mapping: six switches, four regular moves, and four slots for each generation gimmick, with Gen 9 Terastallization occupying actions 22-25.
- Apply the legal-action mask before sampling or choosing an action; illegal actions must have exactly zero probability.
- Encode global field state, side conditions, weather, terrain, turn number, remaining Pokémon, HP, status, stat boosts, revealed species/items/abilities/moves, active matchups, available-move properties, and unknown-value masks.
- Include the previous eight turn events for both players: move, switch, damage bucket, status, faint, and failed-action indicators.
- Use categorical embeddings for species, moves, types, items, abilities, and status; concatenate normalized numeric features.
- Include posterior summaries over plausible random sets: entropy, top-set mass, plausible item/ability/Tera counts, and explicit unknown masks. Never expose simulator ground truth that was not revealed in the player's POV.
- Use a shared two-layer, 256-unit MLP with separate 128-unit policy and value heads.
- Defer GRU and Transformer policies until after month one.

### Training progression

1. Establish `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`, and a stronger custom heuristic using damage, type advantage, switching, status, setup, recovery, hazards, and Terastallization rules.
   - Add a bounded search-informed teacher only after the heuristic is stable. It may use public random-set priors and revealed information, but not hidden simulator state.
2. Train classical baselines:
   - Multinomial logistic regression for action imitation.
   - Logistic regression and gradient boosting for win-value prediction.
3. Pilot human replay reconstruction:
   - Fetch 500 recent `gen9randombattle` replays rated at least 1500; broaden to 1400 only if fewer than 500 exist.
   - Pass the gate if at least 95% of battles reconstruct, at least 99.5% of recorded choices map to legal actions, and POV-leakage tests pass.
   - If passed, scale to as many as 25,000 qualifying replays and train behavioral cloning with masked cross-entropy.
   - If failed by the end of week two, generate cloning data from the strong heuristic with epsilon values 0.05, 0.15, and 0.30; do not delay PPO.
4. Initialize the Stable-Baselines3 masked actor-critic from the behavioral-cloning checkpoint.
5. Use these PPO defaults: learning rate `3e-4`, `gamma=0.99`, entropy coefficient `0.01`, batch size `128`, and 3,072 rollout steps divided across workers.
6. Use these shaped-reward defaults: victory `+30`, opposing faint `+2`, HP differential `+1`, and inflicted status `+0.5`, applied symmetrically. Evaluate a terminal-only reward as an ablation.
7. Apply an opponent curriculum:
   - Early: Random and max-base-power opponents.
   - Middle: Simple and custom heuristics.
   - Late: Heuristics plus three frozen historical checkpoints.
8. Evaluate a deterministic hybrid policy that falls back to the custom heuristic when the policy's top-two probability gap is below thresholds `{0.05, 0.10, 0.20}`. Select the threshold only on the frozen validation tournament.
9. For close model comparisons, reuse the same generated team matchup with seats swapped and run a pre-registered sequential test. Report split-pair rate and measured variance reduction; do not promise a fixed reduction before measurement.

[Metamon](https://github.com/UT-Austin-RPL/metamon) provides evidence that replay reconstruction, imitation learning, offline RL, self-play, and heuristic/model ensembles can produce high-performing Pokémon agents. Its ideas may be adapted, but it will not become a runtime dependency because it does not natively target Gen 9 Random Battle.

## Four-week execution and scaling

### Week 1 - Foundation

- **Contributor A:** Simulator container, environment wrapper, observation/action schema, and replay logging.
- **Contributor B:** CLI, baselines, tournament harness, W&B/Hugging Face setup, and CI.
- Complete one-command local battles and 1,000-battle baseline tournaments.
- Freeze the evaluation seeds and schema version.
- Freeze the append-only raw-log schema and experiment manifest before collecting training data.
- Validate matched-team, seat-swapped replay feasibility; fall back to fixed-seed unpaired evaluation if the simulator cannot reproduce both seats exactly.

### Week 2 - Supervised learning

- **Contributor A:** Replay API client and reconstruction pilot.
- **Contributor B:** Classical models, behavioral-cloning network, and checkpoint conversion into the SB3 policy.
- Apply the replay fidelity gate and immediately use the heuristic fallback if it fails.
- Produce behavioral-cloning accuracy, masked cross-entropy, and local battle results.
- Add random-set posterior summary features only after POV-leakage and calibration tests pass.

### Week 3 - PPO and self-play

- Run local smoke jobs up to 100,000 steps.
- Benchmark 8, 16, and 32 parallel environments; use the configuration with the highest stable battles per hour and zero protocol errors.
- Run six 250,000-step configurations covering two learning rates, two entropy settings, and reward variants.
- Promote the best two to approximately two million steps each on RunPod.
- Fine-tune the winner for up to five million additional steps with the frozen-checkpoint opponent pool.
- Allocate cloud budget: 10% setup, 60% sweeps and training, 20% final fine-tuning, and 10% reserve.

### Week 4 - Selection and ladder

- Evaluate each finalist in 1,000 matched battles against every frozen baseline, swapping sides and reusing seeds where possible.
- Select exactly one checkpoint and hybrid threshold before any public ladder games.
- Verify inference latency, model loading, credential isolation, and replay capture.
- Use a dedicated bot account, one concurrent public battle, no human move intervention, and no model or configuration changes during the 300-game run.
- Publish the full rating curve, peak and final rating, GXE, Glicko-1 with rating deviation, local tournament table, ablations, architecture, costs, limitations, and representative failure cases. Peak 1500 remains the target, but it is never reported without the full curve.

## Tests, gates, and assumptions

### Tests

- **Unit:** Feature ranges, unknown-information masking, eight-turn history ordering, 26-action round trips, reward symmetry, schema validation, and deterministic encoding.
- **Integration:** Docker server readiness, a complete random battle, parallel environments, replay round trip, reconnect/rejoin, matched-team seat swapping, checkpoint save/load parity, and cloud-container smoke run.
- **ML:** Every sampled action is legal; masked logits have zero probability; behavioral-cloning weights load into PPO; fixed seeds reproduce evaluation inputs.

### Release gates

- Zero invalid choices across at least 10,000 local decisions.
- No crashes across 1,000 consecutive battles.
- Inference p95 below 500 ms on the development Mac.
- Local heuristic benchmark confidence interval clears 50%.
- The locked artifact records the code commit, simulator commit, dependency lock, configuration, random seeds, feature schema, and SHA-256 checkpoint hash.
- Every training and ladder episode has a complete raw protocol record and a reproducibility manifest.

### Contingency rules

- **Plateau:** If PPO does not improve against heuristics after two million steps, spend remaining compute on feature/reward ablations and the hybrid policy, not a larger architecture.
- **Budget:** Once 80% of the $50 cloud allowance is spent, stop new sweeps and reserve the remainder for the best checkpoint.
- **Inference gate:** If posterior features fail held-out calibration or leak hidden information, remove them from the model and keep only revealed-state features.
- **Evaluation gate:** If paired-seat reproduction is not deterministic, use frozen unpaired seeds and document the resulting uncertainty instead of patching the simulator during the sprint.
- **Ladder safety:** Train only on the local server. Run one public battle at a time, respect timers and throttles, identify the account as a bot, and stop if requested by Pokémon Showdown staff.

### Assumptions

- Two contributors each contribute 8-12 hours per week.
- Local development uses the M5 Mac with 24 GB RAM.
- RunPod provides the final CUDA environment.
- Weights & Biases and Hugging Face accounts are available.
- A rating of 1500 is a target rather than a guarantee.
- All public replay and ladder use respects Pokémon Showdown limits and policies.
