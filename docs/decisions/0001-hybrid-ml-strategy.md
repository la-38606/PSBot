# Decision 0001: Keep the ML spine and adopt search-plan rigor

**Date:** 2026-07-21
**Status:** Accepted

## Context

The original roadmap used classical models, behavioral cloning, and PPO self-play. A later proposal replaced that approach with search and Bayesian set inference. We reviewed that proposal against the project's learning goals, current tooling, and primary sources.

## Valid changes adopted

- Treat hidden-set uncertainty as a first-class input. Gen 9 Random Battle uses generated teams without Team Preview, so revealed-state-only posterior summaries can improve both heuristics and learned policies.
- Use a strong, bounded search-informed policy as a teacher, fallback, and comparison point.
- Add matched-team, seat-swapped evaluation where deterministic reproduction is verified. Measure the actual variance reduction rather than claiming one in advance.
- Use sequential tests for close ablations and Wilson intervals for fixed-opponent win rates.
- Freeze raw protocol logging and experiment manifests before collecting data.
- Report full ladder curves, GXE, and Glicko-1 with rating deviation alongside peak and final ratings.
- Add timer, reconnect, invalid-action, POV-leakage, and state-tracking failure gates.
- Pre-commit cut lines so failed replay reconstruction, posterior calibration, or PPO progress does not consume the month.

## Changes rejected or narrowed

- Removing ML conflicts with the project's explicit purpose and course context.
- A Gen 9 OU competition result does not establish expected strength in Gen 9 Random Battle.
- `foul-play` is GPL-3.0 and remains an external yardstick; PSBot does not copy or vendor its implementation.
- Claims of first-ever deployed contributions require a dedicated literature review and are not month-one success criteria.
- The full-time, weekend-heavy 25-day schedule conflicts with the agreed capacity of two contributors at 8-12 hours each per week.
- Public ladder bots are not unconditionally approved. Current community policy discusses limiters and intervention when bots harm human experience, so PSBot uses one identifiable account, one concurrent game, and stops if requested.
- Patching the simulator for paired evaluation is not allowed during month one unless deterministic reproduction works through supported interfaces.

## Result

PSBot remains an original `poke-env` project. Its performance path is now:

1. Reliable heuristic and search-informed teachers.
2. Classical action/value baselines.
3. Behavioral cloning from human replays when the reconstruction gate passes, otherwise from exploratory teacher data.
4. Masked PPO self-play with a frozen opponent pool.
5. A deterministic hybrid fallback selected only on held-out local tournaments.

Primary references:

- [Pokémon Showdown Gen 9 Random Battle format](https://github.com/smogon/pokemon-showdown/blob/master/config/formats.ts)
- [`poke-env` Gen 9 Random Battle PPO example](https://poke-env.readthedocs.io/en/latest/examples/reinforcement_learning.html)
- [`foul-play` repository](https://github.com/pmariglia/foul-play)
- [Metamon](https://github.com/UT-Austin-RPL/metamon)
- [Smogon ladder-bot policy discussion](https://www.smogon.com/forums/threads/ladder-bots-and-usage-based-tiering.3774656/)
