# PSBot

An autonomous Pokémon Showdown bot for Gen 9 Random Battles, built on game-tree search over Bayesian
inference about the opponent's hidden team — not machine learning.

**Status:** sprint day 1/25 (2026-07-21 → 2026-08-14)

The interesting parts:

- **It measures itself honestly.** Every strength claim comes from a sequential test on seat-swapped game
  pairs, so the team you were dealt cancels out. Bounds, stopping rules and a ledger of every run are fixed in
  advance.
- **It values information, not just damage.** A move that tells you what the opponent is holding is worth
  something; the search prices that explicitly.
- **It changes gears in the endgame.** When few Pokémon remain and the opponent's sets are pinned down, it
  stops estimating and starts solving.

## Quickstart

Nothing runs yet — the repo is at day 1. See **[Start here — day 1, hour 1](PROJECT_PLAN.md#13-glossary-reading-list-runbook)**.

Full architecture, timeline, gates and cut lines: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**.
Per-pillar writeups land in `docs/writeups/`.

This project previously targeted a reinforcement-learning approach (behavioral cloning + PPO self-play);
see [PROJECT_PLAN.md §0](PROJECT_PLAN.md#0-changelog-open-decisions-assumptions) for why it changed.
