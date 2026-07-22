# Day 1 foundation

## Completed

- Restored and enhanced the ML/PPO roadmap after reviewing the search-based proposal.
- Pinned Python 3.11, application dependencies, and the Pokémon Showdown simulator commit.
- Added the `psbot` CLI contract and environment-backed settings.
- Added versioned episode and transition schemas with legal-action validation.
- Added fixed-window history and uncertainty-summary primitives.
- Added baseline factories, Wilson intervals, paired-evaluation summaries, and run manifests.
- Added Docker Compose, configuration files, CI, linting, typing, and unit tests.

## Day 2 entry point

1. Implement the first local `RandomPlayer` versus `RandomPlayer` battle command.
2. Persist a complete raw protocol record that validates against schema v1.
3. Verify matched-team reproduction through supported Showdown inputs.
4. Start the replay-reconstruction pilot with a small fixture set.
