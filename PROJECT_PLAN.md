# PSBot — Gen 9 Random Battle bot: search + Bayesian set inference

**Status:** first committed revision · 25-day sprint · D1 = Tue 2026-07-21 → D25 = Fri 2026-08-14

A portfolio reader can stop after §3. Everything below the **Execution** rule is for the builder.

---

## 0. Changelog, open decisions, assumptions

**2026-07-21 — supersedes the reinforcement-learning plan at commit `b1e789f`** (behavioral cloning + PPO
self-play, `poke-env` `SinglesEnv`, RunPod, peak-1500 target). No code had been written; earlier drafts were
never committed. Why: the RL route's differentiator is engineering execution on a well-trodden path
(Metamon), whereas search + inference has identifiable whitespace and produces falsifiable results inside 25
days. PPO is cut as out of scope — no ML infrastructure, no compute budget, and learned evaluation is a
stated non-goal (§11). Supervised opponent modelling survives only as post-sprint N6, and never selects our
own moves.

> **BLOCKING — resolve before this plan is executed.** `.gitignore` carries `# Local course material`, and
> the superseded plan's stated learning goal was to progress through heuristic baselines, classical
> supervised learning, behavioral cloning and PPO. **If an external rubric requires a machine-learning
> component, do not take this plan as a clean swap.** Keep search + inference as the spine and add one
> bounded side-track — supervised opponent-policy models per rating band, fitted on this sprint's own logged
> games — which satisfies the requirement and feeds post-sprint N6. Equally: if *learning RL* was the actual
> goal rather than shipping a strong bot, the framing needs to change, not the plan.

**Assumptions.**

- Staffing: **[CONFIRM] — declare one number here** (contributors × hours/week). The superseded plan assumed
  2 × 8–12 h/week; the calendar below assumes full-time solo at ~8 h/day, weekends included.
- Build work is **~25–30 engineer-days (ED)**, 1 ED = 6 focused hours. Arena, SPRT and soak blocks are
  *wall-clock-bound* and cost near-zero extra calendar under part-time staffing.
- Calendar = ED ÷ (weekly hours ÷ 6), plus wall-clock blocks running in parallel. Publish **one** track.
- Hardware: M5 Mac, 24 GB. Sustained arena throughput is unknown until measured on D3–4 and every SPRT
  estimate in §4 is contingent on it.
- Six sprint days are weekends (D5–6 Jul 25–26, D12–13 Aug 1–2, D19–20 Aug 8–9), and the N4 block (D18–20)
  runs Fri–Sun.

---

## 1. Vision and success criteria

Ship, in 25 days, a fully autonomous `gen9randombattle` ladder bot that (a) reproduces a strong
search-plus-inference baseline, and (b) carries **at least two contributions not previously implemented in a
deployed Pokémon agent**, each validated by a pre-registered gate and documented in a short writeup.

1600 Elo in 25 days is **not** a goal. Claiming it would set the project up to fail; the sprint optimizes for
validated novelty per day, not peak rating. The long-term goal survives as the post-sprint roadmap (§11).

SPRT is used for pillar claims (A-vs-B, paired seats). Wilson win-rate CIs are used for baseline claims
against fixed opponents. Ladder rating is descriptive and **never** a controlled comparison.

| # | Criterion | Instrument | Pass condition |
|---|---|---|---|
| 1 | Autonomy | Frozen unattended run, D23–D25 | Target 48 h; **floor** 24 h clean + a documented failure mode. One documented process restart with zero code and zero config change is permitted and reported — the autonomy clock resets, the game count does not |
| 2a | Baseline not broken | vs. max-damage, ≥200 unpaired games | Wilson 95% CI lower bound ≥ 85% |
| 2b | Baseline competent | vs. `poke-env` `SimpleHeuristicsPlayer`, n=200 | Wilson 95% LB > 50% |
| 2c | Yardstick | vs. foul-play — **pinned** commit, identical per-move node budget, identical set-data version | Band 25–40%; 40% is a stretch. foul-play ranked #1 in Gen 9 OU at the NeurIPS 2025 PokéAgent Challenge, so this is not a modest target |
| 2d | Ladder | Frozen hash-recorded build, ≥150 consecutive games | Report GXE, Glicko-1 rating ± RD, final rating, W/L, N, full curve. **Quote no rating while RD > 50. No peak rating anywhere.** Expected band 1250–1400, reported not optimized. Below 1150 after ≥150 games ⇒ local benchmarks are declared miscalibrated and that gap becomes a named section of the writeup |
| 3 | Novelty | ≥2 pillars merged, each passing its **pre-registered** gate + shipping a prior-art note | N1: full-game paired SPRT. N4: endgame-suite delta + reported SPRT estimate. N5: held-out N5-on/off calibration. **M cannot count** — it is the harness and cannot be SPRT'd against a build without it. N5 is isolated by the D24–25 ablation, not by its D11–14 gate |
| 4 | Record | Per-pillar writeups + Elo-over-anchor graph + public repo | Shipped by D25 |

**Pre-registered results table** — empty on day 1, filled as gates pass. Instrument and pass condition are
fixed now; only the result column is written later. *(sprint day 1/25 — all rows pending.)*

---

## 2. Prior art: what is already done

Novelty claims only mean something against an honest map. Default phrasing throughout is *"first in a
deployed Pokémon agent"* or *"under-explored in deployed bots."* The word **novel** requires positive
evidence.

| Approach | Who | Status |
|---|---|---|
| Root-parallelised MCTS with Decoupled UCT over sampled sets | foul-play (current default) | The open-source reference. Expectiminimax remains a selectable mode; pmariglia's stated reason for the switch is that monte-carlo handles simultaneous moves better. **foul-play is not a maximin bot** |
| Competition result | NeurIPS 2025 PokéAgent Challenge | FoulPlay ranked #1, Gen 9 OU |
| Bayesian-ish set inference from reveals, damage, speed | foul-play | Established |
| Nash equilibrium move selection | foul-play `nash_equilibrium` bot (via Gambit) | Implemented and widely reported as playing badly (issue #65 enumerates ~10 failure modes). **No author post-mortem located** — finding one is a search task with a null outcome allowed, not an available artifact |
| RL from human replay corpora | Metamon | Established (imitation + offline RL; already population-exploitative) |
| LLM-guided move selection / minimax | PokéChamp, PokéLLMon | Established |
| MCTS / decoupled UCT for simultaneous moves | Academic Showdown work; poke-engine | Established |
| VOI / information-gain terms inside PIMCTS/ISMCTS | Sinclair et al., LNCS 2023 | Established **outside** Pokémon; ρ-POMDP is the formalism, Reconnaissance Blind Chess an entire competition |
| Per-matchup 1v1 Nash via `scipy.linprog` | salm.dev | Same tool, known teams, no live posterior |
| Greedy Tera heuristics; heuristic bots | Various | Established; sparring partners only |

**Discipline.** The prior-art sweep runs on **D2, not D21** — 4 hours, five notes (M, B-lite+N5, N1, N4, N3),
each carrying: nearest prior implementation + URL, what we must build *differently* because of it, which
benchmark to reuse, and the pre-agreed honest phrasing. N1's and N4's notes are gate items on D7 and D18. A
1-hour D21 refresh covers anything published mid-sprint. Scheduling this check *after* 36% of the sprint has
been spent building N1 and N4 would let it change an adjective and nothing else.

---

## 3. Pillars

| ID | Pillar | Claim | Nearest prior art | Role | Prereq | Gate | ED | Cut trigger |
|---|---|---|---|---|---|---|---|---|
| **M** | Paired-seat SPRT harness | Sequential paired-matchup testing in random battles, with the VRF measured for gen9randombattle | Fishtest pentanomial; mirrored fixed-N evaluation is common | Core, first | — | Harness validation (§6 D3–4) | 2 | none — no fallback exists |
| **B-lite + N5** | Set inference w/ tempered inaction evidence | Explicit *calibrated* inaction evidence in a rule-based bot | Metamon learns it implicitly | Core | M, search | SPRT + held-out calibration | 4 | §7 |
| **N1** | Information-valued search (EVSI) | First explicit value-of-information term in a deployed Pokémon agent; first measurement of the clairvoyance gap in Pokémon | Sinclair 2023; Frank & Basin 1998 | Core | B-lite | Paired SPRT vs. scale=0 | 3 | null by D16 |
| **N4** | Posterior-conditioned endgame solving | First Showdown bot that switches solver mode when the game gets small | Chess tablebases; poker subgame solving; salm.dev | Core | B-lite, calibration | Endgame-suite delta (primary) | 3 | pre-gate f < 0.22 |
| **N3** | Timing tells | Opponent think-time as a posterior input — plausibly first anywhere | Poker timing tells | Offline study only | D1 logging | Effect sizes vs. permutation null | 1 | none — a null is publishable |
| **N7** | Variance-aware node allocation | Reallocating a fixed node budget across turns | Chess time management | **Cut-line substitute, not scheduled** | search | Rides D24–25 ablation | 0.5 | — |
| N2 | Information concealment | — | InfoChess (arXiv 2604.15373) | **Post-sprint** (§11) | — | — | — | — |
| N6 | Rating-band exploitation | — | — | **Post-sprint** (§11) | — | — | — | — |

### M — Paired-seat SPRT *(core, built first)*

**Idea.** Random-battle outcomes are dominated by team-matchup luck. Generate the matchup once, play it twice
with the seats swapped, score the pair, run SPRT on the pair statistic. Keep the name *paired pentanomial
SPRT* — it points at the Fishtest prior art whose port is the claim — with one parenthetical: with ~zero
draws two of the five pair cells vanish and the statistic is effectively **trinomial** on pair scores {0,1,2}
(ties scored ½; revert to the full pentanomial if the observed tie rate exceeds ~0.5%). The variance we
cancel is *team-matchup luck*, not draw structure.

**Honest arithmetic.** VRF = 1 ⁄ (2·(1 − P_split)), where P_split is the fraction of paired matchups scoring
1-1. P_split 0.55 → 1.11; 0.60 → 1.25; 0.65 → 1.43; 0.70 → 1.67. **A 3× reduction would require P_split =
0.833** — i.e. the team draw deciding ~85% of games before turn 1, which is false for a format where the
reference bot sustains a high GXE. Expect **10–40% fewer games, not "severalfold."** P_split is
agent-dependent; re-measure once the D9 baseline exists.

### B-lite + N5 — set inference *(core)*

Exact weighted enumeration over randbats sets; hard filtering on reveals; **damage-roll inversion** — note
the asymmetry: damage *to us* is exact since we know our own HP, damage *we deal* is quantized to displayed
integer percentages. Species-fixed levels and near-uniform spreads collapse the free parameters to
item/ability/Tera, so both are often identifiable from one or two hits. This is the highest-yield inference
channel in the format.

- **`gen9randombattle` has no team preview; species are learned only on switch-in.** This is the defining
  information structure of the format.
- **Tera type is a first-class hidden variable**: per-species Tera lists are public and typically 2–8 wide,
  revealed only by use, narrowed by a revealed Tera Blast.
- Unrevealed team slots are sampled from the species marginal filtered **only** by free constraints
  (species/base-species uniqueness against the revealed five, plus type caps if they read straight off the
  generator). Full joint team-constraint modelling is a named non-goal (§11).
- **Importance-weighted determinization**, specified precisely: deterministic top-K enumeration with
  renormalised weights and a **reported residual mass** (1 − top-K mass) — *not* i.i.d. sampling then
  weighting by mass, which double-applies the density. Determinize hidden state only; damage rolls, accuracy
  and secondaries stay as chance nodes. Factorise per Pokémon. Report ESS = (Σp)²/Σp² rather than K. Half-day
  offline study on 500 logged positions: smallest K whose root argmax agrees with K=64 on ≥95% of positions
  becomes production. *A small twist over foul-play's most-likely-set sampling, not a claimed pillar.*

**N5 guard rails.** A free turn where the opponent didn't click X lowers the posterior on sets containing X,
with the likelihood tempered (raised to λ < 1) and per-turn evidence weight capped. Plus:

- **Hard zero-evidence list** — apply no inaction likelihood under Choice lock / Gorilla Tactics, Taunt,
  Encore, Disable, Torment, sleep/freeze, recharge, out of PP, Imprison, Truant/Slow Start, or no legal
  alternative. (Trapped-cannot-switch is evidence about *switching*, not about move choice.) Tempering fixes
  an overconfident likelihood, not a wrong one.
- Never treat a `[from]`-tagged move (Sleep Talk, Copycat, Dancer, Magic Bounce, Instruct, Transform/Mimic)
  or Struggle as a revealed move.
- The likelihood multiplies **set weights inside the enumeration**, never as an independent per-move factor —
  so it propagates to role, item and Tera exactly, and for free.

Novelty phrasing, verbatim: *Metamon learns this implicitly, so we don't claim the concept, we claim the
explicit calibrated treatment.* Measure calibration (log-loss / Brier), not top-k hit rate. Stated
limitation: N5's evidence is **weakest at the 1250–1400 band we target**, because opponents there misplay
constantly — which is the hook for post-sprint N6. Rating-band conditioning is N6 and stays deferred.

### N1 — Information-valued search *(core)*

**Idea.** Partition worlds by the observation class *c* our action induces next turn (their revealed move,
item proc, damage bracket, speed order):

```
U_uninf(a) = max over ONE shared continuation action a' of  Σ_k p_k · V_k(a→a')
U_ii(a)    = Σ_c P(c|a) · max_a' Σ_{k∈c} (p_k / P(c|a)) · V_k(a→a')
EVSI(a)    = U_ii(a) − U_uninf(a)  ≥ 0
```

The max over continuation actions is taken **inside the observation class, not inside the world**. This is a
change to the *aggregation order*, reusing values the search already produced.

**Why this and not entropy.** Entropy reduction and VOI are different quantities, and the difference is the
whole pillar: VOI is exactly zero when the observation cannot move the argmax, while entropy is generically
positive. An entropy bonus therefore pays real tempo to resolve the 4th move slot and the EV spread — which
dominate joint set entropy and are near-decision-irrelevant — while the one bit that flips whole decisions
(Choice-locked or not) earns almost no entropy credit. Both cost the same to compute.

**Why it is a repair, not a bolt-on.** Determinization/PIMC is provably information-blind: inside each
sampled world the hidden state is known, so the value of information is structurally zero. U_clair ≥ U_ii ≥
U_uninf, so a determinized search both destroys VOI *and* over-estimates its own value. Adding a bonus on top
of an already-clairvoyance-inflated value would double-count. **N1 is the targeted repair of a documented
defect of our own architecture.**

**One free parameter only** — a scale/myopia correction, default 1.0, justified by D9's calibration putting V
in win-probability units. **No λ sweep.** Entropy survives as a reported diagnostic only.

**Measure.** Paired SPRT vs. the scale=0 build. Free deliverable: publish **U_clair − U_ii by turn number** —
the clairvoyance gap in Pokémon, never measured. Qualitative signatures to look for are the randbats-real
ones, not the textbook one: pivot moves (U-turn/Volt Switch/Flip Turn/Chilly Reception/Parting Shot) revealing
their switch-in, hazards testing Heavy-Duty Boots, forcing a speed comparison to rule Choice Scarf in or out,
baiting Tera before committing a win condition. Protect-to-scout is the human illustration, not the workload.

**Risk.** Most information in randbats arrives **for free**, because the opponent must act every turn and each
action reveals a move — so marginal VOI concentrates in the one-shot resources (Tera, Choice lock, Boots).
EVSI(a) may also be near-constant across *a* when observations are driven mostly by the opponent's forced
action. Pre-registered failure signature: under an entropy objective, expect Protect/U-turn/status-probe spam
that worsens with λ — if that shows up, the objective is wrong, not the idea.

### N4 — Posterior-conditioned endgame solving *(core)*

Exact at fully-resolved 1v1; depth-capped matrix solving at 2v2.

**The honest premise.** `gen9randombattle` has **no team preview**, so at 2v2 one opposing mon may never have
been seen, and **Tera becomes the dominant residual unknown** because it is revealed only by use. A 2v2 turn
is up to ~10 actions per side (4 moves + 1 switch + 4 Tera-variants) ≈ 100 joint actions, each a chance node
over 16 damage rolls × 1/24 crit × accuracy (Focus Blast 70%, Hydro Pump/Stone Edge 80%) × ~30% secondaries ×
50/50 speed ties re-rolled every turn, across the posterior-consistent world combos; randbats endgames
commonly run 8–20 turns with recovery, hazards and Leftovers chip. **Only the fully-resolved 1v1 subgame is
exactly solvable.** The word *exact* is struck as a general descriptor, and so is "trivially fast at this
size."

**What it is.** Equilibrium computation within a determinized world-ensemble; a **best response to a frozen
belief**, i.e. *unsafe subgame solving* in the Brown–Sandholm sense (arXiv 1705.02955). Exact given our
posterior; not equilibrium-exact and not unexploitable. Against a ladder population that is a feature — but
the writeup must not say "near-exact solution."

**What still differs from depth-2 search:** equilibrium solution of simultaneous-move nodes instead of
maximin, exact terminals, and far greater depth.

**Trigger.** ≤2 mons per side **AND** every opposing team slot revealed (explicitly: no unseen 6th) **AND**
each opposing mon's posterior within *k* set-combos at ≥90% mass, **with Tera type as a combo dimension**,
**AND** per-move PP tracked on both sides.

**Build.** Transitions via `poke_engine.generate_instructions(state, s1_move, s2_move)` applied and undone
through `apply_instructions()` / `reverse_instructions()` — no Node subprocess inside the search loop; the
Node sim stays a golden-diff oracle. **PP is a correctness precondition, not a detail**: monotonically
decreasing PP terminating in Struggle is what makes the stall/recovery graph a finite DAG (verify poke-engine
exposes per-move PP on D7–8). Memoize on the canonical exact state (hp1, hp2, PP vector, status, boosts,
field, tera-used). Hard node budget from measured throughput, calibrated eval as leaf fallback. Solve nodes
with a dependency-free small-matrix zero-sum solver (a few hundred regret-matching iterations on a ≤10×10
matrix, microseconds in numpy) — **not** scipy per node: a ~0.54 ms 5×5 HiGHS solve is 54 s at 1e5 nodes and
busts the move clock. Bin damage rolls **only within KO-equivalence classes, never across a faint boundary.**

**Measure.** The **endgame suite leads**: positions harvested by log-filtering the D5–D17 streams (1–2 hours,
not a day), persisted as `State.to_string()` under `suites/endgame/` with expected-outcome metadata and a
pytest-marked runner. Report the solver-invocation split — both-sides-resolved (exact) / one-sided (maximin
vs. clairvoyant) / residual uncertainty (approximate). The full-game paired SPRT still runs but is reported
as a fixed-N Elo point estimate with CI, not pass/fail — which is also the honest writeup line: *measurable in
the subgame, below our full-game detection threshold.* Objection to pre-empt: foul-play's DUCT already
approximates the matrix solution in a tiny endgame, so the harvested suite is what isolates the marginal gain.

### N3 — Timing tells *(offline study only; a null is publishable)*

**Premise, corrected.** `|t:|` carries a UNIX timestamp in **whole seconds** and heads every protocol update
block (a checked `gen9randombattle` log has 28 `|t:|` lines against 19 `|turn|` lines) — **not one per
action**. There is no per-decision clock, so regular-turn wall time is approximately *max(our think time,
their think time)* at 1-second resolution.

Two channels rescue it. **(i) Forced-switch blocks after a faint give an uncensored single-player clock**,
because only one side is choosing — ~7–11 faints per battle, ~3–5 opponent-only observations per game, and it
is exactly where the consequential "which check do I bring in" decision lives. **This is the primary
channel.** (ii) `|inactive|` is not binary: its free-text payload carries per-player seconds remaining — parse
and log it from D1 as a coarse long-tail channel.

Log our own per-decision latency from D1 and treat regular-turn opponent think time as **right-censored at our
latency**, analysed with a survival/Tobit model rather than a plain conditional fit; exclude or separately
model forced-switch sub-turns. One pre-registered hypothesis, tested with a **stratified permutation null**
(shuffle durations within turn-number × rating-band strata, 1,000 shuffles), reporting effect size in seconds
plus rank-biserial — *not* a plug-in mutual-information estimate, which is upward-biased at these sample
sizes. State expected N up front. This is a data-collection defect, not an analysis defect: it cannot be fixed
on D21, because twenty days of logs would already be corrupted.

### N7 — Variance-aware node allocation *(cut-line substitute, not scheduled)*

A **fixed total node budget per game, reallocated across turns** — not more wall clock — so it is
throughput-neutral and SPRT-able against an equal-node-budget uniform-allocation baseline. Rule: after the
shallow pass, compute the probability that the top-2 action gap exceeds the calibrated eval noise band;
deepen only while that probability is low. It is the pre-committed reroute if B-lite or N1 fails, and it takes
the D23-window stretch slot vacated by N2. No separate SPRT block — it rides the D24–25 ablation batch.

---

<a name="execution"></a>
## ═══ Execution ═══

## 4. Measurement contract

Every gate in §6 points here.

**(a) Bounds.** In-sprint pillar gates run at **elo0 = 0, elo1 = 20, α = β = 0.05**. Use [0,10] only for a
pillar whose [0,20] point estimate already exceeds +30 Elo; [0,35] for screens and ablations. Fishtest-style
[0,2] is out of reach on hobby compute. **Accepting H1 at [0,20] does not license writing "+20 Elo"** — every
accept/reject is reported with the point estimate and 95% CI. "Rejects H1 at [0,20]" means *worth under 20
Elo*, not *the pillar failed*.

**(b) Expected games (unpaired).** At [0,10]: a boundary-true effect costs ~6.4k games, a true +5 Elo (the
real worst case) ~10.5k, a genuine +40 terminates in ~900. At [0,20], roughly a quarter of each. Divide by the
VRF measured on D3–4. **The budget risk is failing candidates and sweeps, not confirming winners.**

**(c) Truncation.** Every gate carries a pre-committed cap of **1,500 pairs**. At cap: stop, report point
estimate + 95% CI, mark **INCONCLUSIVE** — the pillar may ship but is not counted as validated. SPRT's
unbounded stopping time is the real schedule bomb.

**(d) Fixed budget.** All arena games run a fixed **node/iteration** budget, single-threaded
(`mcts(state, iterations=N, threads=1)`), never `duration_ms`; time budgets are ladder-only. Record the node
budget in every run; changing it is a build change requiring a new anchor. Honest consequence: pillar effects
are validated at the arena budget and we do not claim they transfer unchanged to the ladder's budget.

**(e) Pairing contract** (testable spec). **Held fixed** across the two games of a pair: the two generated
teams in full (species, level, moves, item, ability, nature, IVs/EVs) — nothing else. **Swapped:** which agent
controls which team *and simultaneously* which protocol slot (p1/p2) it occupies, so each agent plays one p1
and one p2 per pair. **Re-randomized with fresh independent seeds:** all in-battle RNG (damage, crits,
accuracy, secondaries, speed ties, sleep/confusion/Rest durations) and each agent's own sampling RNG — never
share one RNG object between agents. Rationale: common-random-numbers coupling is unattainable because
Showdown consumes one PRNG stream in event order and the streams desynchronise on the first divergent move;
independent draws keep Var(pair) = 2·E[r(1−r)] exact and the analysis one-parameter.

**(f) How paired teams are obtained** — not an open question, and not free. `PlayerOptions.seed` exists in
`sim/global-types.ts` and `Battle.getTeam` honours it, but `server/room-battle.ts` writes only
`{name, avatar, team, rating}` into `>player`, so **on the websocket path a per-player seed is not
forwarded**. Three named options, **decided in 1–2 hours on D3, not investigated**: **(a) [default]** ~3-line
patch to our pinned local server forwarding a per-player seed into `>player` — keeps the arena on the real
websocket path, costs a maintained fork; (b) drive the arena directly on `BattleStream` /
`pokemon-showdown simulate-battle` — no patch, but arena and ladder then exercise different transports;
(c) a custom local format accepting packed teams plus a pre-generation step. Whichever is chosen, add a
conformance check: the arena path and the ladder path must produce identical agent decisions on identical
states.

**(g) Integrity.** A **public run ledger** logs every arena run ever launched — including abandoned and re-run
ones — with build hash, bounds, LLR trace, node budget, set-data hash and stopping reason. This is what stops
"≥2 pillars must pass" from becoming "re-run until two pass." Parameter tuning uses **fixed-N A/B with common
random numbers, never SPRT** (k SPRTs at α=0.05 inflate false-accept to 1−0.95^k ≈ 23% at k=5, and the argmax
is biased upward by ~1.16 estimator-SDs); SPRT is reserved for the single accept/reject gate on the selected
configuration, run on **fresh matchups never used in the screen**. Bonferroni is explicitly rejected as
unaffordable (α=0.01 raises the boundary 2.944 → 4.595, ~56% more games per test).

**(h) Frozen anchor.** The D9 search baseline is frozen, hash-recorded and never rebuilt; every subsequent
version plays it in the background queue from the day that version exists, so the Elo-over-version graph has a
real y-axis instead of a chain of pairwise deltas. Per-pillar deltas are **not additive and will not be
summed**. The graph is descriptive; only the pre-registered gates are confirmatory.

---

## 5. Architecture

**Chassis — decided D1, both branches costed, fork is the default.** foul-play (GPL-3.0) already ships
`fp/battle/protocol.py` (~87 KB), `fp/battle/inference.py` (~29 KB — damage-roll inversion, speed brackets,
Boots inference), `fp/search/poke_engine_helpers.py` (~14 KB, the `Battle`→`State` adapter), 
`fp/data/sets/randbats.py`, its own 6 KB websocket client, and ~273 KB of protocol regression tests; deps are
`requests` + `websockets` + `python-dateutil` + `poke-engine`, no `poke-env`.

- **Fork branch:** the repo becomes GPL-3.0; D5–6 and D11–14 collapse to delta work; **unmodified upstream
  becomes the clean A/B control** for every paired SPRT (gate: paired SPRT vs. foul-play-unmodified at [0,20]
  does not accept H0, reporting an Elo point estimate whose 95% CI lower bound > −10). Success criterion 1(a)
  ("reproduces a strong search-plus-inference baseline") must be rewritten or deleted. **Do not raise the
  ladder band** — re-baseline after the first 100 ladder games.
- **Greenfield branch:** add ~12 ED, and name the `poke-env` → poke-engine `State` adapter as its own 3–5 ED
  block. This work is otherwise invisible.
- **Cut line:** if the chassis is not laddering by end of block 1, fork and re-baseline.

Deciding this on D12 instead of D1 is the plan's biggest schedule risk: §5's own principle is *the
originality lives in N1/N4/N5/M, not in re-porting a damage formula*, and the greenfield branch spends 10 of
25 days re-porting exactly that.

**Buy the engine, build the brain.** poke-engine (Rust, PyO3) for transitions; the pinned Node sim as ground
truth. Honesty cost: less of the stack is ours — acceptable, since the originality is in N1/N4/N5/M. A custom
engine is post-sprint if still wanted.

**poke-engine is not a mechanics de-risk.** Its own README says it is "nowhere near as complete or robust as
the PokemonShowdown battle engine." A **nightly golden-diff harness** replays K sampled turns from logged
battles through both poke-engine and the pinned Node sim and reports divergence rate. Engine/sim divergence is
a named, measured error source. Note the pre-named fallback ("foul-play components") is the *same engine
family* and is therefore **not** a fallback for a mechanics-coverage failure.

**What the PyO3 boundary exposes:** `calculate_damage`, `generate_instructions`, `id` (root payoff matrix),
`mcts` (per-move `total_score` and `visits`), `State` (constructible, `from_string`/`to_string`),
`apply_instructions`/`reverse_instructions` — and **no evaluation hook**.

**Search baseline, resolved D7 (not D15).** poke-engine's MCTS/DUCT called as a **black box** over
importance-weighted posterior-sampled worlds, matching foul-play's actual architecture. Depth-2 expectiminimax
is kept only as a small-tree debug oracle. Because there is no eval hook, **N1 operates at the root**: take
per-world root payoff matrices from `id()` and per-move values from `mcts()`, and compute EVSI by re-maxing
over those cached per-world values under posterior reweighting by observation class — no nested re-search, no
Rust fork. Rejected alternative, stated: our own depth-2 Python search over `generate_instructions` buys
leaf-level VOI but loses the engine's search quality.

**Root shared-strategy LP — shipped as its own SPRT-gated change, not an assumed baseline swap.** Solve once
per turn at the root: maximize Σ_k p_k·t_k s.t. t_k ≤ Σ_a x_a·V_k(a,b) ∀k, ∀b∈B_k; Σ_a x_a = 1; x ≥ 0. For
|A|=10, K=16, |B|=10 that is 26 vars / 160 constraints, ~1–3 ms. One strategy shared across all worlds in our
information set is what eliminates **strategy fusion** — averaging per-world solutions is an equilibrium of
nothing, and a 50/50 blend of two pure per-world optima can be dominated by both. The pure-strategy
restriction recovers max_a Σ_k p_k min_b V_k(a,b) exactly, so the LP *value* dominates by construction — but
value dominance is not a win-rate gain against a non-best-responding population, hence the gate. **Do not
apply it recursively at every node** (~10–30 s/move at a 1e4-node budget).

**Eval — named, not hand-rolled.** Start from poke-engine's / foul-play's published evaluation: HP fraction
per mon, alive count, hazards by layer and side, status, boosts, item consumed, Tera remaining, and a coarse
remaining-team type-coverage term standing in for the deferred Pillar C. Eval quality sets the D10 yardstick,
the D9 calibration units, and whether any pillar's effect is visible above the noise floor.

**Data.** Pin **one** randbats set-data version and **one** sim commit for the whole sprint; record both
hashes in every battle log and every SPRT run; re-pull exactly once before the D23 ladder campaign.
Pinned-local vs. live-ladder set drift is a known systematic bias on ladder numbers. Commit the pinned
snapshot (`data/randbats/` un-ignored) so the version is reproducible, not merely diagnosable. *Optional if
D5–9 has slack:* generate ~1e5 teams from the pinned sim for the exact within-role joint set table; otherwise
use pkmn/randbats role-conditional data and name within-role independence as a known approximation.

---

## 6. The 25-day timeline

Gates attach to **block completion, never to a date**. If actual effort is below the declared rate by D9, the
plan re-baselines to the cut-line set (M + B-lite + one pillar) rather than sliding dates. D-labels are block
names, cross-referenced by §7.

| Block | Dates | ED | Wall-clock-bound | Deliverables | Gate |
|---|---|---|---|---|---|
| **D1–2** | Jul 21–22 | 2 | partly | Pinned local server; registered bot account; random bot on local **and** ladder; **log schema v1 frozen with its validation test green before the first ladder game**; autonomy primitives (supervisor with restart-rate limiting — max 5/hour then quiesce; gzipped per-battle log rotation with a disk cap; conformance to the server's *measured* throttle; the rule that **the state tracker is a pure fold over the protocol stream, replayable from scratch**); 45-min protocol spike parsing `\|t:\|` and `\|inactive\|` against five real logs; **4-hour prior-art sweep (5 notes)**; move-clock + time-bank validation | 50 ladder games, zero illegal choices, zero timeouts; 10,000-decision legality fuzz clean; schema test green; 5 prior-art notes filed |
| **D3–4** | Jul 23–24 | 2 | yes | **Pillar M**: paired-seat arena, SPRT runner, CSV + `make report` (not a bespoke dashboard). Decide the paired-team mechanism (§4f, 1–2 h) | Run entirely with zero-compute agents so ~6,000 games is minutes. (i) **A/A null**: baseline vs. byte-identical copy, 1,500 pairs — must fail to reject H0, 95% CI must contain 0 (±11 Elo noise floor at that N; **not** ±5). (ii) **Monotone known-signal**: ε-random handicap at ε ∈ {0.02, 0.05, 0.10}, fixed N, Elo monotone decreasing. (iii) Report P̂_split, VRF with Wilson CI, and sustained **arena battles/hour on the dev box while the dev box is in use**. (iv) Smoke: max-damage > random concludes in <100 games. *Nothing here is pass/fail on M — there is no fallback if M "fails."* |
| **D5–6** | Jul 25–26 *(wknd)* | 3 | partly | State tracker + max-damage bot. **Replay corpus**: ~300 recent `gen9randombattle` replays via the official replay API, **plus** a targeted subset with ≥10 battles each of Illusion, Imposter, Revival Blessing. **Verify input-log retrieval for autogenerated-team formats** — input logs carry the generated teams, i.e. ground truth for the opponent's full set, which becomes the labelled evaluation set for D11–14. Reconnect/backoff/re-login and mid-battle rejoin land here (they need the pure-fold property) | Zero **unexplained** divergence on search-input fields (HP, status, boosts, hazards, weather/terrain/screens/Trick Room, known moves, known items, Tera used); no unfixed divergence in a mechanic occurring in >1% of the corpus, with a named exclusion list, each an open ticket. ≥95% of battles reconstruct end-to-end; ≥99.5% of recorded choices map to legal actions. No crash across 1,000 consecutive local battles. Ladder item relabelled: "N games completed, zero protocol errors, zero timeouts; rating reported as a wide band, explicitly not used for comparison" |
| **D7–9** + ½ D10 | Jul 27–29 | 4 | no | **First line of D7 is set-prior-lite** (load pinned set table, hard-filter on revealed move/item/ability/Tera, sample worlds by prior weight — no inversion, no inaction evidence, no importance weighting): the depth-2 search cannot run without *some* rule for filling unknown sets. Search baseline on poke-engine. **Resolve on D7:** (i) whether the search has a decision ply *after* the observation ply (EVSI needs it; fallback is entropy over a coarse behaviour-relevant partition — Choice-locked y/n, Boots y/n, OHKO-range y/n — declared a fallback); (ii) whether poke-engine exposes per-move PP (N4's DAG depends on it). Golden-diff vs. Node sim covering endgame mechanics specifically. **D9: eval→win-prob calibration fit.** Old D10 folded in as a half-day | ≥200 games vs. max-damage, Wilson LB ≥85%; `SimpleHeuristicsPlayer` n=200, LB >50%; calibration curve published; foul-play head-to-head recorded with commit, node budget and set-data version pinned |
| **D11–14** | Jul 31–Aug 3 | 4 | partly | **B-lite + N5**: enumeration posterior, damage inversion, reveals, Tera posterior, tempered inaction evidence with the zero-evidence list, deterministic top-K importance weighting with residual mass reported, half-day offline K study. **Refit eval→win-prob calibration at D14** — N1's scale and N4's thresholds are denominated in a curve D9 fitted against a belief model B-lite has now replaced | Package SPRT pass vs. the D9 build at [0,20]; **and** N5-ON vs. N5-OFF improves held-out log-loss/Brier on the *same* games, offline, zero extra games played; top-3 set accuracy + log-loss reported on the input-log labelled set, never on the tuning games |
| **D15–17** | Aug 4–6 | 3 | partly | **N1**: EVSI restructuring of the aggregation order; root shared-strategy LP as a *separately gated* change; clairvoyance-gap instrumentation. **No λ sweep.** If the scale must move off 1.0: three values, fixed-N, bounded to ≤10% of the calibrated win-prob scale, with a 300-pair screen used **only** to rule out gross mis-scaling (state that ±35 Elo is all that screen resolves), then one confirm SPRT at [0,20] on fresh matchups. **End of D17, half-day, free: the N4 pre-gate** | SPRT pass vs. the D14 build; clairvoyance-gap (U_clair − U_ii) curve by turn published |
| **D18–20** | Aug 7–9 *(Fri–Sun)* | 3 | partly | **N4 — pre-gate first.** Replay the logged D1–D17 **arena self-play** games (the population the SPRT samples, not ladder logs) and measure *f* = fraction of games reaching the trigger while calibrated win-prob is still in [0.2, 0.8]. Pre-commit: assuming a generous conditional-edge ceiling of 0.10 score units, **f < 0.22 ⇒ the pillar cannot clear ~15 Elo** — loosen the trigger once (≤3 mons, or drop the posterior-mass condition and solve sampled worlds under the node budget), re-measure, and cut only if the loosened trigger still misses. Then build: trigger, PP-keyed memoization, regret-matching node solve, node budget, harvested suite | **Leads with** endgame-suite winrate delta + solver-invocation split. Full-game paired SPRT runs but is reported as a fixed-N Elo point estimate with CI, not pass/fail. Arithmetic published: 15 Elo = 0.022 score units, so 20% firing × 5% edge = 7 Elo (undetectable at any affordable bound); 50%×5% = 17; 30%×8% = 17 |
| **D21** | Aug 10 | 1 | no | **N3 offline timing study** (full day, since the sweep moved to D2) + 1-hour prior-art refresh | Effect sizes with permutation-null p-values; forced-switch channel separated from censored regular-turn data. **A null is publishable** |
| **D22** | Aug 11 | 1 | no | **Attended soak**: run the final build, fix crashes, deliberately force disconnects, exercise rejoin. **Fixes allowed; this day does not count toward the autonomy claim** | Build tagged and hashed at end of day |
| **D23 00:00 → D25 00:00** | Aug 12–14 | 0 | yes | **Frozen unattended run**: no code, no config changes. The 48 h autonomy claim and the ≥150-game ladder run are the **same run**, ending at the later of 48 h and 150 games; hard stop D25 06:00. Arena capped to cores−2 workers (or ablations moved to a second machine); ladder move-time p95 monitored, throttling the arena if it exceeds budget — a git worktree isolates code, not CPU. N7 takes this window's stretch slot if quiet | §1 criteria 1 and 2d |
| **D24–25** | Aug 13–14 *(overlapping the frozen run)* | 2 | partly | **Untouchable.** ONE pre-registered leave-one-out SPRT at [0,20] on the headline pillar; Elo-over-version graph from the frozen-anchor background queue (each version already has 1,500+ pairs against the common anchor at zero deadline cost), labelled descriptive; the N4 exploitability number computed **here**, not in D18–20; optional external-comparability run on PokéChamp's 1,000-position benchmark **with caveats stated** (Gen 8, one mon per side, no switching, win-filtered); writeups against `docs/writeups/TEMPLATE.md` | Ablation result + ≥2 validated-pillar writeups shipped |

---

## 7. Cut lines and risks

Pre-committed, so variance doesn't make the decisions.

| Risk | Early-warning signal | Pre-committed action |
|---|---|---|
| Baseline gate slips > 2 days | D9 | Drop N4, keep N1 |
| B-lite fails its SPRT | D14 | N1 is meaningless — VOI is identically zero when the posterior doesn't change the argmax. Pivot D15–17 to N4 early, D18–20 to N7 + B-lite debugging |
| N1 null | D16 | At most half of D17 on the decision-relevant fallback, then declare null and start N4 early |
| N4 pre-gate f < 0.22 after one loosening | D17/D18 | Cut N4; reallocate to N7 + eval tuning |
| Chassis not laddering | end of block 1 | Fork foul-play and re-baseline |
| Effort below declared rate | D9 | Re-baseline to the cut-line set — **do not slide dates** |
| poke-engine bindings friction | D8 | Decide by end of D8, don't thrash. Note the "foul-play components" fallback is the same engine family and is **not** a fallback for a mechanics-coverage failure; the real fallback there is the named exclusion list + golden-diff-measured error rate |
| Eval noise drowns pillar effects | D9 calibration | If calibrated win-prob differences between candidate moves sit inside the noise band, N1's scale has nothing to work with — the cut lines fire |
| Arena throughput below plan | D3–4 | Widen gates to [0,35] or cut a pillar; run SPRTs at a fixed reduced node budget with a one-time validation that pillar ordering is preserved at full budget |
| Novelty collision | D2 sweep (not D21) | Fallback phrasing per §2 |
| Ladder sample size | throughout | Report a band with RD; lean on local paired testing for strong claims |
| Scope creep | throughout | The deferred list (§11) is the contract |

**Both novel pillars can't land** → the sprint deliverable becomes M + B-lite + the timing study + an honest
writeup. Methodological novelty (M) alone is a real contribution; say so and mean it.

**Nothing un-cuts the writeups. Days 24–25 are untouchable — undocumented results don't exist.** Resist any
urge to add "unless we're close."

---

## 8. Autonomy and ladder operations

Fully autonomous loop, permitted by ladder rules; the 2026 Smogon policy discussion concluded a ban is not
enforceable, with action only where bots negatively affect human experience, and concerns usage-based tiering
in OU-style tiers rather than random battles. No bug exploitation. **Never ladder two of our own accounts** —
all bot-vs-bot on the local server.

One registered account with a bot-identifying name; one concurrent battle; message spacing at the *measured*
server throttle; reconnect with backoff; respect the timer. **Mid-battle crash recovery via rejoin** (Showdown
resends the room's battle log on join) — which is exactly why the state tracker must be a pure fold over the
protocol stream.

Log everything from game one: protocol stream, `|t:|` timestamps, `|inactive|` payloads, opponent rating, our
own per-decision latency, decision traces (search values, posterior per turn), randbats data hash, sim commit,
server version.

---

## 9. Engineering contract

**Release gates.** Zero invalid choices across ≥10,000 local decisions (D1–2). No crash across 1,000
consecutive local battles (D5–6 — a harness a few hours old cannot credibly run 1,000 battles). Every logged
battle carries a complete protocol stream.

**Move clock.** Validate the pinned server's actual per-turn allowance **and battle time bank** on D1–2. Set
p95 per-move at or below a fixed fraction of the per-turn allowance, with a hard abort well inside it; a
global best-so-far abort must return a legal move on **every** code path; a bank-remaining floor below which
search collapses to depth 1. Enforced from D7–9, re-measured the day N1 lands and the day N4 lands. Timing out
is an instant loss and a silent killer of the autonomy claim — and Showdown charges per-turn time against a
bank, so a bot inside its per-move p95 for 40 turns can still lose on the bank.

**Version manifest** on every arena row and the ladder build: bot git sha, poke-engine version + Cargo feature
flags, sim commit, `uv.lock` hash, config, search iteration budget, team seed, battle seed, pair id, pillar
toggle vector, randbats data hash.

**Runtime pillar flags** from the day each pillar lands (`--voi-scale`, `--endgame`, `--inaction-temper`,
`--root-lp`), with the toggle vector recorded in every result row — so D24–25 ablation is a config sweep, not
git archaeology. A 1-hour D1 decision that saves ~2 ED.

**Tests.** *Unit:* posterior never assigns zero mass to a revealed-consistent set; damage-inversion bounds;
the tempered-likelihood cap actually caps; state-tracker Illusion / Imposter / Revival Blessing cases;
observation-class partition sums to 1. *Integration:* Docker sim readiness, full local battle, replay
round-trip, reconnect/rejoin, and **paired-seed determinism — the same seed pair reproduces the same two
teams, green before D4.**

**State-tracker landmine register**, split by failure mode. *Breaks state tracking / damage calc:*
Zoroark-Hisui + Illusion (and Illusion Level Mod); Ditto/Imposter; mid-battle form changes altering stats or
max HP — Palafin/Zero to Hero, Zygarde/Power Construct, Terapagos and Ogerpon, Mimikyu/Disguise,
Eiscue/Ice Face, Aegislash/Stance Change, Morpeko/Hunger Switch; item transfer via Trick/Switcheroo/Knock Off;
Revival Blessing; trapping and forced-switch effects (Magnet Pull, Shed Tail, Wimp Out/Emergency Exit)
altering the legal switch set. *Breaks inference specifically:* Substitute and Focus Sash/Sturdy/Endure/
Multiscale truncating damage (poisons roll inversion); Covert Cloak (learned only when a secondary fails to
proc); Supreme Overlord scaling with fainted allies; Protosynthesis/Quark Drive announcing the boosted stat;
Sleep Clause Mod constraining legal actions. Palafin and Zygarde change max HP mid-battle, silently corrupting
HP-percentage inversion — the exact channel B-lite depends on — and it fails silently.

**Toolchain, kept from the superseded plan:** Python 3.11 + `uv` + committed `uv.lock`; pinned Showdown sim in
Docker; pytest; ruff; a `psbot doctor` preflight; CLI trimmed to `doctor` / `ladder` / `arena` / `report`.
**Cut with the RL lineage:** PyTorch, Stable-Baselines3, Gymnasium, scikit-learn, W&B, Hugging Face, RunPod,
the $50 CUDA budget, the 26-action mask, observation encoding, MLP/GRU, shaped rewards, the opponent
curriculum, PPO sweeps, mypy, pre-commit. CI reduces to one job: `ruff check && pytest -m "not slow"`.

**poke-engine install landmines** — in `doctor` and in the risk table: sdist-only; Rust/Cargo required to pip
install; **generation is a compile-time Cargo feature** (foul-play pins `poke-engine==0.0.48` with
`--features poke-engine/terastallization --no-default-features`); foul-play's Dockerfile pins `pip==24.2`
because `--config-settings` requires it; switching features needs `--force-reinstall --no-cache-dir` because
pip's wheel cache does not key on config-settings. Verify on D1 whether `uv sync` supports per-package
config-settings, else add an explicit `make poke_engine` target plus a doctor assertion on installed version +
features. Mark engine-dependent tests `slow` so default CI stays pure-Python.

**Compute budget.** The arena runs overnight every night from D4 as a background job queue at fixed priority,
and each ablation is submitted the moment its build exists. A 16-vCPU cloud box for three nights would roughly
4× SPRT throughput for well under $50 — the highest-leverage optional spend.

---

## 10. Data contract (log schema v1)

Frozen on D1, validated by a test **before the first ladder game** — a schema revised on D11 makes the D1–D10
logs unusable for the D21 study exactly when the cost is unrecoverable.

**Freeze only the irrecoverable fields on D1:** battle id, format, source (ladder/arena/local), our seat,
winner, full raw protocol stream, `|t:|` timestamps, `|inactive|` payloads, both players' ratings, our own
per-decision latency, sim commit, poke-engine version, randbats data hash, schema version, hashed usernames.

Decision-trace fields that do not exist until D7 (top-k search values, legal action set) and D11 (posterior
entropy, top-3 set hypotheses per opposing mon, toggle vector) are **additive, versioned column adds**,
re-derivable by replaying the stored stream. v1 → v2 is not a rewrite.

**Storage:** raw = append-only gzipped JSONL, one file per battle; a `psbot analysis build` step materialises
partitioned Parquet when D21 needs it (moving polars/pyarrow from a D1 dependency to a D21 one). A
DuckDB-over-JSONL query helper serves the D11–14 log-loss, the D18–20 endgame harvest and the D21 timing study
as one tool rather than three ad-hoc pandas hacks.

**Hygiene, kept verbatim from the superseded plan:** split by complete battle, never by turn; deduplicate by
battle id; strip usernames from processed and published artifacts; **never write unrevealed opponent
information into our own POV record** — the last one is load-bearing here, because it is the only thing
keeping the B-lite posterior-calibration evaluation from being contaminated.

---

## 11. Deferred, non-goals, and post-sprint roadmap

| Item | Why not now | Post-sprint priority |
|---|---|---|
| **Pillar A** — Nash over pruned matrices | Contingent on **finding or reconstructing** a failure analysis (a search task with a null outcome allowed, not an available artifact) and on demonstrated eval calibration. N4 is its beachhead: Nash throughout the tree with a noisy heuristic eval amplifies eval noise; Nash in a terminal exact subgame has no eval noise to amplify | 1 |
| **N2** — information concealment | **Self-play SPRT is the plan's only validation instrument, and in self-play the opponent is maximally belief-responsive because it is us** — N2 is the one pillar this methodology structurally cannot validate, which disqualifies it from a sprint whose thesis is validated novelty per day. Also correct the claim: we can compute the hard-information-consistent *hypothesis set* exactly, but the posterior over it requires an assumed opponent-inference model, and the joint over our unrevealed team additionally requires the generator's role/type correlations, which are not product-form. Free consolation: preferring the less-revealing move among near-equal ones falls out of a mixed root strategy with no μ term | 2 |
| **N6** — rating-band population models | Supervised opponent-policy models fitted per band from this sprint's own logged games (opponent rating logged from day one). This is *opponent modelling*, not learned evaluation, and never selects our own moves — **the only place the retired behavioral-cloning lineage survives** | 3 |
| **N3 in-engine** | Gated on the D21 study finding signal | 4 |
| **Pillar C** — win-condition eval | Multi-week build; a coarse type-coverage term stands in for it | 5 |
| **Pillar D** — Tera as a planned resource | Worthwhile, not novel enough to displace N1/N4 | 6 |
| Full deception (inducing mistakes) | Beyond concealment | 7 |
| Custom Java/Rust engine | The biggest schedule risk in both prior plans | 8 |
| Full joint team-constraint modelling of unrevealed slots | Named non-goal — requires hand-porting `random-teams.ts` into a belief model | — |
| Rating-band-conditioned inaction likelihoods | Explicitly N6, not in-sprint | — |

**Non-goals (sprint):** doubles, other formats, teambuilding formats, learned evaluation, LLM components,
custom engine, GUI beyond CSV + `make report`.

---

## 12. Decision log

| Date | Decision |
|---|---|
| 2026-07-21 | Autonomous laddering confirmed permissible; no self-pairing on ladder |
| 2026-07-21 | RL lineage retired pre-implementation (`b1e789f`); supervised opponent modelling survives only as post-sprint N6, never for our own move selection |
| 2026-07-21 | Headline metric changed from peak ladder rating to validated-pillar count; rating demoted to a reported band with RD |
| 2026-07-21 | Engine: poke-engine (Rust) for search transitions; pinned Node sim for ground truth; custom engine deferred |
| 2026-07-21 | Sprint pillars: M + B-lite/N5, N1, N4 core; N3 offline-only; N7 cut-line substitute; N2 and N6 post-sprint |
| 2026-07-21 | SPRT operating bounds: elo0=0, elo1=20 in-sprint; [0,10] only above a +30 point estimate; [0,35] for screens. Truncation cap 1,500 pairs → INCONCLUSIVE |
| 2026-07-21 | N1's objective is EVSI (decision-relevant), not posterior entropy; no λ sweep |
| 2026-07-21 | Prior-art sweep moved from D21 to D2 |
| *open — before this plan is executed* | Does an external rubric require an ML component? (§0) |
| *open — decide D1* | Chassis: fork foul-play (GPL-3.0, default) vs. greenfield on poke-env |
| *open — decide D1* | Which foul-play commit, search mode, node budget and set-data version constitute the D9 yardstick |
| *open — decide D3, 1–2 h* | Paired-team mechanism: §4f options (a) patch pinned server [default] / (b) BattleStream / (c) custom format |
| *open — before D4* | Public repo from day 1 vs. at first SPRT pass — leaning day 1; an empty pre-registered results table reads as rigour |

---

## 13. Glossary, reading list, runbook

**Glossary.** *SPRT* — sequential probability ratio test; stops as soon as evidence is decisive. *LLR* — log
likelihood ratio, the SPRT's running statistic. *elo0/elo1* — the null and alternative Elo bounds. *Pentanomial
/ trinomial pair statistic* — the score of a seat-swapped game pair; trinomial {0,1,2} when draws are absent.
*VRF* — variance reduction factor from pairing. *GXE* — Showdown's expected win-rate vs. an average player.
*Glicko RD* — rating deviation; a rating with RD > 50 is not yet meaningful. *Determinization / PIMC* —
sampling a full hidden state and searching it as perfect information. *Importance-weighted determinization* —
weighting sampled worlds by posterior mass. *Tempered likelihood* — a likelihood raised to λ < 1 to blunt
overconfident evidence. *EVSI / VOI* — expected value of sample information: the decision-quality gain from an
observation. *Clairvoyance gap* — U_clair − U_ii, how much a determinized search over-values its own position.
*Maximin* — best worst-case action. *Strategy fusion* — the error of picking a different action per sampled
world when you cannot actually tell them apart. *Tera* — Terastallization, a once-per-battle type change.

**Reading list.** PROTOCOL.md; SIM-PROTOCOL.md; `pokemon-showdown-client/WEB-API.md`; pkmn/randbats;
foul-play source; pmariglia's Smogon thread and blog; poke-engine repo + Python bindings; damage-calc; Metamon
and PokéChamp papers; Fishtest pentanomial/paired-game docs and issue #348 (ρ ≈ −0.15); Frank & Basin 1998 and
Long/Sturtevant/Buro/Bowling 2010 (strategy fusion / PIMC); Sinclair et al. LNCS 2023
(doi 10.1007/978-3-031-34017-8_6, information gain inside PIMCTS/ISMCTS); Reconnaissance Blind Chess
(rbc.jhuapl.edu); Brown & Sandholm arXiv 1705.02955 (safe subgame solving); InfoChess arXiv 2604.15373;
salm.dev/writing/winning-pokemon-showdown/ (offline per-matchup payoff matrices via `scipy.linprog` — same
tool, known teams, no live posterior); poker timing-tell literature.

**Writeup template** — `docs/writeups/TEMPLATE.md`, 800–1200 words, fixed sections: claim / prior-art check /
method / SPRT result (bounds, LLR, pairs, seats, node budget) / ablation delta / limitation / repro command.
Four writeups from a blank page in two days will overrun; four against a fixed template will not.

**Start here — day 1, hour 1.**

1. Clone the repo; `uv sync`.
2. `docker compose up` the pinned Showdown sim; record the commit.
3. Register the bot account (bot-identifying name); put credentials in `.env` from `.env.example`.
4. `psbot doctor` — Python, sim reachable, poke-engine version + Cargo features, credentials.
5. Decide the chassis (§5) and record it in §12.
6. Freeze log schema v1 (§10); write its validation test; make it green.
7. First local random-vs-random battle end to end.
8. Autonomy primitives: supervisor, log rotation, measured throttle, pure-fold state rule.
9. Measure the server's per-turn allowance and battle time bank (§9).
10. First ladder game; confirm the log validates against schema v1.
11. 45-minute protocol spike: parse `|t:|` and `|inactive|` on five real replay logs.
12. Start the 4-hour prior-art sweep; commit the five notes.
