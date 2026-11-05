# EXPERIMENTS.md — the five experiments, their committed results, and how to rerun them

Single source of truth: `reports/numbers.json`, written by

```bash
python -m experiments.build_numbers       # reruns experiments 1, 2, 3 and 5 (slow by design: each big-M solve runs out its full 120 s limit)
```

on the committed synthetic dataset (`data/synthetic/`, knobs in `SCHEMA.md`). Every
table below is rendered from that file by `python tools/build_figures.py` and pasted
verbatim; no number here is typed by hand. The SAA experiment (§4) is heavier and has
its own committed output; `build_numbers` only summarizes it.

**Pinned settings** (`experiments/common.py`, mirrored in `numbers.json →
experiment_settings`): 24 scenarios (seed 11), main budget $1,500, no set-spend cap,
rel gap 1e-4 (the operating 0.005 would let gap noise violate the EVPI/VSS
inequalities), 120 s per-solve limit. Every solve's timing is appended to the
untracked `out/solve_log.csv`. **All results are in-sample, conditional on the pinned
scenario set** — §4 is the experiment that measures what that caveat costs.

## 1. EVPI / VSS

Rerun: `python -m experiments.evpi_vss`

RP is the two-stage optimum re-valued through the same `evaluate_plan` accounting as
every other plan (the module checks solver objective and re-valuation agree). EEV holds
the mean-scenario plan's buy units *and* price levels fixed over the full set. WS
solves each of the 24 scenarios (all of them, so it is not a subset approximation) at
the same 1e-4 rel gap setting as every experiment solve. `evpi_vss` raises if any WS
solve stops before meeting that gap; the check is newer than the committed
`numbers.json`, which records no per-scenario WS status, so for the committed run it is
unverified per scenario, and `tests/test_evpi_vss.py` re-runs the check on every CI
run. `EEV ≤ RP ≤ WS` holds up to that gap: `tests/test_evpi_vss.py` asserts each
inequality with a slack of exactly `mip_gap × max(1, |RP|)` and no more (definitions
and inequalities: `MODEL.md` §5).

| quantity (expected profit, $) | value |
| --- | --- |
| RP (two-stage stochastic optimum) | 1,568.38 |
| EEV (mean-value plan, re-valued on the scenarios) | 1,488.31 |
| WS (wait-and-see bound) | 1,673.45 |
| EVPI = WS − RP | 105.07 (6.70% of RP) |
| VSS = RP − EEV | 80.07 (5.11% of RP) |

Reading: perfect demand foresight would be worth 6.70% of RP, and actually solving the
stochastic program instead of planning on the average scenario earns 5.11% of RP. The
mean-value plan fails in the usual way — the average scenario washes out the downside
tail, so it overbuys exactly the units the bad scenarios cannot sell (its own solver
objective, 1,669.23 on the mean scenario, is an overestimate the re-valuation corrects
to 1,488.31).

## 2. Greedy baseline vs the exact MILP

Rerun: `python -m experiments.greedy_baseline`

The greedy picks each product's best standalone price level, then buys in marginal
expected-profit-per-dollar order (min-orders enter as all-or-nothing blocks). The MILP
optimizes units and prices jointly. Same scenarios, same fee model, same level filter,
same accounting — the gap isolates the value of joint optimization. No approximation
guarantee is claimed; see "What this does NOT claim" in the README.

| budget ($) | exact MILP ($) | greedy ($) | gap ($) | gap (% of MILP) |
| --- | --- | --- | --- | --- |
| 250.00 | 591.84 | 560.72 | 31.12 | 5.26% |
| 500.00 | 801.95 | 767.20 | 34.76 | 4.33% |
| 1,000.00 | 1,198.88 | 1,164.54 | 34.34 | 2.86% |
| 1,500.00 | 1,568.38 | 1,503.52 | 64.85 | 4.14% |
| 2,500.00 | 2,054.08 | 1,949.37 | 104.71 | 5.10% |

Measured gap range: 2.86%–5.26%. The committed figure `docs/figs/hero.png` (the README
hero) plots this table.

## 3. GDP transform: hull vs big-M

Rerun: `python -m experiments.hull_vs_bigm`

Identical instances, identical solver, rel gap 1e-4, 120 s limit. The two MILPs
describe the same feasible set, so a big-M incumbent can trail but never beat the
hull optimum by more than the gap allows (asserted in code with a slack of
2 × 1e-4 × |hull objective|). On this model, big-M's LP relaxation is
5.1x–13.2x looser than hull's, and big-M hits the time limit at every budget while
hull reaches a plan that is optimal within a 0.01% gap in seconds:

| budget ($) | transform | incumbent ($) | LP bound ($) | build (s) | solve (s) | result |
| --- | --- | --- | --- | --- | --- | --- |
| 500.00 | hull | 801.95 | 804.18 | 1.04 | 2.1 | optimal within a 0.01% gap |
| 500.00 | bigm | 750.30 | 10,642.59 | 0.17 | 120.6 | time limit (120 s) |
| 1,500.00 | hull | 1,568.38 | 1,572.60 | 1.31 | 3.8 | optimal within a 0.01% gap |
| 1,500.00 | bigm | 1,372.83 | 10,745.40 | 0.18 | 120.3 | time limit (120 s) |
| 2,500.00 | hull | 2,054.08 | 2,089.31 | 1.05 | 4.0 | optimal within a 0.01% gap |
| 2,500.00 | bigm | 1,896.07 | 10,745.40 | 0.17 | 120.7 | time limit (120 s) |

Hull pays ~6x more build time (it adds disaggregated variables per disjunct) and earns
it back ~30–60x over in solve time — against a big-M that still had not reached
the 0.01% gap when stopped. This measured difference is why
`transform="hull"` is the default (`MODEL.md` §1; Grossmann & Trespalacios 2013).

## 4. SAA convergence (the optimizer's curse, measured)

Rerun: `python -m experiments.saa_convergence` — the heavy run (40 MILPs: 4 scenario
counts × 10 seeds, plus 1,000-scenario evaluations); it rewrites the committed
`reports/saa_convergence.csv` and `reports/saa_convergence.png`. Never run in CI.

For each N ∈ {20, 40, 80, 160} and 10 fresh seeds: solve on N training scenarios, then
value the chosen plan on 1,000 evaluation scenarios no training draw used. The
in-sample objective systematically overstates the plan's real value — the optimizer's
curse (Smith & Winkler) — and the gap shrinks as N grows, though not monotonically
(N=80's gap, 9.98, sits above N=40's 7.44 — seed noise at 10 seeds per N), which is
the validation that earns the "SAA" label used in this section:

![SAA convergence: in-sample vs out-of-sample value by training scenario count](reports/saa_convergence.png)

| N (training scenarios) | in-sample mean ($) | in-sample std | out-of-sample mean ($) | out-of-sample std | curse gap ($) | solve (s, mean) |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 1,540.91 | 15.15 | 1,525.14 | 3.70 | 15.77 | 3.1 |
| 40 | 1,537.38 | 8.31 | 1,529.94 | 2.70 | 7.44 | 5.4 |
| 80 | 1,543.52 | 9.15 | 1,533.53 | 1.54 | 9.98 | 10.5 |
| 160 | 1,537.42 | 4.15 | 1,534.30 | 1.19 | 3.12 | 24.0 |

Reading: from N=20 to N=160 the out-of-sample std shrinks 3.70 → 1.19 and the curse
gap shrinks 15.77 → 3.12 — non-monotonically on the way (7.44 at N=40, 9.98 at N=80) —
while mean solve time grows 3.1 s → 24.0 s. The pinned
24-scenario experiment set trades some of that stability for speed; §5 measures what
that trade costs.

## 5. Scenario-set seed spread

Rerun: part of `python -m experiments.build_numbers` (`rp_seed_spread` in
`numbers.json`).

RP re-solved on 5 scenario sets that differ only in the generator seed, at the pinned
N=24:

RP across scenario-set seeds [11, 12, 13, 14, 15]: 1,568.38, 1,523.35, 1,562.07, 1,546.05, 1,590.38
mean 1,558.05, std 25.08, spread 4.30% of the mean

So a single headline RP at N=24 carries real seed noise (max−min spread: 4.30% of the
mean) — which is why the docs always name the pinned seed and label results in-sample.

## 6. End-to-end demo numbers

Rerun: `python -m milo.demo` (same pinned settings as the e2e golden test:
24 scenarios, budget $1,500, 8 bootstrap draws, 12-scenario budget curve).

Strategy comparison on the optimal plan's buy list — fixed prices vs the optimizer's
jointly chosen prices, identical scenarios and accounting:

| pricing strategy (same buy list) | expected ($) | p10 ($) | p90 ($) | sell-through | leftover value ($) |
| --- | --- | --- | --- | --- | --- |
| keep | 1,162.81 | 1,122.93 | 1,183.72 | 0.993 | 15.90 |
| match_market | 748.03 | 746.53 | 750.21 | 0.998 | 1.46 |
| market_minus_5 | 704.02 | 702.68 | 706.01 | 0.999 | 1.37 |
| market_plus_10 | 932.04 | 910.25 | 937.11 | 0.997 | 3.53 |
| optimal (Pyomo) | 1,492.34 | 1,382.82 | 1,584.73 | 0.966 | 70.83 |

| budget ($) | expected profit ($) | spend ($) |
| --- | --- | --- |
| 0.00 | 345.79 | 0.00 |
| 500.00 | 812.27 | 499.57 |
| 1,500.00 | 1,497.63 | 1,497.18 |

Note the shape of the optimal row: it accepts a *lower* sell-through (0.966 vs 0.993)
and more leftover value, because clearing every unit is not the objective — expected
profit is. (The budget-curve row at $1,500 differs from the strategy-table optimum
because the curve re-solves on its 12-scenario subset.)
