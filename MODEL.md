# MODEL.md — formulation, estimation, and the flag vocabulary

This file defines the model the code implements and the P/M/D vocabulary the docstrings
use. Results live in `EXPERIMENTS.md`; every number there comes from
`reports/numbers.json`.

## 1. The optimization problem

The planner solves a **two-stage stochastic MILP in extensive form with fixed,
effectively trivial recourse — all decisions are here-and-now; the second stage prices
each scenario (sales capped at demand, leftovers salvaged), no recourse decisions**.

Stage 1 picks, per product `i`:

- integer buy units `x_i`, with an all-or-nothing minimum order — a disjunction
  `(min_order_i ≤ x_i ≤ max_units_i) ∨ (x_i = 0)`;
- exactly one price level `k` from a 0.80×–1.15× grid around today's price (5% steps),
  as an XOR disjunction; levels above 1.6× the market price are dropped (M13).

Stage 2, per scenario ω, is pure accounting: sales `s_iω = min(D_ikω, on_hand_i + x_i)`,
leftovers `L_iω = on_hand_i + x_i − s_iω` salvaged at `salvage_frac_i × cost_i`.

Objective (maximize):

```
E_ω[ Σ_i  n_ik · s_iω  +  ρ_i c_i · L_iω ]  −  Σ_i c_i · x_i
```

where `n_ik` is the net cash per unit at level `k`:
`price · (1 − commission − card_fee_rate · card_share) − order_fee / units_per_order`
(`milo/fees.py`; commission is a required per-store config value, the card fee 2.9% and
flat order fee $0.30 default to standard published card-processing rates). Ordinary
constraints: the restock budget, an optional per-set spend cap, and per-product
`max_units`. A price level `k` is kept only when `n_ik` beats the salvage value (M1):
a model that "sells" below salvage would refuse to sell in reality.

Both disjunctions are written with `pyomo.gdp` and turned into a MILP by `gdp.hull`
(default) or `gdp.bigm`, then solved with HiGHS. The hull (convex-hull) reformulation
of a disjunction gives the tightest possible continuous relaxation of each disjunct,
at the cost of more variables; big-M is smaller but its relaxation is only as good as
its M constants (Grossmann & Trespalacios 2013). On this model the difference is not
cosmetic — see the measured table in `EXPERIMENTS.md` §3.

## 2. Demand estimation (P1–P3)

The demand unit is a *family*: one seller's interchangeable listings of one product
(e.g. the same pack across sets lives in one listing family). Exposure counts only
**in-stock** open days (D1): a sold-out day says nothing about demand.

- **P1 — raw rate**: units per in-stock week per family.
- **P2 — shrunk rate**: a gamma-Poisson posterior per family with a prior per product
  type — **method-of-moments empirical Bayes (Robbins 1956; Morris 1983) with a
  variance floor so the prior never collapses**. Posterior:
  `Gamma(a_type + units, b_type + exposure_weeks)`; the posterior mean is the shrunk
  weekly rate and the posterior quantiles give per-family p10/p50/p90. Thin families
  (≤2 sales, or under 2.5 in-stock weeks) carry an explicit flag.
- **P3 — time effects**: store-traffic trend × outside-market demand index × holiday
  uplift. The holiday uplift is a scenario input, not estimated from the data (D8).
  P3 is validated by a backtest (fit before a split date, predict after it), run with
  and without the traffic step.

## 3. Price response (P4–P7): the elasticity evidence ladder

This is the strongest statistics in the repo, and it is deliberately defensive: every
rung can be rejected, and a rejected rung falls to the next one.

- **P4 — relative price**: `m_it = our price / market price` per unit-day, with the
  market price from the daily stitched feed table (M17), condition-matched (M18) and
  smoothed over 7 days so one odd sale cannot move `m`.
- **P5 — sensitivity, from evidence in priority order**:
  1. *Evidence 0b — fixed-effects Poisson panel*: `log E[units_it] = unit FE + week FE
     + day-of-week FE + e_below·log(m)⁻ + e_above·log(m)⁺ + β·log(market units)`, fit
     by a hand-rolled IRLS (`milo/pricing.py: poisson_irls`; no statsmodels), on
     in-stock unit-days of units whose own price actually moved. Uncertainty comes
     from a **unit-level cluster bootstrap** (resample whole units with replacement);
     the p10/p90 of the bootstrap draws become the high/low sensitivity scenarios.
     *Rejection guard (M12/M20)*: the fit is discarded when the base slope is ≥ 0 or
     the bootstrap interval crosses 0 — a sign-violating or insignificant fit never
     reaches the optimizer.
  2. *Evidence 1 — Huber cross-section of markups*: `log(units per in-stock week) ~
     type dummies + set popularity + slope·log(m)` across both sellers' items, with a
     Huber loss so one item priced far over market cannot dominate (M13). Same guards:
     wrong sign, interval crossing 0, or too few items (< 25) rejects the fit.
  3. *Fallback — assumed ranges* per type group (sealed/single), below/above market.
- **Envelope clipping**: whatever rung wins is clipped to a wide envelope around the
  assumed range, so a noisy accepted fit still cannot produce absurd elasticities (a
  binding clip is recorded in the result's `note` column); the
  above-market slope is forced to be at least as steep as the below-market slope (P6's
  kink direction).
- **P6 — kinked demand curve**: `h(m) = m^e_below` for `m ≤ 1`, `m^e_above` above —
  demand falls faster when priced over market than it rises when priced under it
  (price-setting newsvendor framing: Petruzzi & Dada 1999).
- **P7 — ordered demand scenarios**: see §4.

## 4. Scenario generation

Demand scenarios are **Monte Carlo draws from the gamma-Poisson posterior predictive
with a comonotone coupling (one shared uniform per unit-scenario through the Poisson
inverse CDF at every price level — simultaneously the demand-monotonicity mechanism
and common random numbers across price levels and compared plans), crossed with a
deterministic three-point elasticity grid (equal weights)**.

Concretely (`pipeline.build_scenarios`): per unit and scenario, a gamma rate draw from
the P2 posterior is scaled by the P3 multipliers and the horizon. The lead-time haircut
(M9) is applied **per product, not per unit**: products already in stock sell over the
whole horizon, including newly bought units; only out-of-stock products wait out the
lead time. *Design decision*: this per-product rule is an approximation — it over-credits
newly bought units on in-stock products by up to `lead_weeks / horizon_weeks` of their
demand — accepted so each product needs only one demand window. The scenario's
elasticity column (low/base/high, cycling every 3 scenarios) sets the P6 multiplier per
price level; one uniform per (unit, scenario) is pushed through the Poisson inverse CDF
at every level, so `D_ikω` is non-increasing in `k` by construction (M11) — the
optimizer validates this property on input and refuses violations. In the docs this
method is called a *sampled-scenario approximation*; the SAA label is reserved for the
convergence experiment that validates it (`EXPERIMENTS.md` §4).

## 5. EVPI and VSS (maximization form)

Over a fixed scenario set with probabilities `p_ω`:

- `RP = max_x E_ω[profit(x, ω)]` — the here-and-now (recourse-problem) optimum;
- `WS = E_ω[max_x profit(x, ω)]` — wait-and-see: one solve per scenario;
- `x̄ = argmax_x profit(x, ω̄)` on the probability-weighted mean scenario `ω̄`, and
  `EEV = E_ω[profit(x̄, ω)]` — the mean-value plan re-valued on the full set.

For maximization, `EEV ≤ RP ≤ WS`, so

- `EVPI = WS − RP ≥ 0` — what perfect foresight would be worth;
- `VSS = RP − EEV ≥ 0` — what solving the stochastic program (instead of planning on
  the average scenario) is worth.

Both are reported **in-sample, conditional on the scenario set** (Birge & Louveaux
2011). All experiment solves run at rel gap 1e-4, not the operating 0.005, so MIP gap
noise can bend the inequalities by at most that gap; `tests/test_evpi_vss.py` asserts
them with a slack of exactly `mip_gap × max(1, |RP|)`.

## 6. Vocabulary: P-steps, M-flags, D-codes

The original build plan numbered its estimation steps (P), modeling rules (M) and data
edge cases (D); docstrings and the solver's `flags` table keep those codes.

**P-steps** — the estimation pipeline: P1 raw rate, P2 shrunk rate, P3 time effects,
P4 relative price, P5 price sensitivity, P6 kinked demand curve, P7 ordered demand
scenarios (all defined above).

**M-flags emitted on `solve_plan(...)["flags"]`** (never silent):

| code | meaning |
| --- | --- |
| `M1` | price level dropped: net cash ≤ salvage value (or all levels dropped → product excluded) |
| `M2` / `M3` | optimum sits at the top / bottom edge of the price grid — the grid, not the data, may be binding |
| `M4` | nothing affordable: budget below the cheapest minimum order |
| `M5` | lumpiness gap: LP relaxation beats the MILP by more than rounding — integrality is costing real money |
| `M10` | HiGHS stopped without reaching the requested rel gap (`mip_gap`; any termination other than convergence, e.g. the time limit), so the plan is not certified optimal within that gap; termination and achieved gap attached |
| `MIN_ORDER` | `max_units < min_order`, so `x_i` is fixed to 0 |

**M-rules referenced in docstrings** (behavior, not emitted flags): M9 lead time (an
out-of-stock product's demand window waits out `lead_weeks`; an in-stock product's does
not — see §4); M11 demand monotone in price via the shared
uniform; M12/M20 elasticity rejection guards; M13 the 1.6× market price cap and the
Huber robustness choice; M17 market-feed stitching, finest granularity first (M17b
drop the still-running bucket, M17c de-duplicate sales, M17d price-outlier filter);
M18 condition matching between store and feed; M22 feed-mapping review flags.

**D-codes** — data edge cases. The two that survive in this repo: D1 (sold-out days
carry no demand information and are excluded from exposure) and D8 (the holiday uplift
is a scenario input, never estimated from the committed window).

## References

1. J. R. Birge and F. Louveaux, *Introduction to Stochastic Programming*, 2nd ed.,
   Springer, 2011 — two-stage recourse, EVPI, VSS.
2. N. C. Petruzzi and M. Dada, "Pricing and the newsvendor problem: a review with
   extensions," *Operations Research* 47(2), 1999.
3. H. Robbins, "An empirical Bayes approach to statistics," *Proc. Third Berkeley
   Symposium*, 1956.
4. C. N. Morris, "Parametric empirical Bayes inference: theory and applications,"
   *JASA* 78(381), 1983.
5. I. E. Grossmann and F. Trespalacios, "Systematic modeling of discrete-continuous
   optimization models through generalized disjunctive programming," *AIChE Journal*
   59(9), 2013 — hull vs big-M relaxation tightness.
6. J. E. Smith and R. L. Winkler, "The optimizer's curse: skepticism and postdecision
   surprise in decision analysis," *Management Science* 52(3), 2006 (working paper
   1996) — cited next to the SAA convergence plot in `EXPERIMENTS.md`.
