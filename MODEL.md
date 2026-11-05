# MODEL.md — formulation, estimation, and the flag vocabulary

This file defines the model the code implements and the P/M/D vocabulary the docstrings
use. Results live in `EXPERIMENTS.md`; every number there comes from
`reports/numbers.json`, `reports/scorecard.json` or `reports/constraint_ladder.json`.

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

## 1b. Business rules: the constraint ladder (`milo/ladder.py`)

`solve_with_rules` builds exactly `solve_plan`'s GDP model and attaches the enabled
rules **before** the hull (or big-M) transform, so they can use the disjunct binaries;
with every rule off the model is identical (same variable and constraint counts,
tested). Each rule has a *kind*, and the kind decides its test:

- a **restriction** shrinks the feasible set, so it can only lower the optimum; its
  price is `obj(looser) − obj(restricted) ≥ 0`, checked within `mip_gap × |bound|`;
- an **accounting** rung changes how money is counted with the same feasible set; its
  value is `obj(new plan, new accounting) − value(old plan, new accounting) ≥ 0`, the
  VSS logic. Fees in the objective (R0b against the fee-blind R0a) is this kind.

**R1 shelf space (restriction).** One linear row, no new variables:
`Σ_i v_i (q_i + x_i) ≤ V`, with `v_i` the space per unit of the product's type (§7).
If on-hand stock alone exceeds `V`, the row would be infeasible; a `SHELF` flag fires
and only space-free buys remain allowed — never a silent infeasible solve.

**R2 weekly price-edit limit (restriction).** In a POS-only store each price change is
a relabel plus a CSV upload. With `y_i,now` the binary of product i's ratio-1.00 price
disjunct (0 if M1 dropped that level) and `o_i` the binary of its order-yes disjunct,
`e_i ∈ {0,1}`:

```
stocked now (q_i > 0):   e_i ≥ 1 − y_i,now
not stocked now:         e_i ≥ (1 − y_i,now) + o_i − 1      (the AND trick: bought AND moved)
never orderable, q_i=0:  no e_i (it can never be on the shelf)
Σ_i e_i ≤ P
```

An unstocked, unbought product never counts; a stocked product whose 1.00 level was
dropped must move, so it is a forced edit — if forced edits exceed `P`, the limit is
raised to that count and an `EDITS` flag says so. The fair "today's prices" baseline of
the scorecard is the same problem as `P = 0` (tested).

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
| `SHELF` | (ladder R1) on-hand stock alone exceeds the shelf capacity; only space-free buys remain allowed |
| `EDITS` | (ladder R2) more stocked products lost their 1.00 price level (M1) than the edit limit allows; the limit is raised to that count |

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

## 7. Assumptions

Every knob the scorecard and the ladder depend on, read from the code that uses it
(rendered by `python tools/render_docs.py`). Source classes: measured / published
default / illustrative / design choice.

<!-- generated:assumptions -->
| knob | value | source | what changes if wrong | flag that fires when it binds |
| --- | --- | --- | --- | --- |
| store commission | 10% | illustrative | every net-cash value moves; the fee-blind row shows how much fees matter | none (a required config value) |
| card fee, per-order fee | 2.9%, $0.30 | published default | small: fees are close to proportional | none |
| price grid | 0.80x to 1.15x today's price | design choice | an optimum at the edge may sit outside the grid | M2 / M3 |
| shelf space per unit (pack-equivalents) | booster_box 12, booster_bundle 4, booster_pack 1, collection 6, etb 8, single 0.5, tin 4 | illustrative | R1's price and the units-by-type shift; swept capacity limits the damage | SHELF when on-hand alone exceeds V |
| shelf capacity V | 50%, 70%, 90%, 100% of today's plan's use | design choice (swept) | the price of R1 is only read at these points | SHELF |
| weekly price-edit limit P | 0, 1, 2, 4, 8, all; the ladder uses P = 4 | design choice (swept) | R2's price at the chosen P | EDITS when forced edits exceed P |
| training scenarios | 24 (seed 11) | design choice (pinned) | in-sample numbers flatter the plan; the curse gap measures by how much | none |
| evaluation scenarios | 1,000 (seed 999) | design choice | out-of-sample values carry sampling noise of their own; the paired SE of each lead is printed beside it | none |
| experiment rel gap | 0.0001 | design choice | every comparison carries mip_gap × \|bound\| of slack | M10 when a solve stops short of it |
| marginal-dollar step | $50 | design choice | a smaller step drowns in gap noise, a larger one averages over a kink | none |
| set-spend cap (operating default) | 35% | design choice | priced as an existing rule; the experiments run without it | none |
<!-- /generated:assumptions -->

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
