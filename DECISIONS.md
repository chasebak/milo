# DECISIONS.md — what was tried, what was measured, what was kept

One entry per upgrade, written as "decided X because measurement Y; alternative Z lost
by N". Nulls and reversals go in too. Every number below is read from
`reports/scorecard.json`, `reports/constraint_ladder.json` or `reports/numbers.json` by
`python tools/render_docs.py`; `tests/test_readme_drift.py` fails if any of it drifts.
All data is synthetic.

<!-- generated:decisions -->
**2026-11-04 · U1 · Lead with the out-of-sample number.** Decided the README headline is out-of-sample profit per restock dollar because the in-sample figure grades the plan on the scenarios it was fitted to: at $1,500 the optimizer's in-sample 0.817 falls to 0.786 on 1,000 fresh scenarios (curse gap 47.02, 3.00%). The in-sample headline lost because it overstates by 0.031 per dollar; the gap is now a KPI of its own.

**2026-11-04 · U1 · The fair baseline re-optimizes the buys.** Decided "today's prices" means `solve_plan` on the ratio-1.00 rows only because that is the strongest policy without price choice: out of sample it earns 1,202.58 at $1,500. The straw man (the optimizer's own buys held at today's prices, 1,190.28) lost by 12.29 (computed before rounding), so it stays a context row only. `today_problem` raises if any product's ratio-1.00 level fails the M1 filter, because that product would then leave this model entirely while P = 0 would force an edit (on the committed data no product is dropped).

**2026-11-04 · U1 · Each policy is measured against its own budget-0 plan.** Decided the denominator base is the same policy's budget-0 plan because repricing the shelf alone is worth 44.24 out of sample and is not something the buys earned. The alternative (one shared today's-prices base) lost because it would credit that repricing to the buys and report 0.815 instead of 0.786.

**2026-11-04 · U1 · Out-of-sample losses ship as printed, with paired errors.** Decided not to assert any out-of-sample ordering because it has no sign guarantee; where the greedy list's per-dollar return beats the optimizer's, `EXPERIMENTS.md` §7 lists it. Every lead now carries the paired standard error over the shared evaluation scenarios: at $1,500 the optimizer leads the greedy list by 60.45 (paired SE 1.30) and by 0.022 per dollar (paired SE 0.001). The pinned seed 11 sits above the 5-seed mean RP (1,558.05) by 10.33, 0.41 seed sds, rank 2 of 5: a mild draw, and the same plan earns 1,521.36 out of sample.

**2026-11-04 · U1 · The marginal dollar uses a $50 step.** Decided a ±$50 finite difference because its solver noise is ±0.0032 per dollar against a signal of 0.652 at $1,500; the average return there (0.817) would overstate the next dollar by 0.165. LP duals were not measured (to measure): a dual of the budget row in a MILP needs a fixed-binary re-solve whose sign conventions this repo has no test for, so the finite difference ships instead.

**2026-11-04 · U2 · Fees are an accounting rung, not a restriction.** Decided to test R0a → R0b with the VSS-style check (value of modeling ≥ 0) instead of "can't raise profit", because removing fees changes how money is counted, not what is feasible. Measured: modeling fees is worth 5.49 in-sample — a near-null, likely because fees are close to proportional — while the fee-blind plan promises 2,068.54 and overstates its own profit by 505.65. Out of sample the fee-blind plan earns 2.35 more than the fee-aware plan (1,523.71 vs 1,521.36, paired SE 0.63): more than two paired SEs, so on the model's own generator modelling fees is a small loss out of sample, not a gain; the null is that the decisions barely move, in either direction. The damage of the textbook model is in the promise, not in the decisions; the null ships.

**2026-11-04 · U2 · Price the rules the model already had.** Decided to price the existing rules as restrictions against a model without them. Measured at $1,500: the 35% set cap costs 74.25, integrality (LP − MILP) 4.22, and the all-or-nothing minimum order 0.00, but the rule binds on 1 of 17 products, which neither plan buys, so its 0.00 is close to vacuous, not a measured null. It stays in the table with that note rather than being dropped, so the rule is visibly untested on this data (pricing it at a budget where that product is bought: to measure).

**2026-11-04 · U3 · Shelf space first, one linear row.** Decided R1 (shelf space came first in the plan because a consignment vendor rents a fixed display) is `Σ v_i (q_i + x_i) ≤ V` with illustrative space per type and a swept V because it adds one row and no variables (solve times, indicative only, are in the table). At 70% of today's plan's shelf it costs 243.18 (15.51% of RP); today's plan uses 378.5 units against that cap of 264.95. One more space unit is worth 3.40, 1.53, 2.14 at 50/70/90% — not monotone, likely because whether a few more units fit one more large box changes the slope (not measured); reported as measured, with the rent-more-shelf rule beside the sweep. Space values live in code (`SPACE_BY_TYPE`), not in a data file, so the committed dataset and its manifest stay frozen.

**2026-11-04 · U4 · Price edits second, on the existing binaries.** Decided R2 uses the AND trick on the price and order disjunct binaries because it adds one binary per product and no big-M: 0.5–9.0 s across the sweep. Today's plan moves 15 stocked products off today's price (recounted independently from the plan, tests). P = 0 costs 355.14; P = 4 alone costs 104.03 and P = 8 40.87. The ladder's cumulative R2 (P = 4 on top of the 70% shelf) costs 66.55 more than R1, against 104.03 alone: the rules overlap, so a rule's price depends on the rungs before it, which is why both tables are printed.

**2026-11-04 · U2-U4 · Tolerance is mip_gap × |bound|, exact checks on small cases.** Decided the committed-problem checks allow 0.0001 × |bound| (about 0.16 at RP) and the exact checks (monotone in V and P, V = ∞ and P = all reproduce the base) run on small random cases at mip_gap = 0, because a fixed 1e-6 match would fail on a HiGHS version change for reasons that are not the model.

**2026-11-04 · Scope · The multi-week cash rung is not built.** Decided to stop at R2 because the plan's Tier 1 ends there; the within-week order (buy → sell → payout) and the open-loop multi-week plan stay in "What to improve next" as to measure.

**2026-11-05 · U1-U4 · The price grid's upper edge binds, and the docs now say so.** At $1,500, 15 of the optimizer's 15 price moves sit at the 1.15× edge, and 18 of 19 ladder solves raise M2. Decided to count that in code (`ladder.grid_edge_moves`, checked against the solver's M2 flag on every ladder record) and print it under the scorecard and on the page, because the lead over today's prices and the P = 0 rung's 355.14 otherwise read as properties of the data when they are largely properties of the 0.80–1.15× grid. Widening the grid (to measure) lost for now because it would change the frozen `numbers.json` and the goldens; the claim is bounded instead.
<!-- /generated:decisions -->

## The Restock Explorer page

Every number below is read from `docs/data/method.json` (written by `python -m experiments.viz build`)
or the reports above.

<!-- generated:page_decisions -->
**2026-11-04 · Page · The slider snaps to 36 solved budgets.** Decided to precompute every policy at $0–$3,000 ($50 steps to $500, then $100 steps) and snap the slider to them, because the page computes nothing and each budget costs two MILP solves (0.1–9.9 s each, 123.8 s for all 72). Interpolating between solved budgets lost because the curve is lumpy: a straight line through two neighbours misses the solved in-sample value by up to 8.80 (at $2,300).

**2026-11-04 · Page · Every budget stores its own buy lists.** Decided to store each policy's plan and buy list per solved budget because the optimal plans are not nested: in 13 of 35 neighbouring steps some product's units fall as the budget rises, so a list of increments would be wrong there.

**2026-11-04 · Page · Quantiles stay in Python.** Decided the p10/p50/p90 band is computed by the model's weighted quantile and copied, because a linear-interpolated percentile (what a browser library would compute) differs by 1.40 at the p10 for the optimizer at $1,500 out-of-sample; the page would then disagree with the README.

**2026-11-04 · Page · Losses ship on the page too.** Decided the curve note lists every solved budget where the greedy list beats the optimizer out of sample — per restock dollar at $50, $100, $200, $250, $300, $350; in total profit at no solved budget — because out-of-sample has no sign guarantee, and since the optimizer still earns more in total at every budget, those per-dollar wins come from greedy's lower budget-0 base and spend, not from more profit. The grid also moved the story: the optimizer's widest lead over the greedy list is 7.10% in-sample at $2,700 (8.09% out of sample at $2,700), not the 5.26% at $250 that the five-budget table shows, so the page quotes the grid and the README keeps its five-budget table labelled as five budgets.

**2026-11-04 · Page · The page names the pinned problem, not the demo's.** Decided the page states its settings (no set-spend cap, 24 training scenarios, seed 11) because the README headline is computed there; the demo's 35% set cap and operating gap give 1,497.63 at $1,500 against RP 1,568.38, and an unlabelled mix would look like drift. CI never re-solves the grid: `python -m experiments.viz --check` re-runs the deterministic greedy list and re-values the 108 stored plans instead of paying the 123.8 s of solves (the tests still solve a few small MILPs and one fresh budget as a smoke test).
<!-- /generated:page_decisions -->
