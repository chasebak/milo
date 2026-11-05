"""The one copy of every "why this metric" box and KPI definition.

The README tables, EXPERIMENTS.md and the page all read this dict (it is also written
into reports/scorecard.json and reports/constraint_ladder.json), so the explanation
cannot drift between them. Settings quoted in the text (scenario counts, seeds, the
time limit, the sweep points) cannot be formatted from experiments/ here, because the
installed `milo` package must not import it; tests/test_why.py asserts each one equals
its constant instead, so changing a constant fails CI until the text follows.
Each entry answers three things:

  question  the owner's question the number answers;
  kind      what kind of number it is (counted / modelled / simulated) and where its
            truth comes from;
  lie       what would make it lie, and which test guards against that.
"""

WHY = {
    "rord": {
        "question": "What does each restock dollar return, on demand the plan never saw?",
        "kind": ("Simulated: each policy's plan is valued on 1,000 fresh scenarios (eval seed 999) "
                 "from the model's own demand generator, minus the same policy's budget-0 plan, "
                 "divided by what it spent."),
        "lie": ("It flatters the optimizer if the evaluation set leaks into training "
                "(test: eval seed 999 is never a training seed) and it is only as true as the "
                "generator, so it is out-of-sample on the model's own generator, not on a real store."),
    },
    "fair_baseline": {
        "question": "Is the lift just repricing, or does planning the buys matter too?",
        "kind": ("Modelled: today's prices held fixed (ratio 1.00) while the buys are re-optimized "
                 "by the same MILP, so the only thing it lacks is price choice."),
        "lie": ("A same-buys-at-today's-prices row would be a straw man; this row re-optimizes the "
                "buys (test: it equals the price-edit limit P = 0 rung within the gap)."),
    },
    "curse_gap": {
        "question": "How much does the in-sample number flatter itself?",
        "kind": ("Simulated: in-sample objective on the 24 training scenarios minus the same plan's "
                 "value on the 1,000 evaluation scenarios."),
        "lie": ("Out-of-sample has no sign guarantee, so the gap is reported, never asserted; "
                "with 24 scenarios seed noise is large (EXPERIMENTS.md §5)."),
    },
    "marginal_dollar": {
        "question": "Should I put in more cash?",
        "kind": ("Modelled: (obj(B + 50) - obj(B - 50)) / 100, from two re-solves; the average "
                 "return hides the last dollar."),
        "lie": ("Each solve may stop up to mip_gap x |bound| from the optimum, so the slope carries "
                "that much noise; the tolerance is printed beside it and budget monotonicity is tested."),
    },
    "fee_blind": {
        "question": "What does ignoring the store's cut overstate?",
        "kind": ("Modelled: a plan solved on list prices (no commission, card or order fees), then "
                 "valued at the true fees."),
        "lie": ("Its own objective must be >= its value at true fees, in total (test); if fees were "
                "close to proportional the plan would barely change, and that null ships as printed."),
    },
    "price_of_rule": {
        "question": "Which business rule costs the most, so which one to relax first?",
        "kind": ("Modelled, in-sample: obj(looser rung) - obj(this rung) for a restriction; for an "
                 "accounting rung, obj(new plan) - value(old plan under the new accounting)."),
        "lie": ("Only true restrictions get the can't-raise-profit test (tolerance mip_gap x |bound|); "
                "an accounting rung gets the VSS-style test instead, and every rung records whether "
                "it solved to the gap."),
    },
    "shelf": {
        "question": "If I rent more shelf, what is one more unit of space worth?",
        "kind": ("Modelled, in-sample: shelf capacity swept at 50/70/90/100% of today's plan's use, with "
                 "illustrative space per product type; the slope is a re-solve with 5 more units."),
        "lie": ("Space values are made up and swept, not measured; shelf use is recounted independently from "
                "the plan (test) and must be <= capacity, and on-hand stock over capacity raises a SHELF flag."),
    },
    "price_edits": {
        "question": "How many relabels a week are worth the staff time?",
        "kind": ("Modelled, in-sample: profit with at most P stocked products moved off today's "
                 "price, P = 0, 1, 2, 4, 8, all."),
        "lie": ("Edits are recounted independently from the plan (stocked and ratio != 1.00) and must be <= P; an "
                "unstocked product never counts, and P = 0 must leave every stocked product at 1.00."),
    },
    "budget_curve": {
        "question": "How much profit does each budget buy, and where does the optimizer's lead open or close?",
        "kind": ("Simulated: each policy's plan at every solved budget, valued on the 1,000 evaluation "
                 "scenarios; the dashed line is the optimizer's in-sample value and the band its "
                 "out-of-sample p10 to p90."),
        "lie": ("A budget between two solved points was never solved, so the slider snaps instead of "
                "interpolating; the in-sample line never falls as the budget grows (test, within "
                "mip_gap x |bound|), but out-of-sample has no such guarantee and is shown as computed."),
    },
    "buy_list": {
        "question": "What exactly should the shop buy at this budget, and which prices move?",
        "kind": ("Modelled: the stored plan at this budget; spend = units x unit cost, and the price is "
                 "the chosen level's shelf price against today's."),
        "lie": ("Spend must stay within the budget and every listed price is a level the model kept "
                "(tests); costs and demand are synthetic, so this demonstrates the method, it is not a "
                "shopping list for a real store."),
    },
    "evpi_vss": {
        "question": "Is planning for uncertainty worth it, and how much could perfect forecasts add?",
        "kind": ("Modelled, in-sample: EEV (the average-scenario plan re-valued on the scenarios) <= RP "
                 "(the stochastic plan) <= WS (perfect foresight per scenario); VSS = RP - EEV and "
                 "EVPI = WS - RP."),
        "lie": ("Both are in-sample on the pinned 24 scenarios, not out-of-sample guarantees; a solve "
                "stopped short of its gap could break the order, so a test asserts EEV <= RP <= WS "
                "within the gap."),
    },
    "saa": {
        "question": "Are 24 training scenarios enough?",
        "kind": ("Simulated: for N = 20 to 160 training scenarios, 10 seeds each, the plan's in-sample "
                 "objective against its value on the 1,000 evaluation scenarios."),
        "lie": ("The gap shrinks with N on average but not at every step, and one seed can sit above or "
                "below the seed mean (EXPERIMENTS.md §5 prints the spread); the evaluation seed is never a "
                "training seed (test)."),
    },
    "hull_bigm": {
        "question": "Does the weekly tool solve fast enough to trust its answer?",
        "kind": ("Counted: wall-clock seconds and termination status per solve, on identical instances "
                 "with the same rel gap and a 120 s limit."),
        "lie": ("Timings depend on the machine and the HiGHS version (one recorded run); the result "
                "column records whether each solve reached the gap, so a timeout is never shown as optimal."),
    },
}

# KPI definitions for the README KPI card (Shared standards (a)). "measured_on" uses
# one of the four fixed labels; "value now" is filled from the generated files.
KPIS = [
    {"id": "rord_out", "name": "Profit per restock dollar, out-of-sample (headline)",
     "formula": ("(E[profit at B] - E[profit of the same policy's budget-0 plan]) / spend at B, "
                 "on 1,000 fresh scenarios (eval seed 999); n/a at B = 0"),
     "why": "What each dollar tied up in stock returns; fresh scenarios stop the plan grading itself",
     "against": "the careful greedy list; today's prices with the buys re-optimized",
     "does": "raises it", "measured_on": "out-of-sample on the model's own generator", "why_key": "rord"},
    {"id": "curse_gap", "name": "Optimizer's-curse gap",
     "formula": "in-sample objective - out-of-sample value of the same plan",
     "why": "How much the in-sample headline flatters itself",
     "against": "the SAA table (EXPERIMENTS.md §4)", "does": "reported, not hidden",
     "measured_on": "out-of-sample on the model's own generator", "why_key": "curse_gap"},
    {"id": "marginal_dollar", "name": "Marginal value of a budget dollar",
     "formula": "(obj(B + 50) - obj(B - 50)) / 100",
     "why": "\"Should I put in more cash?\" The average hides the last dollar",
     "against": "the average return at B (in-sample profit per restock dollar)",
     "does": "shows where to stop adding budget",
     "measured_on": "synthetic, in-sample (modelled on the 24 training scenarios, not truth)",
     "why_key": "marginal_dollar"},
    {"id": "price_of_rule", "name": "Price of each business rule",
     "formula": "obj(previous rung) - obj(this rung), in $ and % of RP",
     "why": "Which rule to relax first", "against": "the rung before",
     "does": "measured per rung",
     "measured_on": "synthetic, in-sample (modelled on the 24 training scenarios, not truth)",
     "why_key": "price_of_rule"},
    {"id": "solve_time", "name": "Solve time within a 0.01% gap",
     "formula": "wall-clock seconds and termination status per solve (rel gap 1e-4, 120 s limit)",
     "why": "A weekly tool must solve fast enough to trust its answer",
     "against": "big-M on identical instances", "does": "hull reaches the gap; big-M times out",
     "measured_on": "synthetic truth (counted on one machine)", "why_key": "hull_bigm"},
]
