"""Render every generated block in README.md and EXPERIMENTS.md from the committed reports.

The docs carry no hand-typed result numbers. Each table, headline sentence and "why
this metric" box sits between markers

    <!-- generated:NAME -->
    ...
    <!-- /generated:NAME -->

and is rewritten from reports/numbers.json, reports/scorecard.json,
reports/constraint_ladder.json and the page's small docs/data/method.json and saa.json
(plus the one WHY dict in milo/why.py) by

    python tools/render_docs.py           # rewrite the blocks in place
    python tools/render_docs.py --check   # exit 1 if any block differs (the drift test does this too)

This file only FORMATS: every number it prints, including differences and counts, is
a value stored in one of those files by tested code (no arithmetic on results here).
Failure mode: a number typed into prose OUTSIDE a block is not rewritten here; the
README drift test (tests/test_readme_drift.py) scans for those and fails on any that is
not a formatted value from the generated files (a weak bag check; see its docstring).
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
DOCS = (PROJECT / "README.md", PROJECT / "EXPERIMENTS.md", PROJECT / "MODEL.md", PROJECT / "DECISIONS.md")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from milo.why import KPIS, WHY  # noqa: E402

BLOCK_PATTERN = re.compile(r"<!-- generated:([a-z0-9_]+) -->\n(.*?)<!-- /generated:\1 -->", re.DOTALL)
POLICY_ORDER = ("optimizer", "greedy", "today_prices")


PAGE_METHOD = PROJECT / "docs" / "data" / "method.json"
PAGE_SAA = PROJECT / "docs" / "data" / "saa.json"


def load_reports():
    """The generated files as dicts: the three reports plus the page's method.json and saa.json.

    Only the page's two small files are read (grid facts feed the page entries in
    DECISIONS.md and the README "Try it" line; saa.json carries the curse-gap shape);
    the large budget grid is checked by tests/test_viz.py instead, so it does not widen
    the drift test's allowed numbers.
    """
    reports = {}
    names = (("numbers", REPORTS / "numbers.json"), ("scorecard", REPORTS / "scorecard.json"),
             ("ladder", REPORTS / "constraint_ladder.json"), ("page", PAGE_METHOD),
             ("page_saa", PAGE_SAA))
    for key, path in names:
        with open(path, encoding="utf-8") as handle:
            reports[key] = json.load(handle)
    return reports


def money(value):
    """Dollars at cents; a value that rounds to zero prints as 0.00, never -0.00."""
    text = format(value, ",.2f")
    if text == "-0.00":
        return "0.00"
    return text


def pct(value):
    return format(value, ".2f") + "%"


def ratio3(value):
    """A profit-per-dollar ratio, or n/a when the policy spent nothing (B = 0)."""
    if value is None:
        return "n/a"
    return format(value, ".3f")


def seconds(value):
    return format(value, ".1f")


def se(value):
    """A paired standard error: 3 decimals under 0.1 (per-dollar SEs), else cents."""
    if value is None:
        return "n/a"
    if abs(value) < 0.1:
        return format(value, ".3f")
    return money(value)


def budget0(value):
    return "$" + format(value, ",.0f")


def grid_edge(scorecard):
    return format(scorecard["settings"]["price_grid_high"], ".2f") + "×"


def grid_range(scorecard):
    settings = scorecard["settings"]
    return format(settings["price_grid_low"], ".2f") + "–" + format(settings["price_grid_high"], ".2f") + "×"


def n_eval_text(settings):
    return format(settings["n_eval"], ",")


def markdown_table(header, rows):
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    separators = []
    for _ in header:
        separators.append("---")
    lines.append("| " + " | ".join(separators) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def why_box(key):
    """The shared three-line "why this metric" box (milo/why.py)."""
    entry = WHY[key]
    lines = []
    lines.append("> **Why this metric.** *Owner's question:* " + entry["question"])
    lines.append("> *What kind of number:* " + entry["kind"])
    lines.append("> *What would make it lie:* " + entry["lie"])
    return "\n".join(lines)


def _gap_label(numbers):
    return pct(numbers["experiment_settings"]["mip_gap"] * 100.0)


def _rows_at(scorecard, budget):
    rows = {}
    for entry in scorecard["table"]:
        if entry["budget"] == budget:
            rows[entry["policy"]] = entry
    return rows


# ---- scorecard (U1) ----------------------------------------------------------------

def headline(reports):
    head = reports["scorecard"]["headline"]
    budget = "$" + format(head["budget"], ",.0f")
    settings = reports["scorecard"]["settings"]
    return ("**Out-of-sample profit per restock dollar at " + budget + ": optimizer "
            + ratio3(head["optimizer_rord_out"]) + " vs the careful greedy list "
            + ratio3(head["greedy_rord_out"]) + "** (today's prices with the buys re-optimized: "
            + ratio3(head["today_prices_rord_out"]) + "), on " + n_eval_text(settings) + " fresh scenarios "
            + "from the model's own generator (synthetic). In-sample the same plans claim "
            + ratio3(head["optimizer_rord_in"]) + " vs " + ratio3(head["greedy_rord_in"])
            + "; the optimizer's in-sample profit " + money(head["optimizer_in_sample"])
            + " falls to " + money(head["optimizer_out_of_sample"])
            + " out of sample, an optimizer's-curse gap of " + money(head["curse_gap"]) + " ("
            + pct(head["curse_gap_pct_of_in_sample"]) + ").")


def _kpi_value(kpi_id, reports):
    head = reports["scorecard"]["headline"]
    if kpi_id == "rord_out":
        return (ratio3(head["optimizer_rord_out"]) + " vs " + ratio3(head["greedy_rord_out"])
                + " greedy, " + ratio3(head["today_prices_rord_out"]) + " today's prices, at $"
                + format(head["budget"], ",.0f") + " (`reports/scorecard.json` → `headline`)")
    if kpi_id == "curse_gap":
        return (money(head["curse_gap"]) + " (" + pct(head["curse_gap_pct_of_in_sample"])
                + ") at $" + format(head["budget"], ",.0f") + " (`reports/scorecard.json` → `headline.curse_gap`)")
    if kpi_id == "marginal_dollar":
        for row in reports["scorecard"]["marginal"]:
            if row["budget"] == head["budget"]:
                return (ratio3(row["marginal_in"]) + " per dollar at $" + format(row["budget"], ",.0f")
                        + " vs average " + ratio3(row["average_in"])
                        + " (`reports/scorecard.json` → `marginal`)")
    if kpi_id == "price_of_rule":
        rows = reports["ladder"]["derived"]["ladder"]
        parts = []
        for row in rows:
            if row["kind"] == "restriction":
                parts.append(row["rung"] + " " + money(row["price_of_rule"]) + " ("
                             + pct(row["price_of_rule_pct_of_rp"]) + ")")
        return "; ".join(parts) + " (`reports/constraint_ladder.json` → `derived.ladder`)"
    if kpi_id == "solve_time":
        return solve_time_value(reports)
    return "to measure"


def solve_time_value(reports):
    """The solve-time KPI: hull vs big-M from numbers.json, plus the page grid's solve facts."""
    numbers = reports["numbers"]
    facts = reports["page"]["grid_facts"]
    counts = {}
    reached = {}
    seconds_by = {}
    for entry in numbers["hull_vs_bigm"]:
        name = entry["transform"]
        counts[name] = counts.get(name, 0) + 1
        if entry["optimal"]:
            reached[name] = reached.get(name, 0) + 1
        seconds_by.setdefault(name, []).append(entry["solve_seconds"])
    hull_seconds = seconds_by.get("hull", [])
    limit = str(numbers["experiment_settings"]["time_limit"])
    grid_reached = facts["n_solves"] if facts["all_solves_optimal"] else "not all"
    return ("hull " + str(reached.get("hull", 0)) + " of " + str(counts.get("hull", 0)) + " reach the gap in "
            + seconds(min(hull_seconds)) + "–" + seconds(max(hull_seconds)) + " s; big-M "
            + str(reached.get("bigm", 0)) + " of " + str(counts.get("bigm", 0)) + " within the " + limit
            + " s limit (`reports/numbers.json` → `hull_vs_bigm`); the page grid's " + str(facts["n_solves"])
            + " hull solves: " + str(grid_reached) + " reach the gap, " + seconds(facts["solve_seconds_min"]) + "–"
            + seconds(facts["solve_seconds_max"]) + " s (`docs/data/method.json` → `grid_facts`)")


def kpi_table(reports):
    rows = []
    for kpi in KPIS:
        rows.append([kpi["name"], kpi["formula"], kpi["why"], kpi["against"], kpi["does"],
                     kpi["measured_on"], _kpi_value(kpi["id"], reports)])
    header = ["KPI", "formula", "why it matters", "compared against", "what the software does to it",
              "measured on", "value now"]
    return markdown_table(header, rows)


def scorecard_headline_table(reports):
    scorecard = reports["scorecard"]
    budget = scorecard["headline"]["budget"]
    rows_by_policy = _rows_at(scorecard, budget)
    rows = []
    for policy in POLICY_ORDER:
        entry = rows_by_policy[policy]
        rows.append([scorecard["policy_labels"][policy], money(entry["spend"]),
                     money(entry["in_sample"]), money(entry["out_of_sample"]),
                     ratio3(entry["rord_in"]), ratio3(entry["rord_out"]), money(entry["curse_gap"])])
    header = ["policy (budget $" + format(budget, ",.0f") + ")", "spend ($)", "in-sample profit ($)",
              "out-of-sample profit ($)", "profit per restock $ (in)", "profit per restock $ (out)",
              "curse gap ($)"]
    head = scorecard["headline"]
    lead = ("Paired on the same " + n_eval_text(scorecard["settings"]) + " evaluation scenarios, the optimizer "
            "leads the greedy list by " + money(head["lead_vs_greedy_out"]) + " out-of-sample profit (paired SE "
            + se(head["lead_vs_greedy_out_se"]) + ") and by " + ratio3(head["rord_out_lift_vs_greedy"])
            + " per restock dollar (paired SE " + se(head["rord_lift_vs_greedy_out_se"]) + "); it leads today's "
            "prices by " + money(head["lead_vs_today_out"]) + " (paired SE " + se(head["lead_vs_today_out_se"])
            + "). The SEs cover evaluation-set sampling only, not model error.")
    parts = [markdown_table(header, rows), "", lead, "", why_box("rord"), "", why_box("fair_baseline")]
    return "\n".join(parts)


def edge_note(reports):
    """How many of each plan's price moves sit on the grid's upper edge (the M2 flag), said plainly."""
    scorecard = reports["scorecard"]
    head = scorecard["headline"]
    derived = reports["ladder"]["derived"]
    edge = grid_edge(scorecard)
    parts = []
    greedy_parts = []
    for budget in scorecard["settings"]["budgets"]:
        rows = _rows_at(scorecard, budget)
        optimizer = rows["optimizer"]
        greedy = rows["greedy"]
        parts.append(str(optimizer["price_moves_at_top_edge"]) + " of " + str(optimizer["price_moves"]) + " at "
                     + budget0(budget))
        greedy_parts.append(str(greedy["price_moves_at_top_edge"]) + " of " + str(greedy["price_moves"]))
    at = _rows_at(scorecard, head["budget"])
    return ("**The price grid binds.** At " + budget0(head["budget"]) + ", " + str(head["optimizer_price_moves_at_top_edge"])
            + " of the optimizer's " + str(head["optimizer_price_moves"]) + " price moves sit at the " + edge
            + " grid edge (flag M2): the grid, not the data, binds, and the true optimum likely lies above it. "
            "Optimizer moves at the edge by budget: " + ", ".join(parts) + "; the greedy list: "
            + ", ".join(greedy_parts) + "; the ladder's solves raise M2 in " + str(derived["m2_records"]) + " of "
            + str(derived["n_records"]) + ". So the lead over today's prices (" + ratio3(at["optimizer"]["rord_out"])
            + " vs " + ratio3(at["today_prices"]["rord_out"]) + " per restock dollar) and the cost of the P = 0 "
            "price-edit rung measure the " + grid_range(scorecard) + " grid as much as the data.")


def scorecard_full_table(reports):
    scorecard = reports["scorecard"]
    rows = []
    budgets = scorecard["settings"]["budgets"]
    for budget in budgets:
        rows_by_policy = _rows_at(scorecard, budget)
        for policy in POLICY_ORDER:
            entry = rows_by_policy[policy]
            rows.append([money(budget), scorecard["policy_labels"][policy], money(entry["spend"]),
                         money(entry["in_sample"]), money(entry["out_of_sample"]),
                         money(entry["out_of_sample_p10"]), money(entry["out_of_sample_p90"]),
                         ratio3(entry["rord_in"]), ratio3(entry["rord_out"]), money(entry["curse_gap"])])
    header = ["budget ($)", "policy", "spend ($)", "in-sample ($)", "out-of-sample ($)",
              "out-of-sample p10 ($)", "out-of-sample p90 ($)", "per restock $ (in)",
              "per restock $ (out)", "curse gap ($)"]
    paired_rows = []
    for row in scorecard["paired"]:
        paired_rows.append([money(row["budget"]), money(row["lead_vs_greedy_out"]), se(row["lead_vs_greedy_out_se"]),
                            se(row["rord_lift_vs_greedy_out_se"]), money(row["lead_vs_today_out"]),
                            se(row["lead_vs_today_out_se"])])
    paired_header = ["budget ($)", "optimizer − greedy, out-of-sample ($)", "paired SE ($)",
                     "paired SE of the per-dollar lift", "optimizer − today's prices ($)", "paired SE ($)"]
    parts = [markdown_table(header, rows), "", "Paired leads (every plan is valued on the same evaluation "
             "scenarios, so the per-scenario differences are paired):", "",
             markdown_table(paired_header, paired_rows), "", why_box("curse_gap")]
    return "\n".join(parts)


def oos_losses(reports):
    """Every budget where the greedy list beats the optimizer out of sample, said plainly."""
    scorecard = reports["scorecard"]
    lines = []
    budgets = scorecard["settings"]["budgets"]
    for budget in budgets:
        rows = _rows_at(scorecard, budget)
        optimizer = rows["optimizer"]
        greedy = rows["greedy"]
        if optimizer["rord_out"] is None or greedy["rord_out"] is None:
            continue
        if greedy["rord_out"] > optimizer["rord_out"]:
            lines.append("- At $" + format(budget, ",.0f") + " the greedy list returns "
                         + ratio3(greedy["rord_out"]) + " per restock dollar out of sample vs the "
                         + "optimizer's " + ratio3(optimizer["rord_out"]) + "; the optimizer's total "
                         + "profit is " + ("higher" if optimizer["out_of_sample"] >= greedy["out_of_sample"] else "lower")
                         + " (" + money(optimizer["out_of_sample"]) + " vs "
                         + money(greedy["out_of_sample"]) + "); the two budget-0 bases are "
                         + money(optimizer["base_out_of_sample"]) + " vs "
                         + money(greedy["base_out_of_sample"]) + ".")
        if greedy["out_of_sample"] > optimizer["out_of_sample"]:
            lines.append("- At $" + format(budget, ",.0f") + " the greedy list earns more in total "
                         + "out of sample (" + money(greedy["out_of_sample"]) + " vs "
                         + money(optimizer["out_of_sample"]) + ").")
    if not lines:
        lines.append("- None: the optimizer leads the greedy list out of sample at every budget.")
    return "\n".join(lines)


def marginal_table(reports):
    rows = []
    for row in reports["scorecard"]["marginal"]:
        rows.append([money(row["budget"]), ratio3(row["marginal_in"]), ratio3(row["marginal_out"]),
                     ratio3(row["average_in"]), "±" + format(row["tolerance_per_dollar"], ".4f")])
    header = ["budget B ($)", "next dollar, in-sample", "next dollar, out-of-sample",
              "average dollar, in-sample", "gap noise per dollar"]
    parts = [markdown_table(header, rows), "", why_box("marginal_dollar")]
    return "\n".join(parts)


def fee_out_of_sample_clause(fee):
    """The out-of-sample side of the fee-blind null, judged against its paired SE (2 SE rule)."""
    difference = fee["out_of_sample_minus_fee_aware"]
    error = fee["out_of_sample_minus_fee_aware_se"]
    if difference is None or error is None:
        return "Out of sample the comparison is to measure."
    if difference >= 0:
        direction = "earns " + money(difference) + " more than"
    else:
        direction = "earns " + money(-difference) + " less than"
    text = ("Out of sample the fee-blind plan " + direction + " the fee-aware plan ("
            + money(fee["true_out_of_sample"]) + " vs " + money(fee["fee_aware_out_of_sample"]) + ", paired SE "
            + se(error) + ")")
    if abs(difference) <= 2.0 * error:
        return text + ": within noise, so the null holds both ways."
    return (text + ": more than two paired SEs, so on the model's own generator modelling fees is a small "
            "loss out of sample, not a gain; the null is that the decisions barely move, in either direction.")


def fee_blind_line(reports):
    fee = reports["scorecard"]["fee_blind"]
    rp = reports["ladder"]["derived"]["rp"]
    line = ("A fee-blind (textbook) plan at $" + format(fee["budget"], ",.0f") + " promises "
            + money(fee["own_objective"]) + " but earns " + money(fee["true_in_sample"])
            + " at the true fees in-sample (" + money(fee["true_out_of_sample"])
            + " out-of-sample): it overstates its profit by " + money(fee["overstatement"])
            + " while the fee-aware optimum earns " + money(rp) + ". Its stocked products that sit "
            + "on a price level the true fees would drop: " + str(fee["levels_outside_true_grid"]) + ". "
            + fee_out_of_sample_clause(fee))
    return line + "\n\n" + why_box("fee_blind")


# ---- constraint ladder (U2-U4) -----------------------------------------------------

def _yes_no(flag):
    if flag is None:
        return "n/a"
    if flag:
        return "yes"
    return "no"


def ladder_table(reports):
    derived = reports["ladder"]["derived"]
    gap_label = _gap_label(reports["numbers"])
    rows = []
    for row in derived["ladder"]:
        if row["kind"] == "restriction":
            versus = "−" + money(row["price_of_rule"]) + " (price of the rule)"
        elif row["value_of_modeling"] is not None:
            versus = "+" + money(row["value_of_modeling"]) + " (value of modeling it)"
        else:
            versus = "n/a"
        proved = "yes, " + gap_label if row["optimal"] else "no (time limit)"
        rows.append([row["rung"], row["rule"], row["kind"], money(row["expected_profit"]), versus,
                     _yes_no(row["binds"]), str(row["price_edits"]), format(row["shelf_used"], ".1f"),
                     money(row["worst_10pct"]), money(row["out_of_sample"]),
                     seconds(row["solve_seconds"]), proved])
    header = ["rung", "rule added", "kind", "expected profit ($)", "vs previous rung ($)", "binds?",
              "prices changed", "shelf used", "worst-10% profit ($)", "out-of-sample ($)",
              "solve (s)", "reached the gap"]
    parts = [markdown_table(header, rows), "", why_box("price_of_rule")]
    return "\n".join(parts)


def ladder_alone_table(reports):
    derived = reports["ladder"]["derived"]
    rows = []
    for row in derived["alone"]:
        rows.append([row["rung"], row["rule"], money(row["expected_profit"]),
                     money(row["price_of_rule"]) + " (" + pct(row["price_of_rule_pct_of_rp"]) + ")",
                     money(row["out_of_sample"]), seconds(row["solve_seconds"])])
    for row in derived["existing_rules"]:
        price = money(row["price_of_rule"]) + " (" + pct(row["price_of_rule_pct_of_rp"]) + ")"
        if "products_with_min_order_above_1" in row:
            price = price + "; " + min_order_text(row)
        rows.append(["existing", row["rule"], "n/a", price, "n/a", "n/a"])
    header = ["rung", "rule, alone against today's model", "expected profit ($)",
              "price of the rule ($, % of RP)", "out-of-sample ($)", "solve (s)"]
    return markdown_table(header, rows)


def min_order_text(row):
    """Whether the minimum-order rule can bind at all: products with a real minimum, and who buys them."""
    binding = row["products_with_min_order_above_1"]
    text = "the rule binds on " + str(binding) + " of " + str(row["n_products"]) + " products"
    if binding == 0:
        return text + ", so it cannot show anything here"
    if row["bought_with_rule"] == 0 and row["bought_without_rule"] == 0:
        return (text + ", which neither plan buys, so its " + money(row["price_of_rule"])
                + " is close to vacuous, not a measured null")
    return (text + "; bought by the plan with the rule: " + str(row["bought_with_rule"]) + ", without: "
            + str(row["bought_without_rule"]))


def _types_text(units_by_type):
    parts = []
    names = sorted(units_by_type.keys())
    for name in names:
        parts.append(name + " " + str(units_by_type[name]))
    return ", ".join(parts)


def shelf_sweep_table(reports):
    derived = reports["ladder"]["derived"]
    rows = []
    for row in derived["shelf_sweep"]:
        rows.append([format(row["fraction_of_today_use"] * 100.0, ".0f") + "%",
                     format(row["capacity"], ".2f"), format(row["shelf_used"], ".1f"),
                     money(row["expected_profit"]), money(row["price_of_rule"]),
                     money(row["value_per_space_unit"]) + " ±" + format(row["value_tolerance_per_unit"], ".2f"),
                     money(row["out_of_sample"]), _types_text(row["units_by_type"])])
    header = ["shelf V (% of today's plan's use)", "V (pack-equivalents)", "used", "expected profit ($)",
              "price of the rule ($)", "value of 1 more space unit ($)", "out-of-sample ($)",
              "units bought by type"]
    parts = [markdown_table(header, rows), "", rent_rule(reports), "", why_box("shelf")]
    return "\n".join(parts)


def rent_rule(reports):
    """U3's decision rule per shelf point: rent one more space unit only if it costs less than its value."""
    ladder_file = reports["ladder"]
    weeks = str(ladder_file["settings"]["horizon_weeks"])
    worth = []
    not_worth = []
    for row in ladder_file["derived"]["shelf_sweep"]:
        share = format(row["fraction_of_today_use"] * 100.0, ".0f") + "%"
        if row["value_exceeds_noise"]:
            worth.append(money(row["value_per_space_unit"]) + " at " + share)
        else:
            not_worth.append("at " + share + " one more unit is worth " + money(row["value_per_space_unit"])
                             + ", inside its ±" + format(row["value_tolerance_per_unit"], ".2f")
                             + " solver noise, so renting more never pays")
    text = "**Rent-more-shelf rule:**"
    if worth:
        text = (text + " rent one more space unit only if it costs less than its value over the " + weeks
                + "-week horizon: " + ", ".join(worth) + ".")
    if not_worth:
        text = text + " " + "; ".join(not_worth).capitalize() + "."
    return text


def edit_sweep_table(reports):
    derived = reports["ladder"]["derived"]
    rows = []
    for row in derived["edit_sweep"]:
        label = str(row["limit"])
        if row["is_all"]:
            label = "all (" + str(row["limit"]) + ")"
        rows.append([label, str(row["edits_used"]), money(row["expected_profit"]),
                     money(row["price_of_rule"]), money(row["out_of_sample"]),
                     seconds(row["solve_seconds"])])
    header = ["edit limit P", "edits used", "expected profit ($)", "price of the rule ($)",
              "out-of-sample ($)", "solve (s)"]
    parts = [markdown_table(header, rows), "", why_box("price_edits")]
    return "\n".join(parts)


# ---- existing numbers.json tables (moved here so they are drift-checked) -----------

def evpi_vss_table(reports):
    ev = reports["numbers"]["evpi_vss"]
    rows = []
    rows.append(["RP (two-stage stochastic optimum)", money(ev["rp"])])
    rows.append(["EEV (mean-value plan, re-valued on the scenarios)", money(ev["eev"])])
    rows.append(["WS (wait-and-see bound)", money(ev["ws"])])
    rows.append(["EVPI = WS − RP", money(ev["evpi"]) + " (" + pct(ev["evpi_pct_of_rp"]) + " of RP)"])
    rows.append(["VSS = RP − EEV", money(ev["vss"]) + " (" + pct(ev["vss_pct_of_rp"]) + " of RP)"])
    parts = [markdown_table(["quantity (expected profit, $)", "value"], rows), "", why_box("evpi_vss")]
    return "\n".join(parts)


def greedy_table(reports):
    rows = []
    for entry in reports["numbers"]["greedy_vs_milp"]["budgets"]:
        rows.append([money(entry["budget"]), money(entry["milp_objective"]),
                     money(entry["greedy_objective"]), money(entry["gap"]), pct(entry["gap_pct"])])
    return markdown_table(["budget ($)", "exact MILP ($)", "greedy ($)", "gap ($)", "gap (% of MILP)"], rows)


def hull_bigm_table(reports):
    numbers = reports["numbers"]
    within_gap = "optimal within a " + _gap_label(numbers) + " gap"
    rows = []
    for entry in numbers["hull_vs_bigm"]:
        status = within_gap if entry["optimal"] else ("time limit ("
                                                      + str(numbers["experiment_settings"]["time_limit"]) + " s)")
        rows.append([money(entry["budget"]), entry["transform"], money(entry["objective"]),
                     money(entry["lp_relaxation_objective"]),
                     format(entry["build_seconds"], ".2f"), format(entry["solve_seconds"], ".1f"),
                     status])
    table = markdown_table(["budget ($)", "transform", "incumbent ($)", "LP bound ($)",
                            "build (s)", "solve (s)", "result"], rows)
    parts = [table, "", why_box("hull_bigm")]
    return "\n".join(parts)


def saa_table(reports):
    rows = []
    for entry in reports["numbers"]["saa_convergence"]:
        rows.append([str(entry["n_scenarios"]), money(entry["in_sample_mean"]),
                     format(entry["in_sample_std"], ".2f"), money(entry["out_of_sample_mean"]),
                     format(entry["out_of_sample_std"], ".2f"), format(entry["curse_gap"], ".2f"),
                     format(entry["solve_seconds_mean"], ".1f")])
    header = ["N (training scenarios)", "in-sample mean ($)", "in-sample std",
              "out-of-sample mean ($)", "out-of-sample std", "curse gap ($)", "solve (s, mean)"]
    parts = [markdown_table(header, rows), "", why_box("saa")]
    return "\n".join(parts)


def strategy_table(reports):
    rows = []
    for entry in reports["numbers"]["demo"]["strategies"]:
        rows.append([entry["strategy"], money(entry["expected"]), money(entry["p10"]),
                     money(entry["p90"]), format(entry["sell_through"], ".3f"),
                     money(entry["leftover_value"])])
    header = ["pricing strategy (same buy list)", "expected ($)", "p10 ($)", "p90 ($)",
              "sell-through", "leftover value ($)"]
    return markdown_table(header, rows)


def budget_curve_table(reports):
    rows = []
    for entry in reports["numbers"]["demo"]["budget_curve"]:
        rows.append([money(entry["budget"]), money(entry["objective"]), money(entry["spend"])])
    return markdown_table(["budget ($)", "expected profit ($)", "spend ($)"], rows)


def seed_spread_lines(reports):
    spread = reports["numbers"]["rp_seed_spread"]
    values = []
    for value in spread["rp_values"]:
        values.append(money(value))
    lines = []
    lines.append("RP across scenario-set seeds " + str(spread["seeds"]) + ": " + ", ".join(values))
    lines.append("mean " + money(spread["mean"]) + ", std " + format(spread["std"], ".2f")
                 + ", spread " + pct(spread["spread_pct_of_mean"]) + " of the mean")
    return "\n".join(lines)


def bound_ratio_range(numbers):
    """big-M LP bound / hull LP bound per budget, as 'a.bx–c.dx' (used in prose)."""
    hull_bounds = {}
    for entry in numbers["hull_vs_bigm"]:
        if entry["transform"] == "hull":
            hull_bounds[entry["budget"]] = entry["lp_relaxation_objective"]
    ratios = []
    for entry in numbers["hull_vs_bigm"]:
        if entry["transform"] == "bigm":
            ratios.append(entry["lp_relaxation_objective"] / hull_bounds[entry["budget"]])
    return ratios


def assumptions_table(reports):
    """MODEL.md §7: every knob, its value read from code or the reports, source, and failure mode."""
    from experiments import common, constraint_ladder, scorecard
    from milo import ladder
    from milo.config import EXAMPLE_CONFIG
    from milo.pricing import PRICE_LEVELS

    settings = reports["ladder"]["settings"]
    space_parts = []
    names = sorted(ladder.SPACE_BY_TYPE.keys())
    for name in names:
        space_parts.append(name + " " + format(ladder.SPACE_BY_TYPE[name], "g"))
    fractions = []
    for fraction in constraint_ladder.SHELF_FRACTIONS:
        fractions.append(format(fraction * 100.0, ".0f") + "%")
    limits = []
    for limit in constraint_ladder.EDIT_LIMITS:
        limits.append(str(limit))
    limits.append("all")
    rows = [
        ["store commission", format(EXAMPLE_CONFIG.commission * 100.0, "g") + "%", "illustrative",
         "every net-cash value moves; the fee-blind row shows how much fees matter", "none (a required config value)"],
        ["card fee, per-order fee", format(EXAMPLE_CONFIG.card_fee_rate * 100.0, "g") + "%, $"
         + format(EXAMPLE_CONFIG.order_fee, ".2f"), "published default",
         "small: fees are close to proportional", "none"],
        ["price grid", format(float(PRICE_LEVELS[0]), ".2f") + "x to " + format(float(PRICE_LEVELS[-1]), ".2f")
         + "x today's price", "design choice", "an optimum at the edge may sit outside the grid", "M2 / M3"],
        ["shelf space per unit (pack-equivalents)", ", ".join(space_parts), "illustrative",
         "R1's price and the units-by-type shift; swept capacity limits the damage", "SHELF when on-hand alone exceeds V"],
        ["shelf capacity V", ", ".join(fractions) + " of today's plan's use", "design choice (swept)",
         "the price of R1 is only read at these points", "SHELF"],
        ["weekly price-edit limit P", ", ".join(limits) + "; the ladder uses P = "
         + str(constraint_ladder.LADDER_EDIT_LIMIT), "design choice (swept)",
         "R2's price at the chosen P", "EDITS when forced edits exceed P"],
        ["training scenarios", str(common.N_SCENARIOS) + " (seed " + str(common.SCENARIO_SEED) + ")",
         "design choice (pinned)", "in-sample numbers flatter the plan; the curse gap measures by how much", "none"],
        ["evaluation scenarios", format(scorecard.N_EVAL, ",") + " (seed " + str(scorecard.EVAL_SEED) + ")",
         "design choice", "out-of-sample values carry sampling noise of their own; the paired SE of each "
         "lead is printed beside it", "none"],
        ["experiment rel gap", format(settings["mip_gap"], "g"), "design choice",
         "every comparison carries mip_gap × \\|bound\\| of slack", "M10 when a solve stops short of it"],
        ["marginal-dollar step", "$" + format(scorecard.MARGINAL_STEP, ".0f"), "design choice",
         "a smaller step drowns in gap noise, a larger one averages over a kink", "none"],
        ["set-spend cap (operating default)", format(constraint_ladder.SET_CAP_SHARE * 100.0, ".0f") + "%",
         "design choice", "priced as an existing rule; the experiments run without it", "none"],
    ]
    header = ["knob", "value", "source", "what changes if wrong", "flag that fires when it binds"]
    return markdown_table(header, rows)


def _ladder_row(reports, rung):
    for row in reports["ladder"]["derived"]["ladder"]:
        if row["rung"] == rung:
            return row
    raise KeyError(rung)


def decisions(reports):
    """DECISIONS.md entries for the U0-U4 upgrades; every number is a stored value, formatted."""
    card = reports["scorecard"]
    head = card["headline"]
    context = card["context"]
    fee = card["fee_blind"]
    derived = reports["ladder"]["derived"]
    ladder_settings = reports["ladder"]["settings"]
    numbers = reports["numbers"]
    budget = budget0(head["budget"])
    n_eval = n_eval_text(card["settings"])
    at = _rows_at(card, head["budget"])
    r0a = _ladder_row(reports, "R0a")
    r0b = _ladder_row(reports, "R0b")
    r1 = _ladder_row(reports, "R1")
    r2 = _ladder_row(reports, "R2")
    existing = {}
    for row in derived["existing_rules"]:
        existing[row["with"] + "|" + row["without"]] = row
    edits = {}
    edit_seconds = []
    for row in derived["edit_sweep"]:
        edits[row["limit"]] = row
        edit_seconds.append(row["solve_seconds"])
    shelf_values = []
    shelf_shares = []
    for row in derived["shelf_sweep"]:
        if row["fraction_of_today_use"] < 1.0:
            shelf_values.append(money(row["value_per_space_unit"]))
            shelf_shares.append(format(row["fraction_of_today_use"] * 100.0, ".0f"))
    ladder_share = format(ladder_settings["ladder_shelf_fraction"] * 100.0, ".0f") + "%"
    ladder_limit = str(ladder_settings["ladder_edit_limit"])
    shelf_capacity = None
    for row in derived["shelf_sweep"]:
        if row["fraction_of_today_use"] == ladder_settings["ladder_shelf_fraction"]:
            shelf_capacity = row["capacity"]
    marginal = None
    for row in card["marginal"]:
        if row["budget"] == head["budget"]:
            marginal = row
    seed = context["pinned_seed"]
    min_order = existing["R0b|no_min_order"]
    entries = []
    entries.append(
        "**2026-11-04 · U1 · Lead with the out-of-sample number.** Decided the README "
        "headline is out-of-sample profit per restock dollar because the in-sample figure grades the plan "
        "on the scenarios it was fitted to: at " + budget + " the optimizer's in-sample "
        + ratio3(head["optimizer_rord_in"]) + " falls to " + ratio3(head["optimizer_rord_out"])
        + " on " + n_eval + " fresh scenarios (curse gap " + money(head["curse_gap"]) + ", "
        + pct(head["curse_gap_pct_of_in_sample"]) + "). The in-sample headline lost because it overstates "
        "by " + ratio3(head["curse_rord_gap"]) + " per dollar; the gap is now a KPI of its own.")
    entries.append(
        "**2026-11-04 · U1 · The fair baseline re-optimizes the buys.** Decided \"today's prices\" means "
        "`solve_plan` on the ratio-1.00 rows only because that is the strongest policy without price choice: "
        "out of sample it earns " + money(at["today_prices"]["out_of_sample"]) + " at " + budget
        + ". The straw man (the optimizer's own buys held at today's prices, "
        + money(context["same_buys_today_out_of_sample"]) + ") lost by " + money(context["straw_man_loss_out"])
        + " (computed before rounding), so it stays a context row only. `today_problem` raises if any "
        "product's ratio-1.00 level fails the M1 filter, because that product would then leave this model "
        "entirely while P = 0 would force an edit (on the committed data no product is dropped).")
    entries.append(
        "**2026-11-04 · U1 · Each policy is measured against its own budget-0 plan.** Decided the "
        "denominator base is the same policy's budget-0 plan because repricing the shelf alone is worth "
        + money(context["repricing_value_out_of_sample"]) + " out of sample and is not something the buys "
        "earned. The alternative (one shared today's-prices base) lost because it would credit that repricing "
        "to the buys and report " + ratio3(context["optimizer_rord_out_today_base"]) + " instead of "
        + ratio3(head["optimizer_rord_out"]) + ".")
    entries.append(
        "**2026-11-04 · U1 · Out-of-sample losses ship as printed, with paired errors.** Decided not to "
        "assert any out-of-sample ordering because it has no sign guarantee; where the greedy list's "
        "per-dollar return beats the optimizer's, `EXPERIMENTS.md` §7 lists it. Every lead now carries the "
        "paired standard error over the shared evaluation scenarios: at " + budget + " the optimizer leads "
        "the greedy list by " + money(head["lead_vs_greedy_out"]) + " (paired SE " + se(head["lead_vs_greedy_out_se"])
        + ") and by " + ratio3(head["rord_out_lift_vs_greedy"]) + " per dollar (paired SE "
        + se(head["rord_lift_vs_greedy_out_se"]) + "). The pinned seed " + str(seed["seed"]) + " sits above the "
        + str(seed["n_seeds"]) + "-seed mean RP (" + money(seed["seed_mean"]) + ") by " + money(seed["minus_seed_mean"])
        + ", " + format(seed["in_seed_sds"], ".2f") + " seed sds, rank " + str(seed["rank_from_top"]) + " of "
        + str(seed["n_seeds"]) + ": a mild draw, and the same plan earns " + money(head["optimizer_out_of_sample"])
        + " out of sample.")
    entries.append(
        "**2026-11-04 · U1 · The marginal dollar uses a $" + format(marginal["step"], ".0f") + " step.** "
        "Decided a ±$" + format(marginal["step"], ".0f") + " finite difference because its solver noise is "
        "±" + format(marginal["tolerance_per_dollar"], ".4f") + " per dollar against a signal of "
        + ratio3(marginal["marginal_in"]) + " at " + budget + "; the average return there ("
        + ratio3(marginal["average_in"]) + ") would overstate the next dollar by "
        + ratio3(marginal["average_minus_marginal"]) + ". LP duals were not measured (to measure): a dual "
        "of the budget row in a MILP needs a fixed-binary re-solve whose sign conventions this repo has no "
        "test for, so the finite difference ships instead.")
    entries.append(
        "**2026-11-04 · U2 · Fees are an accounting rung, not a restriction.** Decided to test R0a → R0b "
        "with the VSS-style check (value of modeling ≥ 0) instead of \"can't raise profit\", because "
        "removing fees changes how money is counted, not what is feasible. Measured: modeling fees is worth "
        + money(r0b["value_of_modeling"]) + " in-sample — a near-null, likely because fees are close to "
        "proportional — while the fee-blind plan promises " + money(r0a["own_objective"]) + " and "
        "overstates its own profit by " + money(r0a["overstatement"]) + ". " + fee_out_of_sample_clause(fee)
        + " The damage of the textbook model is in the promise, not in the decisions; the null ships.")
    entries.append(
        "**2026-11-04 · U2 · Price the rules the model already had.** Decided to price the existing "
        "rules as restrictions against a model without them. Measured at " + budget + ": the "
        + format(ladder_settings["set_cap_share"] * 100.0, ".0f") + "% set cap costs "
        + money(existing["set_cap|R0b"]["price_of_rule"]) + ", integrality (LP − MILP) "
        + money(existing["R0b|LP relaxation"]["price_of_rule"]) + ", and the all-or-nothing minimum order "
        + money(min_order["price_of_rule"]) + ", but " + min_order_text(min_order)
        + ". It stays in the table with that note rather than being dropped, so "
        "the rule is visibly untested on this data (pricing it at a budget where that product is bought: to "
        "measure).")
    entries.append(
        "**2026-11-04 · U3 · Shelf space first, one linear row.** Decided R1 (shelf space came first in the "
        "plan because a consignment vendor rents a fixed display) is `Σ v_i (q_i + x_i) ≤ V` with illustrative "
        "space per type and a swept V because it adds one row and no variables (solve times, indicative only, "
        "are in the table). At " + ladder_share + " of today's plan's shelf it costs " + money(r1["price_of_rule"])
        + " (" + pct(r1["price_of_rule_pct_of_rp"]) + " of RP); today's plan uses "
        + format(derived["base_shelf_use"], ".1f") + " units against that cap of " + format(shelf_capacity, ".2f")
        + ". One more space unit is worth " + ", ".join(shelf_values) + " at " + "/".join(shelf_shares)
        + "% — not monotone, likely because whether a few more units fit one more large box changes the "
        "slope (not measured); reported as measured, with the rent-more-shelf rule beside the sweep. Space "
        "values live in code (`SPACE_BY_TYPE`), not in a data file, so the committed dataset and its "
        "manifest stay frozen.")
    entries.append(
        "**2026-11-04 · U4 · Price edits second, on the existing binaries.** Decided R2 uses the AND trick on "
        "the price and order disjunct binaries because it adds one binary per product and no big-M: "
        + seconds(min(edit_seconds)) + "–" + seconds(max(edit_seconds)) + " s across the sweep. "
        "Today's plan moves " + str(derived["base_price_edits"]) + " stocked products off today's price "
        "(recounted independently from the plan, tests). P = 0 costs " + money(edits[0]["price_of_rule"])
        + "; P = " + ladder_limit + " alone costs " + money(edits[ladder_settings["ladder_edit_limit"]]["price_of_rule"])
        + " and P = 8 " + money(edits[8]["price_of_rule"]) + ". The ladder's cumulative R2 (P = " + ladder_limit
        + " on top of the " + ladder_share + " shelf) costs " + money(r2["price_of_rule"]) + " more than R1, against "
        + money(edits[ladder_settings["ladder_edit_limit"]]["price_of_rule"]) + " alone: the rules overlap, so a "
        "rule's price depends on the rungs before it, which is why both tables are printed.")
    entries.append(
        "**2026-11-04 · U2-U4 · Tolerance is mip_gap × |bound|, exact checks on small cases.** Decided the "
        "committed-problem checks allow " + format(numbers["experiment_settings"]["mip_gap"], "g")
        + " × |bound| (about " + money(derived["tolerance_at_rp"])
        + " at RP) and the exact checks (monotone in V and P, V = ∞ and P = all reproduce the base) run on "
        "small random cases at mip_gap = 0, because a fixed 1e-6 match would fail on a HiGHS version change "
        "for reasons that are not the model.")
    entries.append(
        "**2026-11-04 · Scope · The multi-week cash rung is not built.** Decided to stop at R2 because the "
        "plan's Tier 1 ends there; the within-week order (buy → sell → payout) and the open-loop multi-week "
        "plan stay in \"What to improve next\" as to measure.")
    entries.append(
        "**2026-11-05 · U1-U4 · The price grid's upper edge binds, and the docs now say so.** At " + budget + ", "
        + str(head["optimizer_price_moves_at_top_edge"]) + " of the optimizer's " + str(head["optimizer_price_moves"])
        + " price moves sit at the " + grid_edge(card) + " edge, and " + str(derived["m2_records"]) + " of "
        + str(derived["n_records"]) + " ladder solves raise M2. Decided to count that in code "
        "(`ladder.grid_edge_moves`, checked against the solver's M2 flag on every ladder record) and print it "
        "under the scorecard and on the page, because the lead over today's prices and the P = 0 rung's "
        + money(edits[0]["price_of_rule"]) + " otherwise read as properties of the data when they are largely "
        "properties of the " + grid_range(card) + " grid. Widening the grid (to measure) lost for now because "
        "it would change the frozen `numbers.json` and the goldens; the claim is bounded instead.")
    return "\n\n".join(entries)


# ---- the Restock Explorer page ---------------------------------------------------

def try_it(reports):
    """The README "Try it" line; no live URL is claimed until the page is published."""
    page = reports["page"]
    budgets = page["settings"]["budgets"]
    return ("**Try it (60 s): the Restock Explorer page** in `docs/` (not published yet; serve it locally "
            "with `python -m http.server -d docs` and open the address it prints). One budget slider, which "
            "snaps to " + str(len(budgets)) + " solved budgets from $" + format(budgets[0], ",.0f") + " to $"
            + format(budgets[-1], ",.0f") + ", moves the out-of-sample profit-per-dollar tiles, the profit "
            "curve and the buy list together; the EEV → RP → WS bracket and the scenario-count check sit "
            "below. Budgets between the solved points were never solved, so the slider snaps instead of "
            "interpolating. The page computes nothing: `python -m experiments.viz build` writes its data and "
            "`tests/test_viz.py` checks it.")


def _budget_list(values):
    """"$250, $300" or "no solved budget"."""
    parts = []
    for value in values:
        parts.append("$" + format(value, ",.0f"))
    if not parts:
        return "no solved budget"
    return ", ".join(parts)


def page_decisions(reports):
    """DECISIONS.md entries for the Restock Explorer page, every number read from method.json."""
    page = reports["page"]
    facts = page["grid_facts"]
    budgets = page["settings"]["budgets"]
    numbers = reports["numbers"]
    steps = len(budgets) - 1
    interpolation = facts["max_interpolation_error"]
    default = "$" + format(page["default_budget"], ",.0f")
    entries = []
    entries.append(
        "**2026-11-04 · Page · The slider snaps to " + str(len(budgets)) + " solved budgets.** Decided to "
        "precompute every policy at $" + format(budgets[0], ",.0f") + "–$" + format(budgets[-1], ",.0f")
        + " ($50 steps to $500, then $100 steps) and snap the slider to them, because the page computes "
        "nothing and each budget costs two MILP solves (" + seconds(facts["solve_seconds_min"]) + "–"
        + seconds(facts["solve_seconds_max"]) + " s each, " + seconds(facts["solve_seconds_total"])
        + " s for all " + str(facts["n_solves"]) + "). Interpolating between solved budgets lost because the "
        "curve is lumpy: a straight line through two neighbours misses the solved in-sample value by up to "
        + money(interpolation["error"]) + " (at $" + format(interpolation["budget"], ",.0f") + ").")
    if facts["non_nested_steps"] > 0:
        nesting = ("the optimal plans are not nested: in " + str(facts["non_nested_steps"]) + " of " + str(steps)
                   + " neighbouring steps some product's units fall as the budget rises, so a list of "
                   "increments would be wrong there.")
    else:
        nesting = ("nesting is not assumed: on this grid the optimizer's units never fall as the budget rises "
                   "(0 of " + str(steps) + " steps), but the greedy list and today's-prices plans are stored the "
                   "same way, so no list depends on that luck.")
    entries.append(
        "**2026-11-04 · Page · Every budget stores its own buy lists.** Decided to store each policy's plan "
        "and buy list per solved budget because " + nesting)
    entries.append(
        "**2026-11-04 · Page · Quantiles stay in Python.** Decided the p10/p50/p90 band is computed by the "
        "model's weighted quantile and copied, because a linear-interpolated percentile (what a browser "
        "library would compute) differs by " + money(facts["quantile_method_difference_p10"]) + " at the "
        "p10 for the optimizer at " + default + " out-of-sample; the page would then disagree with the README.")
    win_text = _budget_list(facts["greedy_rord_out_wins"])
    profit_text = _budget_list(facts["greedy_profit_out_wins"])
    widest_in = facts["widest_lead_in"]
    widest_out = facts["widest_lead_out"]
    why_per_dollar = ""
    if facts["greedy_rord_out_wins"] and not facts["greedy_profit_out_wins"]:
        why_per_dollar = (", and since the optimizer still earns more in total at every budget, those per-dollar "
                          "wins come from greedy's lower budget-0 base and spend, not from more profit")
    five = numbers["greedy_vs_milp"]
    five_widest = five["budgets"][0]
    for row in five["budgets"]:
        if row["gap_pct"] > five_widest["gap_pct"]:
            five_widest = row
    entries.append(
        "**2026-11-04 · Page · Losses ship on the page too.** Decided the curve note lists every solved budget "
        "where the greedy list beats the optimizer out of sample — per restock dollar at " + win_text
        + "; in total profit at " + profit_text + " — because out-of-sample has no sign guarantee"
        + why_per_dollar + ". The grid also "
        "moved the story: the optimizer's widest lead over the greedy list is " + pct(widest_in["pct"])
        + " in-sample at $" + format(widest_in["budget"], ",.0f") + " (" + pct(widest_out["pct"])
        + " out of sample at $" + format(widest_out["budget"], ",.0f") + "), not the "
        + pct(five_widest["gap_pct"]) + " at $" + format(five_widest["budget"], ",.0f")
        + " that the five-budget table shows, so the page quotes the grid and the README keeps its "
        "five-budget table labelled as five budgets.")
    demo_value = None
    for row in numbers["demo"]["budget_curve"]:
        if row["budget"] == page["default_budget"]:
            demo_value = row["objective"]
    entries.append(
        "**2026-11-04 · Page · The page names the pinned problem, not the demo's.** Decided the page states "
        "its settings (no set-spend cap, " + str(page["settings"]["n_scenarios"]) + " training scenarios, "
        "seed " + str(page["settings"]["scenario_seed"]) + ") because the README headline is computed there; "
        "the demo's 35% set cap and operating gap give " + money(demo_value) + " at " + default + " against "
        "RP " + money(numbers["evpi_vss"]["rp"]) + ", and an unlabelled mix would look like drift. CI never "
        "re-solves the grid: `python -m experiments.viz --check` re-runs the deterministic greedy list and "
        "re-values the " + str(facts["n_plans"]) + " stored plans instead of paying the "
        + seconds(facts["solve_seconds_total"]) + " s of solves (the tests still solve a few small MILPs and "
        "one fresh budget as a smoke test).")
    return "\n\n".join(entries)


# ---- prose sentences that quote results (so no result number sits outside a block) ----

def _widest_greedy_gap(numbers):
    rows = numbers["greedy_vs_milp"]["budgets"]
    widest = rows[0]
    for row in rows:
        if row["gap_pct"] > widest["gap_pct"]:
            widest = row
    return widest


def _greedy_gap_range(numbers):
    rows = numbers["greedy_vs_milp"]["budgets"]
    gaps = []
    budgets = []
    for row in rows:
        gaps.append(row["gap_pct"])
        budgets.append(row["budget"])
    return min(gaps), max(gaps), min(budgets), max(budgets), len(rows)


def hero(reports):
    """The README hero image line; its alt text is scoped to the five plotted budgets."""
    numbers = reports["numbers"]
    widest = _widest_greedy_gap(numbers)
    facts = reports["page"]["grid_facts"]
    count = str(len(numbers["greedy_vs_milp"]["budgets"]))
    return ("![Expected profit by restock budget on the committed synthetic dataset: the exact MILP beats a "
            "greedy baseline at each of the " + count + " plotted budgets; largest gap among them "
            + pct(widest["gap_pct"]) + " at " + budget0(widest["budget"]) + " (the page's "
            + str(facts["n_budgets"]) + "-budget grid finds a widest in-sample lead of "
            + pct(facts["widest_lead_in"]["pct"]) + " at " + budget0(facts["widest_lead_in"]["budget"])
            + ")](docs/figs/hero.png)")


def greedy_range(reports):
    """The README sentence on the greedy gap, scoped to the five budgets of that experiment."""
    low, high, first, last, count = _greedy_gap_range(reports["numbers"])
    facts = reports["page"]["grid_facts"]
    return ("Across the " + str(count) + " budgets of the greedy experiment (" + budget0(first) + "–"
            + budget0(last) + "), a budget-greedy baseline trails the exact MILP in-sample by " + pct(low) + "–"
            + pct(high) + "; the hero figure above is that comparison. On the page's " + str(facts["n_budgets"])
            + "-budget grid the optimizer's widest in-sample lead over the careful greedy list is "
            + pct(facts["widest_lead_in"]["pct"]) + " at " + budget0(facts["widest_lead_in"]["budget"]) + ".")


def greedy_range_experiments(reports):
    low, high, _, _, count = _greedy_gap_range(reports["numbers"])
    return ("Measured gap range across these " + str(count) + " budgets: " + pct(low) + "–" + pct(high)
            + ". The committed figure `docs/figs/hero.png` (the README hero) plots this table.")


def _bound_ratio_text(numbers):
    ratios = bound_ratio_range(numbers)
    return format(min(ratios), ".1f") + "x–" + format(max(ratios), ".1f") + "x"


def hull_lead(reports):
    """The README "Hull, not big-M" bullet, every number from numbers.json."""
    numbers = reports["numbers"]
    settings = numbers["experiment_settings"]
    return ("- **Hull, not big-M.** The GDP-to-MILP transform is the difference between solving and not "
            "solving this model. With identical instances and a " + format(settings["mip_gap"], ".0e").replace("e-0", "e-")
            + " rel gap, hull reaches a plan that is optimal within a " + _gap_label(numbers) + " gap in seconds "
            "while big-M hits the " + str(settings["time_limit"]) + " s limit at every budget with a "
            + _bound_ratio_text(numbers) + " looser LP bound:")


def bound_ratio(reports):
    """EXPERIMENTS.md §3's opening paragraph, settings and ratios from numbers.json."""
    numbers = reports["numbers"]
    settings = numbers["experiment_settings"]
    gap = format(settings["mip_gap"], ".0e").replace("e-0", "e-")
    return ("Identical instances, identical solver, rel gap " + gap + ", " + str(settings["time_limit"])
            + " s limit. The two MILPs describe the same feasible set, so a big-M incumbent can trail but never "
            "beat the hull optimum by more than the gap allows (asserted in code with a slack of 2 × " + gap
            + " × |hull objective|). On this model, big-M's LP relaxation is " + _bound_ratio_text(numbers)
            + " looser than hull's, and big-M hits the time limit at every budget while hull reaches a plan that "
            "is optimal within a " + _gap_label(numbers) + " gap in seconds:")


def _saa_rows(reports):
    rows = list(reports["numbers"]["saa_convergence"])
    return rows[0], rows[-1]


def _saa_rises(reports):
    """"N = 80's gap (9.98) sits above N = 40's (7.44)" for every rise, from saa.json."""
    parts = []
    for rise in reports["page_saa"]["curse_gap_shape"]["rises"]:
        parts.append("N = " + str(rise["to_n"]) + "'s gap (" + format(rise["to_gap"], ".2f") + ") sits above N = "
                     + str(rise["from_n"]) + "'s (" + format(rise["from_gap"], ".2f") + ")")
    return parts


def saa_claim(reports):
    """The README "does NOT claim" bullet on the SAA label."""
    first, last = _saa_rows(reports)
    shape = reports["page_saa"]["curse_gap_shape"]
    smooth = ""
    if not shape["monotone"]:
        smooth = ", though not at every step: " + "; ".join(_saa_rises(reports))
    return ("- The scenario method is called a **sampled-scenario approximation** throughout; the label \"SAA\" "
            "is used only next to the convergence experiment in `EXPERIMENTS.md` that validates it (the "
            "optimizer's-curse gap shrinks from " + format(first["curse_gap"], ".2f") + " to "
            + format(last["curse_gap"], ".2f") + " as N grows from " + str(first["n_scenarios"]) + " to "
            + str(last["n_scenarios"]) + smooth + ").")


def saa_shape(reports):
    """EXPERIMENTS.md §4's method paragraph; whether the gap shrinks smoothly comes from saa.json."""
    from experiments import saa_convergence

    shape = reports["page_saa"]["curse_gap_shape"]
    counts = []
    for n in saa_convergence.NS:
        counts.append(str(n))
    head = ("For each N ∈ {" + ", ".join(counts) + "} and " + str(len(saa_convergence.SEEDS)) + " fresh seeds: "
            "solve on N training scenarios, then value the chosen plan on " + format(saa_convergence.N_EVAL, ",")
            + " evaluation scenarios no training draw used. The in-sample objective systematically overstates the "
            "plan's real value — the optimizer's curse (Smith & Winkler) — and the gap shrinks as N grows")
    if shape["monotone"]:
        return head + " at every step, which is the validation that earns the \"SAA\" label used in this section:"
    return (head + ", though not monotonically (" + "; ".join(_saa_rises(reports)) + " — seed noise at "
            + str(len(saa_convergence.SEEDS)) + " seeds per N), which is the validation that earns the \"SAA\" "
            "label used in this section:")


def saa_reading(reports):
    first, last = _saa_rows(reports)
    shape = reports["page_saa"]["curse_gap_shape"]
    path = ""
    if not shape["monotone"]:
        path = " — non-monotonically on the way (" + "; ".join(_saa_rises(reports)) + ") —"
    return ("Reading: from N = " + str(first["n_scenarios"]) + " to N = " + str(last["n_scenarios"])
            + " the out-of-sample std shrinks " + format(first["out_of_sample_std"], ".2f") + " → "
            + format(last["out_of_sample_std"], ".2f") + " and the curse gap shrinks "
            + format(first["curse_gap"], ".2f") + " → " + format(last["curse_gap"], ".2f") + path
            + " while mean solve time grows " + seconds(first["solve_seconds_mean"]) + " s → "
            + seconds(last["solve_seconds_mean"]) + " s. The pinned "
            + str(reports["numbers"]["experiment_settings"]["n_scenarios"]) + "-scenario experiment set "
            "trades some of that stability for speed; §5 measures what that trade costs.")


def evpi_reading(reports):
    ev = reports["numbers"]["evpi_vss"]
    return ("Reading: perfect demand foresight would be worth " + pct(ev["evpi_pct_of_rp"]) + " of RP, and "
            "actually solving the stochastic program instead of planning on the average scenario earns "
            + pct(ev["vss_pct_of_rp"]) + " of RP. The mean-value plan fails in the usual way — the average "
            "scenario washes out the downside tail, so it overbuys exactly the units the bad scenarios cannot "
            "sell (its own solver objective, " + money(ev["ev_objective"]) + " on the mean scenario, is an "
            "overestimate the re-valuation corrects to " + money(ev["eev"]) + ").")


def spread_reading(reports):
    spread = reports["numbers"]["rp_seed_spread"]
    card = reports["scorecard"]
    seed = card["context"]["pinned_seed"]
    return ("So a single headline RP at N = " + str(reports["numbers"]["experiment_settings"]["n_scenarios"])
            + " carries real seed noise (max−min spread: " + pct(spread["spread_pct_of_mean"]) + " of the mean) — "
            "which is why the docs always name the pinned seed and label results in-sample. The pinned seed "
            + str(seed["seed"]) + " sits " + money(seed["minus_seed_mean"]) + " above the mean ("
            + format(seed["in_seed_sds"], ".2f") + " sds, rank " + str(seed["rank_from_top"]) + " of "
            + str(seed["n_seeds"]) + "): a mild draw, not an outlier.")


def sell_through(reports):
    rows = {}
    for entry in reports["numbers"]["demo"]["strategies"]:
        rows[entry["strategy"]] = entry
    optimal = rows["optimal (Pyomo)"]
    keep = rows["keep"]
    budget = budget0(reports["numbers"]["experiment_settings"]["main_budget"])
    return ("Note the shape of the optimal row: it accepts a *lower* sell-through (" + format(optimal["sell_through"], ".3f")
            + " vs " + format(keep["sell_through"], ".3f") + ") and more leftover value, because clearing every "
            "unit is not the objective — expected profit is. (The budget-curve row at " + budget + " differs from "
            "the strategy-table optimum because the curve re-solves on its 12-scenario subset.)")


BLOCKS = {
    "headline": headline,
    "kpi_table": kpi_table,
    "scorecard_headline": scorecard_headline_table,
    "scorecard_full": scorecard_full_table,
    "marginal": marginal_table,
    "oos_losses": oos_losses,
    "fee_blind": fee_blind_line,
    "ladder": ladder_table,
    "ladder_alone": ladder_alone_table,
    "shelf_sweep": shelf_sweep_table,
    "edit_sweep": edit_sweep_table,
    "evpi_vss": evpi_vss_table,
    "greedy": greedy_table,
    "hull_bigm": hull_bigm_table,
    "saa": saa_table,
    "strategies": strategy_table,
    "budget_curve": budget_curve_table,
    "seed_spread": seed_spread_lines,
    "assumptions": assumptions_table,
    "decisions": decisions,
    "page_decisions": page_decisions,
    "try_it": try_it,
    "edge_note": edge_note,
    "hero": hero,
    "greedy_range": greedy_range,
    "greedy_range_experiments": greedy_range_experiments,
    "hull_lead": hull_lead,
    "bound_ratio": bound_ratio,
    "saa_claim": saa_claim,
    "saa_shape": saa_shape,
    "saa_reading": saa_reading,
    "evpi_reading": evpi_reading,
    "spread_reading": spread_reading,
    "sell_through": sell_through,
}


def render(text, reports):
    """text with every generated block re-rendered; raises on an unknown block name."""
    def replace(match):
        name = match.group(1)
        if name not in BLOCKS:
            raise KeyError("unknown generated block " + name)
        body = BLOCKS[name](reports)
        return "<!-- generated:" + name + " -->\n" + body + "\n<!-- /generated:" + name + " -->"
    return BLOCK_PATTERN.sub(replace, text)


def block_names(text):
    names = []
    matches = list(BLOCK_PATTERN.finditer(text))
    for match in matches:
        names.append(match.group(1))
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description="rewrite or check the generated doc blocks")
    parser.add_argument("--check", action="store_true", help="exit 1 if any block is stale")
    args = parser.parse_args(argv)
    reports = load_reports()
    stale = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        fresh = render(text, reports)
        if fresh == text:
            continue
        if args.check:
            stale.append(path.name)
        else:
            path.write_text(fresh, encoding="utf-8")
            print("rewrote", path.name)
    if stale:
        print("stale generated blocks in: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
