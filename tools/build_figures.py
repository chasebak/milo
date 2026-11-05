"""Build docs/figs/hero.png and print every doc table/number from reports/numbers.json.

The docs (README.md, EXPERIMENTS.md) carry no hand-typed numbers: every table and
headline value they show is printed by this script from reports/numbers.json and
pasted verbatim. Rerun after regenerating numbers.json:

    python tools/build_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
NUMBERS = PROJECT / "reports" / "numbers.json"
FIGS = PROJECT / "docs" / "figs"

MILP_COLOR = "#2a78d6"    # blue, same pair as reports/saa_convergence.png
GREEDY_COLOR = "#eb6834"  # orange
INK = "#333333"
MUTED = "#707070"


def load_numbers():
    with open(NUMBERS, encoding="utf-8") as handle:
        return json.load(handle)


def money(value):
    return format(value, ",.2f")


def pct(value):
    return format(value, ".2f") + "%"


def hero_figure(numbers, png_path):
    """Profit vs budget: exact MILP (hull) vs the greedy baseline, from greedy_vs_milp."""
    rows = numbers["greedy_vs_milp"]["budgets"]
    budgets = []
    milp_values = []
    greedy_values = []
    gaps_pct = []
    for row in rows:
        budgets.append(row["budget"])
        milp_values.append(row["milp_objective"])
        greedy_values.append(row["greedy_objective"])
        gaps_pct.append(row["gap_pct"])

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=110)
    ax.plot(budgets, milp_values, color=MILP_COLOR, linewidth=2, marker="o",
            markersize=6, zorder=3, label="exact MILP")
    ax.plot(budgets, greedy_values, color=GREEDY_COLOR, linewidth=2, marker="o",
            markersize=6, zorder=3, label="greedy baseline")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    label_index = len(budgets) - 2
    label_budget = budgets[label_index]
    ax.annotate("exact MILP (Pyomo GDP hull + HiGHS)",
                xy=(label_budget, milp_values[label_index]), xytext=(-8, 14),
                textcoords="offset points", color=INK, fontsize=10, ha="center")
    ax.annotate("greedy baseline",
                xy=(label_budget, greedy_values[label_index]), xytext=(8, -20),
                textcoords="offset points", color=INK, fontsize=10, ha="center")

    worst_index = 0
    position = 0
    for gap in gaps_pct:
        if gap > gaps_pct[worst_index]:
            worst_index = position
        position += 1
    worst_x = budgets[worst_index]
    worst_y = greedy_values[worst_index]
    worst_label = ("largest measured gap: " + pct(gaps_pct[worst_index])
                   + " at budget " + format(worst_x, ",.0f"))
    ax.annotate(worst_label, xy=(worst_x, worst_y), xytext=(14, -4),
                textcoords="offset points", color=MUTED, fontsize=9, va="top")

    ax.set_xlabel("restock budget ($)", color=INK)
    ax.set_ylabel("expected profit ($)", color=INK)
    ax.set_title("Joint restock + pricing: exact MILP vs greedy baseline",
                 color=INK, fontsize=12)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    fig.text(0.01, 0.01,
             "committed synthetic dataset; in-sample, conditional on the pinned 24-scenario set",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


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


def evpi_vss_table(numbers):
    ev = numbers["evpi_vss"]
    rows = []
    rows.append(["RP (two-stage stochastic optimum)", money(ev["rp"])])
    rows.append(["EEV (mean-value plan, re-valued on the scenarios)", money(ev["eev"])])
    rows.append(["WS (wait-and-see bound)", money(ev["ws"])])
    rows.append(["EVPI = WS − RP", money(ev["evpi"]) + " (" + pct(ev["evpi_pct_of_rp"]) + " of RP)"])
    rows.append(["VSS = RP − EEV", money(ev["vss"]) + " (" + pct(ev["vss_pct_of_rp"]) + " of RP)"])
    return markdown_table(["quantity (expected profit, $)", "value"], rows)


def greedy_table(numbers):
    rows = []
    gap_values = []
    for entry in numbers["greedy_vs_milp"]["budgets"]:
        gap_values.append(entry["gap_pct"])
        rows.append([money(entry["budget"]), money(entry["milp_objective"]),
                     money(entry["greedy_objective"]), money(entry["gap"]),
                     pct(entry["gap_pct"])])
    table = markdown_table(["budget ($)", "exact MILP ($)", "greedy ($)", "gap ($)", "gap (% of MILP)"], rows)
    gap_range = pct(min(gap_values)) + "–" + pct(max(gap_values))
    return table, gap_range


def hull_bigm_table(numbers):
    # "optimal" in numbers.json means HiGHS stopped because the relative MIP gap reached
    # the experiment tolerance, not that it closed the gap to zero. The label says how
    # close, read from the pinned setting so it cannot drift from the solves themselves.
    gap_percent = numbers["experiment_settings"]["mip_gap"] * 100.0
    within_gap = "optimal within a " + pct(gap_percent) + " gap"
    rows = []
    bound_ratios = []
    hull_bounds = {}
    for entry in numbers["hull_vs_bigm"]:
        if entry["transform"] == "hull":
            hull_bounds[entry["budget"]] = entry["lp_relaxation_objective"]
    for entry in numbers["hull_vs_bigm"]:
        if entry["transform"] == "bigm":
            ratio = entry["lp_relaxation_objective"] / hull_bounds[entry["budget"]]
            bound_ratios.append(ratio)
        status = within_gap if entry["optimal"] else "time limit (120 s)"
        rows.append([money(entry["budget"]), entry["transform"], money(entry["objective"]),
                     money(entry["lp_relaxation_objective"]),
                     format(entry["build_seconds"], ".2f"), format(entry["solve_seconds"], ".1f"),
                     status])
    table = markdown_table(["budget ($)", "transform", "incumbent ($)", "LP bound ($)",
                            "build (s)", "solve (s)", "result"], rows)
    ratio_range = format(min(bound_ratios), ".1f") + "x–" + format(max(bound_ratios), ".1f") + "x"
    return table, ratio_range


def saa_table(numbers):
    rows = []
    for entry in numbers["saa_convergence"]:
        rows.append([str(entry["n_scenarios"]), money(entry["in_sample_mean"]),
                     format(entry["in_sample_std"], ".2f"), money(entry["out_of_sample_mean"]),
                     format(entry["out_of_sample_std"], ".2f"), format(entry["curse_gap"], ".2f"),
                     format(entry["solve_seconds_mean"], ".1f")])
    header = ["N (training scenarios)", "in-sample mean ($)", "in-sample std",
              "out-of-sample mean ($)", "out-of-sample std", "curse gap ($)", "solve (s, mean)"]
    return markdown_table(header, rows)


def strategy_table(numbers):
    rows = []
    for entry in numbers["demo"]["strategies"]:
        rows.append([entry["strategy"], money(entry["expected"]), money(entry["p10"]),
                     money(entry["p90"]), format(entry["sell_through"], ".3f"),
                     money(entry["leftover_value"])])
    header = ["pricing strategy (same buy list)", "expected ($)", "p10 ($)", "p90 ($)",
              "sell-through", "leftover value ($)"]
    return markdown_table(header, rows)


def budget_curve_table(numbers):
    rows = []
    for entry in numbers["demo"]["budget_curve"]:
        rows.append([money(entry["budget"]), money(entry["objective"]), money(entry["spend"])])
    return markdown_table(["budget ($)", "expected profit ($)", "spend ($)"], rows)


def seed_spread_lines(numbers):
    spread = numbers["rp_seed_spread"]
    values = []
    for value in spread["rp_values"]:
        values.append(money(value))
    lines = []
    lines.append("RP across scenario-set seeds " + str(spread["seeds"]) + ": " + ", ".join(values))
    lines.append("mean " + money(spread["mean"]) + ", std " + format(spread["std"], ".2f")
                 + ", spread " + pct(spread["spread_pct_of_mean"]) + " of the mean")
    return "\n".join(lines)


def main():
    numbers = load_numbers()
    hero_path = hero_figure(numbers, FIGS / "hero.png")
    print("wrote", hero_path.relative_to(PROJECT))

    print("\n== EVPI / VSS table ==")
    print(evpi_vss_table(numbers))
    greedy_md, gap_range = greedy_table(numbers)
    print("\n== greedy vs MILP table ==")
    print(greedy_md)
    print("greedy gap range across budgets: " + gap_range)
    hull_md, ratio_range = hull_bigm_table(numbers)
    print("\n== hull vs bigm table ==")
    print(hull_md)
    print("big-M LP bound looser than hull by: " + ratio_range)
    print("\n== SAA convergence table ==")
    print(saa_table(numbers))
    print("\n== demo strategy table ==")
    print(strategy_table(numbers))
    print("\n== demo budget curve ==")
    print(budget_curve_table(numbers))
    print("\n== RP seed spread ==")
    print(seed_spread_lines(numbers))
    ev = numbers["evpi_vss"]
    print("\n== headline ==")
    print("RP " + money(ev["rp"]) + " | EEV " + money(ev["eev"]) + " | WS " + money(ev["ws"])
          + " | EVPI " + money(ev["evpi"]) + " (" + pct(ev["evpi_pct_of_rp"]) + " of RP)"
          + " | VSS " + money(ev["vss"]) + " (" + pct(ev["vss_pct_of_rp"]) + " of RP)")
    print("EV solver objective on the mean scenario: " + money(ev["ev_objective"]))
    demo = numbers["demo"]
    print("demo plan objective " + money(demo["plan_objective"]))


if __name__ == "__main__":
    main()
