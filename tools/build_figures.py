"""Build docs/figs/hero.png from reports/numbers.json and print every generated doc block.

The docs (README.md, EXPERIMENTS.md) carry no hand-typed numbers: every table and
headline value sits in a generated block that tools/render_docs.py rewrites from the
committed reports. Rerun after regenerating the reports:

    python tools/build_figures.py      # the hero figure (+ prints the blocks)
    python tools/render_docs.py        # rewrite the blocks in README.md / EXPERIMENTS.md
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_docs  # noqa: E402

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
    worst_label = ("largest gap of these " + str(len(budgets)) + " budgets: " + pct(gaps_pct[worst_index])
                   + " at \\$" + format(worst_x, ",.0f"))
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


def main():
    numbers = load_numbers()
    hero_path = hero_figure(numbers, FIGS / "hero.png")
    print("wrote", hero_path.relative_to(PROJECT))
    # Every doc table is rendered by tools/render_docs.py (one copy of each renderer);
    # print them here too so a rerun shows what the docs will carry.
    reports = render_docs.load_reports()
    names = sorted(render_docs.BLOCKS.keys())
    for name in names:
        print("\n== " + name + " ==")
        print(render_docs.BLOCKS[name](reports))


if __name__ == "__main__":
    main()
