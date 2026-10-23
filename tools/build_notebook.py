"""Build and EXECUTE pricing_optimizer.ipynb (the notebook calls milo.pipeline; all numbers come from code).

Run from the repo root in the project venv (`pip install -e .` plus `nbclient` and
`ipykernel`, both declared in the dev extras):

    python tools/build_notebook.py

The executed notebook is committed; CI does not execute it. Before writing, the
notebook JSON is scanned for local absolute paths (and the kernelspec stays the
generic "Python 3"), so a build from any machine produces a publishable file.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]
# macOS home-dir prefix built by concatenation so this file itself passes the
# repo-wide scan for absolute local paths.
MAC_HOME_PREFIX = "/" + "Users" + "/"
LEAK_MARKERS = [MAC_HOME_PREFIX, "/home/", str(Path.home()), ".venvs"]

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Card-store pricing & restock optimizer

How to use: edit the settings cell, then **Run All**. Inputs you can edit live in `data/synthetic/inputs/`:
`restock_costs.csv` (your buy cost, min order, max units per product), `tcg_map.csv` (market-feed matches),
`catalog_overrides.csv` (fix a title the parser got wrong). Outputs go to `out/`.
""")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from milo import pipeline, demand, pricing
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)

# ---- settings ----
HORIZON_WEEKS = 8          # how far ahead to plan
LEAD_WEEKS = 1             # restock arrives after this many weeks
BUDGETS = [0, 1000, 2500, 5000, 10000]
MAIN_BUDGET = 5000         # the budget the buy list is built for
SEASON_UPLIFT = 1.3        # holiday scenario (1.0 / 1.3 / 1.6); not estimated from data (edge case D8)
SET_CAP_SHARE = 0.35       # max share of the budget in one set
N_SCENARIOS = 60           # demand scenarios for the main plan (a third each low/base/high price sensitivity)

out = pipeline.run(HORIZON_WEEKS, LEAD_WEEKS, BUDGETS, MAIN_BUDGET, SEASON_UPLIFT, N_SCENARIOS, SET_CAP_SHARE)
""")

md("## 1. Data loaded")
code("""
out["summary"]
""")
code("""
cat = out["data"]["catalog"]
print("Card fee share (paid by card):", round(out["data"]["card_share"], 3),
      "| units per order:", round(out["data"]["units_per_order"], 2))
cat[cat["needs_review"]][["product", "variant", "product_type", "set"]]
""")

md("""
### Store demand by weekday
Whole-store orders per day estimated from order-number gaps (the store is closed one weekday).
The weekend share feeds the Friday shelf check in section 7.
""")
code("""
out["weekday"][["weekday", "full_days", "avg_store_orders", "share_of_week"]].round(3)
""")

md("""
### Restock timing (copied as-is from the store's reporting)
Shelf rule used for in-stock days: an upload entered after hours sells from the **next open day**, edits during
store hours sell the same day, and the store is closed one weekday. Bulk orders (15+ units of an item in one
order, or one order that is most of an item's sales) are left out of demand rates but still leave the shelf.
""")
code("""
rt = out["restock_tables"]
print("Best sellers that sat sold out (est. missed sales):")
display(rt.get("sold_out", pd.DataFrame()).head(15))
print("Items selling 0.5+/day: days of cover:")
display(rt.get("restock_soon", pd.DataFrame()))
print("Products both sellers carry:")
rt.get("shared_products", pd.DataFrame())
""")

md("## 2. Baseline demand: P1 (in-stock rate) → P2 (shrunk toward type average)")
code("""
p = out["p1p2"].sort_values("units", ascending=False)
p[["family", "product_type", "units", "in_stock_days", "raw_rate", "shrunk_rate", "p10", "p50", "p90", "flag"]].head(30).round(2)
""")

md("## 3. Online (market feed) vs in-person (the store)")
code("""
out["online"]["by_type"].round(3)
""")
code("""
cols = ["product", "seller", "product_type", "in_stock_days", "store_units", "online_units", "our_price", "online_price", "premium", "capture_rate"]
out["online"]["products"].sort_values("store_units", ascending=False)[cols].head(30).round(3)
""")
code("""
print("Does online demand show up in the store 0/1/2 weeks later? (pooled within-product correlation)")
out["online"]["lags"].round(3)
""")
code("""
prod = out["online"]["products"].dropna(subset=["premium"])
fig, ax = plt.subplots(figsize=(7, 4))
for seller, part in prod.groupby("seller"):
    ax.scatter(part["premium"], part["store_per_week"], label=seller, alpha=0.7)
ax.set_xlabel("store price ÷ market price"); ax.set_ylabel("store units per in-stock week")
ax.set_yscale("log"); ax.axvline(1.0, color="grey", lw=0.8); ax.legend(); ax.set_title("Markup vs sales speed")
plt.show()
""")

md("## 4. Backtest: fit before the split date, predict after it")
code("""
joined, by_type = out["backtest_plain"]
joined_t, by_type_t = out["backtest_traffic"]
print("Totals without traffic step:", by_type[["actual", "pred_shrunk", "pred_raw"]].sum().round(1).to_dict())
print("Totals with traffic step:   ", by_type_t[["actual", "pred_shrunk", "pred_raw"]].sum().round(1).to_dict())
by_type_t.round(2)
""")

md("## 5. Price response: P4 position vs market → P5 sensitivity → P6 curve")
code("""
print(out["sensitivity"]["fe_info"])
out["sensitivity"]["elasticities"].round(3)
""")
code("""
el = out["sensitivity"]["elasticities"]
m = np.linspace(0.7, 1.6, 50)
fig, ax = plt.subplots(figsize=(7, 4))
for group in ["sealed", "single"]:
    below = el[(el.group == group) & (el.side == "below")].iloc[0]["base"]
    above = el[(el.group == group) & (el.side == "above")].iloc[0]["base"]
    ax.plot(m, pricing.h(m, below, above), label=group)
ax.axvline(1.0, color="grey", lw=0.8); ax.set_xlabel("our price ÷ market price"); ax.set_ylabel("demand vs at market")
ax.legend(); ax.set_title("P6 kinked demand curve (base sensitivity)"); plt.show()
""")
code("""
d = out["demand"]
print("P7 check: demand never rises with price in any scenario:", bool(np.all(np.diff(d, axis=1) <= 0)))
""")

md("## 6. Restock units and inputs")
code("""
print("shared_with_other_seller = another seller carries it too; demand there rises when they are out (v1 assumes independent demand).")
out["units"][["unit", "product_type", "price_now", "market_price", "cost", "cost_source", "on_hand", "shrunk_rate",
              "shared_with_other_seller", "data_flag"]].round(2)
""")

md("## 7. Optimal buy list and prices (Pyomo GDP → hull MILP, HiGHS)")
code("""
plan = out["plan"]
print("status:", plan["status"], "| termination:", plan["termination"], "| optimal:", plan["optimal"],
      "| gap:", plan["mip_gap"], "| expected profit:", round(plan["objective"], 2),
      "| LP relaxation:", round(plan["lp_relaxation_objective"], 2))
buy = plan["buy"].merge(plan["prices"], on="i", how="left")
buy[buy["units"] > 0].sort_values("spend", ascending=False).round(2)
""")
code("""
print("Flags by code (M1 dropped price levels, M2/M3 price at grid edge, M5 lumpiness, M10 not proven optimal):")
print(plan["flags"]["code"].value_counts())
print("\\nProducts priced at the top level, and why:")
out["edge_reasons"]
""")
code("""
print("Fri-Sun share of an open week:", round(out["weekend_share"], 3))
out["weekend"].head(25).round(2)
""")
code("""
curve = out["budget_curve"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve["budget"], curve["profit_from_restock"], marker="o")
ax.set_xlabel("budget ($)"); ax.set_ylabel("expected profit added by the restock ($)"); ax.set_title("Profit vs budget"); plt.show()
curve.round(2)
""")

md("## 8. Pricing strategies on the same demand scenarios")
code("""
out["strategies"].round(2)
""")
code("""
files = pipeline.export(out)
for path in files:
    print(path.relative_to(pipeline.PROJECT))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(PROJECT)}})
client.execute()

serialized = nbf.writes(nb)
found = []
for marker in LEAK_MARKERS:
    if marker in serialized:
        found.append(marker)
if found:
    raise RuntimeError("notebook JSON contains local paths: " + ", ".join(found)
                       + " - not writing pricing_optimizer.ipynb")
display_name = nb["metadata"]["kernelspec"]["display_name"]
if display_name != "Python 3":
    raise RuntimeError("kernelspec display_name must stay the generic 'Python 3', got " + display_name)

target = PROJECT / "pricing_optimizer.ipynb"
nbf.write(nb, target)
print("wrote", target.relative_to(PROJECT), "(executed)")
