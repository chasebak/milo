"""The constraint ladder: business rules attached to the restock + pricing model.

`solve_with_rules` builds exactly the model `solve_plan` builds (same inputs, same
M1 level filter, same flags), attaches the enabled rules to the GDP model BEFORE the
hull/big-M transform, then solves and reads the plan out with the same helper.
`solve_plan` and its frozen signature (INTERFACES.md) do not change.

Every rule is typed by its KIND, because the kind decides which test fits it:

- "restriction": a rule the plan must obey (shelf space, a price-edit limit, the
  existing set cap and min order, integrality). It can only shrink the feasible set,
  so the true optimum can only fall. The price of the rule is
  obj(looser rung) - obj(this rung) >= 0, checked by `restriction_holds` within a
  tolerance of mip_gap x |bound| (the looser rung's dual bound).
- "accounting": a change in how money is counted (fees in the objective). Feasible
  sets are the same, so "can't raise profit" is the WRONG test; the value of
  modeling it is obj(new plan, new accounting) - value(old plan under new
  accounting) >= 0, the same logic as VSS, checked by `accounting_holds`.

Failure mode: the tolerance is only as good as the solver's reported bound; a solve
that stops on the time limit has a loose bound and the checks pass vacuously, so
every rung also records `optimal` and the M10 flag fires.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from milo.optimize import _add_flag, _build_model, _prepare, _solve_and_read

RUNG_KINDS = ("restriction", "accounting")

# Shelf space per unit, in pack-equivalent facings (one booster pack = 1). These are
# ILLUSTRATIVE design values, not measured from any store; the experiments sweep the
# capacity instead of trusting a real shelf size. Kept in code (not a data file) so
# the committed dataset and its manifest stay frozen.
SPACE_BY_TYPE = {
    "single": 0.5,
    "booster_pack": 1.0,
    "booster_bundle": 4.0,
    "tin": 4.0,
    "collection": 6.0,
    "etb": 8.0,
    "booster_box": 12.0,
}

TODAY_RATIO = 1.00


@dataclass(frozen=True)
class Rules:
    """Business rules; every field None means that rule is off.

    shelf_capacity: V, total space (pack-equivalents) for on-hand plus bought units.
        Needs a `space` column on products (see `space_per_unit`).
    price_edit_limit: P, the most stocked products whose price may move off today's
        level (ratio 1.00) this week.
    """
    shelf_capacity: float | None = None
    price_edit_limit: int | None = None


def space_per_unit(units):
    """Space per unit for each unit id, from its product_type via SPACE_BY_TYPE.

    Raises on an unknown type (a silent default would understate shelf use).
    """
    values = {}
    rows = units[["unit", "product_type"]].itertuples(index=False)
    for row in rows:
        if row.product_type not in SPACE_BY_TYPE:
            raise ValueError("no shelf space default for product_type " + str(row.product_type))
        values[row.unit] = SPACE_BY_TYPE[row.product_type]
    return pd.Series(values, name="space", dtype=float)


def with_space(products, units):
    """A copy of products with the `space` column added from the unit table."""
    space = space_per_unit(units)
    out = products.copy()
    out["space"] = space.reindex(out.index).to_numpy()
    if out["space"].isna().any():
        raise ValueError("some products have no shelf space value")
    return out


def fee_blind_prices(prices):
    """The textbook price table: net cash per unit replaced by the list price.

    A fee-blind plan believes it keeps the whole sticker price. Because
    net_cash <= price on every row, the same plan's own (gross) objective is always
    >= its value at the true fees; the gap is what ignoring fees overstates.
    """
    out = prices.copy()
    out["net_cash"] = out["price"].astype(float)
    return out


def today_levels(prices):
    """Dict unit id -> the level k whose ratio is today's price (1.00)."""
    levels = {}
    rows = prices[["i", "k", "ratio"]].itertuples(index=False)
    for row in rows:
        if abs(float(row.ratio) - TODAY_RATIO) < 1e-9:
            levels[row.i] = int(row.k)
    return levels


def shelf_use(products, buy_units):
    """Recounted shelf use: sum_i space_i * (on_hand_i + x_i), over every product.

    Products the model excluded still sit on the shelf, so their on-hand counts too.
    """
    total = 0.0
    product_ids = list(products.index)
    for i in product_ids:
        stock = float(products.at[i, "on_hand"]) + float(buy_units.get(i, 0))
        total += float(products.at[i, "space"]) * stock
    return total


def count_price_edits(products, prices, buy_units, price_level):
    """Recounted price edits: stocked products whose chosen ratio is not 1.00.

    Rule: an unstocked product (on_hand + x = 0) never counts, whatever level the
    solver left on it; a stocked product whose 1.00 level was filtered out (M1) must
    move, so it counts. A product outside the model (no level at all) keeps today's
    price and does not count. Returns the sorted list of edited unit ids.
    """
    ratio = {}
    rows = prices[["i", "k", "ratio"]].itertuples(index=False)
    for row in rows:
        ratio[(row.i, int(row.k))] = float(row.ratio)
    edited = []
    product_ids = list(products.index)
    for i in product_ids:
        stock = float(products.at[i, "on_hand"]) + float(buy_units.get(i, 0))
        if stock <= 0 or i not in price_level:
            continue
        chosen_ratio = ratio[(i, int(price_level[i]))]
        if abs(chosen_ratio - TODAY_RATIO) > 1e-9:
            edited.append(i)
    return sorted(edited)


def kept_levels(products, prices):
    """Dict unit id -> the levels the M1 filter keeps (net cash > salvage), in grid order.

    The same rule as optimize._filter_levels: level k of product i is kept only if
    n_ik > salvage_frac_i x cost_i. A product with no kept level is absent.
    """
    salvage = {}
    product_ids = list(products.index)
    for i in product_ids:
        salvage[i] = float(products.at[i, "salvage_frac"]) * float(products.at[i, "cost"])
    kept = {}
    rows = prices[["i", "k", "net_cash"]].itertuples(index=False)
    for row in rows:
        if row.i not in salvage:
            continue
        if float(row.net_cash) > salvage[row.i]:
            kept.setdefault(row.i, []).append(int(row.k))
    out = {}
    units = list(kept.keys())
    for i in units:
        out[i] = sorted(kept[i])
    return out


def grid_edge_moves(products, prices, buy_units, price_level):
    """How many recounted price moves sit on the edge of the price grid (the M2/M3 test).

    Counts, over stocked products (on_hand + x > 0) that the model priced:
      moved        the chosen ratio is not 1.00 (count_price_edits);
      moved_top    moved AND the chosen level is the highest level M1 kept (what the
                   solver flags M2: the grid's upper edge, not the data, binds);
      moved_bottom moved AND the chosen level is the lowest kept level (M3);
      m2 / m3      every stocked product at the highest / lowest kept level, moved or
                   not (the solver's M2 / M3 flag count for this plan).
    Failure mode: an optimum on the edge says the true optimum may lie outside the
    grid, so any lift that comes from those moves is bounded by the grid design.
    """
    kept = kept_levels(products, prices)
    edited = set(count_price_edits(products, prices, buy_units, price_level))
    counts = {"moved": len(edited), "moved_top": 0, "moved_bottom": 0, "m2": 0, "m3": 0}
    product_ids = list(products.index)
    for i in product_ids:
        stock = float(products.at[i, "on_hand"]) + float(buy_units.get(i, 0))
        if stock <= 0 or i not in price_level or i not in kept:
            continue
        chosen = int(price_level[i])
        levels = kept[i]
        top = chosen == levels[-1]
        bottom = chosen == levels[0] and not top
        if top:
            counts["m2"] += 1
        if bottom:
            counts["m3"] += 1
        if i in edited and top:
            counts["moved_top"] += 1
        if i in edited and bottom:
            counts["moved_bottom"] += 1
    return counts


def rung_tolerance(mip_gap, bound):
    """Slack allowed by a solve stopped at relative gap mip_gap: mip_gap x max(1, |bound|)."""
    if bound is None or not np.isfinite(bound):
        return float("inf")
    return float(mip_gap) * max(1.0, abs(float(bound)))


def restriction_holds(restricted_objective, looser_objective, looser_bound, mip_gap):
    """A restriction can't raise profit: obj(restricted) <= obj(looser) + mip_gap x |bound|.

    Use only for true restrictions (the rung's feasible set is a subset of the looser
    rung's). Failure mode: a rule that also changes the accounting is not a pure
    restriction and can legitimately fail this.
    """
    tolerance = rung_tolerance(mip_gap, looser_bound)
    return bool(restricted_objective <= looser_objective + tolerance)


def accounting_holds(new_objective, old_plan_new_value, new_bound, mip_gap):
    """Modeling the accounting can't hurt: obj(new) >= value(old plan, new accounting) - tol.

    The old plan is feasible in the new model, so the new optimum is at least its value.
    """
    tolerance = rung_tolerance(mip_gap, new_bound)
    return bool(new_objective >= old_plan_new_value - tolerance)


def model_size(model):
    """(variables, integer variables, active constraints) of a transformed model."""
    n_vars = 0
    n_int = 0
    variables = model.component_data_objects(pyo.Var, descend_into=True)
    for var in variables:
        n_vars += 1
        if var.is_integer() or var.is_binary():
            n_int += 1
    n_cons = 0
    constraints = model.component_data_objects(pyo.Constraint, active=True, descend_into=True)
    for _ in constraints:
        n_cons += 1
    return n_vars, n_int, n_cons


def _attach_shelf(model, products, valid, capacity, flags, report):
    """R1: sum_i v_i (q_i + x_i) <= V as one linear row; no new variables.

    If on-hand stock alone exceeds V, the row would be infeasible; instead a SHELF flag
    is raised and buys are capped at zero shelf (only space-free products could be
    bought), never a silent infeasible solve.
    """
    if "space" not in products.columns:
        raise ValueError("shelf rule needs a `space` column on products (see with_space)")
    on_hand_use = 0.0
    product_ids = list(products.index)
    for i in product_ids:
        on_hand_use += float(products.at[i, "space"]) * float(products.at[i, "on_hand"])
    room = float(capacity) - on_hand_use
    if room < 0:
        _add_flag(flags, None, "SHELF",
                  f"on-hand stock alone uses {on_hand_use:.2f} > shelf capacity "
                  f"{float(capacity):.2f}; no shelf-using buys allowed")
        room = 0.0
    buy_use = 0
    included = list(valid.keys())
    for i in included:
        buy_use = buy_use + float(products.at[i, "space"]) * model.x[i]
    model.shelf = pyo.Constraint(expr=buy_use <= room)
    report["shelf"] = {"capacity": float(capacity), "on_hand_use": on_hand_use}


def _attach_price_edits(model, products, prices, valid, limit, flags, report):
    """R2: at most P stocked products move off today's price (the AND trick).

    e_i in {0, 1}; y_now = the binary of product i's ratio-1.00 price disjunct (0 if
    M1 dropped that level); o_i = the binary of its order-yes disjunct.
    Stocked now (q_i > 0): e_i >= 1 - y_now.
    Not stocked now:       e_i >= (1 - y_now) + o_i - 1  (edit only if it is bought AND moved).
    Never orderable and not stocked: e_i = 0.   sum_i e_i <= P.
    Products whose 1.00 level was dropped while stocked are FORCED edits; if they
    outnumber P the limit is raised to that count and an EDITS flag says so.
    """
    now = today_levels(prices)
    order_ids = set(model.J)
    edit_ids = []
    forced = []
    included = list(valid.keys())
    for i in included:
        on_hand = float(products.at[i, "on_hand"])
        if on_hand > 0 or i in order_ids:
            edit_ids.append(i)
            if on_hand > 0 and now.get(i) not in valid[i]:
                forced.append(i)
    model.E = pyo.Set(initialize=edit_ids, ordered=True)
    model.edit = pyo.Var(model.E, within=pyo.Binary)
    model.edit_link = pyo.ConstraintList()
    for i in edit_ids:
        k_now = now.get(i)
        if k_now in valid[i]:
            moved = 1 - model.prod[i].price[k_now].binary_indicator_var
        else:
            moved = 1
        if float(products.at[i, "on_hand"]) > 0:
            model.edit_link.add(model.edit[i] >= moved)
        else:
            ordered = model.order_yes[i].binary_indicator_var
            model.edit_link.add(model.edit[i] >= moved + ordered - 1)
    effective = int(limit)
    if len(forced) > effective:
        _add_flag(flags, None, "EDITS",
                  f"{len(forced)} stocked products lost their 1.00 level (M1) and must move; "
                  f"edit limit {int(limit)} raised to {len(forced)}")
        effective = len(forced)
    total = 0
    for i in edit_ids:
        total = total + model.edit[i]
    model.edit_limit = pyo.Constraint(expr=total <= effective)
    report["price_edits"] = {"limit": int(limit), "effective_limit": effective,
                             "forced": sorted(forced)}


def build_with_rules(products, prices, demand, rules=None, probs=None, budget=0.0,
                     set_cap_share=1.0, transform="hull"):
    """The transformed model with the rules attached, before any solve.

    Returns (model, setup, rule_report, build_seconds). With rules=None or Rules() this
    is exactly the model solve_plan builds.
    """
    if rules is None:
        rules = Rules()
    setup = _prepare(products, prices, demand, probs, budget, transform)
    flags = setup["flags"]
    report = {}
    t0 = time.perf_counter()
    model = _build_model(products, setup["demand"], setup["probs"], setup["levels"],
                         setup["valid"], setup["net_cash"], budget, set_cap_share, flags)
    if rules.shelf_capacity is not None:
        _attach_shelf(model, products, setup["valid"], rules.shelf_capacity, flags, report)
    if rules.price_edit_limit is not None:
        _attach_price_edits(model, products, prices, setup["valid"], rules.price_edit_limit,
                            flags, report)
    pyo.TransformationFactory("gdp." + transform).apply_to(model)
    build_seconds = time.perf_counter() - t0
    return model, setup, report, build_seconds


def solve_with_rules(products, prices, demand, rules=None, probs=None, budget=0.0,
                     set_cap_share=1.0, transform="hull", time_limit=120, mip_gap=0.005,
                     lp_relaxation=False):
    """solve_plan plus business rules; same result keys plus the extras below.

    Extra keys: objective_bound (the solver's dual bound), rule_report (per rule: the
    limit, the use recounted independently from the plan, slack and binds), n_variables,
    n_integer_variables, n_constraints. With rules=None or Rules() the model is
    identical to solve_plan's (same counts, same objective; tested).
    """
    model, setup, report, build_seconds = build_with_rules(
        products, prices, demand, rules=rules, probs=probs, budget=budget,
        set_cap_share=set_cap_share, transform=transform)
    n_vars, n_int, n_cons = model_size(model)

    result, res = _solve_and_read(model, products, setup, build_seconds, time_limit,
                                  mip_gap, lp_relaxation)
    bound = res.objective_bound
    result["objective_bound"] = float(bound) if bound is not None else float("nan")
    result["n_variables"] = n_vars
    result["n_integer_variables"] = n_int
    result["n_constraints"] = n_cons

    buy_units = {}
    buy_rows = list(result["buy"].itertuples(index=False))
    for row in buy_rows:
        buy_units[row.i] = int(row.units)
    price_level = {}
    price_rows = list(result["prices"].itertuples(index=False))
    for row in price_rows:
        if row.level is not None and not pd.isna(row.level):
            price_level[row.i] = int(row.level)
    if "shelf" in report:
        used = shelf_use(products, buy_units)
        slack = report["shelf"]["capacity"] - used
        smallest = float(products["space"].min())
        report["shelf"]["used"] = used
        report["shelf"]["slack"] = slack
        report["shelf"]["binds"] = bool(slack < smallest)
    if "price_edits" in report:
        edited = count_price_edits(products, prices, buy_units, price_level)
        report["price_edits"]["used"] = len(edited)
        report["price_edits"]["edited"] = edited
        report["price_edits"]["binds"] = bool(len(edited) >= report["price_edits"]["effective_limit"])
    result["rule_report"] = report
    return result
