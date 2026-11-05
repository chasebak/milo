"""Two-stage stochastic restock + pricing MILP, written as a GDP (pyomo.gdp).

Stage 1: units to buy x_i and one price level per product.
Stage 2 (per scenario ω): units sold s, leftovers L, net revenue R.
The GDP is turned into a MILP with gdp.hull (default) or gdp.bigm and
solved with HiGHS via pyomo.contrib.solver ('highs').
"""

import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.gdp import Disjunct, Disjunction

FLAG_COLUMNS = ["i", "code", "message"]


def _add_flag(flags, i, code, message):
    flags.append({"i": i, "code": code, "message": message})


def _check_inputs(products, prices, demand, probs):
    """Validate shapes, probabilities and demand ordering; return levels, probs."""
    needed = ["cost", "on_hand", "min_order", "max_units", "set_code", "salvage_frac"]
    for col in needed:
        if col not in products.columns:
            raise ValueError(f"products is missing column {col}")
    for col in ["i", "k", "ratio", "price", "net_cash"]:
        if col not in prices.columns:
            raise ValueError(f"prices is missing column {col}")

    levels = sorted(prices["k"].astype(int).unique())
    n_products = len(products.index)
    if demand.ndim != 3:
        raise ValueError("demand must have shape (n_products, n_levels, n_scenarios)")
    if demand.shape[0] != n_products or demand.shape[1] != len(levels):
        raise ValueError(
            f"demand shape {demand.shape} does not match "
            f"({n_products}, {len(levels)}, n_scenarios)"
        )
    if np.any(demand < 0):
        raise ValueError("demand must be non-negative")
    steps = np.diff(demand, axis=1)
    if np.any(steps > 1e-9):
        raise ValueError("demand must be non-increasing in price level k")

    n_scen = demand.shape[2]
    if probs is None:
        probs = np.full(n_scen, 1.0 / n_scen)
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (n_scen,):
        raise ValueError("probs must have one entry per scenario")
    if np.any(probs < 0) or abs(probs.sum() - 1.0) > 1e-6:
        raise ValueError("probs must be non-negative and sum to 1")
    return levels, probs


def _price_tables(prices):
    """Dicts (i, k) -> net_cash, price, ratio."""
    net_cash = {}
    price = {}
    ratio = {}
    rows = prices[["i", "k", "ratio", "price", "net_cash"]].itertuples(index=False)
    for row in rows:
        key = (row.i, int(row.k))
        net_cash[key] = float(row.net_cash)
        price[key] = float(row.price)
        ratio[key] = float(row.ratio)
    return net_cash, price, ratio


def _filter_levels(products, levels, net_cash, flags):
    """M1: keep level k only if n_ik > ρ_i·c_i. Returns dict i -> kept levels."""
    valid = {}
    product_ids = list(products.index)
    for i in product_ids:
        salvage = float(products.at[i, "salvage_frac"]) * float(products.at[i, "cost"])
        kept = []
        for k in levels:
            key = (i, k)
            if key not in net_cash:
                continue
            n = net_cash[key]
            if n > salvage:
                kept.append(k)
            else:
                msg = (f"level {k} dropped: net cash {n:.2f} <= salvage "
                       f"{salvage:.2f} (model would refuse to sell)")
                _add_flag(flags, i, "M1", msg)
        if len(kept) == 0:
            _add_flag(flags, i, "M1", "all price levels dropped; product excluded (x=0)")
        else:
            valid[i] = kept
    return valid


def _build_model(products, demand, probs, levels, valid, net_cash,
                 budget, set_cap_share, flags):
    """Build the GDP model (before transformation)."""
    product_ids = list(products.index)
    level_pos = {}
    level_list = list(enumerate(levels))
    for pos, k in level_list:
        level_pos[k] = pos
    row_pos = {}
    product_rows = list(enumerate(product_ids))
    for pos, i in product_rows:
        row_pos[i] = pos

    included = list(valid.keys())
    n_scen = demand.shape[2]
    scenarios = list(range(n_scen))

    # Per-product bounds (these are also the big-M values for gdp.bigm).
    x_ub = {}
    s_ub = {}
    r_ub = {}
    for i in included:
        units_max = int(products.at[i, "max_units"])
        on_hand = float(products.at[i, "on_hand"])
        x_ub[i] = max(units_max, 0)
        kept_pos = []
        for k in valid[i]:
            kept_pos.append(level_pos[k])
        max_demand = float(demand[row_pos[i], kept_pos, :].max())
        s_ub[i] = min(max_demand, on_hand + x_ub[i])
        best_net = 0.0
        for k in valid[i]:
            best_net = max(best_net, net_cash[(i, k)])
        r_ub[i] = best_net * s_ub[i]

    m = pyo.ConcreteModel("restock_pricing")
    m.I = pyo.Set(initialize=included, ordered=True)
    m.W = pyo.Set(initialize=scenarios, ordered=True)

    def x_bounds(model, i):
        return (0, x_ub[i])

    m.x = pyo.Var(m.I, within=pyo.NonNegativeIntegers, bounds=x_bounds)

    # One block per product holds its scenario vars and its price disjunction.
    # (Small per-product indexed vars keep pyomo's hull transform fast.)
    def product_block(b, i):
        on_hand = float(products.at[i, "on_hand"])
        b.K = pyo.Set(initialize=valid[i], ordered=True)
        b.s = pyo.Var(m.W, within=pyo.NonNegativeReals, bounds=(0, s_ub[i]))
        b.L = pyo.Var(m.W, within=pyo.NonNegativeReals, bounds=(0, on_hand + x_ub[i]))
        b.R = pyo.Var(m.W, within=pyo.NonNegativeReals, bounds=(0, r_ub[i]))

        # Disjunction 1: exactly one price level for this product.
        def price_disjunct(d, k):
            n = net_cash[(i, k)]
            d_row = demand[row_pos[i], level_pos[k], :]
            d.cap = pyo.ConstraintList()
            d.rev = pyo.ConstraintList()
            for w in scenarios:
                d.cap.add(b.s[w] <= float(d_row[w]))
                d.rev.add(b.R[w] == n * b.s[w])

        b.price = Disjunct(b.K, rule=price_disjunct)
        options = []
        for k in valid[i]:
            options.append(b.price[k])
        b.pick = Disjunction(expr=options, xor=True)

    m.prod = pyo.Block(m.I, rule=product_block)

    # Disjunction 2: order (m_i <= x_i <= U_i) or not (x_i = 0).
    order_ids = []
    for i in included:
        min_order = int(products.at[i, "min_order"])
        if x_ub[i] < min_order:
            m.x[i].fix(0)
            _add_flag(flags, i, "MIN_ORDER",
                      f"max_units {x_ub[i]} < min_order {min_order}; x fixed to 0")
        else:
            order_ids.append(i)
    m.J = pyo.Set(initialize=order_ids, ordered=True)

    def order_yes(d, i):
        model = d.model()
        d.lo = pyo.Constraint(expr=model.x[i] >= int(products.at[i, "min_order"]))
        d.hi = pyo.Constraint(expr=model.x[i] <= x_ub[i])

    def order_no(d, i):
        model = d.model()
        d.zero = pyo.Constraint(expr=model.x[i] == 0)

    m.order_yes = Disjunct(m.J, rule=order_yes)
    m.order_no = Disjunct(m.J, rule=order_no)

    def order_choice(model, i):
        return [model.order_yes[i], model.order_no[i]]

    m.order = Disjunction(m.J, rule=order_choice, xor=True)

    # Ordinary constraints.
    spend = 0
    for i in included:
        spend = spend + float(products.at[i, "cost"]) * m.x[i]
    m.spend = pyo.Expression(expr=spend)
    m.budget = pyo.Constraint(expr=m.spend <= float(budget))

    m.set_cap = pyo.ConstraintList()
    if set_cap_share < 1.0:
        set_codes = list(products.loc[included, "set_code"].unique())
        for g in set_codes:
            set_spend = 0
            for i in included:
                if products.at[i, "set_code"] == g:
                    set_spend = set_spend + float(products.at[i, "cost"]) * m.x[i]
            m.set_cap.add(set_spend <= float(set_cap_share) * float(budget))

    m.balance = pyo.ConstraintList()
    for i in included:
        on_hand = float(products.at[i, "on_hand"])
        for w in scenarios:
            m.balance.add(m.prod[i].s[w] + m.prod[i].L[w] == on_hand + m.x[i])

    # Objective: expected (net revenue + salvage of leftovers) minus buy cost.
    expected = 0
    for w in scenarios:
        pi = float(probs[w])
        for i in included:
            salvage = float(products.at[i, "salvage_frac"]) * float(products.at[i, "cost"])
            expected = expected + pi * (m.prod[i].R[w] + salvage * m.prod[i].L[w])
    m.profit = pyo.Objective(expr=expected - m.spend, sense=pyo.maximize)
    return m


def _run_highs(model, time_limit, mip_gap):
    """Solve with HiGHS; never raise on a non-optimal result."""
    opt = SolverFactory("highs")
    res = opt.solve(
        model,
        time_limit=time_limit,
        rel_gap=mip_gap,
        load_solutions=False,
        raise_exception_on_nonoptimal_result=False,
    )
    has_solution = res.solution_status in (SolutionStatus.feasible, SolutionStatus.optimal)
    if has_solution:
        res.solution_loader.load_vars()
    return res, has_solution


def _gap(res):
    """Relative MIP gap from incumbent and bound (nan if unknown)."""
    inc = res.incumbent_objective
    bound = res.objective_bound
    if inc is None or bound is None:
        return float("nan")
    return abs(bound - inc) / max(abs(inc), 1e-9)


def _prepare(products, prices, demand, probs, budget, transform):
    """Validated inputs, price tables, the M1 level filter and the M4 flag.

    Shared by solve_plan and milo.ladder.solve_with_rules so both see exactly the
    same levels, flags and flag order (the goldens pin the flag order).
    """
    demand = np.asarray(demand, dtype=float)
    levels, probs = _check_inputs(products, prices, demand, probs)
    if transform not in ("hull", "bigm"):
        raise ValueError("transform must be 'hull' or 'bigm'")
    net_cash, price, ratio = _price_tables(prices)
    flags = []

    valid = _filter_levels(products, levels, net_cash, flags)

    # M4: nothing affordable.
    cheapest = float("inf")
    included = list(valid.keys())
    for i in included:
        order_cost = float(products.at[i, "cost"]) * int(products.at[i, "min_order"])
        cheapest = min(cheapest, order_cost)
    if budget <= 0 or budget < cheapest:
        _add_flag(flags, None, "M4",
                  f"nothing affordable: budget {budget:.2f} < cheapest minimum order "
                  f"{cheapest:.2f}")
    return {"demand": demand, "levels": levels, "probs": probs, "net_cash": net_cash,
            "price": price, "ratio": ratio, "flags": flags, "valid": valid}


def _solve_and_read(model, products, setup, build_seconds, time_limit, mip_gap,
                    lp_relaxation):
    """Solve a transformed model and read the plan out; returns (result dict, HiGHS results).

    The result dict has exactly the frozen solve_plan keys (INTERFACES.md); callers
    that need more (the dual bound, a rule report) read it from the second value.
    """
    valid = setup["valid"]
    ratio = setup["ratio"]
    price = setup["price"]
    flags = setup["flags"]

    t1 = time.perf_counter()
    res, has_solution = _run_highs(model, time_limit, mip_gap)
    solve_seconds = time.perf_counter() - t1

    termination = res.termination_condition
    optimal = termination == TerminationCondition.convergenceCriteriaSatisfied
    gap = _gap(res)
    if not optimal:
        _add_flag(flags, None, "M10",
                  f"not proven optimal: termination={termination.name}, gap={gap:.4f}")

    objective = float("nan")
    buy_rows = []
    price_rows = []
    if has_solution:
        objective = float(pyo.value(model.profit))
        product_ids = list(products.index)
        for i in product_ids:
            cost = float(products.at[i, "cost"])
            units = 0
            if i in valid:
                units = int(round(pyo.value(model.x[i])))
            buy_rows.append({"i": i, "units": units, "spend": units * cost})
            if i not in valid:
                continue
            chosen = None
            for k in valid[i]:
                y = model.prod[i].price[k].binary_indicator_var.value
                if y is not None and y > 0.5:
                    chosen = k
            price_rows.append({"i": i, "level": chosen, "ratio": ratio[(i, chosen)],
                               "price": price[(i, chosen)]})
            stock = float(products.at[i, "on_hand"]) + units
            if stock > 0 and chosen == max(valid[i]):
                _add_flag(flags, i, "M2", f"optimum at grid edge: highest level {chosen}")
            elif stock > 0 and chosen == min(valid[i]):
                _add_flag(flags, i, "M3", f"optimum at grid edge: lowest level {chosen}")

    # M5: LP relaxation (integrality + disjunct binaries relaxed).
    lp_objective = float("nan")
    if lp_relaxation and has_solution:
        relaxed = model.clone()
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(relaxed)
        lp_res, lp_ok = _run_highs(relaxed, time_limit, mip_gap)
        if lp_ok:
            lp_objective = float(pyo.value(relaxed.profit))
            lump = lp_objective - objective
            if lump > 1e-6 * max(1.0, abs(objective)):
                _add_flag(flags, None, "M5",
                          f"LP relaxation {lp_objective:.2f} vs MILP {objective:.2f}: "
                          f"lumpiness gap {lump:.2f}")

    buy = pd.DataFrame(buy_rows, columns=["i", "units", "spend"])
    chosen_prices = pd.DataFrame(price_rows, columns=["i", "level", "ratio", "price"])
    result = {
        "status": res.solution_status.name,
        "termination": termination.name,
        "optimal": bool(optimal),
        "mip_gap": gap,
        "objective": objective,
        "lp_relaxation_objective": lp_objective,
        "buy": buy,
        "prices": chosen_prices,
        "flags": pd.DataFrame(flags, columns=FLAG_COLUMNS),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "model": model,
    }
    return result, res


def solve_plan(products, prices, demand, probs=None, budget=0.0, set_cap_share=1.0,
               transform="hull", time_limit=120, mip_gap=0.005, lp_relaxation=True):
    """Pick buy units and one price level per product to maximize expected profit."""
    setup = _prepare(products, prices, demand, probs, budget, transform)

    t0 = time.perf_counter()
    model = _build_model(products, setup["demand"], setup["probs"], setup["levels"],
                         setup["valid"], setup["net_cash"], budget, set_cap_share,
                         setup["flags"])
    pyo.TransformationFactory("gdp." + transform).apply_to(model)
    build_seconds = time.perf_counter() - t0

    result, _ = _solve_and_read(model, products, setup, build_seconds, time_limit,
                                mip_gap, lp_relaxation)
    return result


def budget_curve(products, prices, demand, budgets, **kw):
    """One solve per budget: budget, objective, spend."""
    kw.setdefault("lp_relaxation", False)
    rows = []
    for b in budgets:
        result = solve_plan(products, prices, demand, budget=b, **kw)
        spend = float(result["buy"]["spend"].sum())
        rows.append({"budget": b, "objective": result["objective"], "spend": spend})
    return pd.DataFrame(rows)
