"""Price-response steps P4-P7 (plan section 4).

P4  m = our price / market price, per unit and day
P5  price sensitivity (elasticity) below and above market, from evidence in priority order
P6  kinked demand curve h(m); demand multiplier for each price level
P7  ordered demand scenarios (shared random number per unit and scenario; see simulate.ordered_demand)
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PRICE_LEVELS = np.round(np.arange(0.80, 1.151, 0.05), 2)
MAX_MARKET_MULTIPLE = 1.6
ASSUMED = {  # low / base / high sensitivity (more negative = more sensitive); below-market slope is gentler
    "sealed": {"below": (-0.5, -1.0, -1.75), "above": (-1.0, -2.0, -3.5)},
    "single": {"below": (-0.25, -0.6, -1.25), "above": (-0.5, -1.2, -2.5)},
}


def type_group(product_type):
    return "single" if product_type == "single" else "sealed"


# ---------- P4 ----------
def relative_price(unit_days, our_price_col="our_price", market_col="market_price"):
    """m_it = our price / market price, with log splits for the kinked fit."""
    out = unit_days.copy()
    out["m"] = out[our_price_col] / out[market_col]
    log_m = np.log(out["m"])
    out["log_m_below"] = np.minimum(log_m, 0.0)
    out["log_m_above"] = np.maximum(log_m, 0.0)
    return out


# ---------- P5: evidence 0b, Poisson with unit fixed effects ----------
def poisson_irls(X, y, weights=None, max_iter=100, tol=1e-8, ridge=1e-6):
    """Plain Poisson regression by iteratively reweighted least squares (no statsmodels)."""
    n_obs, n_par = X.shape
    if weights is None:
        weights = np.ones(n_obs)
    beta = np.zeros(n_par)
    mean_y = max(float(np.average(y, weights=weights)), 1e-6)
    beta[0] = np.log(mean_y) if n_par else 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w = weights * mu
        XtW = X.T * w
        lhs = XtW @ X + ridge * np.eye(n_par)
        rhs = XtW @ z
        new_beta = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def fe_design(panel, slope_cols):
    """Design matrix: intercept, unit dummies (drop first), week dummies (drop first), slope columns."""
    units = pd.get_dummies(panel["unit"], prefix="u", drop_first=True, dtype=float)
    weeks = pd.get_dummies(panel["week"].astype(str), prefix="w", drop_first=True, dtype=float)
    parts = [pd.Series(1.0, index=panel.index, name="const"), units, weeks]
    # Day-of-week effects: weekends carry most store orders, so a busy Sunday must not read as a price effect.
    if "dow" in panel:
        days = pd.get_dummies(panel["dow"], prefix="d", drop_first=True, dtype=float)
        parts.append(days)
    parts.append(panel[slope_cols].astype(float))
    design = pd.concat(parts, axis=1)
    return design


def fit_fixed_effects(panel, slope_cols=("log_m_below", "log_m_above", "log_market_units"),
                      n_boot=100, seed=7):
    """Evidence 0b: log E[units] = unit FE + week FE + slopes. Bootstrap by unit for low/base/high.

    panel columns: unit, week, units (in-stock days only), weight, and the slope columns.
    """
    slope_cols = list(slope_cols)
    design = fe_design(panel, slope_cols)
    y = panel["units"].to_numpy(dtype=float)
    w = panel["weight"].to_numpy(dtype=float) if "weight" in panel else None
    beta = poisson_irls(design.to_numpy(), y, w)
    estimates = dict(zip(design.columns, beta))
    base = {col: float(estimates[col]) for col in slope_cols}
    rng = np.random.default_rng(seed)
    unit_ids = panel["unit"].unique()
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        pieces = []
        for copy_no, unit_id in enumerate(picked):
            piece = panel[panel["unit"] == unit_id].copy()
            piece["unit"] = str(unit_id) + "#" + str(copy_no)
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        sample_design = fe_design(sample, slope_cols)
        sample_w = sample["weight"].to_numpy(dtype=float) if "weight" in sample else None
        sample_beta = poisson_irls(sample_design.to_numpy(), sample["units"].to_numpy(dtype=float), sample_w)
        draws.append(sample_beta[-len(slope_cols):])
    draws = np.array(draws)
    result = {}
    for index, col in enumerate(slope_cols):
        column_draws = draws[:, index] if len(draws) else np.array([base[col]])
        result[col] = {"base": base[col], "p10": float(np.percentile(column_draws, 10)),
                       "p90": float(np.percentile(column_draws, 90))}
    return result


# ---------- P5: evidence 1, cross-section of markups (Huber) ----------
def fit_cross_section(items):
    """log(units per in-stock week) ~ type dummies + set popularity + slope x log(price / market).

    items columns: rate (units/week), m (price/market), product_type, set_popularity (optional).
    Robust (Huber) so one item at 175% of market can't dominate (M13).
    """
    usable = items[(items["rate"] > 0) & items["m"].notna() & (items["m"] > 0)].copy()
    if len(usable) < 5:
        return None
    types = pd.get_dummies(usable["product_type"], prefix="t", drop_first=True, dtype=float)
    columns = [np.ones(len(usable)), np.log(usable["m"].to_numpy())]
    names = ["const", "log_m"]
    if "set_popularity" in usable and usable["set_popularity"].notna().all():
        columns.append(np.log1p(usable["set_popularity"].to_numpy(dtype=float)))
        names.append("set_popularity")
    for col in types.columns:
        columns.append(types[col].to_numpy())
        names.append(col)
    X = np.column_stack(columns)
    y = np.log(usable["rate"].to_numpy(dtype=float))

    def residuals(beta):
        return X @ beta - y

    fit = least_squares(residuals, np.zeros(X.shape[1]), loss="huber", f_scale=0.5)
    jac = fit.jac
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(np.sum(fit.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(jac.T @ jac) * sigma2
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    slope = float(fit.x[1])
    return {"slope": slope, "se": se, "n": int(len(y)), "names": names, "coef": fit.x.tolist()}


def combine_elasticities(fe_fit=None, cross_fit=None, min_items=25):
    """Pick low/base/high per type group from the strongest usable evidence, clipped to assumed ranges.

    Rungs are evaluated in order (MODEL.md section 3): a present-but-rejected fixed-effects fit falls
    through to the cross-section rung, and a rejected cross-section falls through to the assumed range.
    Guard (M12/M20): reject a fit whose base slope is >= 0 or whose interval crosses 0, or with too few items.
    Returns DataFrame: group, side, low, base, high, source, note.
    """
    rows = []
    groups = list(ASSUMED.keys())
    for group in groups:
        for side in ("below", "above"):
            low, base, high = ASSUMED[group][side]
            source = "assumed range"
            notes = []
            accepted = False
            # Rung 1: fixed-effects Poisson panel (evidence 0b).
            fe_key = "log_m_below" if side == "below" else "log_m_above"
            if fe_fit is not None and group in fe_fit and fe_key in fe_fit[group]:
                est = fe_fit[group][fe_key]
                interval_ok = est["p90"] < 0 and est["base"] < 0
                if interval_ok:
                    base = est["base"]
                    low = est["p90"]
                    high = est["p10"]
                    source = "own sales vs daily market price (fixed effects)"
                    accepted = True
                else:
                    notes.append("fixed-effects fit rejected (sign or interval crosses 0)")
            # Rung 2: cross-section of markups (evidence 1), evaluated whenever rung 1 was not accepted.
            if not accepted and cross_fit is not None and group in cross_fit and cross_fit[group] is not None:
                fit = cross_fit[group]
                ok = fit["slope"] < 0 and fit["n"] >= min_items and np.isfinite(fit["se"])
                if ok and fit["slope"] + 1.28 * fit["se"] < 0:
                    base = fit["slope"]
                    low = fit["slope"] + 1.28 * fit["se"]
                    high = fit["slope"] - 1.28 * fit["se"]
                    source = "cross-section of markups (both sellers)"
                    accepted = True
                else:
                    rejected = "cross-section rejected (n=" + str(fit["n"]) + ", slope=" + str(round(fit["slope"], 2)) + ")"
                    notes.append(rejected)
            # Rung 3: the assumed range (low/base/high still hold it when no rung above was accepted).
            a_low, a_base, a_high = ASSUMED[group][side]
            # Clip to a wide envelope around the assumed range so a noisy fit can't produce absurd values.
            envelope_hi = min(a_high * 1.5, -0.05)
            envelope_lo = max(a_low * 0.5, -0.01)
            clipped_base = float(np.clip(base, envelope_hi, envelope_lo))
            clipped_low = float(np.clip(low, envelope_hi, envelope_lo))
            clipped_high = float(np.clip(high, envelope_hi, envelope_lo))
            clip_binds = (clipped_base != float(base)) or (clipped_low != float(low)) or (clipped_high != float(high))
            if clip_binds:
                notes.append("envelope clip binding")
            base = clipped_base
            low = clipped_low
            high = clipped_high
            rows.append({"group": group, "side": side, "low": low, "base": base, "high": high,
                         "source": source, "note": "; ".join(notes)})
    table = pd.DataFrame(rows)
    # The above-market slope must be at least as steep as below (plan P6); enforce on base values.
    for group in groups:
        below = table[(table["group"] == group) & (table["side"] == "below")].index[0]
        above = table[(table["group"] == group) & (table["side"] == "above")].index[0]
        for col in ("low", "base", "high"):
            if table.at[above, col] > table.at[below, col]:
                table.at[above, col] = table.at[below, col]
    return table


# ---------- P6 ----------
def h(m, e_below, e_above):
    """Kinked demand curve: m^e_below at or under market, m^e_above above (both e < 0)."""
    m = np.asarray(m, dtype=float)
    return np.where(m <= 1.0, np.power(m, e_below), np.power(m, e_above))


def level_multipliers(m_now, e_below, e_above, levels=PRICE_LEVELS):
    """Demand multiplier at each price level r (price = r x today's price) relative to today."""
    current = h(m_now, e_below, e_above)
    multipliers = []
    for ratio in levels:
        multipliers.append(float(h(ratio * m_now, e_below, e_above) / current))
    return np.array(multipliers)


def price_grid(units, levels=PRICE_LEVELS, max_market_multiple=MAX_MARKET_MULTIPLE):
    """One row per (unit, level) with price; levels above 1.6x market are dropped (M13)."""
    rows = []
    for _, unit in units.iterrows():
        market = unit.get("market_price")
        unit_rows = []
        for k, ratio in enumerate(levels):
            price = float(unit["price_now"]) * float(ratio)
            over_cap = pd.notna(market) and market > 0 and price > max_market_multiple * market
            unit_rows.append({"i": unit["unit"], "k": k, "ratio": float(ratio), "price": price, "over_cap": over_cap})
        kept = []
        for row in unit_rows:
            if not row["over_cap"]:
                kept.append(row)
        # If even today's price is above the cap (already priced well over market), keep every level and flag it:
        # the cap must never remove the price the item is actually selling at.
        if not kept:
            for row in unit_rows:
                row["cap_note"] = "today's price is already above 1.6x market"
            kept = unit_rows
        rows.extend(kept)
    return pd.DataFrame(rows)


# ---------- strategies ----------
def strategy_levels(units, prices, rule):
    """Pick a price level per unit for a named strategy. rule in: keep, match_market, market_minus_5, market_plus_10."""
    target_vs_market = {"match_market": 1.00, "market_minus_5": 0.95, "market_plus_10": 1.10}
    chosen = {}
    for _, unit in units.iterrows():
        options = prices[prices["i"] == unit["unit"]]
        if options.empty:
            continue
        if rule == "keep" or pd.isna(unit.get("market_price")) or rule not in target_vs_market:
            nearest = (options["ratio"] - 1.0).abs().idxmin()
        else:
            target_price = target_vs_market[rule] * float(unit["market_price"])
            nearest = (options["price"] - target_price).abs().idxmin()
        chosen[unit["unit"]] = int(options.loc[nearest, "k"])
    return chosen
