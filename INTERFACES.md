# Frozen interfaces

The three optimizer-facing signatures below are the frozen public interface. Everything
else in the repo may change — the experiments, the notebook and the docs all build on
these calls, and the committed dataset may be regenerated — but these signatures and
their return shapes do not.

## milo/optimize.py

```python
def solve_plan(products, prices, demand, probs=None, budget=0.0, set_cap_share=1.0,
               transform="hull", time_limit=120, mip_gap=0.005, lp_relaxation=True):
    """Pick buy units and one price level per product to maximize expected profit."""
```

Returns a dict with keys: `status`, `termination`, `optimal`, `mip_gap`, `objective`,
`lp_relaxation_objective`, `buy` (DataFrame: i, units, spend), `prices` (DataFrame: i,
level, ratio, price), `flags` (DataFrame: i, code, message), `build_seconds`,
`solve_seconds`, `model`.

```python
def budget_curve(products, prices, demand, budgets, **kw):
    """One solve per budget: budget, objective, spend."""
```

Returns a DataFrame with columns `budget`, `objective`, `spend` (one row per entry in
`budgets`, in order).

## milo/simulate.py

```python
def evaluate_plan(products, prices, demand, buy_units, price_level, probs=None):
    """Profit per scenario for a fixed buy + price plan."""
```

Returns `(profit, summary)`: a numpy array of length n_scenarios and a dict with keys
`expected`, `p10`, `p50`, `p90`, `sell_through`, `leftover_value`.

## Shared input contract

- `products`: DataFrame indexed by unit id with columns `cost`, `on_hand`, `min_order`,
  `max_units`, `set_code`, `salvage_frac`.
- `prices`: DataFrame with columns `i`, `k`, `ratio`, `price`, `net_cash`.
- `demand`: numpy array of shape (n_products, n_levels, n_scenarios), non-negative and
  non-increasing in `k`.
- `probs`: optional scenario probabilities (uniform when None).

## Not frozen: milo/ladder.py

`solve_with_rules(products, prices, demand, rules=None, probs=None, budget=0.0,
set_cap_share=1.0, transform="hull", time_limit=120, mip_gap=0.005, lp_relaxation=False)`
returns every `solve_plan` key plus `objective_bound`, `rule_report`, `n_variables`,
`n_integer_variables` and `n_constraints`. With `rules=None` it builds exactly the
`solve_plan` model (tested). It is new and may still change; `solve_plan` does not.
