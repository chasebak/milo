"""Write report JSON the same way everywhere: sorted keys, floats at 10 significant digits.

Why 10 digits: the committed files must not churn on the last bits of a float between
HiGHS builds or platforms, and tests compare recomputed values to the file at this
same rounding, never with a hand-picked 1e-9.
Failure mode: two values that differ only past the 10th significant digit compare
equal; nothing in this repo reports that precision.
"""
import hashlib
import json
import math

import numpy as np

SIGNIFICANT_DIGITS = 10


def round_sig(value, digits=SIGNIFICANT_DIGITS):
    """value rounded to `digits` significant digits (non-finite values become None)."""
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    magnitude = int(math.floor(math.log10(abs(value))))
    return round(value, digits - 1 - magnitude)


def rounded(obj):
    """A JSON-ready copy: floats rounded, numpy scalars unwrapped, NaN -> None."""
    if isinstance(obj, dict):
        out = {}
        keys = list(obj.keys())
        for key in keys:
            out[str(key)] = rounded(obj[key])
        return out
    if isinstance(obj, (list, tuple)):
        items = []
        for item in obj:
            items.append(rounded(item))
        return items
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round_sig(obj)
    return obj


def dump(obj, path):
    """Write obj to path as rounded, sorted-key JSON (one trailing newline)."""
    text = json.dumps(rounded(obj), indent=1, sort_keys=True, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return path


def fingerprint(products, prices, arrays):
    """A short content hash of the optimizer inputs, used by every --check.

    16 hex characters (64 bits) is plenty to notice a changed input and stays clear of
    the 64-hex pattern the privacy test bans.
    """
    digest = hashlib.sha256()
    # CSV text at 10 significant digits, not pandas' internal hash, so the value does
    # not depend on the pandas version CI happens to install.
    digest.update(products.to_csv(float_format="%.10g").encode("utf-8"))
    digest.update(prices.to_csv(float_format="%.10g").encode("utf-8"))
    for array in arrays:
        values = np.asarray(array, dtype=float)
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(np.round(values, 9)).tobytes())
    return digest.hexdigest()[:16]
