"""The one "why this metric" dict (milo/why.py) quotes settings; each must equal its constant.

milo/why.py cannot format its text from experiments/ (the installed package must not
import it), and the README drift test skips generated blocks, where this text renders.
So every quoted setting is pulled out of the text by a pattern and compared to the
constant it names. Changing N_EVAL, a seed, the time limit or a sweep point fails here
until the text follows. A control plants a wrong value and must be caught.
"""
import re

from experiments import common, constraint_ladder, saa_convergence, scorecard
from milo.why import KPIS, WHY


def _all_text():
    parts = []
    keys = sorted(WHY.keys())
    for key in keys:
        entry = WHY[key]
        parts.append(entry["question"])
        parts.append(entry["kind"])
        parts.append(entry["lie"])
    for kpi in KPIS:
        fields = ("name", "formula", "why", "against", "does", "measured_on")
        for field in fields:
            parts.append(kpi[field])
    return "\n".join(parts)


def _percent_list(fractions):
    parts = []
    for fraction in fractions:
        parts.append(format(fraction * 100.0, ".0f"))
    return "/".join(parts)


def _limit_list(limits):
    parts = []
    for limit in limits:
        parts.append(str(limit))
    return ", ".join(parts)


def expected_literals():
    """(pattern, expected text of every capture group, constant name) for each quoted setting."""
    step = format(scorecard.MARGINAL_STEP, ".0f")
    width = format(2.0 * scorecard.MARGINAL_STEP, ".0f")
    return [
        (r"([\d,]+) fresh scenarios", (format(scorecard.N_EVAL, ","),), "scorecard.N_EVAL"),
        (r"([\d,]+) evaluation scenarios", (format(saa_convergence.N_EVAL, ","),), "saa_convergence.N_EVAL"),
        (r"eval seed (\d+)", (str(scorecard.EVAL_SEED),), "scorecard.EVAL_SEED"),
        (r"(?<!to )\b(\d+) training scenarios", (str(common.N_SCENARIOS),), "common.N_SCENARIOS"),
        (r"pinned (\d+) scenarios", (str(common.N_SCENARIOS),), "common.N_SCENARIOS"),
        (r"with (\d+) scenarios", (str(common.N_SCENARIOS),), "common.N_SCENARIOS"),
        (r"N = (\d+) to (\d+)", (str(min(saa_convergence.NS)), str(max(saa_convergence.NS))), "saa_convergence.NS"),
        (r"(\d+) seeds each", (str(len(saa_convergence.SEEDS)),), "saa_convergence.SEEDS"),
        (r"(\d+) s limit", (str(common.EXPERIMENT_TIME_LIMIT),), "common.EXPERIMENT_TIME_LIMIT"),
        (r"swept at ([\d/]+)%", (_percent_list(constraint_ladder.SHELF_FRACTIONS),), "constraint_ladder.SHELF_FRACTIONS"),
        (r"P = ([\d, ]+), all", (_limit_list(constraint_ladder.EDIT_LIMITS),), "constraint_ladder.EDIT_LIMITS"),
        (r"(\d+) more units", (format(constraint_ladder.SHELF_STEP, ".0f"),), "constraint_ladder.SHELF_STEP"),
        (r"obj\(B \+ (\d+)\) - obj\(B - (\d+)\)\) / (\d+)", (step, step, width), "scorecard.MARGINAL_STEP"),
        (r"within a ([\d.]+)% gap", (format(common.EXPERIMENT_GAP * 100.0, ".2f"),), "common.EXPERIMENT_GAP"),
        (r"rel gap (1e-\d+)", (format(common.EXPERIMENT_GAP, ".0e").replace("e-0", "e-"),), "common.EXPERIMENT_GAP"),
    ]


def stale_literals(text, literals):
    """Every quoted setting in text that differs from its constant, as (constant, found, expected)."""
    stale = []
    for pattern, expected, name in literals:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            found = match.groups()
            if found != expected:
                stale.append((name, found, expected))
    return stale


def test_every_quoted_setting_equals_its_constant():
    assert stale_literals(_all_text(), expected_literals()) == []


def test_every_pattern_is_used_at_least_once():
    # a pattern that never matches would let a reworded sentence escape the check
    text = _all_text()
    for pattern, _, name in expected_literals():
        assert re.search(pattern, text), name + " is no longer quoted; update the pattern list"


def test_control_a_planted_wrong_setting_is_caught():
    text = _all_text() + "\nvalued on 2,000 fresh scenarios with a 60 s limit, swept at 50/75%"
    stale = stale_literals(text, expected_literals())
    names = []
    for name, _, _ in stale:
        names.append(name)
    assert "scorecard.N_EVAL" in names
    assert "common.EXPERIMENT_TIME_LIMIT" in names
    assert "constraint_ladder.SHELF_FRACTIONS" in names
