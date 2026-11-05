"""README drift test (U0): every README number equals a generated file after formatting.

Two layers, both able to fail:
1. Every generated block (<!-- generated:NAME --> ... <!-- /generated:NAME -->) in
   README.md and EXPERIMENTS.md must equal what tools/render_docs.py renders from
   reports/numbers.json, reports/scorecard.json and reports/constraint_ladder.json.
2. Every result-looking number typed OUTSIDE a block (a decimal like 2.86 or a
   thousands-comma value like 1,568.38) must be one of the generated files' values
   after the docs' formatting, or sit on a short allowlist of non-result literals
   (Python versions, the price grid), each with its reason.

Limits of layer 2, measured, so nobody reads more into it than it checks: it only asks
whether a token appears ANYWHERE in a bag of every value in every generated file (in
each format below, plus abs() and x100), which is thousands of strings. A hand-typed
number therefore passes by chance fairly often (a review measured about 16% of random
2-decimal numbers under 10, and more for 1-decimal ones before .1f was dropped from the
formats), and the bag cannot tell WHICH value a token claims to be. So layer 2 is a
backstop, not proof: every result sentence lives in a generated block (layer 1, exact),
and the prose outside blocks is kept to settings and a short allowlist. A bare integer
in prose (a count like "24 scenarios") is not scanned at all, because integers collide
with settings and years.
"""
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_PATHS = (REPO / "README.md", REPO / "EXPERIMENTS.md", REPO / "DECISIONS.md")
BLOCK_PATHS = DOC_PATHS + (REPO / "MODEL.md",)   # MODEL.md prose holds model inputs, so blocks only
REQUIRED_README_BLOCKS = ("headline", "kpi_table", "scorecard_headline", "oos_losses", "marginal",
                          "evpi_vss", "ladder", "ladder_alone", "fee_blind", "shelf_sweep",
                          "edit_sweep", "strategies", "hull_bigm", "try_it", "edge_note", "hero",
                          "greedy_range", "hull_lead", "saa_claim")

# Non-result literals allowed in prose, with the reason each is not a result.
ALLOWED_LITERALS = {
    "3.12": "Python version (CI matrix 3.12-3.13)",
    "3.13": "Python version (CI matrix 3.12-3.13)",
    "0.80": "price grid lower edge (0.80x today's price)",
    "1.15": "price grid upper edge (1.15x today's price)",
    "0.005": "the operating rel gap default of solve_plan",
    "1.0": "a card-fee share of 1.0, the failure a contract test catches (README design notes)",
}

NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
FORMATS = (",.2f", ".2f", ".3f", ".4f", ",.0f", ".0f")


def _render_docs():
    spec = importlib.util.spec_from_file_location("render_docs", REPO / "tools" / "render_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk_numbers(obj, out):
    if isinstance(obj, dict):
        values = list(obj.values())
        for value in values:
            _walk_numbers(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _walk_numbers(value, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))


def allowed_numbers(reports, extra_values):
    """Every generated value, formatted every way the docs format numbers (and x100 as %)."""
    values = []
    names = sorted(reports.keys())
    for name in names:
        _walk_numbers(reports[name], values)
    values.extend(extra_values)
    allowed = set()
    for value in values:
        scaled = (value, abs(value), value * 100.0)
        for candidate in scaled:
            for spec in FORMATS:
                allowed.add(format(candidate, spec))
    return allowed


def prose_numbers(text):
    """(line number, token) for result-looking numbers outside generated blocks and code."""
    found = []
    in_block = False
    in_code = False
    lines = text.splitlines()
    numbered = enumerate(lines, start=1)
    for line_number, line in numbered:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if stripped.startswith("<!-- generated:"):
            in_block = True
            continue
        if stripped.startswith("<!-- /generated:"):
            in_block = False
            continue
        if in_block or in_code:
            continue
        cleaned = re.sub(r"`[^`]*`", "", line)
        matches = list(NUMBER_PATTERN.finditer(cleaned))
        for match in matches:
            end = match.end()
            after = cleaned[end:end + 2]
            before = cleaned[max(0, match.start() - 1):match.start()]
            # version strings like 6.10.1 and tokens glued to letters (v1.2) are not results
            if re.match(r"\.\d", after) or re.match(r"[A-Za-z.]", before):
                continue
            found.append((line_number, match.group(0)))
    return found


def _context():
    render_docs = _render_docs()
    reports = render_docs.load_reports()
    extra = list(render_docs.bound_ratio_range(reports["numbers"]))
    return render_docs, reports, allowed_numbers(reports, extra)


def unexplained_numbers(text, allowed):
    bad = []
    tokens = prose_numbers(text)
    for line_number, token in tokens:
        if token in allowed or token in ALLOWED_LITERALS:
            continue
        bad.append((line_number, token))
    return bad


def test_generated_blocks_equal_the_reports():
    render_docs, reports, _ = _context()
    for path in BLOCK_PATHS:
        text = path.read_text(encoding="utf-8")
        assert render_docs.render(text, reports) == text, (
            path.name + " has a stale generated block; run python tools/render_docs.py")


def test_readme_carries_every_required_block_and_leads_with_out_of_sample():
    render_docs = _render_docs()
    text = (REPO / "README.md").read_text(encoding="utf-8")
    names = render_docs.block_names(text)
    for name in REQUIRED_README_BLOCKS:
        assert name in names, "README.md lost its generated block " + name
    # DECISIONS.md U1: the out-of-sample number leads, even though it is lower.
    assert names[0] == "headline"
    reports = render_docs.load_reports()
    assert render_docs.headline(reports).startswith("**Out-of-sample profit per restock dollar")


def test_every_prose_number_is_a_generated_value():
    _, _, allowed = _context()
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        bad = unexplained_numbers(text, allowed)
        assert bad == [], path.name + " has numbers typed by hand (line, token): " + str(bad)


def test_control_a_planted_prose_number_is_caught():
    _, _, allowed = _context()
    text = (REPO / "README.md").read_text(encoding="utf-8")
    planted = text + "\nThe optimizer earns 1,234.57 more than greedy.\n"
    bad = unexplained_numbers(planted, allowed)
    tokens = []
    for _, token in bad:
        tokens.append(token)
    assert tokens == ["1,234.57"]


def test_control_a_hand_edited_block_is_caught():
    render_docs, reports, _ = _context()
    text = (REPO / "README.md").read_text(encoding="utf-8")
    real = render_docs.headline(reports)
    assert real in text
    edited = text.replace(real, real.replace("0.786", "0.799"))
    assert render_docs.render(edited, reports) != edited


def test_control_code_and_blocks_are_skipped_but_prose_is_not():
    sample = ("a 2.50 here\n```\n9.99 in code\n```\n<!-- generated:x -->\n8.88\n"
              "<!-- /generated:x -->\nversion 6.10.1 and `7.77` skipped\n")
    tokens = []
    found = prose_numbers(sample)
    for _, token in found:
        tokens.append(token)
    assert tokens == ["2.50"]


def test_reports_are_valid_json_with_sorted_keys():
    names = ("scorecard.json", "constraint_ladder.json")
    for name in names:
        raw = (REPO / "reports" / name).read_text(encoding="utf-8")
        data = json.loads(raw)
        again = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        assert raw == again, name + " is not written by milo.jsonout.dump"


FORBIDDEN_KEYS = ("tcg_id", "sku", "cost_source")
REPORT_SIZE_BUDGET = 200_000   # bytes per new report file


def _keys(obj, out):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        for key in keys:
            out.add(str(key))
            _keys(obj[key], out)
    elif isinstance(obj, list):
        for item in obj:
            _keys(item, out)


def test_new_reports_carry_no_forbidden_keys_and_fit_their_budget():
    names = ("scorecard.json", "constraint_ladder.json")
    for name in names:
        path = REPO / "reports" / name
        assert path.stat().st_size <= REPORT_SIZE_BUDGET, name
        keys = set()
        _keys(json.loads(path.read_text(encoding="utf-8")), keys)
        for forbidden in FORBIDDEN_KEYS:
            assert forbidden not in keys, name + " carries " + forbidden


def test_control_forbidden_key_walk_finds_a_nested_key():
    keys = set()
    _keys({"a": [{"b": {"sku": 1}}]}, keys)
    assert "sku" in keys
