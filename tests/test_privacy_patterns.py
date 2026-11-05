"""Public privacy check: generic leak patterns in every file git would publish.

Why: this repo is public and runs on synthetic data only. Privacy is checked
in two layers (PROOF_UPGRADE_PLAN.md, "Shared standards" (c) 6):
- this public file holds GENERIC patterns only, so it can sit in a public
  repo without naming what it guards;
- real store, vendor and staff names live in one private deny file outside
  every repo, checked locally before each push. A deny file that lands
  inside the repo is itself a leak, so the last test fails on one.

Scope: tracked files plus untracked files that are not ignored (what the next
commit would pick up). Every pattern runs over each published PATH as well
as each text line; a path hit is reported as line 0, the same way the private
deny check reports one. Why: a photo named after its cert number publishes
the cert in the file name even though its bytes are never read.
Binary files (a NUL byte in the first 8 KiB) have their paths checked but
their contents skipped. Failure mode: a cert number or home path inside a
binary's bytes (image metadata, a pickled frame) passes here, and gitleaks
in CI does not cover it either (it looks for secrets, not cert numbers or
home paths); keep binaries synthetic-generated, never copied from real files.
Failure output names file, line and pattern only, never the matched text, so
a public CI log cannot repeat the leak it caught.
Failure mode: these patterns miss a leak that has no generic shape (a real
name in plain words); that is what the private deny check is for.
The planted examples in the control tests are built from pieces, so this file
never matches its own patterns.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY_SNIFF_BYTES = 8192

# Built from pieces so this file does not contain the path prefix it bans.
HOME_PATH_PATTERN = re.compile("/" + "Users" + "/")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
HEX64_PATTERN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
# An 8-10 digit run with digit-only boundaries: not part of a longer digit
# run, not the fraction of a decimal ("0.1234...") and not the whole part of
# one. Letters and "_" may touch it on purpose: a cert is often written as
# "PSA" + digits or "psa_" + digits + ".jpg", and a \w boundary let both pass.
CERT_PATTERN = re.compile(r"(?<![\d.])\d{8,10}(?!\d|\.\d)")
NOREPLY_DOMAIN = "users.noreply.github.com"


def git_paths(extra_args):
    """Repo-relative paths from `git ls-files -z` with the given flags.

    Why -z: synthetic data files have spaces in their names; NUL-separated
    output keeps them whole.
    """
    command = ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *extra_args]
    completed = subprocess.run(command, capture_output=True, check=True)
    output = completed.stdout.decode("utf-8")
    raw_paths = output.split("\0")
    paths = []
    for raw_path in raw_paths:
        if raw_path:
            paths.append(raw_path)
    return paths


def is_notebook_image_line(path, line):
    """True for a base64 image payload line inside a Jupyter notebook.

    Why: an embedded PNG is a long random string that can contain a digit
    run or a hex run by chance; it carries no text a person wrote.
    """
    if not path.endswith(".ipynb"):
        return False
    stripped = line.lstrip()
    return stripped.startswith('"image/')


def published_paths():
    """Every path the next commit would publish: tracked plus unignored untracked."""
    tracked = git_paths(["--cached"])
    untracked = git_paths(["--others", "--exclude-standard"])
    return sorted(set(tracked) | set(untracked))


def read_lines(root, paths):
    """(path, line number, text) to scan: each path as line 0, then each text line.

    Why line 0 comes first and before the binary skip: a file name is
    published even when its contents are not read, so a cert-named photo
    must still be caught.
    """
    lines = []
    for path in paths:
        lines.append((path, 0, path))
        full_path = root / path
        if not full_path.is_file():
            continue
        data = full_path.read_bytes()
        head = data[:BINARY_SNIFF_BYTES]
        if b"\0" in head:
            continue
        text = data.decode("utf-8", errors="replace")
        text_lines = text.splitlines()
        numbered = enumerate(text_lines, start=1)
        for line_number, line in numbered:
            if is_notebook_image_line(path, line):
                continue
            lines.append((path, line_number, line))
    return lines


def read_published_lines():
    """(path, line number, text) for every path and text line git would publish."""
    paths = published_paths()
    return read_lines(REPO_ROOT, paths)


@pytest.fixture(scope="module")
def published_lines():
    return read_published_lines()


def author_emails():
    """Author addresses declared in pyproject.toml, lowercased.

    Why: the author's own address is public by choice (LICENSE, package
    metadata); reading it from pyproject.toml keeps it out of this file and
    allows exactly that address, not a whole domain.
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        return set()
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    project = pyproject.get("project", {})
    authors = project.get("authors", [])
    emails = set()
    for author in authors:
        email = author.get("email", "")
        if email:
            emails.add(email.lower())
    return emails


def email_is_allowed(address, allowed_emails):
    """The author's own address and GitHub noreply addresses are allowed."""
    lowered = address.lower()
    if lowered in allowed_emails:
        return True
    return lowered.endswith("@" + NOREPLY_DOMAIN)


def find_hits(lines, pattern, pattern_name, allow=None):
    """'file:line: pattern' for each line where pattern matches (minus allowed).

    The matched text is left out on purpose: a public CI log must not repeat
    the leak it caught.
    """
    hits = []
    for path, line_number, line in lines:
        matches = pattern.findall(line)
        for match in matches:
            if allow is not None and allow(match):
                continue
            hits.append(f"{path}:{line_number}: {pattern_name}")
            break
    return hits


def deny_named_paths(paths):
    """Paths whose file name starts with 'deny' (any case)."""
    found = []
    for path in paths:
        name = Path(path).name.lower()
        if name.startswith("deny"):
            found.append(path)
    return found


def test_no_home_directory_paths(published_lines):
    """An absolute macOS home path leaks the author's account name and local layout."""
    hits = find_hits(published_lines, HOME_PATH_PATTERN, "home-directory path")
    assert hits == []


def test_no_email_addresses(published_lines):
    """An email address identifies a real person or account; only the author's own is allowed."""
    allowed_emails = author_emails()

    def allow(address):
        return email_is_allowed(address, allowed_emails)

    hits = find_hits(published_lines, EMAIL_PATTERN, "email address", allow=allow)
    assert hits == []


def test_no_64_hex_strings(published_lines):
    """A 64-hex run is shaped like a store admin persisted-query hash or an API secret."""
    hits = find_hits(published_lines, HEX64_PATTERN, "64-hex string")
    assert hits == []


def test_no_cert_like_digit_runs(published_lines):
    """A bare 8-10 digit run is shaped like a real grading-cert number.

    The portfolio's synthetic ids never take that shape: SYN-nnnnnn certs and
    7-digit product ids from 9,000,000 (cardshop-agents), card_NNNN
    (grace-grading-roi), csyn/gsyn ids (cardsegments), 6-digit product ids
    (milo). Decimals are excluded, because simulated results print with
    many decimal places.
    """
    hits = find_hits(published_lines, CERT_PATTERN, "cert-like digit run")
    assert hits == []


def test_no_deny_file_in_repo():
    """A deny* file inside the repo would publish the private names it exists to guard.

    Ignored files count too (--others without --exclude-standard): an
    ignored deny file is one `git add -f` away from being public.
    """
    tracked = git_paths(["--cached"])
    on_disk = git_paths(["--others"])
    all_paths = sorted(set(tracked) | set(on_disk))
    found = deny_named_paths(all_paths)
    assert found == []


# Controls: each check must be able to fail, so each pattern is shown to catch
# a planted example and to pass the synthetic shapes that sit beside it.


def planted_lines(text):
    return [("planted.txt", 1, text)]


def test_control_home_path_matches_planted_example():
    planted = "see " + "/" + "Users" + "/someone/project/data.csv"
    hits = find_hits(planted_lines(planted), HOME_PATH_PATTERN, "home-directory path")
    assert hits == ["planted.txt:1: home-directory path"]
    clean = find_hits(planted_lines("data/synthetic/users.csv"), HOME_PATH_PATTERN, "x")
    assert clean == []


def test_control_email_matches_planted_example_and_allows_noreply():
    planted = "contact " + "someone" + "@" + "example.org" + " today"
    hits = find_hits(planted_lines(planted), EMAIL_PATTERN, "email address")
    assert hits == ["planted.txt:1: email address"]
    noreply = "12345+bot" + "@" + NOREPLY_DOMAIN
    assert email_is_allowed(noreply, set())
    author = "author" + "@" + "example.org"
    assert email_is_allowed(author.upper(), {author})
    other = "someone" + "@" + "example.org"
    assert not email_is_allowed(other, {author})


def test_control_hex64_matches_planted_example_only_at_64():
    planted = "hash=" + "ab12" * 16
    hits = find_hits(planted_lines(planted), HEX64_PATTERN, "64-hex string")
    assert hits == ["planted.txt:1: 64-hex string"]
    short = "hash=" + ("ab12" * 16)[:63]
    long_run = "hash=" + "ab12" * 16 + "c"
    assert find_hits(planted_lines(short), HEX64_PATTERN, "x") == []
    assert find_hits(planted_lines(long_run), HEX64_PATTERN, "x") == []


def test_control_cert_pattern_matches_planted_examples():
    planted_runs = ["1234" + "5678", "1234" + "56789", "1234" + "567890"]
    for run in planted_runs:
        planted = "cert " + run + " graded 10"
        hits = find_hits(planted_lines(planted), CERT_PATTERN, "cert-like digit run")
        assert hits == ["planted.txt:1: cert-like digit run"]


def test_control_cert_pattern_matches_digits_glued_to_letters():
    """A cert written next to letters or "_" must still be caught."""
    digits = "1234" + "5678"
    glued_texts = ["PSA" + digits, "psa_" + digits + ".jpg", "cert#" + digits + "x"]
    for text in glued_texts:
        hits = find_hits(planted_lines(text), CERT_PATTERN, "cert-like digit run")
        assert hits == ["planted.txt:1: cert-like digit run"], text


def test_control_cert_named_binary_file_is_caught_by_its_path(tmp_path):
    """A binary photo named after its cert: contents skipped, path still flagged at line 0."""
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    photo_name = "photos/psa_" + "1234" + "5678" + ".jpg"
    photo_path = tmp_path / photo_name
    photo_path.write_bytes(b"\xff\xd8\xff\0binary")
    lines = read_lines(tmp_path, [photo_name])
    assert lines == [(photo_name, 0, photo_name)]
    hits = find_hits(lines, CERT_PATTERN, "cert-like digit run")
    assert hits == [f"{photo_name}:0: cert-like digit run"]
    clean_name = "photos/card_0042.jpg"
    clean_lines = read_lines(tmp_path, [clean_name])
    assert find_hits(clean_lines, CERT_PATTERN, "x") == []


def test_control_cert_pattern_passes_synthetic_shapes():
    synthetic_texts = [
        "cert SYN-" + "123456",
        "product 9000001",
        "card_0042",
        "csyn000000,gsyn0007",
        "share 0.1234" + "5678",
        "total 1234" + "5678.5",
        "epoch 1234" + "5678901",
    ]
    for text in synthetic_texts:
        hits = find_hits(planted_lines(text), CERT_PATTERN, "cert-like digit run")
        assert hits == [], text


def test_control_deny_file_name_check_catches_planted_paths():
    paths = ["data/deny.txt", "Denylist.csv", "tests/test_x.py", "docs/undeny.md"]
    found = deny_named_paths(paths)
    assert found == ["data/deny.txt", "Denylist.csv"]


def test_control_notebook_image_lines_are_the_only_skipped_lines():
    image_line = '      "image/png": "iVBORw0KGgo"'
    assert is_notebook_image_line("walkthrough.ipynb", image_line)
    assert not is_notebook_image_line("README.md", image_line)
    assert not is_notebook_image_line("walkthrough.ipynb", '    "print(1)"')
