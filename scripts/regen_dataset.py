"""Regenerate the committed synthetic dataset at data/synthetic/ with the default knobs.

Run from the repo root:  python scripts/regen_dataset.py
Then rerun scripts/regen_goldens.py so the e2e goldens match the new dataset.
The committed dataset was frozen in one pass (knobs in SCHEMA.md); it is not
regenerated again afterwards.
"""
import json
from pathlib import Path

from milo import synth

REPO = Path(__file__).resolve().parents[1]


def main():
    target = REPO / "data" / "synthetic"
    manifest = synth.generate_dataset(target, **synth.DEFAULTS)
    print("wrote dataset to", target)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
