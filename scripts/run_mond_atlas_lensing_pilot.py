"""Ingest/replay a bounded SLACS source pilot; creates a new immutable receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mond_atlas_lensing_pilot import ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/mond_atlas_lensing_pilot_v1.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory inside work/gravity-first-principles/mond-atlas-lensing-pilot-001')
    parser.add_argument('--offline', action='store_true', help='Require cached, hash-verified downloads')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, offline=args.offline), indent=2))


if __name__ == '__main__':
    main()
