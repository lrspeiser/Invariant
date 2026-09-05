"""Compare saved small length signals with their numerical grid changes."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root/'work/gravity-first-principles'
    result_path, cards_path = base/'tensor-poisson-001/result.json', base/'length-screening-local-001/result.json'
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), result_path, cards_path]
    hashes = {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes[result_path.relative_to(root).as_posix()] == '5c5d19ea954df993f7e4e2104d6495257eb7b7995c0bb9cf56fb12f0ad656306'
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')

    write('started.json', {'input_hashes':hashes, 'started_utc':datetime.now(UTC).isoformat(),
        'scope':'Post-run descriptive sensitivity, not a newly preregistered significance threshold or observational test.'})
    result = json.loads(result_path.read_bytes())
    cards = {r['card']['id']:r['card'] for r in json.loads(cards_path.read_bytes())['rows']}
    groups = []
    for length in [.001, .01, .1, 1., 10.]:
        rows = [c for c in result['comparisons'] if cards[c['card']]['length_pc'] == length]
        assert len(rows) == 216
        ratios = [{**r, 'change_over_peak_signal':r['maximum_scaled_length_signal_change']/r['maximum_scaled_length_signal']}
                  for r in rows if r['maximum_scaled_length_signal'] > 0]
        groups.append({'length_pc':length, 'comparisons':len(rows),
            'maximum_scaled_signal':max(r['maximum_scaled_length_signal'] for r in rows),
            'maximum_scaled_signal_change':max(r['maximum_scaled_length_signal_change'] for r in rows),
            'zero_peak_signal_count':len(rows)-len(ratios),
            'worst_ratio_case':max(ratios, key=lambda r:r['change_over_peak_signal'])})
    write('result.json', {'groups':groups,
        'ratio_definition':'Maximum normalized change divided by maximum normalized reference signal within the same card/distance/thickness comparison; maxima can occur at different radii.',
        'interpretation':'A full-force tolerance does not establish small-signal accuracy. No observational significance follows from these ratios.',
        'next_work':'Propagate source representations and refine angular response; inspect cancellation from subtracting nearly equal full fields for the smallest lengths.',
        'new_observational_scores':0, 'physical_exclusions':0})
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
