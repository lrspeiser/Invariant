"""Export the overlap audit and an explicitly illustrative orbital-speed plot."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = ROOT/'work/gravity-first-principles/overlap-elastic-001'
    raw = (run/'result.json').read_bytes()
    assert sha256(raw).hexdigest() == json.loads((run/'receipt.json').read_bytes())['result_sha256']
    (args.output/'Gravity-overlap-elastic-summary.json').write_bytes(raw)
    (args.output/'Gravity-overlap-elastic-assessment.md').write_bytes((ROOT/'docs/GRAVITY_OVERLAP_ELASTIC_RESULTS.md').read_bytes())
    r = np.geomspace(.03, 20, 500)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    fig.set_facecolor('#f6f5f1')
    ax.set_facecolor('#f6f5f1')
    ax.loglog(r, np.sqrt(1/r), color='#707780', linewidth=2.3, label='Newtonian compact source')
    ax.loglog(r, np.sqrt(1/r+1), color='#167b75', linewidth=2.8, label='Added 1/r pull: speed levels out')
    ax.loglog(r, np.sqrt(1/r+r*r), color='#bd6036', linewidth=2.8, label='Added spring pull: speed rises')
    ax.set(xlabel='Distance / reference length L', ylabel='Circular speed / reference speed',
        title='Stronger relative gravity does not require a growing pull')
    ax.legend(loc='lower left', frameon=False, fontsize=10)
    ax.grid(which='major', alpha=.17)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.99, -.025, 'Same compact source and Newtonian term. Chosen extra-force strengths; no measured data.',
        ha='right', va='top', fontsize=9, color='#444444')
    fig.savefig(args.output/'Gravity-overlap-elastic-illustration.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Exported overlap assessment, retained result and illustrative figure.')


if __name__ == '__main__':
    main()
