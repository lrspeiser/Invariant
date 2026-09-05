"""Report completed precise-tail source and independent derivative controls."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['source-tail-002', 'source-tail-verification-001', 'source-tail-003', 'source-tail-verification-003', 'source-tail-controls-003']
    records, hashes = {}, {}
    for name in names:
        p = ROOT/'work/gravity-first-principles'/name/'result.json'
        data = p.read_bytes()
        digest = sha256(data).hexdigest()
        if digest != json.loads((p.parent/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError(f'Result changed: {name}')
        records[name], hashes[name] = json.loads(data), digest
    failed = ROOT/'work/gravity-first-principles/source-tail-verification-002/failure.json'
    hashes['source-tail-verification-002/failure.json'] = sha256(failed.read_bytes()).hexdigest()
    source = records['source-tail-003']
    verification = records['source-tail-verification-003']
    source_pass = all(row['within_all_registered_tail_completion_targets'] for row in source['summary'])
    derivative_pass = verification['all_registered_fine_checks_pass']
    lines = []
    for row in source['summary']:
        e = row['worst_source_identity_errors']
        change = row['maximum_refinement_changes']
        lines.append(f"| {row['variant']['id']} | {change['force']:.6g} | {change['hessian']:.6g} | {change['third']:.6g} | {e['density']['value']:.6g} | {e['density_gradient']['value']:.6g} |")
    differences = []
    for name in ['source-tail-verification-001', 'source-tail-verification-003']:
        v = records[name]
        fine = [row for row in v['rows'] if row['step_kpc'] == min(v['step_sizes_kpc'])]
        for row in fine:
            maximum = {key: max(check['maximum_errors'][key]['value'] for check in row['checks']) for key in ['gradient', 'hessian', 'third']}
            differences.append(f"| {name} | {row['variant']['id']} | {maximum['gradient']:.6g} | {maximum['hessian']:.6g} | {maximum['third']:.6g} |")
    summary = {'result_hashes': hashes, 'source_grid_pass': source_pass, 'independent_derivative_pass': derivative_pass,
        'sampled_reference_pass': source_pass and derivative_pass, 'source_summary': source['summary'],
        'derivative_verification': verification, 'implementation_controls': records['source-tail-controls-003'],
        'production_interpolant_validated': False, 'complete_nonlinear_action_solver': False,
        'new_observational_scores': 0, 'new_physical_gravity_rejections': 0, 'validated_universal_gravity_laws': 0}
    verdict = ('The repaired source passes its registered source-grid checks, and the active correction passes its independent derivative checks.'
               if summary['sampled_reference_pass'] else 'The repaired reference remains numerically unqualified: at least one registered check fails.')
    next_step = ('Next, build and validate a fast representation derived from one C3 potential, then evaluate the full action flux and its separate Poisson solve.'
                 if summary['sampled_reference_pass'] else 'Next, diagnose the retained failing points without changing the source or relaxing targets; repeat the affected checks before production use.')
    text = f'''# Precise omitted-potential calculation: retained result

{verdict}
This is a numerical milestone for the existing source model, not a new
astronomical result or a validated gravity law.

## Fixed source and arithmetic changes

The physical source, all 4,715 locations per thickness, the 60--80 kpc join
and numerical thresholds are unchanged. The canonical calculation uses
50 decimal digits for cancellation and the low-k Bessel band below 8 kpc^-1.
The stored source coefficients and surface transforms are retained; the
same-order radial Gauss rule is refined accurately. Nine cases per thickness
include eight separate variations, including 35 digits and a wider accurate
band below 16 kpc^-1. All derivatives remain derivatives of one potential.

| Thickness | Force refinement | Hessian refinement | Third refinement | Density identity | Density-gradient identity |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(lines)+'''

The registered source-grid targets are 0.0001 for force, 0.002 for Hessian and
density, 0.01 for the third tensor and density gradient, and 0.000001 for
potential overlap in GM/r units. Other quantities use their inherited
field/source scales; they are not fractional errors in vanishing density.

## Independent potential derivatives

| Verification | Thickness | Fine gradient error | Fine Hessian error | Fine third-tensor error |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(differences)+f'''

The fine-step target remains 0.0001. This verifier differentiates the newly
added active correction (1-w) delta psi, normalized by full-field scales.
Inherited Hankel and exterior derivatives have separate earlier controls;
this is not a new finite-difference audit of the entire matched potential.
The verifier checks every original point
at both 0.001 and 0.0005 kpc steps, including both one-sided radial interface
stencils, axis parity and the exact active join weight. It loads checked
execution snapshots. Only unused stencil offsets away from interfaces are
omitted from the expensive evaluation grid; no comparison is removed.

The original failed source join, serialized partial tail file, double-precision
derivative failure and initial Gauss-weight unit-control failure remain in
their original evidence directories. The first precise derivative verifier
also stopped on an inactive-row indexing error; its failure and executed
script are retained as source-tail-verification-002. A regression control and
full stencil-index preflight now precede the expensive calculation. No failure
is relabeled as a success.
The two new precision controls and the focused implementation suite passed
at their recorded execution hashes. The verifier supports both legacy and precise execution snapshots; the new
numerical verification checks the precise correction.

## Remaining work

{next_step}
Successful sampled checks do not establish uniform continuum error bounds or
production interpolation accuracy. Galaxy predictions must then be tested
with the same global constants used for clusters and the Solar System.

The user-proposed overlap and elastic directions remain open under their
recorded conservation, mass-scaling and known-family constraints. Complete
light coupling, dynamical stability, source uncertainty, direct outer-star
observations and independent confirmation remain requirements of the full
discovery goal. No new observational score or physical exclusion is added.

## Result hashes

'''+ '\n'.join(f'- `{name}`: `{digest}`' for name, digest in hashes.items())+'\n'
    (args.output/'Gravity-precise-tail-results.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-precise-tail-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf8', newline='\n')
    (ROOT/'docs/GRAVITY_PRECISE_TAIL_RESULTS.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-precise-tail-method.md').write_bytes((ROOT/'docs/GRAVITY_PRECISE_TAIL_METHOD.md').read_bytes())


if __name__ == '__main__':
    main()
