"""Report source-grid gains alongside the retained derivative failure."""
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
    directory = ROOT/'work/gravity-first-principles'
    names = ['matched-source-001', 'source-tail-002', 'source-tail-verification-001', 'tail-precision-diagnostic-001']
    records, hashes = {}, {}
    for name in names:
        path = directory/name/'result.json'
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != json.loads((path.parent/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError(f'Result changed: {name}')
        records[name] = json.loads(path.read_bytes())
        hashes[name] = digest
    failure = directory/'source-tail-001/failure.json'
    hashes['source-tail-001/failure.json'] = sha256(failure.read_bytes()).hexdigest()
    incomplete = directory/'source-tail-001/fields_primary_reference.json'
    hashes['source-tail-001/partial-fields'] = sha256(incomplete.read_bytes()).hexdigest()
    source_lines, verification_lines, diagnostic_lines = [], [], []
    for name in names[:2]:
        for row in records[name]['summary']:
            errors = row['worst_source_identity_errors']
            admitted = row['within_all_registered_matched_targets'] if name == names[0] else row['within_all_registered_tail_completion_targets']
            source_lines.append(f"| {name} | {row['variant']['id']} | {row['maximum_refinement_changes']['third']:.6g} | {errors['density']['value']:.6g} | {errors['density_gradient']['value']:.6g} | {admitted} |")
    verification = records['source-tail-verification-001']
    fine = [row for row in verification['rows'] if row['step_kpc'] == min(verification['step_sizes_kpc'])]
    for row in fine:
        for check in row['checks']:
            e = check['maximum_errors']
            verification_lines.append(f"| {row['variant']['id']} | {check['direction']} / {check['stencil']} | {e['gradient']['value']:.6g} | {e['hessian']['value']:.6g} | {e['third']['value']:.6g} |")
    for row in records['tail-precision-diagnostic-001']['rows']:
        last = row['precision_cases'][-1]
        diagnostic_lines.append(f"| {row['component']} | {row['ordinary_finite_first']-row['analytic_first']:.6g} | {last['finite_first_minus_analytic']:.6g} |")
    summary = {'result_hashes': hashes, 'stages': {names[0]: records[names[0]]['summary'], names[1]: records[names[1]]['summary'],
        names[2]: verification, names[3]: records[names[3]]['rows']},
        'source_grid_targets_pass': all(row['within_all_registered_tail_completion_targets'] for row in records[names[1]]['summary']),
        'independent_derivative_targets_pass': verification['all_registered_fine_checks_pass'],
        'production_provider_admitted': False, 'complete_nonlinear_action_solver': False,
        'new_observational_scores': 0, 'new_physical_gravity_rejections': 0, 'validated_universal_gravity_laws': 0}
    text = '''# Matched source and omitted potential tail: retained results

The leading omitted short-scale potential fixes the registered source-grid
errors, but its current floating-point evaluation fails independent derivative
verification. It remains provisional and is not admitted for new gravity-law
scores. A focused precision diagnostic identifies a practical repair direction.

No physical source, gravity constants, observational responses or reserved
data changed. These are numerical results, not new astronomical validation.

## Source-grid comparison

The new 60--80 kpc join retains all prior coordinates and adds dense sampling
across the disk taper and transition. Each source thickness has 115 radial by
41 vertical coordinates: 4,715 locations. The finite-cutoff audit compares five
one-factor refinements. Tail completion repeats those comparisons and adds a
log-source radial-quadrature refinement.

| Run | Thickness | Largest third-tensor refinement | Density identity error | Density-gradient identity error | All registered source-grid targets met |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(source_lines)+'''

The fixed source-grid tolerances are 0.0001 for force, 0.002 for Hessian and
density, 0.01 for third tensor and density gradient, and 0.000001 for the
near/exterior potential difference in GM/r units. Tensor discrepancies use
the inherited full-field scales; density errors use the physical source with
Hessian-based floors. These are not uniform fractional errors in density,
which vanishes at many locations.

The uncompleted source's worst density-gradient errors occur at the fixed
cosine taper endpoint R=36 kpc, z=0. The radial density is C1 there, so a
finite-wavenumber integral converges slowly in its higher spatial derivatives.
The added term is derived from the omitted potential integral. The density
trace and its gradient are still computed from that potential, never imposed.
The exact physical source and all numerical targets are unchanged.

The first tail run ended with a JSON serialization error for numpy.longdouble.
Its failure record and partially written field file are retained. The second
run changes only serialization and completes. The partial file is not a valid
completed scientific result.

## Independent derivative failure

The snapshot-based verifier differentiates the active correction (1-w) delta
psi at every original location, including the axis, radial source interfaces,
both disk sides and the exterior join. It uses 0.001 and 0.0005 kpc steps;
both one-sided stencils are tested at source interfaces. The fine-step target
was fixed at 0.0001 on the full-field force/Hessian/third-tensor scales.

| Thickness | Direction / stencil | Gradient error | Hessian error | Third-tensor error |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(verification_lines)+'''

The largest fine-step discrepancy occurs in a radial derivative at R=66.5
kpc, z=0 and exceeds the registered target. The nominal-source maximum rises
from 0.000343685 to 0.000743872 as the step is halved; the half-height maximum
rises from 0.00274889 to 0.00594980. The source-grid pass therefore does not
qualify the provider for production. All failures remain in the evidence tree.

## Focused precision diagnostic

The logarithmic expression subtracts large terms to obtain a tiny omitted
potential. This Windows runtime provides 52 stored mantissa bits for both
float64 and numpy.longdouble. The latter does not add arithmetic precision.
At the already exposed R=66.5 kpc point, the diagnostic reuses the same source,
mass and transform inputs. It evaluates cancellation arithmetic and low-k
Bessel functions with 50 digits, with a compensated sum for higher k.

| Radial component | Ordinary finite derivative minus analytic derivative | Higher-precision difference, accurate Bessel below k=8 |
| --- | --- | --- |
'''+ '\n'.join(diagnostic_lines)+'''

These are absolute radial A_K derivative differences, not the normalized
three-dimensional tensor errors in the preceding table. This single-point
diagnostic identifies cancellation as a major contributor. It is not a new
full-grid verification or a production repair.

## Next work and scientific limits

Implement a stable evaluation of the same scalar tail potential and preserve
its derivatives consistently. Repeat every source, refinement and derivative
gate without loosening targets or changing the source. Then validate a fast
representation derived from one C3 potential, the full nonlinear action flux
and a separate Poisson solve. Only after those checks can the same global
gravity constants be tested again against galaxies, clusters and local data.

The current cluster/local tension, conditional galaxy comparisons, incomplete
light coupling and untested stability remain unchanged. Environment-dependent
response and matter-current mechanisms remain research directions, not
validated explanations. The discovery goal remains active.

## Retained result hashes

'''+ '\n'.join(f'- `{name}`: `{digest}`' for name, digest in hashes.items())+'\n'
    (args.output/'Gravity-source-tail-results.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-source-tail-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf8', newline='\n')
    (ROOT/'docs/GRAVITY_SOURCE_TAIL_RESULTS.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-source-tail-derivation.md').write_bytes((ROOT/'docs/GRAVITY_SOURCE_TAIL_DERIVATION.md').read_bytes())


if __name__ == '__main__':
    main()
