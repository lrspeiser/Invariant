"""Export retained exterior successes and the failed first potential join."""
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
    names = ['exterior-moment-001', 'exterior-verification-001', 'potential-join-001',
             'exterior-moment-002', 'exterior-verification-002']
    records, hashes = {}, {}
    for name in names:
        path = directory/name/'result.json'
        if not path.exists():
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != json.loads((path.parent/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError(f'Result changed: {name}')
        hashes[name] = digest
        records[name] = json.loads(path.read_bytes())
    summary = {'result_hashes': hashes, 'new_observational_scores': 0,
        'new_physical_gravity_rejections': 0, 'validated_universal_gravity_laws': 0,
        'complete_isolated_action_solver': False, 'stages': {}}
    exterior_lines, join_lines, verification_lines = [], [], []
    for name, value in records.items():
        if name.startswith('exterior-moment'):
            summary['stages'][name] = value['summary']
            for r in value['summary']:
                radius = value['config']['canonical_minimum_radius_kpc']
                order = max(value['config']['multipole_orders'])
                exterior_lines.append(f"| {name} | {r['variant']['id']} | {order} | {radius:g} | {r['uniform_series_tail_bound_at_admission_radius']['third_tensor']:.6g} | {r['maximum_canonical_cross_method_difference']['third_tensor']:.6g} | {r['within_registered_exterior_targets']} |")
        elif name.startswith('exterior-verification'):
            summary['stages'][name] = value
            fine = [r for r in value['rows'] if r['fractional_step'] == min(value['fractional_stencil_steps'])]
            verification_lines.append(f"- {name}: {value['verified_input_snapshots']} execution snapshots checked; "
                f"maximum fine-stencil scaled discrepancy {max(max(r['maximum_errors'].values()) for r in fine):.6g}; "
                f"all registered verification checks passed: {value['all_registered_checks_pass']}.")
        else:
            summary['stages'][name] = value['summary']
            for r in value['summary']:
                e = r['maximum_source_identity_errors']
                join_lines.append(f"| {r['variant']['id']} | {r['maximum_refinement_changes']['force']:.6g} | {r['maximum_refinement_changes']['third']:.6g} | {e['near_density_gradient']:.6g} | {e['joined_density_gradient']:.6g} | {r['within_all_registered_join_targets']} |")
    text = '''# Exterior source and potential join: retained results

No new gravity law is validated by these calculations. They repair the
Newtonian source derivatives needed by the current universal-length action.
No new gravity parameters, observational responses, raw data or reserved
confirmation products were opened or scored.

The original 80--120 kpc join failed its registered derivative targets. The
failure is retained. A higher-order exterior representation is a subsequent
numerical experiment; it does not retroactively pass the failed join.

## Exterior reference

| Run | Source | Highest order | Admission radius (kpc) | Uniform omitted-third-series bound | Maximum direct third-derivative difference | All exterior targets met |
| --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(exterior_lines)+'''

These exterior quantities use monopole units GM/r^(n+1), including the full
Cartesian tensor norms. The series bound is uniform over angle for the ideal
positive compact source; numerical moments and physical vertical tails are
separate checks. The direct comparison uses independent spatial quadrature
through infinite height. Its sampled agreement is not a uniform theorem for
the physical continuum tail. No gravitational-family rejection follows.

## First join failure

| Source | Largest force refinement | Largest third-tensor refinement | Near-field density-gradient identity error | Joined density-gradient identity error | All join targets met |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(join_lines)+'''

The fixed 80--120 kpc audit retains all 545 registered near-domain points and
152 join points per source. Its source-scaled tolerances were 0.0001 for force,
0.002 for Hessian and density, and 0.01 for the third tensor and density
gradient. These scales differ from the exterior monopole normalizers above.

The largest near-field density-gradient discrepancy occurred at R=96 kpc,
z=0 for both thicknesses. The largest third-tensor change was the cutoff
200-to-400 comparison at R=120 kpc, z=0. Radial integration refinement also
matters. Potential and force agreement alone did not diagnose these errors.
All derivatives of the join include the complete product rule, and neither
the trace nor its gradient is overwritten by the known source density.

The subsequent exterior experiment raises the expansion to order 128 and
tests admission from 60 kpc, retaining the original numerical tolerances.
That can support a new 60--80 kpc join, but the new joined potential has not
yet been audited. All old failures remain in the evidence tree.

## Independent checks

'''+ '\n'.join(verification_lines)+'''

The verification uses checked execution snapshots, 80-digit recombination of
the stored source moment integrals, and fourth-order derivative stencils at
every registered exterior point. The sum check is not another quadrature
method. Symbolic synthetic tests separately differentiate a nonspherical
joined potential through third order, including axis and reflection cases.

## Next requirements

Construct and audit the revised matched potential with denser coverage across
the source taper and transition; preserve errors at every point. Validate a
production representation derived from one C3 potential, then use its field
and derivatives in the complete action flux and a separate Poisson solve.
Only after those numerical checks can we repeat the fixed-parameter galaxy
comparison. Direct outer-star dynamics, lensing, full Solar System predictions,
stability and untouched confirmation remain open scientific requirements.

The goal remains active. These numerical results do not alter the earlier
conditional cluster, galaxy or Solar System comparison counts.

## Result hashes

'''+ '\n'.join(f'- `{name}`: `{digest}`' for name, digest in hashes.items())+'\n'
    (args.output/'Gravity-exterior-and-join-results.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-exterior-and-join-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf8', newline='\n')
    (ROOT/'docs/GRAVITY_EXTERIOR_AND_JOIN_RESULTS.md').write_text(text, encoding='utf8', newline='\n')
    (args.output/'Gravity-exterior-and-join-derivation.md').write_bytes((ROOT/'docs/GRAVITY_EXTERIOR_AND_JOIN_DERIVATION.md').read_bytes())


if __name__ == '__main__':
    main()
