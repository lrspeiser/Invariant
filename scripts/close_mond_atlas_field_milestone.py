"""Preserve the exploratory results and apply the repository's admission audit.

This does not perform new motion scoring or refit any source/solver parameter.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
from mond_atlas_common import ROOT,read_json,write_json,digest


def close(previous,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    policy=ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md'
    audit=dict(status='SOURCE_BLOCKED_FOR_ADMITTED_MOTION_SCORING',
        policy=str(policy.relative_to(ROOT)),policy_sha256=digest(policy),
        original_exploratory_response_comparison_retained=True,scientific_benchmark_admitted=False,
        required_admission_disposition_frozen_before_implementation=False,
        all_required_source_and_implementation_gates_completed_before_response_access=False,
        source_parameters_fitted_to_target_motions=False,
        numerical_refinement_selected_from_motion_residuals=False,
        reason='A conditional source sensitivity model and numerical checks were frozen, but the new builder did not record the required formal admission disposition with a complete paper/source benchmark and source validation before the published curve was inspected. Some source and likelihood gates remain unresolved.',
        correction='Retain the exploratory comparison with this disclosure. Exclude it from admitted scientific scores and sample counts. Do not claim retrospective preregistration or a pristine holdout.',
        numerical_scope='Eleven real-source conditional Newtonian/QUMOND field runs and their numerical convergence checks remain inspectable. A complete observed 3D mass reconstruction and motion likelihood are not established.',
        prospective_requirements=['Freeze a complete source/paper/benchmark admission package before any new observational scoring.',
            'Resolve source photometric normalization, geometry/depth and missing mass treatment; test projection and instrument closure.',
            'Preserve NGC2903 and all previously opened seed responses as development-exposed; use new galaxy/group/survey holdouts for confirmation.'],
        primary_sources_identified_for_next_package=[
            {'role':'stellar ICA measurement','url':'https://arxiv.org/abs/1410.0009'},
            {'role':'stellar conversion','url':'https://arxiv.org/abs/1402.5210'},
            {'role':'THINGS measurement','url':'https://arxiv.org/abs/0810.2125'},
            {'role':'HERACLES measurement','url':'https://arxiv.org/abs/0905.4742'},
            {'role':'QUMOND equations','url':'https://arxiv.org/abs/0911.5464'}])
    write_json(output/'admission-audit.json',audit)
    # Existing comparison artifacts are preserved verbatim, not recalculated.
    for name in ('conditional-motion-comparison.csv','conditional-motion-scores.csv','structure-comparison.csv','stellar-dust-check.json','validation.log'):
        (output/name).write_bytes((previous/name).read_bytes())
    status=read_json(previous/'execution-status.json');status.update(
        admitted_scientific_motion_comparisons=0,exploratory_response_comparison_status=audit['status'],
        required_source_admission_complete=False,previous_report_superseded=str(previous.relative_to(ROOT)))
    status['next_required'].insert(0,'Read admission-audit.json. Freeze and verify the complete prospective admission package before any additional observational scoring; the retained comparison is not an admitted benchmark.')
    write_json(output/'execution-status.json',status)
    verification=read_json(previous/'verification.json');verification.update(
        status='NUMERICAL_TESTS_PASS_OBSERVATIONAL_ADMISSION_INCOMPLETE',
        source_builder_admitted_for_motion_scoring=False,earlier_comparison_reclassified=True)
    write_json(output/'verification.json',verification)
    text=(previous/'README.md').read_text(encoding='utf-8')
    first,rest=text.split('\n',1)
    addendum='''

**Admission correction:** The new source builder did not complete the repository's
required admission record before the published rotation comparison was made.
The retained 10.6% versus 28.3% comparison is **exploratory and not an admitted
scientific benchmark**. The result does not count toward a validated galaxy
sample. Numerical convergence alone does not supply missing source validation.
See [the admission audit](admission-audit.json). No source parameters were fitted
to these response values, and the additional grid refinement was driven by a
numerical vector discrepancy, but those facts do not repair the missing record.
Future scoring must follow the
[repository admission policy](../../../docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md).
'''
    text=first+addendum+rest
    text=text.replace('## Actual comparison with published motion','## Retained exploratory comparison with published motion — not admitted')
    text=text.replace('The next decisive test is whether','After the source admission package is complete, the next decisive test is whether')
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    bindings={str(p.relative_to(ROOT)):digest(p) for p in previous.iterdir() if p.is_file()}
    bindings.update({str(Path(__file__).relative_to(ROOT)):digest(__file__),str(policy.relative_to(ROOT)):digest(policy)})
    write_json(output/'input-bindings.json',bindings)
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:'):assert (output/link).is_file(),link
    print(dict(report=str(output/'README.md'),numerical_tests=verification['tests_run'],admitted_motion_comparisons=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();close(args.previous.resolve(),args.output.resolve())
