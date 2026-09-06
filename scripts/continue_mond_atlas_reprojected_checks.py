"""Apply the same numerical gates to a previously unverified source alternative."""
from __future__ import annotations
import argparse
from pathlib import Path
from mond_atlas_common import ROOT,read_json,write_json,digest
from run_mond_atlas_reprojected_fields import execute
from check_mond_atlas_field_pattern import forces,vector_difference


def main(args):
    check_config=read_json(args.config);config=read_json(ROOT/check_config['parent_config']);prior=ROOT/check_config['parent_run']
    summary=read_json(prior/'summary.json');assert digest(ROOT/check_config['parent_config'])==summary['config_sha256']
    for group in ('code_hashes','source_bindings'):
        for path,expected in summary[group].items():assert digest(ROOT/path)==expected,path
    gravity=read_json(ROOT/config['gravity_protocol']);case=next(c for c in config['stellar_cases'] if c['id']==check_config['case_id'])
    if args.output.exists() or args.private.exists():raise FileExistsError('immutable check output')
    args.output.mkdir(parents=True);args.private.mkdir(parents=True);checks={}
    for spec in check_config['checks']:
        execute(case,config,gravity,spec['half_width_kpc'],spec['spacing_kpc'],args.output,args.private,spec['id'])
        checks[spec['id']]=vector_difference(forces(prior,case['id']),forces(args.output,spec['id']))
    passed=all(v<(check_config['maximum_ring_vector_rms_gate'] if 'maximum_ring' in k else check_config['vector_relative_rms_gate']) for c in checks.values() for k,v in c.items())
    write_json(args.output/'summary.json',dict(status='MIXED_SOURCE_NUMERICAL_AUDIT',admission_disposition='SOURCE_BLOCKED',config=check_config,
        config_sha256=digest(args.config),checks=checks,mixed_model_numerical_gates_pass=passed,
        bindings={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/check_config['parent_config'],prior/'summary.json',ROOT/'scripts/run_mond_atlas_reprojected_fields.py')},
        new_full_field_runs=len(checks),response_files_opened=[],kinematic_response_scores_computed=0,goal_complete=False))
    print(dict(checks=checks,mixed_model_numerical_gates_pass=passed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_reprojected_mixed_convergence_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    main(a)
