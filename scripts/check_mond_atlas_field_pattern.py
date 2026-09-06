"""Stricter vector-convergence and symmetry follow-up to the first galaxy field.

Triggered because azimuth-averaged force convergence can hide errors in lateral
force. This audit retains the original radial gates and does not relabel them.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,digest,write_json
from run_mond_atlas_ngc2903_fields import run_one


def forces(folder,name):
    keys=('radius_kpc','angle_deg','newton_inward','newton_tangential','mond_inward','mond_tangential')
    return np.array([[float(r[k]) for k in keys] for r in csv.DictReader((folder/(name+'-forces.csv')).open())])


def vector_difference(a,b):
    if not np.array_equal(a[:,:2],b[:,:2]):raise ValueError('force sampling differs')
    result={}
    for label,start in (('newton',2),('mond',4)):
        delta=b[:,start:start+2]-a[:,start:start+2]
        result[label+'_vector_relative_rms']=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))/np.mean(np.sum(b[:,start:start+2]**2,axis=1))))
        rings=[]
        for radius in np.unique(a[:,0]):
            use=a[:,0]==radius
            rings.append(float(np.sqrt(np.mean(np.sum(delta[use]**2,axis=1))/np.mean(np.sum(b[use,start:start+2]**2,axis=1)))))
        result[label+'_maximum_ring_relative_rms']=max(rings)
    return result


def main(args):
    config=read_json(args.protocol);audit=read_json(args.source/'source-audit.json')
    if digest(args.protocol)!=audit['protocol_sha256']:raise ValueError('source protocol changed')
    packet_path=ROOT/audit['source_packet']
    if digest(packet_path)!=audit['source_packet_sha256']:raise ValueError('source packet changed')
    if args.output.exists() or args.private.exists():raise FileExistsError('immutable output')
    args.output.mkdir(parents=True);args.private.mkdir(parents=True)
    with np.load(packet_path) as f:packet={k:f[k] for k in f.files}
    cases={c['id']:c for c in config['cases']}
    frozen=dict(trigger='nominal .5 -> .25 kpc lateral full-vector comparison exceeded 3 percent aggregate RMS despite passing the original radial-mean gate',
        fine_spacing_kpc=[.125,.125,.125],symmetry_spacing_kpc=[.25,.25,.125],half_width_kpc=24.,
        vector_rms_gate=.03,maximum_ring_vector_rms_gate=.05,
        source_refit=False,target_velocities_consumed=False,adaptive_numerical_check=True)
    write_json(args.output/'declared-followup.json',frozen)
    run_one(packet,config,cases['axisymmetrized'],24.,[.25,.25,.125],args.output,args.private,'axisym_lateral_refined')
    run_one(packet,config,cases['nominal'],24.,[.125,.125,.125],args.output,args.private,'lateral_finer')
    comparisons={
        'base_to_half_step':vector_difference(forces(args.previous,'nominal'),forces(args.previous,'lateral_refined')),
        'half_to_quarter_step':vector_difference(forces(args.previous,'lateral_refined'),forces(args.output,'lateral_finer')),
        'vertical':vector_difference(forces(args.previous,'nominal'),forces(args.previous,'vertical_refined')),
        'box':vector_difference(forces(args.previous,'nominal'),forces(args.previous,'larger_box'))}
    fine=comparisons['half_to_quarter_step']
    passed=all(v<(frozen['maximum_ring_vector_rms_gate'] if 'maximum_ring' in k else frozen['vector_rms_gate']) for k,v in fine.items())
    write_json(args.output/'vector-audit.json',dict(status='CONDITIONAL_FIELD_VECTOR_AUDIT',comparisons=comparisons,
        refined_vector_convergence_pass=passed,source_deprojection_identified=False,admitted_motion_prediction=False,
        bindings=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in (Path(__file__),ROOT/'scripts/run_mond_atlas_ngc2903_fields.py',args.previous/'field-audit.json',args.source/'source-audit.json')]))
    print(comparisons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/mond_atlas_ngc2903_field_v1.json')
    for k in ('source','previous','output','private'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    main(a)
