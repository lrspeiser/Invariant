"""Full-field sensitivity of source-reprojected depth alternatives, no motions."""
from __future__ import annotations
import argparse,gc,time
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
import mond_atlas_rectangular_fields as rf
from run_mond_atlas_ngc2903_fields import interpolate2,force_profiles
from check_mond_atlas_field_pattern import forces,vector_difference


def vertical_cell_density(axis,spacing,height):
    lo=axis-spacing/2;hi=axis+spacing/2
    def cdf(z):return np.where(z<0,.5*np.exp(-np.abs(z)/height),1-.5*np.exp(-np.abs(z)/height))
    return (cdf(hi)-cdf(lo))/spacing


def build_density(star_case,config,gravity,half,spacing):
    axes=[np.linspace(-half,half,int(round(2*half/h))+1) for h in spacing]
    x,y=np.meshgrid(axes[0],axes[1],indexing='ij');density=np.zeros(tuple(len(a) for a in axes));masses={};resampling={}
    conv=gravity['conversions']
    components=[('stellar',star_case['source'],conv['nominal_stellar_ml'],star_case['vertical_components']),
        ('atomic_helium',config['gas_cases']['atomic_helium'],1.,[[1.,config['gas_cases']['height_kpc']]]),
        ('molecular_helium',config['gas_cases']['co21'],conv['alpha_co10_including_helium']/conv['co21_to_co10'],[[1.,config['gas_cases']['height_kpc']]])]
    for name,path,factor,layers in components:
        with np.load(ROOT/path) as f:surface=f['intrinsic_effective_surface']*factor*1e6;axis=f['axis']
        expected=float(surface.sum()*(axis[1]-axis[0])**2);sampled=interpolate2(surface,axis,x,y)
        ratio=expected/(sampled.sum()*spacing[0]*spacing[1]);sampled*=ratio
        assert abs(sum(f for f,h in layers)-1)<1e-12 and all(f>=0 and h>0 for f,h in layers)
        vertical=sum(f*vertical_cell_density(axes[2],spacing[2],h) for f,h in layers)
        density+=sampled[:,:,None]*vertical[None,None,:]
        masses[name]=expected;resampling[name]=float(ratio)
    return density,axes,dict(component_mass_msun=masses,resampling_normalization=resampling,
        finite_grid_total_mass_msun=float(density.sum()*np.prod(spacing)))


def execute(star_case,config,gravity,half,spacing,output,private,label):
    start=time.monotonic();print('START '+label,flush=True)
    rho,axes,source=build_density(star_case,config,gravity,half,spacing)
    G=gravity['gravity']['G_kpc_kms2_per_msun'];a0=gravity['gravity']['a0_kms2_per_kpc']
    bn,bm,moments=rf.multipole_boundary(rho,axes,G,a0)
    pn,pm,residuals=rf.solve(rho,spacing,bn,bm,G,a0);del rho,bn,bm;gc.collect()
    rows,profile,plane=force_profiles(pn,pm,axes,spacing,gravity);del pn,pm;gc.collect()
    write_csv(output/(label+'-forces.csv'),rows);write_csv(output/(label+'-profile.csv'),profile)
    np.savez_compressed(private/(label+'-midplane.npz'),**plane)
    result=dict(id=label,star_case=star_case,half_width_kpc=half,spacing_kpc=spacing,source=source,moments=moments,
        residuals=residuals,seconds=time.monotonic()-start,profile=profile)
    write_json(output/(label+'-result.json'),result)
    print('DONE '+label+' '+str(round(result['seconds'],2))+' seconds',flush=True)
    return result


def main(args):
    config=read_json(args.config);assert config['admission_disposition']=='SOURCE_BLOCKED';gravity=read_json(ROOT/config['gravity_protocol'])
    products={}
    for key in ('projection_run','mixed_projection_run'):
        audit=read_json(ROOT/config[key]);assert audit['all_optimizers_converged']
        for p in audit['products']:products[p['path'].replace('\\','/')]=p['sha256']
    paths=[c['source'] for c in config['stellar_cases']]+[config['gas_cases'][k] for k in ('atomic_helium','co21')]
    for p in paths:assert digest(ROOT/p)==products[p],p
    if args.output.exists() or args.private.exists():raise FileExistsError('immutable output')
    args.output.mkdir(parents=True);args.private.mkdir(parents=True)
    cases={c['id']:c for c in config['stellar_cases']};results={}
    for name,c in cases.items():results[name]=execute(c,config,gravity,config['half_width_kpc'],config['base_grid_spacing_kpc'],args.output,args.private,name)
    comparisons={}
    for check in config['numerical_followups']:
        label=check['id'];execute(cases[check['case_id']],config,gravity,check['half_width_kpc'],check['spacing_kpc'],args.output,args.private,label)
        comparisons[label]=vector_difference(forces(args.output,check['case_id']),forces(args.output,label))
    passed=all(v<(config['maximum_ring_vector_rms_gate'] if 'maximum_ring' in k else config['vector_relative_rms_gate']) for c in comparisons.values() for k,v in c.items())
    sensitivity=[]
    for a,b in zip(results['refitted_thin']['profile'],results['refitted_mixed']['profile']):
        sensitivity.append(dict(radius_kpc=a['radius_kpc'],thin_mond_force_speed_kms=a['mond_force_speed_kms'],mixed_mond_force_speed_kms=b['mond_force_speed_kms'],
            mixed_minus_thin_mond_speed_fraction=b['mond_force_speed_kms']/a['mond_force_speed_kms']-1,
            thin_mond_tangential_fraction=a['mond_tangential_fraction'],mixed_mond_tangential_fraction=b['mond_tangential_fraction']))
    write_csv(args.output/'source-depth-force-sensitivity.csv',sensitivity)
    write_json(args.output/'summary.json',dict(status='CONDITIONAL_IMAGE_CONSISTENT_FORCE_DIAGNOSTIC_NOT_ADMITTED_FOR_MOTION',admission_disposition='SOURCE_BLOCKED',
        config=config,config_sha256=digest(args.config),source_models=2,full_field_runs=2+len(config['numerical_followups']),
        thin_model_numerical_checks=comparisons,thin_model_numerical_gates_pass=passed,mixed_model_convergence_validated=False,
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'scripts/mond_atlas_rectangular_fields.py',ROOT/'scripts/run_mond_atlas_ngc2903_fields.py',ROOT/'scripts/check_mond_atlas_field_pattern.py')},
        source_bindings={p:digest(ROOT/p) for p in paths+[config['gravity_protocol'],config['projection_run'],config['mixed_projection_run']]},
        response_files_opened=[],kinematic_response_scores_computed=0,goal_complete=False))
    print(dict(full_field_runs=2+len(config['numerical_followups']),thin_model_numerical_gates_pass=passed,goal_complete=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_reprojected_fields_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    args=p.parse_args()
    for k,v in vars(args).items():setattr(args,k,v.resolve())
    main(args)
