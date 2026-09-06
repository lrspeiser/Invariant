"""Replay, then refine the mixed stellar source without loading global FFT work."""
from __future__ import annotations
import argparse,gc,io,os,shutil,time,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from run_mond_atlas_ngc2903_fields import interpolate2,force_profiles
from run_mond_atlas_reprojected_fields import vertical_cell_density
from check_mond_atlas_field_pattern import forces,vector_difference
import mond_atlas_blocked_fields as bf


def source_components(case,parent,gravity,half,spacing):
    axes=[np.linspace(-half,half,int(round(2*half/h))+1) for h in spacing]
    if any(abs((a[1]-a[0])-h)>1e-12 for a,h in zip(axes,spacing)):raise ValueError('nonintegral grid')
    x,y=np.meshgrid(axes[0],axes[1],indexing='ij');conversion=gravity['conversions']
    specs=[('stellar',case['source'],conversion['nominal_stellar_ml'],case['vertical_components']),
        ('atomic_helium',parent['gas_cases']['atomic_helium'],1.,[[1.,parent['gas_cases']['height_kpc']]]),
        ('molecular_helium',parent['gas_cases']['co21'],conversion['alpha_co10_including_helium']/conversion['co21_to_co10'],[[1.,parent['gas_cases']['height_kpc']]])]
    components=[];masses={};normalizations={}
    for name,path,factor,layers in specs:
        with np.load(ROOT/path) as packet:surface=packet['intrinsic_effective_surface']*factor*1e6;axis=packet['axis']
        if not np.isfinite(surface).all() or np.any(surface<0):raise ValueError('invalid baryon surface')
        expected=float(surface.sum()*(axis[1]-axis[0])**2);sampled=interpolate2(surface,axis,x,y)
        ratio=expected/float(sampled.sum()*spacing[0]*spacing[1]);sampled*=ratio
        if abs(sum(f for f,h in layers)-1)>1e-12 or not all(f>=0 and h>0 for f,h in layers):raise ValueError('invalid vertical components')
        vertical=sum(f*vertical_cell_density(axes[2],spacing[2],h) for f,h in layers)
        components.append((sampled,vertical));masses[name]=expected;normalizations[name]=ratio
    moments=bf.moments_separable(components,axes,spacing)
    return components,axes,moments,dict(component_mass_msun=masses,resampling_normalization=normalizations,
        finite_grid_total_mass_msun=moments['mass_msun'])


def execute(case,parent,gravity,config,half,spacing,output,private,label):
    start=time.monotonic();folder=private/label;folder.mkdir(exist_ok=False)
    components,axes,moments,source=source_components(case,parent,gravity,half,spacing)
    shape=tuple(len(a) for a in axes);interior=tuple(n-2 for n in shape)
    estimate=8*(3*int(np.prod(shape))+int(np.prod(interior)))+4096
    available=shutil.disk_usage(folder).free
    if available<estimate+config['workspace_disk_reserve_bytes']:raise OSError('insufficient workspace disk for declared field maps and reserve')
    progress_log=[]
    def progress(stage):
        item=dict(stage=stage,elapsed_seconds=time.monotonic()-start);progress_log.append(item)
        write_json(output/(label+'-progress.json'),dict(status='RUNNING',pid=os.getpid(),label=label,**item))
        print(label+': '+stage,flush=True)
    progress('allocate global disk-backed fields '+str(shape))
    pn=bf.array_file(folder/'newton-potential.npy',shape)
    pm=bf.array_file(folder/'mond-potential.npy',shape)
    q=bf.array_file(folder/'qumond-source.npy',shape)
    work=bf.array_file(folder/'spectral-work.npy',interior)
    G=gravity['gravity']['G_kpc_kms2_per_msun'];a0=gravity['gravity']['a0_kms2_per_kpc']
    bf.fill_boundary(pn,axes,moments,G,a0,'newton');bf.fill_boundary(pm,axes,moments,G,a0,'mond')
    def source_block(lo,hi):
        rho=sum(surface[lo:hi,:,None]*vertical[None,None,:] for surface,vertical in components)
        return 4*np.pi*G*rho
    kw=dict(slab_rows=config['physical_slab_rows'],max_elements=config['maximum_transform_block_elements'])
    rn=bf.poisson_stream(source_block,pn,work,spacing,progress=lambda s:progress('Newton '+s),**kw)
    progress('Newton solve complete; generate combined-field nonlinear source')
    bf.qumond_stream(pn,q,spacing,a0,slab_rows=config['physical_slab_rows'],progress=progress)
    rm=bf.poisson_stream(lambda lo,hi:q[lo:hi],pm,work,spacing,progress=lambda s:progress('MOND '+s),**kw)
    rows,profile,plane=force_profiles(pn,pm,axes,spacing,gravity)
    write_csv(output/(label+'-forces.csv'),rows);write_csv(output/(label+'-profile.csv'),profile)
    np.savez_compressed(folder/'midplane.npz',**plane)
    progress('hash retained field maps')
    for a in (pn,pm,q,work):a.flush()
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(folder.iterdir())]
    result=dict(id=label,shape=list(shape),half_width_kpc=half,spacing_kpc=spacing,source=source,moments=moments,
        numerical=dict(newton=rn,mond=rm),profile=profile,seconds=time.monotonic()-start,files=files,
        field_disk_estimate_bytes=estimate,free_disk_before_allocation_bytes=available,
        storage_scope='Global field maps are disk backed; temporary FFT and flux arrays use declared blocks. Resident OS page cache is not measured or bounded here.',
        progress_log=progress_log)
    write_json(output/(label+'-result.json'),result)
    write_json(output/(label+'-progress.json'),dict(status='COMPLETE',label=label,pid=os.getpid(),seconds=result['seconds']))
    print(label+' COMPLETE '+str(round(result['seconds'],2))+' seconds',flush=True)
    return result


def main(args):
    config=read_json(args.config);assert config['admission_disposition']=='SOURCE_BLOCKED'
    replay=config['real_source_validation_replay'];parent=read_json(ROOT/replay['parent_config']);gravity=read_json(ROOT/parent['gravity_protocol'])
    prior=read_json(ROOT/'work/gravity-first-principles/mond-atlas-field-003/summary.json')
    assert digest(ROOT/replay['parent_config'])==prior['config_sha256']
    for group in ('source_bindings','code_hashes'):
        for path,expected in prior[group].items():assert digest(ROOT/path)==expected,path
    case=next(c for c in parent['stellar_cases'] if c['id']==replay['source_case'])
    if args.output.exists() or args.private.exists():raise FileExistsError('immutable outputs')
    args.output.mkdir(parents=True);args.private.mkdir(parents=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_blocked.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (args.output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    execute(case,parent,gravity,config,replay['half_width_kpc'],replay['spacing_kpc'],args.output,args.private,'mixed_replay')
    original=ROOT/replay['existing_result'];original_label=read_json(original)['id']
    comparison=vector_difference(forces(original.parent,original_label),forces(args.output,'mixed_replay'))
    max_replay=max(comparison.values());replay_pass=max_replay<config['benchmarks']['source_replay_relative_force_error_max']
    write_json(args.output/'replay-audit.json',dict(status='PASS' if replay_pass else 'BENCHMARK_FAILED',comparison=comparison,
        reference_result=str(original.relative_to(ROOT)),reference_sha256=digest(original)))
    if not replay_pass:raise RuntimeError('existing galaxy replay failed before refinement')
    gc.collect();fine=config['required_refinement']
    execute(case,parent,gravity,config,fine['half_width_kpc'],fine['spacing_kpc'],args.output,args.private,'mixed_finer')
    refinement=vector_difference(forces(args.output,'mixed_replay'),forces(args.output,'mixed_finer'))
    passed=all(v<(config['benchmarks']['mixed_refinement_maximum_ring_relative_rms_max'] if 'maximum_ring' in k else config['benchmarks']['mixed_refinement_vector_relative_rms_max']) for k,v in refinement.items())
    write_json(args.output/'summary.json',dict(status='MIXED_SOURCE_GLOBAL_REFINEMENT',admission_disposition='SOURCE_BLOCKED',
        config=config,config_sha256=digest(args.config),replay_pass=replay_pass,replay=comparison,mixed_refinement=refinement,
        mixed_refinement_gates_pass=passed,source_models_changed=0,new_full_field_runs=2,numerical_benchmark_tests=tests.testsRun,
        input_bindings={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/replay['parent_config'],ROOT/parent['gravity_protocol'],original)},
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'scripts/mond_atlas_blocked_fields.py',ROOT/'scripts/mond_atlas_rectangular_fields.py',ROOT/'scripts/run_mond_atlas_reprojected_fields.py',ROOT/'tests/test_mond_atlas_blocked.py')},
        response_files_opened=[],kinematic_response_scores_computed=0,admitted_galaxy_cube_predictions=0,goal_complete=False))
    print(dict(replay_pass=replay_pass,refinement=refinement,mixed_refinement_gates_pass=passed,goal_complete=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_blocked_fields_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():setattr(a,k,v.resolve())
    main(a)
