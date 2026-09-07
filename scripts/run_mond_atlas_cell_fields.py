"""Checked conditional Newton/QUMOND fields from cell-integrated source fits."""
from __future__ import annotations
import argparse
import datetime
import gc
import os
import shutil
import sys
import time
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from mond_atlas_source_cells import component_cells
from mond_atlas_force_sampling import sample_force, convergence
import mond_atlas_blocked_fields as bf


def sampling(config):
    points,metadata,groups=[],[],[]
    for z in config['sampling']['height_kpc']:
        for r in config['sampling']['radius_kpc']:
            for phi in np.arange(config['sampling']['azimuth_count'])*2*np.pi/config['sampling']['azimuth_count']:
                points.append([r*np.cos(phi),r*np.sin(phi),z])
                metadata.append(dict(radius_kpc=r,height_kpc=z,azimuth_radian=float(phi)))
                groups.append(f'{r}:{z}')
    return np.array(points),metadata,np.array(groups)


def build_case(case, grid):
    half=grid['half_width_kpc'];spacing=grid['spacing_kpc']
    axes=[np.linspace(-half,half,round(2*half/h)+1) for h in spacing]
    components,details=[],[]
    for spec in case['components']:
        path=ROOT/spec['path']
        if digest(path)!=spec['sha256']:
            raise ValueError('frozen source packet changed')
        with np.load(path) as packet:
            if not np.array_equal(packet['vertical_layers'],spec['vertical_layers']):
                raise ValueError('image and field vertical distributions differ')
            plane,vertical,receipt=component_cells(packet['intrinsic_effective_surface'],
                packet['latent_axis'],axes,spec['vertical_layers'],spec['conversion_to_msun_pc2'])
        components.append((plane,vertical))
        details.append(dict(id=spec['id'],**receipt))
    moments=bf.moments_separable(components,axes,spacing)
    expected=sum(r['full_source_mass_msun'] for r in details)
    source=dict(components=details,full_source_mass_msun=expected,
        finite_grid_mass_msun=moments['mass_msun'],
        relative_full_mass_error=abs(moments['mass_msun']/expected-1),
        mass_renormalization_applied=False)
    return components,axes,moments,source


def execute(case,grid,config,output,private,points,metadata):
    label=case['id']+'_'+grid['id'];folder=private/label
    folder.mkdir(exist_ok=False)
    start=time.monotonic();progress_log=[]
    def progress(stage):
        item=dict(stage=stage,elapsed_seconds=time.monotonic()-start)
        progress_log.append(item)
        write_json(output/(label+'-progress.json'),dict(status='RUNNING',pid=os.getpid(),label=label,**item))
    print('START '+label,flush=True)
    components,axes,moments,source=build_case(case,grid)
    spacing=grid['spacing_kpc'];shape=tuple(len(a) for a in axes)
    if source['relative_full_mass_error']>config['gates']['source_full_integral_relative']:
        raise ValueError('source cell integral lost mass before solve')
    pn=bf.array_file(folder/'newton-potential.npy',shape)
    pm=bf.array_file(folder/'mond-potential.npy',shape)
    q=bf.array_file(folder/'qumond-source.npy',shape)
    work=bf.array_file(folder/'spectral-work.npy',tuple(n-2 for n in shape))
    gravity=config['gravity'];G=gravity['G_kpc_kms2_per_msun'];a0=gravity['a0_kms2_per_kpc']
    bf.fill_boundary(pn,axes,moments,G,a0,'newton')
    bf.fill_boundary(pm,axes,moments,G,a0,'mond')
    def rhs(lo,hi):
        return 4*np.pi*G*sum(p[lo:hi,:,None]*v[None,None,:] for p,v in components)
    storage=config['storage']
    kw=dict(slab_rows=storage['physical_slab_rows'],max_elements=storage['maximum_transform_block_elements'])
    rn=bf.poisson_stream(rhs,pn,work,spacing,progress=lambda s:progress('Newton '+s),**kw)
    bf.qumond_stream(pn,q,spacing,a0,slab_rows=kw['slab_rows'],progress=progress)
    rm=bf.poisson_stream(lambda lo,hi:q[lo:hi],pm,work,spacing,progress=lambda s:progress('MOND '+s),**kw)
    origin=[a[0] for a in axes]
    gn=sample_force(pn,origin,spacing,points)
    gm=sample_force(pm,origin,spacing,points)
    rows=[]
    for i,meta in enumerate(metadata):
        row=dict(meta)
        for law,force in [('newton',gn[i]),('mond',gm[i])]:
            phi=meta['azimuth_radian']
            row.update({law+'_g'+axis:float(v) for axis,v in zip('xyz',force)})
            row[law+'_inward']=float(-force[0]*np.cos(phi)-force[1]*np.sin(phi))
            row[law+'_tangential']=float(-force[0]*np.sin(phi)+force[1]*np.cos(phi))
        rows.append(row)
    write_csv(output/(label+'-forces.csv'),rows)
    np.savez_compressed(folder/'sampled-forces.npz',points=points,newton=gn,mond=gm)
    for array in (pn,pm,q,work):array.flush()
    del pn,pm,q,work
    gc.collect()
    progress('hash saved fields')
    assets=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p),bytes=p.stat().st_size)
            for p in sorted(folder.iterdir()) if p.is_file()]
    result=dict(id=label,case_id=case['id'],grid=grid,shape=list(shape),source=source,moments=moments,
        numerical=dict(newton=rn,mond=rm),assets=assets,seconds=time.monotonic()-start,
        progress_log=progress_log,observed_response_scores=0,source_disposition='SOURCE_BLOCKED')
    write_json(output/(label+'-result.json'),result)
    write_json(output/(label+'-progress.json'),dict(status='COMPLETE',pid=os.getpid(),label=label,seconds=result['seconds']))
    print(f'DONE {label}: {result["seconds"]:.2f}s',flush=True)
    return result,gn,gm


def run(config_path,output,private):
    config=read_json(config_path)
    if config['admission_disposition']!='SOURCE_BLOCKED':
        raise ValueError('source-only numerical diagnostic required')
    if output.exists() or private.exists():
        raise FileExistsError('immutable output exists')
    if not output.is_relative_to(ROOT/'work/gravity-first-principles') or not private.is_relative_to(ROOT/'work/private'):
        raise ValueError('outside public/private research paths')
    package=ROOT/'work/gravity-first-principles/mond-atlas-ngc2976-field-001'
    freeze=read_json(package/'freeze.json')
    for path,expected in freeze['bindings'].items():
        if digest(ROOT/path)!=expected:raise ValueError('frozen input changed: '+path)
    estimates=[]
    for grid in config['grids']:
        shape=np.array([round(2*grid['half_width_kpc']/h)+1 for h in grid['spacing_kpc']],dtype=np.int64)
        estimates.append(int(8*(3*np.prod(shape)+np.prod(shape-2)))+1024**2)
    required=sum(estimates)*len(config['source_cases'])
    free=shutil.disk_usage(ROOT).free
    if free<required+config['storage']['workspace_disk_reserve_bytes']:
        raise OSError('insufficient disk for complete frozen field ensemble plus reserve')
    output.mkdir(parents=True);private.mkdir(parents=True)
    code_paths=[Path(__file__),ROOT/'scripts/mond_atlas_source_cells.py']
    code_paths += [ROOT/'tests'/(name+'.py') for name in config['benchmark_modules']]
    write_json(output/'execution-start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=os.getpid(),config_sha256=digest(config_path),frozen_inputs=freeze['bindings'],
        code_bindings={p.relative_to(ROOT).as_posix():digest(p) for p in code_paths},
        estimated_private_bytes=required,free_disk_before_bytes=free,source_packets_opened=False,
        observed_response_files_opened=[],declared_fields=16))
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in config['benchmark_modules'])
    with threadpool_limits(limits=1):
        with (output/'unit-tests.log').open('w',encoding='utf-8') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not tests.wasSuccessful() or tests.skipped:
            write_json(output/'failure.json',dict(status='BENCHMARK_FAILED',source_packets_opened=False))
            raise ValueError('independent source/field/sampling benchmark failed')
        points,metadata,groups=sampling(config)
        write_json(output/'numerical-admission.json',dict(tests_passed=tests.testsRun,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_packets_opened=False,
            observed_response_files_opened=[],disposition='SOURCE_BLOCKED'))
        results=[];checks=[]
        start=time.monotonic()
        for case in config['source_cases']:
            baseline=None
            for grid in config['grids']:
                result,gn,gm=execute(case,grid,config,output,private,points,metadata)
                results.append(result)
                if baseline is None:baseline=(gn,gm)
                else:
                    for law,ref,trial in [('newton',baseline[0],gn),('mond',baseline[1],gm)]:
                        metrics=convergence(ref,trial,groups)
                        passed=all(metrics[k] is not None and metrics[k]<config['gates'][k] for k in metrics)
                        checks.append(dict(case=case['id'],grid=grid['id'],law=law,**metrics,gates_pass=passed))
                    write_csv(output/'numerical-convergence.csv',checks)
        integrity=[]
        for result in results:
            residual=max(result['numerical'][law]['relative_pde_residual'] for law in ('newton','mond'))
            integrity.append(dict(id=result['id'],maximum_relative_pde_residual=residual,
                source_relative_mass_error=result['source']['relative_full_mass_error'],
                gates_pass=residual<config['gates']['relative_pde_residual'] and
                    result['source']['relative_full_mass_error']<config['gates']['source_full_integral_relative']))
        write_csv(output/'field-integrity.csv',integrity)
        write_json(output/'summary.json',dict(status='CONDITIONAL_CELL_INTEGRATED_FIELDS',disposition='SOURCE_BLOCKED',
            config=config,config_sha256=digest(config_path),new_field_runs=len(results),source_cases=len(config['source_cases']),
            independent_benchmark_tests=tests.testsRun,all_numerical_gates_pass=all(r['gates_pass'] for r in checks+integrity),
            checks=checks,integrity=integrity,elapsed_seconds=time.monotonic()-start,
            results=[dict(id=r['id'],result_file=(output/(r['id']+'-result.json')).relative_to(ROOT).as_posix(),
                          result_sha256=digest(output/(r['id']+'-result.json'))) for r in results],
            observed_response_files_opened=[],new_observed_gravity_scores=0,new_lensing_scores=0,goal_complete=False))
    print('COMPLETE: all numerical gates '+str(all(r['gates_pass'] for r in checks+integrity)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_ngc2976_cells_field_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args()
    run(args.config.resolve(),args.output.resolve(),args.private.resolve())
