"""Check actual three-component fields above the disk, retaining all failures."""
from __future__ import annotations
import argparse, io, unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_force_sampling import sample_force,convergence


def run(config_path,output):
    config=read_json(config_path)
    if config['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('conditional numerical audit only')
    folder=ROOT/config['field_directory'];summary=read_json(folder/'summary.json')
    protocol=read_json(ROOT/config['field_protocol'])
    if digest(ROOT/config['field_protocol'])!=summary['config_sha256']:raise ValueError('field protocol changed')
    for path,expected in summary['bindings'].items():
        if digest(ROOT/path)!=expected:raise ValueError('field input changed: '+path)
    if output.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths=[config_path,Path(__file__),ROOT/'scripts/mond_atlas_force_sampling.py',
           ROOT/'tests/test_mond_atlas_force_sampling.py',folder/'summary.json',ROOT/config['field_protocol']]
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_force_sampling.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    labels=np.array([(r,z,a) for z in config['heights_kpc'] for r in config['radii_kpc']
                     for a in np.arange(config['azimuth_count'])*360/config['azimuth_count']],float)
    radius,height,angle=labels.T;theta=np.deg2rad(angle)
    points=np.column_stack((radius*np.cos(theta),radius*np.sin(theta),height))
    mirrored=points.copy();mirrored[:,2]*=-1
    groups=np.repeat(np.arange(len(config['heights_kpc'])*len(config['radii_kpc'])),config['azimuth_count'])
    results={};checks=[];reflection=[];assets=[]
    for case in protocol['stellar_cases']:
        for grid in protocol['grids']:
            label=case['id']+'_'+grid['id'];path=folder/(label+'-result.json');result=read_json(path)
            bindings[str(path.relative_to(ROOT))]=digest(path)
            sample={};rows=[]
            for theory,filename in [('newton','newton-potential.npy'),('mond','mond-potential.npy')]:
                asset=next(p for p in result['files'] if Path(p['path']).name==filename)
                if digest(ROOT/asset['path'])!=asset['sha256']:raise ValueError('potential changed')
                potential=np.load(ROOT/asset['path'],mmap_mode='r')
                f=sample_force(potential,[-result['half_width_kpc']]*3,result['spacing_kpc'],points)
                reflected=sample_force(potential,[-result['half_width_kpc']]*3,result['spacing_kpc'],mirrored)
                reflected[:,2]*=-1
                error=float(np.sqrt(np.sum((f-reflected)**2)/np.sum(f**2)))
                reflection.append(dict(case=label,theory=theory,z_reflection_relative_rms=error,
                    gate_pass=error<config['benchmarks']['z_reflection_relative_rms_max']))
                assets.append(asset);sample[theory]=f
                inward=-(f[:,0]*np.cos(theta)+f[:,1]*np.sin(theta))
                tangential=-f[:,0]*np.sin(theta)+f[:,1]*np.cos(theta)
                for i,(r,z,a) in enumerate(labels):
                    rows.append(dict(theory=theory,radius_kpc=r,height_kpc=z,angle_deg=a,
                        gx_kms2_per_kpc=f[i,0],gy_kms2_per_kpc=f[i,1],gz_kms2_per_kpc=f[i,2],
                        inward_kms2_per_kpc=inward[i],tangential_kms2_per_kpc=tangential[i],
                        downward_kms2_per_kpc=-f[i,2]))
                del potential
            write_csv(output/(label+'-forces.csv'),rows);results[label]=sample
            if grid['id']!='base':
                for theory in ('newton','mond'):
                    comparison=convergence(results[case['id']+'_base'][theory],sample[theory],groups)
                    passed=(comparison['vector_relative_rms']<config['benchmarks']['aggregate_three_component_force_relative_rms_max'] and
                            comparison['maximum_group_relative_rms']<config['benchmarks']['maximum_radius_height_group_force_relative_rms_max'])
                    checks.append(dict(case=case['id'],perturbation=grid['id'],theory=theory,**comparison,full_vector_gates_pass=passed))
            print(label+' off-plane sampled and potential hashes verified',flush=True)
    sensitivity=[]
    for theory in ('newton','mond'):
        thin=results['common_thin_lateral'][theory];mixed=results['common_mixed_lateral'][theory]
        for group in np.unique(groups):
            use=groups==group;t=theta[use];a=thin[use];b=mixed[use]
            inward_a=-(a[:,0]*np.cos(t)+a[:,1]*np.sin(t));inward_b=-(b[:,0]*np.cos(t)+b[:,1]*np.sin(t))
            tangential_a=-a[:,0]*np.sin(t)+a[:,1]*np.cos(t);tangential_b=-b[:,0]*np.sin(t)+b[:,1]*np.cos(t)
            r,z,_=labels[use][0]
            sensitivity.append(dict(theory=theory,radius_kpc=r,height_kpc=z,
                thin_mean_inward=float(inward_a.mean()),mixed_mean_inward=float(inward_b.mean()),
                inward_mean_fractional_change=float(inward_b.mean()/inward_a.mean()-1),
                thin_mean_downward=float(-a[:,2].mean()),mixed_mean_downward=float(-b[:,2].mean()),
                downward_mean_fractional_change=float(b[:,2].mean()/a[:,2].mean()-1),
                thin_tangential_rms_over_inward_mean=float(np.sqrt(np.mean(tangential_a**2))/inward_a.mean()),
                mixed_tangential_rms_over_inward_mean=float(np.sqrt(np.mean(tangential_b**2))/inward_b.mean()),
                vector_difference_over_thin_rms=float(np.sqrt(np.sum((b-a)**2)/np.sum(a**2)))))
    write_csv(output/'numerical-checks.csv',checks);write_csv(output/'reflection-checks.csv',reflection)
    write_csv(output/'conditional-source-sensitivity.csv',sensitivity)
    passed=all(c['full_vector_gates_pass'] for c in checks) and all(c['gate_pass'] for c in reflection)
    write_json(output/'summary.json',dict(status='OFFPLANE_THREE_COMPONENT_NUMERICAL_AUDIT',
        admission_disposition='SOURCE_BLOCKED',config=config,bindings=bindings,verified_potential_assets=assets,
        full_field_runs_sampled=len(results),force_sampling_points_per_field=len(points),unit_tests_passed=tests.testsRun,
        checks=checks,offplane_full_vector_gates_pass=passed,
        maximum_vertical_component_relative_difference=max(c['vertical_component_relative_rms'] for c in checks),
        reflection_gates_pass=all(c['gate_pass'] for c in reflection),
        kinematic_response_scores_computed=0,admitted_galaxy_cube_predictions=0,goal_complete=False))
    print(dict(offplane_full_vector_gates_pass=passed,checks=checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_offplane_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
