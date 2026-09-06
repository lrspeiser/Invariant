"""Run every declared stationary-noise surrogate on the actual pilot geometry."""
from __future__ import annotations
import argparse,csv,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest,fits_primary_header
from mond_atlas_smoothing_null import native_to_coarse_operator,normalized_covariance,selected_channel_covariance,draw_separable,center_outer_statistic


def run(config_path,output):
    config=read_json(config_path);native_path=ROOT/config['native_summary'];native=read_json(native_path)
    replay_path=ROOT/config['preprocessing_summary'];replay=read_json(replay_path)
    preprocessing=read_json(ROOT/config['preprocessing_protocol'])
    if config['admission_disposition']!='SOURCE_BLOCKED' or not replay['all_replay_gates_pass']:raise ValueError('unvalidated prerequisite')
    checked={}
    for summary in [native,replay]:
        for group in ['bindings','source_bindings']:
            for relative,expected in summary[group].items():
                if relative not in checked:checked[relative]=digest(ROOT/relative)
                if checked[relative]!=expected:raise ValueError('prerequisite changed: '+relative)
    if output.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths=[config_path,native_path,replay_path,ROOT/config['preprocessing_protocol'],Path(__file__),
        ROOT/'scripts/mond_atlas_smoothing_null.py',ROOT/'scripts/mond_atlas_preprocessing.py',
        ROOT/'scripts/mond_atlas_native_spectral.py',ROOT/'tests/test_mond_atlas_smoothing_null.py']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,observed_statistics_already_development_exposed=True))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_smoothing_null.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    with (replay_path.parent/'galaxies.csv').open(newline='') as stream:observed={r['galaxy']:r for r in csv.DictReader(stream) if int(r['candidate_channels'])>0}
    audit_path=ROOT/native['config']['cached_cube_audit'];audits={r['name']:r for r in read_json(audit_path)}
    sources={str(audit_path.relative_to(ROOT)):digest(audit_path)};rows=[];cases=[];ensemble={};observed_ensemble={}
    factor=preprocessing['operation']['block_factor'];spatial=preprocessing['spatial_diagnostic']
    for galaxy_index,(name,observation) in enumerate(sorted(observed.items())):
        record_path=native_path.parent/(name+'.json');record=read_json(record_path);sources[str(record_path.relative_to(ROOT))]=digest(record_path)
        h=fits_primary_header(ROOT/record['source_path']);audit=audits[name];pixel=abs(h['CDELT1'])*3600
        nx,ny=int(h['NAXIS1']),int(h['NAXIS2']);yy,xx=np.indices((ny//factor,nx//factor))
        x=xx*factor+(factor-1)/2;y=yy*factor+(factor-1)/2
        radius=np.hypot((x+1-h['CRPIX1'])*h['CDELT1']*3600,(y+1-h['CRPIX2'])*h['CDELT2']*3600)
        a,b=spatial['inner_projection_plane_radius_arcsec'];inner=(radius>=a)&(radius<b)
        a,b=spatial['outer_projection_plane_radius_arcsec'];outer=(radius>=a)&(radius<b)
        target=float(observation['coarse_median_inner_outer_scale_ratio'])
        for spatial_index,spatial_name in enumerate(config['spatial_surrogates']):
            native_sigma=0. if spatial_name=='independent_native_pixels' else audit['native_beam_arcsec'][0]/(2.354820045*pixel)
            extra_sigma=audit['extra_smoothing_sigma_arcsec']/pixel
            oy=native_to_coarse_operator(ny,factor,extra_sigma,native_sigma);ox=native_to_coarse_operator(nx,factor,extra_sigma,native_sigma)
            cy,ly=normalized_covariance(oy);cx,lx=normalized_covariance(ox)
            for spectral_index,spectral_name in enumerate(config['spectral_surrogates']):
                cc,lc=selected_channel_covariance(record['provenance'],spectral_name=='hanning_parent_channels')
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],galaxy_index,spatial_index,spectral_index]))
                values=[]
                for start in range(0,config['replicates_per_case'],config['batch_replicates']):
                    batch=min(config['batch_replicates'],config['replicates_per_case']-start)
                    draws=draw_separable(rng,batch,lc,ly,lx);values.extend(center_outer_statistic(draws,inner,outer).tolist())
                values=np.array(values);key=(spatial_name,spectral_name)
                ensemble.setdefault(key,[]).append(values);observed_ensemble.setdefault(key,[]).append(target)
                for index,value in enumerate(values):rows.append(dict(galaxy=name,spatial_surrogate=spatial_name,spectral_surrogate=spectral_name,replicate=index,simulated_median_channel_mad_ratio=value))
                q=np.quantile(values,config['interval_quantiles']);exceed=int(np.sum(values>=target))
                case=dict(galaxy=name,spatial_surrogate=spatial_name,spectral_surrogate=spectral_name,
                    channels=len(cc),replicates=len(values),observed_median_channel_mad_ratio=target,
                    reference_q025=float(q[0]),reference_median=float(q[1]),reference_q975=float(q[2]),
                    upper_tail_reference_count=exceed,upper_tail_reference_fraction=(1+exceed)/(len(values)+1),
                    observed_outside_reference_interval=bool(target<q[0] or target>q[2]),
                    native_correlation_filter_sigma_pixels=native_sigma,extra_filter_sigma_pixels=extra_sigma,
                    conditional_gaussian_surrogate_only=True,observational_p_value=False)
                cases.append(case);print(name,spatial_name,spectral_name,'obs',round(target,3),'interval',np.round(q[[0,2]],3).tolist(),'tail',round(case['upper_tail_reference_fraction'],4),flush=True)
    joint=[]
    for key,arrays in ensemble.items():
        values=np.mean(np.log(np.array(arrays)),axis=0);target=float(np.mean(np.log(observed_ensemble[key])));q=np.quantile(values,config['interval_quantiles']);exceed=int(np.sum(values>=target))
        joint.append(dict(spatial_surrogate=key[0],spectral_surrogate=key[1],galaxies=len(arrays),replicates=len(values),
            observed_mean_log_galaxy_ratio=target,reference_q025=float(q[0]),reference_median=float(q[1]),reference_q975=float(q[2]),
            upper_tail_reference_count=exceed,upper_tail_reference_fraction=(1+exceed)/(len(values)+1),
            observational_p_value=False))
    write_csv(output/'replicates.csv',rows);write_csv(output/'galaxy-cases.csv',cases);write_csv(output/'ensemble-cases.csv',joint)
    result=dict(status='CONDITIONAL_STATIONARY_SMOOTHING_NULL_EXECUTED',admission_disposition='SOURCE_BLOCKED',config=config,
        bindings=bindings,source_bindings=sources,galaxies=len(observed),surrogate_cases=len(cases),
        galaxy_statistic_replicates=len(rows),conditional_ensemble_cases=joint,independent_unit_tests=tests.testsRun,
        observed_ratios_outside_reference_interval=[dict(galaxy=c['galaxy'],spatial=c['spatial_surrogate'],spectral=c['spectral_surrogate']) for c in cases if c['observed_outside_reference_interval']],
        actual_instrument_covariance_validated=False,selection_mask_validated=False,new_motion_fits=0,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print(dict(galaxies=len(observed),cases=len(cases),replicates=len(rows),ensemble=joint),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_smoothing_null_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
