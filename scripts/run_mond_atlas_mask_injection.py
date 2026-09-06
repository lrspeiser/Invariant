"""Execute paired signal injections for the fixed consecutive-channel rule."""
from __future__ import annotations
import argparse,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_mask_injection import noise_cube,source_template,consecutive_mask,recovery_metrics


def summarize(rows,field):
    values=np.array([r[field] for r in rows],float);q=np.quantile(values,[.025,.5,.975])
    return {field+'_mean':float(np.mean(values)),field+'_q025':float(q[0]),field+'_median':float(q[1]),field+'_q975':float(q[2])}


def run(config_path,output):
    config=read_json(config_path)
    if config['admission_disposition']!='SOURCE_BLOCKED' or config['threshold_fitting_permitted']:raise ValueError('fixed control scope required')
    if output.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths=[config_path,Path(__file__),ROOT/'scripts/mond_atlas_mask_injection.py',ROOT/'scripts/mond_atlas_smoothing_null.py',
        ROOT/'scripts/mond_atlas_preprocessing.py',ROOT/'tests/test_mond_atlas_mask_injection.py']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,real_galaxy_motion_values_used=False))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_mask_injection.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    instrument=config['instrument'];sources=config['sources'];detection=config['detection'];simulation=config['simulation']
    rows=[];noise_rows=[];cases=[];noiseless=[];noise_summaries=[]
    for branch_index,branch in enumerate(instrument['spectral_branches']):
        templates=[]
        for spatial in sources['intrinsic_spatial_gaussian_fwhm_arcsec']:
            for spectral in sources['intrinsic_spectral_gaussian_fwhm_channels']:
                template,center=source_template(instrument,spatial,spectral,branch)
                templates.append((spatial,spectral,template,center))
                for amplitude in sources['observed_peak_snr']:
                    noiseless.append(dict(spectral_branch=branch,spatial_fwhm_arcsec=spatial,spectral_fwhm_channels=spectral,
                        observed_peak_snr=amplitude,**recovery_metrics(amplitude*template,template,amplitude,center,detection)))
        rng=np.random.default_rng(np.random.SeedSequence([simulation['seed'],branch_index]))
        for replicate in range(simulation['noise_realizations_per_branch']):
            noise=noise_cube(rng,instrument,branch)
            mask=consecutive_mask(noise,detection['threshold_sigma'],detection['consecutive_channels'])
            noise_rows.append(dict(spectral_branch=branch,replicate=replicate,
                noise_voxel_fraction_selected=float(mask.mean()),selected_noise_flux=float(noise[mask].sum()),
                any_noise_selected=bool(mask.any()),noise_mean_square=float(np.mean(noise*noise))))
            for spatial,spectral,template,center in templates:
                for amplitude in sources['observed_peak_snr']:
                    rows.append(dict(spectral_branch=branch,replicate=replicate,spatial_fwhm_arcsec=spatial,
                        spectral_fwhm_channels=spectral,observed_peak_snr=amplitude,
                        **recovery_metrics(noise+amplitude*template,template,amplitude,center,detection)))
            if (replicate+1)%16==0:print(branch,'completed noise realizations',replicate+1,'/',simulation['noise_realizations_per_branch'],flush=True)
        for spatial,spectral,template,center in templates:
            for amplitude in sources['observed_peak_snr']:
                selected=[r for r in rows if r['spectral_branch']==branch and r['spatial_fwhm_arcsec']==spatial and r['spectral_fwhm_channels']==spectral and r['observed_peak_snr']==amplitude]
                cases.append(dict(spectral_branch=branch,spatial_fwhm_arcsec=spatial,spectral_fwhm_channels=spectral,observed_peak_snr=amplitude,
                    paired_noise_realizations=len(selected),peak_selection_fraction=float(np.mean([r['peak_selected'] for r in selected])),
                    fraction_retaining_at_least_half_true_flux=float(np.mean([r['true_flux_fraction_retained']>=.5 for r in selected])),
                    **summarize(selected,'true_flux_fraction_retained'),**summarize(selected,'measured_selected_flux_over_true')))
        selected=[r for r in noise_rows if r['spectral_branch']==branch]
        noise_summaries.append(dict(spectral_branch=branch,noise_realizations=len(selected),
            **summarize(selected,'noise_voxel_fraction_selected'),**summarize(selected,'selected_noise_flux'),
            any_noise_selection_fraction=float(np.mean([r['any_noise_selected'] for r in selected])),
            average_noise_mean_square=float(np.mean([r['noise_mean_square'] for r in selected]))))
        write_csv(output/(branch+'-cases.csv'),[r for r in cases if r['spectral_branch']==branch])
    write_csv(output/'injection-trials.csv',rows);write_csv(output/'noise-trials.csv',noise_rows)
    write_csv(output/'injection-cases.csv',cases);write_csv(output/'noise-summary.csv',noise_summaries);write_csv(output/'noiseless-cases.csv',noiseless)
    result=dict(status='FIXED_CONSECUTIVE_CHANNEL_MASK_INJECTION_CONTROLS_EXECUTED',admission_disposition='SOURCE_BLOCKED',config=config,bindings=bindings,
        spectral_branches=len(instrument['spectral_branches']),source_families_per_branch=len(templates),
        noise_realizations=len(noise_rows),injection_cases=len(cases),paired_injection_trials=len(rows),noiseless_controls=len(noiseless),
        noise_summaries=noise_summaries,independent_unit_tests=tests.testsRun,
        actual_THINGS_mask_reconstructed=False,real_galaxy_selection_function_validated=False,
        new_motion_fits=0,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print({k:result[k] for k in ['noise_realizations','injection_cases','paired_injection_trials','noiseless_controls','noise_summaries']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_mask_injection_v1.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
