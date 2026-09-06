"""Replay selected native planes and assess coarse spatial scale transfer."""
from __future__ import annotations
import argparse,csv,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_spectral import NativeCube,robust_region
from mond_atlas_preprocessing import gaussian_plane_float32,block_mean,original_offset_mask


def run(config_path,output):
    config=read_json(config_path);native_config=read_json(ROOT/config['native_protocol'])
    native_path=ROOT/config['native_summary'];native_result=read_json(native_path)
    cached_bindings=read_json(ROOT/config['cached_packet_bindings'])['source_bindings']
    if config['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('instrument diagnostic only')
    for relative,expected in native_result['bindings'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('native stage binding changed: '+relative)
    if output.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths=[config_path,ROOT/config['native_protocol'],native_path,ROOT/config['cached_packet_bindings'],Path(__file__),
        ROOT/'scripts/mond_atlas_native_spectral.py',ROOT/'scripts/mond_atlas_preprocessing.py',
        ROOT/'tests/test_mond_atlas_preprocessing.py',ROOT/'scripts/run_gravity_cube_pilot.py']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,new_motion_fits=0))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_preprocessing.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    audits={r['name']:r for r in read_json(ROOT/native_config['cached_cube_audit'])}
    with (native_path.parent/'native-candidate-channel-diagnostics.csv').open(newline='') as stream:
        native_rows={(r['galaxy'],int(r['stored_channel_index'])):r for r in csv.DictReader(stream)}
    sources={};rows=[];galaxies=[]
    for item in sorted(read_json(ROOT/native_config['cube_receipt'])['files'],key=lambda r:r['name']):
        name=item['name'];parent_path=native_path.parent/(name+'.json');parent=read_json(parent_path)
        sources[str(parent_path.relative_to(ROOT))]=digest(parent_path)
        selected=parent['provenance']['retained_continuum_fit_stored_indices']
        if not selected:
            galaxies.append(dict(galaxy=name,candidate_channels=0,status='NO_DIRECTLY_MAPPED_RETAINED_CANDIDATES',replay_pass=None));continue
        cube_path=ROOT/item['file'];packet_path=ROOT/config['cached_packets']/(name+'.npz')
        for path,expected in [(cube_path,item['sha256']),(packet_path,cached_bindings[str(packet_path.relative_to(ROOT))])]:
            actual=digest(path)
            if actual!=expected:raise ValueError('source changed: '+str(path))
            sources[str(path.relative_to(ROOT))]=actual
        cube=NativeCube(cube_path)
        try:
            h=cube.header;audit=audits[name]
            if h['CTYPE3']!='FELO-HEL' or h['CDELT3']>=0 or audit['spectral_input']!='VOPT-F2W':raise ValueError('unsupported spectral-order replay contract')
            with np.load(packet_path,allow_pickle=False) as packet:
                east,north,cached=packet['east'],packet['north'],packet['cube']
            operation=config['operation'];factor=operation['block_factor'];offset_mask=original_offset_mask(east,north,operation)
            yy,xx=np.indices(east.shape);native_x=xx*factor+(factor-1)/2;native_y=yy*factor+(factor-1)/2
            radius=np.hypot((native_x+1-h['CRPIX1'])*h['CDELT1']*3600,(native_y+1-h['CRPIX2'])*h['CDELT2']*3600)
            inner_range=config['spatial_diagnostic']['inner_projection_plane_radius_arcsec'];outer_range=config['spatial_diagnostic']['outer_projection_plane_radius_arcsec']
            inner=(radius>=inner_range[0])&(radius<inner_range[1]);outer=(radius>=outer_range[0])&(radius<outer_range[1])
            sigma=audit['extra_smoothing_sigma_arcsec']/(abs(h['CDELT1'])*3600)
            details=[]
            for channel in selected:
                native=cube.sample_plane(channel).astype(np.float32)
                smoothed=gaussian_plane_float32(native,sigma,operation['gaussian_truncate_sigma'])
                coarse=block_mean(smoothed,factor);offset=float(coarse[offset_mask].astype(float).mean());replayed=coarse-offset
                # The historical subtraction was float32 minus a float64 vector
                # broadcast over a cube, which promotes the complete cube to float64.
                replayed=coarse.astype(float)-offset
                stored=cube.shape[0]-1-channel;reference=cached[stored]
                difference=replayed-reference;absolute=float(np.max(np.abs(difference)))
                relative=float(np.sqrt(np.sum(difference*difference)/np.sum(reference*reference)))
                a,_=robust_region(replayed[inner]);b,_=robust_region(replayed[outer])
                gate=absolute<=config['replay_gates']['maximum_absolute_error_jy_per_native_beam'] and relative<=config['replay_gates']['relative_image_rms_error_max']
                native_row=native_rows[name,channel]
                row=dict(galaxy=name,native_stored_channel_index=channel,cached_channel_index=stored,
                    maximum_absolute_replay_error=absolute,relative_rms_replay_error=relative,replay_pass=gate,
                    original_background_offset_jy_per_native_beam=offset,gaussian_sigma_native_pixels=sigma,
                    inner_coarse_pixels=a['pixels'],outer_coarse_pixels=b['pixels'],
                    inner_coarse_mad_jy_per_native_beam=a['mad_scale_jy_per_beam'],outer_coarse_mad_jy_per_native_beam=b['mad_scale_jy_per_beam'],
                    coarse_inner_to_outer_mad_ratio=a['mad_scale_jy_per_beam']/b['mad_scale_jy_per_beam'],
                    native_inner_to_outer_mad_ratio=float(native_row['inner_to_outer_mad_scale_ratio']),
                    coarse_inner_tail_ratio=a['upper_to_lower_90pct_tail_ratio'],coarse_outer_tail_ratio=b['upper_to_lower_90pct_tail_ratio'],
                    coarse_inner_above3_mad=a['above_3_mad_fraction'],coarse_inner_belowminus3_mad=a['below_minus3_mad_fraction'],
                    coarse_outer_above3_mad=b['above_3_mad_fraction'],coarse_outer_belowminus3_mad=b['below_minus3_mad_fraction'])
                rows.append(row);details.append(row)
            summary=dict(galaxy=name,candidate_channels=len(selected),status='SPATIAL_PREPROCESSING_REPLAY_EXECUTED',
                replay_pass=all(r['replay_pass'] for r in details),maximum_absolute_replay_error=max(r['maximum_absolute_replay_error'] for r in details),
                maximum_relative_rms_replay_error=max(r['relative_rms_replay_error'] for r in details),
                native_median_inner_outer_scale_ratio=float(np.median([r['native_inner_to_outer_mad_ratio'] for r in details])),
                coarse_median_inner_outer_scale_ratio=float(np.median([r['coarse_inner_to_outer_mad_ratio'] for r in details])),
                coarse_inner_outer_scale_ratio_min=min(r['coarse_inner_to_outer_mad_ratio'] for r in details),
                coarse_inner_outer_scale_ratio_max=max(r['coarse_inner_to_outer_mad_ratio'] for r in details))
            galaxies.append(summary);write_json(output/(name+'.json'),dict(summary=summary,channels=details));print(summary,flush=True)
        finally:cube.close()
    write_csv(output/'galaxies.csv',galaxies);write_csv(output/'channels.csv',rows)
    result=dict(status='PREPROCESSING_REPLAY_PASS' if all(r['replay_pass'] for r in rows) else 'PREPROCESSING_REPLAY_FAILED',
        admission_disposition='SOURCE_BLOCKED',config=config,bindings=bindings,source_bindings=sources,
        source_galaxies=len(galaxies),replayed_galaxies=sum(r['candidate_channels']>0 for r in galaxies),replayed_channels=len(rows),
        all_declared_candidates_replayed=len(rows)==native_result['retained_historical_continuum_candidate_channels'],
        all_replay_gates_pass=bool(rows) and all(r['replay_pass'] for r in rows),independent_unit_tests=tests.testsRun,
        physically_complete_instrument_response_validated=False,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print({k:result[k] for k in ['status','replayed_galaxies','replayed_channels','all_declared_candidates_replayed']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_preprocessing_replay_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
