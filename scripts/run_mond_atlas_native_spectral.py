"""Execute native spectral provenance and candidate-channel spatial diagnostics."""
from __future__ import annotations
import argparse,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_spectral import NativeCube,history_provenance,continuum_controls,robust_region


def spatial_diagnostic(cube,selected,config):
    if not selected:return dict(status='NO_RETAINED_HISTORICAL_CONTINUUM_CHANNELS',channels=[]),[]
    stride=config['sample_native_pixel_stride'];h=cube.header
    yy,xx=np.mgrid[0:cube.shape[1]:stride,0:cube.shape[2]:stride]
    radius=np.hypot((xx+1-h['CRPIX1'])*h['CDELT1']*3600,(yy+1-h['CRPIX2'])*h['CDELT2']*3600)
    border=config['pixel_border'];interior=(xx>=border)&(yy>=border)&(xx<cube.shape[2]-border)&(yy<cube.shape[1]-border)
    regions={}
    for role,key in [('inner','inner_radius_arcsec'),('outer','outer_radius_arcsec')]:
        lo,hi=config[key];regions[role]=interior&(radius>=lo)&(radius<hi)
    rows=[];z={role:{} for role in regions}
    for channel in selected:
        plane=cube.sample_plane(channel,stride);row=dict(stored_channel_index=channel)
        for role,mask in regions.items():
            stats,standardized=robust_region(plane[mask]);z[role][channel]=standardized
            row.update({role+'_'+key:value for key,value in stats.items()})
        row['inner_to_outer_mad_scale_ratio']=row['inner_mad_scale_jy_per_beam']/row['outer_mad_scale_jy_per_beam']
        rows.append(row)
    pairs=[(i,i+1) for i in selected if i+1 in selected]
    result=dict(status='HISTORICAL_CONTINUUM_CHANNEL_SPATIAL_DIAGNOSTIC',channels=selected,channel_count=len(selected),
        projection_plane_radius_from_FITS_reference_pixel=True,spatial_sampling_stride_native_pixels=stride,
        inner_pixels=int(regions['inner'].sum()),outer_pixels=int(regions['outer'].sum()),
        inner_to_outer_mad_ratio_min=min(r['inner_to_outer_mad_scale_ratio'] for r in rows),
        inner_to_outer_mad_ratio_median=float(np.median([r['inner_to_outer_mad_scale_ratio'] for r in rows])),
        inner_to_outer_mad_ratio_max=max(r['inner_to_outer_mad_scale_ratio'] for r in rows),adjacent_stored_channel_pairs=pairs,
        limitations=['Historical continuum selection is not a proof of zero HI or an independent epoch.',
            'Native spatial samples are correlated; sample counts do not supply independent-beam significance.',
            'Robust scales are compared on the same historically selected channels; they do not validate transfer to other channels.'])
    for role in regions:
        products=np.concatenate([np.clip(z[role][a],-3,3)*np.clip(z[role][b],-3,3) for a,b in pairs]) if pairs else None
        result[role+'_adjacent_clipped_product']=float(np.mean(products)) if products is not None else None
    return result,rows


def run(config_path,output):
    config=read_json(config_path)
    if config['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('source/instrument diagnostic only')
    if output.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths=[config_path,ROOT/config['cube_receipt'],ROOT/config['cached_cube_audit'],Path(__file__),
        ROOT/'scripts/mond_atlas_native_spectral.py',ROOT/'scripts/mond_atlas_common.py',ROOT/'tests/test_mond_atlas_native_spectral.py']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,
        exposure='Native header histories were inspected to declare this contract. Existing galaxies are development-exposed. No new galaxy motion fit occurs.'))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_native_spectral.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    rows=[];measurements=[];sources={}
    for item in sorted(read_json(ROOT/config['cube_receipt'])['files'],key=lambda r:r['name']):
        path=ROOT/item['file'];actual=digest(path)
        if actual!=item['sha256']:raise ValueError('native cube hash changed: '+item['name'])
        sources[str(path.relative_to(ROOT))]=actual;cube=NativeCube(path)
        try:
            if cube.header['BUNIT'].strip()!='JY/BEAM':raise ValueError('unexpected native units')
            provenance=history_provenance(cube.cards,cube.shape[0]);control=continuum_controls(provenance)
            if provenance['direct_channel_mapping']:
                spatial,channel_rows=spatial_diagnostic(cube,provenance['retained_continuum_fit_stored_indices'],config['native_spatial_transfer_diagnostic'])
            else:spatial,channel_rows=dict(status='HISTORY_MAPPING_UNRESOLVED'),[]
            measurements.extend(dict(galaxy=item['name'],**row) for row in channel_rows)
            header_path=output/(item['name']+'-header.txt');header_path.write_text('\n'.join(cube.cards)+'\n',encoding='ascii',newline='\n')
            result=dict(galaxy=item['name'],source_path=str(path.relative_to(ROOT)),source_sha256=actual,shape=cube.shape,
                header_bytes=cube.offset,original_padded_header_sha256=cube.header_sha256,
                extracted_header_text_sha256=digest(header_path),provenance=provenance,
                conditional_continuum_covariance=control,native_spatial_diagnostic=spatial,
                new_gravity_motion_scores=0,admitted_cube_likelihood=False)
            write_json(output/(item['name']+'.json'),result)
            summary=dict(galaxy=item['name'],stored_channels=cube.shape[0],uvlin_history_groups=len(provenance['uvlin_groups']),
                direct_channel_mapping=provenance['direct_channel_mapping'],
                retained_historical_continuum_channels=len(provenance['retained_continuum_fit_stored_indices']) if provenance['direct_channel_mapping'] else None,
                historical_parent_fit_channels=len(provenance['continuum_fit_parent_indices_zero_based']) if provenance['direct_channel_mapping'] else None,
                polynomial_order=provenance.get('polynomial_order'),unresolved_reasons=';'.join(provenance['unresolved_reasons']),
                spatial_diagnostic_status=spatial['status'],inner_to_outer_mad_ratio_median=spatial.get('inner_to_outer_mad_ratio_median'))
            rows.append(summary);print(summary,flush=True)
        finally:cube.close()
    write_csv(output/'galaxies.csv',rows);write_csv(output/'native-candidate-channel-diagnostics.csv',measurements)
    result=dict(status='NATIVE_SPECTRAL_PROVENANCE_AND_CONDITIONAL_COVARIANCE_EXECUTED',admission_disposition='SOURCE_BLOCKED',
        config=config,bindings=bindings,source_bindings=sources,galaxies=len(rows),
        direct_history_mapping_galaxies=[r['galaxy'] for r in rows if r['direct_channel_mapping']],
        unresolved_history_mapping_galaxies=[r['galaxy'] for r in rows if not r['direct_channel_mapping']],
        retained_historical_continuum_candidate_channels=len(measurements),
        native_spatial_diagnostic_galaxies=[r['galaxy'] for r in rows if r['spatial_diagnostic_status']=='HISTORICAL_CONTINUUM_CHANNEL_SPATIAL_DIAGNOSTIC'],
        conditional_continuum_covariance_cases=2*sum(r['direct_channel_mapping'] for r in rows),
        independent_unit_tests=tests.testsRun,certified_line_free_channels=0,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print({k:result[k] for k in ['galaxies','direct_history_mapping_galaxies','unresolved_history_mapping_galaxies','retained_historical_continuum_candidate_channels']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_native_spectral_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
