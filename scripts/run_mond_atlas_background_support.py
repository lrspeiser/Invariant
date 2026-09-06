"""Audit the existing background annulus against released HI support."""
from __future__ import annotations
import argparse, copy, hashlib, io, unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest, fits_primary_header
from mond_atlas_image_io import read_primary_image
from mond_atlas_background_support import block_fraction, dilate_disk, support_overlap, spectral_diagnostics
from run_mond_atlas_noise import masks
from run_mond_atlas_noise_robustness import reversed_masks


def same_spatial_grid(moment, cube):
    keys = ['NAXIS1','NAXIS2','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','CROTA1','CROTA2']
    keys += [prefix+str(i)+'_'+str(j) for prefix in ['PC','CD'] for i in [1,2] for j in [1,2]]
    checked = {}
    for key in keys:
        a, b = moment.get(key), cube.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            equal = bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))
        else: equal = a == b
        if not equal: raise ValueError('cube/MOM0 spatial grid mismatch: '+key)
        checked[key] = a
    # Do not silently treat a spectral-to-spatial coupled WCS as a 2D map.
    for header in [moment, cube]:
        for prefix in ['PC','CD']:
            for i in [1,2]:
                for j in [3,4]:
                    if header.get(prefix+str(i)+'_'+str(j),0) != 0:
                        raise ValueError('spectral/spatial WCS coupling unsupported')
    if not np.isclose(abs(moment['CDELT1']), abs(moment['CDELT2']), rtol=1e-12, atol=1e-15):
        raise ValueError('coarse warning disk requires square native angular pixels')
    return checked


def run(config_path, output):
    config = read_json(config_path); parent = read_json(ROOT/config['parent_noise_protocol'])
    partition = read_json(ROOT/config['partition_protocol'])
    prior_path = ROOT/config['partition_results']; prior = read_json(prior_path)
    if config['admission_disposition'] != 'SOURCE_BLOCKED': raise ValueError('source/background diagnostic only')
    for path, expected in prior['bindings'].items():
        if digest(ROOT/path) != expected: raise ValueError('prior binding changed: '+path)
    if output.exists(): raise FileExistsError('immutable outputs')
    output.mkdir(parents=True)
    paths = [config_path, ROOT/config['parent_noise_protocol'], ROOT/config['partition_protocol'], prior_path,
        ROOT/config['native_moment_receipt'], ROOT/config['native_cube_receipt'], ROOT/config['cube_audit'],
        Path(__file__), ROOT/'scripts/mond_atlas_background_support.py', ROOT/'scripts/mond_atlas_image_io.py',
        ROOT/'scripts/mond_atlas_common.py', ROOT/'scripts/run_mond_atlas_noise.py',
        ROOT/'scripts/run_mond_atlas_noise_robustness.py', ROOT/'tests/test_mond_atlas_background_support.py']
    bindings = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json', dict(config=config, bindings=bindings,
        previous_exposure='All galaxies and old background partitions are development-exposed.', galaxy_motion_scoring=False))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_mond_atlas_background_support.py')
    log = io.StringIO(); tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(), encoding='utf-8', newline='\n')
    if not tests.wasSuccessful(): raise RuntimeError(log.getvalue())
    moments = {r['name']:r for r in read_json(ROOT/config['native_moment_receipt'])['files'] if r['resolution']=='NA' and r['moment']==0}
    cubes = {r['name']:r for r in read_json(ROOT/config['native_cube_receipt'])['files']}
    overlaps = []; spectral_rows = []; galaxies = []; sources = {}
    for audit in read_json(ROOT/config['cube_audit']):
        name = audit['name']; moment = moments[name]; raw_cube = cubes[name]
        old_path = prior_path.parent/(name+'.json'); old = read_json(old_path)
        packet_path = ROOT/parent['source_packets']/(name+'.npz')
        for path, expected in [(ROOT/moment['file'],moment['sha256']), (ROOT/raw_cube['file'],raw_cube['sha256']),
                               (packet_path,old['summary']['packet_sha256'])]:
            actual = digest(path)
            if actual != expected: raise ValueError('source hash changed: '+str(path))
            sources[str(path.relative_to(ROOT))] = actual
        sources[str(old_path.relative_to(ROOT))] = digest(old_path)
        if raw_cube['sha256'] != audit['cube_sha256']: raise ValueError('cube preprocessing receipt mismatch')
        image, header = read_primary_image(ROOT/moment['file'])
        grid = same_spatial_grid(header, fits_primary_header(ROOT/raw_cube['file']))
        support = np.isfinite(image) & (image > 0)
        fraction = block_fraction(support, config['block_factor'])
        conservation = float(abs(fraction.sum()*config['block_factor']**2-support.sum()))
        if conservation > config['independent_benchmarks']['block_support_conservation_absolute_error_max']:
            raise ArithmeticError('support area not conserved')
        pixel_arcsec = abs(header['CDELT1'])*3600*config['block_factor']
        dilation = int(np.ceil(4*audit['extra_smoothing_sigma_arcsec']/pixel_arcsec))+2
        expanded = dilate_disk(fraction > 0, dilation)
        with np.load(packet_path, allow_pickle=False) as packet:
            east, north = packet['east'], packet['north']
            if fraction.shape != east.shape or east.shape != tuple(audit['shape'][1:]):
                raise ValueError('native-to-coarse grid size mismatch')
            lo, hi = parent['sky_annulus_arcsec']; radius = np.hypot(east,north)
            annulus = (radius > lo) & (radius < hi)
            cube = np.where(annulus[None,:,:], packet['cube'], 0.)
        previous = {p['split']:p for p in old['partitions']}; details = []; spectra = []
        for seed in partition['split_seeds']:
            cfg = copy.deepcopy(parent); cfg['split_seed'] = seed
            for direction in partition['calibration_validation_directions']:
                train, test = masks(east,north,cfg) if direction=='forward' else reversed_masks(east,north,cfg)
                label = str(seed)+'-'+direction
                identity = hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                if identity != previous[label]['split_sha256']: raise ValueError('partition changed: '+label)
                for role, mask in [('calibration',train),('validation',test)]:
                    row = dict(galaxy=name, split=label, split_sha256=identity, role=role, **support_overlap(mask,fraction,expanded))
                    overlaps.append(row); details.append(row)
                if seed==config['fixed_spatial_seed'] and direction in config['spectral_directions']:
                    result = spectral_diagnostics(cube,train,test,config['spectral_partitions']['outer_band_fraction_per_end'],config['spectral_partitions']['lags'])
                    spectra.append(dict(split=label, split_sha256=identity, **result))
                    for role in ['calibration','validation']:
                        for band, values in result[role].items():
                            for lag in values['lags']:
                                spectral_rows.append(dict(galaxy=name,split=label,role=role,band=band,
                                    **{k:v for k,v in values.items() if k!='lags'},**lag))
        summary = dict(galaxy=name, partitions=len(previous), previous_all_splits_pass=old['summary']['all_declared_splits_pass'],
            coarse_pixel_arcsec=pixel_arcsec, extra_smoothing_sigma_arcsec=audit['extra_smoothing_sigma_arcsec'],
            warning_dilation_radius_cells=dilation, native_positive_pixels=int(support.sum()),
            native_negative_pixels=int(np.sum(np.isfinite(image)&(image<0))), native_nonfinite_pixels=int(np.sum(~np.isfinite(image))),
            support_conservation_absolute_error=conservation,
            annulus_direct_pixel_fraction=support_overlap(annulus,fraction,expanded)['direct_pixel_fraction'],
            annulus_expanded_pixel_fraction=support_overlap(annulus,fraction,expanded)['expanded_pixel_fraction'],
            minimum_direct_partition_fraction=min(r['direct_pixel_fraction'] for r in details),
            maximum_direct_partition_fraction=max(r['direct_pixel_fraction'] for r in details),
            minimum_expanded_partition_fraction=min(r['expanded_pixel_fraction'] for r in details),
            maximum_expanded_partition_fraction=max(r['expanded_pixel_fraction'] for r in details))
        galaxies.append(summary)
        write_json(output/(name+'.json'),dict(summary=summary,verified_spatial_grid=grid,partitions=details,spectral_diagnostics=spectra))
        print(name,dict(direct_range=[summary['minimum_direct_partition_fraction'],summary['maximum_direct_partition_fraction']],
            expanded_range=[summary['minimum_expanded_partition_fraction'],summary['maximum_expanded_partition_fraction']]),flush=True)
    write_csv(output/'galaxies.csv',galaxies); write_csv(output/'partitions.csv',overlaps); write_csv(output/'spectral-diagnostics.csv',spectral_rows)
    result = dict(status='RELEASED_HI_SUPPORT_BACKGROUND_AUDIT_EXECUTED', admission_disposition='SOURCE_BLOCKED',
        config=config, bindings=bindings, source_bindings=sources, galaxies=len(galaxies), partition_role_evaluations=len(overlaps),
        fixed_seed_spectral_partitions=len(galaxies)*len(config['spectral_directions']), independent_unit_tests=tests.testsRun,
        galaxies_with_direct_partition_overlap=[r['galaxy'] for r in galaxies if r['maximum_direct_partition_fraction']>0],
        galaxies_with_expanded_partition_overlap=[r['galaxy'] for r in galaxies if r['maximum_expanded_partition_fraction']>0],
        new_masks_selected=0, certified_line_free_channels=0, galaxy_motion_scores_computed=0,
        admitted_galaxy_cube_predictions=0, goal_complete=False)
    write_json(output/'summary.json',result); print({k:result[k] for k in ['galaxies','partition_role_evaluations','galaxies_with_direct_partition_overlap','galaxies_with_expanded_partition_overlap']},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_background_emission_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); run(args.config.resolve(),args.output.resolve())
