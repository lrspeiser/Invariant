"""Execute a registered, source-only tracer pilot with immutable receipts."""
from __future__ import annotations
import argparse
import copy
import json
import re
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from mond_atlas_image_io import read_primary_image
from mond_atlas_registered_source import inclination, transfer_matrix, source_coordinates, rebin_tracer

REPORT = ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001'


def load_sources(config):
    sources = config['assets']; bound = {}
    for row in sources.values():
        if row['name'] != config['object_id']:
            raise ValueError('source identity differs')
        path = ROOT/row['file']
        if digest(path) != row['sha256']:
            raise ValueError('changed source bytes: '+str(path))
        bound[path.relative_to(ROOT).as_posix()] = row['sha256']
    stellar, sh = read_primary_image(ROOT/sources['STELLAR_MASS_MAP']['file'])
    mask, mh = read_primary_image(ROOT/sources['STELLAR_ICA_MASK']['file'])
    hi, hh = read_primary_image(ROOT/sources['HI_MOM0']['file'])
    co, ch = read_primary_image(ROOT/sources['CO21_MOM0']['file'])
    err, eh = read_primary_image(ROOT/sources['CO21_EMOM0']['file'])
    keys = ['CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
            'CD1_1','CD1_2','CD2_1','CD2_2','CDELT1','CDELT2']
    if stellar.shape != mask.shape or any(sh.get(k) != mh.get(k) for k in keys):
        raise ValueError('stellar mask coordinates differ')
    if co.shape != err.shape or any(ch.get(k) != eh.get(k) for k in keys):
        raise ValueError('CO error coordinates differ')
    if str(sh.get('BUNIT','')).lower() != 'mjy/sr' or hh.get('BUNIT') != 'JY/B*M/S' or ch.get('BUNIT') != 'K KM/S' or eh.get('BUNIT') != 'K KM/S':
        raise ValueError('unrecognized tracer unit')
    beam = next(re.search(r'BMAJ=\s*([\d.E+-]+)\s+BMIN=\s*([\d.E+-]+)', line)
                for line in sources['HI_MOM0']['beam_and_blanking_history'] if 'CLEAN BMAJ=' in line)
    major, minor = [float(t)*3600 for t in beam.groups()]
    frequency = float(hh['RESTFREQ'])/1e9
    if not 1.4 < frequency < 1.45 or min(major, minor) <= 0:
        raise ValueError('HI frequency/beam changed')
    factor = .001*1222000/(frequency**2*major*minor)*1.823e18/1.248e20*config['conversions']['helium_factor_hi']
    record = next(r for r in read_json(ROOT/config['p1_image_record'])['objects'] if r['name'] == config['object_id'])
    if record['selected_wcs'] != 'linear_tan':
        raise ValueError('P1 core TAN lacks prior calibration support')
    p1path = ROOT/record['image_file']
    if digest(p1path) != record['image_sha256']:
        raise ValueError('P1 bytes changed')
    bound[p1path.relative_to(ROOT).as_posix()] = record['image_sha256']
    _, h1 = read_primary_image(p1path)
    transfers = []
    for name in ('stellar_transfer', 'stellar_transfer_alternative'):
        t = read_json(ROOT/config[name])
        if t['galaxy'] != config['object_id'] or not t['both_relative_and_prior_absolute_pass']:
            raise ValueError('stellar absolute/relative transfer not supported')
        transfers.append(t)
        bound[config[name]] = digest(ROOT/config[name])
    inputs = [
        ('stellar_luminosity', stellar, sh, np.isfinite(stellar)&(mask == 0), config['conversions']['stellar_lsun_pc2_per_mjy_sr'], None),
        ('atomic_helium', hi, hh, np.isfinite(hi)&(hi != 0), factor, None),
        ('co21', co, ch, np.isfinite(co)&np.isfinite(err)&(err > 0), 1., err),
    ]
    meta = dict(hi_beam_arcsec=[major,minor], hi_frequency_ghz=frequency,
                hi_surface_factor=factor, units=dict(stellar_luminosity='Lsun/pc2',atomic_helium='Msun/pc2 including helium',co21='K km/s'),
                inherited_stellar_sip_present=any(k.endswith('_ORDER') for k in sh),
                stellar_sip_applied=False, native_beams_matched=False)
    return inputs, h1, transfers, bound, meta


def independent_source_wcs(inputs, p1, transfers, geometry):
    rows = []
    # Core wcslib calls explicitly omit SIP, matching the declared source model.
    w1 = WCS(p1).celestial; w1.sip = None
    for name, image, header, good, _, _ in inputs:
        iy, ix = np.nonzero(good); choose = np.linspace(0,len(ix)-1,min(300,len(ix))).astype(int)
        xy = np.column_stack((ix[choose],iy[choose]))
        w = WCS(header).celestial; w.sip = None
        for t in (transfers if name == 'stellar_luminosity' else [None]):
            transform = transfer_matrix(p1,t['fit']['shift']) if t else None
            v = source_coordinates(header,xy,geometry,transform)[3]
            reference = w.wcs_pix2world(xy,0)
            if t:
                reference = w1.wcs_pix2world(w1.wcs_world2pix(reference,0)+t['fit']['shift'],0)
            ra,dec = np.deg2rad(reference).T
            rv = np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
            error = float(np.max(np.linalg.norm(v-rv,axis=1))*206264.806247)
            rows.append(dict(component=name,translation=t['fit']['shift'] if t else None,
                             checked_pixels=len(xy),world_max_error_arcsec=error))
    return rows


def execute(config_path, output, private):
    config = read_json(config_path)
    if config['admission_disposition'] != 'SOURCE_BLOCKED':
        raise ValueError('only source-only blocked disposition implemented')
    if not output.is_relative_to(ROOT/'work/gravity-first-principles') or not private.is_relative_to(ROOT/'work/private'):
        raise ValueError('output path outside assigned public/private locations')
    if output.exists() or private.exists():
        raise FileExistsError('immutable output exists')
    for path, expected in read_json(REPORT/'freeze.json')['bindings'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('preflight changed: '+path)
    output.mkdir(parents=True); private.mkdir(parents=True)
    bindings = {p.relative_to(ROOT).as_posix():digest(p) for p in
                [config_path,Path(__file__),ROOT/'scripts/mond_atlas_registered_source.py',
                 ROOT/'scripts/build_mond_atlas_ngc2903_source.py',ROOT/'scripts/mond_atlas_image_io.py',
                 ROOT/'scripts/mond_atlas_common.py',ROOT/'tests/test_mond_atlas_registered_source.py',
                 ROOT/config['p1_image_record']]}
    write_json(output/'execution-start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
               bindings=bindings,source_arrays_opened=0,observed_response_arrays_opened=0))
    sys.path.insert(0,str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_mond_atlas_registered_source')
    with (output/'unit-tests.log').open('w',encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write_json(output/'numerical-gate.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
               passed=result.wasSuccessful() and not result.skipped,source_arrays_opened=0,observational_scoring_allowed=False))
    if not result.wasSuccessful() or result.skipped:
        raise ValueError('source numerical benchmark failed before construction')
    start = time.perf_counter()
    inputs,p1,transfers,source_bindings,meta = load_sources(config)
    checks = independent_source_wcs(inputs,p1,transfers,config['geometry'])
    write_json(output/'actual-source-wcs-check.json',checks)
    if any(r['world_max_error_arcsec'] > config['benchmarks']['astropy_world_error_arcsec_max'] for r in checks):
        raise ValueError('actual source WCS benchmark failed')
    cases = []; annuli = []; mass_rows = []; nominal = None
    with threadpool_limits(limits=1):
        for case in config['cases']:
            geometry = dict(config['geometry'])
            geometry.update({k:v for k,v in case.items() if k in geometry})
            grid = dict(config['source_grid']);distance_scale=geometry['distance_mpc']/config['geometry']['distance_mpc']
            for key in ('half_width_kpc','spacing_kpc','annulus_width_kpc','taper_start_kpc','cutoff_kpc'):
                grid[key] *= distance_scale
            transfer = transfers[int(case.get('reverse_transfer',False))]
            transform = transfer_matrix(p1,transfer['fit']['shift'])
            nsub = case.get('pixel_subdivisions',config['pixel_subdivisions'])
            packed = {}; reports = {}; comparison = {}
            for name,image,header,good,conversion,error in inputs:
                arrays,report,rings = rebin_tracer(image,header,good,geometry,grid,conversion,
                    transform if name == 'stellar_luminosity' else None,nsub,error)
                for key,value in arrays.items():
                    if value is not None:packed[name+'_'+key] = value
                reports[name] = report
                annuli.extend(dict(case=case['id'],component=name,**r) for r in rings)
                if nominal is not None:
                    ref = nominal[name+'_observed'];delta = arrays['observed']-ref
                    comparison[name] = dict(coordinate_matched_signed_relative_l1=float(np.sum(abs(delta))/max(np.sum(abs(ref)),1e-30)))
            if nominal is None:nominal = copy.deepcopy(packed)
            for conv in config['mass_conversion_cases']:
                for fill in ('zero','annular'):
                    masses = dict(stellar_msun=reports['stellar_luminosity']['conditional_'+fill+'_integral']*conv['stellar_ml'],
                        atomic_helium_msun=reports['atomic_helium']['conditional_'+fill+'_integral'],
                        molecular_helium_msun=reports['co21']['conditional_'+fill+'_integral']*conv['alpha_co10']/conv['r21'])
                    mass_rows.append(dict(case=case['id'],conversion=conv['id'],fill=fill,**masses,total_msun=sum(masses.values())))
            target = private/(case['id']+'.npz');np.savez_compressed(target,**packed)
            row = dict(id=case['id'],geometry=dict(geometry,inclination_deg=inclination(geometry)),grid=grid,
                       stellar_translation_p1_pixels=transfer['fit']['shift'],pixel_subdivisions=nsub,
                       components=reports,comparison_to_nominal_on_same_angular_index=comparison,
                       packet=target.relative_to(ROOT).as_posix(),packet_sha256=digest(target))
            write_json(output/('case-'+case['id']+'.json'),row);cases.append(row)
            print(case['id']+' complete',flush=True)
    write_csv(output/'source-annuli.csv',annuli);write_csv(output/'conditional-masses.csv',mass_rows)
    summary = dict(status='CONDITIONAL_TRACER_SOURCE_PILOT_EXECUTED',disposition='SOURCE_BLOCKED',object_id=config['object_id'],
                   elapsed_seconds=time.perf_counter()-start,source_metadata=meta,source_bindings=source_bindings,
                   cases=cases,mass_cases=mass_rows,source_3d_truth_observed=False,new_gravity_scores=0,
                   new_observed_motion_likelihoods=0,observed_response_arrays_opened=0,
                   source_processing_kinematically_independent=False,
                   limitations=config['source_policy'],prior_initial_control_failure='Uniform test box clipped inclined corners; original test retained. Enclosing box fixed without changing 1e-12 conservation tolerance or production implementation.')
    write_json(output/'summary.json',summary)
    write_json(output/'artifact-hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print('Completed source-only pilot: '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_ngc2976_source_v1.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args();execute(args.config.resolve(),args.output.resolve(),args.private.resolve())
