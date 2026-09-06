"""Freeze/check NGC3198 bindings, then call the unchanged registered-source runner.

No new reconstruction equations. All actual-header tests precede packet construction.
Use preflight once, then run with a fresh run-NNN public/private pair. Never overwrite.
"""
from __future__ import annotations
import argparse
import gzip
import io
import json
import os
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
import numpy as np
from astropy.io import fits
from threadpoolctl import threadpool_limits, threadpool_info
from mond_atlas_common import ROOT, read_json, write_json, digest
from mond_atlas_registered_source import source_coordinates, transfer_matrix, inclination
from build_mond_atlas_registered_source import reference_wcs, execute, REPORT as LEGACY

PUBLIC = ROOT/'work/gravity-first-principles/mond-atlas-generic-source-002'
PRIVATE = ROOT/'work/private/mond-atlas-generic-source-002'
CONFIG = ROOT/'configs/mond_atlas_ngc3198_source_v1.json'
RECOVERY = ROOT/'work/gravity-first-principles/mond-atlas-baryon-recovery-001'


def utc():
    return datetime.now(timezone.utc).isoformat()


def bind(bindings, path, expected=None):
    path = Path(path).resolve()
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError('changed binding: '+str(path))
    bindings[path.relative_to(ROOT).as_posix()] = value


def verify_bindings(bindings):
    for path, value in bindings.items():
        if digest(ROOT/path) != value:
            raise ValueError('changed frozen binding: '+path)


def audit_headers(config, bindings):
    """Hash original assets; read headers only, not FITS data arrays."""
    headers = {}; rows = []
    p1 = next(r for r in read_json(ROOT/config['p1_image_record'])['objects'] if r['name']=='NGC3198')
    absolute = read_json(ROOT/config['absolute_astrometry'])
    if p1['selected_wcs'] != 'linear_tan' or not absolute['footprint_strict_pass']:
        raise ValueError('absolute core-WCS support absent')
    if 'NGC3198' not in read_json(ROOT/config['absolute_astrometry_summary'])['footprint_strict_pass']:
        raise ValueError('absolute aggregate admission absent')
    if absolute['image_sha256'] != p1['image_sha256'] or absolute['catalog_sha256'] != p1['catalog_sha256']:
        raise ValueError('absolute evidence binds different bytes')
    for key in ('image', 'catalog'):
        bind(bindings, ROOT/p1[key+'_file'], p1[key+'_sha256'])
    ph = dict(fits.getheader(ROOT/p1['image_file']))
    transfers = []
    for key in ('stellar_transfer','stellar_transfer_alternative'):
        t = read_json(ROOT/config[key]); transfers.append(t)
        if t['galaxy'] != 'NGC3198' or not all(t[k] for k in ('relative_transfer_pass','prior_absolute_footprint_pass','both_relative_and_prior_absolute_pass')):
            raise ValueError('transfer not admitted')
        if t['validation_after']['relative_rms'] > .05 or t['validation_after']['correlation'] < .99:
            raise ValueError('relative-transfer frozen gate failed')
        for path, value in t['source_bindings'].items():
            bind(bindings, ROOT/path, value)
        bind(bindings, ROOT/t['private_samples'], t['private_samples_sha256'])
        if t['source_bindings'].get(Path(p1['image_file']).as_posix()) != p1['image_sha256']:
            raise ValueError('transfer P1 mismatch')
        for role in ('STELLAR_MASS_MAP','STELLAR_ICA_MASK'):
            a=config['assets'][role]
            if t['source_bindings'].get(Path(a['file']).as_posix()) != a['sha256']:
                raise ValueError('transfer P5 mismatch')
    expected_units={'STELLAR_MASS_MAP':'MJy/sr','STELLAR_ICA_MASK':'MJy/sr','HI_MOM0':'JY/B*M/S','CO21_MOM0':'K KM/S','CO21_EMOM0':'K KM/S'}
    for role, a in config['assets'].items():
        if a['name'] != 'NGC3198': raise ValueError('asset identity mismatch')
        bind(bindings, ROOT/a['file'], a['sha256'])
        if (ROOT/a['file']).stat().st_size != a['bytes']: raise ValueError('asset size mismatch')
        h = dict(fits.getheader(ROOT/a['file'])); headers[role]=h
        if h.get('BUNIT') != expected_units[role]: raise ValueError('unit mismatch')
        if h.get('OBJECT','NGC3198').strip().replace(' ','') != 'NGC3198': raise ValueError('header identity mismatch')
        history = [str(x) for x in h.get('HISTORY',[])]
        if role=='HI_MOM0' and not all(x in history for x in a['beam_and_blanking_history']):
            raise ValueError('HI beam/blanking history differs from original header')
        n,m=int(h['NAXIS1']),int(h['NAXIS2'])
        x,y,_,_=source_coordinates(h,np.array([[0,0],[n-1,0],[0,m-1],[n-1,m-1]]),config['geometry'])
        keep=('OBJECT','NAXIS','NAXIS1','NAXIS2','CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2','CDELT1','CDELT2','BUNIT','BMAJ','BMIN','BPA','RESTFREQ')
        rows.append(dict(role=role,url=a['url'],file=Path(a['file']).as_posix(),sha256=a['sha256'],bytes=a['bytes'],header={k:h[k] for k in keep if k in h},inherited_sip=any(k.endswith('_ORDER') for k in h),header_rectangle_corners_kpc=np.column_stack((x,y)).tolist(),rectangle_is_valid_coverage=False,hi_history=history if role=='HI_MOM0' else None,semantic_unit='integer ICA labels; inherited BUNIT ignored' if role=='STELLAR_ICA_MASK' else expected_units[role]))
    for a,b in [('STELLAR_MASS_MAP','STELLAR_ICA_MASK'),('CO21_MOM0','CO21_EMOM0')]:
        ha,hb=headers[a],headers[b]
        if (ha['NAXIS1'],ha['NAXIS2']) != (hb['NAXIS1'],hb['NAXIS2']) or reference_wcs(ha).to_header()!=reference_wcs(hb).to_header():
            raise ValueError('paired image spatial coordinates differ')
    return headers,ph,transfers,dict(assets=rows,p1=dict(url=p1['image_url'],file=p1['image_file'],selected_wcs=p1['selected_wcs'],original_status=p1['status']),absolute_footprint=dict(finite_validation=absolute['finite_validation'],median_arcsec=absolute['footprint_validation_median_arcsec'],p90_arcsec=absolute['footprint_validation_p90_arcsec'],strict_pass=True),relative_transfers=[dict(path=config[k],shift=t['fit']['shift'],validation=t['validation_after'],scale_and_background_applied=False) for k,t in zip(('stellar_transfer','stellar_transfer_alternative'),transfers)],source_arrays_opened=0,observed_response_arrays_opened=0)


def geometry_checks(config, headers, ph, transfers):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_mond_atlas_registered_source import independent_mapping
    rows=[]; gates=config['benchmarks'];p1=reference_wcs(ph)
    h1=dict(p1.to_header());h1.update(NAXIS=2,NAXIS1=int(ph['NAXIS1']),NAXIS2=int(ph['NAXIS2']))
    for case in config['cases']:
        if case['id'] in ('pixels_one','pixels_two'):continue
        g=dict(config['geometry']);g.update({k:v for k,v in case.items() if k in g});g['inclination_deg']=inclination(g)
        for role in ('STELLAR_MASS_MAP','HI_MOM0','CO21_MOM0'):
            h=headers[role];n,m=int(h['NAXIS1']),int(h['NAXIS2'])
            xx,yy=np.meshgrid(np.linspace(1,n-2,11),np.linspace(1,m-2,11));xy=np.column_stack((xx.ravel(),yy.ravel()))
            w=reference_wcs(h);hh=dict(w.to_header());hh.update(NAXIS=2,NAXIS1=n,NAXIS2=m)
            t=transfers[int(case.get('reverse_transfer',False))] if role=='STELLAR_MASS_MAP' else None
            tr=transfer_matrix(ph,t['fit']['shift']) if t else None
            actual=source_coordinates(h,xy,g,tr);shift=t['fit']['shift'] if t else None
            reference=independent_mapping(hh,xy,g,h1 if t else None,shift)
            world_error=float(np.max(np.linalg.norm(actual[3]-reference[2],axis=1))*206264.806247)
            delta=[]
            for direction in ([1.,0],[0,1.]):
                a=independent_mapping(hh,xy+direction,g,h1 if t else None,shift);b=independent_mapping(hh,xy-direction,g,h1 if t else None,shift)
                delta.append(np.column_stack(((a[0]-b[0])/2,(a[1]-b[1])/2)))
            area=np.abs(delta[0][:,0]*delta[1][:,1]-delta[0][:,1]*delta[1][:,0])*np.cos(np.deg2rad(g['inclination_deg']))
            area_error=float(np.max(abs(actual[2]/area-1)));pixel_error=None
            if t:
                v=actual[3];sky=np.column_stack((np.rad2deg(np.arctan2(v[:,1],v[:,0]))%360,np.rad2deg(np.arcsin(v[:,2]))))
                pixel_error=float(np.max(abs(p1.wcs_world2pix(sky,0)-(p1.wcs_world2pix(w.wcs_pix2world(xy,0),0)+shift))))
            passed=world_error<gates['astropy_world_error_arcsec_max'] and area_error<gates['projected_area_finite_difference_relative_max'] and (pixel_error is None or pixel_error<gates['astropy_composed_pixel_error_max'])
            rows.append(dict(case=case['id'],role=role,samples=len(xy),world_error_arcsec_max=world_error,finite_difference_area_relative_max=area_error,composed_pixel_error_max=pixel_error,passed=passed))
    return rows


def preflight():
    if PUBLIC.exists() or PRIVATE.exists():raise FileExistsError('immutable pilot root already exists')
    PUBLIC.mkdir(parents=True)
    try:
        c=read_json(CONFIG);base=read_json(ROOT/'configs/mond_atlas_ngc2976_source_v1.json')
        if c['object_id']!='NGC3198' or c['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('scope changed')
        for key in ('benchmarks','mass_conversion_cases','conversions','pixel_subdivisions'):
            if c[key]!=base[key]:raise ValueError('sensitivity family changed: '+key)
        if [x['id'] for x in c['cases']]!=[x['id'] for x in base['cases']]:raise ValueError('case family changed')
        sys.path.insert(0,str(ROOT/'tests'))
        with (PUBLIC/'preconstruction-unit-tests.log').open('x',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_mond_atlas_registered_source'))
        gate=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful() and result.testsRun==9 and not result.skipped,completed_utc=utc(),source_arrays_opened=0,observed_response_arrays_opened=0)
        write_json(PUBLIC/'preconstruction-numerical-gate.json',gate)
        if not gate['passed']:raise ValueError('nine independent source tests failed')
        bindings={}
        files=[CONFIG,Path(__file__),ROOT/'configs/mond_atlas_ngc2976_source_v1.json',ROOT/'configs/mond_atlas_stellar_transfer_v1.json',ROOT/'configs/mond_atlas_astrometry_v1.json',ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md',RECOVERY/'geometry.json',RECOVERY/'source-manifest.json',RECOVERY/'conversion-metadata.json',ROOT/'work/gravity-first-principles/stellar-co-acquisition-001/receipt.json',ROOT/'work/gravity-first-principles/things-observable-acquisition-003/receipt.json',LEGACY/'freeze.json']
        files += [ROOT/c[k] for k in ('p1_image_record','stellar_transfer','stellar_transfer_alternative','absolute_astrometry','absolute_astrometry_summary')]
        files += [ROOT/'scripts'/n for n in ('build_mond_atlas_registered_source.py','mond_atlas_registered_source.py','build_mond_atlas_ngc2903_source.py','mond_atlas_image_io.py','mond_atlas_common.py','mond_atlas_baryon_recovery.py')]
        files += [ROOT/'tests/test_mond_atlas_registered_source.py']
        for p in files:bind(bindings,p)
        verify_bindings(read_json(LEGACY/'freeze.json')['bindings'])
        # Exact original CDS rows are independently reselected from the recovered bytes.
        from mond_atlas_baryon_recovery import CATALOG_COLUMNS,PIPELINE_COLUMNS,parse_row
        for s in read_json(RECOVERY/'source-manifest.json')['sources']:
            bind(bindings,ROOT/s['path'],s['sha256'])
            if s['id'] in ('s4g_catalog','s4g_pipeline4'):
                raw=(ROOT/s['path']).read_bytes();raw=gzip.decompress(raw) if s['id']=='s4g_catalog' else raw
                row=next(x for x in raw.decode().splitlines() if x.split() and x.split()[0]=='NGC3198')
                parsed=parse_row(row,CATALOG_COLUMNS if s['id']=='s4g_catalog' else PIPELINE_COLUMNS,pipeline=s['id']=='s4g_pipeline4',verified_release=True)
                if parsed!=c['geometry_record']['catalog' if s['id']=='s4g_catalog' else 'pipeline4']:raise ValueError('original geometry row mismatch')
        headers,ph,transfers,audit=audit_headers(c,bindings)
        write_json(PUBLIC/'input-audit.json',audit)
        checks=geometry_checks(c,headers,ph,transfers)
        write_json(PUBLIC/'preconstruction-actual-header-checks.json',dict(checks=checks,source_arrays_opened=0,passed=all(r['passed'] for r in checks),completed_utc=utc()))
        if not all(r['passed'] for r in checks):raise ValueError('independent actual-header geometry failed')
        note='''# NGC3198 source-only preflight

SOURCE_BLOCKED. This pilot reuses unchanged registered-source equations. No source
packet has been constructed at freeze; no motion, velocity cube, lensing response,
field operator or gravity score is allowed. Numerical CPU thread limit is one.

The largest photometric radius is the P4 375 arcsec outer isophote (25.4291 kpc
at 13.987 Mpc). Round up to a 28 kpc cutoff; linear taper spans 24–28 kpc.
Grid axis centers span +/-32 kpc at 125 pc spacing (513 by 513); cell edges
extend by half a cell. Annuli are 250 pc. These are conditional computational
choices, not new measurements. Header rectangles extend beyond this field,
especially after inclination stretching, and include invalid image areas.
Finite-field loss, taper loss, signed negatives and absent coverage stay explicit.

The twelve geometry/registration/quadrature cases, six conversion/fill alternatives
per case and all thresholds are frozen in the config before reconstruction.
Distance cases scale all physical lengths together to hold angular support fixed.
Both fitted P1 translation receipts are bound; their flux scales/offsets are unused.
P1 core TAN omits inherited SIP under the existing finite-footprint Gaia evidence.
The earlier all-catalog Gaia failure remains historical, not overwritten.

Nine independent analytic/synthetic tests and 30 actual-header checks pass before
packet construction. Actual checks cover 121 pixels per component and every
distinct geometry/registration case using wcslib and finite differences. The runner
repeats nine tests and checks actual supported image pixels before rebinning.
Any failure is preserved; no threshold or aperture may be retuned from responses.
Cell/annular coverage thresholds .5/.2 and the 3% annular refinement flag are
unchanged from generic-source-001. A flag is a numerical limitation, not a pass.

Native beams are retained without matching/deconvolution. P5 flux is not mass;
M/L, CO conversion and excitation are assumed. HI blanked zeros are unobserved;
CO remains signed until conditional nonnegative fill. EMOM0 is a diagnostic,
not propagated covariance. No unique three-dimensional mass is established.
'''
        (PUBLIC/'PREFLIGHT.md').write_text(note,encoding='utf-8')
        for p in PUBLIC.iterdir():
            if p.is_file():bind(bindings,p)
        write_json(PUBLIC/'freeze.json',dict(created_utc=utc(),disposition='SOURCE_BLOCKED',source_packet_construction_started=False,new_operator_implementation=False,source_arrays_opened=0,observed_response_arrays_opened=0,bindings=bindings,legacy_freeze_also_verified=True,hardware=dict(numerical_threads=1,gpu_used=False,libraries=threadpool_info())))
        print('NGC3198 preflight frozen: 9 tests and '+str(len(checks))+' actual-header checks pass',flush=True)
    except Exception:
        write_json(PUBLIC/'preflight-failure.json',dict(utc=utc(),traceback=traceback.format_exc(),source_arrays_opened=0))
        raise


def run(run_id):
    if not run_id.startswith('run-') or len(run_id)!=7 or not run_id[4:].isdigit():raise ValueError('run-NNN required')
    freeze=read_json(PUBLIC/'freeze.json');verify_bindings(freeze['bindings'])
    output=PUBLIC/run_id;private=PRIVATE/run_id
    if output.exists() or private.exists():raise FileExistsError('immutable output already exists')
    receipt=PUBLIC/(run_id+'-checked-start.json')
    if receipt.exists():raise FileExistsError('run attempt already recorded')
    write_json(receipt,dict(started_utc=utc(),freeze_sha256=digest(PUBLIC/'freeze.json'),config_sha256=digest(CONFIG),verified_bindings=len(freeze['bindings']),output=output.relative_to(ROOT).as_posix(),private=private.relative_to(ROOT).as_posix(),source_arrays_opened=0,observed_response_arrays_opened=0))
    try:
        execute(CONFIG.resolve(),output.resolve(),private.resolve())
    except Exception:
        write_json(PUBLIC/(run_id+'-failure.json'),dict(utc=utc(),traceback=traceback.format_exc(),disposition='SOURCE_BLOCKED',paths_may_not_be_reused=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['preflight','run','verify-freeze']);parser.add_argument('--run-id',default='run-001');args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.mode=='preflight':preflight()
        elif args.mode=='run':run(args.run_id)
        else:verify_bindings(read_json(PUBLIC/'freeze.json')['bindings']);print('Frozen bindings pass')
