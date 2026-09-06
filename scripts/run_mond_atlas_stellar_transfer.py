"""Execute relative P1/P5 source transfer without opening galaxy motion data."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import minimum_filter,map_coordinates
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_image_io import read_primary_image,plain_tan_pixel_to_world,plain_tan_world_to_pixel
from mond_atlas_stellar_transfer import fit_shift,sample,score,flux_fit


def acquire(name,private,limit):
    target=private/(name+'.nonstellar.fits')
    url=f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P5/{name}.nonstellar.fits'
    if target.exists():
        receipt=read_json(private/(name+'-download.json'))
        if digest(target)!=receipt['sha256']:raise ValueError('cached download changed')
        return target,receipt
    temporary=target.with_suffix('.fits.part'); n=0; h=hashlib.sha256()
    with urllib.request.urlopen(url,timeout=90) as response,temporary.open('wb') as stream:
        if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('download exceeds remaining cap')
        while block:=response.read(1024*1024):
            n+=len(block)
            if n>limit:raise ValueError('download exceeds remaining cap')
            stream.write(block);h.update(block)
        receipt=dict(url=url,path=target.relative_to(ROOT).as_posix(),bytes=n,sha256=h.hexdigest(),
            downloaded_at_utc=datetime.now(timezone.utc).isoformat(),final_url=response.url,
            content_type=response.headers.get('Content-Type'))
    if target.exists():raise FileExistsError(target)
    temporary.rename(target);write_json(private/(name+'-download.json'),receipt)
    return target,receipt


def source_check(name,config,p1record,assets,absolute,private,remaining):
    p1path=ROOT/p1record['image_file'];starpath=ROOT/assets['STELLAR_MASS_MAP']['file'];maskpath=ROOT/assets['STELLAR_ICA_MASK']['file']
    for path,expected in [(p1path,p1record['image_sha256']),(starpath,assets['STELLAR_MASS_MAP']['sha256']),(maskpath,assets['STELLAR_ICA_MASK']['sha256'])]:
        if digest(path)!=expected:raise ValueError('bound source changed: '+str(path))
    dustpath,download=acquire(name,private,remaining)
    p1,h1=read_primary_image(p1path);stars,h5=read_primary_image(starpath);mask,hm=read_primary_image(maskpath);dust,hd=read_primary_image(dustpath)
    if not stars.shape==mask.shape==dust.shape:raise ValueError('P5 component shapes differ')
    keys=('CRVAL1','CRVAL2','CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2','CTYPE1','CTYPE2')
    if any(h5.get(k)!=hd.get(k) or h5.get(k)!=hm.get(k) for k in keys):raise ValueError('P5 component coordinates differ')
    if any(str(h.get('BUNIT','')).lower()!='mjy/sr' for h in [h1,h5,hd]):raise ValueError('unexpected flux unit')
    if p1record['selected_wcs']!='linear_tan':raise ValueError('P1 plain TAN not supported by prior calibration')
    stride=config['sampling_stride_p5_pixels'];yy,xx=np.mgrid[0:stars.shape[0]:stride,0:stars.shape[1]:stride]
    xy5=np.column_stack([xx.ravel(),yy.ravel()]); world=plain_tan_pixel_to_world(xy5,h5)
    xy1=plain_tan_world_to_pixel(world[:,0],world[:,1],h1)
    # Independent WCS implementation, explicitly matching the declared plain-TAN choice.
    w1=WCS(h1).celestial;w1.sip=None;w5=WCS(h5).celestial;w5.sip=None
    check=w1.wcs_world2pix(w5.wcs_pix2world(xy5[::101],0),0)
    wcs_error=float(np.max(np.abs(check-xy1[::101])))
    if wcs_error>config['benchmarks']['wcs_astropy_pixel_absolute_tolerance']:raise ValueError('independent WCS mapping mismatch')
    margin=int(np.ceil(config['search_radius_p1_pixels']))+2
    support=minimum_filter(np.isfinite(p1).astype(np.uint8),size=2*margin+1,mode='constant',cval=0)
    supported=map_coordinates(support,xy1.T[::-1],order=0,mode='constant',cval=0)>0
    reconstruct=stars[yy,xx].ravel()+dust[yy,xx].ravel()
    block=config['partition_block_p5_pixels'];guard=config['block_edge_guard_pixels']
    valid=supported&(mask[yy,xx].ravel()==0)&np.isfinite(reconstruct)
    valid&=(xx.ravel()%block>=guard)&(xx.ravel()%block<block-guard)&(yy.ravel()%block>=guard)&(yy.ravel()%block<block-guard)
    calibration=(xx.ravel()//block+yy.ravel()//block)%2==config['calibration_parity']
    if sum(valid&calibration)<config['minimum_calibration_samples']:raise ValueError('insufficient calibration footprint')
    threshold=float(np.percentile(reconstruct[valid&calibration],config['bright_percentile_of_calibration_p5_reconstruction']))
    bright=valid&(reconstruct>threshold);cal=bright&calibration;test=bright&~calibration
    if sum(cal)<config['minimum_calibration_samples'] or sum(test)<config['minimum_validation_samples']:raise ValueError('insufficient bright samples')
    fit=fit_shift(p1,xy1[cal],reconstruct[cal],config['search_radius_p1_pixels'],config['integer_step_pixels'])
    zero=sample(p1,xy1[cal],[0,0]);za,zb=flux_fit(zero,reconstruct[cal])
    before=score(sample(p1,xy1[test],[0,0]),reconstruct[test],za,zb)
    after=score(sample(p1,xy1[test],fit['shift']),reconstruct[test],fit['scale'],fit['offset'])
    gates=config['gates'];interior=max(abs(v) for v in fit['shift'])<config['search_radius_p1_pixels']-gates['maximum_shift_boundary_margin_pixels']
    passed=bool(after['relative_rms']<gates['validation_relative_rms_max'] and after['correlation'] is not None and after['correlation']>gates['validation_correlation_min'] and interior and fit['scale']>gates['minimum_positive_flux_scale'])
    quadrant=[]
    for q in range(4):
        chosen=test&((xx.ravel()>=stars.shape[1]/2).astype(int)+2*(yy.ravel()>=stars.shape[0]/2).astype(int)==q)
        if sum(chosen)>=20:quadrant.append(dict(quadrant=q,**score(sample(p1,xy1[chosen],fit['shift']),reconstruct[chosen],fit['scale'],fit['offset'])))
        else:quadrant.append(dict(quadrant=q,samples=int(sum(chosen)),relative_rms=None,correlation=None))
    # Sample coordinates and held-out residuals are provenance diagnostics, not new mass arrays.
    packet=private/(name+'-samples.npz')
    np.savez_compressed(packet,xy5=xy5[bright],xy1=xy1[bright],reconstruction=reconstruct[bright],
        calibration=calibration[bright],p1_before=sample(p1,xy1[bright],[0,0]),p1_after=sample(p1,xy1[bright],fit['shift']))
    result=dict(galaxy=name,status='RELATIVE_TRANSFER_PASS' if passed else 'RELATIVE_TRANSFER_FAIL',
        relative_transfer_pass=passed,prior_absolute_footprint_pass=name in absolute,
        both_relative_and_prior_absolute_pass=bool(passed and name in absolute),
        calibration_samples=int(sum(cal)),validation_samples=int(sum(test)),bright_threshold_p5_mjy_sr=threshold,
        p1_shape=list(p1.shape),p5_shape=list(stars.shape),fit=fit,validation_before=before,validation_after=after,
        validation_quadrants=quadrant,independent_wcs_pixel_max_error=wcs_error,
        inherited_sip_present=dict(p1=any(k.startswith(('A_','B_','AP_','BP_')) for k in h1),p5=any(k.startswith(('A_','B_','AP_','BP_')) for k in h5)),
        source_bindings={p.relative_to(ROOT).as_posix():digest(p) for p in [p1path,starpath,maskpath,dustpath]},
        nonstellar_download=download,private_samples=packet.relative_to(ROOT).as_posix(),private_samples_sha256=digest(packet),
        new_gravity_scores=0,absolute_photometric_calibration=False,independent_observing_epoch=False)
    return result


def main(args):
    config=read_json(args.config);output=args.output.resolve();private=ROOT/config['private_directory']
    if output.exists():raise FileExistsError('immutable output directory')
    output.mkdir(parents=True);private.mkdir(parents=True,exist_ok=True)
    bindings=[args.config,Path(__file__),ROOT/'scripts/mond_atlas_stellar_transfer.py',ROOT/'tests/test_mond_atlas_stellar_transfer.py',
        ROOT/'scripts/mond_atlas_image_io.py']+[ROOT/config[k] for k in ['p1_inputs','p5_inputs','absolute_astrometry']]
    write_json(output/'prospective-bindings.json',dict(config=config,bindings={p.relative_to(ROOT).as_posix():digest(p) for p in bindings}))
    import unittest
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    with (output/'unit-tests.log').open('w',encoding='utf-8',newline='\n') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_mond_atlas_stellar_transfer'))
    if not tests.wasSuccessful():raise ValueError('benchmark failed before source alignment')
    p1={r['name']:r for r in read_json(ROOT/config['p1_inputs'])['objects']}
    allassets=read_json(ROOT/config['p5_inputs'])['files'];absolute=read_json(ROOT/config['absolute_astrometry'])['footprint_strict_pass']
    results=[];errors=[];downloaded=0
    for name in config['objects']:
        try:
            assets={r['role']:r for r in allassets if r['name']==name}
            with threadpool_limits(limits=1):result=source_check(name,config,p1[name],assets,absolute,private,config['download_limit_bytes']-downloaded)
            downloaded+=result['nonstellar_download']['bytes'];results.append(result);write_json(output/(name+'.json'),result)
            print(name,result['status'],result['fit']['shift'],result['validation_after'],flush=True)
        except Exception as exc:
            error=dict(galaxy=name,error_type=type(exc).__name__,error=str(exc));errors.append(error);write_json(output/(name+'-error.json'),error);print(error,flush=True)
    write_json(output/'summary.json',dict(status='SOURCE_TRANSFER_EXECUTED' if not errors else 'SOURCE_TRANSFER_EXECUTED_WITH_ERRORS',
        disposition='SOURCE_BLOCKED',galaxies=len(results),errors=errors,results=results,download_bytes=downloaded,unit_tests_passed=tests.testsRun,
        relative_pass=[r['galaxy'] for r in results if r['relative_transfer_pass']],
        joint_prior_absolute_and_relative_pass=[r['galaxy'] for r in results if r['both_relative_and_prior_absolute_pass']],
        current_full_field_admissions=0,new_motion_scores=0,limitations=config['nonclaims']))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_stellar_transfer_v1.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.config=a.config.resolve();main(a)
