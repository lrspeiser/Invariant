"""Frozen recipe transfer; support checks precede all cube-value access."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_covariance import block_geometry,extract_background,regularized_covariance
import mond_atlas_noise_mode_power as mp
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-transfer-001'
PRIVATE=ROOT/'work/private/mond-atlas-noise-transfer-001'

def support_check(name):
    receipt=ROOT/f'work/gravity-first-principles/mond-atlas-native-spectral-001/{name}.json'
    r=read_json(receipt);cube=ROOT/r['source_path'];moment=cube.with_name(cube.name.replace('CUBE','MOM0'))
    h=fits.getheader(cube);mh=fits.getheader(moment)
    keys=['NAXIS1','NAXIS2','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2']
    assert all(h[k]==mh[k] for k in keys)
    assert h['CTYPE1']=='RA---SIN' and h['CTYPE2']=='DEC--SIN'
    assert not any(abs(h.get(k,0))>0 for k in ['CROTA1','CROTA2'])
    m=fits.getdata(moment).squeeze();assert m.shape==(h['NAXIS2'],h['NAXIS1'])
    px=abs(h['CDELT1'])*3600;py=abs(h['CDELT2'])*3600
    excluded=(m>0)|~np.isfinite(m);distance=distance_transform_edt(~excluded,sampling=(py,px))
    yy,xx=np.indices(m.shape);east=(xx+1-h['CRPIX1'])*h['CDELT1']*3600;north=(yy+1-h['CRPIX2'])*h['CDELT2']*3600
    radius=np.hypot(east,north);keep=(radius>=550)&(radius<=680)&(distance>120)&(xx>=42)&(yy>=42)&(xx<m.shape[1]-42)&(yy<m.shape[0]-42)
    supports={'training':keep&(east < -90),'validation':keep&(east>90)}
    contract=dict(block_core_side_native_pixels=24,block_lattice_step_native_pixels=48,block_lattice_center_origin_native_pixels=24,inner_training_folds=3)
    rows=block_geometry(supports,contract);counts={k:sum(r['region']==k for r in rows) for k in supports}
    return dict(name=name,channels=int(h['NAXIS3']),counts=counts,eligible=min(counts.values())>=12,cube_path=cube.relative_to(ROOT).as_posix(),cube_sha256_expected=r['source_sha256'],moment_path=moment.relative_to(ROOT).as_posix(),moment_sha256=digest(moment),receipt_sha256=digest(receipt),header={k:h[k] for k in keys},excluded_positive_or_nonfinite_pixels=int(excluded.sum())),rows

if __name__=='__main__':
    out=P/'support001';out.mkdir(parents=True,exist_ok=False);checks=[]
    for name in ['NGC3198','NGC2903']:
        record,rows=support_check(name);checks.append(record)
        write_csv(out/f'{name}-cores.csv',rows) if rows else None
        if record['eligible']:break
    write_json(out/'summary.json',dict(checks=checks,cube_values_read=False,preflight_sha256=digest(P/'PREFLIGHT.md'),script_sha256=digest(Path(__file__))))
    print(json.dumps(checks,indent=2))
