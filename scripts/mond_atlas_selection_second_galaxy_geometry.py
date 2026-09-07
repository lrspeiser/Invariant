"""Metadata and source-footprint-only padding gate, no cube arrays."""
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
from mond_atlas_native_selection import beam_from_history,beam_covariance,gaussian_kernel,FWHM_SIGMA
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-selection-second-galaxy-001'
h=fits.getheader(R/'work/private/things-observable-12gal-003/NGC_3198_NA_CUBE_THINGS.FITS');beam=beam_from_history(h)
native=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],h['CDELT2']*3600,h['CDELT1']*3600)
extra=np.eye(2)*(30/FWHM_SIGMA/abs(h['CDELT1']*3600))**2-native;kernel=gaussian_kernel(extra);radius=40+len(kernel)//2
moment=R/'work/private/things-observable-12gal-003/NGC_3198_NA_MOM0_THINGS.FITS';m=fits.getdata(moment).squeeze();distance=distance_transform_edt(~((m>0)|~np.isfinite(m)),sampling=(abs(h['CDELT2'])*3600,abs(h['CDELT1'])*3600))
rows=list(csv.DictReader((R/'work/gravity-first-principles/mond-atlas-noise-transfer-001/support001/NGC3198-cores.csv').open(newline='',encoding='utf-8')));results=[]
for row in rows:
    y,x=int(row['center_y']),int(row['center_x']);y0,y1,x0,x1=y-radius,y+radius+1,x-radius,x+radius+1;reasons=[]
    if min(y0,x0)<42 or y1>m.shape[0]-42 or x1>m.shape[1]-42:reasons.append('image_edge_guard')
    if min(y0,x0)<0 or y1>m.shape[0] or x1>m.shape[1]:reasons.append('outside_image')
    else:
        if distance[y0:y1,x0:x1].min()<=120:reasons.append('expanded_source_footprint')
        e=(np.arange(x0,x1)+1-h['CRPIX1'])*h['CDELT1']*3600
        if not np.all(e>90 if row['region']=='validation' else e < -90):reasons.append('east_west_guard')
    results.append(dict(core=row['block_id'],region=row['region'],center_y=y,center_x=x,eligible=not reasons,reasons=reasons))
counts={k:sum(r['eligible'] and r['region']==k for r in results) for k in ['training','validation']}
out=P/'geometry001';out.mkdir(exist_ok=False)
result=dict(status='FULL_PADDING_ELIGIBLE' if min(counts.values())>=3 else 'INSUFFICIENT_FULL_PADDING_SUPPORT',counts=counts,extra_kernel_radius=len(kernel)//2,required_radius=radius,patch_side=2*radius+1,beam=beam,rows=results,cube_values_read=False,files={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'PREFLIGHT.md',P/'PADDING_ADDENDUM.md',moment]})
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','files']}))
