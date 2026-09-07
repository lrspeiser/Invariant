"""Source-only integrity and metadata audit: never materialize cube responses."""
from pathlib import Path
import json,hashlib,sys,csv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,default=lambda x:x.item())
def main():
    out=P/'run003';out.mkdir(exist_ok=False)
    files={};headers=[];expected={};source_arrays=[];models=[]
    for name in ['NGC2976','NGC3198']:
        r=read(ROOT/f'work/gravity-first-principles/mond-atlas-native-spectral-001/{name}.json');path=ROOT/r['source_path'];expected[path]=r['source_sha256']
        h=fits.getheader(path);w=WCS(h).spectral;v=w.pixel_to_world_values(np.arange(h['NAXIS3']));assert np.isfinite(v).all() and (np.all(np.diff(v)<0) or np.all(np.diff(v)>0))
        headers.append(dict(galaxy=name,path=path.relative_to(ROOT).as_posix(),shape=[h[f'NAXIS{i}'] for i in [3,2,1]],expected_shape=r['shape'],shape_matches=[h[f'NAXIS{i}'] for i in [3,2,1]]==r['shape'],bunit=h.get('BUNIT'),spectral_ctype=list(w.wcs.ctype),spectral_cunit=[str(x) for x in w.wcs.cunit],specsys=w.wcs.specsys,spectral_first_last=[float(v[0]),float(v[-1])],beam_deg=[h.get('BMAJ'),h.get('BMIN'),h.get('BPA')],pixel_deg=[h.get('CDELT1'),h.get('CDELT2')],cube_values_materialized=False))
    s=read(ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json')
    for path,h in s['source_bindings'].items():expected[ROOT/path]=h
    cfg=read(ROOT/'configs/mond_atlas_ngc3198_source_v1.json')
    for asset in cfg['assets'].values():
        if isinstance(asset,dict) and 'file' in asset and 'sha256' in asset:expected[ROOT/asset['file']]=asset['sha256']
    bound=read(ROOT/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json')
    for case in bound['source_cases']:
        for c in case['components']:expected[ROOT/c['path']]=c['sha256']
    for path,h in expected.items():
        present=path.exists();got=digest(path) if present else None
        files[path.relative_to(ROOT).as_posix()]=dict(present=present,bytes=path.stat().st_size if present else None,sha256=got,expected_sha256=h,matches=got==h)
        if present and path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as z:
                entry={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),finite_count=int(np.isfinite(z[k]).sum()) if np.issubdtype(z[k].dtype,np.number) else None,size=int(z[k].size)) for k in z.files}
                for k in ['source','surface','source_coefficients']:
                    if k in z:entry[k]['nonnegative']=bool(np.all(z[k]>=0))
            source_arrays.append(dict(path=path.relative_to(ROOT).as_posix(),arrays=entry))
    for name,rel in [('NGC2976','mond-atlas-noise-dc-regularization-001/run001/models-before-east.json'),('NGC3198','mond-atlas-noise-transfer-001/run001/model-before-east.json')]:
        path=ROOT/'work/gravity-first-principles'/rel;a=read(path);ms=a.get('models',a.get('base_models'));n=len(a['mean']);rows=[]
        for band,m in ms.items():
            C=np.asarray(m['C']);power=np.asarray(m['power'])
            if name=='NGC2976' and band=='DC':C=np.diag(a['dc_candidates'][a['selected']]['variance'])
            e=np.linalg.eigvalsh(C);rows.append(dict(band=band,covariance_shape=list(C.shape),channels=n,positive_definite=bool(e.min()>0),min_eigenvalue=float(e.min()),power_length=len(power),power_positive=bool((power>0).all())))
        models.append(dict(galaxy=name,path=path.relative_to(ROOT).as_posix(),sha256=digest(path),units='mean mJy/native restoring beam; covariance (mJy/native beam)^2',rows=rows,spatial_mode_total=sum(r['power_length'] for r in rows)))
    # Header-only source support. Native pixel lattice is unrelated to measured intensity.
    h=fits.getheader(ROOT/'work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS');w=WCS(h).celestial;g=s['cases'][0]['geometry'];origin=SkyCoord(g['ra_deg']*u.deg,g['dec_deg']*u.deg);pa=np.deg2rad(g['pa_deg']);ci=np.cos(np.deg2rad(g['inclination_deg']));D=g['distance_mpc']*1000
    margin=D*np.deg2rad(1.5/3600)/ci # conservative one-pixel deprojected margin between samples
    rows=[]
    for j,y in enumerate(range(12,h['NAXIS2'],24)):
        for i,x in enumerate(range(12,h['NAXIS1'],24)):
            if min(x-6,y-6)<42 or x+6>=h['NAXIS1']-42 or y+6>=h['NAXIS2']-42:continue
            xx,yy=np.meshgrid(x-6-.5+np.arange(13),y-6-.5+np.arange(13));sky=w.pixel_to_world(xx,yy);east,north=origin.spherical_offsets_to(sky.icrs);e=east.rad*D;n=north.rad*D
            radius=np.hypot(e*np.sin(pa)+n*np.cos(pa),(e*np.cos(pa)-n*np.sin(pa))/ci)
            if radius.min()>=.75+margin and radius.max()<=2.5-margin:rows.append(dict(i=i,j=j,x_center=x,y_center=y,x_start=x-6,x_stop=x+6,y_start=y-6,y_stop=y+6,subset='training' if (i+j)%2==0 else 'evaluation',min_radius=float(radius.min()),max_radius=float(radius.max())))
    with (out/'frozen-geometric-apertures.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0] if rows else ['none']);writer.writeheader();writer.writerows(rows)
    count={k:sum(r['subset']==k for r in rows) for k in ['training','evaluation']}
    result=dict(status='SOURCE_BLOCKED_SCORER_ADAPTERS_MISSING',files=files,cube_headers=headers,source_arrays=source_arrays,noise_models=models,apertures=dict(counts=count,minimum6_per_subset_pass=min(count.values())>=6,all_channels=42,source_intensity_used=False,historical_exposure='both subsets previously exposed; source conditioned developmental evaluation only',conservative_interpixel_margin_kpc=margin),all_expected_hashes_match=all(v['matches'] for v in files.values()),observed_source_region_cube_values_opened=False,new_observed_scores=0,reserved_responses_opened=False,new_raw_bytes=0)
    save(out/'integrity.json',result);save(out/'bindings.json',{p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),P/'PREFLIGHT.md']})
    print(json.dumps(dict(files=len(files),arrays=len(source_arrays),hashes=result['all_expected_hashes_match'],apertures=count,minimum6_per_subset_pass=result['apertures']['minimum6_per_subset_pass'])))
if __name__=='__main__':main()
