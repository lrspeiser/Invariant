"""Acquire official THINGS products for the previously source-selected 12 objects."""
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
from astropy.io import fits
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/things-observable-acquisition-002'
PRIVATE=ROOT/'work/private/things-observable-12gal-002'
D.mkdir(exist_ok=False);PRIVATE.mkdir(exist_ok=False)
(D/'runner.py').write_bytes(Path(__file__).read_bytes())
def save(name,d): (D/name).write_text(json.dumps(d,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index=ROOT.parent/'things-data-index.html';paper=ROOT.parent/'things-paper.html'
shutil.copyfile(index,D/'publisher-index.html');shutil.copyfile(paper,D/'measurement-paper.html')
soup=BeautifulSoup(index.read_bytes(),'html.parser')
links={urljoin('https://things.www3.mpia.de/Data.html',a['href']) for a in soup.find_all(href=True)}
prepath=ROOT/'configs/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1.json'
pre=json.loads(prepath.read_text());assets=[]
for obj in pre['object_source_contracts']:
    for spec in obj['hi_files']:
        for moment in (0,1,2):
            url=spec['url'].replace('MOM0',f'MOM{moment}')
            if url not in links:
                matches=[u for u in links if u.replace('_','')==url.replace('_','')]
                assert len(matches)==1, (url,matches)
                url=matches[0]
            assets.append(dict(name=obj['object_id'],resolution='NA' if '_NA_' in url else 'RO',
                moment=moment,url=url,file=PRIVATE/Path(url).name))
save('registration.json',dict(selection='All 12 objects in pre-existing source preflight, without new response-based pruning.',
    preflight_sha256=sha(prepath),index_sha256=sha(index),paper_sha256=sha(paper),
    source_url='https://things.www3.mpia.de/Data.html',paper_url='https://arxiv.org/html/0810.2125v1',
    assets=[{**a,'file':str(a['file'].relative_to(ROOT))} for a in assets],
    scope='Development data acquisition. Past project exposure prevents a pristine confirmation claim.',
    noise_rule='MOM2 is velocity dispersion, never uncertainty on MOM1. Published channel noise alone does not give moment-map covariance.'))
# Parse the mapping/noise table, preserving both NA and RO entries.
tables=[];psoup=BeautifulSoup(paper.read_bytes(),'html.parser')
for table in psoup.find_all('table'):
    if 'Weighting' in table.get_text() and 'noise' in table.get_text():
        tables.append([[c.get_text(' ',strip=True) for c in row.find_all(['td','th'])] for row in table.find_all('tr')])
save('published_noise_tables.json',dict(tables=tables,warning='Published survey metadata, not per-pixel propagated uncertainties. Check release/header beam differences.'))

def acquire(asset):
    p=asset['file'];tmp=p.with_suffix('.part')
    with urllib.request.urlopen(asset['url'],timeout=60) as response,tmp.open('wb') as f:
        shutil.copyfileobj(response,f,length=1024*1024)
    tmp.rename(p)
    h=fits.getheader(p);a=np.squeeze(fits.getdata(p))
    history='\n'.join(str(x) for x in h.get('HISTORY',[]))
    return {**{k:v for k,v in asset.items() if k!='file'},'file':str(p.relative_to(ROOT)),
        'bytes':p.stat().st_size,'sha256':sha(p),'shape':list(a.shape),'bunit':h.get('BUNIT'),
        'finite_fraction':float(np.isfinite(a).mean()),'zero_fraction':float((a==0).mean()),
        'beam_and_blanking_history':[x for x in history.splitlines() if any(t in x for t in ['CLEAN BMAJ','NBLANK','PIXVAL'])]}
receipts=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(acquire,a):a for a in assets}
    for future in concurrent.futures.as_completed(futures):
        a=futures[future]
        try:
            receipts.append(future.result());print(f'Acquired {a["name"]} {a["resolution"]} MOM{a["moment"]}',flush=True)
        except Exception as e:errors.append(dict(url=a['url'],error=repr(e)))
        save('receipt_partial.json',dict(files=receipts,errors=errors))
save('receipt.json',dict(status='COMPLETE' if not errors else 'INCOMPLETE',files=receipts,errors=errors,
    network_bytes=sum(a['bytes'] for a in receipts),calibrated_pixel_noise=False,raw_cube_acquired=False))
print(json.dumps(dict(files=len(receipts),errors=errors)),flush=True)
