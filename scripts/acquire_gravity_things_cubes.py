import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from astropy.io import fits
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/things-cube-acquisition-001';D.mkdir(exist_ok=False)
P=ROOT/'work/private/things-observable-12gal-003'
source=json.loads((ROOT/'work/gravity-first-principles/things-observable-acquisition-003/receipt.json').read_text())
assets=[a for a in source['files'] if a['resolution']=='NA' and a['moment']==0]
(D/'runner.py').write_bytes(Path(__file__).read_bytes())
def save(n,v):(D/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
save('registration.json',dict(purpose='Acquire standard unblanked natural-weighting HI cubes to audit channel noise and recover detection support.',
    source_paper='https://arxiv.org/html/0810.2125v1',names=sorted(a['name'] for a in assets),
    caveat='Standard cubes have uniform noise but uncorrected residual flux scaling. Do not substitute their flux directly for published rescaled moment maps.'))
def acquire(a):
    url=a['url'].replace('MOM0','CUBE');path=P/Path(url).name;tmp=path.with_suffix('.part')
    digest=hashlib.sha256()
    with urllib.request.urlopen(url,timeout=60) as r,tmp.open('wb') as f:
        while True:
            block=r.read(4*1024*1024)
            if not block:break
            f.write(block);digest.update(block)
    h=fits.getheader(tmp);tmp.rename(path)
    return dict(name=a['name'],url=url,file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
        sha256=digest.hexdigest(),header={k:h.get(k) for k in ['NAXIS1','NAXIS2','NAXIS3','NAXIS4','BUNIT','CTYPE3','CDELT3']})
files=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures={pool.submit(acquire,a):a for a in assets}
    for future in concurrent.futures.as_completed(futures):
        a=futures[future]
        try:files.append(future.result());print('Cube '+a['name'],flush=True)
        except Exception as e:errors.append(dict(name=a['name'],error=repr(e)))
        save('partial.json',dict(files=files,errors=errors))
save('receipt.json',dict(status='COMPLETE' if not errors else 'INCOMPLETE',files=files,errors=errors,
    bytes=sum(a['bytes'] for a in files)))
