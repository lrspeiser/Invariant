import concurrent.futures,json,hashlib,urllib.request,shutil
from pathlib import Path
from astropy.io import fits
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/stellar-co-acquisition-001';D.mkdir(exist_ok=False)
P=ROOT/'work/private/stellar-co-12gal-001';P.mkdir(exist_ok=False)
(D/'runner.py').write_bytes(Path(__file__).read_bytes())
pre=json.loads((ROOT/'configs/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1.json').read_text())
assets=[]
for o in pre['object_source_contracts']:
 for a in o['stellar_branch']['files']+o['molecular_files']:
  assets.append(dict(name=o['object_id'],**a))
(D/'registration.json').write_text(json.dumps({'assets':assets,'scope':'Source maps and coverage; no missing CO converted into measured zero density.'},indent=2))
def one(a):
 p=P/(a['name']+'__'+a['role']+('.fits.gz' if a['url'].endswith('.gz') else '.fits'));tmp=p.with_suffix('.part')
 with urllib.request.urlopen(a['url'],timeout=60) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f,1024*1024)
 tmp.rename(p);h=fits.getheader(p)
 return dict(**a,file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_bytes=p.stat().st_size,bunit=h.get('BUNIT'))
files=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 fs={pool.submit(one,a):a for a in assets}
 for f in concurrent.futures.as_completed(fs):
  try:files.append(f.result())
  except Exception as e:errors.append({'asset':fs[f],'error':repr(e)})
  (D/'partial.json').write_text(json.dumps({'files':files,'errors':errors},indent=2));print(len(files),len(errors),flush=True)
(D/'receipt.json').write_text(json.dumps({'status':'COMPLETE' if not errors else 'INCOMPLETE','files':files,'errors':errors},indent=2))
