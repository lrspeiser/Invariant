"""Inspect already accessed development stellar files; no response data reads."""
import hashlib
import json
from pathlib import Path
from astropy.io import fits

root=Path(__file__).parent/'Invariant'
packet_path=root/'work/gravity-first-principles/xcop-pressure-002/source_preflight.json'
packet=json.loads(packet_path.read_bytes())
rows=[]
for p in packet['packets']:
    if p['stellar'] is None:
        continue
    access=next(a for a in p['access'] if a.get('role')=='stellar_mass')
    path=root/access['path']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==access['sha256']
    with fits.open(path) as hdus:
        metadata=[dict(hdu=i,name=h.name,columns=list(h.columns.names),
                       comments=list(h.header.get('COMMENT',[]))) for i,h in enumerate(hdus) if i>0]
        table=hdus[access['hdu']].data
        assert list(map(float,table['MSTAR']))==p['stellar']['mass_msun']
        assert list(map(float,table['RADIUS']))==p['stellar']['radius_kpc']
    rows.append(dict(cluster=p['cluster'],path=access['path'],sha256=digest,hdus=metadata,
                     retained_columns=list(p['stellar']),selected_hdu=access['hdu']))
out=dict(rows=rows,packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
    paper='https://arxiv.org/abs/2007.01084',section='4.1; arXiv PDF page 13',
    release='https://dominiqueeckert.wixsite.com/xcop/data',
    finding='Source paper describes projected cumulative profiles and a 0.75 correction at R500 only. Current pipeline treats retained masses as spherical enclosed mass. No file-specific deprojection evidence has been established. Lower/upper uncertainty columns exist but were omitted from the derived packet.',
    disposition='Existing stellar-containing spherical cluster scores are conditional on an unverified geometry mapping; withhold physical ranking or exclusion pending source reconstruction.',
    raw_pressure_or_reserved_files_accessed=False)
dest=root/'work/gravity-first-principles/stellar-projection-audit-001'
dest.mkdir(exist_ok=False)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(dest/'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(rows[0],indent=2))
print('Verified five existing source hashes and exact packet mass/radius identity')
