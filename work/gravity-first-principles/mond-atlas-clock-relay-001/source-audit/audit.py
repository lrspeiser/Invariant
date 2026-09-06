"""Source-only inventory; no residual calculation, target statistics or fitting."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
paths = ['configs/sparc_rotation_curves_full_v1.json',
         'configs/sparc_surface_brightness_exploration_v1.json',
         'work/gravity-first-principles/map-response-metadata-001/SPARC_Lelli2016c.mrt',
         'work/gravity-first-principles/sparc-pattern-analysis-001/registration.json',
         'work/private/matched-concentration-001/Rotmod_LTG.zip']
files = {p: {'sha256': hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),
             'bytes': (ROOT/p).stat().st_size} for p in paths}
curves = json.loads((ROOT/paths[0]).read_text(encoding='utf-8'))
photo = json.loads((ROOT/paths[1]).read_text(encoding='utf-8'))
history = json.loads((ROOT/paths[3]).read_text(encoding='utf-8'))
names = set(history['names'])
assert names == {g['galaxy'] for g in photo['galaxies']}
curves_by_name = {g['name']: g for g in curves['galaxies']}
photo_by_name = {g['galaxy']: g for g in photo['galaxies']}
meta = {}
for line in (ROOT/paths[2]).read_text(encoding='utf-8').splitlines():
    f = line.split()
    if f and f[0] in names:
        assert len(f) == 19
        meta[f[0]] = dict(distance_mpc=float(f[2]), distance_error_mpc=float(f[3]),
            inclination_deg=float(f[5]), inclination_error_deg=float(f[6]),
            luminosity_1e9_lsun=float(f[7]), luminosity_error_1e9_lsun=float(f[8]),
            rdisk_kpc=float(f[11]), hi_mass_1e9_msun=float(f[13]), quality=int(f[17]))
rows = []
with zipfile.ZipFile(ROOT/paths[4]) as archive:
    lookup = {Path(n).name:n for n in archive.namelist()}
    for name in sorted(names):
        g = curves_by_name[name]
        raw = archive.read(lookup[g['provenance']['source_file']])
        assert hashlib.sha256(raw).hexdigest() == g['provenance']['source_file_sha256']
        tokens = [line.split() for line in raw.decode('utf-8').splitlines() if line.strip() and not line.startswith('#')]
        assert all(len(row)==8 for row in tokens)
        # Source columns only: do not compare response values or compute response eligibility.
        assert [[r[j] for j in [0,3,4,5]] for r in tokens] == [[r[j] for j in [0,3,4,5]] for r in g['rows']]
        assert [r[6:8] for r in tokens] == photo_by_name[name]['rows']
        source = np.array([[r[j] for j in [0,3,4,5,6,7]] for r in tokens],float)
        rad,gas,disk,bulge,sbd,sbb = source.T
        vbar2 = gas*np.abs(gas)+.5*disk**2+.7*bulge**2
        rows.append(dict(name=name, rows=len(tokens), source_member_sha256=hashlib.sha256(raw).hexdigest(),
            all_source_values_finite=bool(np.isfinite(source).all()),
            all_radii_positive=bool((rad>0).all()), radii_strictly_increasing=bool((np.diff(rad)>0).all()),
            nominal_baryonic_v2_positive=bool((vbar2>0).all()),
            surface_brightness_nonnegative=bool(((sbd>=0)&(sbb>=0)).all()),
            **meta[name]))
out = dict(scope='SOURCE_ONLY_INVENTORY_NOT_RESPONSE_SCORING', files=files,
    full_catalog=dict(galaxies=len(curves['galaxies']), rows=sum(len(g['rows']) for g in curves['galaxies'])),
    admitted_historical_development=dict(galaxies=len(rows),rows=sum(r['rows'] for r in rows)),
    protected_other_names=sorted(set(curves_by_name)-names),
    prior_exposure=history['scope'], columns=curves['columns'], primary_source=curves['source'],
    mass_to_light_convention=curves['mass_to_light_convention'],
    access_disclosure='A preliminary Get-Content header inspection encountered minified full JSON and displayed/truncated existing response rows. No scores, fits, residuals, response eligibility or response summary statistics were computed. This is not a fresh blind sample.',
    objects=rows)
(DEST/'inventory.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(galaxies=len(rows),rows=sum(r['rows'] for r in rows),
    failed_source_flags=[{k:v for k,v in r.items() if v is False or k=='name'} for r in rows if any(v is False for v in r.values())]),indent=2))
