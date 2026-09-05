"""Recover existing stellar bounds and validate spherical forward projection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from astropy.io import fits

root=Path(__file__).parent/'Invariant'
old_path=root/'work/gravity-first-principles/xcop-pressure-002/source_preflight.json'
old=json.loads(old_path.read_bytes())
rows=[]
for p in old['packets']:
    if p['stellar'] is None:
        continue
    a=next(a for a in p['access'] if a.get('role')=='stellar_mass')
    path=root/a['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==a['sha256']
    with fits.open(path) as h:
        d=h[2].data
        arrays={k:np.array(d[k],float) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']}
        assert np.all(np.isfinite(list(arrays.values())))
        assert np.all(arrays['MSTAR_LO']<=arrays['MSTAR'])
        assert np.all(arrays['MSTAR']<=arrays['MSTAR_HI'])
        assert arrays['MSTAR'].tolist()==p['stellar']['mass_msun']
        units={k:h[2].columns[k].unit for k in arrays}
        raw_units={k:h[1].columns[k].unit for k in h[1].columns.names}
    rows.append(dict(cluster=p['cluster'],source=a,units=units,raw_extension_units=raw_units,
        columns={k:v.tolist() for k,v in arrays.items()},
        geometry='projected per associated paper; file-specific transformation not documented',
        uncertainty='MSTAR_LO and MSTAR_HI bracket all retained masses; do not treat as deviations. Confidence convention and joint covariance unresolved.'))

def projected_mass(radius, density, nodes):
    # Mass in a cylinder = full inner shells + polar caps of outer shells.
    # u=R/r regularizes the infinite outer interval; rationalize 1-sqrt(1-u²).
    q,w=leggauss(nodes)
    u=(q+1)/2
    inner=quad(lambda r:4*np.pi*r*r*density(r),0,radius,epsabs=1e-12,epsrel=1e-12)[0]
    outer=4*np.pi*radius**3*np.sum(w/2*density(radius/u)/(u*u*(1+np.sqrt(1-u*u))))
    return inner+outer

controls=[]
for scale in [.3,1.,7.]:
    for mass in [.2,5.]:
        density=lambda r: 3*mass/(4*np.pi*scale**3)*(1+(r/scale)**2)**-2.5
        for ratio in [.01,.1,1.,10.,100.]:
            radius=scale*ratio
            expected=mass*radius**2/(radius**2+scale**2)
            computed=[projected_mass(radius,density,n) for n in [128,256,512]]
            errors=[abs(x/expected-1) for x in computed]
            assert errors[-1]<1e-7
            controls.append(dict(scale=scale,mass=mass,radius=radius,exact_projected_mass=expected,
                                 numerical=computed,relative_errors=errors))
out=dict(source_packet_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
    stellar_packets=rows, projection_controls=controls,
    scope='Recovered existing source columns; Plummer analytic forward-projection controls only. No cluster source fit, covariance assumption, gravity scores or candidate admission.',
    all_projection_controls_pass=True)
dest=root/'work/gravity-first-principles/projected-stellar-packet-001'
dest.mkdir(exist_ok=False)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
print('Recovered',len(rows),'stellar profiles;',len(controls),'analytic projection controls passed')
print('Worst relative projection error:',max(c['relative_errors'][-1] for c in controls))
