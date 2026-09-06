"""Independent face-difference flux replay of saved refraction/Newton potentials."""
import json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_refraction_program import ROOT,P,MANIFEST,G,source,digest,save

def run():
    summary=json.loads((P/'run001/summary.json').read_text(encoding='utf-8'))
    cases={c['id']:c for c in json.loads(MANIFEST.read_text(encoding='utf-8'))['source_cases']}
    rows=[]
    for r in summary['runs']:
        path=ROOT/r['packet_path'];assert digest(path)==r['packet_sha256']
        with np.load(path,allow_pickle=False) as p:
            phi=p['potential'];axes=[p['axes_'+a] for a in 'xyz']
        spacing=[a[1]-a[0] for a in axes];rho,masses=source(cases[r['case']],axes)
        eps=np.ones_like(rho) if r['model']=='newton' else .2+.8*rho/(rho+1e7)
        interior=(slice(1,-1),)*3;divergence=np.zeros(tuple(n-2 for n in phi.shape));netflux=0.
        for axis,h in enumerate(spacing):
            a=[slice(None)]*3;b=a.copy();a[axis]=slice(0,-1);b[axis]=slice(1,None)
            flux=np.diff(phi,axis=axis)/h*2/(1/eps[tuple(a)]+1/eps[tuple(b)])
            crop=[slice(1,-1)]*3;crop[axis]=slice(None)
            divergence+=np.diff(flux,axis=axis)[tuple(crop)]/h
            lo=[slice(1,-1)]*3;hi=lo.copy();lo[axis]=0;hi[axis]=-1
            area=np.prod([spacing[j] for j in range(3) if j!=axis])
            netflux+=float((flux[tuple(hi)].sum()-flux[tuple(lo)].sum())*area)
        rhs=4*np.pi*G*rho[interior]
        residual=float(np.linalg.norm(divergence-rhs)/np.linalg.norm(rhs))
        expected=float(rhs.sum()*np.prod(spacing));balance=abs(netflux/expected-1)
        assert residual<1e-8 and balance<1e-8
        rows.append(dict(case=r['case'],grid=r['grid'],model=r['model'],independent_physical_relative_residual=residual,boundary_flux=netflux,enclosed_rhs_integral=expected,independent_boundary_flux_relative_error=balance))
    save(P/'run001/independent-flux-review.json',dict(status='PASS_DISCRETE_EQUATIONS_NOT_FIELD_CONVERGENCE',runs=rows,source_adapter_shared=True,operator_reimplemented_as_face_differences=True,observed_response_arrays_opened=0,bindings={p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),ROOT/'scripts/mond_atlas_refraction_program.py',ROOT/'scripts/mond_atlas_source_resolution.py',MANIFEST,P/'run001/summary.json']}))
    print('Independent face flux equations pass for',len(rows),'potentials; refinement failures remain.')

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
