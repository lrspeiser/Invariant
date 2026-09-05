"""Source-only positive spherical-shell bracket feasibility; no gravity fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

root=Path(__file__).parent/'Invariant'
path=root/'work/gravity-first-principles/projected-stellar-packet-001/result.json'
packet=json.loads(path.read_bytes())
dest=root/'work/gravity-first-principles/stellar-shell-aperture-001'
dest.mkdir(exist_ok=False)
registration=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    grids=[dict(nodes=128,outer_factor=10)],
    inner_scale_factor=.1, acceptance_maximum_fractional_bound_violation=1e-8,
    objective='Minimize common additive bracket relaxation as fraction of each measured mass; nonnegative thin spherical shell masses. No pressure or gravity data.',
    limits='Conditional aperture localization only. Omitted inner observations remain unresolved and no complete source is admitted. Positive shells permit nonmonotone density. Each grid includes all measured radii and geometric midpoints plus exterior samples. Singular shells are feasibility probes, not differentiable sources admissible for the gravity action. No likelihood or statistical confidence claim.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
old=json.loads((root/'work/gravity-first-principles/xcop-pressure-002/source_preflight.json').read_bytes())
r500={p['cluster']:p['r500_kpc'] for p in old['packets']}
expanded=[]
for source in packet['stellar_packets']:
    for cutoff in [0.,.02,.1,.5]:
        mask=np.array(source['columns']['RADIUS'])>=cutoff*r500[source['cluster']]
        expanded.append({**source,'cutoff_r500':cutoff,'omitted_count':int(sum(~mask)),
                        'columns':{k:np.array(v)[mask].tolist() for k,v in source['columns'].items()}})
for p in expanded:
    c=p['columns']
    r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
    for grid in registration['grids']:
        scales=np.unique(np.r_[r,np.sqrt(r[:-1]*r[1:]),np.geomspace(.1*r[0],grid['outer_factor']*r[-1],grid['nodes'])])
        # Exact projected polar-cap fraction of a thin spherical shell.
        normalization=m[-1]
        u=np.minimum(r[:,None]/scales[None,:],1.)
        design=(u*u/(1+np.sqrt(1-u*u)))*normalization/m[:,None]
        constraints=np.r_[np.c_[design,-np.ones(len(r))],np.c_[-design,-np.ones(len(r))]]
        bounds=np.r_[hi/m,-lo/m]
        objective=np.r_[np.zeros(len(scales)),1.]
        result=linprog(objective,A_ub=constraints,b_ub=bounds,bounds=(0,None),method='highs',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        row=dict(cluster=p['cluster'],cutoff_r500=p['cutoff_r500'],omitted_count=p['omitted_count'],grid=grid,solver_status=int(result.status),solver_message=result.message)
        if result.success:
            masses=result.x[:-1]*normalization
            projected=design@result.x[:-1]*m
            violation=float(max(0,np.max((lo-projected)/m),np.max((projected-hi)/m)))
            assert abs(violation-result.x[-1])<2e-8
            assert np.all(masses>=0)
            row.update(scales_kpc=scales.tolist(),component_mass_msun=masses.tolist(),
                projected_mass_msun=projected.tolist(),minimum_relaxation_fraction=float(result.x[-1]),
                verified_maximum_bound_violation_fraction=violation,
                within_original_brackets=violation<=1e-8,
                positive_components=int(sum(masses>0)))
        rows.append(row)
        print(p['cluster'],p['cutoff_r500'],grid,'relaxation',row.get('minimum_relaxation_fraction'),flush=True)
out=dict(registration=registration,rows=rows,new_gravity_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
