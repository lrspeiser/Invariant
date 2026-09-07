"""Real-source conditional balance on the independently resolved radial range."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from mond_atlas_hi_column import smooth
from mond_atlas_pressure_support import SurfaceColumn, surface_balance

R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-column-balance-001'
FACTORS={'baseline':(1.,1.,1.), 'stellar_ML_0.4':(2/3,1.,1.),
         'stellar_ML_0.8':(4/3,1.,1.), 'HI_0.8':(1.,.8,1.),
         'HI_1.2':(1.,1.2,1.), 'CO_0.5':(1.,1.,.5), 'CO_2':(1.,1.,2.)}


def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    table=R/'work/gravity-first-principles/mond-atlas-column-force-001/run003/column-force-table.csv'
    packet=R/'work/private/mond-atlas-hi-column-001/run001/f4-column.npz'
    paths=[Path(__file__),P/'PREFLIGHT.md',table,packet,R/'scripts/mond_atlas_hi_column.py',
           R/'scripts/mond_atlas_pressure_support.py',
           R/'work/gravity-first-principles/mond-atlas-column-force-001/run002/review.json',
           R/'work/gravity-first-principles/mond-atlas-column-force-001/run003/review.json',
           R/'work/gravity-first-principles/mond-atlas-hi-column-001/CLOSURE_ADDENDUM.md']
    (out/'bindings.json').write_text(json.dumps({p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    for report in paths[6:8]:
        gates=json.loads(report.read_text())['gates']
        aperture=[gate for gate in gates if gate['scope']=='aperture']
        assert aperture and all(gate['passed'] for gate in aperture),str(report)
    with table.open(newline='') as f:
        source=[row for row in csv.DictReader(f) if .75<=float(row['r_kpc'])<=2.5]
    with np.load(packet) as z:
        raw_r=z['raw_radius_kpc'];raw_sigma=z['raw_sigma_HI_msun_pc2']
    output=[];summaries=[];speeds={}
    for case in sorted(set(row['case'] for row in source)):
        def records(component):
            return sorted([row for row in source if row['case']==case and row['component']==component],key=lambda row:float(row['r_kpc']))
        total=records('total');r=np.array([float(row['r_kpc']) for row in total]);sigma=np.array([float(row['sigma_hi_msun_pc2']) for row in total])
        assert len(r)>2 and np.all(sigma>0)
        smoothed,derivative=smooth(raw_r,raw_sigma,r)
        components=[records(name) for name in ('stellar_luminosity','atomic_helium','co21')]
        for rows in components:
            assert np.array_equal([float(row['r_kpc']) for row in rows],r)
            assert np.allclose([float(row['sigma_hi_msun_pc2']) for row in rows],sigma,rtol=1e-12)
        for model,column in [('newton','newton_gbar'),('newton_plus_log','newton_plus_log_gbar')]:
            base=np.array([[float(row[column]) for row in rows] for rows in components])
            assert np.allclose(base.sum(axis=0),[float(row[column]) for row in total],rtol=1e-12,atol=1e-10)
            for material,factors in FACTORS.items():
                g=np.array(factors)@base
                for support in (5.,10.,15.):
                    hi=factors[1]
                    balance=surface_balance(SurfaceColumn(r,sigma*hi,smoothed*hi*support**2,derivative*hi*support**2),g)
                    valid=balance.feasible
                    speed=balance.speed() if valid.all() else None
                    summaries.append(dict(case=case,model=model,material=material,pressure_reference_km_s=support,
                        status=balance.status,negative_rotation_squared_count=int((~valid).sum()),
                        minimum_rotation_squared=float(balance.rotation_squared.min()),
                        speed_min=None if speed is None else float(speed.min()),
                        speed_max=None if speed is None else float(speed.max()),
                        local_effective_pressure_speed_range=(support*np.sqrt(smoothed/sigma))[[np.argmin(smoothed/sigma),np.argmax(smoothed/sigma)]].tolist()))
                    if speed is not None:speeds[(case,model,material,support)]=speed
                    for i,radius in enumerate(r):
                        output.append(dict(case=case,model=model,material=material,pressure_reference_km_s=support,
                            r_kpc=radius,raw_sigma_HI=sigma[i]*hi,pressure=smoothed[i]*hi*support**2,
                            pressure_gradient=derivative[i]*hi*support**2,inward_force=g[i],
                            pressure_support_km2_s2=balance.support[i],rotation_squared=balance.rotation_squared[i],
                            speed_km_s='' if speed is None else speed[i],whole_profile_feasible=bool(valid.all())))
    with (out/'profiles.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    comparisons=[]
    for case in sorted(set(row['case'] for row in source)):
        for support in (5.,10.,15.):
            n=speeds.get((case,'newton','baseline',support));l=speeds.get((case,'newton_plus_log','baseline',support))
            if n is not None and l is not None:
                comparisons.append(dict(case=case,pressure_reference_km_s=support,median_extra_speed_km_s=float(np.median(l-n)),
                    extra_speed_range_km_s=[float((l-n).min()),float((l-n).max())]))
    summary=dict(status='CONDITIONAL_REAL_SOURCE_BALANCE_ONLY',profiles=len(summaries),rows=len(output),
        feasible_profiles=sum(s['status']=='STEADY_CIRCULAR_SOLUTION' for s in summaries),
        models=summaries,baseline_comparisons=comparisons,observed_gravity_scores=0,
        evaluated_radial_range_kpc=[.75,2.5],full_emission_support_validated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='models'}))


if __name__=='__main__':run()
