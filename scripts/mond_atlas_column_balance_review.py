"""Compare only already-declared source/pressure alternatives; no fitting."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-column-balance-001'
path=P/'run002/profiles.csv'
with path.open(newline='') as f:rows=list(csv.DictReader(f))
groups={}
for row in rows:
    key=(row['case'],row['model'],row['material'],float(row['pressure_reference_km_s']))
    groups.setdefault(key,[]).append(row)
vectors={key:np.array([float(r['speed_km_s']) for r in sorted(values,key=lambda r:float(r['r_kpc']))])
         for key,values in groups.items() if all(r['whole_profile_feasible']=='True' for r in values)}
results=[]
for case in sorted(set(k[0] for k in vectors)):
    for support in (5.,10.,15.):
        baseline=vectors[case,'newton','baseline',support]
        extra=vectors[case,'newton_plus_log','baseline',support]-baseline
        for material in ('stellar_ML_0.4','stellar_ML_0.8','HI_0.8','HI_1.2','CO_0.5','CO_2'):
            delta=vectors[case,'newton',material,support]-baseline
            results.append(dict(case=case,support_reference_km_s=support,material=material,
                median_speed_change_km_s=float(np.median(delta)),
                rms_speed_change_km_s=float(np.sqrt(np.mean(delta**2))),
                log_extra_rms_speed_change_km_s=float(np.sqrt(np.mean(extra**2))),
                signed_cosine_with_log_change=float(delta@extra/(np.linalg.norm(delta)*np.linalg.norm(extra))),
                rms_difference_from_fixed_log_prediction_km_s=float(np.sqrt(np.mean((delta-extra)**2)))))
pressure=[]
for case in sorted(set(k[0] for k in vectors)):
    for model in ('newton','newton_plus_log'):
        delta=vectors[case,model,'baseline',15.]-vectors[case,model,'baseline',5.]
        pressure.append(dict(case=case,model=model,reference_change='5_to_15_km_s',
            median_speed_change_km_s=float(np.median(delta)),speed_change_range=[float(delta.min()),float(delta.max())]))
result=dict(status='PREDECLARED_PREDICTION_SENSITIVITY_NOT_OBSERVATIONAL_EVIDENCE',
            formula_or_mass_parameters_fitted=0, material_comparisons=results,pressure_comparisons=pressure,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(P/'prediction-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(stellar_upper=[r for r in results if r['material']=='stellar_ML_0.8'],pressure=pressure)))
