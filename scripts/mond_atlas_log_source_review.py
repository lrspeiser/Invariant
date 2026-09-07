"""Replay convergence and conditional geometry comparisons from saved vectors."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-log-source-001/run001'
def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def vec(rows):return np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in rows])
def main():
    for p,h in json.loads((P/'bindings.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    rows=read(P/'fields.csv');old=read(R/'work/gravity-first-principles/mond-atlas-spatial-program-001/fields.csv');cases=sorted(set(r['case'] for r in rows));gates=[];patterns=[]
    for case in cases:
        for component in sorted(set(r['component'] for r in rows)):
            for model in ['log_extra','newton','newton_plus_log']:
                values=[vec([r for r in rows if r['case']==case and r['component']==component and r['model']==model and int(r['level'])==level]) for level in [1,2]];a,b=values
                rms=float(np.linalg.norm(a-b)/np.linalg.norm(b));worst=float(np.max(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)))
                gates.append(dict(case=case,component=component,model=model,rms=rms,max_point=worst,passed=rms<.01 and worst<.03))
        for radius in [1.,3.,6.]:
            for z in [0.,.4]:
                def select(data,model=None):return [r for r in data if r['case']==case and r['component']=='total' and int(r['level'])==2 and float(r['r'])==radius and float(r['z'])==z and (model is None or r['model']==model)]
                lr=select(rows,'log_extra');lg=vec(lr);ng=vec(select(rows,'newton'));fg=vec(select(old));theta=np.array([float(r['theta']) for r in lr]);er=np.column_stack([np.cos(theta),np.sin(theta),np.zeros(12)]);et=np.column_stack([-np.sin(theta),np.cos(theta),np.zeros(12)])
                pattern=dict(case=case,r=radius,z=z)
                for name,v in [('log_extra',lg),('newton',ng),('finite_extra',fg)]:
                    cylindrical=np.column_stack([np.sum(v*er,axis=1),np.sum(v*et,axis=1),v[:,2]])
                    pattern[name+'_azimuth_vector_variation']=float(np.linalg.norm(cylindrical-cylindrical.mean(axis=0))/np.linalg.norm(cylindrical));pattern[name+'_tangential_fraction']=float(np.linalg.norm(cylindrical[:,1])/np.linalg.norm(cylindrical));pattern[name+'_mean_inward']=float(-cylindrical[:,0].mean())
                pattern['log_to_newton_rms']=float(np.linalg.norm(lg)/np.linalg.norm(ng));pattern['log_to_finite_rms']=float(np.linalg.norm(lg)/np.linalg.norm(fg));patterns.append(pattern)
    # Component sums independently reconstruct saved total fields.
    sumerr=0.
    for case in cases:
        for level in range(3):
            for model in ['log_extra','newton','newton_plus_log']:
                subset=[r for r in rows if r['case']==case and int(r['level'])==level and r['model']==model]
                total=vec([r for r in subset if r['component']=='total']);components=[vec([r for r in subset if r['component']==c]) for c in ['stellar_luminosity','atomic_helium','co21']];sumerr=max(sumerr,float(np.max(abs(total-sum(components)))))
    result=dict(status='NUMERICAL_GATES_REPLAYED',gates=gates,patterns=patterns,component_sum_error=sumerr,passed_gates=sum(g['passed'] for g in gates),total_gates=len(gates),observed_response_scored=False)
    (P/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps([g for g in gates if not g['passed']],indent=2));print('passes',result['passed_gates'],'/',len(gates))
if __name__=='__main__':main()
