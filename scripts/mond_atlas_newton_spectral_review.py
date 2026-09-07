"""Saved-field convergence audit and isolated real-space reference at far points."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-newton-spectral-001/run001'
def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def vec(rows):return np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in rows])
def main():
    for p,h in json.loads((P/'bindings.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    rows=read(P/'fields.csv');cases=sorted(set(r['case'] for r in rows));components=sorted(set(r['component'] for r in rows));gates=[];sumerr=0
    for case in cases:
        for component in components:
            def select(grid):return vec([r for r in rows if r['case']==case and r['component']==component and r['grid']==grid])
            for name,ga,gb in [('early_box','box32','box64'),('box','box64','box96'),('resolution','box96','fine96')]:
                a,b=select(ga),select(gb);rms=float(np.linalg.norm(a-b)/np.linalg.norm(b));worst=float(np.max(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)));gates.append(dict(case=case,component=component,comparison=name,rms=rms,max_point=worst,passed=rms<.01 and worst<.03,required=name!='early_box'))
        for grid in ['box32','box64','box96','fine96']:
            subset=[r for r in rows if r['case']==case and r['grid']==grid];total=vec([r for r in subset if r['component']=='total']);parts=[vec([r for r in subset if r['component']==c]) for c in components if c!='total'];sumerr=max(sumerr,float(np.max(abs(total-sum(parts)))))
    sym=max(abs(float(r['gz'])) for r in rows if float(r['z'])==0);assert sym<1e-10
    req=[g for g in gates if g['required']];result=dict(gates=gates,required_gates=len(req),required_passes=sum(g['passed'] for g in req),midplane_vertical_max=sym,component_sum_error=sumerr,observed_response_scored=False)
    (P/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps([g for g in gates if g['component']=='total'],indent=2));print('Required passes',result['required_passes'],len(req))
if __name__=='__main__':main()
