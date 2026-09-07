"""Radial-only interpolation refinement; old force gates are not overwritten."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-column-force-001/run003'
def main():
    with (P/'radial-fields.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    data={}
    for r in rows:data.setdefault((r['case'],r['component'],r['grid'],int(r['azimuth_nodes'])),[]).append(r)
    gates=[]
    for key,rr in data.items():
        if key[-1]!=2048:continue
        radius=np.array([float(r['r_kpc']) for r in rr]);value=np.array([float(r['gbar_inward']) for r in rr])
        for step in [.025,.0125]:
            coarse=np.isclose(radius/step,np.rint(radius/step));mid=(~coarse)&np.isclose(radius/(step/2),np.rint(radius/(step/2)));pred=PchipInterpolator(radius[coarse],value[coarse])(radius[mid]);truth=value[mid]
            for scope,mask in [('full',np.ones(len(truth),bool)),('aperture',(radius[mid]>=.75)&(radius[mid]<=2.5))]:
                delta=pred[mask]-truth[mask];rms=float(np.linalg.norm(delta)/np.linalg.norm(truth[mask]));worst=float(np.max(abs(delta/truth[mask])));gates.append(dict(case=key[0],component=key[1],grid=key[2],coarse_spacing=step,scope=scope,rms=rms,max_point=worst,passed=rms<.01 and worst<.03))
    table=[]
    for case in sorted(set(r['case'] for r in rows)):
        for component in ['stellar_luminosity','atomic_helium','co21','total']:
            for n,l in zip(data[(case,component,'nf96',2048)],data[(case,component,'l32fine',2048)]):table.append(dict(case=case,component=component,r_kpc=n['r_kpc'],sigma_hi_msun_pc2=n['sigma_hi_msun_pc2'],newton_gbar=n['gbar_inward'],log_extra_gbar=l['gbar_inward'],newton_plus_log_gbar=float(n['gbar_inward'])+float(l['gbar_inward']),newton_radial_rms_deviation=n['radial_rms_deviation'],log_radial_rms_deviation=l['radial_rms_deviation'],newton_tangential_rms=n['tangential_rms'],log_tangential_rms=l['tangential_rms']))
    with (P/'column-force-table.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    out=dict(gates=gates,passed=sum(g['passed'] for g in gates),total=len(gates),table_rows=len(table));(P/'review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([g for g in gates if not g['passed']],indent=2))
if __name__=='__main__':main()
