"""Independent source-column table and gate replay, without response data."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-column-force-001';O=P/'run002'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    for p,h in json.loads((O/'bindings.json').read_text()).items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    with (O/'radial-fields.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    data={}
    for r in rows:data.setdefault((r['case'],r['component'],r['grid'],int(r['azimuth_nodes'])),[]).append(r)
    gates=[];rad=np.array([float(r['r_kpc']) for r in next(iter(data.values()))]);cases=sorted(set(r['case'] for r in rows));components=['stellar_luminosity','atomic_helium','co21','total']
    def values(key):return np.array([float(r['gbar_inward']) for r in data[key]])
    def compare(case,component,name,a,b,radii):
        for scope,mask in [('full',np.ones(len(radii),bool)),('aperture', (radii>=.75)&(radii<=2.5))]:
            aa,bb=a[mask],b[mask];delta=abs(aa-bb);rel=np.divide(delta,abs(bb),out=np.where(delta<1e-10,0.,np.inf),where=bb!=0);rms=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb));worst=float(np.max(rel));gates.append(dict(case=case,component=component,comparison=name,scope=scope,rms=rms,max_point=worst,passed=rms<.01 and worst<.03))
    for case in cases:
        for component in components:
            for name,a,b in [('newton_box','n64','n96'),('newton_grid','n96','nf96'),('log_vertical','l32mid','l32vertical'),('log_grid','l32vertical','l32fine'),('log_box','l32vertical','l48box')]:compare(case,component,name,values((case,component,a,2048)),values((case,component,b,2048)),rad)
            for grid in ['nf96','l32fine']:
                compare(case,component,grid+'_angular',values((case,component,grid,1024)),values((case,component,grid,2048)),rad)
                vv=values((case,component,grid,2048));coarse=np.isclose(rad/.05,np.rint(rad/.05));mid=(~coarse)&(rad>.05)&(rad<6);interp=PchipInterpolator(rad[coarse],vv[coarse])(rad[mid]);compare(case,component,grid+'_radial_interpolation',interp,vv[mid],rad[mid])
    # Linear decomposition and positive HI support checked on every stored rule.
    maxsum=0.
    for case in cases:
        for grid in sorted(set(r['grid'] for r in rows)):
            for nt in [512,1024,2048]:maxsum=max(maxsum,float(np.max(abs(values((case,'total',grid,nt))-sum(values((case,c,grid,nt)) for c in components[:-1])))))
    assert maxsum<1e-9 and all(float(r['sigma_hi_msun_pc2'])>0 for r in rows)
    output=[]
    for case in cases:
        for c in components:
            n=data[(case,c,'nf96',2048)];l=data[(case,c,'l32fine',2048)]
            for a,b in zip(n,l):output.append(dict(case=case,component=c,r_kpc=a['r_kpc'],sigma_hi_msun_pc2=a['sigma_hi_msun_pc2'],newton_gbar=a['gbar_inward'],log_extra_gbar=b['gbar_inward'],newton_plus_log_gbar=float(a['gbar_inward'])+float(b['gbar_inward']),newton_radial_rms_deviation=a['radial_rms_deviation'],log_radial_rms_deviation=b['radial_rms_deviation'],newton_tangential_rms=a['tangential_rms'],log_tangential_rms=b['tangential_rms']))
    with (O/'column-force-table.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    result=dict(gates=gates,passed=sum(g['passed'] for g in gates),total=len(gates),component_sum_max=maxsum,table_rows=len(output),observed_spectra_accessed=False);save(O/'review.json',result);print(json.dumps([g for g in gates if not g['passed']],indent=2));print(result['passed'],len(gates))
if __name__=='__main__':main()
