"""Independent saved-force convergence and baseline sensitivity arithmetic."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001/fields001'
def main():
    for path,h in json.loads((P/'bindings.json').read_text()).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
    with gzip.open(P/'radial-fields.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    with gzip.open(R/'work/gravity-first-principles/mond-atlas-column-force-001/run004/column-force-table.csv.gz','rt') as f:baseline=list(csv.DictReader(f))
    data={}
    for row in rows:data.setdefault((row['case'],row['component'],row['grid'],row['azimuth_nodes']),[]).append(row)
    radii=np.array([float(r['r_kpc']) for r in next(iter(data.values()))]);gates=[];differences=[];case_names=sorted(set(r['case'] for r in rows))
    def get(case,component,grid,nt='2048'):return np.array([float(r['gbar_inward']) if r['gbar_inward'] else np.nan for r in data[(case,component,grid,nt)]])
    for case in case_names:
        for component in ['stellar_luminosity','atomic_helium','co21','total']:
            for check,ga,gb,nt in [('newton_grid','nc','nf','2048'),('log_grid','lc','lf','2048'),('newton_angular','nf','nf','1024'),('log_angular','lf','lf','1024')]:
                a,b=get(case,component,ga,nt),get(case,component,gb)
                for scope,mask in [('full',np.ones(len(radii),bool)),('aperture',(radii>=.75)&(radii<=2.5))]:
                    valid=np.isfinite(a[mask])&np.isfinite(b[mask]);delta=a[mask][valid]-b[mask][valid];ref=b[mask][valid];rms=float(np.linalg.norm(delta)/np.linalg.norm(ref));point=float(np.max(abs(delta/ref)));gates.append(dict(case=case,component=component,check=check,scope=scope,rms=rms,max_point=point,undefined_points=int(np.sum(~valid)),passed=bool(np.all(valid) and rms<.01 and point<.03)))
        baselinecase='f4-stars-h0p4' if case=='common30_stars_h0.4' else 'f4-stars-h0p1'
        base={round(float(r['r_kpc']),8):r for r in baseline if r['case']==baselinecase and r['component']=='total'}
        for model,grid,key in [('newton','nf','newton_gbar'),('log','lf','log_extra_gbar')]:
            val=get(case,'total',grid);old=np.array([float(base[round(r,8)][key]) for r in radii]);mask=(radii>=.75)&(radii<=2.5);difference=val[mask]-old[mask];differences.append(dict(case=case,baseline=baselinecase,model=model,scope='aperture',difference_rms_relative_to_baseline=float(np.linalg.norm(difference)/np.linalg.norm(old[mask])),median_ratio=float(np.median(val[mask]/old[mask])),minimum_ratio=float(np.min(val[mask]/old[mask])),maximum_ratio=float(np.max(val[mask]/old[mask]))))
    result=dict(gates=gates,differences=differences,passed=sum(g['passed'] for g in gates),total=len(gates),source_to_spectra_admitted=False,observed_spectra_accessed=False);(P/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('gates',result['passed'],len(gates));print(json.dumps(differences,indent=2));print('failed',[g for g in gates if not g['passed']])
if __name__=='__main__':main()
