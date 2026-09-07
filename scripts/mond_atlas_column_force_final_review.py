"""Independent final-table arithmetic and prescribed complete-model gate replay."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-column-force-001';O=P/'run004'
def main():
    for path,h in json.loads((O/'bindings.json').read_text()).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
    with gzip.open(O/'column-force-table.csv.gz','rt',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    data={}
    for r in rows:data.setdefault((r['case'],r['component']),[]).append(r)
    r=np.array([float(v['r_kpc']) for v in next(iter(data.values()))]);emitter=(r>=.0736569564)&(r<=6.0121585431);records=[];component_records=[]
    def array(case,c,k):return np.array([float(v[k]) for v in data[(case,c)]])
    for case in ['f4-stars-h0p1','f4-stars-h0p4']:
        for c in ['stellar_luminosity','atomic_helium','co21','total']:
            for model,key,delta in [('newton','newton_gbar','newton_grid_delta'),('log','log_extra_gbar','log_grid_delta')]:
                value=array(case,c,key);error=array(case,c,delta);a=value[emitter];e=error[emitter];component_records.append(dict(case=case,component=c,model=model,scope='actual_emitter_radial_extent',rms=float(np.linalg.norm(e)/np.linalg.norm(a)),max_point=float(np.max(abs(e/a))),max_absolute=float(np.max(abs(e))),passed=bool(np.linalg.norm(e)/np.linalg.norm(a)<.01 and np.max(abs(e/a))<.03)))
        for key in ['newton_gbar','log_extra_gbar','newton_grid_delta','log_grid_delta']:
            assert np.max(abs(array(case,'total',key)-sum(array(case,c,key) for c in ['stellar_luminosity','atomic_helium','co21'])))<1e-9
    variants=[('primary','f4-stars-h0p1',[1,1,1]),('stars_ML_0.4','f4-stars-h0p1',[2/3,1,1]),('stars_ML_0.8','f4-stars-h0p1',[4/3,1,1]),('HI_0.8','f4-stars-h0p1',[1,.8,1]),('HI_1.2','f4-stars-h0p1',[1,1.2,1]),('CO_0.5','f4-stars-h0p1',[1,1,.5]),('CO_2','f4-stars-h0p1',[1,1,2]),('stars_h0.4','f4-stars-h0p4',[1,1,1])]
    for name,case,factors in variants:
        keys=['newton_gbar','log_extra_gbar','newton_grid_delta','log_grid_delta','newton_angular_delta','log_angular_delta'];a={k:sum(f*array(case,c,k) for f,c in zip(factors,['stellar_luminosity','atomic_helium','co21'])) for k in keys}
        for model in ['newton','newton_plus_log']:
            value=a['newton_gbar']+(a['log_extra_gbar'] if model!='newton' else 0)
            for error_type in ['grid','angular']:
                delta=a['newton_'+error_type+'_delta']+(a['log_'+error_type+'_delta'] if model!='newton' else 0)
                for scope,mask in [('full',np.ones(len(r),bool)),('actual_emitter_radial_extent',emitter)]:
                    rms=float(np.linalg.norm(delta[mask])/np.linalg.norm(value[mask]));point=float(np.max(abs(delta[mask]/value[mask])));records.append(dict(variant=name,model=model,check=error_type,scope=scope,rms=rms,max_point=point,passed=rms<.01 and point<.03))
            for lo,hi,step in [(.05,.25,.0005),(.25,5.95,.0125),(5.95,6.025,.0005)]:
                mask=(r>=lo)&(r<=hi);rr=r[mask];vv=value[mask];coarse=np.isclose((rr-lo)/step,np.rint((rr-lo)/step));mid=~coarse;e=PchipInterpolator(rr[coarse],vv[coarse])(rr[mid])-vv[mid];rms=float(np.linalg.norm(e)/np.linalg.norm(vv[mid]));point=float(np.max(abs(e/vv[mid])));records.append(dict(variant=name,model=model,check='radial',scope=f'{lo}:{hi}',rms=rms,max_point=point,passed=rms<.01 and point<.03))
    result=dict(component_emitter_gates=component_records,prescribed_complete_model_gates=records,complete_model_passed=sum(v['passed'] for v in records),complete_model_total=len(records),source_parameters_refit=False,failed_component_gates_not_erased=True,observed_spectra_accessed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(O/'independent-final-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('complete',result['complete_model_passed'],len(records));print('component emitter fail',[v for v in component_records if not v['passed']]);print('complete fail',[v for v in records if not v['passed']])
if __name__=='__main__':main()
