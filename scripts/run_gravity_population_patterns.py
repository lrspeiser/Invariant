"""Recheck historical stellar-population leads against stronger controls."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from run_gravity_broad_patterns import save,csvsave,split
from run_gravity_matched_concentration import bootstrap_mean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/gravity-first-principles/population-patterns-001'
BASE=['log_stellar_mass','log_half_light_radius','log_surface_density','axis_ratio','sersic_index','g_minus_r_color','redshift','log_surface_brightness','log_snr']
GROUPS={'break':['dn4000'],'balmer':['hdelta_a','hgamma_a','hbeta'],
        'star_formation':['signed_log_halpha','log_specific_sfr'],
        'all_population':['dn4000','d4000','hdelta_a','hgamma_a','hbeta','signed_log_halpha','log_specific_sfr'],
        'crossing_proxy':['mass_size_crossing_proxy']}

def rows(item):
    stem='item-12-manga-dynamical-age' if item==12 else 'item-13-manga-relaxation-mergers'
    directory=ROOT/('runs/gravity/roadmap/'+stem+'-v1-source');p=directory/'extraction-summary.json'
    receipt=json.loads((ROOT/('runs/gravity/roadmap/'+stem+'-v1.json')).read_text())
    # Git's Windows checkout changes the terminal line ending; authenticate the
    # original LF representation without changing any parsed source content.
    normalized=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert normalized==receipt['inputs']['extraction_summary_sha256']
    data=json.loads(p.read_text());assert data['counts']['confirmation_response_rows']==0
    sample=json.loads((directory/'sample-manifest.json').read_text())
    allowed={r['plateifu'] for r in sample['objects'] if r['role']=='exploration'}
    records=data['rows']
    assert all(r['quality_pass'] and r['plateifu'] in allowed for r in records)
    for r in records:
        a=float(r['halpha_ew']);r['signed_log_halpha']=float(np.sign(a)*np.log1p(abs(a)))
    return records,dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_LF_sha256=normalized)

def matrix(records,keys):
    return np.array([[float(r[k]) for k in keys] for r in records])

def estimator(kind,value):
    if kind=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=value))
    return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=int(value),max_features=1.,random_state=9143,n_jobs=2)

def choose(X,y,names,kind,salt):
    folds=split(names,salt,3);grid=[.03,.3,3.,30.] if kind=='ridge' else [5,15,30]
    losses=[]
    for value in grid:
        pred=np.full(len(y),np.nan)
        for f in range(3):
            use=folds!=f;model=estimator(kind,value);model.fit(X[use],y[use]);pred[~use]=model.predict(X[~use])
        losses.append(np.mean((pred-y)**2))
    return grid[int(np.argmin(losses))]

def evaluate(records,kind,base=BASE,groups=GROUPS):
    y=np.log10(matrix(records,['stellar_sigma_1re_km_s'])[:,0]);names=np.array([r['plateifu'] for r in records]);folds=np.array([r['outer_fold'] for r in records],int)
    columns={'baseline':base,**{k:base+v for k,v in groups.items()}};pred={k:np.full(len(y),np.nan) for k in columns};selections=[]
    for key,keys in columns.items():
        X=matrix(records,keys)
        for f in range(5):
            tr=folds!=f;te=~tr
            value=choose(X[tr],y[tr],names[tr],kind,'population-inner'+str(f));model=estimator(kind,value);model.fit(X[tr],y[tr]);pred[key][te]=model.predict(X[te]);selections.append(dict(group=key,outer_fold=f,value=value))
    summary=[];baseline=(pred['baseline']-y)**2
    for key in groups:
        err=(pred[key]-y)**2;delta=baseline-err
        summary.append(dict(group=key,model=kind,galaxies=len(y),mse_gain_percent=float(100*(1-err.mean()/baseline.mean())),baseline_rmse_dex=float(np.sqrt(baseline.mean())),full_rmse_dex=float(np.sqrt(err.mean())),galaxies_improving=int(sum(delta>0)),gain_bootstrap95=bootstrap_mean(delta)))
    exported=[dict(plateifu=str(n),y=float(y[i]),**{k:float(v[i]) for k,v in pred.items()}) for i,n in enumerate(names)]
    return dict(summary=summary,selections=selections),exported

def transport(train,test,kind):
    y=np.log10(matrix(train,['stellar_sigma_1re_km_s'])[:,0]);target=np.log10(matrix(test,['stellar_sigma_1re_km_s'])[:,0]);names=np.array([r['plateifu'] for r in train]);pred={};selected={}
    for key,keys in {'baseline':BASE,**{k:BASE+v for k,v in GROUPS.items()}}.items():
        X=matrix(train,keys);Z=matrix(test,keys);value=choose(X,y,names,kind,'transport-inner');model=estimator(kind,value);model.fit(X,y);pred[key]=model.predict(Z);selected[key]=value
    baseline=(pred['baseline']-target)**2;result=[]
    for key in GROUPS:
        err=(pred[key]-target)**2;result.append(dict(group=key,model=kind,train_galaxies=len(train),test_galaxies=len(test),mse_gain_percent=float(100*(1-err.mean()/baseline.mean())),baseline_rmse_dex=float(np.sqrt(baseline.mean())),full_rmse_dex=float(np.sqrt(err.mean())),gain_bootstrap95=bootstrap_mean(baseline-err)))
    return dict(summary=result,selected_on_training_only=selected),[dict(plateifu=r['plateifu'],y=float(target[i]),**{k:float(v[i]) for k,v in pred.items()}) for i,r in enumerate(test)]

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);assert not any(OUT.iterdir())
    a,srca=rows(12);b,srcb=rows(13)
    assert not {r['mangaid'] for r in a}&{r['mangaid'] for r in b}
    # Source-only crossing clock is exactly a linear mass/size combination.
    crossing=matrix(a,['mass_size_crossing_proxy'])[:,0];design=np.column_stack([np.ones(len(a)),matrix(a,['log_stellar_mass','log_half_light_radius'])]);coef=np.linalg.lstsq(design,crossing,rcond=None)[0]
    alias_residual=float(np.max(abs(crossing-design@coef)))
    result=dict(status='COMPLETED_POPULATION_PROXY_RECHECK_NOT_GRAVITY_CAUSALITY',sources=[srca,srcb],sample_overlap=0,crossing_alias_max_residual=alias_residual,runs={},transport={},sensitivities={},disturbance={})
    for kind in ['ridge','trees']:
        print('population',kind,flush=True)
        scores,pred=evaluate(a,kind);result['runs'][kind]=scores;csvsave(OUT/('predictions_'+kind+'.csv'),pred)
        transferred,pred=transport(a,b,kind);result['transport'][kind]=transferred;csvsave(OUT/('transport_'+kind+'.csv'),pred)
        for label,subset in [('snr10',[r for r in a if float(r['snr_med_g'])>=10]),('sigma70',[r for r in a if float(r['stellar_sigma_1re_km_s'])>=70])]:
            score,pred=evaluate(subset,kind,groups={'all_population':GROUPS['all_population']});result['sensitivities'][kind+'_'+label]=score
        # Broad request also permits a new development recheck of existing imaging
        # disturbance predictors. Age features remain in the common baseline.
        disturbances={'asymmetry':['asymmetry'],'clumpiness':['clumpiness'],'tidal_bar':['tidal','bar_strength'],'all_disturbance':['asymmetry','clumpiness','tidal','bar_strength','concentration']}
        score,pred=evaluate(b,kind,base=BASE+GROUPS['all_population'],groups=disturbances);result['disturbance'][kind]=score;csvsave(OUT/('disturbance_'+kind+'.csv'),pred)
    result['elapsed_seconds']=time.time()-started;save(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['runs','transport','sensitivities','disturbance']},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
