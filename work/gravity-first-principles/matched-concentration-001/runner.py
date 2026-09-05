"""Development-only matched acceleration/concentration test; see frozen protocol."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'work/gravity-first-principles/matched-concentration-001'
PROTOCOL = ROOT/'configs/gravity_matched_concentration_v1.json'
KNOTS = np.array(json.loads(PROTOCOL.read_text())['flexible_knots_log10_m_s2'])
CONVERSION = 1e6/3.085677581491367e19

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def read_inputs():
    sources = ['configs/sparc_rotation_curves_full_v1.json',
               'configs/sparc_surface_brightness_exploration_v1.json',
               'work/gravity-first-principles/map-response-metadata-001/SPARC_Lelli2016c.mrt']
    curves = {g['name']: g for g in json.loads((ROOT/sources[0]).read_text())['galaxies']}
    photo = json.loads((ROOT/sources[1]).read_text())
    assert photo['counts']['confirmation_galaxies'] == 0
    meta = {}
    for line in (ROOT/sources[2]).read_text().splitlines():
        fields = line.split()
        name = fields[0] if fields else ''
        if name in curves:
            assert len(fields)==19, (name, fields)
            meta[name] = dict(inc=float(fields[5]), q=int(fields[17]))
    assert len(meta) == 175
    rows = []
    for g in photo['galaxies']:
        name = g['galaxy']; c = curves[name]
        assert len(g['rows']) == len(c['rows'])
        for i, (v, p) in enumerate(zip(c['rows'], g['rows'])):
            rows.append(dict(name=name, row=i, r=float(v[0]), v=float(v[1]), ev=float(v[2]),
                gas=float(v[3]), disk=float(v[4]), bul=float(v[5]),
                sd=float(p[0]), sb=float(p[1]), **meta[name]))
    return rows, sources

def prepare(rows, ml=.5, qmax=2, inclination=(30,80)):
    selected=[]; exclusions={}
    for r in rows:
        b2 = r['gas']*abs(r['gas']) + ml*r['disk']**2 + 1.4*ml*r['bul']**2
        sigma = ml*r['sd'] + 1.4*ml*r['sb']
        reason = ('quality' if r['q']>qmax else 'inclination' if not inclination[0]<=r['inc']<=inclination[1]
            else 'nonpositive' if min(r['r'],r['v'],b2,sigma)<=0 else 'velocity_error' if r['ev']/r['v']>.1 else None)
        if reason:
            exclusions[reason]=exclusions.get(reason,0)+1; continue
        selected.append(dict(**r,x=float(np.log10(b2/r['r']*CONVERSION)),
            y=float(np.log10(r['v']**2/r['r']*CONVERSION)),z=float(np.log10(sigma)),b2=b2))
    names, counts=np.unique([r['name'] for r in selected],return_counts=True)
    admitted=set(names[counts>=5]); selected=[r for r in selected if r['name'] in admitted]
    arrays={k:np.array([r[k] for r in selected]) for k in selected[0]}
    return arrays,exclusions

def basis(x):
    return np.stack([np.interp(x,KNOTS,np.eye(len(KNOTS))[i]) for i in range(len(KNOTS))],axis=1)

def weights(names):
    unique,count=np.unique(names,return_counts=True)
    lookup=dict(zip(unique,count))
    return np.array([1/lookup[n] for n in names])

def smooth_fit(B,target,w):
    penalty=np.diff(np.eye(B.shape[1]),n=2,axis=0)
    return np.linalg.solve(B.T@(w[:,None]*B)+.01*penalty.T@penalty+np.eye(B.shape[1])*1e-10,
                           B.T@(w*target))

def rar(x,loga):
    return x-np.log10(-np.expm1(-np.sqrt(10**(x-loga))))

def fit_predict(d,train,test):
    x,y,z=d['x'],d['y'],d['z']; w=weights(d['name'][train]); sw=np.sqrt(w)
    B=basis(x); zfit=smooth_fit(B[train],z[train],w); zr=z-B@zfit
    base=smooth_fit(B[train],y[train],w)
    residual=y[train]-B[train]@base
    # Joint penalized fit with the same acceleration basis, plus one coefficient.
    A=np.column_stack([B,zr]); penalty=np.zeros((len(KNOTS)-2,A.shape[1]));penalty[:,:-1]=np.diff(np.eye(len(KNOTS)),n=2,axis=0)
    coef=np.linalg.solve(A[train].T@(w[:,None]*A[train])+.01*penalty.T@penalty+np.eye(A.shape[1])*1e-10,
                         A[train].T@(w*y[train]))
    r0=least_squares(lambda p:sw*(rar(x[train],p[0])-y[train]),[-10.],bounds=([-12.],[-9.]))
    r1=least_squares(lambda p:sw*(rar(x[train],p[0])+p[1]*zr[train]-y[train]),
                     [r0.x[0],0.],bounds=([-12.,-1.],[-9.,1.]))
    assert r0.success and r1.success
    support=[]
    for idx in np.flatnonzero(test):
        neighbors=train & (abs(x-x[idx])<=.15)
        support.append(len(np.unique(d['name'][neighbors]))>=5 and
                       np.min(z[neighbors],initial=np.inf)<=z[idx]<=np.max(z[neighbors],initial=-np.inf))
    return dict(flexible=B[test]@base, flexible_density=A[test]@coef,
                rar=rar(x[test],r0.x[0]),rar_density=rar(x[test],r1.x[0])+r1.x[1]*zr[test],
                support=np.array(support), beta_flexible=float(coef[-1]),beta_rar=float(r1.x[1]))

def bootstrap_mean(a,seed=853,n=4000):
    a=np.asarray(a);rng=np.random.default_rng(seed)
    return np.quantile(np.mean(rng.choice(a,(n,len(a)),replace=True),axis=1),[.025,.975]).tolist()

def cross_validate(d, folds=None):
    names=np.unique(d['name']); columns=['flexible','flexible_density','rar','rar_density']
    predictions={k:np.full(len(d['x']),np.nan) for k in columns}; support=np.zeros(len(d['x']),bool); coefficients=[]
    if folds is None:folds=[[n] for n in names]
    for held in folds:
        test=np.isin(d['name'],held); result=fit_predict(d,~test,test)
        for k in columns:predictions[k][test]=result[k]
        support[test]=result['support']
        coefficients.append({k:result[k] for k in ['beta_flexible','beta_rar']})
    scores=[]
    for name in names:
        use=(d['name']==name)&support
        if np.sum(use)<3:continue
        row=dict(name=str(name),positions=int(np.sum(use)))
        row.update({k:float(np.mean((d['y'][use]-predictions[k][use])**2)) for k in columns})
        scores.append(row)
    summary=dict(galaxies=len(names),positions=len(d['x']),scored_galaxies=len(scores),
                 scored_positions=sum(r['positions'] for r in scores),coefficient_medians={k:float(np.median([r[k] for r in coefficients])) for k in coefficients[0]})
    for base in ['rar','flexible']:
        a=np.array([r[base] for r in scores]);b=np.array([r[base+'_density'] for r in scores])
        summary[base]=dict(baseline_rmse_dex=float(np.sqrt(a.mean())),density_rmse_dex=float(np.sqrt(b.mean())),
            mse_improvement_percent=float(100*(1-b.mean()/a.mean())),galaxies_improving=int(np.sum(b<a)),
            mean_mse_difference=float(np.mean(b-a)),mse_difference_bootstrap95=bootstrap_mean(b-a))
    return summary,scores,predictions,support

def matched(d):
    # Representatives and pairing depend only on source quantities; y read after selection.
    reps=[]
    bins=np.floor(d['x']/.1).astype(int)
    for name in np.unique(d['name']):
        for b in np.unique(bins[d['name']==name]):
            use=(d['name']==name)&(bins==b)
            reps.append(dict(name=str(name),bin=int(b),x=float(np.median(d['x'][use])),
                z=float(np.median(d['z'][use])),indices=np.flatnonzero(use)))
    candidates=[]
    for i,a in enumerate(reps):
        for j in range(i+1,len(reps)):
            b=reps[j];dx=abs(a['x']-b['x']);dz=abs(a['z']-b['z'])
            if a['name']!=b['name'] and a['bin']==b['bin'] and dx<=.05 and dz>=.5:
                candidates.append((dx,-dz,a['name'],b['name'],i,j))
    used=set();pairs=[]
    for *_,i,j in sorted(candidates):
        a,b=reps[i],reps[j]
        if a['name'] in used or b['name'] in used:continue
        used.update([a['name'],b['name']]);lo,hi=sorted([a,b],key=lambda r:r['z'])
        # Subtract the fixed published RAR, avoiding a fitted target-selected pair definition.
        residual=d['y']-rar(d['x'],np.log10(1.2e-10))
        difference=float(np.median(residual[lo['indices']])-np.median(residual[hi['indices']]))
        pairs.append(dict(diffuse=lo['name'],dense=hi['name'],x_diffuse=lo['x'],x_dense=hi['x'],
            concentration_ratio=float(10**(hi['z']-lo['z'])),diffuse_minus_dense_residual_dex=difference))
    values=np.array([r['diffuse_minus_dense_residual_dex'] for r in pairs])
    return dict(pairs=len(pairs),diffuse_has_larger_residual=int(np.sum(values>0)),
        mean_difference_dex=float(values.mean()),median_difference_dex=float(np.median(values)),
        mean_bootstrap95=bootstrap_mean(values),median_density_ratio=float(np.median([r['concentration_ratio'] for r in pairs]))),pairs

def fivefold(names):
    ordered=sorted(names,key=lambda n:hashlib.sha256(str(n).encode()).hexdigest())
    return [ordered[i::5] for i in range(5)]

def controls(d):
    name=np.unique(d['name'])[0];test=d['name']==name
    before=fit_predict(d,~test,test); mutated={k:v.copy() for k,v in d.items()};mutated['y'][test]+=10
    after=fit_predict(mutated,~test,test)
    assert all(np.array_equal(before[k],after[k]) for k in ['rar','rar_density','flexible','flexible_density','support'])
    p1=matched(d)[1];p2=matched(mutated)[1]
    assert [(p['diffuse'],p['dense']) for p in p1]==[(p['diffuse'],p['dense']) for p in p2]
    w=weights(d['name']);B=basis(d['x']);zr=d['z']-B@smooth_fit(B,d['z'],w)
    folds=fivefold(np.unique(d['name'])); synthetic={k:v.copy() for k,v in d.items()}
    synthetic['y']=rar(d['x'],np.log10(1.2e-10))
    null=cross_validate(synthetic,folds)[0]
    synthetic['y']-=.15*zr
    injection=cross_validate(synthetic,folds)[0]
    assert abs(null['coefficient_medians']['beta_rar'])<1e-6
    assert injection['coefficient_medians']['beta_rar']<-.1 and injection['rar']['mse_improvement_percent']>50
    # Counterexample: true gravity has no density term; errors in inferred stellar mass
    # change both the acceleration predictor and concentration feature coherently.
    rng=np.random.default_rng(9814);names=np.unique(d['name']); draws=[]
    for i in range(64):
        mock={k:v.copy() for k,v in d.items()}
        errors=dict(zip(names,rng.normal(0,.1,len(names))));massfactor=np.array([10**errors[n] for n in d['name']])
        galaxy_scatter=dict(zip(names,rng.normal(0,.08,len(names))))
        mock['y']=rar(d['x'],np.log10(1.2e-10))+np.array([galaxy_scatter[n] for n in d['name']])+rng.normal(0,.02,len(d['x']))
        b2=d['gas']*abs(d['gas'])+massfactor*(.5*d['disk']**2+.7*d['bul']**2)
        # Signed gas can yield nonpositive inferred force after perturbation; retain
        # only mathematically defined rows, recording their count per draw.
        good=b2>0
        mock['x']=np.log10(np.maximum(b2,1e-100)/d['r']*CONVERSION)
        mock['z']=d['z']+np.log10(massfactor)
        mock={k:v[good] for k,v in mock.items()}
        s=cross_validate(mock,folds)[0]
        draws.append(dict(draw=i,excluded_rows=int(np.sum(~good)),beta=s['coefficient_medians']['beta_rar'],
                          improvement=s['rar']['mse_improvement_percent']))
    return dict(held_target_leakage_check=True,pair_selection_target_invariance=True,
        synthetic_null=null,synthetic_injection=injection,shared_mass_error_null=draws)

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    # Refuse to overwrite any completed evidence; an empty failed-start directory
    # can be reused after repairing an input parser or runtime error.
    DEST.mkdir(parents=True,exist_ok=True)
    assert not any(DEST.iterdir()), 'output directory is not empty'
    rows,sources=read_inputs();d,excluded=prepare(rows)
    primary,scores,pred,support=cross_validate(d);match,pairs=matched(d)
    sensitivity={}
    for label,kw in [('ml_0.4',dict(ml=.4)),('ml_0.6',dict(ml=.6)),('quality_1',dict(qmax=1)),('inclination_40_75',dict(inclination=(40,75)))]:
        alt,_=prepare(rows,**kw);sensitivity[label]=cross_validate(alt)[0]
    check=controls(d)
    acceleration_bins=[]
    for low,high in [(-14,-11),(-11,-10),(-10,-9),(-9,-7)]:
        use=(d['x']>=low)&(d['x']<high);ratios=[];fractions=[]
        for n in np.unique(d['name'][use]):
            r=10**(d['y'][use&(d['name']==n)]-d['x'][use&(d['name']==n)])
            ratios.append(np.median(r));fractions.append(np.mean((r>=.8)&(r<=1.2)))
        if ratios:acceleration_bins.append(dict(low=low,high=high,galaxies=len(ratios),positions=int(np.sum(use)),
            median_of_galaxy_median_pull_ratios=float(np.median(ratios)),mean_galaxy_fraction_within20percent=float(np.mean(fractions))))
    result=dict(status='COMPLETED_EXPLORATORY_STELLAR_SURFACE_CONCENTRATION_TEST',primary=primary,
        matched_pairs=match,sensitivities=sensitivity,selection_exclusions=excluded,acceleration_bins=acceleration_bins,
        total_3d_density_measured=False,independent_confirmation=False,admitted_gravity_laws=0,
        protocol=json.loads(PROTOCOL.read_text()),sources=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sources])
    save(DEST/'result.json',result);save(DEST/'controls.json',check)
    write_csv(DEST/'galaxy-scores.csv',scores);write_csv(DEST/'matched-pairs.csv',pairs)
    records=[]
    for i in range(len(d['x'])):
        record={k:(v[i].item() if hasattr(v[i],'item') else v[i]) for k,v in d.items()}
        record.update({k:float(v[i]) for k,v in pred.items()});record['prediction_support']=bool(support[i]);records.append(record)
    write_csv(DEST/'positions.csv',records)
    fig,ax=plt.subplots(1,3,figsize=(14,4.4))
    sc=ax[0].scatter(d['x'],d['y']-d['x'],c=d['z'],s=6,cmap='viridis',alpha=.55)
    ax[0].axhline(0,color='black',ls='--');ax[0].set(xlabel='log10 ordinary-matter acceleration (m/s²)',ylabel='log10 required pull / ordinary-matter pull',title='Weak predicted pull: larger discrepancy')
    fig.colorbar(sc,ax=ax[0],label='log10 stellar surface density proxy')
    values=np.array([p['diffuse_minus_dense_residual_dex'] for p in pairs]);ax[1].hist(values,bins=12,color='#467da9');ax[1].axvline(0,color='black',ls='--')
    ax[1].set(xlabel='Diffuse minus dense residual (dex)',ylabel='Disjoint galaxy pairs',title='Matched pull: positive supports diffuse boost')
    labels=['RAR','RAR + density','Flexible','Flexible + density']; vals=[primary[b]['baseline_rmse_dex'] if 'density' not in l else primary[b]['density_rmse_dex'] for l,b in zip(labels,['rar','rar','flexible','flexible'])]
    ax[2].bar(range(4),vals,color=['#467da9','#d49a3b']*2);ax[2].set_xticks(range(4),labels,rotation=25,ha='right');ax[2].set(ylabel='Held-galaxy RMS error (dex)',title='Lower error is better')
    fig.tight_layout();fig.savefig(DEST/'matched-concentration.png',dpi=160);plt.close(fig)
    (DEST/'runner.py').write_bytes(Path(__file__).read_bytes());(DEST/'protocol.json').write_bytes(PROTOCOL.read_bytes())
    print(json.dumps(dict(primary=primary,matched=match,sensitivities=sensitivity,acceleration_bins=acceleration_bins),indent=2))

if __name__=='__main__':main()
