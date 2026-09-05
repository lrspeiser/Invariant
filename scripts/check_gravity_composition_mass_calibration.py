"""Post-screen control: does a universal stellar M/L absorb the gas-share lead?"""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_gravity_broad_patterns as screen
import run_gravity_matched_concentration as prior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/gravity-first-principles/composition-mass-control-001'
GRID=[.3,.4,.5,.6,.7,.8]

def model(d,train):
    base=screen.baseline(d,train,'flexible');B=prior.basis(d['x']);w=prior.weights(d['name'][train]);F=d['F'][:,d['keys'].index('local_atomic_force_share')]
    F=F-B@prior.smooth_fit(B[train],F[train],w);F/=max(np.sqrt(np.average(F[train]**2,weights=w)),1e-5)
    matrices=[];tags=[]
    for shape in ['linear','tanh']:
        for gate in ['all','weak']:
            c=F if shape=='linear' else np.tanh(F)
            if gate=='weak':c=c/(1+10**d['x']/1.2e-10)
            c=c-B@prior.smooth_fit(B[train],c[train],w);c/=max(np.sqrt(np.average(c[train]**2,weights=w)),1e-5)
            for penalty in [.1,1,10]:
                coef=np.sum(w*c[train]*(d['y'][train]-base[train]))/(np.sum(w*c[train]**2)+penalty*np.sum(w))
                matrices.append(base+np.clip(c*coef,-.5,.5));tags.append(dict(shape=shape,gate=gate,penalty=penalty,coefficient=float(coef)))
    return base,np.column_stack(matrices),tags

def run(data):
    nominal=data[.5];n=len(nominal['x']);names=nominal['name'];outer=screen.split(names,'broad-A',5)
    preds={k:np.full(n,np.nan) for k in ['fixed_mass','calibrated_mass','calibrated_mass_composition']};choices=[];support=np.zeros(n,bool)
    for fold in range(5):
        train=outer!=fold;test=~train;inner=screen.split(names,'mass-inner'+str(fold),3);base_losses=[];extra_losses=[]
        for ml in GRID:
            bl=[];el=[]
            for k in range(3):
                tr=train&(inner!=k);va=train&(inner==k);b,p,t=model(data[ml],tr)
                bl.append(screen.score_matrix(data[ml],b[:,None],va)[0]);el.append(screen.score_matrix(data[ml],p,va))
            base_losses.append(np.mean(bl));extra_losses.append(np.mean(el,axis=0))
        bml=GRID[int(np.argmin(base_losses))];mi,ci=np.unravel_index(np.argmin(extra_losses),np.shape(extra_losses));cml=GRID[mi]
        preds['fixed_mass'][test]=model(data[.5],train)[0][test]
        preds['calibrated_mass'][test]=model(data[bml],train)[0][test]
        b,p,t=model(data[cml],train);preds['calibrated_mass_composition'][test]=p[test,ci]
        choices.append(dict(fold=fold,baseline_ml=bml,composition_ml=cml,**t[ci]))
        # Same support as the main nominal-acceleration screen for every comparison.
        good=screen.support(nominal,train,test)
        for name in np.unique(names[test]):
            if sum(good&(names==name))<3:good[names==name]=False
        support|=good
    rows=[]
    for name in np.unique(names[support]):
        use=(names==name)&support
        row=dict(name=str(name),positions=int(sum(use)))
        row.update({k:float(np.mean((nominal['y'][use]-p[use])**2)) for k,p in preds.items()});rows.append(row)
    fixed=np.array([r['fixed_mass'] for r in rows]);base=np.array([r['calibrated_mass'] for r in rows]);extra=np.array([r['calibrated_mass_composition'] for r in rows])
    return dict(galaxies=len(rows),positions=sum(r['positions'] for r in rows),selections=choices,
        fixed_mass_rmse=float(np.sqrt(fixed.mean())),calibrated_mass_rmse=float(np.sqrt(base.mean())),composition_rmse=float(np.sqrt(extra.mean())),
        calibration_gain_percent=float(100*(1-base.mean()/fixed.mean())),incremental_composition_gain_percent=float(100*(1-extra.mean()/base.mean())),
        incremental_gain_bootstrap95=prior.bootstrap_mean(base-extra),galaxy_scores=rows)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    screen.save(OUT/'protocol.json',dict(status='Post-screen diagnostic fixed before its response scores; not independent confirmation',grid=GRID,
        comparison='Nested universal stellar M/L calibration with and without the gas-share term; equal galaxy scores and fixed nominal support',
        synthetic='Known RAR from M/L=0.7 plus galaxy scatter, analyzed with nominal 0.5 inputs and the same calibration machinery. This is a constructed counterexample, not a probability model.'))
    data={ml:screen.make_data(ml=ml) for ml in GRID};nom=data[.5]
    for d in data.values():
        assert np.array_equal(d['name'],nom['name']) and np.array_equal(d['row'],nom['row'])
    real=run(data)
    rng=np.random.default_rng(62388);scatter=dict(zip(np.unique(nom['name']),rng.normal(0,.08,len(np.unique(nom['name'])))))
    source=screen.make_data(ml=.7);y=prior.rar(source['x'],np.log10(1.2e-10))+np.array([scatter[n] for n in nom['name']])+rng.normal(0,.02,len(nom['name']))
    synthetic=run({ml:{**d,'y':y} for ml,d in data.items()})
    result=dict(status='COMPLETED_POST_SCREEN_MASS_CALIBRATION_CONTROL',real=real,known_mass_mismatch_simulation=synthetic)
    screen.save(OUT/'result.json',result);print(json.dumps(result,indent=2)[:4200])

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
