"""Post-hoc continuous radial return models, not a three-dimensional relay."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from mond_atlas_clock_relay import G,A0,nfw_mass_shape
from mond_atlas_common import ROOT,digest,write_json,write_csv,read_json
from mond_atlas_pattern_learning import galaxy_folds
from run_mond_atlas_clock_relay import load_sources

OWN=ROOT/'work/gravity-first-principles/mond-atlas-nonclock-robust-001/return'
BOUNDS={
 'mond_adjusted':dict(mf=(.8,2.),a0_factor=(.1,3.)),
 'finite_mix':dict(mf=(.8,2.),A=(0.,100.),length_factor=(.1,30.),t=(0.,1.),q=(0.,1.)),
 'truncated_point_kernel':dict(mf=(.8,2.),A=(0.,100.),length_factor=(.1,30.),t=(0.,1.),C=(3.,100.)),
 'finite_flat_bridge':dict(mf=(.8,2.),eta=(0.,10.),delta=(.1,10.),C=(1.,100.))}
FAMILIES=['mond_fixed']+list(BOUNDS)


def extra_acceleration(r,rd,GM,family,p):
    r=np.asarray(r);rM=np.sqrt(GM/A0)
    if family=='finite_flat_bridge':
        return p['eta']*np.sqrt(GM*A0)*r/((r+p['delta']*rd)**2*(1+r/(p['C']*rM)))
    L=p['length_factor']*rd**(1-p['t'])*rM**p['t'];x=r/L
    if family=='truncated_point_kernel':return p['A']*GM/r**2*nfw_mass_shape(np.minimum(x,p['C']))
    if family=='finite_mix':return p['A']*GM/L**2*((1-p['q'])/(1+x)**2+p['q']*x/(1+x)**3)
    raise ValueError(family)


def predict(s,family,p):
    r=s['r'];mf=p.get('mf',1.)
    vb2=s['gas']*abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2)
    if family.startswith('mond'):
        gb=vb2/r;a0=A0*p.get('a0_factor',1.);return .5*np.log10(.5*r*(gb+np.sqrt(gb*gb+4*a0*gb)))
    GM=G*1e9*(.5*mf*s['luminosity']+1.33*s['hi'])
    return .5*np.log10(vb2+r*extra_acceleration(r,s['rd'],GM,family,p))


def fit(s,y,gi,train,family):
    if family=='mond_fixed':return dict(mf=1.,a0_factor=1.),[]
    # Slice response and sources before optimizer construction; held y never enters objective.
    mask=np.asarray(train,bool)[gi];ss={k:np.asarray(v)[mask] for k,v in s.items()};yy=np.asarray(y)[mask];gg=gi[mask]
    unique,counts=np.unique(gg,return_counts=True);count=dict(zip(unique,counts))
    weights=np.array([1/np.sqrt(len(unique)*count[g]) for g in gg])
    bounds=BOUNDS[family];keys=list(bounds);lo=np.array([bounds[k][0] for k in keys]);hi=np.array([bounds[k][1] for k in keys])
    def unpack(u):return dict(zip(keys,map(float,lo+u*(hi-lo))))
    starts=[]
    for start in [.15,.5,.85]:
        try:
            opt=least_squares(lambda u:(predict(ss,family,unpack(u))-yy)*weights,np.full(len(keys),start),bounds=(np.zeros(len(keys)),np.ones(len(keys))),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            starts.append(dict(start=start,kind='full',parameters=unpack(opt.x),training_mse=float(opt.fun@opt.fun),success=bool(opt.success),status=int(opt.status),nfev=int(opt.nfev),message=str(opt.message),normalized_parameters=opt.x.tolist()))
        except Exception as exc:starts.append(dict(start=start,kind='full',success=False,error=repr(exc)))
    if not family.startswith('mond'):
        amplitude='eta' if family=='finite_flat_bridge' else 'A'
        for start in [.15,.5,.85]:
            def zero(u):
                p=unpack(np.full(len(keys),.5));p[amplitude]=0.;p['mf']=float(.8+1.2*u[0]);return p
            try:
                opt=least_squares(lambda u:(predict(ss,family,zero(u))-yy)*weights,[start],bounds=([0.],[1.]),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                starts.append(dict(start=start,kind='zero_amplitude',parameters=zero(opt.x),training_mse=float(opt.fun@opt.fun),success=bool(opt.success),status=int(opt.status),nfev=int(opt.nfev),message=str(opt.message),normalized_parameters=None))
            except Exception as exc:starts.append(dict(start=start,kind='zero_amplitude',success=False,error=repr(exc)))
    good=[v for v in starts if v['success'] and np.isfinite(v['training_mse'])]
    if not good:raise RuntimeError('All starts failed '+family+json.dumps(starts))
    selected=min(good,key=lambda v:v['training_mse'])
    return selected['parameters'],starts


def run():
    started=time.time();out=OWN/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        config=read_json(ROOT/'configs/mond_atlas_clock_relay_v1.json');inventory=read_json(ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001/source-audit/inventory.json')
        bound=[Path(__file__),ROOT/'tests/test_mond_atlas_return_robust.py',OWN/'PREFLIGHT.md',OWN/'TRANSFER_ADDENDUM.md',OWN/'PRE_ACCESS_NUMERICAL_REPAIR.md',ROOT/'scripts/run_mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_pattern_learning.py',ROOT/'configs/mond_atlas_clock_relay_v1.json']
        for k in ['source_archive','source_metadata','registered_development_names']:
            p=ROOT/config[k]
            if digest(p)!=inventory['files'][config[k]]['sha256']:raise RuntimeError('Source hash mismatch')
            bound.append(p)
        write_json(out/'pre-access-bindings.json',dict(status='POST_HOC_DEVELOPMENT_ONLY',bounds=BOUNDS,files={p.relative_to(ROOT).as_posix():digest(p) for p in bound}))
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_return_robust.py','-v'],cwd=ROOT,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
        if test.returncode:raise RuntimeError('Pre-access tests failed')
        s,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
        if len(names)!=102:raise RuntimeError('Historical cohort differs')
        write_json(out/'access-receipt.json',dict(members=members,reserved_bodies_opened=0,eligible_names=names,galaxies=len(names),radii=len(y),historically_exposed=True))
        write_csv(out/'exclusions.csv',exclusions);gi=np.array([names.index(n) for n in ids]);starts=[];selected=[];held=[];pergal=[];metrics=[];biases=[];edges=[]
        for seed in config['fold_seeds']:
            folds=galaxy_folds(names,seed,config['fold_count']);write_csv(out/f'folds-{seed}.csv',[dict(galaxy=n,fold=int(folds[j])) for j,n in enumerate(names)])
            for family in FAMILIES:
                hp=np.empty(len(y))
                for fold in range(config['fold_count']):
                    p,attempts=fit(s,y,gi,folds!=fold,family);rowmask=folds[gi]==fold;hp[rowmask]=predict({k:v[rowmask] for k,v in s.items()},family,p)
                    selected.append(dict(seed=seed,family=family,fold=fold,parameters=p));starts.extend(dict(seed=seed,family=family,fold=fold,**v) for v in attempts)
                    for k,v in p.items():
                        if family=='mond_fixed':continue
                        low,high=BOUNDS[family][k];u=(v-low)/(high-low);edges.append(dict(seed=seed,family=family,fold=fold,parameter=k,value=v,lower_bound=u<=1e-4,upper_bound=u>=1-1e-4))
                residual=hp-y
                for j,n in enumerate(ids):held.append(dict(seed=seed,family=family,galaxy=n,fold=int(folds[gi[j]]),radial_index=int(radial_ids[j]),r_kpc=float(s['r'][j]),r_over_Rd=float(s['r'][j]/s['rd'][j]),log10_predicted_speed=float(hp[j]),log10_predicted_over_observed=float(residual[j])))
                mses=[]
                for j,n in enumerate(names):
                    mask=gi==j;mse=float(np.mean(residual[mask]**2));mses.append(mse);pergal.append(dict(seed=seed,family=family,galaxy=n,mse_logspeed=mse,signed_bias_dex=float(np.mean(residual[mask]))))
                metrics.append(dict(seed=seed,family=family,mse=float(np.mean(mses)),rmse_dex=float(np.sqrt(np.mean(mses)))))
                for group,mask in [('inner_r_over_Rd_lt1',s['r']<s['rd']),('outer_r_over_Rd_ge3',s['r']>=3*s['rd'])]:
                    vals=[np.mean(residual[(gi==j)&mask]) for j in range(len(names)) if np.any((gi==j)&mask)]
                    sq=[np.mean(residual[(gi==j)&mask]**2) for j in range(len(names)) if np.any((gi==j)&mask)]
                    biases.append(dict(seed=seed,family=family,group=group,galaxies=len(vals),signed_bias_dex=float(np.mean(vals)),rmse_dex=float(np.sqrt(np.mean(sq)))))
                print(seed,family,metrics[-1]['rmse_dex'],flush=True)
        write_json(out/'all-optimizer-starts.json',starts);write_json(out/'selected-parameters.json',selected)
        for file,rows in [('held-predictions.csv',held),('per-galaxy.csv',pergal),('metrics.csv',metrics),('radial-bias.csv',biases),('parameter-boundaries.csv',edges)]:write_csv(out/file,rows)
        transfer_starts=[];transfer_selected=[];transfer_predictions=[];transfer_galaxies=[];transfer_metrics=[]
        gasrich=np.array([1.33*meta[n]['hi']/(1.33*meta[n]['hi']+.5*meta[n]['luminosity'])>=.5 for n in names])
        for direction,train in [('gas_rich_to_stellar_rich',gasrich),('stellar_rich_to_gas_rich',~gasrich)]:
            for family in FAMILIES:
                p,attempts=fit(s,y,gi,train,family);mask=~train[gi];indices=np.flatnonzero(mask)
                hp=predict({k:v[mask] for k,v in s.items()},family,p);residual=hp-y[mask]
                transfer_selected.append(dict(direction=direction,family=family,parameters=p));transfer_starts.extend(dict(direction=direction,family=family,**v) for v in attempts)
                for j,i in enumerate(indices):transfer_predictions.append(dict(direction=direction,family=family,galaxy=ids[i],radial_index=int(radial_ids[i]),log10_predicted_speed=float(hp[j]),log10_predicted_over_observed=float(residual[j])))
                mses=[]
                for k in np.flatnonzero(~train):
                    local=gi[mask]==k;mse=float(np.mean(residual[local]**2));mses.append(mse)
                    transfer_galaxies.append(dict(direction=direction,family=family,galaxy=names[k],mse_logspeed=mse,signed_bias_dex=float(np.mean(residual[local]))))
                transfer_metrics.append(dict(direction=direction,family=family,training_galaxies=int(train.sum()),held_galaxies=int((~train).sum()),mse=float(np.mean(mses)),rmse_dex=float(np.sqrt(np.mean(mses)))))
        write_json(out/'transfer-all-starts.json',transfer_starts);write_json(out/'transfer-selected.json',transfer_selected)
        for file,rows in [('transfer-predictions.csv',transfer_predictions),('transfer-galaxies.csv',transfer_galaxies),('transfer-metrics.csv',transfer_metrics)]:write_csv(out/file,rows)
        means={f:float(np.mean([v['mse'] for v in metrics if v['family']==f])) for f in FAMILIES}
        summary=[dict(family=f,rmse_dex=float(np.sqrt(means[f])),gain_vs_fixed_percent=100*(1-means[f]/means['mond_fixed']),gain_vs_adjusted_percent=100*(1-means[f]/means['mond_adjusted']),selected_bound_events=sum((v['lower_bound'] or v['upper_bound']) for v in edges if v['family']==f)) for f in FAMILIES]
        write_json(out/'summary.json',dict(status='POST_HOC_DEVELOPMENT_ONLY',galaxies=len(names),radii=len(y),metrics=summary,optimizer_attempts=len(starts),failed_attempts=[v for v in starts if not v['success']],transfer_metrics=transfer_metrics,transfer_failed_attempts=[v for v in transfer_starts if not v['success']],seconds=time.time()-started,mechanism_identified=False))
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        write_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':run()
