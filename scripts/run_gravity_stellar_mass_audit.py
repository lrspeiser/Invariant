"""Zero-halo stellar-mass audit: independent stellar anchors and Newtonian galaxy tests."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar, minimize, LinearConstraint
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acquire_gravity_mass_audit import PRIVATE, ROOT, save
import run_gravity_matched_concentration as sparc
import run_gravity_population_patterns as population

OUT = ROOT / 'work/gravity-first-principles/stellar-mass-audit-002'
PROTOCOL = ROOT / 'configs/gravity_stellar_mass_audit_v1.json'
LN10 = np.log(10.)


def csvsave(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def folds(names, salt, n=5):
    order = sorted(set(names), key=lambda x: hashlib.sha256((salt+str(x)).encode()).hexdigest())
    mapping = {s:i % n for i,s in enumerate(order)}
    return np.array([mapping[s] for s in names])


def groupmeans(values, names):
    return np.array([np.mean(values[np.asarray(names)==s]) for s in sorted(set(names))])


def boot(values, seed=73881, n=4000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    return np.quantile(rng.choice(values, (n, len(values))).mean(axis=1), [.025,.975]).tolist()


def metrics(actual, predicted, names):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    e = np.log10(predicted / actual); relative = predicted/actual-1
    return dict(objects=len(actual), systems=len(set(names)),
                log_rmse=float(np.sqrt(groupmeans(e*e, names).mean())),
                rms_fractional_error_percent=float(100*np.sqrt(groupmeans(relative**2, names).mean())),
                median_absolute_fractional_error_percent=float(100*np.median(abs(relative))),
                mean_predicted_minus_actual_percent=float(100*groupmeans(relative,names).mean()),
                mean_bias_bootstrap95_percent=[100*x for x in boot(groupmeans(relative,names))],
                within10percent=int(sum(abs(relative)<=.1)),
                actual_exceeds_prediction_by_factor2=int(sum(actual > 2*predicted)))


def comparison(y, base, full, names):
    a = groupmeans((np.log10(base/y))**2,names)
    b = groupmeans((np.log10(full/y))**2,names)
    return dict(mse_gain_percent=float(100*(1-b.mean()/a.mean())),
                paired_log_mse_gain_bootstrap95=boot(a-b))


def optical_design(data, kind, degree):
    v = np.array([r['MV'] for r in data]); k = np.array([r['MK'] for r in data])
    x = (v-14)/4 if kind.startswith('V') else (k-8)/2
    columns = [x**i for i in range(1,degree+1)]
    if 'color' in kind: columns.append(v-k)
    return np.column_stack(columns)


def optical_choice(data, y, names, train, kind):
    inner = folds(names, 'optical-inner', 3); options = []
    for degree in [2,3]:
        X = optical_design(data,kind,degree)
        for penalty in [.01,.1,1.]:
            errors = []
            for f in range(3):
                tr = train & (inner!=f); va = train & (inner==f)
                model = make_pipeline(StandardScaler(),Ridge(alpha=penalty)).fit(X[tr],y[tr])
                errors.extend(groupmeans((model.predict(X[va])-y[va])**2,names[va]))
            options.append((float(np.mean(errors)),degree,penalty))
    _, degree, penalty = min(options)
    X=optical_design(data,kind,degree)
    model=make_pipeline(StandardScaler(),Ridge(alpha=penalty)).fit(X[train],y[train])
    return model.predict(X), dict(degree=degree,penalty=penalty)


def optical_audit(data):
    names=np.array([r['system'] for r in data]); mass=np.array([r['mass'] for r in data]); y=np.log10(mass)
    predictions={k:np.zeros(len(y)) for k in ['V','K','V_color','K_color']}; selections=[]
    for held in sorted(set(names)):
        tr=names!=held
        assert not set(names[tr]) & set(names[~tr])
        for kind in predictions:
            p,choice=optical_choice(data,y,names,tr,kind);predictions[kind][~tr]=10**p[~tr]
            selections.append(dict(held=str(held),model=kind,**choice))
    held=sorted(set(names))[0];tr=names!=held
    p0,_=optical_choice(data,y,names,tr,'K_color');mut=y.copy();mut[~tr]+=1
    p1,_=optical_choice(data,mut,names,tr,'K_color')
    assert np.array_equal(p0,p1)
    result=dict(stars=len(data),binaries=len(set(names)),mass_range=[float(mass.min()),float(mass.max())],
                validation='Leave entire binary out; nested training-only choice',
                metrics={k:metrics(mass,p,names) for k,p in predictions.items()},
                comparisons={a+'_to_'+b:comparison(mass,predictions[a],predictions[b],names)
                             for a,b in [('V','K'),('V','V_color'),('K','K_color')]},
                held_target_invariance=True,selections=selections,
                limitations=['Only two measured bands and 14 nearby HST binaries; not all stellar populations.',
                             'Common parallax affects dynamical mass and absolute magnitudes; grouped folds do not remove shared calibration systematics.',
                             'Reported prediction errors include observational uncertainty and do not measure an intrinsic universal scatter.'])
    rows=[dict(**r,**{k+'_prediction':float(p[i]) for k,p in predictions.items()}) for i,r in enumerate(data)]
    csvsave(OUT/'binary-optical-predictions.csv',rows)
    save(OUT/'binary-optical.json',result)
    return result


def pair_arrays(rows):
    return dict(k1=np.array([r['MK1'] for r in rows]), k2=np.array([r['MK2'] for r in rows]),
                y=np.log10([r['mass'] for r in rows]),z=np.array([r['feh'] for r in rows]),
                names=np.array([r['system'] for r in rows]))


def poly_design(k):
    x=(np.asarray(k)-7.5)/2.5
    return np.stack([x**i for i in range(6)],axis=-1)


def pair_predict(p,d,metal=False):
    a=poly_design(d['k1'])@p[:6]; b=poly_design(d['k2'])@p[:6]
    return np.logaddexp(LN10*a,LN10*b)/LN10 + (p[6]*d['z'] if metal else 0)


def pair_fit(d,tr,metal=False):
    # Initial line is calculated solely from the training systems.
    xm=((d['k1'][tr]+d['k2'][tr])/2-7.5)/2.5
    slope,intercept=np.polyfit(xm,d['y'][tr]-np.log10(2),1)
    start=np.zeros(7 if metal else 6);start[:2]=[intercept,slope]
    xgrid=(np.linspace(3.5,11.5,801)-7.5)/2.5
    derivative=np.column_stack([np.zeros(len(xgrid))]+[i*xgrid**(i-1) for i in range(1,6)])
    constraint=np.column_stack([derivative,np.zeros(len(xgrid))]) if metal else derivative
    A=poly_design(d['k1'][tr]);B=poly_design(d['k2'][tr])
    def objective(p):
        r=pair_predict(p,d,metal)[tr]-d['y'][tr]
        mix=1/(1+np.exp(np.clip(LN10*((B-A)@p[:6]),-700,700)))
        jac=mix[:,None]*A+(1-mix[:,None])*B
        if metal:jac=np.column_stack([jac,d['z'][tr]])
        reg=np.zeros_like(p);reg[2:6]=1e-10*p[2:6]
        return .5*np.mean(r*r)+.5e-10*np.sum(p[2:6]**2),jac.T@r/len(r)+reg
    fit=minimize(objective,start,jac=True,method='SLSQP',constraints=[LinearConstraint(constraint,-np.inf,0)],
                 options={'ftol':1e-13,'maxiter':1500})
    assert fit.success and np.max(derivative@fit.x[:6])<1e-7, fit.message
    return fit.x


def pair_cv(d,metal=False):
    split=folds(d['names'],'mass-anchor-A');pred=np.zeros(len(split));coeff=[]
    for f in range(5):
        tr=split!=f;p=pair_fit(d,tr,metal);pred[~tr]=pair_predict(p,d,metal)[~tr]
        coeff.append(p.tolist())
    return 10**pred,coeff


def binary_anchor(data):
    rows=data['mann']; d=pair_arrays(rows); mass=10**d['y']; names=d['names']
    primary,c0=pair_cv(d); metal,c1=pair_cv(d,True)
    p=pair_fit(d,np.ones(len(rows),bool))
    ext=data['external_EB']; k=np.array([r['MK'] for r in ext]); target=np.array([r['mass'] for r in ext]); groups=np.array([r['system'] for r in ext])
    assert not set(names)&set(groups)
    prediction=10**(poly_design(k)@p)
    # Published relation is a benchmark reproduction, not independent training evidence.
    published=np.array([-.642,-.208,-8.43e-4,7.87e-3,1.42e-4,-2.13e-4])
    published_prediction=10**(np.column_stack([(k-7.5)**i for i in range(6)])@published)
    good=np.array([r['e_mass']/r['mass']<=.05 for r in rows]); gd={a:b[good] for a,b in d.items()}
    gp,gc=pair_cv(gd);gm,gmc=pair_cv(gd,True)
    # Coherent one-sigma distance calibration stresses, not confidence intervals.
    distance=[]
    for sign in [-1,1]:
        perturbed=[dict(r) for r in rows]
        for r in perturbed:
            old=r['plx'];new=old+sign*r['e_plx']
            r['mass']*=(old/new)**3
            r['MK1']+=5*np.log10(new/old);r['MK2']+=5*np.log10(new/old)
        pd=pair_arrays(perturbed);pc=pair_fit(pd,np.ones(len(rows),bool));pe=10**(poly_design(k)@pc)
        distance.append(dict(sign=sign,external_metrics=metrics(target,pe,groups),
                             median_prediction_change_percent=float(100*np.median(pe/prediction-1))))
    tr=folds(names,'mass-anchor-A')!=0;before=pair_fit(d,tr);mut={a:b.copy() for a,b in d.items()};mut['y'][~tr]+=1
    after=pair_fit(mut,tr);assert np.array_equal(before,after)
    synth={a:b.copy() for a,b in d.items()};true=np.array([-.65,-.5,0,0,0,0]);synth['y']=pair_predict(true,d)+np.log10(1.5)
    sp,_=pair_cv(synth);injection_error=float(np.max(abs(sp/(10**synth['y'])-1)))
    assert injection_error<1e-5
    orbit_errors=[abs(r['orbital_mass_replay']/r['mass']-1) for r in rows if 'orbital_mass_replay' in r]
    result=dict(calibration_systems=len(rows),external_EB_stars=len(ext),external_EB_systems=len(set(groups)),
                calibration_metrics=metrics(mass,primary,names),metallicity_metrics=metrics(mass,metal,names),
                metallicity_comparison=comparison(mass,primary,metal,names),
                metallicity_log_mass_coefficients=[c[-1] for c in c1],
                external_metrics=metrics(target,prediction,groups),published_external_reproduction=metrics(target,published_prediction,groups),
                coefficients=p.tolist(),definition='log10(M/Msun)=sum(c_i*((M_Ks-7.5)/2.5)^i); calibration uses sums of component predictions',
                coefficients_per_fold=c0,
                quality_sensitivity=dict(systems=int(sum(good)),baseline=metrics(10**gd['y'],gp,gd['names']),
                                         metallicity=metrics(10**gd['y'],gm,gd['names']),comparison=comparison(10**gd['y'],gp,gm,gd['names'])),
                parallax_stresses=distance,
                orbital_replay=dict(systems=len(orbit_errors),max_fractional_difference=float(max(orbit_errors)),median_fractional_difference=float(np.median(orbit_errors)),
                                    note='Rounded marginal central orbital values and mass may differ; this is an audit, not a refit of the astrometry'),
                controls=dict(held_target_invariance=True,known_50percent_mass_scale_recovered_max_error=injection_error),
                limitations=['Our fit uses central measurements and equal system weights, not the original joint likelihood/posterior.',
                             'External EB masses are independent of this fit; their separated Ks light uses spectral-template conversion of eclipse contrasts.',
                             'Original authors already analyzed these validation stars; this is a reproducible reanalysis, not new pristine confirmation.',
                             'Nearby low-mass stars do not determine galaxy-wide faint-star counts, remnant fractions, or integrated 3.6-micron mass-to-light ratios.'])
    csvsave(OUT/'binary-anchor-predictions.csv',[dict(system=r['system'],observed_mass=r['mass'],predicted_mass=float(primary[i]),metallicity_prediction=float(metal[i])) for i,r in enumerate(rows)])
    csvsave(OUT/'binary-external-predictions.csv',[dict(**r,predicted_mass=float(prediction[i]),published_prediction=float(published_prediction[i])) for i,r in enumerate(ext)])
    save(OUT/'binary-anchor.json',result)
    return result


def force(d,alpha,gas_scale=1.,bulge_alpha=None):
    if bulge_alpha is None:bulge_alpha=alpha
    return gas_scale*d['gas']*abs(d['gas']) + .5*alpha*d['disk']**2 + .7*bulge_alpha*d['bul']**2


def logspeed(d,alpha,gas_scale=1.,bulge_alpha=None):
    # A nonpositive force is retained as an invalid circular prediction, not dropped.
    return .5*np.log10(np.maximum(force(d,alpha,gas_scale,bulge_alpha),1e-12))


def fit_scale(d,use,gas_scale=1.):
    w=sparc.weights(d['name'][use]);target=np.log10(d['v'][use])
    fn=lambda t:np.average((logspeed(d,np.exp(t),gas_scale)[use]-target)**2,weights=w)
    result=minimize_scalar(fn,bounds=np.log([.05,40.]),method='bounded',options={'xatol':1e-10})
    assert result.success
    return float(np.exp(result.x))


def galaxy_audit(d,gas_scale=1.,export=False):
    names=d['name']; v=d['v'];n=len(v); split=folds(names,'mass-only-galaxies-A')
    pred_global=np.zeros(n);scales=[]
    for f in range(5):
        tr=split!=f;a=fit_scale(d,tr,gas_scale);pred_global[~tr]=10**logspeed(d,a,gas_scale)[~tr]
        scales.append(a)
    fixed=10**logspeed(d,1.,gas_scale);pergal=np.zeros(n);innerpred=np.zeros(n);two=np.zeros(n);outer=np.zeros(n,bool)
    stellar=.5*d['disk']**2+.7*d['bul']**2;gas=gas_scale*d['gas']*abs(d['gas'])
    req=(v*v-gas)/stellar;records=[]
    for name in sorted(set(names)):
        use=names==name;ids=np.flatnonzero(use);ids=ids[np.argsort(d['r'][ids])]
        cut=(len(ids)+1)//2;train=np.zeros(n,bool);train[ids[:cut]]=True;outer[ids[cut:]]=True
        a=fit_scale(d,use,gas_scale);ai=fit_scale(d,train,gas_scale)
        pergal[use]=10**logspeed(d,a,gas_scale)[use];innerpred[use]=10**logspeed(d,ai,gas_scale)[use]
        if np.any(d['bul'][use]>0):
            fit=least_squares(lambda p:logspeed(d,np.exp(p[0]),gas_scale,np.exp(p[1]))[use]-np.log10(v[use]),
                              np.log([a,a]),bounds=(np.log([.05,.05]),np.log([40.,40.])),max_nfev=1500)
            assert fit.success;ad,ab=np.exp(fit.x)
        else:ad=ab=a
        two[use]=10**logspeed(d,ad,gas_scale,ab)[use]
        third=max(1,int(np.ceil(len(ids)/3)));lo=req[ids[:third]];hi=req[ids[-third:]]
        il=float(np.median(lo));oh=float(np.median(hi));err=abs(pergal[use]/v[use]-1)
        records.append(dict(name=str(name),positions=len(ids),alpha_all=a,alpha_inner=ai,
                            alpha_disk_two_component=float(ad),alpha_bulge_two_component=float(ab),
                            required_alpha_inner_third=il,required_alpha_outer_third=oh,
                            outer_inner_required_ratio=oh/il if il>0 and oh>0 else None,
                            fit_all_median_absolute_speed_error_percent=float(100*np.median(err)),
                            fit_all_p90_absolute_speed_error_percent=float(100*np.quantile(err,.9)),
                            inner_fit_outer_median_prediction_ratio=float(np.median(innerpred[ids[cut:]]/v[ids[cut:]])),
                            global_log_mse=float(np.mean(np.log10(pred_global[use]/v[use])**2))))
    bins=[]
    for low,high in [(-20,-11),(-11,-10),(-10,-9),(-9,0)]:
        use=(d['x']>=low)&(d['x']<high)&(req>0)
        vals=[np.median(req[use&(names==s)]) for s in sorted(set(names[use]))]
        if vals: bins.append(dict(log_gbar_low=low,log_gbar_high=high,galaxies=len(vals),positions=int(sum(use)),median_required_stellar_multiplier=float(np.median(vals))))
    reqratios=np.array([r['outer_inner_required_ratio'] for r in records if r['outer_inner_required_ratio'] is not None])
    result=dict(galaxies=len(records),positions=n,dark_matter_term=0,gas_scale=gas_scale,
                global_alpha_by_fold=scales,
                fixed=metrics(v,fixed,names),global_held_galaxy=metrics(v,pred_global,names),
                all_radii_per_galaxy_fit_diagnostic=metrics(v,pergal,names),
                all_radii_two_component_fit_diagnostic=metrics(v,two,names),
                outer_fixed=metrics(v[outer],fixed[outer],names[outer]),
                outer_global=metrics(v[outer],pred_global[outer],names[outer]),
                outer_after_inner_calibration=metrics(v[outer],innerpred[outer],names[outer]),
                inner_to_outer_ratio=dict(galaxies=len(reqratios),median=float(np.median(reqratios)),above_two=int(sum(reqratios>2)),above_one=int(sum(reqratios>1))),
                median_best_fit_stellar_multiplier=float(np.median([r['alpha_all'] for r in records])),
                per_galaxy_fit_at_upper_bound=int(sum(r['alpha_all']>39.9 for r in records)),
                per_galaxy_90percent_radii_within10percent_speed=int(sum(r['fit_all_p90_absolute_speed_error_percent']<=10 for r in records)),
                required_alpha_negative_or_zero_positions=int(sum(req<=0)),
                invalid_fixed_force=int(sum(force(d,1,gas_scale)<=0)),
                invalid_inner_force_outer=int(sum((innerpred<=1e-6)&outer)),
                required_mass_by_acceleration=bins)
    if export:
        csvsave(OUT/'galaxy-mass-requirements.csv',records)
        csvsave(OUT/'galaxy-radius-predictions.csv',[dict(name=str(names[i]),row=int(d['row'][i]),radius_kpc=float(d['r'][i]),
                 observed_speed=float(v[i]),e_speed=float(d['ev'][i]),log_gbar=float(d['x'][i]),
                 required_stellar_multiplier=float(req[i]),gas_force_share=float(gas[i]/(gas[i]+stellar[i])),
                 fixed_prediction=float(fixed[i]),global_prediction=float(pred_global[i]),fit_all_prediction=float(pergal[i]),
                 two_component_fit_all_prediction=float(two[i]),inner_calibrated_prediction=float(innerpred[i]),outer_test=bool(outer[i])) for i in range(n)])
    return result


def sparc_audit():
    rows,sources=sparc.read_inputs();d,excluded=sparc.prepare(rows)
    primary=galaxy_audit(d,export=True);sensitivity={}
    for label,kwargs in [('Q1',dict(qmax=1)),('inclination_40_75',dict(inclination=(40,75)))]:
        subset,_=sparc.prepare(rows,**kwargs);sensitivity[label]=galaxy_audit(subset)
    for scale in [.5,2.]:sensitivity['gas_scale_'+str(scale)]=galaxy_audit(d,gas_scale=scale)
    mock={k:v.copy() for k,v in d.items()};mock['v']=np.sqrt(force(d,3.));recovered=fit_scale(mock,np.ones(len(d['v']),bool))
    assert abs(recovered-3)<1e-5
    tr=folds(d['name'],'mass-only-galaxies-A')!=0;a=fit_scale(d,tr)
    mut={k:v.copy() for k,v in d.items()};mut['v'][~tr]*=2;b=fit_scale(mut,tr)
    assert a==b
    replay=(d['v']**2-d['gas']*abs(d['gas']))/(.5*d['disk']**2+.7*d['bul']**2)
    replay_error=float(np.max(abs(force(d,replay)-d['v']**2)))
    assert replay_error<1e-7
    result=dict(primary=primary,sensitivities=sensitivity,selection_exclusions=excluded,
                sources=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sources],
                controls=dict(known_threefold_stellar_mass_scale_recovered=recovered,held_target_invariance=True,analytic_replay_max_v2_error=replay_error),
                limitations=['No RAR or halo is included; predictions use Newtonian sums of existing gas and stellar force templates.',
                             'Per-radius required scaling is not a reconstructed spatial stellar-mass profile: gravity from disks is nonlocal.',
                             'Distance, inclination, warps, pressure support, noncircular motion, and radial covariance are not refitted by this 1D audit.',
                             'Gas templates primarily trace HI plus helium. Missing molecular/hot gas and stellar population gradients are not comprehensively calibrated.',
                             'Free per-galaxy mass fitting is a requirement diagnostic; these fitted masses are not independent evidence that the stars are heavier.',
                             'Individual-star Ks mass calibration cannot be numerically transferred into an integrated 3.6-micron M/L without a population model and independent population constraints.'])
    save(OUT/'galaxy-mass-only.json',result)
    return result


def manga_audit():
    a,sa=population.rows(12);b,sb=population.rows(13)
    assert not {r['mangaid'] for r in a}&{r['mangaid'] for r in b}
    base=['log_half_light_radius','axis_ratio','sersic_index','redshift','log_surface_brightness','log_snr']
    additions={'color':['g_minus_r_color'],
               'color_and_spectrum':['g_minus_r_color','dn4000','d4000','hdelta_a','hgamma_a','hbeta','signed_log_halpha']}
    assert not set(base+sum(additions.values(),[])) & {'log_stellar_mass','log_surface_density','log_specific_sfr','mass_size_crossing_proxy'}
    result=dict(sources=[sa,sb],features=dict(base=base,additions=additions),cross_validation={},transport={},mass_derived_predictors_used=False,
                target='Stellar velocity dispersion, not individual stellar mass or circular speed',
                pristine_confirmation=False,limitations=['Both development samples were historically examined.',
                'Catalog g-r uses absolute photometric magnitudes and corrections; these are not all raw photometric bands.',
                'Original sample admission used catalog mass validity. This test removes mass from model inputs, not from historical sample selection.',
                'Measured spectral summaries and velocity dispersion share spectra and pipeline fitting. Predictive gains are not independent mass determinations.'])
    for kind in ['ridge','trees']:
        print('MaNGA light-only',kind,flush=True)
        score,pred=population.evaluate(a,kind,base=base,groups=additions);result['cross_validation'][kind]=score
        csvsave(OUT/('manga-'+kind+'-predictions.csv'),pred)
        y=np.log10(population.matrix(a,['stellar_sigma_1re_km_s'])[:,0]);target=np.log10(population.matrix(b,['stellar_sigma_1re_km_s'])[:,0]);names=np.array([r['plateifu'] for r in a]);predictions={};selection={}
        for tag,keys in {'baseline':base,**{k:base+v for k,v in additions.items()}}.items():
            X=population.matrix(a,keys);Z=population.matrix(b,keys)
            choice=population.choose(X,y,names,kind,'light-only-transport')
            model=population.estimator(kind,choice).fit(X,y);predictions[tag]=model.predict(Z);selection[tag]=choice
        loss=(predictions['baseline']-target)**2;summary=[]
        for tag in additions:
            err=(predictions[tag]-target)**2
            summary.append(dict(group=tag,train_galaxies=len(a),test_galaxies=len(b),mse_gain_percent=float(100*(1-err.mean()/loss.mean())),
                                baseline_rmse_dex=float(np.sqrt(loss.mean())),full_rmse_dex=float(np.sqrt(err.mean())),paired_mse_gain_bootstrap95=boot(loss-err)))
        result['transport'][kind]=dict(summary=summary,selected_on_training=selection)
        csvsave(OUT/('manga-'+kind+'-transport.csv'),[dict(plateifu=r['plateifu'],y=float(target[i]),**{k:float(p[i]) for k,p in predictions.items()}) for i,r in enumerate(b)])
    save(OUT/'manga-light-only.json',result)
    return result


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=False)
    # Save the exact prescoring protocol; failed runs preserve their directory.
    protocol=PROTOCOL.read_bytes();(OUT/'protocol.json').write_bytes(protocol)
    data=json.loads((PRIVATE/'parsed-binaries.json').read_text(encoding='utf-8'))
    manifest=json.loads((PRIVATE/'source-manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((PRIVATE/'parsed-binaries.json').read_bytes()).hexdigest()==manifest['data_sha256']
    save(OUT/'source-manifest.json',manifest)
    print('binary visible / infrared',flush=True);optical=optical_audit(data['benedict'])
    print('independent binary masses',flush=True);anchor=binary_anchor(data)
    print('Newtonian stellar-mass requirements',flush=True);galaxies=sparc_audit()
    print('light-only galaxy predictions',flush=True);manga=manga_audit()
    summary=dict(status='COMPLETED_SCOPED_ZERO_DARK_MATTER_STELLAR_MASS_AUDIT',dark_matter_contribution=0,
                 protocol_sha256=hashlib.sha256(protocol).hexdigest(),elapsed_seconds=time.time()-started,
                 binary_optical={k:v for k,v in optical.items() if k!='selections'},
                 binary_anchor={k:v for k,v in anchor.items() if k!='coefficients_per_fold'},
                 galaxies=galaxies['primary'],manga={k:v for k,v in manga.items() if k not in ['sources','features']},
                 all_baryonic_mass_alternatives_resolved=False,independent_galaxy_population_mass_calibration_completed=False,
                 admitted_gravity_laws=0)
    save(OUT/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],elapsed_seconds=summary['elapsed_seconds'],
                         binary_external=anchor['external_metrics'],galaxy_global=galaxies['primary']['global_held_galaxy'])))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='A new, empty result directory; relative paths are resolved from the repository root')
    args=parser.parse_args();OUT=args.output if args.output.is_absolute() else ROOT/args.output
    with threadpool_limits(limits=1): main()
