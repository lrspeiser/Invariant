"""Replay independent source checks and targeted robustness checks for the mass audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

import run_gravity_stellar_mass_audit as audit
from acquire_gravity_mass_audit import parse_sources, save, PRIVATE, ROOT
import run_gravity_population_patterns as pop

OUT=audit.OUT


def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))


def transfer(a,b,kind,base,groups):
    y=np.log10(pop.matrix(a,['stellar_sigma_1re_km_s'])[:,0]); target=np.log10(pop.matrix(b,['stellar_sigma_1re_km_s'])[:,0])
    names=np.array([r['plateifu'] for r in a]);pred={}
    for key,keys in {'baseline':base,**{k:base+v for k,v in groups.items()}}.items():
        X=pop.matrix(a,keys);Z=pop.matrix(b,keys);choice=pop.choose(X,y,names,kind,'light-distance-transport')
        pred[key]=pop.estimator(kind,choice).fit(X,y).predict(Z)
    loss=(pred['baseline']-target)**2
    return {k:dict(mse_gain_percent=float(100*(1-np.mean((p-target)**2)/loss.mean())),
                   paired_mse_gain_bootstrap95=audit.boot(loss-(p-target)**2)) for k,p in pred.items() if k!='baseline'}


def main():
    # These rules are fixed before any scores in this addendum are computed.
    plan=dict(status='Post-primary robustness checks, not new confirmation',
              optical='Score the already-frozen predictions on stars whose V and K both lie within their training binaries ranges. No model is refitted or selected using this subset.',
              metallicity='Source flags d mark extrapolated L-dwarf metallicities. Compare both models on the same 60 systems after excluding these two source-invalid metallicities.',
              manga='Replace redshift by log10 redshift so a linear predictor can represent distance scaling; repeat color comparisons with both previously declared model families.',
              verification='Reparse original binary source tables, verify source hashes and shared-system identities, test held-target invariance, replay galaxy predictions from original force templates, and replay MaNGA targets from their source responses.')
    path=OUT/'robustness-protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==plan
    else:save(path,plan)
    summary=read('summary.json');assert summary['dark_matter_contribution']==0
    assert summary['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    manifest=read('source-manifest.json')
    for r in manifest['sources']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    data=parse_sources();stored=json.loads((PRIVATE/'parsed-binaries.json').read_text(encoding='utf-8'));assert data==stored
    import csv
    optical=list(csv.DictReader((OUT/'binary-optical-predictions.csv').open(encoding='utf-8')))
    names=np.array([r['system'] for r in optical]);v=np.array([float(r['MV']) for r in optical]);k=np.array([float(r['MK']) for r in optical])
    support=np.zeros(len(names),bool)
    for i in range(len(names)):
        train=names!=names[i]
        support[i]=v[train].min()<=v[i]<=v[train].max() and k[train].min()<=k[i]<=k[train].max()
    mass=np.array([float(r['mass']) for r in optical]);pred={p:np.array([float(r[p+'_prediction']) for r in optical]) for p in ['V','K','V_color','K_color']}
    domain=dict(stars=int(sum(support)),excluded_names=[r['name'] for i,r in enumerate(optical) if not support[i]],
                metrics={p:audit.metrics(mass[support],x[support],names[support]) for p,x in pred.items()},
                comparisons={a+'_to_'+b:audit.comparison(mass[support],pred[a][support],pred[b][support],names[support]) for a,b in [('V','K'),('V','V_color'),('K','K_color')]})
    valid=[r for r in data['mann'] if r['feh_flag']!='d'];d=audit.pair_arrays(valid)
    p,_=audit.pair_cv(d);q,coef=audit.pair_cv(d,True)
    metallicity=dict(systems=len(valid),baseline=audit.metrics(10**d['y'],p,d['names']),
                     with_metallicity=audit.metrics(10**d['y'],q,d['names']),comparison=audit.comparison(10**d['y'],p,q,d['names']),
                     coefficients=[c[-1] for c in coef],excluded=[r['system'] for r in data['mann'] if r['feh_flag']=='d'])
    # Flux addition must recover the system magnitude without using the mass labels.
    flux_errors=[]
    for r in data['mann']:
        m1=r['MK1']-5*np.log10(r['plx'])+10;m2=r['MK2']-5*np.log10(r['plx'])+10
        flux_errors.append(abs(-2.5*np.log10(10**(-.4*m1)+10**(-.4*m2))-r['Ks']))
    assert max(flux_errors)<1e-10
    # Independent scalar replay of every exported Newtonian curve point.
    rows,_=audit.sparc.read_inputs();d,_=audit.sparc.prepare(rows)
    original={(str(d['name'][i]),int(d['row'][i])):i for i in range(len(d['v']))}
    exported=list(csv.DictReader((OUT/'galaxy-radius-predictions.csv').open(encoding='utf-8')))
    scales={r['name']:r for r in csv.DictReader((OUT/'galaxy-mass-requirements.csv').open(encoding='utf-8'))}
    sf=audit.folds(d['name'],'mass-only-galaxies-A');global_alpha=read('galaxy-mass-only.json')['primary']['global_alpha_by_fold']
    for r in exported:
        i=original[(r['name'],int(r['row']))];s=.5*d['disk'][i]**2+.7*d['bul'][i]**2;gas=d['gas'][i]*abs(d['gas'][i])
        for column,alpha in [('fixed_prediction',1.),('global_prediction',global_alpha[sf[i]]),
                             ('fit_all_prediction',float(scales[r['name']]['alpha_all'])),
                             ('inner_calibrated_prediction',float(scales[r['name']]['alpha_inner']))]:
            assert abs(float(r[column])-np.sqrt(gas+alpha*s))<1e-8
        assert abs(gas+float(r['required_stellar_multiplier'])*s-d['v'][i]**2)<1e-7
    a,sa=pop.rows(12);b,sb=pop.rows(13)
    assert len({r['mangaid'] for r in a})==len(a) and len({r['mangaid'] for r in b})==len(b)
    assert not {r['mangaid'] for r in a}&{r['mangaid'] for r in b}
    for sample,item in [(a,'item-12-manga-dynamical-age'),(b,'item-13-manga-relaxation-mergers')]:
        raw=json.loads((ROOT/('runs/gravity/roadmap/'+item+'-v1-source/response-source.json')).read_text(encoding='utf-8'))
        mapping={r['plateifu']:float(r['stellar_sigma_1re']) for r in raw['records']}
        assert all(abs(float(r['stellar_sigma_1re_km_s'])-mapping[r['plateifu']])<1e-7 for r in sample)
    spec=read('manga-light-only.json')['features'];features=spec['base']+spec['additions']['color_and_spectrum']
    X=pop.matrix(a,features);y=np.log10(pop.matrix(a,['stellar_sigma_1re_km_s'])[:,0]);tr=np.array([r['outer_fold']!=0 for r in a]);names=np.array([r['plateifu'] for r in a])
    checks={}
    for kind in ['ridge','trees']:
        choice=pop.choose(X[tr],y[tr],names[tr],kind,'held-target-check');model=pop.estimator(kind,choice).fit(X[tr],y[tr]);before=model.predict(X[~tr])
        mutant=y.copy();mutant[~tr]+=2;choice2=pop.choose(X[tr],mutant[tr],names[tr],kind,'held-target-check')
        after=pop.estimator(kind,choice2).fit(X[tr],mutant[tr]).predict(X[~tr])
        delta=float(np.max(abs(before-after)))
        # Threaded tree prediction can sum estimator outputs in a different
        # order. This tolerance is many orders below measurement precision.
        assert choice==choice2 and delta<1e-12, (kind,choice,choice2,delta)
        checks[kind]=dict(passed=True,max_log10_prediction_difference=delta,absolute_tolerance=1e-12)
    # Sensitivity to a physically useful transform of the distance descriptor.
    for r in a+b:r['log_redshift']=float(np.log10(float(r['redshift'])))
    base=['log_redshift' if k=='redshift' else k for k in spec['base']]
    manga={}
    for kind in ['ridge','trees']:
        print('distance descriptor sensitivity',kind,flush=True)
        cv,_=pop.evaluate(a,kind,base=base,groups=spec['additions'])
        manga[kind]=dict(cross_validation=cv['summary'],transport=transfer(a,b,kind,base,spec['additions']))
    result=dict(status='PASS_TARGETED_SOURCE_REPLAY_AND_ROBUSTNESS_CHECKS',optical_interpolation=domain,
                valid_metallicity=metallicity,manga_log_distance=manga,
                source_hashes_verified=len(manifest['sources']),binary_source_reparse_exact=True,
                flux_addition_max_magnitude_error=max(flux_errors),newtonian_radius_predictions_replayed=len(exported),
                manga_response_values_replayed=len(a)+len(b),unique_manga_galaxies_verified=True,manga_held_target_invariance=checks,
                no_independent_galaxy_population_mass_calibration=True)
    save(OUT/'verification.json',result)
    print(json.dumps(dict(status=result['status'],optical_support=domain['stars'],valid_metallicity=metallicity['comparison'],manga_log_distance=manga)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='Result directory to verify')
    args=parser.parse_args();OUT=args.output if args.output.is_absolute() else ROOT/args.output
    audit.OUT=OUT
    with threadpool_limits(limits=1):main()
