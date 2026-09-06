"""Independent algebraic evaluator; never imports the repair predictor."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_clock_relay import load_sources

P=Path(__file__).resolve().parent
RUN=P.parent/'run001'


def readcsv(name):
    with (RUN/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def independent(s,c):
    # Work in SI, independently from the original kpc/km-per-second evaluator.
    kpc=3.085677581491367e19
    gconst=4.30091727003628e-6*kpc*1e6
    a0=1.2e-10
    radius=s['r']*kpc;diskradius=s['rd']*kpc;mf=c['mf']
    vb2=1e6*(s['gas']*np.abs(s['gas'])+.5*mf*s['disk']**2+.7*mf*s['bulge']**2)
    mass=1e9*(.5*mf*s['luminosity']+1.33*s['hi']);mu=gconst*mass
    family=c['family']
    if family.startswith('mond'):
        gb=vb2/radius
        effective_a0=a0*c.get('a0_factor',1.)
        # nu(y) formulation rather than quadratic expression in original evaluator.
        acceleration=gb*(.5+np.sqrt(.25+effective_a0/gb))
    else:
        if family=='clock_potential':
            psi=c['clock_factor']*np.sqrt(mu*a0)
            u=mu/(psi*(radius+diskradius))
            extra=c['beta']*psi/(radius+diskradius)*(u/(1+u))
        else:
            length=c['length_factor']*np.sqrt(mu/a0)
            if family=='kernel_point':
                u=np.minimum(radius/length,c['cutoff'])
                # Integrate t/(1+t)^2 using high-order Gauss-Legendre quadrature.
                nodes,weights=np.polynomial.legendre.leggauss(96)
                t=.5*u[:,None]*(nodes+1)
                m=.5*u*np.sum(weights*t/(1+t)**2,axis=1)
                extra=c['eta']*mu*m/radius**2
            else:
                core_weight={'finite_p2':0.,'finite_p3':1.}.get(family,c.get('q',0.))
                extra=c['eta']*mu/(radius+length)**2*((1-core_weight)+core_weight*radius/(radius+length))
        acceleration=vb2/radius+extra
    return np.log10(np.sqrt(radius*acceleration)/1000)


def main():
    config=json.loads((ROOT/'configs/mond_atlas_clock_relay_v1.json').read_text(encoding='utf-8'))
    # Same loader means response ingestion is shared; force implementation is independent.
    s,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
    candidates=json.loads((RUN/'candidates.json').read_text(encoding='utf-8'))
    pred=np.array([independent(s,c) for c in candidates]);gi=np.array([names.index(n) for n in ids])
    loss=np.array([np.mean((pred[:,gi==i]-y[None,gi==i])**2,axis=1) for i in range(len(names))]).T
    rowlookup={(n,int(r)):j for j,(n,r) in enumerate(zip(ids,radial_ids))}
    max_pred=0.;max_residual=0.
    held=readcsv('all-held-radial-residuals.csv')
    for row in held:
        j=rowlookup[(row['galaxy'],int(row['radial_index']))];ix=int(row['candidate_index']);p=pred[ix,j]
        max_pred=max(max_pred,abs(p-float(row['log10_predicted_speed'])))
        max_residual=max(max_residual,abs(p-y[j]-float(row['log10_predicted_over_observed'])))
    folds={seed:{r['galaxy']:int(r['fold']) for r in readcsv(f'folds-{seed}.csv')} for seed in config['fold_seeds']}
    choices=json.loads((RUN/'selections.json').read_text(encoding='utf-8'))
    choice_lookup={};choice_errors=[]
    for row in choices:
        seed=row['seed'];f=row['family'];fold=row['fold']
        train=np.array([folds[seed][n]!=fold for n in names])
        options=[i for i,c in enumerate(candidates) if c['family']==f]
        selected=options[int(np.argmin(loss[options][:,train].mean(axis=1)))]
        if selected!=row['candidate_index']:choice_errors.append(row)
        choice_lookup[(seed,f,fold)]=selected
    max_train=0.
    training=readcsv('all-training-losses.csv')
    for row in training:
        seed=int(row['seed']);fold=int(row['fold']);ix=int(row['candidate_index'])
        train=np.array([folds[seed][n]!=fold for n in names])
        max_train=max(max_train,abs(float(loss[ix,train].mean())-float(row['training_mse'])))
    max_gal=0.
    gal=readcsv('galaxy-scores.csv')
    for row in gal:
        n=row['galaxy'];seed=int(row['seed']);f=row['family'];i=names.index(n)
        ix=choice_lookup[(seed,f,folds[seed][n])]
        max_gal=max(max_gal,abs(float(loss[ix,i])-float(row['mse_logspeed'])))
    receipt=dict(independent_implementation='SI-unit evaluator; 96-point quadrature for kernel; shared source loader explicitly retained',galaxies=len(names),radii=len(y),candidates=len(candidates),held_rows=len(held),training_rows=len(training),galaxy_score_rows=len(gal),choices=len(choices),choice_mismatches=len(choice_errors),max_abs_logspeed=max_pred,max_abs_logspeed_residual=max_residual,max_abs_training_mse=max_train,max_abs_galaxy_mse=max_gal,threshold=1e-10,reserved_member_bodies_opened=0)
    receipt['passed']=max(max_pred,max_residual,max_train,max_gal)<1e-10 and not choice_errors
    receipt['files']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*RUN.glob('*')] if p.is_file()}
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
    if not receipt['passed']:raise RuntimeError('Independent replay failed')


if __name__=='__main__':main()
