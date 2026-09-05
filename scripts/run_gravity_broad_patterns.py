"""Nested, whole-galaxy screen of observable gravity-pattern families."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
import run_gravity_matched_concentration as prior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/gravity-first-principles/broad-patterns-001'
CONFIG=ROOT/'configs/gravity_broad_pattern_search_v1.json'
ALIASES={'predicted_period','predicted_speed','predicted_angular_momentum','spherical_equivalent_density'}
DIAGNOSTICS={'distance','distance_uncertainty','inclination','inclination_uncertainty','quality','radial_sampling','radial_coverage'}

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),newline='\n')

def csvsave(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def log(a):
    a=np.asarray(a,dtype=float);return np.log10(np.where(a>0,a,np.nan))

def make_data(ml=.5,qmax=2,inclination=(30,80)):
    rows,sources=prior.read_inputs();d,_=prior.prepare(rows,ml,qmax,inclination)
    meta={}
    for line in (ROOT/sources[-1]).read_text().splitlines():
        f=line.split()
        if f and f[0] in set(d['name']):
            meta[f[0]]=dict(zip(['T','D','eD','method','inc','einc','L','eL','Re','Ie','Rd','I0','HI','RHI','Vflat','eVflat','Q'],map(float,f[1:18])))
    features={}; descriptions={}
    def add(name,values,description):
        features[name]=np.asarray(values,float);descriptions[name]=description
    m=lambda field:np.array([meta[n][field] for n in d['name']])
    r=d['r'];b2=d['b2'];sigma=ml*d['sd']+1.4*ml*d['sb']; gas2=d['gas']*abs(d['gas'])
    add('radius',log(r),'Physical radius; kpc')
    add('radius_over_Re',log(r/m('Re')),'Radius relative to photometric effective radius')
    add('radius_over_Rd',log(r/m('Rd')),'Radius relative to stellar disk scale length')
    add('stellar_surface_density',log(sigma),'Inclination-corrected stellar surface density; not volume density')
    add('effective_brightness',log(m('Ie')),'Global effective stellar surface brightness')
    add('central_brightness',log(m('I0')),'Global central stellar disk surface brightness')
    add('stellar_luminosity',log(m('L')),'Total 3.6 micron luminosity, not independent stellar mass')
    add('atomic_mass',log(m('HI')),'Total catalog atomic hydrogen mass')
    add('global_atomic_fraction',1.33*m('HI')/(1.33*m('HI')+ml*m('L')),'Atomic/stellar mass-mixture proxy using single global stellar M/L; no complete H2')
    add('local_atomic_force_share',gas2/(abs(gas2)+ml*d['disk']**2+1.4*ml*d['bul']**2),'Bounded signed gas share of component radial force magnitude; not local density')
    add('bulge_force_share',1.4*ml*d['bul']**2/(abs(gas2)+ml*d['disk']**2+1.4*ml*d['bul']**2),'Bulge share of component radial force magnitude')
    add('bulge_light_share',1.4*ml*d['sb']/sigma,'Local bulge fraction of stellar surface mass proxy')
    add('mass_size_compactness',log(ml*m('L')/m('Re')),'Stellar luminosity-based global mass/size proxy')
    add('gas_to_stellar_size',log(m('RHI')/m('Re')),'Published HI radius / stellar effective radius; missing HI radii stay missing')
    add('morphology',m('T'),'Published Hubble type; a proxy for population/structure, not a physical law')
    add('predicted_period',log(2*np.pi*r/np.sqrt(b2)),'Baryon-only circular period up to fixed units; exact acceleration/radius rewrite, not orbit age')
    add('predicted_speed',.5*log(b2),'Baryon-only circular speed; exact acceleration/radius rewrite')
    add('predicted_angular_momentum',log(r*np.sqrt(b2)),'Baryon-only specific circular angular momentum; exact acceleration/radius rewrite')
    add('spherical_equivalent_density',log(b2/r**2),'Spherical-equivalent mean-density proxy; exact acceleration/radius rewrite, not true galaxy volume density')
    profile={k:np.full(len(r),np.nan) for k in ['force_slope','force_curvature','stellar_slope','stellar_curvature','radial_profile_waviness','truncated_potential','potential_shape','interior_light_fraction','exterior_light_fraction','inner_stellar_contrast','outer_stellar_contrast','inner_force_contrast','outer_force_contrast','radial_sampling','radial_coverage']}
    for name in np.unique(d['name']):
        source=[p for p in rows if p['name']==name]
        rr=np.array([p['r'] for p in source]);ss=np.array([ml*p['sd']+1.4*ml*p['sb'] for p in source])
        bb=np.array([p['gas']*abs(p['gas'])+ml*p['disk']**2+1.4*ml*p['bul']**2 for p in source]);gg=bb/rr
        good=(rr>0)&(ss>0)&(bb>0);rr,ss,bb,gg=rr[good],ss[good],bb[good],gg[good]
        if len(rr)<5:continue
        lr=log(rr);lg=log(gg);ls=log(ss)
        slope=np.gradient(lg,lr);curv=np.gradient(slope,lr);sslope=np.gradient(ls,lr);scurv=np.gradient(sslope,lr)
        # Integral of the actual radial source force, with explicitly assumed finite
        # point-mass continuation. This is not a measured escape potential.
        integ=cumulative_trapezoid(gg,rr,initial=0);phi=integ[-1]-integ+bb[-1]
        area=cumulative_trapezoid(ss*rr,rr,initial=0);fraction=area/area[-1]
        waviness=np.full(len(rr),np.nan)
        for j in range(len(rr)):
            near=abs(lr-lr[j])<.3
            if np.sum(near)>=4:
                fit=np.polyfit(lr[near]-lr[j],ls[near],1)
                waviness[j]=ls[j]-fit[1]
        selected=np.flatnonzero(d['name']==name)
        for key,val in [('force_slope',slope),('force_curvature',curv),('stellar_slope',sslope),('stellar_curvature',scurv),('truncated_potential',log(phi)),('potential_shape',log(phi/bb)),('interior_light_fraction',fraction),('exterior_light_fraction',1-fraction),('radial_profile_waviness',waviness),('radial_sampling',np.gradient(rr)/rr),('radial_coverage',rr/rr[-1])]:
            profile[key][selected]=np.interp(r[selected],rr,val,left=np.nan,right=np.nan)
        for prefix,factor in [('inner',.5),('outer',2.)]:
            for label,values in [('stellar',ls),('force',lg)]:
                a=np.interp(log(r[selected]*factor),lr,values,left=np.nan,right=np.nan)
                b=np.interp(log(r[selected]),lr,values,left=np.nan,right=np.nan)
                profile[prefix+'_'+label+'_contrast'][selected]=a-b
    for name,value in profile.items():add(name,value,'Derived radial source-profile descriptor; '+('finite observed radial range and point-mass tail' if 'potential' in name else 'no 3D topology inferred'))
    add('distance',log(m('D')),'Measurement/population diagnostic: catalog distance')
    add('distance_uncertainty',m('eD')/m('D'),'Measurement diagnostic: relative distance uncertainty')
    add('inclination',m('inc'),'Measurement diagnostic: viewing inclination')
    add('inclination_uncertainty',m('einc'),'Measurement diagnostic: inclination uncertainty')
    add('quality',m('Q'),'Measurement diagnostic: rotation-curve quality code')
    keys=list(features);F=np.column_stack([features[k] for k in keys]);assert not np.any(np.isinf(F))
    return dict(**d,F=F,keys=keys,descriptions=descriptions,sources=sources)

def split(names,salt,k):
    unique=sorted(np.unique(names),key=lambda n:hashlib.sha256((salt+str(n)).encode()).hexdigest())
    lookup={n:i%k for i,n in enumerate(unique)};return np.array([lookup[n] for n in names])

def baseline(d,train,kind):
    w=prior.weights(d['name'][train]);B=prior.basis(d['x'])
    if kind=='flexible':return B@prior.smooth_fit(B[train],d['y'][train],w)
    fit=least_squares(lambda p:np.sqrt(w)*(prior.rar(d['x'][train],p[0])-d['y'][train]),[-10.],bounds=([-12.],[-9.]))
    assert fit.success;return prior.rar(d['x'],fit.x[0])

def design(d,train):
    F=d['F'].copy();w=prior.weights(d['name'][train]);missing=~np.isfinite(F)
    med=np.nanmedian(F[train],axis=0);med=np.where(np.isfinite(med),med,0.)
    F=np.where(missing,med,F);B=prior.basis(d['x'])
    adjustment=np.column_stack([prior.smooth_fit(B[train],F[train,j],w) for j in range(F.shape[1])])
    F-=B@adjustment
    scale=np.sqrt(np.sum(w[:,None]*F[train]**2,axis=0)/np.sum(w));scale=np.maximum(scale,1e-4)
    F=np.clip(F/scale,-5,5);F[missing]=0
    weak=1/(1+10**d['x']/1.2e-10);mat=[];labels=[]
    for j,key in enumerate(d['keys']):
        for shape in ['linear','tanh','square']:
            raw=F[:,j] if shape=='linear' else np.tanh(F[:,j]) if shape=='tanh' else F[:,j]**2
            for gate in ['all','weak']:
                a=raw if gate=='all' else raw*weak
                # Residualize every transform as well, so nonlinear concentration
                # terms do not merely buy a more flexible acceleration baseline.
                a=a-B@prior.smooth_fit(B[train],a[train],w);a[missing[:,j]]=0
                rms=max(np.sqrt(np.sum(w*a[train]**2)/sum(w)),1e-5)
                mat.append(a/rms);labels.append((key,shape,gate))
    return np.column_stack(mat),labels

def candidates(d,train,kind):
    base=baseline(d,train,kind);C,labels=design(d,train);w=prior.weights(d['name'][train]);res=d['y'][train]-base[train]
    numerator=C[train].T@(w*res);denominator=np.sum(w[:,None]*C[train]**2,axis=0)
    pred=[];tags=[];coefs=[]
    for penalty in [.1,1.,10.]:
        coef=numerator/(denominator+penalty*sum(w))
        pred.append(base[:,None]+np.clip(C*coef,-.5,.5));tags.extend([(*l,penalty) for l in labels]);coefs.extend(coef.tolist())
    # Joint descriptor fit can reveal combinations missed by a one-variable test.
    physical=[i for i,l in enumerate(labels) if l[0] not in ALIASES|DIAGNOSTICS and l[1:] == ('linear','all')]
    J=C[:,physical]
    for penalty in [.1,1.,10.]:
        coef=np.linalg.solve(J[train].T@(w[:,None]*J[train])+penalty*sum(w)*np.eye(len(physical)),J[train].T@(w*res))
        pred.append((base+np.clip(J@coef,-.5,.5))[:,None]);tags.append(('combined','ridge','all',penalty));coefs.append(float(np.linalg.norm(coef)))
    return base,np.column_stack(pred),tags,coefs

def support(d,train,test):
    mask=np.zeros(len(d['x']),bool)
    for i in np.flatnonzero(test):mask[i]=len(np.unique(d['name'][train&(abs(d['x']-d['x'][i])<=.15)]))>=5
    return mask

def score_matrix(d,pred,use):
    return np.mean([np.mean((d['y'][(d['name']==n)&use,None]-pred[(d['name']==n)&use])**2,axis=0) for n in np.unique(d['name'][use])],axis=0)

def evaluate(d,salt='broad-A',kind='flexible',only_selector=False):
    outer=split(d['name'],salt,5);allkeys=d['keys']+['combined','selected_physical']
    if only_selector:allkeys=['selected_physical']
    pred={k:np.full(len(d['x']),np.nan) for k in ['baseline',*allkeys]};scored=np.zeros(len(d['x']),bool);choices=[]
    for fold in range(5):
        train=outer!=fold;test=~train;inner=split(d['name'],salt+'inner'+str(fold),3)
        validation=[]
        for j in range(3):
            tr=train&(inner!=j);va=train&(inner==j)
            _,P,tags,_=candidates(d,tr,kind);validation.append(score_matrix(d,P,va))
        innerloss=np.mean(validation,axis=0)
        base,P,tags,coefs=candidates(d,train,kind);pred['baseline'][test]=base[test]
        scoped=support(d,train,test)
        for name in np.unique(d['name'][test]):
            if np.sum(scoped&(d['name']==name))<3:scoped[d['name']==name]=False
        scored|=scoped
        for key in allkeys:
            eligible=[i for i,t in enumerate(tags) if (t[0] not in ALIASES|DIAGNOSTICS if key=='selected_physical' else t[0]==key)]
            ix=min(eligible,key=lambda i:innerloss[i]);pred[key][test]=P[test,ix]
            choices.append(dict(outer_fold=fold,model=key,descriptor=tags[ix][0],shape=tags[ix][1],gate=tags[ix][2],penalty=tags[ix][3],coefficient=coefs[ix]))
    scores=[]
    for n in np.unique(d['name'][scored]):
        use=(d['name']==n)&scored;y=d['y'][use];v=d['v'][use]
        for key in allkeys:
            b=pred['baseline'][use];p=pred[key][use];vb=v*10**((b-y)/2);vp=v*10**((p-y)/2)
            scores.append(dict(name=str(n),model=key,positions=int(sum(use)),base_mse=float(np.mean((y-b)**2)),model_mse=float(np.mean((y-p)**2)),base_fractional_mse=float(np.mean((vb/v-1)**2)),model_fractional_mse=float(np.mean((vp/v-1)**2)),base_kms_mse=float(np.mean((vb-v)**2)),model_kms_mse=float(np.mean((vp-v)**2))))
    summary=[]
    for key in allkeys:
        rs=[r for r in scores if r['model']==key];a=np.array([r['base_mse'] for r in rs]);b=np.array([r['model_mse'] for r in rs]);delta=a-b
        row=dict(model=key,galaxies=len(rs),positions=sum(r['positions'] for r in rs),mse_gain_percent=float(100*(1-b.mean()/a.mean())),base_rmse_dex=float(np.sqrt(a.mean())),model_rmse_dex=float(np.sqrt(b.mean())),galaxies_improving=int(sum(delta>0)),mse_gain_bootstrap95=prior.bootstrap_mean(delta),diagnostic=key in DIAGNOSTICS,algebraic_alias=key in ALIASES)
        for metric in ['fractional','kms']:
            row[metric+'_gain_percent']=float(100*(1-np.mean([r['model_'+metric+'_mse'] for r in rs])/np.mean([r['base_'+metric+'_mse'] for r in rs])))
        summary.append(row)
    return dict(summary=summary,choices=choices,galaxy_scores=scores),pred,scored

def controls(d,observed):
    f=split(d['name'],'broad-A',5);train=f!=0;test=~train
    a= candidates(d,train,'flexible');changed={**d,'y':d['y'].copy()};changed['y'][test]+=5
    b=candidates(changed,train,'flexible');assert np.array_equal(a[1][test],b[1][test])
    # Radius/period/speed/angular-momentum/mean-density alias identities.
    F={k:d['F'][:,i] for i,k in enumerate(d['keys'])};lr=F['radius'];x=d['x'];constant=np.log10(prior.CONVERSION)
    assert np.allclose(F['predicted_speed'],.5*(x+lr-constant))
    assert np.allclose(F['predicted_period'],np.log10(2*np.pi)+.5*(lr-x+constant))
    assert np.allclose(F['predicted_angular_momentum'],.5*x+1.5*lr-.5*constant)
    assert np.allclose(F['spherical_equivalent_density'],x-lr-constant)
    # Keep each galaxy's residual pattern intact, flip its sign, and redo selection.
    # This null probes search optimism under approximate symmetric residuals.
    full=np.ones(len(x),bool);base=baseline(d,full,'flexible');res=d['y']-base;rng=np.random.default_rng(860503);null=[]
    for draw in range(99):
        signs=dict(zip(np.unique(d['name']),rng.choice([-1.,1.],len(np.unique(d['name'])))))
        mock={**d,'y':base+res*np.array([signs[n] for n in d['name']])}
        ans=evaluate(mock,only_selector=True)[0]['summary'][0];null.append(ans['mse_gain_percent'])
        if draw%20==0:print('selection null draw',draw,flush=True)
    obs=next(r['mse_gain_percent'] for r in observed['summary'] if r['model']=='selected_physical')
    return dict(held_target_mutation_invariant=True,algebraic_identities_pass=True,
        selector_null_gains_percent=null,observed_selector_gain_percent=obs,
        approximate_exceedance=(1+sum(v>=obs for v in null))/100,
        null_scope='Whole-galaxy wild sign null; full nested selection repeated; approximate diagnostic, not formal causal significance')

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=False);d=make_data()
    save(OUT/'protocol.json',json.loads(CONFIG.read_text()))
    descriptors=[dict(id=k,description=d['descriptions'][k],missing_fraction=float(np.mean(~np.isfinite(d['F'][:,i]))),role='diagnostic' if k in DIAGNOSTICS else 'alias' if k in ALIASES else 'source') for i,k in enumerate(d['keys'])]
    save(OUT/'descriptors.json',descriptors)
    runs={}
    for label,kwargs,kind,salt in [('primary',{},'flexible','broad-A'),('split_B',{},'flexible','broad-B'),('rar',{},'rar','broad-A'),('ml_0.4',dict(ml=.4),'flexible','broad-A'),('ml_0.6',dict(ml=.6),'flexible','broad-A'),('quality_1',dict(qmax=1),'flexible','broad-A'),('inclination_40_75',dict(inclination=(40,75)),'flexible','broad-A')]:
        print('running',label,flush=True);data=d if not kwargs else make_data(**kwargs)
        result,pred,supported=evaluate(data,salt,kind);runs[label]=result['summary'];save(OUT/(label+'.json'),result)
        if label=='primary':
            primary=result
            records=[]
            for i in range(len(data['x'])):
                r=dict(name=str(data['name'][i]),row=int(data['row'][i]),x=float(data['x'][i]),y=float(data['y'][i]),supported=bool(supported[i]))
                r.update({k:float(v[i]) for k,v in pred.items()});records.append(r)
            csvsave(OUT/'predictions.csv',records)
    checks=controls(d,primary);save(OUT/'controls.json',checks)
    save(OUT/'result.json',dict(status='COMPLETED_DEVELOPMENT_OBSERVABLE_FAMILY_SCREEN',galaxies=len(np.unique(d['name'])),positions=len(d['x']),descriptors=len(d['keys']),physical_descriptors=sum(r['role']=='source' for r in descriptors),runs=runs,elapsed_seconds=time.time()-started,independent_confirmation=False,causal_mechanism_established=False,all_possible_theories_exhausted=False))
    print(json.dumps(dict(top=sorted(runs['primary'],key=lambda r:-r['mse_gain_percent'])[:8],selector_null=checks['approximate_exceedance'],seconds=time.time()-started),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
