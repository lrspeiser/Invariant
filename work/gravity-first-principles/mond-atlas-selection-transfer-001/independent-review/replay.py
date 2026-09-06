"""Independent aggregation, paired masks and synthetic-template audit."""
import csv,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_native_selection import recovery,select_runs,integrated_gaussian,spectral_matrix
from mond_atlas_selection_transfer import intrinsic
OWN=Path(__file__).resolve().parent;RUN=OWN.parent/'run001'


def read(name):return list(csv.DictReader((RUN/name).open(encoding='utf-8')))
def number(v):return float(v=='True') if v in ['True','False'] else float(v)
def loop_mask(x,sigma):
    mask=np.zeros_like(x,dtype=bool)
    for y,z in itertools.product(range(x.shape[1]),range(x.shape[2])):
        for k in range(len(x)-2):
            if all(x[j,y,z]>2*sigma[j] for j in range(k,k+3)):mask[k:k+3,y,z]=True
    return mask


def run():
    trials=read('trials.csv');aggregates=read('case-summary.csv');pairs=read('paired-morphology.csv');summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'));maximum=0.;metricnames=['true_flux_fraction_retained','peak_selected','paired_selected_flux_difference_over_reference','selected_noisy_flux_over_reference']
    keys=['group','branch','center','amplitude','kind'];groups={}
    for row in trials:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    recovery_counts=dict(empirical=0,gaussian=0,noiseless=0);pair_counts=recovery_counts.copy()
    for row in aggregates:
        values=groups[tuple(row[k] for k in keys)];assert len(values)==int(row['n'])
        for metric in metricnames:
            v=np.array([number(r[metric]) for r in values]);sd=float(v.std(ddof=1)) if len(v)>1 else 0.
            for suffix,actual in [('mean',v.mean()),('sd',sd),('min',v.min()),('max',v.max())]:maximum=max(maximum,abs(actual-float(row[metric+'_'+suffix])))
            if row['group']=='gaussian':maximum=max(maximum,abs(sd/np.sqrt(len(v))-float(row[metric+'_conditional_mc_se'])))
        retained=np.mean([number(v['true_flux_fraction_retained']) for v in values]);paired=np.mean([number(v['paired_selected_flux_difference_over_reference']) for v in values]);passed=retained>=.9 and abs(paired-1)<=.1
        assert passed==(row['adequate_recovery']=='True');recovery_counts[row['group']]+=int(passed)
    maxpair=0.;worst={}
    for row in pairs:
        key=tuple(row[k] for k in keys);one=sorted(groups[key],key=lambda v:int(v['draw']));base=sorted(groups[key[:-1]+('rotation',)],key=lambda v:int(v['draw']));assert [v['draw'] for v in one]==[v['draw'] for v in base]
        d=np.array([float(a['true_flux_fraction_retained'])-float(b['true_flux_fraction_retained']) for a,b in zip(one,base)]);sd=float(d.std(ddof=1)) if len(d)>1 else 0.
        for key2,val in [('mean_retention_difference',d.mean()),('sd',sd),('minimum',d.min()),('maximum',d.max())]:maxpair=max(maxpair,abs(val-float(row[key2])))
        passed=abs(d.mean())<=.05;assert passed==(row['transfer_gate_pass']=='True');pair_counts[row['group']]+=int(passed)
        if row['group'] not in worst or abs(d.mean())>abs(worst[row['group']]['mean_retention_difference']):worst[row['group']]=dict(branch=row['branch'],center=int(row['center']),amplitude=int(row['amplitude']),kind=row['kind'],mean_retention_difference=float(d.mean()))
    assert maximum<1e-12 and maxpair<1e-12 and recovery_counts==summary['recovery_cases_pass'] and pair_counts==summary['transfer_pairs_pass']
    counts={g:sum(r['group']==g for r in trials) for g in summary['counts']};assert counts==summary['counts']
    rng=np.random.default_rng(925);shape=(9,4,5);n=rng.normal(size=shape);d=rng.normal(size=shape);ns=rng.normal(size=shape)*.3;ds=rng.random(shape);ps=rng.random(shape);sigma=np.linspace(.2,.5,9);amp=1.3;ff=.7
    mask=loop_mask(d+amp*ds,sigma);base=loop_mask(d,sigma);actual=recovery(n,d,ns,ds,ps,sigma,amp,ff);total=(amp*ps).sum();selected=(n+amp*ns)[mask].sum();reference=dict(peak_selected=bool(mask[np.unravel_index(np.argmax(ds),shape)]),true_flux_fraction_retained=float((amp*ps)[mask].sum()/total),reference_flux_jy_kms=float(total*ff),post_continuum_flux_over_reference=float((amp*ns).sum()/total),selected_noisy_flux_over_reference=float(selected/total),paired_selected_flux_difference_over_reference=float((selected-n[base].sum())/total),selected_voxel_fraction=float(mask.mean()))
    assert mask.tolist()==select_runs(d+amp*ds,sigma).tolist();recoveryerr=max(abs(float(reference[k])-float(actual[k])) for k in reference);assert recoveryerr<1e-12
    flux=[];qerror=[]
    for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
        h,grid,width=spectral_matrix(64,branch)
        for center in [10,20,30]:
            values=[intrinsic(kind,grid,width,center)[0].sum()*width for kind in ['rotation','warp','streaming']];flux.append(max(abs(np.array(values)/values[0]-1)))
    for center,width in [(0.,1.),(.35,.5),(-1.2,1.)]:
        for g in [-2.,0.,2.]:
            exact=quad(lambda t:np.exp(-4*np.log(2)*(t-center)**2/2**2),g-width/2,g+width/2,epsabs=1e-13)[0]/width
            qerror.append(abs(exact-float(integrated_gaussian(g,center,2.,width))))
    bindings=json.loads((RUN/'pre-access-bindings.json').read_text(encoding='utf-8'))['bindings']
    for p,h in bindings.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    result=dict(status='PASS_WITH_SCOPE_LIMITS',trial_counts=counts,aggregate_cases=len(aggregates),paired_cases=len(pairs),aggregate_max_absolute_error=maximum,paired_max_absolute_error=maxpair,recovery_case_pass=recovery_counts,morphology_pair_pass=pair_counts,worst_morphology_differences=worst,independent_mask_exact=True,independent_recovery_max_error=recoveryerr,intrinsic_flux_relative_spread=max(flux),line_integral_quadrature_max_error=max(qerror),all_input_hashes_verified=len(bindings),raw_background_or_observed_velocities_opened=False,limits=['Synthetic line-of-sight channel fields, not dynamical warped density reconstruction.','Empirical patches overlap and were previously screened/exposed; not independent empty sky.','Full production trials not rerun; saved aggregates and paired gates replayed from per-trial outputs.','Conditional Gaussian spectral/spatial noise does not validate the real dirty-beam pipeline.'])
    (OWN/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
