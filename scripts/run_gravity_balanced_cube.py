"""Response-blind radial-support repair of the spatial cube validation split."""
import hashlib,json,shutil,traceback
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from run_gravity_constrained_campaign import fit_object
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/constrained-cube-balanced-001';D.mkdir(exist_ok=False)
P=ROOT/'work/private/constrained-cube-balanced-001';P.mkdir(exist_ok=False)
shutil.copy2(__file__,D/'runner.py')
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
registration=dict(reason='Original geometric checkerboard can separate inner test radii from outer training radii. This is a radial extrapolation confound.',
 selection='Reuse all original guarded block interiors; choose block labels using geometry only, balancing radial bins and approaching/receding geometry. No spectra, amplitudes or velocities enter assignment.',
 radius_edges_arcsec=[48,100,160,240,330,450],candidate_assignments=4096,minimum_points_each_side_in_radial_bin=8,
 objective='Squared fractional count imbalance per radius/major-axis-sign cell, plus penalty for empty calibration or test radial bins; require both folds in an admitted radial bin.',
 guard='Existing 120 arcsec block guard; every candidate block has the same guard regardless of its label.',
 initialization='Recompute systemic and rotation seeds using the NEW training spectra only. No parameters fitted to an earlier partition are reused.',
 fitting='Same constrained model and multistart rule as constrained-cube-real-001.',
 scope='Previously exposed development data with a repaired validation design, not independent confirmation. No new gas term is fitted.')
save(D/'registration.json',registration)
source=json.loads((ROOT/'work/gravity-first-principles/conditional-cube-pilot-001/registration.json').read_text())
def partition(p,name):
    yy,xx=np.indices(p['radius'].shape);candidate=p['train_mask']|p['test_mask']
    blockid=(yy//16)*100+xx//16;ids=np.unique(blockid[candidate]);edges=np.array(registration['radius_edges_arcsec'])
    bins=np.clip(np.digitize(p['radius'],edges)-1,0,4)
    major=p['east']*np.sin(p['pa'])+p['north']*np.cos(p['pa'])
    cell=bins*2+(major>=0);counts=np.zeros((len(ids),10))
    for i,b in enumerate(ids):counts[i]=np.bincount(cell[candidate&(blockid==b)],minlength=10)
    rng=np.random.default_rng(2081+sum(map(ord,name)));labels=rng.integers(0,2,(4096,len(ids)));labels[:,0]=1
    totals=counts.sum(axis=0);train=labels@counts;test=totals-train
    score=np.sum(((train-totals/2)/(totals+1))**2,axis=1)
    tr=train.reshape(-1,5,2).sum(axis=2);te=test.reshape(-1,5,2).sum(axis=2)
    enough=(tr+te)>=16
    score+=10*np.sum(enough&((tr<8)|(te<8)),axis=1)
    choice=int(np.argmin(score));assignment=labels[choice].astype(bool)
    selectedids=ids[assignment];newtrain=candidate&np.isin(blockid,selectedids);newtest=candidate&~newtrain
    traincount=np.bincount(bins[newtrain],minlength=5);testcount=np.bincount(bins[newtest],minlength=5)
    admitted=(traincount>=8)&(testcount>=8)
    common=admitted[bins];newtrain &= common;newtest &= common
    if min(newtrain.sum(),newtest.sum())<30:raise ValueError('Insufficient common-radius spatial support')
    separation=float(np.min(distance_transform_edt(~newtrain)[newtest])*12)
    assert separation>=120 and not np.any(newtrain&newtest)
    return newtrain,newtest,dict(block_ids=ids.tolist(),train_block_ids=selectedids.tolist(),
        assignment_objective=float(score[choice]),radial_train_counts=traincount.tolist(),radial_test_counts=testcount.tolist(),
        admitted_radial_bins=admitted.tolist(),train_pixels=int(newtrain.sum()),test_pixels=int(newtest.sum()),
        minimum_center_separation_arcsec=separation,selection_reads_response=False)

results=[];audits=[];failures=[]
for name in source['names']:
    try:
        p=dict(np.load(ROOT/'work/private/conditional-cube-pilot-001'/(name+'.npz')))
        train,test,audit=partition(p,name)
        # Explicit response mutation check: only geometry can determine labels.
        altered=p.copy();altered['cube']=np.zeros_like(p['cube']);altered['amplitude']=np.zeros_like(p['amplitude'])
        tx,vx,_=partition(altered,name);assert np.array_equal(tx,train) and np.array_equal(vx,test)
        p['train_mask']=train;p['test_mask']=test
        centers=(p['velocity_edges'][:-1]+p['velocity_edges'][1:])/2
        positive=np.maximum(p['cube'][:,train],0);flux=positive.sum(axis=0);velocity=(centers@positive)/np.maximum(flux,1e-12)
        major=p['east']*np.sin(p['pa'])+p['north']*np.cos(p['pa']);ct=major[train]/np.maximum(p['radius'][train],1)
        design=np.column_stack([np.ones(len(ct)),np.sin(p['inc'])*ct]);coefs=np.linalg.lstsq(design*np.sqrt(flux[:,None]),velocity*np.sqrt(flux),rcond=None)[0]
        p['vsys_initial']=float(coefs[0]);p['speed_scale']=max(abs(coefs[1]),40.);p['rotation_initial']=np.array([0,.7,1,1,1])*coefs[1]
        np.savez_compressed(P/(name+'.npz'),**p)
        audit.update(name=name,vsys_initial=float(coefs[0]),rotation_seed_kms=float(coefs[1]),response_mutation_check_pass=True)
        audits.append(audit);save(D/'mask-audit.json',audits)
        print('START',name,audit['train_pixels'],audit['test_pixels'],audit['admitted_radial_bins'],flush=True)
        fits=fit_object(p);row=dict(name=name,fits=fits);results.append(row);save(D/(name+'.json'),row)
        print('DONE',name,[(f['mode'],round(f['test_loss'],3),f['optimizer_success']) for f in fits],flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        failures.append(dict(name=name,error=repr(e),traceback=traceback.format_exc()));print('FAIL',name,repr(e),flush=True)
    save(D/'failures.json',failures)
save(D/'result.json',dict(status='COMPLETE_DIAGNOSTIC' if not failures else 'INCOMPLETE',objects=results,failures=failures,registration=registration))
