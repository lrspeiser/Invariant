"""Independent scalar point renderer and saved refinement arithmetic."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,csv,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad,dblquad
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'scripts'))
from mond_atlas_native_emission_sampled import load_instrument,render_spectra
inst=load_instrument();C=inst['beam_covariance_yx_pixel2'];inv=np.linalg.inv(C);halo=6*np.sqrt(np.linalg.eigvalsh(C).max());area=2*np.pi*np.sqrt(np.linalg.det(C))
normalization=dblquad(lambda y,x:np.exp(-np.array([y,x])@inv@np.array([y,x])/2)/area,-halo,halo,lambda _: -halo,lambda _:halo,epsabs=1e-11,epsrel=1e-11)[0]
xy=np.array([[468.17,444.81],[490.72,468.13],[540.5,530.75]]);vel=np.array([-21.,8.,35.]);flux=np.array([.2,.3,.5]);sigma=np.array([3.,7.,12.]);W=np.zeros((15,3))
for a,ap in enumerate(inst['apertures']):
    for j,(x,y) in enumerate(xy):
        total=0
        for row in range(ap['y_start'],ap['y_stop']):
            for column in range(ap['x_start'],ap['x_stop']):
                d=np.array([row-y,column-x])
                if max(abs(d))<=halo:total+=math.exp(-d@inv@d/2)/normalization
        W[a,j]=total/144
history=json.loads((R/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json').read_text(encoding='utf-8'))['provenance'];cal=np.array(history['continuum_fit_parent_indices_zero_based']);keep=np.arange(11,53);X=np.column_stack([np.ones(63),np.arange(63)]);dv=abs(inst['velocity_increment_km_s']);errors=[]
for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
    grid=np.arange(63.) if branch=='boxcar_independent' else np.arange(65.)-1 if branch=='boxcar_hanning_full' else (np.arange(127.)-1)/2;width=.5 if branch.endswith('decimated') else 1.;parent=np.zeros((63,3))
    for j,(mu,sd) in enumerate(zip(vel,sigma)):
        density=[]
        for g in grid:
            center=inst['first_parent_velocity_km_s']+g*inst['velocity_increment_km_s'];delta=dv*width
            density.append(quad(lambda u:math.exp(-.5*((u-mu)/sd)**2)/(math.sqrt(2*math.pi)*sd),center-delta/2,center+delta/2,epsabs=1e-13)[0]/delta)
        parent[:,j]=density if branch=='boxcar_independent' else np.convolve(density,[.25,.5,.25],mode='valid')[::2 if branch.endswith('decimated') else 1]
    trend=X[keep]@np.linalg.lstsq(X[cal],parent[cal],rcond=None)[0];expected=1000*(W*flux)@(parent[keep]-trend).T;actual=render_spectra(xy,vel,flux,sigma,inst,branch)['spectra_mjy_beam'];errors.append(float(np.max(abs(expected-actual))))
assert max(errors)<1e-10
summary=json.loads((P/'sampled-run001/summary.json').read_text(encoding='utf-8'));packet=R/summary['private_spectra'];assert hashlib.sha256(packet.read_bytes()).hexdigest()==summary['private_sha256'];refinement=[]
with np.load(packet) as data:
    for row in csv.DictReader((P/'sampled-run001/refinement.csv').open(newline='',encoding='utf-8')):
        lo=data[f"{row['kind']}_{row['coarse_nr']}_{row['branch']}"][int(row['aperture'])];hi=data[f"{row['kind']}_{row['fine_nr']}_{row['branch']}"][int(row['aperture'])];L1=np.sum(abs(hi-lo))/np.sum(abs(hi));cent=abs(np.sum(hi*inst['velocity_km_s'])/hi.sum()-np.sum(lo*inst['velocity_km_s'])/lo.sum());refinement.extend([abs(L1-float(row['profile_L1'])),abs(cent-float(row['centroid_km_s']))]);assert (L1<.01 and cent<.5)==(row['passed']=='True')
assert max(refinement)<1e-12
receipt=dict(status='PASS_INDEPENDENT_NATIVE_EMISSION_REVIEW',independent_scalar_emitter_errors=errors,independent_continuum_least_squares=True,independent_spectral_bin_quadrature=True,independent_double_integral_beam_normalization=normalization,refinement_rows=len(refinement)//2,refinement_arithmetic_error=max(refinement),observed_source_spectra_read=0)
(P/'independent-review').mkdir(exist_ok=True);(P/'independent-review/receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
