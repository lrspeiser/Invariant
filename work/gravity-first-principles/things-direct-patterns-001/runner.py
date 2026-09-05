"""Direct HI brightness -> antisymmetric LOS motion, whole-galaxy development CV."""
import run_gravity_observable_patterns as fit
import json
import re
import time
import sys
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS

ROOT=fit.ROOT
D=Path(sys.argv[1]);D.mkdir(exist_ok=False)
(D/'runner.py').write_bytes(Path(__file__).read_bytes())
(D/'run_gravity_observable_patterns.py').write_bytes((ROOT/'scripts/run_gravity_observable_patterns.py').read_bytes())
ACQ=ROOT/'work/gravity-first-principles/things-observable-acquisition-003'
receipt=json.loads((ACQ/'receipt.json').read_text());assert receipt['status']=='COMPLETE'
assets={(a['name'],a['resolution'],a['moment']):a for a in receipt['files']}
names=sorted({a['name'] for a in receipt['files']})
photo={g['galaxy']:g for g in json.loads((ROOT/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
raw={g['name']:g for g in json.loads((ROOT/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
assert set(names)<=photo.keys() and set(names)==raw.keys()
geom={}
for g in json.loads((ROOT/'configs/open_gravity_rg_s4g_geometry_source_v1.json').read_text())['objects']:
    q=1-g['outer_ellipticity'];inc=np.rad2deg(np.arccos(np.sqrt((q*q-.13**2)/(1-.13**2))))
    geom[g['object_id']]=dict(ra=g['ra_deg'],dec=g['dec_deg'],pa=g['outer_position_angle_deg'],inc=inc,
        provenance='Independent photometric orientation with assumed intrinsic axis ratio 0.13',flag=g['orientation_flag'])
for g in json.loads((ROOT/'configs/open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1.json').read_text())['objects']:
    geom[g['object_id']]=dict(ra=g['ra_deg'],dec=g['dec_deg'],pa=g['position_angle_deg'],inc=g['primary_inclination_deg'],
        provenance=g['geometry_provenance'],flag='kinematic_geometry' if g['object_id']!='UGC04305' else 'photometric')
scenarios={'nominal':([.5,.7],0,0),'lighter_stars':([.35,.5],0,0),'heavier_stars':([.65,.9],0,0),
    'inclination_minus5':([.5,.7],-5,0),'inclination_plus5':([.5,.7],5,0),
    'pa_minus5':([.5,.7],0,-5),'pa_plus5':([.5,.7],0,5)}
registration=dict(names=names,geometry=geom,scenarios=scenarios,
    source_hash=fit.sha(ACQ/'receipt.json'),scope='Previously exposed 12-galaxy development data, not independent confirmation.',
    target='Half absolute velocity difference at diametrically opposite sky locations after intensity-weighted smoothing to a circular 20 arcsec Gaussian sigma. Cancels systemic velocity; measures odd LOS motion, not total circular acceleration.',
    source='Actual HI MOM0 intensity; beam corrected to circular sigma 20,40,80 arcsec. Added broad/local log intensity contrasts and opposite-side log intensity asymmetry.',
    baseline='Published SPARC source components -> nominal RAR circular speed -> photometric or published kinematic projection -> intensity-weighted smoothing; no rotation-curve velocities used as predictors.',
    mask='Nonzero finite MOM0; MOM1 finite nonzero; Gaussian support local>.98, 80 arcsec>.75, velocity support>.98; reference projected speed>10 km/s; source radial support>.98; absolute cos(theta)>.5. Both sides required.',
    sampling='Fixed 20-pixel grid, one side of each opposite pair, minimum 10 pairs per galaxy; whole galaxies hold out, not pixels. Geometry scenarios reuse the common admitted grid across all scenarios and both products.',
    validation='Leave one galaxy out; three inner whole-galaxy folds select parameters. No target-galaxy nuisance fitting.',
    algorithms={'ridge':[.01,.1,1.],'trees':[(3,40),(7,80)],'gpu_rbf_features':[(1.,.01),(1.,.1),(3.,.01),(3.,.1)]},
    limitations=['MOM2 is gas velocity spread, not an error bar; no chi-square or noise-calibrated significance.',
    'Opposite-pair motion can include noncircular flows, warps and asymmetric profiles.',
    'Seven-object metadata include kinematically inferred geometry; predictions are conditional on those published values.',
    'Standard intensity-weighted convolution of released moments is an approximation to full cube forward modeling.',
    'Shared observations and masks make natural/robust agreement a processing check, not independent replication.',
    'Coverage cuts define detected gas; no physical void or unique 3D reconstruction is inferred.'])
fit.save(D/'registration.json',registration)

def cov(major,minor,pa):
    p=np.deg2rad(pa);u=np.array([-np.sin(p),np.cos(p)]);v=np.array([np.cos(p),np.sin(p)])
    return major**2*np.outer(u,u)+minor**2*np.outer(v,v)

def kernel(c):
    eig=np.linalg.eigvalsh(c);assert min(eig)>0
    n=int(np.ceil(5*np.sqrt(max(eig))));yy,xx=np.mgrid[-n:n+1,-n:n+1];a=np.linalg.inv(c)
    k=np.exp(-.5*(a[0,0]*xx*xx+2*a[0,1]*xx*yy+a[1,1]*yy*yy));return cp.asarray(k/k.sum())

def smooth(a,k):return fftconvolve(cp.asarray(a),k,mode='same')
def sample(a,points):return map_coordinates(a,points,order=1,mode='constant',cval=0.)

def source_fields(name,g,shape,header,ml,di,dp):
    ny,nx=shape;yy,xx=np.indices(shape)
    wcs=WCS(header).celestial;cx,cy=wcs.all_world2pix(g['ra'],g['dec'],0)
    # SIN tangent-plane offsets; subdegree footprint. No line-of-sight depth is assigned.
    px=abs(header['CDELT1'])*3600
    east=-(xx-cx)*px;north=(yy-cy)*px
    pa=np.deg2rad(g['pa']+dp);inc=np.deg2rad(np.clip(g['inc']+di,10,85))
    major=east*np.sin(pa)+north*np.cos(pa);minor=east*np.cos(pa)-north*np.sin(pa)
    radius=np.hypot(major,minor/np.cos(inc))*raw[name]['distance_mpc']*1000*np.pi/(180*3600)
    ct=major/np.maximum(np.hypot(major,minor/np.cos(inc)),1e-12)
    r,_,_,gas,disk,bulge=np.array(raw[name]['rows'],float).T
    vg=np.interp(radius,r,gas);vd=np.interp(radius,r,disk);vb=np.interp(radius,r,bulge)
    vbar=vg*abs(vg)+ml[0]*vd**2+ml[1]*vb**2
    positive=vbar>0;y=np.maximum(vbar,1e-8)/np.maximum(radius,1e-8)/3702.81458
    vrar=np.sqrt(np.maximum(vbar,1e-8)/(-np.expm1(-np.sqrt(y))))
    los=vrar*np.sin(inc)*ct
    domain=(radius>=r[0])&(radius<=r[-1])&positive
    sb=np.interp(radius,r,np.array(photo[name]['rows'],float).sum(axis=1))
    fg=vg**2/np.maximum(vg**2+ml[0]*vd**2+ml[1]*vb**2,1e-8)
    return dict(los=los,domain=domain,radius=radius,ct=ct,y=y,sb=sb,fg=fg,cx=float(cx),cy=float(cy),inc=float(inc))

def benchmarks():
    k=kernel(np.eye(2)*4);a=np.ones((101,101))*7
    z=cp.asnumpy(smooth(a,k)/smooth(np.ones_like(a),k))
    assert np.max(abs(z-7))<1e-10
    x=np.linspace(-30,30,61);v=700+3*x;opp=700-3*x
    assert np.max(abs(.5*abs(v-opp)-3*abs(x)))<1e-12
    # Intensity weighting and a symmetric beam preserve a linear velocity field away from edges.
    xx=np.broadcast_to(x,(61,61));blur=cp.asnumpy(smooth(700+3*xx,k)/smooth(np.ones_like(xx),k))
    assert np.max(abs(blur[15:-15,15:-15]-(700+3*xx)[15:-15,15:-15]))<1e-9
    h=fits.getheader(ROOT/assets[('NGC3198','NA',0)]['file']);w=WCS(h).celestial
    p=np.array([[200.,300.],[512.,512.],[800.,700.]]);q=w.all_world2pix(w.all_pix2world(p,0),0)
    err=float(np.max(abs(p-q)));assert err<1e-6
    fit.save(D/'controls.json',dict(constant_preserved=True,opposite_pair_systemic_cancellation=True,
        manufactured_linear_velocity_beam_preserved=True,wcs_roundtrip_pixel_error=err))

def build_all():
    bundles={};audit=[]
    for name in names:
        g=geom[name];variants={};checks=[]
        for resolution in ('NA','RO'):
            paths=[ROOT/assets[(name,resolution,j)]['file'] for j in range(3)]
            h=fits.getheader(paths[0]);a=np.squeeze(fits.getdata(paths[0])).astype(float)
            hist='\n'.join(str(s) for s in h.get('HISTORY',[]))
            beams=re.findall(r'CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)',hist)
            assert beams;bm,bn,bp=map(float,beams[-1]);pixel=abs(h['CDELT1'])*3600
            beam=cov(bm*3600/2.354820045,bn*3600/2.354820045,bp)
            for path in paths[1:]:
                hh=fits.getheader(path)
                assert all(h[k]==hh[k] for k in ['NAXIS1','NAXIS2','CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2'])
            good=np.isfinite(a)&(a!=0);intensity=np.where(good,a/1000/(np.pi/(4*np.log(2))*bm*bn*3600**2),0.)
            kernels=[kernel((np.eye(2)*s*s-beam)/pixel**2) for s in (20.,40.,80.)]
            supports=[cp.asnumpy(smooth(good.astype(float),k)) for k in kernels]
            sums=[cp.asnumpy(smooth(intensity,k)) for k in kernels]
            means=[s/np.maximum(c,1e-12) for s,c in zip(sums,supports)]
            # Velocity is loaded only after the source-only image fields exist.
            v=np.squeeze(fits.getdata(paths[1])).astype(float)/1000
            vgood=np.isfinite(v)&(v!=0)&good
            denom=cp.asnumpy(smooth(np.where(vgood,intensity,0),kernels[0]))
            vm=cp.asnumpy(smooth(np.where(vgood,intensity*v,0),kernels[0]))/np.maximum(denom,1e-20)
            vcov=cp.asnumpy(smooth(vgood.astype(float),kernels[0]))
            shape=a.shape;f0=source_fields(name,g,shape,h,[.5,.7],0,0)
            ys,xs=np.mgrid[10:shape[0]:20,10:shape[1]:20]
            # Fix one half-plane in the reference geometry to avoid duplicate pairs.
            one=sample(f0['ct'],[ys.ravel(),xs.ravel()])>0
            points=np.array([ys.ravel()[one],xs.ravel()[one]],float)
            opposite=np.array([2*f0['cy']-points[0],2*f0['cx']-points[1]])
            velocity=.5*abs(sample(vm,points)-sample(vm,opposite))
            average=lambda field:.5*(sample(field,points)+sample(field,opposite))
            support_min=lambda field:np.minimum(sample(field,points),sample(field,opposite))
            local=average(means[0]);broad1=average(means[1]);broad2=average(means[2])
            extra=np.column_stack([np.log10(np.maximum(broad1,1e-20)/np.maximum(local,1e-20)),
                np.log10(np.maximum(broad2,1e-20)/np.maximum(local,1e-20)),
                abs(np.log10(np.maximum(sample(means[0],points),1e-20)/np.maximum(sample(means[0],opposite),1e-20)))])
            basevalid=(support_min(supports[0])>.98)&(support_min(supports[2])>.75)&(support_min(vcov)>.98)&(local>0)&(broad1>0)&(broad2>0)
            for scenario,(ml,di,dp) in scenarios.items():
                fields=source_fields(name,g,shape,h,ml,di,dp)
                model=cp.asnumpy(smooth(intensity*fields['los'],kernels[0]))/np.maximum(sums[0],1e-20)
                predicted=.5*abs(sample(model,points)-sample(model,opposite))
                radial_support=cp.asnumpy(smooth(intensity*fields['domain'],kernels[0]))/np.maximum(sums[0],1e-20)
                valid=basevalid&(support_min(radial_support)>.98)&(predicted>10)&(support_min(abs(fields['ct']))>.5)
                rr=average(fields['radius']);y=average(fields['y']);sb=average(fields['sb']);fg=average(fields['fg'])
                # Local inputs include coverage at all scales, so aperture support itself is a comparator.
                X=np.column_stack([np.log10(np.maximum(y,1e-20)),np.log10(np.maximum(rr,1e-20)),np.log10(1+sb),fg,
                    np.log10(np.maximum(local,1e-20)),average(abs(fields['ct'])),average(supports[0]),average(supports[1]),average(supports[2]),
                    np.full(len(rr),np.sin(fields['inc'])),np.full(len(rr),np.log10(raw[name]['distance_mpc']))])
                variants[(resolution,scenario)]=dict(name=name,r=rr,v=velocity,error=np.ones(len(rr)),vrar=predicted,
                    target=velocity/np.maximum(predicted,1e-20)-1,X={'local':X,'gas_surroundings':np.column_stack([X,extra])},
                    valid=valid,points=points)
            checks.append(dict(resolution=resolution,nonzero_fraction=float(good.mean()),beam_arcsec=[bm*3600,bn*3600],
                candidate_pairs=len(velocity),source_intensity_and_features_independent_of_velocity_values=True))
        common=np.logical_and.reduce([v['valid'] for v in variants.values()])
        count=int(common.sum());audit.append(dict(name=name,usable_pairs=count,minimum_pairs=10,admitted=count>=10,checks=checks,geometry=g))
        if count>=10:
            for key,o in variants.items():
                o.pop('valid');o.pop('points')
                for k in ['r','v','error','vrar','target']:o[k]=o[k][common]
                o['X']={k:x[common] for k,x in o['X'].items()}
                bundles.setdefault(key,[]).append(o)
        fit.save(D/'data_audit_partial.json',audit)
        print(f'Source and motion {name}: {count} common pairs',flush=True)
    fit.save(D/'data_audit.json',audit)
    return bundles

def run(bundles):
    results=[]
    for (resolution,scenario),objects in bundles.items():
        object_names=[o['name'] for o in objects];n=len(objects);assert n>=5
        for algorithm,parameters in registration['algorithms'].items():
            for group in ('local','gas_surroundings'):
                predictions=[None]*n;folds=[]
                for outer in range(n):
                    train=[i for i in range(n) if i!=outer];inner=fit.foldmap(object_names,train,'things-direct-round1',3)
                    losses=[]
                    for param in parameters:
                        loss=[]
                        for infold in range(3):
                            tr=[i for i in train if inner[i]!=infold];va=[i for i in train if inner[i]==infold]
                            pp=fit.train_predict(objects,tr,va,group,algorithm,param)
                            loss.extend(np.mean((objects[i]['target']-p)**2) for i,p in zip(va,pp))
                        losses.append(float(np.mean(loss)))
                    best=parameters[int(np.argmin(losses))]
                    predictions[outer]=fit.train_predict(objects,train,[outer],group,algorithm,best)[0]
                    folds.append(dict(test=object_names[outer],selected=best,inner_losses=losses))
                scored=fit.score(objects,predictions)
                result=dict(resolution=resolution,scenario=scenario,algorithm=algorithm,group=group,names=object_names,
                    pairs=sum(len(o['v']) for o in objects),fractional_gain=scored['fractional_mse_gain'],
                    kms_gain=scored['kms_mse_gain'],galaxies_improved=scored['galaxies_improved'],
                    nonpositive=scored['nonpositive_predictions'],before=scored['before'],after=scored['after'],folds=folds)
                results.append(result);fit.save(D/'results_partial.json',results)
                if scenario=='nominal':
                    fit.save(D/f'predictions_{resolution}_{algorithm}_{group}.json',[
                        dict(name=o['name'],r=o['r'].tolist(),observed_pair_speed=o['v'].tolist(),
                            rar_pair_speed=o['vrar'].tolist(),predicted_pair_speed=(o['vrar']*(1+p)).tolist(),
                            gas_features=o['X']['gas_surroundings'][:,-3:].tolist()) for o,p in zip(objects,predictions)])
                print(f'{resolution} {scenario} {algorithm} {group}: frac {result["fractional_gain"]:.3f} km {result["kms_gain"]:.3f}',flush=True)
    fit.save(D/'result.json',dict(registration=registration,models=results,independent_confirmation=False,new_laws=0))

start=time.perf_counter();benchmarks();bundles=build_all();run(bundles)
fit.save(D/'completion.json',dict(status='COMPLETED',seconds=time.perf_counter()-start))
