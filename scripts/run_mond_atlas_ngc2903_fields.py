"""Execute conditional full QUMOND fields from an observed NGC2903 source map.

No observed motions are loaded. All reported speeds are radial force-equivalent
diagnostics. The source ensemble is illustrative and is not a mass posterior.
"""
from __future__ import annotations
import argparse,gc,time
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,digest,write_json,write_csv
import mond_atlas_rectangular_fields as rf


def interpolate2(array,axis,x,y):
    h=axis[1]-axis[0];u=(np.asarray(x)-axis[0])/h;v=(np.asarray(y)-axis[0])/h
    ix=np.floor(u).astype(int);iy=np.floor(v).astype(int)
    inside=(ix>=0)&(iy>=0)&(ix<len(axis)-1)&(iy<len(axis)-1)
    i=np.clip(ix,0,len(axis)-2);j=np.clip(iy,0,len(axis)-2);du=u-ix;dv=v-iy
    result=(1-du)*(1-dv)*array[i,j]+du*(1-dv)*array[i+1,j]+(1-du)*dv*array[i,j+1]+du*dv*array[i+1,j+1]
    return np.where(inside,result,0.)


def symmetrize(surface,axis,annulus_width):
    x,y=np.meshgrid(axis,axis,indexing='ij');ring=np.floor(np.hypot(x,y)/annulus_width).astype(int)
    count=np.bincount(ring.ravel());total=np.bincount(ring.ravel(),weights=surface.ravel())
    return (total/count)[ring]


def density_for_case(packet,config,case,half,spacing):
    axes=[np.linspace(-half,half,int(round(2*half/h))+1) for h in spacing]
    if any(abs((a[1]-a[0])-h)>1e-10 for a,h in zip(axes,spacing)):raise ValueError('box not divisible by step')
    x,y=np.meshgrid(axes[0],axes[1],indexing='ij');rho=np.zeros(tuple(len(a) for a in axes))
    factors={'stellar_luminosity':case['ml'],'atomic_helium':1.,'co21':case['alpha_co10']/case['r21']}
    masses={};normalizations={}
    for component,factor in factors.items():
        source_axis=packet[component+'_axis'];surface=packet[component+'_'+case['fill']]*factor
        if case.get('axisymmetrize'):surface=symmetrize(surface,source_axis,config['source_grid']['annulus_width_kpc'])
        surface=surface*1e6 # pc^-2 -> kpc^-2
        expected=float(surface.sum()*(source_axis[1]-source_axis[0])**2)
        sampled=interpolate2(surface,source_axis,x,y)
        sampled_mass=float(sampled.sum()*spacing[0]*spacing[1])
        if sampled_mass<=0:raise ValueError('empty resampled source')
        norm=expected/sampled_mass;sampled*=norm
        height=case['star_height_kpc'] if component=='stellar_luminosity' else case['gas_height_kpc']
        vertical=np.exp(-np.abs(axes[2])/height);vertical/=vertical.sum()*spacing[2]
        rho+=sampled[:,:,None]*vertical[None,None,:]
        masses[component]=expected;normalizations[component]=norm
    return rho,axes,dict(component_masses_msun=masses,resampling_normalization=normalizations,
        discrete_total_msun=float(rho.sum()*np.prod(spacing)),vertical_family='discretely normalized exp(-abs(z)/height)')


def force_profiles(pn,pm,axes,spacing,config):
    mid=len(axes[2])//2;gx_n,gy_n=np.gradient(-pn[:,:,mid],*spacing[:2],edge_order=2)
    gx_m,gy_m=np.gradient(-pm[:,:,mid],*spacing[:2],edge_order=2)
    radii=np.arange(config['numerics']['force_radius_min_kpc'],config['numerics']['force_radius_max_kpc']+.01,.5)
    angles=np.arange(72)*2*np.pi/72;rows=[];summary=[]
    a0=config['gravity']['a0_kms2_per_kpc']
    for r in radii:
        c,s=np.cos(angles),np.sin(angles);x=r*c;y=r*s
        nx=interpolate2(gx_n,axes[0],x,y);ny=interpolate2(gy_n,axes[0],x,y)
        mx=interpolate2(gx_m,axes[0],x,y);my=interpolate2(gy_m,axes[0],x,y)
        nr=-(nx*c+ny*s);nt=-nx*s+ny*c;mr=-(mx*c+my*s);mt=-mx*s+my*c
        norm=np.hypot(nx,ny);nu=np.zeros_like(norm);positive=norm>0;nu[positive]=.5+np.sqrt(.25+a0/norm[positive])
        algebraic=nu*nr
        for i,angle in enumerate(angles):
            rows.append(dict(radius_kpc=float(r),angle_deg=float(np.rad2deg(angle)),newton_inward=float(nr[i]),newton_tangential=float(nt[i]),
                mond_inward=float(mr[i]),mond_tangential=float(mt[i]),algebraic_inward=float(algebraic[i])))
        summary.append(dict(radius_kpc=float(r),newton_inward_mean=float(np.mean(nr)),mond_inward_mean=float(np.mean(mr)),
            newton_force_speed_kms=float(np.sqrt(max(0,r*np.mean(nr)))),mond_force_speed_kms=float(np.sqrt(max(0,r*np.mean(mr)))),
            newton_radial_azimuth_cv=float(np.std(nr)/np.mean(nr)),mond_radial_azimuth_cv=float(np.std(mr)/np.mean(mr)),
            newton_tangential_fraction=float(np.sqrt(np.mean(nt*nt))/np.mean(nr)),mond_tangential_fraction=float(np.sqrt(np.mean(mt*mt))/np.mean(mr)),
            full_minus_algebraic_fraction=float(np.mean(mr-algebraic)/np.mean(mr))))
    return rows,summary,dict(newton_midplane=pn[:,:,mid],mond_midplane=pm[:,:,mid],axis=axes[0])


def run_one(packet,config,case,half,spacing,output,private,label):
    start=time.monotonic();print('START '+label,flush=True)
    rho,axes,source=density_for_case(packet,config,case,half,spacing)
    G=config['gravity']['G_kpc_kms2_per_msun'];a0=config['gravity']['a0_kms2_per_kpc']
    bn,bm,moments=rf.multipole_boundary(rho,axes,G,a0)
    pn,pm,residuals=rf.solve(rho,spacing,bn,bm,G,a0)
    del rho,bn,bm;gc.collect()
    rows,profile,plane=force_profiles(pn,pm,axes,spacing,config)
    del pn,pm;gc.collect()
    write_csv(output/(label+'-forces.csv'),rows);write_csv(output/(label+'-profile.csv'),profile)
    np.savez_compressed(private/(label+'-midplane.npz'),**plane)
    result=dict(id=label,case=case,half_width_kpc=half,spacing_kpc=list(spacing),shape=[len(a) for a in axes],
        source=source,moments=moments,residuals=residuals,seconds=time.monotonic()-start,profile=profile)
    write_json(output/(label+'-result.json'),result)
    print('DONE '+label+' '+str(round(result['seconds'],2))+' seconds',flush=True)
    return result


def compare(left,right):
    rows=[]
    for a,b in zip(left['profile'],right['profile']):
        if a['radius_kpc']!=b['radius_kpc']:raise ValueError('radius mismatch')
        rows.append(dict(radius_kpc=a['radius_kpc'],newton_relative=b['newton_inward_mean']/a['newton_inward_mean']-1,
                         mond_relative=b['mond_inward_mean']/a['mond_inward_mean']-1))
    return dict(newton_relative_force_rms=float(np.sqrt(np.mean([r['newton_relative']**2 for r in rows]))),
                mond_relative_force_rms=float(np.sqrt(np.mean([r['mond_relative']**2 for r in rows]))),radii=rows)


def main(args):
    config=read_json(args.protocol);audit=read_json(args.source/'source-audit.json')
    if digest(args.protocol)!=audit['protocol_sha256']:raise ValueError('source protocol changed')
    packet_path=ROOT/audit['source_packet']
    if digest(packet_path)!=audit['source_packet_sha256']:raise ValueError('source packet hash mismatch')
    if args.output.exists() or args.private.exists():raise FileExistsError('immutable output already exists')
    args.output.mkdir(parents=True);args.private.mkdir(parents=True)
    with np.load(packet_path) as loaded:packet={k:loaded[k] for k in loaded.files}
    results={};num=config['numerics'];spacing=num['base_spacing_kpc'];half=num['base_half_width_kpc']
    cases=config['cases'][:1] if args.nominal_only else config['cases']
    for case in cases:results[case['id']]=run_one(packet,config,case,half,spacing,args.output,args.private,case['id'])
    checks={}
    if args.convergence:
        nominal=config['cases'][0]
        specs=[('larger_box',num['larger_half_width_kpc'],spacing),('vertical_refined',half,[spacing[0],spacing[1],spacing[2]/2]),
               ('lateral_refined',half,[spacing[0]/2,spacing[1]/2,spacing[2]])]
        for label,box,h in specs:
            results[label]=run_one(packet,config,nominal,box,h,args.output,args.private,label)
            checks[label]=compare(results['nominal'],results[label])
    all_pass=bool(checks) and all(max(v['newton_relative_force_rms'],v['mond_relative_force_rms'])<num['convergence_relative_rms_gate'] for v in checks.values())
    write_json(args.output/'field-audit.json',dict(status='CONDITIONAL_FULL_FIELD_DEVELOPMENT_DIAGNOSTIC',object_id=config['object_id'],
        source_audit_sha256=digest(args.source/'source-audit.json'),protocol_sha256=digest(args.protocol),
        bindings=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in (Path(__file__),ROOT/'scripts/mond_atlas_rectangular_fields.py',ROOT/'scripts/mond_atlas_fields.py')],
        full_field_cases_executed=list(results),convergence=checks,declared_convergence_gates_pass=all_pass,
        astrophysical_likelihood_admitted=False,external_field_observationally_constrained=False,aqual_comparison_complete=False,
        limitations=config['source_policy']))
    print('COMPLETE; convergence gates: '+str(all_pass),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/mond_atlas_ngc2903_field_v1.json')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--nominal-only',action='store_true');p.add_argument('--convergence',action='store_true')
    a=p.parse_args()
    for key in ('protocol','source','output','private'):setattr(a,key,getattr(a,key).resolve())
    main(a)
