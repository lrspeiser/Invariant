"""Independent packet arithmetic, integer labels, distance invariance and v1 preservation."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-ngc3198-recovery-001'
OLD=ROOT/'work/gravity-first-principles/mond-atlas-generic-source-002'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
COMPONENTS=['stellar_luminosity','atomic_helium','co21']

def run():
    out=P/'verification';out.mkdir(exist_ok=False)
    config=read(OLD/'correction-preflight-001/config-ngc3198-source-v2.json')
    summary=read(P/'run-001/summary.json');before=read(OLD/'run-001/summary.json')
    oldcases={r['id']:r for r in before['cases']};cases={r['id']:r for r in summary['cases']}
    assert len(cases)==12 and len(summary['mass_cases'])==72
    component_checks=[];saved={};bindings={}
    for case in summary['cases']:
        name=case['id'];packet=ROOT/case['packet'];oldpacket=ROOT/oldcases[name]['packet']
        assert digest(packet)==case['packet_sha256'] and digest(oldpacket)==oldcases[name]['packet_sha256']
        bindings[packet.relative_to(ROOT).as_posix()]=digest(packet)
        with np.load(packet,allow_pickle=False) as a,np.load(oldpacket,allow_pickle=False) as old:
            h=case['grid']['spacing_kpc'];layout=case['grid']['dimensionless_layout'];half=layout['half_width_cells'];width=layout['annulus_width_cells']
            jj,kk=np.mgrid[-half:half+1,-half:half+1];sq=jj*jj+kk*kk
            ring=np.fromiter((math.isqrt(int(v))//width for v in sq.flat),dtype=int).reshape(sq.shape)
            radius=np.sqrt(sq);start=layout['taper_start_cells'];cut=layout['cutoff_cells']
            taper=np.where(radius<=start,1.,np.where(radius<cut,(cut-radius)/(cut-start),0.))
            for comp in COMPONENTS:
                observed=a[comp+'_observed'];coverage=a[comp+'_coverage'];mean=a[comp+'_mean']
                assert np.array_equal(observed,old[comp+'_observed'])
                assert np.array_equal(coverage,old[comp+'_coverage'])
                area=coverage*h*h;flux=observed*h*h*1e6;nr=ring.max()+1
                rarea=np.bincount(ring.ravel(),weights=area.ravel(),minlength=nr)
                rflux=np.bincount(ring.ravel(),weights=flux.ravel(),minlength=nr)
                rcov=rarea/(np.bincount(ring.ravel(),minlength=nr)*h*h)
                rmean=np.divide(rflux,rarea*1e6,out=np.full(nr,np.nan),where=rarea>0)
                ok=(rcov>=.2)&np.isfinite(rmean);index=np.arange(nr)
                profile=np.interp(index,index[ok],np.maximum(rmean[ok],0),left=0,right=0)
                reconstructed=np.where((coverage>=.5)&np.isfinite(mean),np.maximum(mean,0),profile[ring])*taper
                err=float(abs(reconstructed-a[comp+'_annular']).sum()/max(abs(a[comp+'_annular']).sum(),1e-30))
                assert err<1e-12,(name,comp,err)
                report=case['components'][comp];values={
                    'untapered_in_field_signed_integral':float(observed.sum()*h*h*1e6),
                    'signed_measured_integral':float((observed*taper).sum()*h*h*1e6),
                    'conditional_zero_integral':float(a[comp+'_zero'].sum()*h*h*1e6),
                    'conditional_annular_integral':float(a[comp+'_annular'].sum()*h*h*1e6)}
                errors={key:abs(value/report[key]-1) for key,value in values.items()}
                assert max(errors.values())<1e-12,(name,comp,errors)
                component_checks.append(dict(case=name,component=comp,measured_and_coverage_unchanged=True,independent_fill_relative_l1=err,integral_relative_errors=errors,negative_observed_cells=int((observed<0).sum())))
                if name in ['nominal','distance_low','distance_high']:saved[(name,comp)]={key:np.array(a[comp+'_'+key]) for key in ['observed','coverage','zero','annular']}
    masserrors=[]
    conv={x['id']:x for x in config['mass_conversion_cases']}
    for row in summary['mass_cases']:
        c=cases[row['case']]['components'];factor=conv[row['conversion']];key='conditional_'+row['fill']+'_integral'
        expected=c['stellar_luminosity'][key]*factor['stellar_ml']+c['atomic_helium'][key]+c['co21'][key]*factor['alpha_co10']/factor['r21']
        err=abs(expected/row['total_msun']-1);assert err<1e-12;masserrors.append(err)
    distance=[]
    for name in ['distance_low','distance_high']:
        scale=cases[name]['geometry']['distance_mpc']/cases['nominal']['geometry']['distance_mpc']
        for comp in COMPONENTS:
            for key in ['input_signed_integral','untapered_in_field_signed_integral','signed_measured_integral','conditional_zero_integral','conditional_annular_integral']:
                err=abs(cases[name]['components'][comp][key]/cases['nominal']['components'][comp][key]/scale**2-1)
                assert err<1e-10,(name,comp,key,err)
                distance.append(dict(case=name,component=comp,quantity=key,relative_error=err))
            for key in ['observed','coverage','zero','annular']:
                a=saved[('nominal',comp)][key];b=saved[(name,comp)][key]
                err=float(abs(a-b).sum()/max(abs(a).sum(),1e-30));assert err<1e-10
        for row in [r for r in summary['mass_cases'] if r['case']==name]:
            nominal=next(r for r in summary['mass_cases'] if r['case']=='nominal' and r['fill']==row['fill'] and r['conversion']==row['conversion'])
            err=abs(row['total_msun']/nominal['total_msun']/scale**2-1);assert err<1e-10
            distance.append(dict(case=name,component=row['conversion']+'_'+row['fill'],quantity='total_msun',relative_error=err))
    annuli=list(csv.DictReader((P/'run-001/source-annuli.csv').open(encoding='utf-8')));convergence=[]
    for comp in COMPONENTS:
        for case in ['pixels_one','pixels_two']:
            get=lambda name:[r for r in annuli if r['case']==name and r['component']==comp and float(r['radius_kpc'])<28]
            a=[];b=[];w=[]
            for u,v in zip(get('nominal'),get(case)):
                if u['signed_mean'] and v['signed_mean'] and u['accepted']=='True' and v['accepted']=='True':a.append(float(u['signed_mean']));b.append(float(v['signed_mean']));w.append(float(u['radius_kpc']))
            a,b,w=map(np.array,[a,b,w]);err=float((w*abs(a-b)).sum()/(w*abs(a)).sum())
            convergence.append(dict(component=comp,comparison=case+'_vs_four',annular_relative_l1=err,over_frozen_3percent_flag=err>.03,matched_annuli=len(a)))
    oldmax=read(OLD/'findings-002/distance-scaling-counterexample.json')['maximum_total_mass_relative_error']
    result=dict(status='DISCRETIZATION_BLOCKER_RESOLVED_SOURCE_ONLY',packets=12,component_cases=36,mass_rows=72,observed_arrays_unchanged=True,old_maximum_total_mass_distance_error=oldmax,new_maximum_distance_error=max(r['relative_error'] for r in distance),maximum_independent_fill_relative_l1=max(r['independent_fill_relative_l1'] for r in component_checks),maximum_integral_relative_error=max(max(r['integral_relative_errors'].values()) for r in component_checks),mass_replay_max_error=max(masserrors),component_checks=component_checks,distance_checks=distance,pixel_convergence=convergence,full_3d_source_admitted=False,new_gravity_scores=0,observed_response_arrays_opened=0,packet_bindings=bindings)
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['component_checks','distance_checks','packet_bindings']},indent=2))

if __name__=='__main__':run()
