"""Evaluate a published gas profile; incomplete, conditional G calibration only."""
import hashlib,json,math,urllib.request
from pathlib import Path
from scipy.integrate import quad
from scipy.special import erfcx
from astropy.constants import m_p,M_sun,pc
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-local-gas-calibration-001'
PROFILES=[('H2',.15,105,'gaussian',.9),('CNM',.8,127,'gaussian',5.54),('WNM1',.13,318,'gaussian',2.24),('WNM2',.077,403,'exponential',1.91),('HII',.0154,1590,'exponential',1.6)]
def analytic(n,h,kind,ell):return n*h/math.sqrt(h*h+2*ell*ell) if kind=='gaussian' else n*erfcx(ell/(math.sqrt(2)*h))
def shape(z,h,kind):return math.exp(-(z/h)**2) if kind=='gaussian' else math.exp(-abs(z)/h)
def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    url='https://arxiv.org/html/1509.05334v1';raw=urllib.request.urlopen(url,timeout=40).read()
    private=R/'work/private/mond-atlas-local-gas-calibration-001';private.mkdir(exist_ok=False);(private/'paper.html').write_bytes(raw)
    bind={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'PREFLIGHT.md']}
    (out/'bindings.json').write_text(json.dumps(dict(files=bind,paper_url=url,paper_sha256=hashlib.sha256(raw).hexdigest(),paper_bytes=len(raw)),indent=2)+'\n')
    # m_p approximates hydrogen mass here. Electron/binding corrections are below0.06%; recorded explicitly.
    factor=1.4*m_p.value*1e6*pc.value**3/M_sun.value
    rows=[];maxerr=0;limits=[];columns=[]
    for name,n,h,kind,reference in PROFILES:
        col=n*h*(math.sqrt(math.pi) if kind=='gaussian' else 2)
        actual_column=col*pc.value*100/1e20
        columns.append(dict(component=name,column_1e20=actual_column,table_column_1e20=reference,relative_difference=actual_column/reference-1,table_HII_includes_extra_gum_component=name=='HII'))
        small=analytic(n,h,kind,1e-5);large=analytic(n,h,kind,1e7);asymptotic=col/(math.sqrt(2*math.pi)*1e7)
        assert abs(small/n-1)<1e-7 and abs(large/asymptotic-1)<1e-7
        limits.append(dict(component=name,small_relative_error=small/n-1,large_relative_error=large/asymptotic-1))
        for ell in [250,500]:
            a=analytic(n,h,kind,ell)
            independent,error=quad(lambda t:2*n*shape(ell*t,h,kind)*math.exp(-t*t/2)/math.sqrt(2*math.pi),0,math.inf,epsabs=1e-13,epsrel=1e-12)
            relative=abs(independent/a-1);maxerr=max(maxerr,relative);assert relative<1e-10 and 0<a<=n
            rows.append(dict(component=name,kind=kind,midplane_nH_cm3=n,height_pc=h,ell_pc=ell,smoothed_nH_cm3=a,smoothed_rho_msun_pc3=a*factor,independent_integral=independent,quadrature_error=error))
        assert analytic(n,h,kind,500)<analytic(n,h,kind,250)
    totals=[]
    for ell in [250,500]:
        rho=sum(v['smoothed_rho_msun_pc3'] for v in rows if v['ell_pc']==ell);u=rho/.01
        totals.append(dict(ell_pc=ell,gas_model_rho_msun_pc3=rho,u=u,epsilon_gas_only=.2+.8*u/(1+u),actual_local_G_calibration=False))
    result=dict(status='CONDITIONAL_GAS_PROFILE_CONVOLUTION_ONLY',observed_motion_scores=0,hydrogen_mass_approximation='proton mass; no electron correction',rho_conversion_factor=factor,rows=rows,totals=totals,column_checks=columns,limit_checks=limits,max_independent_relative_error=maxerr,full_stellar_table_admitted=False,spatial_local_inhomogeneity_modeled=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(totals))
if __name__=='__main__':run()
