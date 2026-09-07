"""Theory-only local-G normalization audit; no observational inputs."""
import hashlib,json,math
from pathlib import Path
from scipy.integrate import quad
R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-normalization-001'
def eps(u):return .2+.8*u/(1+u)
def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    bindings={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'PREFLIGHT.md',R/'work/gravity-first-principles/mond-atlas-execution-025/README.md']}
    (out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    rows=[];maximum=0
    for ulab in [0,.01,.1,1,10,100]:
        for uenv in [0,.01,.1,1,10,100]:
            elab,eenv=eps(ulab),eps(uenv)
            expected=elab/eenv;previous=None
            for radius in [1,2,8]:
                # Independent spherical Gauss flux with G_measured=mass=1.
                bare_g=elab;flux=-4*math.pi*bare_g
                radial_g=flux/(4*math.pi*radius**2*eenv)
                measured_ratio=-radial_g*radius**2
                maximum=max(maximum,abs(measured_ratio/expected-1))
                assert radial_g<0
                if previous:assert abs(radial_g/previous[1]-(previous[0]/radius)**2)<1e-12
                previous=(radius,radial_g)
            if ulab==uenv:assert abs(expected-1)<1e-12
            rows.append(dict(u_lab=ulab,u_environment=uenv,epsilon_lab=elab,epsilon_environment=eenv,relative_to_measured_newton=expected))
    assert eps(0)==.2 and abs(eps(1e15)-1)<1e-12 and maximum<1e-12
    sources=[]
    for ell in [.25,.5]:
        integral,error=quad(lambda t:4*math.pi*t*t*math.exp(-t*t/2)/(2*math.pi)**1.5,0,math.inf,epsabs=1e-12)
        assert abs(integral-1)<1e-10
        u=1/((2*math.pi)**1.5*ell**3*1e7)
        sources.append(dict(ell_kpc=ell,source_mass_msun=1,peak_u=u,maximum_epsilon_increment=.8*u/(1+u),gaussian_integral=integral,quadrature_error=error))
    result=dict(status='PASS_THEORY_NORMALIZATION_CONTROLS',observational_tests=0,maximum_relative_flux_error=maximum,constant_environment_cases=rows,isolated_source_bounds=sources,physical_G_calibration_resolved=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(maximum_relative_flux_error=maximum,isolated_source_bounds=sources)))
if __name__=='__main__':run()
