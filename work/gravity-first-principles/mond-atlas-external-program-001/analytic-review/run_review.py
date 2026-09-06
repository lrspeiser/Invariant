"""Independent dielectric-sphere interface and freely-falling-frame checks."""
import hashlib,json
from pathlib import Path
import numpy as np

OWN=Path(__file__).resolve().parent


def coefficients(ein,eout):return 3*eout/(ein+2*eout),(ein-eout)/(ein+2*eout)


def potential(x,E=1.,ein=1.,eout=.2,a=1.,side=None):
    x=np.asarray(x,float);r=np.linalg.norm(x);A,B=coefficients(ein,eout)
    inside=(r<a) if side is None else side=='inside'
    return -A*E*x[2] if inside else -E*x[2]+B*E*a**3*x[2]/r**3


def acceleration(x,E=1.,ein=1.,eout=.2,a=1.,side=None):
    x=np.asarray(x,float);r=np.linalg.norm(x);A,B=coefficients(ein,eout);z=np.array([0.,0.,1.])
    inside=(r<a) if side is None else side=='inside'
    return A*E*z if inside else E*z+B*E*a**3*(3*x[2]*x/r**5-z/r**3)


def run():
    if (OWN/'results.json').exists():raise RuntimeError('Immutable existing result')
    checks=[];A,B=coefficients(1,.2)
    independent=np.linalg.solve(np.array([[1,1],[1,-.4]]),[1,.2]);assert np.max(abs(independent-[A,B]))<1e-12
    maximum=dict(potential=0.,normal_flux=0.,tangential_field=0.)
    for theta in [0,.1,.4,.8,1.2,np.pi/2,2,2.6,np.pi]:
        for phi in [0,.7,2.1]:
            n=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            for E in [-2,0,.3,1,5]:
                pi=potential(n,E,side='inside');po=potential(n,E,side='outside');gi=acceleration(n,E,side='inside');go=acceleration(n,E,side='outside')
                errors=dict(potential=abs(pi-po),normal_flux=abs(gi@n-.2*(go@n)),tangential_field=float(np.linalg.norm((gi-(gi@n)*n)-(go-(go@n)*n))))
                for k,v in errors.items():maximum[k]=max(maximum[k],float(v))
                assert max(errors.values())<1e-12
                checks.append(dict(theta=theta,phi=phi,E=E,**errors))
    gradient_errors=[];laplacians=[];linear=[];uniform=[]
    for x in [[.1,.2,.3],[-.2,.1,-.5],[1.5,.4,.8],[2.,-1.,-.3],[0.,0.,3.]]:
        x=np.array(x);h=2e-4
        gradient=np.array([(potential(x+h*axis)-potential(x-h*axis))/(2*h) for axis in np.eye(3)])
        lap=sum(potential(x+h*axis)-2*potential(x)+potential(x-h*axis) for axis in np.eye(3))/h**2
        err=float(np.max(abs(gradient+acceleration(x))));assert err<1e-6 and abs(lap)<1e-6
        gradient_errors.append(err);laplacians.append(float(lap))
        for E in [-2,0,.3,1,5]:
            linear.append(float(np.max(abs(acceleration(x,E)-E*acceleration(x)))))
            uniform.append(float(np.max(abs(acceleration(x,E,ein=.7,eout=.7)-[0,0,E]))))
    assert max(linear+uniform)<1e-12
    origin=acceleration([0,0,0]);relative=[]
    for p in [[0,0,0],[.1,.2,.3],[0,0,.8],[0,0,2.],[2.,0,0]]:
        g=acceleration(p);relative.append(dict(point=p,field=g.tolist(),relative_to_center=(g-origin).tolist(),relative_to_applied_background=(g-[0,0,1]).tolist()))
    result=dict(status='THEORY_BENCHMARK_ONLY',eps_in=1,eps_out=.2,sphere_radius=1,A=A,B=B,boundary_checks=len(checks),max_interface_errors=maximum,max_potential_gradient_error=max(gradient_errors),max_abs_laplacian=max(map(abs,laplacians)),max_linearity_error=max(linear),max_uniform_medium_error=max(uniform),freefall_examples=relative,bindings={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OWN/'PREFLIGHT.md']})
    (OWN/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
