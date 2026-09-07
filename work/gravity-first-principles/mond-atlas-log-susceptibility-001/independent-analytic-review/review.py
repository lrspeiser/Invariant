"""Independent differentiation/quadrature/completed-square checks."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import quad
P=Path(__file__).resolve().parent
G,M,L,b,C=1.,2.3,1.7,.05,.021
def potential(r):return -C*M*np.log1p(L/np.hypot(r,b))
def field(r):
    s=np.hypot(r,b)
    return C*M*L*r/(s*s*(s+L))
def density(r):
    s=np.hypot(r,b)
    return C*M*L/(4*np.pi*G)*(L*s*s+b*b*(3*s+2*L))/(s**4*(s+L)**2)
force_error=0.;mass_error=0.
for r in np.geomspace(.001,100,31):
    d=r*1e-4
    derivative=(potential(r-2*d)-8*potential(r-d)+8*potential(r+d)-potential(r+2*d))/(12*d)
    force_error=max(force_error,abs(derivative-field(r)))
    integrated=quad(lambda x:4*math.pi*x*x*density(x),0,r,epsabs=1e-13,epsrel=1e-12)[0]
    mass_error=max(mass_error,abs(integrated-r*r*field(r)/G))
    assert density(r)>0
total,error=quad(lambda r:4*math.pi*r*r*density(r),0,np.inf,epsabs=1e-13,epsrel=1e-12)
expected=C*M*L/G
assert force_error<1e-8 and mass_error<1e-10 and abs(total/expected-1)<1e-10
# Jaffe convention rho=M_eff*L/[4*pi*r²*(r+L)²].
jaffe=quad(lambda r:expected*L/(r+L)**2,0,np.inf,epsabs=1e-13,epsrel=1e-12)[0]
assert abs(jaffe/expected-1)<1e-12
central=3*C*M*L/(4*np.pi*G*b*b*(b+L))
assert abs(density(1e-9)/central-1)<1e-12
assert abs(field(1e9)*1e18/(C*M*L)-1)<1e-8
# Complete the square with independent random canonical states.
rng=np.random.default_rng(6029);square_error=0
for _ in range(100):
    mi,mj=rng.uniform(.1,3,2);r=rng.uniform(0,10);omega=.2;eta=.15;q,p=rng.normal(size=2)
    s=np.hypot(r,b);h=eta*np.sqrt(mi*mj*np.log1p(L/s));cc=omega**2*eta**2/2
    original=p*p/2+omega**2*q*q/2-omega**2*q*h-G*mi*mj/s
    completed=p*p/2+omega**2*(q-h)**2/2-cc*mi*mj*np.log1p(L/s)-G*mi*mj/s
    square_error=max(square_error,abs(original-completed))
    bound=-(G/b+cc*np.log1p(L/b))*mi*mj
    assert original>=bound-1e-12
assert square_error<1e-12
result=dict(status='PASS_INDEPENDENT_ANALYTIC_REVIEW',force_absolute_error=force_error,enclosed_mass_absolute_error=mass_error,total_mass=total,expected_total_mass=expected,total_mass_quadrature_error=error,jaffe_mass=jaffe,completed_square_error=square_error,observational_tests=0,protocol_sha256=hashlib.sha256((P.parent/'PREFLIGHT.md').read_bytes()).hexdigest())
(P/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
