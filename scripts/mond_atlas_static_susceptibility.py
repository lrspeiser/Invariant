"""Minimal counterterm removal with conservative mechanical controls."""
from pathlib import Path
import itertools,json,hashlib,csv,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-static-susceptibility-001'
ETA=.15; B=.05; G=1.; L=1.
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def calc(x,v,m,omega,q=None,w=None,eta=ETA):
    pairs=np.array(list(itertools.combinations(range(len(m)),2)));i,j=pairs.T
    d=x[i]-x[j];r2=np.sum(d*d,axis=1);mm=m[i]*m[j]
    h=eta*np.sqrt(mm)*np.exp(-r2/(2*L*L))
    if q is None:q=h.copy()
    if w is None:w=np.zeros_like(q)
    grad=-h[:,None]*d/L**2
    f=-G*mm[:,None]*d/(r2+B*B)[:,None]**1.5+omega**2*q[:,None]*grad
    force=np.zeros_like(x);np.add.at(force,i,f);np.add.at(force,j,-f)
    vn=-G*np.sum(mm/np.sqrt(r2+B*B));osc=.5*np.sum(w*w+omega**2*q*q)-omega**2*np.sum(q*h)
    energy=.5*np.sum(m[:,None]*v*v)+vn+osc
    momentum=np.sum(m[:,None]*v,axis=0);angular=np.cross(x,m[:,None]*v).sum(axis=0)
    return force/m[:,None],float(energy),momentum,angular,-omega**2*(q-h),h,float(osc)

def rhs(m,omega):
    n=len(m);k=n*(n-1)//2
    def fun(t,z):
        x=z[:3*n].reshape(n,3);v=z[3*n:6*n].reshape(n,3);q=z[6*n:6*n+k];w=z[6*n+k:]
        a,e,p,j,dd,h,osc=calc(x,v,m,omega,q,w)
        return np.r_[v.ravel(),a.ravel(),w,dd]
    return fun

def controls():
    rng=np.random.default_rng(9062901);rows=[]
    x=rng.normal(size=(4,3));v=rng.normal(size=(4,3))*.1;m=np.array([.4,.7,1.,2.])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    for omega in [.2,2.]:
        h=calc(x,v,m,omega)[5];q=h+.03*np.sqrt(np.array([m[i]*m[j] for i,j in itertools.combinations(range(4),2)]));w=.01*np.ones(6)
        a,e,p,j,dd,h,osc=calc(x,v,m,omega,q,w);z=np.r_[x.ravel(),v.ravel(),q,w];dz=rhs(m,omega)(0,z)
        def energy(state):return calc(state[:12].reshape(4,3),state[12:24].reshape(4,3),m,omega,state[24:30],state[30:])[1]
        de=abs((energy(z+1e-5*dz)-energy(z-1e-5*dz))/2e-5)/max(1,abs(e))
        forceerror=0.
        for k in range(12):
            delta=np.zeros_like(z);delta[k]=1e-5
            numerical=-(energy(z+delta)-energy(z-delta))/2e-5
            forceerror=max(forceerror,abs(numerical-(a*m[:,None]).ravel()[k]))
        rotated=calc(x@rotation.T+[1,2,3],v@rotation.T,m,omega,q,w)[0]
        boosted=calc(x,v+[1,-2,3],m,omega,q,w)[0]
        cov=max(float(np.max(abs(rotated-a@rotation.T))),float(np.max(abs(boosted-a))))
        totalforce=float(np.linalg.norm(np.sum(m[:,None]*a,axis=0)))
        minimum_errors=[];curvature=[]
        for r in [0,.1,1,3]:
            hh=ETA*np.sqrt(.4*2)*np.exp(-r*r/2)
            solution=minimize_scalar(lambda qq:.5*omega**2*qq*qq-omega**2*qq*hh,bracket=(-1,1),method='brent',options={'xtol':1e-13})
            minimum_errors.append(abs(solution.fun+.5*omega**2*hh*hh));curvature.append(omega**2)
        # Manufacture coincident children, mapping cross-source oscillator states.
        f=np.tile([.3,.7],4);parent=np.repeat(np.arange(4),2);ms=np.repeat(m,2)*f
        xs=np.repeat(x,2,axis=0);vs=np.repeat(v,2,axis=0)
        pairindex={pair:k for k,pair in enumerate(itertools.combinations(range(4),2))}
        qs=[];ws=[];internal=0.
        for i,jj in itertools.combinations(range(8),2):
            pi,pj=parent[i],parent[jj]
            if pi==pj:
                qs.append(ETA*np.sqrt(ms[i]*ms[jj]));ws.append(0.)
                internal+=-G*ms[i]*ms[jj]/B-.5*omega**2*ETA**2*ms[i]*ms[jj]
            else:
                k=pairindex[(int(pi),int(pj))];factor=np.sqrt(f[i]*f[jj])
                qs.append(q[k]*factor);ws.append(w[k]*factor)
        aa,ee,*_=calc(xs,vs,ms,omega,np.array(qs),np.array(ws))
        splitforce=float(np.max(abs(aa-np.repeat(a,2,axis=0))));splitenergy=abs(ee-internal-e)
        lower=-(G/B+.5*omega**2*ETA**2)*sum(m[i]*m[j] for i,j in itertools.combinations(range(4),2))
        assert e>=lower
        zero=calc(x,v,m,omega,np.zeros(6),np.zeros(6),eta=0)[0]
        newton=calc(x,v,m,0)[0];zerodiff=float(np.max(abs(zero-newton)))
        row=dict(omega=omega,energy_directional_error=de,finite_difference_force_error=forceerror,
            covariance_error=cov,total_force=totalforce,scalar_minimum_error=max(minimum_errors),
            oscillator_curvature_min=min(curvature),split_acceleration_error=splitforce,split_energy_error_after_internal_constant=splitenergy,
            zero_coupling_newton_error=zerodiff,energy_lower_bound=lower,manufactured_energy=e)
        row['passed']=max(de,forceerror)<1e-6 and max(cov,totalforce,splitforce,splitenergy)<1e-10 and max(minimum_errors)<1e-10 and zerodiff<1e-12
        assert row['passed'];rows.append(row)
        t=np.linspace(0,40,401);hh=.3
        sol=solve_ivp(lambda tt,y:[y[1],-omega**2*(y[0]-hh)],[0,40],[0,0],t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13)
        error=float(np.max(abs(sol.y[0]-hh*(1-np.cos(omega*t)))))
        ee=.5*(sol.y[1]**2+omega**2*sol.y[0]**2)-omega**2*hh*sol.y[0]
        assert error<1e-8 and max(abs(ee))<1e-8
        rows.append(dict(control='fixed_radius_oscillator',omega=omega,coordinate_error=error,energy_error=float(max(abs(ee))),settles_to_equilibrium=False))
    return rows

def static_tests():
    rows=[]
    for omega in [.2,2.]:
        for r in [1e-4,.01,.1,1,np.sqrt(1.5),2,5,10]:
            aextra=omega**2*ETA**2*r*np.exp(-r*r)
            anewton=r/(r*r+B*B)**1.5
            rows.append(dict(omega=omega,r=r,extra_accel_per_unit_source_mass=aextra,
                softened_newton_accel_per_unit_source_mass=anewton,extra_over_softened_newton=aextra/anewton,
                extra_over_unsoftened_newton=aextra*r*r))
    masschecks=[dict(source_mass=m,newton_accel=m/(1+B*B)**1.5,
         extra_accel=4*ETA**2*m*np.exp(-1),fraction=4*ETA**2*np.exp(-1)*(1+B*B)**1.5) for m in [1,1e3,1e6]]
    negative=dict(inverse_radius_h=[dict(r=r,Ueq=-.5*4*ETA**2/r**2) for r in [1,.1,.01,.001]],
        linear_radius_h=[dict(r=r,Ueq=-.5*4*ETA**2*r**2,outward_force=4*ETA**2*r) for r in [1,2,4,8,16]],
        naive_mass_independent_h_equal_split_cross_energy_ratio=4.,
        fixed_particle_mass_coincident_energy_scaling=[dict(N=n,pair_count=n*(n-1)//2,gaussian_energy=-.5*4*ETA**2*n*(n-1)/2) for n in [2,4,8,16]])
    return dict(static_rows=rows,high_acceleration_mass_scaling=masschecks,
        high_acceleration_universal_newton_limit_pass=False,extended_flat_halo_asymptote_pass=False,
        fixed_total_mass_bounded=True,thermodynamic_extensivity=False,negative_controls=negative)

def run():
    out=P/'run001';out.mkdir(exist_ok=False);start=time.time()
    old=ROOT/'work/gravity-first-principles/mond-atlas-dynamic-program-001'
    sources=[Path(__file__),P/'PREFLIGHT.md',ROOT/'scripts/mond_atlas_dynamic_program.py',ROOT/'scripts/mond_atlas_dynamic_partition.py',ROOT/'scripts/mond_atlas_dynamic_partition_strength.py',
        old/'run001/results.json',old/'short-time-results.json',old/'partition-repair/run001/results.json',old/'partition-repair/strength-sweep/results.json',old/'audit-and-inventory.json']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sources};save(out/'bindings.json',bound)
    save(out/'pre-integration-controls.json',controls());save(out/'static-susceptibility.json',static_tests())
    rows=[];series=[];m=np.array([1.,.1]);n=2
    for omega,r,kind in itertools.product([.2,2.],[1.,2.],['circular','perturbed']):
        x=np.array([[-m[1]/m.sum()*r,0,0],[m[0]/m.sum()*r,0,0]])
        vc=np.sqrt(m.sum()*r*r*((r*r+B*B)**-1.5+omega**2*ETA**2*np.exp(-r*r)))
        v=np.array([[0,-m[1]/m.sum()*vc,0],[0,m[0]/m.sum()*vc,0]])
        if kind=='perturbed':v*=.97
        h=calc(x,v,m,omega)[5];z=np.r_[x.ravel(),v.ravel(),h,np.zeros(1)];sols=[]
        for resolution,step,rtol,atol in [('base',.1,1e-8,1e-10),('fine',.05,1e-10,1e-12)]:
            sol=solve_ivp(rhs(m,omega),[0,40],z,t_eval=np.linspace(0,40,401),method='DOP853',max_step=step,rtol=rtol,atol=atol)
            assert sol.success;energy=[];mom=[];ang=[];radii=[];oscs=[]
            exacterror=0.
            for t,y in zip(sol.t,sol.y.T):
                xx=y[:6].reshape(2,3);vv=y[6:12].reshape(2,3);a,e,p,j,dd,hh,osc=calc(xx,vv,m,omega,y[12:13],y[13:])
                energy.append(e);mom.append(p);ang.append(j);rad=float(np.linalg.norm(xx[1]-xx[0]));radii.append(rad);oscs.append(osc)
                if kind=='circular':
                    theta=vc/r*t;rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
                    exacterror=max(exacterror,float(np.max(abs(xx-x@rot.T))))
                if resolution=='fine':series.append(dict(omega=omega,r0=r,kind=kind,time=float(t),radius=rad,energy=e,internal_signed_energy=osc,q=float(y[12]),q_equilibrium=float(hh[0])))
            ed=float(np.max(abs(np.array(energy)-energy[0]))/max(1,abs(energy[0])))
            pd=float(np.max(np.linalg.norm(np.array(mom)-mom[0],axis=1)))
            jd=float(np.max(np.linalg.norm(np.array(ang)-ang[0],axis=1)))
            row=dict(omega=omega,r0=r,kind=kind,resolution=resolution,energy_drift=ed,momentum_drift=pd,angular_drift=jd,
                min_radius=min(radii),max_radius=max(radii),initial_energy=energy[0],final_energy=energy[-1],
                internal_energy_min=min(oscs),internal_energy_max=max(oscs),conservation_pass=max(ed,pd,jd)<1e-5,
                exact_circular_error=exacterror if kind=='circular' else None,
                exact_circular_pass=exacterror<1e-6 if kind=='circular' else None)
            rows.append(row);sols.append(sol)
        diff=float(np.linalg.norm(sols[0].y[:6,-1]-sols[1].y[:6,-1])/max(1,np.linalg.norm(sols[1].y[:6,-1])))
        rows[-1].update(trajectory_difference=diff,trajectory_pass=diff<1e-3)
        print(omega,r,kind,'energy',rows[-1]['energy_drift'],'trajectory',diff,flush=True)
    with (out/'timeseries.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=series[0]);w.writeheader();w.writerows(series)
    assert all(sha(ROOT/k)==v for k,v in bound.items())
    save(out/'results.json',dict(status='THEORY_BENCHMARK_ONLY',integrations=len(rows),seconds=time.time()-start,results=rows,
         new_observed_scores=0,all_bound_inputs_reverified=True))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
