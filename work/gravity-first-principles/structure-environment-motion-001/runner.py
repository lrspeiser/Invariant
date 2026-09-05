"""First-round controlled mechanisms: structure, environment, and relative motion."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
base=Path(__file__).parent; root=base/'Invariant'
dest=root/'work/gravity-first-principles/structure-environment-motion-001'
dest.mkdir(exist_ok=False)
registration=dict(units='G=1, reference source mass=1, length unit=1; illustrative parameters, no astronomical calibration.',
    static_potential='U_ij=-m_i m_j/d [1+sum alpha (1-exp(-d/L))].',
    static_cards={'Newton':[], 'short':[[1.,.3]], 'long':[[1.,3.]], 'two_ranges':[[.5,.3],[.5,3.]]},
    structure='Same total point-clump mass, center of mass and outer containment radius; instantaneous snapshots, not equilibrium galaxies.',
    motion_lagrangian='L_mech=sum m_i v_i^2/2 - U_Newton + eta/2 sum_(i<j) (m_i m_j/Mref) exp(-d/Lv) |v_i-v_j|^2; Mref=1, Lv=1.',
    motion_scope='An explicitly specified new conservative toy mechanics, not the earlier static filtered action. No light, relativistic, or observational admission.',
    motion_eta=[0.,.1,1.],numerical_tolerance=1e-8,
    real_data='Only previously exposed per-galaxy SPARC scores and baryonic descriptors; descriptive strata, no new fit.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())

def acceleration(probe,pos,mass,card):
    delta=pos-np.array(probe); d=np.linalg.norm(delta,axis=1)
    factor=np.ones_like(d)
    for alpha,L in card:
        u=d/L
        factor+=alpha*(-np.expm1(-u)-u*np.exp(-u))
    return np.sum((mass*factor/d**3)[:,None]*delta,axis=0)

def shell(R,n):
    mu,w=roots_legendre(n); phi=np.arange(2*n)*np.pi/n
    xy=np.sqrt(1-mu[:,None]**2)
    pos=np.stack(np.broadcast_arrays(R*xy*np.cos(phi),R*xy*np.sin(phi),R*mu[:,None]),axis=-1).reshape(-1,3)
    mass=np.broadcast_to(w[:,None]/(4*n),(n,2*n)).reshape(-1)
    return pos,mass

theta=np.arange(16)*2*np.pi/16
layouts={'central':(np.zeros((1,3)),np.ones(1)),
         'two_clumps':(np.array([[-.7,0,0],[.7,0,0]]),np.ones(2)/2),
         'ring':(np.column_stack([.7*np.cos(theta),.7*np.sin(theta),np.zeros(16)]),np.ones(16)/16),
         'spherical_shell':shell(.7,64)}
structure=[]
split_error=0.
for name,(pos,mass) in layouts.items():
    for radius in [1.2,2.,4.,8.]:
        for angle in [0.,np.pi/4]:
            unit=np.array([np.cos(angle),np.sin(angle),0.]); probe=radius*unit
            for model,card in registration['static_cards'].items():
                acc=acceleration(probe,pos,mass,card)
                split=acceleration(probe,np.repeat(pos,2,axis=0),np.repeat(mass/2,2),card)
                split_error=max(split_error,float(np.linalg.norm(acc-split)/max(np.linalg.norm(acc),1e-20)))
                inward=-float(acc@unit); transverse=float(np.linalg.norm(acc+inward*unit))
                structure.append(dict(layout=name,r=radius,angle=float(angle),model=model,
                    inward=inward,transverse=transverse,newton_point_ratio=inward*radius**2))

# Exact spherical shell environment, tested against numerical integration.
environment=[]
for n in [32,64]:
    pos,mass=shell(2.,n)
    for r in [.25,.5,1.]:
        for shell_mass in [1.,10.]:
            for model,card in registration['static_cards'].items():
                measured=-acceleration([r,0,0],pos,mass*shell_mass,card)[0]
                exact=0.
                for alpha,L in card:
                    z=r/L
                    exact+=alpha*shell_mass*np.exp(-2/L)/2*(z*np.cosh(z)-np.sinh(z))/(L*z*z)
                central=-acceleration([r,0,0],np.zeros((1,3)),np.ones(1),card)[0]
                environment.append(dict(nodes=n,r=r,shell_mass=shell_mass,model=model,
                    shell_inward=float(measured),exact=float(exact),absolute_error=float(abs(measured-exact)),
                    fraction_of_central_pull=float(measured/central)))

# Change radial placement while preserving shell mass and spherical symmetry.
spillover=[]
for model,card in registration['static_cards'].items():
    for r in [1.2,2.,4.,8.]:
        values=[]
        for R in [.3,.8]:
            pos,mass=shell(R,64)
            values.append(-acceleration([r,0,0],pos,mass,card)[0])
        spillover.append(dict(model=model,r=r,change=float(values[1]-values[0]),
            change_over_newton_point=float((values[1]-values[0])*r*r)))

def mechanics(pos,vel,mass,eta):
    n=len(mass); A=np.diag(mass.copy()); rhs=np.zeros_like(pos)
    kinetic=.5*np.sum(mass[:,None]*vel*vel); potential=0.
    for i in range(n):
        for j in range(i):
            delta=pos[i]-pos[j]; d=np.linalg.norm(delta); unit=delta/d
            v=vel[i]-vel[j]; c=eta*mass[i]*mass[j]; weight=np.exp(-d)
            grad=-weight*unit; dw=grad@v
            F=-mass[i]*mass[j]*unit/d**2
            motion=.5*c*grad*(v@v)-c*dw*v
            rhs[i]+=F+motion; rhs[j]-=F+motion
            A[i,i]+=c*weight; A[j,j]+=c*weight; A[i,j]-=c*weight; A[j,i]-=c*weight
            kinetic+=.5*c*weight*(v@v); potential-=mass[i]*mass[j]/d
    return np.linalg.solve(A,rhs),float(kinetic+potential),float(np.linalg.eigvalsh(A).min())

angle=np.arange(8)*2*np.pi/8
source_pos=np.column_stack([.7*np.cos(angle),.7*np.sin(angle),np.zeros(8)])
tangent=np.column_stack([-np.sin(angle),np.cos(angle),np.zeros(8)])
motions={'co_rotation':.5*tangent,'counter_rotation':.5*tangent*((-1.)**np.arange(8))[:,None],
         'alternating_radial':.5*source_pos/.7*((-1.)**np.arange(8))[:,None]}
motion=[]
for name,source_vel in motions.items():
    for tracer_speed in [-.5,0.,.5]:
        pos=np.vstack([source_pos,[3.,0,0]]); vel=np.vstack([source_vel,[0,tracer_speed,0]])
        mass=np.array([1/8]*8+[1e-5])
        for eta in registration['motion_eta']:
            acc,energy,eigenvalue=mechanics(pos,vel,mass,eta)
            boosted=mechanics(pos,vel+np.array([.7,-.2,.3]),mass,eta)[0]
            momentum=np.sum(mass[:,None]*acc,axis=0)
            checks=[]
            for dt in [1e-4,1e-5]:
                ep=mechanics(pos+dt*vel,vel+dt*acc,mass,eta)[1]
                em=mechanics(pos-dt*vel,vel-dt*acc,mass,eta)[1]
                checks.append(dict(step=dt,energy_derivative=(ep-em)/(2*dt)))
            motion.append(dict(motion=name,tracer_speed=tracer_speed,eta=eta,
                source_kinetic_energy=float(.5*np.sum((1/8)*source_vel**2)),
                tracer_acceleration=acc[-1].tolist(),total_force_norm=float(np.linalg.norm(momentum)),
                uniform_boost_error=float(np.max(abs(acc-boosted))),minimum_inertia_eigenvalue=eigenvalue,
                energy_checks=checks))

# Exposed development descriptors. No confirmation response rows accessed.
p=root/'runs/gravity/g4/conditional-formula-generator-v4.json'
historical=json.loads(p.read_text()); strata=[]
for feature in ['surface_density','gas_dominance','bulge_dominance','baryonic_compactness']:
    rows=sorted(historical['galaxies'],key=lambda x:float(x['generated_formula']['condition_values'][feature]))
    for label,subset in [('lowest_quarter',rows[:35]),('highest_quarter',rows[-35:])]:
        gains=[1-float(x['candidate_score']['chi_square'])/float(x['rar_score']['chi_square']) for x in subset]
        strata.append(dict(feature=feature,group=label,n=len(subset),improved=sum(x>0 for x in gains),
            median_fractional_gain=float(np.median(gains)),
            feature_range=[float(subset[0]['generated_formula']['condition_values'][feature]),float(subset[-1]['generated_formula']['condition_values'][feature])]))

out=dict(registration=registration,structure=structure,environment=environment,spillover=spillover,motion=motion,
    historical_focusing_strata=strata,historical_receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    controls=dict(source_splitting_error=split_error,
        shell_exact_max_absolute_error=max(x['absolute_error'] for x in environment),
        maximum_total_force_norm=max(x['total_force_norm'] for x in motion),
        maximum_boost_error=max(x['uniform_boost_error'] for x in motion),
        maximum_fine_energy_derivative=max(abs(x['energy_checks'][-1]['energy_derivative']) for x in motion)),
    new_observational_fits=0,admitted_laws=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out['controls'],indent=2))
print('Historical strata',json.dumps(strata))
