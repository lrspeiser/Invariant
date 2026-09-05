"""Additional asymmetric mechanical controls and user-facing first-round report."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
base=Path(__file__).parent; root=base/'Invariant'; outputs=base.parent/'outputs'
dest=root/'work/gravity-first-principles/structure-environment-motion-001'
d=json.loads((dest/'result.json').read_text())
tree=ast.parse((dest/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'frozen_first_round_functions','exec'))
rng=np.random.default_rng(731)
pos=rng.normal(size=(6,3)); vel=.2*rng.normal(size=(6,3)); mass=np.arange(1,7)/21
angle=.47; R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
mechanical_checks=[]
for eta in [0.,.1,1.]:
    acc,E,minimum=mechanics(pos,vel,mass,eta)
    rotated=mechanics(pos@R.T+np.array([1.,-2.,.7]),vel@R.T,mass,eta)[0]
    errors=[]
    for dt in [1e-4,1e-5,1e-6]:
        ep=mechanics(pos+dt*vel,vel+dt*acc,mass,eta)[1]
        em=mechanics(pos-dt*vel,vel-dt*acc,mass,eta)[1]
        errors.append(dict(dt=dt,energy_rate=(ep-em)/(2*dt)))
    mechanical_checks.append(dict(eta=eta,energy_checks=errors,
        covariance_error=float(np.max(abs(rotated-acc@R.T))),
        total_force_norm=float(np.linalg.norm(np.sum(mass[:,None]*acc,axis=0))),minimum_inertia_eigenvalue=minimum))

# Nested shells have the same mass and outer radius as the outer-shell control.
p1,m1=shell(.3,64); p2,m2=shell(.8,64)
nested=[]
for model,card in d['registration']['static_cards'].items():
    for r in [1.2,2.,4.]:
        single=-acceleration([r,0,0],p2,m2,card)[0]
        multi=-acceleration([r,0,0],np.vstack([p1,p2]),np.concatenate([m1/2,m2/2]),card)[0]
        nested.append(dict(model=model,r=r,single_shell_force=float(single),nested_force=float(multi),
            fractional_change=float((multi-single)/single)))
extra=dict(asymmetric_mechanical_controls=mechanical_checks,nested_shells=nested,
    description='Supplemental tests after the initial snapshots; fixed seeds and exact unchanged formulas. Numerical controls are not a relativistic or global stability proof.')
(dest/'supplement.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
(dest/'supplement_runner.py').write_bytes(Path(__file__).read_bytes())
assert max(x['covariance_error'] for x in mechanical_checks)<1e-8
assert max(abs(x['energy_checks'][1]['energy_rate']) for x in mechanical_checks)<1e-8
assert all(x['minimum_inertia_eigenvalue']>0 for x in mechanical_checks)
assert max(d['controls'].values())<1e-8

baseline=next(-x['tracer_acceleration'][0] for x in d['motion'] if x['eta']==0 and x['tracer_speed']==0)
motion_table=[]
for name in ['co_rotation','counter_rotation','alternating_radial']:
    row=next(x for x in d['motion'] if x['motion']==name and x['eta']==1 and x['tracer_speed']==0)
    motion_table.append((name,100*((-row['tracer_acceleration'][0])/baseline-1)))
hashes={f:hashlib.sha256((dest/f).read_bytes()).hexdigest() for f in ['result.json','supplement.json']}
report='''# First round: structure, surroundings, motion, and spillover

This round compares exact, explicitly specified toy mechanisms. All parameters are illustrative dimensionless values, not fitted physical constants. It does not claim a new gravity law or new observational fit. The results identify behaviors we can now distinguish experimentally.

## 1. Arrangement matters, but much of that is already Newtonian

We placed the same unit mass at the center, in two separated clumps, in a ring, and on a spherical shell. At radius 2 along the clump axis, the two-clump layout gives 45.8% more inward Newtonian pull than a central point mass; the ring gives 10.4% more. A spherical shell gives the same exterior Newtonian pull as the central mass. Thus a claim that 'clumping changes gravity' is insufficient: a new law must predict a difference beyond ordinary geometry.

The range-dependent force comes from the pair potential

    U_ij = -G m_i m_j/d [1 + sum alpha (1-exp(-d/L))].

This is the already discussed Newtonian-plus-subtracted-Yukawa family, not the newer filtered-curvature action. The force enhancement is 1+sum alpha[1-(1+d/L)exp(-d/L)]. We compared Newtonian gravity, L=0.3, L=3, and an equal-weight mixture. Each single-range amplitude was 1, and the mixture used 0.5 for each range. The total mass and a common containing region were fixed; layouts were instantaneous point-clump snapshots, not stable stellar systems. Singular coincident source interactions were not evolved.

Two concentric shells versus one outer shell also produce different exterior forces in the range model, while Newtonian gravity gives the same exterior result. Replacing every mass element with two half-mass elements at the identical position changes the static external calculation by less than 2e-14 relative: bookkeeping does not create extra gravity. Physical clump count and arrangement are distinct from simulation-cell count.

## 2. Outer matter can change the inner pull

We placed a shell of radius 2 and mass 10 around a central unit mass, measuring the pull at radius 1. In Newtonian gravity the shell contributes zero. In the short-range model it adds 3.39% to the central-only model pull; in the long-range model it adds 9.20%; the mixture adds 5.49%. These numbers depend on the selected toy ranges and masses, not observed galaxy parameters.

The shell contribution was checked against an exact spherical integral, with maximum absolute disagreement below 5e-15. It points toward the center in this particular model. This is an environmental effect generated by the same pair law, without declaring the system a galaxy or cluster. Whether its sign and magnitude are useful for real systems remains untested. Real surroundings are not generally perfect shells.

## 3. A correction can be localized, but it has spillover

We moved a shell's fixed mass from radius 0.3 to 0.8. Outside both shells, the Newtonian force did not change. The modified pull weakened. The magnitude of that change, divided by the Newtonian point-source pull at each location, was:

| Probe radius | Short range L=0.3 | Long range L=3 |
|---:|---:|---:|
'''
for r in [1.2,2.,4.,8.]:
    vals=[next(abs(x['change_over_newton_point'])*100 for x in d['spillover'] if x['model']==model and x['r']==r) for model in ['short','long']]
    report+=f'| {r:g} | {vals[0]:.6g}% | {vals[1]:.6g}% |\n'
report+='''
Reversing the redistribution reverses the change. The short-range model therefore gives an example of a substantial nearby correction with tiny distant effects. The long-range model spreads the change more broadly. This does not make gravity a fixed resource or allow independent tuning for individual bodies. It quantifies the tradeoff we would need to test when improving an inner rotation curve while preserving the outskirts.

## 4. Motion can matter, but only after specifying a different mechanism

All static pair models give identical instantaneous forces for identical positions and masses, regardless of assigned velocities. To explore the user's motion idea we added a separate, explicit velocity-dependent kinetic coupling to Newtonian mechanics:

    L = sum m_i |v_i|²/2 - U_Newton
        + eta/2 sum_(i<j) (m_i m_j/Mref) exp(-d/Lv) |v_i-v_j|².

Mref=Lv=1 are fixed reference scales in this toy experiment; eta=0, 0.1, 1. This is a modification of the mechanical response, not merely an extra static force. For eta>=0 its additional kinetic quadratic form is nonnegative. Relative velocities make it unchanged by adding the same velocity to the whole system. It is spatially nonlocal and instantaneous, with no relativistic or photon completion.

Eight equal clumps occupy identical ring positions and have the same ordinary kinetic energy. They either rotate together, alternate clockwise/counterclockwise, or alternate radial inward/outward motion. A small tracer sits outside. At eta=1, with the tracer initially stationary, the extra inward acceleration relative to the Newtonian case is:

| Source motion | Extra inward tracer acceleration |
|---|---:|
'''
for name,value in motion_table:
    report+=f"| {name.replace('_',' ')} | {value:.3f}% |\n"
report+='''
Moving the tracer with or against the rotation also changes the result. Thus both the source motion and the body's motion can matter in this particular construction. It is not evidence that such a coupling exists in nature. Counter-rotation and radial motion are controlled patterns, not full random-motion populations. The coupled source accelerations are solved too; sources are not pinned while receiving no reaction.

The finite system passed total momentum, uniform-boost, rotation/translation, and directional energy-derivative controls, including an asymmetric state. These checks distinguish it from an arbitrary coherence multiplier. They do not establish long-term stability, a continuum limit, Solar System compliance, or a universal theory. The source motions and speed-dependent correction would need independent data beyond a rotation-curve mass table.

## 5. Existing SPARC evidence argues against automatic extra attraction

We re-examined the already exposed focusing formula, without fitting anything. Among the 35 galaxies highest in its surface-density descriptor, it improved only 14 and the median galaxy's score worsened by 31.6%. Among the highest-compactness 35, it improved 12 and the median worsened by 30.2%. These are post-selection descriptive partitions of the same historical development set, not fresh significance tests. The descriptors are the frozen baryonic summaries, not direct 3D densities.

This supports a practical direction for development: permit a signed or suppressed correction and report galaxy-by-galaxy outcomes. It does not prove that density causes the failure. The historical score is an uncertainty-weighted squared velocity error, not a percentage error in speed. The bulge descriptor has many ties, so its arbitrary rank-based quarters are retained in JSON but not interpreted as clean physical classes. Nuisance mass, geometry, and measurement errors still need investigation.

## Choices for brainstorming

1. Prioritize arrangement and environment with a short-range component to control spillover, keeping Newtonian geometry as the mandatory comparison.
2. Keep a broader component only if it improves a repeated outer-galaxy pattern without damaging the inner fit or mass scaling. The finite fixed linear range family does not by itself solve the universal mass-scaling problem.
3. Treat motion as a separate candidate track. The first conservative toy demonstrates how to test it, but adopting its arbitrary coupling would be premature. Co-rotation versus counter-rotation at fixed structure is more discriminating than an undifferentiated 'spin boosts gravity' claim.

Next real-data step after review: use the existing SPARC residual atlas to pick source-defined compact/diffuse and radial-feature comparisons, propagate mass/geometry uncertainty, and test unchanged one-range/two-range predictions. New motion-sensitive predictions require suitable independent kinematics; the mass model must not use the velocities it is asked to predict. No simulation success here is an observational success.
'''
report+='\nEvidence hashes:\n'+ '\n'.join(f'- {name}: {value}' for name,value in hashes.items())+'\n'
(dest/'report.md').write_text(report,encoding='utf-8')
(outputs/'Gravity-first-round-structure-environment-motion.md').write_text(report,encoding='utf-8')
print(json.dumps(extra['asymmetric_mechanical_controls']))
print('Motion enhancements',motion_table)
print('Nested shells',json.dumps(nested[:4]))
