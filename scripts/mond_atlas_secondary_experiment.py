"""Single-generation, conservative pair response; theory benchmark only."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.special import roots_laguerre
from mond_atlas_halo_return import mass_shape, field, numerical_gradient


def disk_sources(M, a, nr, nphi):
    nodes, weights = roots_laguerre(nr)
    phi = np.arange(nphi)*2*np.pi/nphi
    radius = a*nodes[:, None]
    positions = np.stack([radius*np.cos(phi), radius*np.sin(phi), np.zeros((nr,nphi))], axis=-1).reshape(-1,3)
    masses = np.broadcast_to((M*weights*nodes/nphi)[:,None], (nr,nphi)).ravel().copy()
    return positions, masses


def secondary(point, positions, masses, eta, L, G=1., cutoff=None):
    delta = np.asarray(point)-np.asarray(positions)
    r = np.linalg.norm(delta, axis=-1)
    if np.any(r<=0):
        raise ValueError('Coincident cusp directions excluded')
    x = r/L if cutoff is None else np.minimum(r/L,cutoff)
    return -np.sum((G*eta*np.asarray(masses)*mass_shape(x,'NFW')/r**3)[:,None]*delta, axis=0)


def potential(point, positions, masses, eta, L, G=1.):
    r = np.linalg.norm(np.asarray(point)-np.asarray(positions),axis=-1)
    if np.any(r<=0):
        raise ValueError('Coincident evaluation excluded')
    return float(-G*eta*np.sum(masses*np.log1p(r/L)/r))


def run():
    root=Path(__file__).resolve().parents[1]
    output=root/'work/gravity-first-principles/mond-atlas-relay-001/distributed'
    if (output/'results.json').exists():
        raise RuntimeError('Immutable result already exists')
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_secondary_experiment.py','-v'],cwd=root,capture_output=True,text=True)
    (output/'test-log.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:
        raise RuntimeError('Pre-evaluation tests failed; see preserved log')
    M,a,L,rho,G=6e10,3.,19.5725,8.53702e6,4.30091727003628e-6
    eta=4*np.pi*rho*L**3/M
    sources=[disk_sources(M,a,n,2*n) for n in (32,64,128)]
    points=[('grid',R,z) for R in [.5,1,2,4,8,16,32,64] for z in [.5,2,8,32]]
    points += [('angular',r*np.cos(np.deg2rad(t)),r*np.sin(np.deg2rad(t))) for r in [8,16,32,64] for t in [15,30,45,60,75,90]]
    rows=[]
    for group,R,z in points:
        p=np.array([R,0,z]); radius=np.linalg.norm(p)
        values=[secondary(p,s,m,eta,L,G) for s,m in sources]
        ref=field(p,rho,L,'NFW',G)
        fine=values[-1]; fn=np.linalg.norm(fine)
        rel=np.linalg.norm(values[1]-fine)/fn
        row=dict(group=group,R_kpc=R,z_kpc=z,r_kpc=radius,
            coarse_base_relative=float(np.linalg.norm(values[0]-values[1])/np.linalg.norm(values[1])),
            base_fine_relative=float(rel),converged=bool(rel<.01),
            response_R=float(fine[0]),response_z=float(fine[2]),
            point_R=float(ref[0]),point_z=float(ref[2]),
            strength_ratio=float(fn/np.linalg.norm(ref)),
            vector_relative_to_point=float(np.linalg.norm(fine-ref)/np.linalg.norm(ref)),
            angle_from_inward_degrees=float(np.degrees(np.arccos(np.clip(np.dot(fine,ref)/fn/np.linalg.norm(ref),-1,1)))))
        rows.append(row)
    with (output/'field-comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    roundness=[]
    for r in [8,16,32,64]:
        subset=[v for v in rows if v['group']=='angular' and abs(v['r_kpc']-r)<1e-8]
        ratios=[v['strength_ratio'] for v in subset]
        roundness.append(dict(radius_kpc=r,min_strength_ratio=min(ratios),max_strength_ratio=max(ratios),
            angular_strength_spread_relative_to_mean=(max(ratios)-min(ratios))/np.mean(ratios),
            max_direction_offset_degrees=max(v['angle_from_inward_degrees'] for v in subset),
            max_resolution_relative=max(v['base_fine_relative'] for v in subset)))
    result=dict(admission='THEORY_BENCHMARK_ONLY',pre_evaluation_tests_passed=True,
        calibrated_eta=eta,quadratures=[[32,64],[64,128],[128,256]],
        convergence_threshold=.01,all_points_converged=all(v['converged'] for v in rows),
        max_resolution_relative=max(v['base_fine_relative'] for v in rows),
        failed_points=[v for v in rows if not v['converged']],roundness=roundness,
        effective_source_budget=[dict(radius_over_L=x,response_to_baryon_ratio=float(eta*mass_shape(x,'NFW')),
            truncated_force_ratio=float(mass_shape(min(x,10),'NFW')/mass_shape(x,'NFW'))) for x in [1,10,100,1000]],
        bindings={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'scripts/mond_atlas_halo_return.py',root/'tests/test_mond_atlas_secondary_experiment.py',output/'PREFLIGHT.md']})
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
