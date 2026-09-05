"""All-profile smooth source feasibility with measured offsets; no gravity fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linprog

def offset_fraction(R,a,d):
    D=np.sqrt(((R-d)**2+a*a)*((R+d)**2+a*a))
    X=R*R-a*a-d*d
    return np.where(X<0,2*a*a*R*R/(D*(D-X)),.5*(1+X/D))

def shell_matrix(r,s,ratio,n):
    q,w=leggauss(n)
    theta=(q+1)*np.pi/4
    weights=w*np.pi/4*np.sin(theta)
    result=np.empty((len(r),len(s)))
    for start in range(0,len(s),8):
        ss=s[start:start+8][None,:,None]
        result[:,start:start+8]=np.sum(offset_fraction(r[:,None,None],ratio*ss,ss*np.sin(theta))*weights,axis=-1)
    return result

root=Path(__file__).parent/'Invariant'
base=root/'work/gravity-first-principles'
paths=[base/'projected-stellar-packet-001/result.json',base/'stellar-measured-centering-001/result.json']
packet,centering=[json.loads(p.read_bytes()) for p in paths]
centers={p['cluster']:p for p in centering['rows']}
dest=base/'smooth-stellar-feasibility-001'
dest.mkdir(exist_ok=False)
registration=dict(input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    shell_grid='All measured radii plus 128 logarithmic radii from 0.1*first to 10*last',
    width_to_shell_radius=[.02,.1,.2],quadrature_nodes=[256,512],
    offset_scales_r500=np.geomspace(.001,.02,17).tolist(),
    matrix_relative_target=1e-6,prediction_change_over_measured_mass_target=1e-6,
    bracket_violation_target=1e-8,
    scope='Every stellar bound retained; only positive smooth shells and Plummer components at measured projected BCG offset. Widths and centers not optimized against gravity; no covariance or complete source admission.')
for p in paths:
    (dest/p.parent.name).mkdir(exist_ok=True)
    (dest/p.parent.name/p.name).write_bytes(p.read_bytes())
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
for p in packet['stellar_packets']:
    c=p['columns']; r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
    center=centers[p['cluster']]
    R500=center['offset_kpc']/center['offset_r500']
    scales=np.array(registration['offset_scales_r500'])*R500
    extra=offset_fraction(r[:,None],scales[None,:],center['offset_kpc'])
    shells=np.unique(np.r_[r,np.geomspace(.1*r[0],10*r[-1],128)])
    for width in registration['width_to_shell_radius']:
        coarse=shell_matrix(r,shells,width,256)
        fine=shell_matrix(r,shells,width,512)
        error=float(np.max(abs(coarse-fine)/np.maximum(abs(fine),1e-300)))
        fraction=np.c_[fine,extra]
        design=fraction*m[-1]/m[:,None]
        A=np.r_[np.c_[design,-np.ones(len(r))],np.c_[-design,-np.ones(len(r))]]
        b=np.r_[hi/m,-lo/m]
        answer=linprog(np.r_[np.zeros(fraction.shape[1]),1.],A_ub=A,b_ub=b,bounds=(0,None),method='highs',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        row=dict(cluster=p['cluster'],shell_width_ratio=width,status=int(answer.status),matrix_relative_change=error)
        if answer.success:
            masses=answer.x[:-1]*m[-1]
            pred=fraction@masses
            pred_coarse=np.c_[coarse,extra]@masses
            change=float(np.max(abs(pred-pred_coarse)/m))
            violation=float(max(0,np.max((lo-pred)/m),np.max((pred-hi)/m)))
            assert abs(violation-answer.x[-1])<2e-8 and np.all(masses>=0)
            row.update(relaxation=float(answer.x[-1]),verified_bracket_violation=violation,
                prediction_change_over_mass=change,
                numerical_pass=error<1e-6 and change<1e-6,
                within_bounds=violation<=1e-8,
                shell_radii_kpc=shells.tolist(),shell_mass_msun=masses[:len(shells)].tolist(),
                offset_kpc=center['offset_kpc'],offset_scales_kpc=scales.tolist(),
                offset_mass_msun=masses[len(shells):].tolist(),projected_mass_msun=pred.tolist())
        rows.append(row)
        (dest/f"case_{p['cluster']}_{width}.json").write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(p['cluster'],width,'relaxation',row.get('relaxation'),'matrix error',error,flush=True)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==registration['input_sha256'][str(p.relative_to(root))] for p in paths)
(dest/'result.json').write_text(json.dumps(dict(registration=registration,rows=rows,gravity_scores=0,
    physical_exclusions=0,all_inputs_unchanged=True),indent=2),encoding='utf-8')
