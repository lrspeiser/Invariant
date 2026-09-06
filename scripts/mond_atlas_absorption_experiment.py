"""Manufactured absorption/relay tests, not an observational gravity fit."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def transmission(tau):
    tau = np.asarray(tau, dtype=float)
    if np.any(tau < 0) or np.any(~np.isfinite(tau)):
        raise ValueError('Optical depth must be finite and nonnegative')
    return np.exp(-tau)


def packet(tau, eta, forward_fraction):
    if not (0 <= eta <= 1.2 and 0 <= forward_fraction <= 1):
        raise ValueError('Out of declared test domain')
    t = float(transmission(tau)); interacting = -np.expm1(-tau)
    # Active eta > 1 is accounted as external input, not negative retention.
    return dict(direct=t, forward=eta*forward_fraction*interacting,
                backward=eta*(1-forward_fraction)*interacting,
                retained=max(0., 1-eta)*interacting,
                external_input=max(0., eta-1)*interacting)


def clumpy_transmission(mean_tau, coverage):
    if not 0 < coverage <= 1:
        raise ValueError('Coverage must lie in (0,1]')
    return 1-coverage+coverage*float(transmission(mean_tau/coverage))


def attenuated_field(point, opacity=1., GM=1.):
    point = np.asarray(point, dtype=float); radius = np.linalg.norm(point)
    if radius == 0 or opacity < 0:
        raise ValueError('Nonzero radius and nonnegative opacity required')
    return -GM*float(transmission(opacity*radius))*point/radius**3


def transverse_screen_field(point, alpha=.8, width=1.):
    """Manufactured local attenuated central field; not a physical slab solver."""
    p = np.asarray(point, dtype=float); r = np.linalg.norm(p)
    tau = alpha*np.exp(-p[1]**2/(2*width**2))
    return -float(transmission(tau))*p/r**3


def curl(field, point, h):
    p=np.asarray(point,dtype=float); eye=np.eye(3)
    jac=np.column_stack([(field(p+h*e)-field(p-h*e))/(2*h) for e in eye])
    return np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])


def analytic_screen_curl(point, alpha=.8, width=1.):
    p=np.asarray(point,dtype=float); r=np.linalg.norm(p)
    tau=alpha*np.exp(-p[1]**2/(2*width**2)); T=float(transmission(tau))
    gradT=np.array([0.,T*tau*p[1]/width**2,0.])
    return np.cross(gradT,-p/r**3)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'work/gravity-first-principles/mond-atlas-relay-001/absorption'
    if (out/'receipt.json').exists():
        raise RuntimeError('Immutable result exists; do not overwrite')
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
        '-p','test_mond_atlas_absorption_experiment.py','-v'],cwd=root,capture_output=True,text=True)
    (out/'test-output.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:
        raise RuntimeError('Independent benchmark failed; see retained test-output.txt')
    rows=[]
    for tau in [0,.001,.01,.1,1,3,10]:
        for eta in [0,.25,.5,1,1.2]:
            for f in [0,.25,.5,1]:
                p=packet(tau,eta,f)
                for n in [1,2,10,100]:
                    rows.append(dict(tau=tau,eta=eta,forward_fraction=f,relays=n,
                        passive=eta<=1,forward_after_n=(p['direct']+p['forward'])**n,
                        one_step_balance_error=sum(p[k] for k in ['direct','forward','backward','retained'])-1-p['external_input'],**p))
    write_csv(out/'relay-grid.csv',rows)
    clumps=[dict(mean_tau=t,coverage=c,uniform_transmission=float(transmission(t)),
            clumpy_transmission=clumpy_transmission(t,c),ratio_to_uniform=clumpy_transmission(t,c)/float(transmission(t)))
            for t in [.1,1,3] for c in [.1,.25,.5,1]]
    write_csv(out/'clumpy-screen.csv',clumps)
    radial=[dict(radius=r,newton_acceleration=1/r**2,attenuated_acceleration=np.linalg.norm(attenuated_field([r,0,0])),
            ratio_to_newton=float(transmission(r)),log_acceleration_slope=-2-r,
            log_circular_speed_slope=(-1-r)/2) for r in [.01,.1,1,3,10]]
    write_csv(out/'radial-attenuation.csv',radial)
    conv=[]
    for n in [8,16,32,64]:
        s=(np.arange(n)+.5)/n;column=float(np.mean(1+s*s))
        conv.append(dict(cells=n,column=column,column_error=abs(column-4/3),
                         transmission=float(transmission(column))))
    write_csv(out/'column-convergence.csv',conv)
    curls=[];p=[2.,1.,0.];expected=analytic_screen_curl(p)
    for h in [.1,.05,.025,.0125,.001,.0001]:
        measured=curl(transverse_screen_field,p,h)
        curls.append(dict(h=h,curl_z=float(measured[2]),analytic_curl_z=float(expected[2]),
                          absolute_error=float(np.linalg.norm(measured-expected))))
    write_csv(out/'curl-convergence.csv',curls)
    summary=dict(disposition='THEORY_BENCHMARK_ONLY',observed_response_rows=0,
        relay_grid_rows=len(rows),passive_max_forward=max(r['forward_after_n'] for r in rows if r['passive']),
        passive_min_forward=min(r['forward_after_n'] for r in rows if r['passive']),
        max_packet_balance_error=max(abs(r['one_step_balance_error']) for r in rows),
        tau1_eta1_isotropic_after10=(np.exp(-1)+(1-np.exp(-1))*.5)**10,
        tau1_cover01_ratio_to_uniform=clumpy_transmission(1,.1)/np.exp(-1),
        manufactured_screen_curl_z=float(expected[2]),
        next_tests=['Derive attractive force and momentum transfer from an interaction law',
        'Test geometry-dependent opacity with a conservative joint matter-field model',
        'Map composition dependence to MICROSCOPE only after specifying coupling',
        'Independent controlled source-shield motion with shield own gravity modeled'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    files=[Path(__file__),root/'tests/test_mond_atlas_absorption_experiment.py',*sorted(out.glob('*'))]
    receipt=dict(disposition='THEORY_BENCHMARK_ONLY',inputs='manufactured dimensionless grids; no observational targets',
        benchmark_before_experiment=True,tests_exit_code=test.returncode,implementation='CPU NumPy; no CUDA required',
        sources=[dict(url='https://arxiv.org/abs/2209.15487',role='primary constraint paper, not fitted data'),
                 dict(url='https://doi.org/10.1103/PhysRevD.63.062002',role='primary shielding critique, not fitted data')],
        files=[dict(path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files if f.is_file()])
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
