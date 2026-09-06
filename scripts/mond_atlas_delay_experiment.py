"""Manufactured memory/feedback benchmark; no observational scoring."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def kernel(n=16, sigma=2.):
    i=np.arange(n); d=np.minimum(i,n-i)
    w=np.exp(-.5*(d/sigma)**2); w/=w.sum()
    return np.array([np.roll(w,j) for j in range(n)])


def step(t, lam=1., alpha=0., tau=1.):
    t=np.asarray(t); gap=1-alpha*lam
    return lam*t/tau if abs(gap)<1e-14 else -lam*np.expm1(-gap*t/tau)/gap


def transfer(omega, lam=1., alpha=0., tau=1.):
    return lam/(1-alpha*lam+1j*omega*tau)


def integrate_mode(times, lam=1., alpha=0., tau=1., omega=None, rtol=1e-10):
    if omega is None:
        f=lambda t,y:(lam-(1-alpha*lam)*y)/tau
        initial=[0.]
    else:
        f=lambda t,y:(lam*np.exp(1j*omega*t)-(1-alpha*lam)*y)/tau
        initial=[transfer(omega,lam,alpha,tau)]
    return solve_ivp(f,(0,float(times[-1])),initial,t_eval=times,method='DOP853',rtol=rtol,atol=rtol*.01).y[0]


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists(): raise FileExistsError('Immutable result already exists')
    rows=[]; max_step=0.; max_frequency=0.
    for alpha in [0,.25,.5,.8,.9,.95,.99,1,1.01,1.1]:
        for tau in [.1,1,10]:
            times=np.linspace(0,10*tau,101)
            numeric=integrate_mode(times,alpha=alpha,tau=tau)
            exact=step(times,alpha=alpha,tau=tau)
            err=float(np.max(np.abs(numeric-exact)/(1+np.abs(exact))))
            max_step=max(max_step,err)
            for omega in [.001,.01,.1,1,10]:
                h=transfer(omega,alpha=alpha,tau=tau)
                # Cover 20 radians or 20 relaxation times, whichever is shorter.
                # Initializing the analytic periodic state isolates integration error.
                duration=min(20/omega,20*tau/(1-alpha)) if alpha<1 else 20/omega
                freq_times=np.linspace(0,duration,101)
                # Unstable integrations cannot follow a bounded particular solution
                # for arbitrarily long times; report transfer formally but do not
                # mislabel it as a physically attained steady state.
                freq_err=None
                if alpha<1:
                    y=integrate_mode(freq_times,alpha=alpha,tau=tau,omega=omega)
                    e=h*np.exp(1j*omega*freq_times)
                    freq_err=float(np.max(np.abs(y-e)/(1+np.abs(e))))
                    max_frequency=max(max_frequency,freq_err)
                rows.append(dict(alpha=alpha,tau=tau,omega=omega,
                    stability='stable' if alpha<1 else ('marginal_forced_linear_growth' if alpha==1 else 'unstable'),
                    uniform_static_response_gain=1/(1-alpha) if alpha<1 else None,
                    relaxation_time=tau/(1-alpha) if alpha<1 else None,
                    response_at_10_tau=float(exact[-1]),
                    fraction_equilibrium_at_10_tau=float(1-np.exp(-10*(1-alpha))) if alpha<1 else None,
                    formal_frequency_amplitude=float(abs(h)),formal_frequency_phase_degrees=float(np.angle(h,deg=True)),
                    step_scaled_error=err,frequency_scaled_error=freq_err))
    with (output/'sweep.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    modes=[]
    for sigma in [1.,2.,4.]:
        vals=np.linalg.eigvalsh(kernel(sigma=sigma))
        for rank,lam in enumerate(vals):
            modes.append(dict(sigma_cells=sigma,rank=rank,eigenvalue=float(lam),growth_rate_alpha_09=float(.9*lam-1),growth_rate_alpha_11=float(1.1*lam-1)))
    with (output/'modes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(modes[0]));w.writeheader();w.writerows(modes)
    convergence=[]
    for n in [16,32,64]:
        k=kernel(n,n/8)
        vals=np.fft.fft(k[0]).real
        convergence.append(dict(n=n,eigenvalue_mode_1=float(vals[1]),eigenvalue_mode_2=float(vals[2])))
    t=np.linspace(0,10,101); exact=step(t,alpha=.9)
    integration_errors={str(tol):float(np.max(np.abs(integrate_mode(t,alpha=.9,rtol=tol)-exact))) for tol in [1e-6,1e-8,1e-10]}
    summary=dict(disposition='THEORY_BENCHMARK_ONLY',observations_opened=0,sweep_count=len(rows),mode_count=len(modes),max_step_scaled_error=max_step,max_frequency_scaled_error=max_frequency,kernel_resolution=convergence,integration_convergence=integration_errors,
        gates_passed=bool(max_step<1e-8 and max_frequency<1e-8),
        kernel_outer_budget={str(x):float(np.log1p(x)-x/(1+x)) for x in [1,10,100,1000,1e6]})
    (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output),indent=2))
