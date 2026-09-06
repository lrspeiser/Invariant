"""Independent frozen Newtonian shell control; not a transport model."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss


def run():
    folder=Path(__file__).resolve().parent
    if (folder/'shell-result.json').exists():raise RuntimeError('Immutable result already exists')
    rows=[];errors=[]
    for n in [16,32,64,128]:
        mu,w=leggauss(n);local=[]
        for r in [0.,1.,3.,4.,6.,10.]:
            actual=float(.5*np.sum(w*(5*mu-r)/(25+r*r-10*r*mu)**1.5))
            expected=0. if r<5 else -1/r**2
            error=abs(actual-expected);local.append(error)
            rows.append(dict(order=n,radius=r,radial_field=actual,exact=expected,absolute_error=error))
        errors.append(max(local))
    with (folder/'shell-grid.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passed=bool(errors[-1]<=1e-10 and errors[0]>errors[1]>errors[2])
    result=dict(disposition='THEORY_BENCHMARK_ONLY' if passed else 'BENCHMARK_FAILED',
        passed=passed,max_errors_by_order=errors,
        interpretation='A uniform Newtonian secondary-source shell cancels inside; no universal conclusion about directional transport',
        files=[dict(path=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
               for f in [Path(__file__),folder/'SHELL_PREFLIGHT.md',folder/'shell-grid.csv']])
    (folder/'shell-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not passed:raise RuntimeError('Shell gate failed')


if __name__=='__main__':run()
