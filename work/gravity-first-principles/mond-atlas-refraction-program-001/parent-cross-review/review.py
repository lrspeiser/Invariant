"""Independent sparse manufactured FV check and saved-vector gate replay."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
import mond_atlas_refraction_program as m
P=Path(__file__).resolve().parent;BASE=P.parent

def independent_dirichlet(rhs,eps,spacing,bc):
    inner=tuple(n-2 for n in rhs.shape);A=lil_matrix((np.prod(inner),np.prod(inner)));b=np.zeros(np.prod(inner))
    for index in np.ndindex(inner):
        center=tuple(i+1 for i in index);row=np.ravel_multi_index(index,inner);b[row]=-rhs[center]
        for axis,h in enumerate(spacing):
            for sign in (-1,1):
                neighbor=list(center);neighbor[axis]+=sign;neighbor=tuple(neighbor)
                coefficient=2/(1/eps[center]+1/eps[neighbor])/h**2
                A[row,row]+=coefficient
                if all(0<i<n-1 for i,n in zip(neighbor,rhs.shape)):
                    column=np.ravel_multi_index(tuple(i-1 for i in neighbor),inner);A[row,column]-=coefficient
                else:b[row]+=coefficient*bc[neighbor]
    phi=bc.copy();phi[(slice(1,-1),)*3]=spsolve(A.tocsr(),b).reshape(inner)
    return phi,float(abs(A-A.T).max())

def main():
    tests=[]
    for n in (9,17):
        axes=[np.linspace(-extent,extent,n) for extent in (1,.8,.6)];spacing=[a[1]-a[0] for a in axes]
        x,y,z=np.meshgrid(*axes,indexing='ij');truth=x*x+2*y*y+3*z*z;epsilon=1+.2*x+.1*y
        rhs=12+2.8*x+1.6*y
        reference,asym=independent_dirichlet(rhs,epsilon,spacing,truth);actual,check=m.solve(rhs,epsilon,spacing,truth)
        discrepancy=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference));analytic=float(np.linalg.norm(reference-truth)/np.linalg.norm(truth))
        assert discrepancy<1e-9 and asym<1e-10 and check['passed']
        tests.append(dict(n=n,independent_direct_vs_CG_relative=discrepancy,manufactured_analytic_error=analytic,matrix_asymmetry=asym))
    assert tests[1]['manufactured_analytic_error']<.4*tests[0]['manufactured_analytic_error']
    read=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
    initial=read(BASE/'run001/sampled-vectors.csv');fine=read(BASE/'finer002/sampled-vectors.csv');gates=[]
    for model in ('newton','refraction'):
        old=[r for r in initial if r['case']=='f4-stars-h0p4' and r['model']==model and r['grid']=='fine'];new=[r for r in fine if r['model']==model]
        points=np.array([[float(r[k]) for k in ('x','y','z')] for r in new]);assert np.allclose(points,[[float(r[k]) for k in ('x','y','z')] for r in old])
        a=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in old]);b=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in new]);rms=float(np.linalg.norm(a-b)/np.linalg.norm(b));groups=[]
        for z in (0,.2,.5,1):
            mask=points[:,2]==z;groups.append(dict(z=z,rms=float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask]))))
        gates.append(dict(model=model,rms=rms,groups=groups,passes=bool(rms<.05 and all(g['rms']<.08 for g in groups))))
    manifest=json.loads(m.MANIFEST.read_text(encoding='utf-8'));case=next(c for c in manifest['source_cases'] if c['id']=='f4-stars-h0p4');boundary=[]
    for name,half,spacing in [('base',[8,8,4],[.25,.25,.125]),('fine',[8,8,4],[.125,.125,.0625]),('box',[12,12,6],[.25,.25,.125])]:
        axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)];rho,record=m.source(case,axes);total=rho.sum();interior=rho[1:-1,1:-1,1:-1].sum()
        boundary.append(dict(grid=name,reported_all_node_mass=float(total*np.prod(spacing)),active_interior_rhs_mass=float(interior*np.prod(spacing)),fraction_on_dirichlet_nodes=float(1-interior/total),note='Shared source adapter; boundary nodes impose Phi and do not enter PDE RHS.'))
    result=dict(status='INDEPENDENT_OPERATOR_PASS_POINT_LAW_FIELD_FAILURE_RETAINED',manufactured=tests,field_gates=gates,boundary_mass=boundary,observed_response_arrays_opened=0)
    (P/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
