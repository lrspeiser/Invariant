"""Separate CPU formula and quadrature replay of selected finest source vectors."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import roots_laguerre
from threadpoolctl import threadpool_limits
from mond_atlas_spatial_program import planar
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-log-source-001'
def main():
    config=json.loads((R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json').read_text());case=config['source_cases'][-1];comp=case['components'][0]
    with np.load(R/comp['path']) as packet:
        xy,mass,expected=planar(packet,comp['conversion_to_msun_pc2'],.03125)
        assert np.allclose(packet['vertical_layers'],comp['vertical_layers'])
    with (P/'run001/fields.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    nodes,weights=roots_laguerre(48);output=[];errors=[]
    for radius,z in [(1.,0.),(1.,.4),(6.,.4)]:
        point=np.array([radius,0,z]);answer=np.zeros((2,4))
        for fraction,height in comp['vertical_layers']:
            for node,w in zip(nodes,weights):
                for sign in [-1,1]:
                    sources=np.column_stack([xy,np.full(len(mass),sign*node*height)]);d=point-sources;r=np.linalg.norm(d,axis=1);s=np.hypot(r,.05);dm=mass*fraction*w*.5
                    answer[0,0]+=np.sum(4.30091727003628e-6*dm/4*np.log(s/(s+4)))
                    answer[0,1:]+=np.einsum('i,ij->j',-4.30091727003628e-6*dm/(s**3*(1+4/s)),d)
                    answer[1,0]+=np.sum(-4.30091727003628e-6*dm/r)
                    answer[1,1:]+=np.einsum('i,ij->j',-4.30091727003628e-6*dm/r**3,d)
        for index,model in enumerate(['log_extra','newton']):
            row=next(t for t in rows if t['case']==case['id'] and t['component']==comp['id'] and t['level']=='2' and t['model']==model and float(t['r'])==radius and float(t['z'])==z and float(t['theta'])==0)
            original=np.array([float(row[k]) for k in ['phi','gx','gy','gz']]);err=float(np.linalg.norm(answer[index]-original)/np.linalg.norm(original));assert err<1e-10;errors.append(err);output.append(dict(r=radius,z=z,model=model,relative_error=err))
    result=dict(status='PASS_SELECTED_FINEST_CPU_REPLAY',case=case['id'],component=comp['id'],independent_vertical_nodes='scipy roots_laguerre versus numpy laggauss',shared_planar_builder='Previously audited exact bilinear cell mass integration; not independently reimplemented here',comparisons=output,max_relative_error=max(errors),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'run001/independent-cpu.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
