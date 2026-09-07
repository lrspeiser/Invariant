"""Selected actual-source direct GPU sums independent of planar FFT convolution."""
import json,hashlib
from pathlib import Path
import numpy as np
import cupy as cp
from scipy.ndimage import map_coordinates
from scipy.special import roots_laguerre
from threadpoolctl import threadpool_limits
from mond_atlas_spatial_program import planar
from mond_atlas_column_force_v2 import source_coeff,kernel,grids,G
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-column-force-001'
def main():
    cfg=json.loads((R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json').read_text());case=next(c for c in cfg['source_cases'] if c['id']=='f4-stars-h0p1');comp=case['components'][0];points=np.array([[1.,.3],[2.,-.5],[5.,1.]])
    with np.load(R/comp['path']) as packet:coeff,mass=source_coeff(packet,comp['conversion_to_msun_pc2'],32.,.015625);xy,dm,expected=planar(packet,comp['conversion_to_msun_pc2'],.015625)
    gg=grids(coeff,kernel(32.,.015625,.1,'log',128));pred=np.column_stack([map_coordinates(a,(points/.015625).T%2048,order=3,mode='grid-wrap') for a in gg]);nodes,weights=roots_laguerre(128);xy=cp.asarray(xy);dm=cp.asarray(dm);direct=[]
    for p in points:
        d=cp.asarray(p)-xy;r2=cp.sum(d*d,axis=1);factor=cp.zeros(len(dm))
        for h,coef in [(.1,.01/(.01-.04)),(.2,-.04/(.01-.04))]:
            for z,w in zip(h*nodes,coef*weights):
                s=cp.sqrt(r2+z*z+.05**2);factor+=w/(s*s*(s+4))
        direct.append(cp.asnumpy(-G*cp.sum((dm*factor)[:,None]*d,axis=0)))
    direct=np.array(direct);err=np.linalg.norm(direct-pred,axis=1)/np.linalg.norm(pred,axis=1);assert np.max(err)<.002
    result=dict(status='PASS_SELECTED_ACTUAL_SOURCE_DIRECT_GPU',source=comp['path'],source_sha256=comp['sha256'],source_mass_msun=mass,cell_mass_msun=expected,planar_cell_spacing=.015625,vertical_order=128,points=points.tolist(),fft=pred.tolist(),direct=direct.tolist(),relative_errors=err.tolist(),maximum_relative_error=float(max(err)),limitation='Independent planar integration; vertical difference-density quadrature shared but separately checked against adaptive integration before fields.',device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'direct-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
