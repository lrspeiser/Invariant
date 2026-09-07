from pathlib import Path
import json,hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mond_atlas_source_resolution import cell_projection_matrix
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001';PRIVATE=ROOT/'work/private/mond-atlas-hi-column-001'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def moment_matrix(cells,width,nodes,h):
 delta=cells[:,None]-nodes[None,:];lo=delta-width/2;hi=delta+width/2
 def F(u):
  q=np.clip(u,-h,h)
  return np.where(q<=0,q*q/2+q**3/(3*h)-h*h/6,-h*h/6+q*q/2-q**3/(3*h))
 A=cell_projection_matrix(cells,width,nodes,h,0)
 return nodes[None,:]*A+(F(hi)-F(lo))/width

def main():
 out=P/'centroid002';out.mkdir(exist_ok=False);priv=PRIVATE/'centroid002';priv.mkdir(parents=True,exist_ok=False)
 old=read(P/'run001/summary.json');assets=[];rows=[]
 for item in old['assets']:
  if item['role']!='factorized intrinsic integrated HI flux':continue
  case=Path(item['path']).name.split('-')[0];sp=ROOT/next(c['source_path'] for c in old['cases'] if c['case']==case)
  with np.load(sp) as z:S=z['intrinsic_effective_surface']/1.36;axis=z['latent_axis'];h=axis[1]-axis[0]
  with np.load(ROOT/item['path']) as z:d={k:z[k] for k in z.files}
  step=float(d['cell_width_kpc']);cells=np.arange(round(16/step))*step-8+step/2;A=cell_projection_matrix(cells,step,axis,h,0);M=moment_matrix(cells,step,axis,h);plane=A@S@A.T;mx=M@S@A.T;my=A@S@M.T;good=plane>0;xx,yy=np.meshgrid(cells,cells,indexing='ij');x=mx[good]/plane[good];y=my[good]/plane[good]
  assert np.max(abs(x-xx[good]))<=step/2+1e-10 and np.max(abs(y-yy[good]))<=step/2+1e-10
  interp=RegularGridInterpolator((axis,axis),S,bounds_error=False,fill_value=0);positive=interp(np.column_stack([x,y]));bad=positive<=0
  # Independent Gaussian integration splits each testcell atallbilinearknots.
  inds=np.flatnonzero(good);chosen=inds[np.linspace(0,len(inds)-1,48,dtype=int)];errs=[];masserrs=[]
  for idx in chosen:
   i,j=np.unravel_index(idx,plane.shape);a,b=cells[i]-step/2,cells[i]+step/2;c,e=cells[j]-step/2,cells[j]+step/2
   ex=np.r_[a,axis[(axis>a)&(axis<b)],b];ey=np.r_[c,axis[(axis>c)&(axis<e)],e];xn=[];xw=[];yn=[];yw=[]
   for edges,ns,ws in [(ex,xn,xw),(ey,yn,yw)]:
    for low,high in zip(edges[:-1],edges[1:]):ns.extend([(low+high)/2-(high-low)/(2*np.sqrt(3)),(low+high)/2+(high-low)/(2*np.sqrt(3))]);ws.extend([(high-low)/2]*2)
   X,Y=np.meshgrid(xn,yn,indexing='ij');values=interp(np.stack([X,Y],axis=-1));w=np.outer(xw,yw)*values;tot=w.sum();cx=(w*X).sum()/tot;cy=(w*Y).sum()/tot;errs.extend([abs(cx-mx[i,j]/plane[i,j]),abs(cy-my[i,j]/plane[i,j])]);masserrs.append(abs(tot-plane[i,j]*step*step)/max(tot,1e-20))
  assert max(errs)<1e-10
  d['x_major_kpc']=x;d['y_minor_kpc']=y;d['radius_kpc']=np.hypot(x,y);d['ring_index_025kpc']=np.floor(d['radius_kpc']/.025).astype(np.int32)
  dest=priv/Path(item['path']).name;np.savez_compressed(dest,**d);assets.append(dict(path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),role=item['role']))
  rows.append(dict(case=case,step_kpc=step,emitters=len(x),radius_min=float(d['radius_kpc'].min()),radius_max=float(d['radius_kpc'].max()),independent_centroid_error_kpc=float(max(errs)),independent_mass_relative_error=float(max(masserrs)),min_underlyingHI=float(positive.min()),positive_location_pass=bool((positive>0).all()),zero_HI_centroid_count=int(bad.sum()),zero_HI_centroid_mass_fraction=float(plane[good][bad].sum()/plane.sum()),mass_flux_arrays_unchanged=True))
 (out/'summary.json').write_text(json.dumps(dict(status='CENTROID_PRIMARY_FINE_PASS_COARSE_FAILURE_RETAINED',assets=assets,cases=rows,observed_response_access=False,script_sha256=sha(Path(__file__)),addendum_sha256=sha(P/'CENTROID_ADDENDUM.md')),indent=2),encoding='utf8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
