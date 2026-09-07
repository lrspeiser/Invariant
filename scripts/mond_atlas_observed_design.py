"""Source-only1512case catalogue and exact-nuisance prospective refinement."""
from mond_atlas_actual_spectra import ROOT,MATERIALS,ActualSpectrumModel,load_instrument
from mond_atlas_native_selection import spectral_matrix
from mond_atlas_actual_spectra_build import save,digest
from pathlib import Path
import numpy as np,cupy as cp,json,itertools
from cupyx.scipy.special import ndtr
P=ROOT/'work/gravity-first-principles/mond-atlas-observed-design-001'
PAIRS=[('planar','p0p0625_z24','p0p03125_z24'),('vertical','p0p0625_z24','p0p0625_z48'),('joint','p0p0625_z24','p0p03125_z48'),('fine_planar_vertical','p0p03125_z24','p0p03125_z48'),('fine_vertical_planar','p0p0625_z48','p0p03125_z48')]
def catalogue():
 base=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-actual-spectra-001/cache001/summary.json').read_text())['assets'];alt=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-alternative-spectra-001/cache001/summary.json').read_text())['assets'];cases=[]
 for source in ['baseline_h0p1','baseline_h0p4','common30','common30_stars_h0.4','missing_zero','missing_annular']:
  assets=base if source.startswith('baseline') else [a for a in alt if a['case']==source];height='h0p4' if source=='baseline_h0p4' else 'h0p1'
  for model,pressure,spin,branch,material in itertools.product(['newton','newton_plus_log'],[5,10,15],[-1,1],['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'],MATERIALS):
   cases.append(dict(id=len(cases),source=source,height_slot=height,model=model,pressure_reference=pressure,spin=spin,branch=branch,material=material,caches={a['label']:dict(path=a['cache'],sha256=a['sha256']) for a in assets}))
 assert len(cases)==1512;return cases
class Bank:
 def __init__(self):self.arrays={};self.inst=load_instrument()
 def load(self,asset):
  path=asset['path']
  if path not in self.arrays:
   assert digest(ROOT/path)==asset['sha256']
   with np.load(ROOT/path) as z:self.arrays[path]={k:z[k] for k in z.files}
  return self.arrays[path]
 def build(self,case,label):
  a=self.load(case['caches'][label]);f=np.array(MATERIALS[case['material']]);h=case['height_slot'];r=a['radius_kpc'];v2=r*(f@a[f'force_{h}_{case["model"]}']+case['pressure_reference']**2*a['pressure_gradient']/a[f'den_{h}']);valid=a['force_supported']&np.isfinite(v2)&(v2>=0);vel=case['spin']*float(a['sin_inclination'])*np.cos(a['phi_rad'][valid])*np.sqrt(v2[valid]);flux=a['grouped_flux'][:,valid]*f[1];unknown=a['grouped_flux'][:,~valid].sum(1)*f[1]
  return Model(self.inst,case['branch'],vel,flux,unknown,int((~valid).sum()))
class Model:
 def __init__(self,inst,branch,velocity,flux,unknown,invalid):
  self.inst=inst;self.velocity=cp.asarray(velocity);self.flux=cp.asarray(flux);self.unknown_flux=unknown;self.invalid=invalid;H,grid,width=spectral_matrix(63,branch);self.width=abs(inst['velocity_increment_km_s'])*width;self.centers=cp.asarray(inst['first_parent_velocity_km_s']+grid*inst['velocity_increment_km_s']);self.H=cp.asarray(H);self.A=cp.asarray(inst['continuum']);self.bound_coeff=1000*abs(inst['continuum']@H).sum(1)/self.width
 def predict_both(self,parameters):
  systemic,sigma,multiplier=ActualSpectrumModel.parameters(parameters);lo=(self.centers[:,None]-self.width/2-self.velocity[None,:]-systemic)/sigma;hi=lo+self.width/sigma;p=cp.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo));parent=(self.flux@(p/self.width).T)@self.H.T;positive=1000*multiplier*parent[:,11:53];signed=1000*multiplier*(parent@self.A.T);return cp.asnumpy(signed),cp.asnumpy(positive)
 def predict(self,parameters):return self.predict_both(parameters)[0]
 def unknown_envelope(self,parameters):return ActualSpectrumModel.parameters(parameters)[2]*self.unknown_flux[:,None]*self.bound_coeff[None,:]
 def __call__(self,indices,parameters):return self.predict(parameters)[indices]
def refine_point(case,parameters,bank=None):
 bank=bank or Bank();models={label:bank.build(case,label) for label in case['caches']};outputs={label:model.predict_both(parameters) for label,model in models.items()};rows=[];v=bank.inst['velocity_km_s']
 for name,lo,hi in PAIRS:
  a,pa=outputs[lo];b,pb=outputs[hi];L=abs(a-b).sum(1)/abs(b).sum(1);delta=abs(pa@v/pa.sum(1)-pb@v/pb.sum(1))
  for i in range(15):rows.append(dict(comparison=name,aperture=i,profile_L1=float(L[i]),positive_centroid_km_s=float(delta[i]),passed=bool(np.isfinite(L[i]) and np.isfinite(delta[i]) and L[i]<.01 and delta[i]<.5)))
 return dict(rows=rows,passed=all(x['passed'] for x in rows),max_unknown_envelope_mjy_beam=max(float(m.unknown_envelope(parameters).max()) for m in models.values()))
if __name__=='__main__':
 P.mkdir(exist_ok=True);target=P/'catalogue.json';assert not target.exists();save(target,catalogue());print('catalogue1512')
