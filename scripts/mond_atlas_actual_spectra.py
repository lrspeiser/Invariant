"""Conditional valid-emitter prediction plus explicit unknown-emitter envelope."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
from mond_atlas_native_emission_sampled import load_instrument,render_grouped_spectra
from mond_atlas_native_selection import spectral_matrix
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-actual-spectra-001'
PRIVATE=ROOT/'work/private/mond-atlas-actual-spectra-001'
MATERIALS={'baseline':(1.,1.,1.),'stellar_ML_0.4':(2/3,1.,1.),'stellar_ML_0.8':(4/3,1.,1.),'HI_0.8':(1.,.8,1.),'HI_1.2':(1.,1.2,1.),'CO_0.5':(1.,1.,.5),'CO_2':(1.,1.,2.)}

class ActualSpectrumModel:
    """Reusable callback; .predict([systemic,sigma,multiplier]) ->15x42."""
    def __init__(self,cache_path,height='h0p1',model='newton',pressure_reference=10.,spin=1,branch='boxcar_independent',material='baseline'):
        self.cache_path=Path(cache_path);self.instrument=load_instrument();self.height=height;self.model=model;self.pressure_reference=float(pressure_reference);self.spin=spin;self.branch=branch;self.material=material
        if height not in ['h0p1','h0p4'] or model not in ['newton','newton_plus_log'] or pressure_reference not in [5.,10.,15.] or spin not in [-1,1] or material not in MATERIALS:raise ValueError('undeclared physical case')
        with np.load(cache_path) as z:
            radius=z['radius_kpc'];phi=z['phi_rad'];grouped=z['grouped_flux'];force=z[f'force_{height}_{model}'];den=z[f'den_{height}'];gradient=z['pressure_gradient'];inside=z['force_supported'];sini=float(z['sin_inclination'])
        factors=np.array(MATERIALS[material]);g=factors@force;v2=np.full(len(radius),np.nan);v2[inside]=radius[inside]*(g[inside]+pressure_reference**2*gradient[inside]/den[inside]);valid=inside&np.isfinite(v2)&(v2>=0)
        self.rotation_squared=v2;self.valid=valid;self.radius=radius;self.invalid_count=int((~valid).sum());self.global_steady_valid=bool(valid.all());self.grouped_flux=grouped[:,valid]*factors[1];self.velocity=spin*sini*np.cos(phi[valid])*np.sqrt(v2[valid]);self.invalid_weighted_flux=grouped[:,~valid].sum(axis=1)*factors[1]
        H,_,width=spectral_matrix(self.instrument['parent_channels'],branch);self.bound_coefficients=1000*np.sum(abs(self.instrument['continuum']@H),axis=1)/(abs(self.instrument['velocity_increment_km_s'])*width)
    @staticmethod
    def parameters(parameters):
        p=np.asarray(parameters,float)
        if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<[-30.,3.,.5]) or np.any(p>[30.,20.,2.]):raise ValueError('Parameters must lie in frozen [systemic,sigma,multiplier] bounds')
        return p
    def predict(self,parameters):
        systemic,sigma,multiplier=self.parameters(parameters)
        return render_grouped_spectra(self.grouped_flux,self.velocity,sigma,self.instrument,self.branch,systemic,multiplier)['spectra_mjy_beam']
    def unknown_envelope(self,parameters):
        _,_,multiplier=self.parameters(parameters)
        return multiplier*self.invalid_weighted_flux[:,None]*self.bound_coefficients[None,:]
    def predict_with_envelope(self,parameters):
        return dict(spectra_mjy_beam=self.predict(parameters),absolute_unknown_envelope_mjy_beam=self.unknown_envelope(parameters),global_steady_valid=self.global_steady_valid,invalid_groups=self.invalid_count)
    __call__=predict
