"""Exact linear regrouping: apply H/A after aperture spectral aggregation."""
from mond_atlas_actual_spectra import ActualSpectrumModel as OriginalModel
from mond_atlas_native_emission import normal_interval
from mond_atlas_native_selection import spectral_matrix
import numpy as np

class ActualSpectrumModel(OriginalModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._H,grid,width=spectral_matrix(self.instrument['parent_channels'],self.branch);self._dv=abs(self.instrument['velocity_increment_km_s'])*width
        self._centers=self.instrument['first_parent_velocity_km_s']+grid*self.instrument['velocity_increment_km_s'];active=np.any(self.grouped_flux>0,axis=0);self._flux=self.grouped_flux[:,active];self._vel=self.velocity[active]
    def predict(self,parameters):
        systemic,sigma,multiplier=self.parameters(parameters);mu=self._vel+systemic
        bins=normal_interval((self._centers[:,None]-self._dv/2-mu)/sigma,(self._centers[:,None]+self._dv/2-mu)/sigma)/self._dv
        aperture_pre=1000*multiplier*self._flux@bins.T
        return aperture_pre@self._H.T@self.instrument['continuum'].T
    __call__=predict
