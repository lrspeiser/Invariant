"""Algebraically reordered float64 CUDA spectral callback; no new physical law."""
import numpy as np
import cupy as cp
from cupyx.scipy.special import ndtr
from mond_atlas_native_selection import spectral_matrix


class GPUCallback:
    def __init__(self, model):
        self.model=model
        self.flux=cp.asarray(model.grouped_flux,dtype=cp.float64)
        self.velocity=cp.asarray(model.velocity,dtype=cp.float64)
        H,grid,width=spectral_matrix(model.instrument['parent_channels'],model.branch)
        self.centers=cp.asarray(model.instrument['first_parent_velocity_km_s']+
                                grid*model.instrument['velocity_increment_km_s'])
        self.width=abs(model.instrument['velocity_increment_km_s'])*width
        self.H=cp.asarray(H,dtype=cp.float64)
        self.A=cp.asarray(model.instrument['continuum'],dtype=cp.float64)

    def predict(self, parameters):
        systemic,sigma,multiplier=self.model.parameters(parameters)
        low=(self.centers[:,None]-self.width/2-self.velocity[None,:]-systemic)/sigma
        high=low+self.width/sigma
        probability=cp.where(low>=0,ndtr(-low)-ndtr(-high),ndtr(high)-ndtr(low))
        # Sum emitters before common linear spectral operators. This is exact
        # algebra, not averaging velocities or replacing the spectrum by moments.
        pregrid=self.flux@(probability/self.width).T
        result=1000*multiplier*(pregrid@self.H.T@self.A.T)
        return cp.asnumpy(result)

    def __call__(self, indices, parameters):
        return self.predict(parameters)[list(indices)]

    def unknown_envelope(self, parameters):
        return self.model.unknown_envelope(parameters)
