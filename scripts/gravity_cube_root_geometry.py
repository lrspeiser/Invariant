"""Bracketed warped-ring radius with implicit differentiation.

Accuracy of a selected root does not establish a unique physical 3D projection.
Folded/multiple-intersection geometries still require a full emitting disk model.
"""
import numpy as np
import torch
from gravity_cube_constrained import ConstrainedCube

class RootGeometryCube(ConstrainedCube):
    root_iterations=36

    def ring_mapping(self,p,mode,r):
        pa=self.base_pa+p[15]*np.deg2rad(10)
        unbounded_inc=self.base_inc+p[16]*np.deg2rad(5)
        dp=torch.zeros_like(r);di=torch.zeros_like(r)
        if mode in ('warp','full'):
            rr=(r/600).clamp(0,1)
            pa=pa+p[17]*np.deg2rad(15)*rr**2
            unbounded_inc=unbounded_inc+p[18]*np.deg2rad(8)*rr**2
            dp=p[17]*np.deg2rad(15)*2*r/600**2*(r<600)
            di=p[18]*np.deg2rad(8)*2*r/600**2*(r<600)
        inc=torch.clamp(unbounded_inc,np.deg2rad(10),np.deg2rad(85))
        di=di*((unbounded_inc>np.deg2rad(10))&(unbounded_inc<np.deg2rad(85)))
        a=self.x*torch.sin(pa)+self.y*torch.cos(pa)
        c=self.x*torch.cos(pa)-self.y*torch.sin(pa);b=c/torch.cos(inc)
        f=torch.sqrt(a*a+b*b+1e-8)
        da=c*dp;db=(-a*dp+c*torch.tan(inc)*di)/torch.cos(inc)
        derivative=(a*da+b*db)/f
        return f,derivative,a,b,inc

    def geometry(self,p,mode):
        if mode not in ('warp','full'):
            f,_,a,b,inc=self.ring_mapping(p,mode,self.radius)
            return f,a,b,inc
        with torch.no_grad():
            lower=torch.zeros_like(self.radius)
            upper=torch.hypot(self.x,self.y)/np.cos(np.deg2rad(85))+1
            for _ in range(self.root_iterations):
                mid=(lower+upper)/2;f=self.ring_mapping(p,mode,mid)[0]
                lower=torch.where(f>mid,mid,lower);upper=torch.where(f>mid,upper,mid)
            root=(lower+upper)/2
        f,derivative,_,_,_=self.ring_mapping(p,mode,root)
        jacobian=1-derivative
        # Implicit dr/dp = (partial F/partial p)/(1-partial F/partial r).
        # Near-singular cases are explicitly flagged by diagnostics, not admitted.
        denominator=torch.where(jacobian.abs()<1e-3,torch.ones_like(jacobian)*1e-3,jacobian).detach()
        radius=root+(f-root)/denominator
        _,_,a,b,inc=self.ring_mapping(p,mode,radius)
        return radius,a,b,inc

    def geometry_diagnostic(self,p,mode='full'):
        with torch.no_grad():
            r,_,_,_=self.geometry(p,mode);f,derivative,_,_,_=self.ring_mapping(p,mode,r)
            mask=self.train|self.test
            # Scan the radius-dependent part for a nonmonotone mapping. Beyond
            # 600 arcsec angles are constant. This is a conservative fold flag.
            minimum=torch.ones_like(r)
            for sample in np.linspace(0,600,121):
                slope=self.ring_mapping(p,mode,torch.ones_like(r)*sample)[1]
                minimum=torch.minimum(minimum,1-slope)
            return dict(max_root_residual_arcsec=float(torch.max(torch.abs(f-r)[mask])),
                minimum_root_jacobian=float(torch.min((1-derivative)[mask])),
                possible_fold_fraction=float(torch.mean((minimum[mask]<=0).float())),
                note='Possible-fold scan is diagnostic; unique line-of-sight structure is not certified.')
