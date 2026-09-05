"""Differentiable, conditional spectral-cube model and numerical controls.

This is a coarse projected kinematic model, not a unique 3D density lift.
The source template is supplied independently of held-out velocity fitting.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from scipy.optimize import minimize

torch.set_num_threads(1)
DEVICE='cuda'

def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=DEVICE)

def hats(radius,knots):
    return torch.stack([torch.clamp(1-torch.abs(radius-k)/(knots[1]-knots[0]),min=0) for k in knots])

class CubeModel:
    def __init__(self,packet):
        self.p=packet
        self.x=tensor(packet['east']);self.y=tensor(packet['north'])
        self.amp=tensor(packet['amplitude']);self.edges=tensor(packet['velocity_edges'])
        self.h,self.w=self.x.shape;self.nc=len(self.edges)-1
        self.radius=tensor(packet['radius']);self.knots=tensor(np.linspace(0,packet['rmax'],5))
        self.basis=hats(self.radius,self.knots)
        self.beam=tensor(packet['beam_transfer'])
        self.base_pa=float(packet['pa']);self.base_inc=float(packet['inc'])
        self.gas=tensor(packet['gas_context'])
        self.rscale=max(float(packet['rmax']),1.)
        self.scale=float(packet['speed_scale'])
        self.vsys=float(packet['vsys_initial'])
        self.data=tensor(packet['cube']);self.white=tensor(packet['whitener'])
        self.train=torch.as_tensor(packet['train_mask'],device=DEVICE)
        self.test=torch.as_tensor(packet['test_mask'],device=DEVICE)

    def convolve(self,cube):
        # Zero-padded circular transfer is linear convolution over the central field.
        padded=torch.nn.functional.pad(cube,(self.w//2,self.w//2,self.h//2,self.h//2))
        out=torch.fft.irfft2(torch.fft.rfft2(padded)*self.beam,s=padded.shape[-2:])
        return out[:,self.h//2:self.h//2+self.h,self.w//2:self.w//2+self.w]

    def render(self,params,mode,gas_beta=0.,context=None):
        # Parameter scales make bounded optimization reasonably conditioned.
        rot=torch.einsum('k,khw->hw',params[:5],self.basis)*self.scale
        sys=self.vsys+params[5]*30
        sigma=3+torch.nn.functional.softplus(params[6])*10
        amplitude=self.amp*torch.exp(torch.einsum('k,khw->hw',params[7:12],self.basis))
        rr=torch.clamp(self.radius/self.rscale,0,1)
        pa=torch.ones_like(rr)*self.base_pa;inc=torch.ones_like(rr)*self.base_inc
        if mode in ('warp','full'):
            pa=pa+params[12]*np.deg2rad(15)*rr**2
            inc=inc+params[13]*np.deg2rad(8)*rr**2
        major=self.x*torch.sin(pa)+self.y*torch.cos(pa)
        minor=(self.x*torch.cos(pa)-self.y*torch.sin(pa))/torch.cos(inc)
        r=torch.sqrt(major**2+minor**2+1e-5);ct=major/r;st=minor/r
        c=self.gas if context is None else context
        rot=rot*(1+gas_beta*c)
        los=rot*ct
        if mode in ('stream','full'):
            vr=self.scale*.3*(params[14]*(1-rr)+params[15]*rr)
            los=los+vr*st
        center=sys+torch.sin(inc)*los
        lower=self.edges[:-1,None,None];upper=self.edges[1:,None,None]
        def profile(mean,width):
            return .5*(torch.erf((upper-mean)/(1.41421356237*width))-torch.erf((lower-mean)/(1.41421356237*width)))/(upper-lower)
        shape=profile(center,sigma)
        if mode in ('asymmetric','full'):
            fraction=params[16]
            lag=params[17]
            secondary=sys+(center-sys)*(1-lag)
            shape=(1-fraction)*shape+fraction*profile(secondary,sigma*1.7)
        return self.convolve(amplitude*shape)

    def loss(self,params,mode,which='train',gas_beta=0.,context=None,penalize=True):
        prediction=self.render(params,mode,gas_beta,context)
        mask=self.train if which=='train' else self.test
        residual=(prediction-self.data)[:,mask]
        whitened=self.white@residual
        value=torch.mean(whitened**2)
        if penalize:
            value=value+.02*torch.mean((params[2:5]-2*params[1:4]+params[:3])**2)+.01*torch.mean(params[7:12]**2)
        return value

    def fit(self,mode,initial=None,maxiter=100):
        p=np.zeros(18,dtype=np.float64) if initial is None else np.array(initial,dtype=float)
        if initial is None:
            p[:5]=self.p['rotation_initial']/self.scale;p[6]=.5;p[16]=.08;p[17]=.15
        active=list(range(12))
        if mode in ('warp','full'):active += [12,13]
        if mode in ('stream','full'):active += [14,15]
        if mode in ('asymmetric','full'):active += [16,17]
        bounds=[(-2.5,2.5)]*5+[(-5,5),(-3,3)]+[(-1.5,1.5)]*5+[(-1,1)]*4+[(0,.4),(0,.6)]
        def objective(x):
            whole=p.copy();whole[active]=x
            t=tensor(whole).requires_grad_(True);loss=self.loss(t,mode);loss.backward()
            return float(loss.detach().cpu()),t.grad.detach().cpu().numpy()[active].astype(float)
        result=minimize(objective,p[active],method='L-BFGS-B',jac=True,bounds=[bounds[i] for i in active],
            options={'maxiter':maxiter,'ftol':2e-8,'gtol':2e-5,'maxls':20})
        p[active]=result.x
        with torch.no_grad():train=float(self.loss(tensor(p),mode,penalize=False));test=float(self.loss(tensor(p),mode,'test',penalize=False))
        return dict(params=p.tolist(),train_loss=train,test_loss=test,optimizer_success=bool(result.success),
            message=str(result.message),iterations=int(result.nit),objective=float(result.fun),mode=mode)

def controls():
    torch.manual_seed(709)
    n=32;yy,xx=np.mgrid[:n,:n];x=(xx-15.5)*3;y=(yy-15.5)*3
    rad=np.hypot(x,y/np.cos(.8));knots=np.linspace(0,100,5)
    fy=np.fft.fftfreq(2*n);fx=np.fft.rfftfreq(2*n)
    beam=np.exp(-2*np.pi**2*1.5**2*(fy[:,None]**2+fx[None,:]**2))
    cov=.25**np.abs(np.arange(32)[:,None]-np.arange(32)[None,:]);white=np.linalg.inv(np.linalg.cholesky(cov*.002**2))
    train=(xx<12);test=(xx>20)
    packet=dict(east=x,north=y,amplitude=np.exp(-rad/35)*10,velocity_edges=np.linspace(-120,120,33),
        radius=rad,rmax=100,beam_transfer=beam,pa=.3,inc=.8,gas_context=np.sin(x/30),speed_scale=100,
        vsys_initial=0,cube=np.zeros((32,n,n)),whitener=white,train_mask=train,test_mask=test,
        rotation_initial=np.array([0,.5,.8,1.,1.])*100)
    m=CubeModel(packet);p=np.zeros(18);p[:5]=[0,.5,.8,1,1];p[6]=.2
    with torch.no_grad():truth=m.render(tensor(p),'rotation').cpu().numpy()
    m.data=tensor(truth)
    # Autograd against central finite differences, including beam and whitening.
    q=p.copy();q[2]+=.03;t=tensor(q).requires_grad_(True);loss=m.loss(t,'rotation');loss.backward()
    eps=.003;lo=q.copy();hi=q.copy();lo[2]-=eps;hi[2]+=eps
    finite=(float(m.loss(tensor(hi),'rotation'))-float(m.loss(tensor(lo),'rotation')))/(2*eps)
    grad=float(t.grad[2]);relative=abs(grad-finite)/max(abs(grad),1)
    assert relative<.01,(relative,grad,finite)
    # Train-only fit must recover unseen noise-free spectra, not just training pixels.
    initial=p.copy();initial[:5]*=.9;initial[5]=.2
    recovered=m.fit('rotation',initial,maxiter=180)
    assert recovered['test_loss']<.02,recovered
    # Whiten a separate synthetic correlated noise draw.
    rng=np.random.default_rng(710);z=rng.normal(size=(32,20000));correlated=np.linalg.cholesky(cov)@z
    empirical=np.cov(np.linalg.inv(np.linalg.cholesky(cov))@correlated)
    whitening=float(np.max(abs(empirical-np.eye(32))));assert whitening<.04
    # Geometric masks do not change with noise; guard creates no shared selected pixels.
    assert not np.any(train&test)
    return dict(gradient_relative_error=relative,noise_free_heldout_whitened_loss=recovered['test_loss'],
        synthetic_whitening_max_covariance_error=whitening,geometric_mask_response_independent=True,
        note='Controls validate numerical conditional cube machinery, not unique physical identification of a warp versus radial streaming.')
