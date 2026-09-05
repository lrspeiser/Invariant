"""Sign-preserving, finer-ring conditional cube model with train-only multistarts.

This improves the projected pilot; finite thickness and native beam velocity
mixing remain separate validation requirements. No gravitational mass law here.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from gravity_cube_model import CubeModel,tensor

def interpolate(r,knots,values):
    index=torch.bucketize(r.contiguous(),knots).clamp(1,len(knots)-1)
    fraction=((r-knots[index-1])/(knots[index]-knots[index-1])).clamp(0,1)
    return values[index-1]*(1-fraction)+values[index]*fraction

class ConstrainedCube(CubeModel):
    def __init__(self,packet):
        super().__init__(packet)
        self.rings=tensor([0,40,80,140,220,320,450,600])
        self.spin=1. if np.sum(packet['rotation_initial'])>=0 else -1.

    def geometry(self,p,mode):
        radius=self.radius
        # Smooth implicit tilted-ring geometry, damped to avoid oscillation.
        for _ in range(6):
            rr=(radius/600).clamp(0,1)
            pa=self.base_pa+p[15]*np.deg2rad(10)
            inc=self.base_inc+p[16]*np.deg2rad(5)
            if mode in ('warp','full'):
                pa=pa+p[17]*np.deg2rad(15)*rr**2
                inc=inc+p[18]*np.deg2rad(8)*rr**2
            inc=torch.clamp(inc,np.deg2rad(10),np.deg2rad(85))
            major=self.x*torch.sin(pa)+self.y*torch.cos(pa)
            minor=(self.x*torch.cos(pa)-self.y*torch.sin(pa))/torch.cos(inc)
            current=torch.sqrt(major**2+minor**2+1e-8)
            radius=.5*(radius+current)
        return current,major,minor,inc

    def render(self,p,mode,gas_beta=0.,context=None):
        current,major,minor,inc=self.geometry(p,mode)
        speed=interpolate(current,self.rings,torch.cat([p.new_zeros(1),p[:7]*200]))
        c=self.gas if context is None else context
        speed=speed*(1+gas_beta*c)
        los=self.spin*speed*major/current
        if mode in ('stream','full'):
            rr=(current/600).clamp(0,1)
            los=los+.3*speed*(p[19]*(1-rr)+p[20]*rr)*minor/current
        center=self.vsys+p[7]*30+torch.sin(inc)*los
        sigma=interpolate(current,tensor([0,220,600]),p[8:11]*10)
        amplitude=self.amp*torch.exp(interpolate(current,tensor([0,140,320,600]),p[11:15]))
        lower=self.edges[:-1,None,None];upper=self.edges[1:,None,None]
        def profile(mean,width):
            return .5*(torch.erf((upper-mean)/(np.sqrt(2)*width))-torch.erf((lower-mean)/(np.sqrt(2)*width)))/(upper-lower)
        shape=profile(center,sigma)
        if mode in ('asymmetric','full'):
            sys=self.vsys+p[7]*30
            shape=(1-p[21])*shape+p[21]*profile(sys+(center-sys)*(1-p[22]),sigma*1.7)
        return self.convolve(amplitude*shape)

    def loss(self,p,mode,which='train',gas_beta=0.,context=None,penalize=True):
        prediction=self.render(p,mode,gas_beta,context)
        mask=self.train if which=='train' else self.test
        residual=self.white@(prediction-self.data)[:,mask]
        value=torch.mean(residual**2)
        if penalize:
            value=value+.02*torch.mean((p[2:7]-2*p[1:6]+p[:5])**2)
            value=value+.01*torch.mean(p[11:15]**2)+.01*torch.mean(p[15:17]**2)
        return value

    @staticmethod
    def bounds():
        return [(0,3)]*7+[(-5,5)]+[(.3,4)]*3+[(-1.5,1.5)]*4+[(-2,2)]*2+[(-1,1)]*4+[(0,.4),(0,.6)]

    def seed(self,factor=1):
        p=np.zeros(23)
        p[:7]=np.interp([40,80,140,220,320,450,600],np.linspace(0,600,5),np.abs(self.p['rotation_initial']))/200*factor
        p[8:11]=1.2;p[21]=.06;p[22]=.2
        return p

    def fit_once(self,mode,initial,maxiter=280):
        p=np.array(initial,dtype=float).copy();active=list(range(17))
        if mode in ('warp','full'):active += [17,18]
        if mode in ('stream','full'):active += [19,20]
        if mode in ('asymmetric','full'):active += [21,22]
        bounds=self.bounds()
        def objective(x):
            whole=p.copy();whole[active]=x
            t=tensor(whole).requires_grad_(True);loss=self.loss(t,mode);loss.backward()
            return float(loss.detach()),t.grad.detach().cpu().numpy()[active].astype(float)
        fit=minimize(objective,p[active],method='L-BFGS-B',jac=True,bounds=[bounds[i] for i in active],
            options={'maxiter':maxiter,'ftol':2e-7,'gtol':1e-4,'maxls':40})
        p[active]=fit.x
        # A float32 plateau can cause line-search termination. Retain its status.
        with torch.no_grad():train=float(self.loss(tensor(p),mode,penalize=False))
        atbound=[int(i) for i in active if min(abs(p[i]-bounds[i][0]),abs(p[i]-bounds[i][1]))<1e-3]
        return dict(params=p.tolist(),train_loss=train,objective=float(fit.fun),mode=mode,
            optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),at_bounds=atbound)

    def fit_multistart(self,mode,seeds,maxiter=280):
        candidates=[self.fit_once(mode,p,maxiter) for p in seeds]
        # No access to held-out spectra until optimization/selection is complete.
        good=[c for c in candidates if c['optimizer_success']]
        pool=good or candidates
        chosen=min(pool,key=lambda a:a['objective']).copy()
        with torch.no_grad():chosen['test_loss']=float(self.loss(tensor(chosen['params']),mode,'test',penalize=False))
        chosen['starts']=candidates;chosen['all_starts_failed']=not bool(good)
        chosen['train_objective_spread']=float(max(c['objective'] for c in candidates)-min(c['objective'] for c in candidates))
        return chosen
