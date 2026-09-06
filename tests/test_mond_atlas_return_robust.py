import sys,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mond_atlas_return_robust import G,A0,BOUNDS,extra_acceleration,predict,fit


def synthetic():
    n=80;r=np.tile(np.geomspace(.2,30,20),4);scale=np.repeat([1.,2.,3.,4.],20)
    return dict(r=r,rd=scale,gas=8+np.sqrt(r),disk=20*np.sqrt(r)/(r+scale),bulge=np.ones(n)*2,luminosity=scale*2,hi=scale*.7,sb=np.ones(n)),np.repeat(np.arange(4),20)


class ReturnRobustTests(unittest.TestCase):
    def test_zero_and_units(self):
        s,gi=synthetic();newton=.5*np.log10(s['gas']**2+.5*s['disk']**2+.7*s['bulge']**2)
        for family in list(BOUNDS)[1:]:
            p={k:(a+b)/2 for k,(a,b) in BOUNDS[family].items()};p['mf']=1.;p['eta' if family=='finite_flat_bridge' else 'A']=0.
            np.testing.assert_allclose(predict(s,family,p),newton,rtol=1e-12)
            p['eta' if family=='finite_flat_bridge' else 'A']=2.
            # Rescale all lengths by u and GM,A0 by u^2,1: rM and Rd both scale u.
            g=extra_acceleration(s['r'],s['rd'],3000.,family,p)
            np.testing.assert_allclose(extra_acceleration(7*s['r'],7*s['rd'],49*3000.,family,p),g,rtol=1e-12)

    def test_density_integral(self):
        p=dict(A=2.,length_factor=1.,t=0.,C=10.)
        for r in [.01,.1,1.,10.,100.]:
            enclosed=quad(lambda u:u/(1+u)**2,0,min(r,10),epsabs=1e-13)[0]
            self.assertAlmostEqual(float(extra_acceleration(r,1.,1.,'truncated_point_kernel',p))*r*r,2*enclosed,places=11)

    def test_potential_mass_and_limits(self):
        for f in list(BOUNDS)[1:]:
            p={k:(a+b)/2 for k,(a,b) in BOUNDS[f].items()};p['length_factor']=2.;p['t']=.5;p['C']=10.
            fn=lambda r:float(extra_acceleration(r,1.,100.,f,p))
            radii=np.geomspace(1e-4,1e8,300);mass=np.array([r*r*fn(r) for r in radii])
            self.assertTrue(np.all(np.diff(mass)>=-1e-8));self.assertGreater(mass[-1],0)
            self.assertLess(abs(mass[-1]/mass[-2]-1),1e-5)
            for r in [.3,2.,20.]:
                h=r*1e-4
                if f=='truncated_point_kernel':
                    cutoff=p['C']*p['length_factor']*np.sqrt(100./A0)**p['t']
                    tail=fn(cutoff)*cutoff**2
                    phi=lambda x:-(quad(fn,x,cutoff,epsabs=1e-9,epsrel=1e-10)[0]+tail/cutoff) if x<cutoff else -tail/x
                else:phi=lambda x:-quad(fn,x,np.inf,epsabs=1e-9,epsrel=1e-10)[0]
                self.assertLess(abs((phi(r+h)-phi(r-h))/(2*h)/fn(r)-1),1e-5)

    def test_planted_and_training_isolation(self):
        s,gi=synthetic();train=np.array([True,True,True,False])
        for family in BOUNDS:
            p={k:a+.35*(b-a) for k,(a,b) in BOUNDS[family].items()};y=predict(s,family,p)
            recovered,starts=fit(s,y,gi,train,family)
            self.assertLess(np.mean((predict(s,family,recovered)[gi<3]-y[gi<3])**2),1e-8)
            changed=y.copy();changed[gi==3]+=50
            again,_=fit(s,changed,gi,train,family)
            np.testing.assert_allclose(list(recovered.values()),list(again.values()),atol=1e-10,rtol=0)


if __name__=='__main__':unittest.main()
