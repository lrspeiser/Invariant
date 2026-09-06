import json
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from mond_atlas_clock_relay import A0, G, candidate_grid, loss_select, nfw_mass_shape, predict_logv


def sources():
    r=np.geomspace(0.2, 60, 31)
    return dict(r=r,gas=np.full_like(r,12.),disk=80*np.sqrt(r/(r+3)),bulge=25/np.sqrt(r+1),sb=150*np.exp(-r/3),luminosity=np.full_like(r,20.),hi=np.full_like(r,3.),rd=np.full_like(r,3.))


class ClockRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'configs/mond_atlas_clock_relay_v1.json').read_text(encoding='utf-8'))
        cls.grid=candidate_grid(cls.config)

    def test_grid_order_count_and_finiteness(self):
        self.assertEqual(len(self.grid),713)
        self.assertEqual(list(dict.fromkeys(c['family'] for c in self.grid)),self.config['families'])
        for c in self.grid:
            self.assertTrue(np.isfinite(predict_logv(sources(),c)).all())
        self.assertEqual(self.grid[0],dict(family='newton_fixed',mf=1.))

    def test_zero_strength_and_absorption(self):
        s=sources()
        for c in self.grid:
            base=predict_logv(s,dict(family='newton_ml',mf=c['mf']))
            if c.get('eta',c.get('beta',c.get('kappa',None)))==0:
                np.testing.assert_allclose(predict_logv(s,c),base,atol=1e-12,rtol=0)
            if c['family']=='absorption_proxy':
                self.assertTrue((predict_logv(s,c)<=base+1e-12).all())

    def test_signed_gas_and_mond_independent(self):
        s=sources();s['gas'][0]=-12
        vb2=s['gas']*abs(s['gas'])+.5*s['disk']**2+.7*s['bulge']**2
        np.testing.assert_allclose(10**(2*predict_logv(s,self.grid[0])),vb2,rtol=1e-10)
        gb=vb2/s['r']; gm=10**(2*predict_logv(s,self.grid[1]))/s['r']
        np.testing.assert_allclose(gm*gm/(gm+A0),gb,rtol=1e-10)
        self.assertAlmostEqual(A0*1e6/3.085677581491367e19,1.2e-10,places=22)

    def test_nfw_quadrature_and_cutoff(self):
        for x in np.geomspace(1e-7,100,30):
            reference=x*x*quad(lambda u:u/(1+x*u)**2,0,1,epsabs=1e-13,epsrel=1e-13)[0]
            self.assertLess(abs(float(nfw_mass_shape(x))/reference-1),1e-10)
        s=sources();s['r']=np.geomspace(1e-4,1e5,31)
        c=dict(family='kernel_point',mf=1.,eta=3.,length_factor=1.,cutoff=10.)
        gm=G*1e9*(.5*s['luminosity']+1.33*s['hi'])
        base=10**(2*predict_logv(s,self.grid[0]))/s['r']
        extra=10**(2*predict_logv(s,c))/s['r']-base
        charge=s['r']**2*extra/gm
        np.testing.assert_allclose(charge[-5:],3*nfw_mass_shape(10.),rtol=1e-10)

    def test_independent_potential_derivatives(self):
        s=sources();r=s['r'];L=s['rd'];GM=G*1e9*(.5*s['luminosity']+1.33*s['hi']);eta=3.
        potentials={
            'finite_p2':lambda t:-eta*GM/(t+L),
            'finite_p3':lambda t:-eta*GM/(t+L)+eta*GM*L/(2*(t+L)**2),
            'clock_potential':lambda t:-eta*A0*L*np.log1p(GM/(A0*L*(t+L)))}
        for family,phi in potentials.items():
            c=dict(family=family,mf=1.,eta=eta,length_factor=1.,beta=eta,clock_factor=1.)
            h=r*1e-3
            derivative=(phi(r-2*h)-8*phi(r-h)+8*phi(r+h)-phi(r+2*h))/(12*h)
            base=10**(2*predict_logv(s,self.grid[0]))/r
            extra=10**(2*predict_logv(s,c))/r-base
            np.testing.assert_allclose(extra,derivative,rtol=1e-6)

    def test_effective_source_and_limits(self):
        x=np.geomspace(1e-7,1e7,1000)
        for charge in (x*x/(1+x)**2,x**3/(1+x)**3,x*x/((x+1)*(x+4))):
            self.assertTrue((np.diff(charge)>0).all())
            self.assertLess(abs(charge[-1]-1),1e-6)
        self.assertLess(abs(float(nfw_mass_shape(1e-7))/(1e-7)**2-.5),1e-7)
        # Fixed mass and length-scaled source implies velocity invariance when M/r is fixed.
        s=sources();scaled={k:np.array(v,copy=True) for k,v in s.items()}
        for key in ('r','rd','luminosity','hi'):scaled[key]*=7
        for c in self.grid:
            if c['family'] in ('kernel_point','finite_p2','finite_p3','finite_mixture'):
                np.testing.assert_allclose(predict_logv(s,c),predict_logv(scaled,c),atol=1e-12,rtol=0)

    def test_planted_recovery_and_no_held_label_leakage(self):
        choices=[c for c in self.grid if c['family']=='clock_potential' and c['mf']==1. and c['beta']>0]
        pred=np.array([predict_logv(sources(),c) for c in choices])
        planted=7;target=pred[planted]
        loss=(pred-target[None,:])**2
        train=np.arange(loss.shape[1])%3!=0
        self.assertEqual(loss_select(loss,train),planted)
        perturbed=loss.copy();perturbed[:,~train]=np.nan
        self.assertEqual(loss_select(perturbed,train),planted)
        self.assertEqual(loss_select(np.zeros_like(loss),train),0)
        with self.assertRaises(ValueError):loss_select(loss,np.zeros_like(train))


if __name__=='__main__':
    unittest.main()
