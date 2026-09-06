import sys,json
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_clock_scale_repair import predict_repaired,FAMILIES
from mond_atlas_clock_relay import G,A0,candidate_grid,predict_logv

def source():
    r=np.geomspace(.1,100,20)
    return dict(r=r,rd=r*0+3,luminosity=r*0+20,hi=r*0+3,gas=r*0+10,disk=r*0+30,bulge=r*0,sb=r*0+20)

class RepairTests(unittest.TestCase):
    def test_all_finite_zero_strength(self):
        config=json.loads((ROOT/'configs/mond_atlas_clock_relay_v1.json').read_text(encoding='utf-8'))
        for c in candidate_grid(config):
            if c['family'] not in FAMILIES:continue
            pred=predict_repaired(source(),c);self.assertTrue(np.isfinite(pred).all())
            if c.get('eta',c.get('beta'))==0:
                np.testing.assert_allclose(pred,predict_logv(source(),dict(family='newton_ml',mf=c['mf'])),atol=1e-12,rtol=0)

    def test_clock_gradient_and_scaling(self):
        s=source();r=s['r'];GM=G*1e9*(.5*s['luminosity']+1.33*s['hi']);psi=np.sqrt(GM*A0);beta=3
        phi=lambda t:-beta*psi*np.log1p(GM/(psi*(t+s['rd'])))
        h=r*1e-3;deriv=(phi(r-2*h)-8*phi(r-h)+8*phi(r+h)-phi(r+2*h))/(12*h)
        c=dict(family='clock_potential',mf=1.,beta=beta,clock_factor=1.)
        extra=(10**(2*predict_repaired(s,c))-550)/r
        np.testing.assert_allclose(extra,deriv,rtol=1e-6)
        np.testing.assert_allclose(np.sqrt(4*GM/A0),2*np.sqrt(GM/A0),rtol=1e-10)
        np.testing.assert_allclose((beta*np.sqrt(4*GM*A0))**2,4*(beta*psi)**2,rtol=1e-10)

    def test_independent_finite_formulas(self):
        s=source();r=s['r'];GM=G*1e9*(.5*s['luminosity']+1.33*s['hi']);L=2*np.sqrt(GM/A0)
        for family,extra in [('finite_p2',3*GM/(r+L)**2),('finite_p3',3*GM*r/(r+L)**3)]:
            c=dict(family=family,mf=1.,eta=3.,length_factor=2.)
            np.testing.assert_allclose(10**(2*predict_repaired(s,c)),550+r*extra,rtol=1e-10)

if __name__=='__main__':unittest.main()
