import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_stellar_transfer import bilinear_reference,sample,fit_shift,score


class StellarTransferTests(unittest.TestCase):
    def test_bilinear_reference(self):
        rng=np.random.default_rng(91); a=rng.normal(size=(20,22)); xy=rng.uniform([1,1],[19,17],size=(80,2))
        np.testing.assert_allclose(sample(a,xy,[0,0]),bilinear_reference(a,xy),atol=1e-12)

    def test_planted_fractional_shift_on_separate_patches(self):
        yy,xx=np.mgrid[:110,:110]
        a=np.exp(-((xx-37)**2+(yy-44)**2)/130)+.7*np.exp(-((xx-67)**2+(yy-65)**2)/90)+.2*np.sin(xx/5)*np.cos(yy/8)
        y,x=np.mgrid[15:95:2,15:95:2]; xy=np.column_stack([x.ravel(),y.ravel()])
        shift=np.array([2.3,-1.7]); target=1.17*bilinear_reference(a,xy+shift)+.031
        train=(xy[:,0]//20+xy[:,1]//20)%2==0
        result=fit_shift(a,xy[train],target[train],4)
        np.testing.assert_allclose(result['shift'],shift,atol=.02)
        held=score(sample(a,xy[~train],result['shift']),target[~train],result['scale'],result['offset'])
        self.assertLess(held['relative_rms'],.005)

    def test_unresolved_flat_calibration_fails(self):
        with self.assertRaises(ValueError):fit_shift(np.ones((40,40)),np.array([[10.,10.],[20.,20.]]),np.array([1.,2.]),2)

    def test_zero_shift_and_axis_order(self):
        yy,xx=np.mgrid[:50,:50]; a=np.sin(xx/3)+np.cos(yy/5)
        xy=np.array([[10.,15.],[20.,30.],[35.,12.],[15.,35.],[31.,30.]])
        truth=sample(a,xy,[0,0]);result=fit_shift(a,xy,truth,2)
        np.testing.assert_allclose(result['shift'],[0,0],atol=1e-6)


if __name__=='__main__':unittest.main()
