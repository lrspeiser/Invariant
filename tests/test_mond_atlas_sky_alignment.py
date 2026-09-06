import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_sky_alignment import vector,sky_features,residualizer,associations
from mond_atlas_pattern_learning import outer_prediction


class SkyTests(unittest.TestCase):
    def test_axis_symmetry_and_rotation(self):
        a=vector(239.,64.3); s=vector(np.array([0,90,239]),np.array([0,0,64.3]))
        np.testing.assert_allclose(np.linalg.norm(s,axis=1),1)
        self.assertAlmostEqual(s[-1]@a,1)
        np.testing.assert_allclose((s@a)**2,(s@(-a))**2)
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose((s@rot)@(a@rot),s@a)

    def test_independent_galactic_rotation(self):
        # Standard J2000 rotation coefficients, independent of Astropy conversion.
        matrix=np.array([[-.0548755604,-.8734370902,-.4838350155],[.4941094279,-.4448296300,.7469822445],[-.8676661490,-.1980763734,.4559837762]])
        ra=np.array([0.,80.,190.,310.]);dec=np.array([0.,-35.,70.,20.])
        _,_,meta=sky_features(ra,dec,{'quadrupole':[224.2,69.2],'octopole':[239.,64.3]})
        np.testing.assert_allclose(vector(ra,dec)@matrix.T,vector(meta['l_deg'],meta['b_deg']),atol=1e-6)

    def test_residual_and_planted(self):
        rng=np.random.default_rng(39);x=rng.normal(size=(100,3));z=rng.normal(size=(100,2));y=2*x[:,0]+.8*z[:,0]
        m,_=residualizer(x);a=np.column_stack([np.ones(100),x]);ref=y-a@np.linalg.lstsq(a,y,rcond=None)[0]
        np.testing.assert_allclose(m@y,ref,atol=1e-12)
        rows,_,_,_,_=associations(x,y,z,['signal','null'],31)
        self.assertAlmostEqual(rows[0]['partial_r'],1.)
        self.assertAlmostEqual(rows[0]['partial_slope_dex_per_feature'],.8)

    def test_predictive_outer_label_isolation(self):
        rng=np.random.default_rng(21);x=rng.normal(size=(50,5));y=rng.normal(size=50);fold=np.arange(50)%5
        cfg={'ridge_penalties':[.1,1.,10.]};p,c=outer_prediction(x,y,fold,0,'linear_ridge',cfg)
        y[fold==0]+=1000;q,d=outer_prediction(x,y,fold,0,'linear_ridge',cfg)
        np.testing.assert_array_equal(p,q);self.assertEqual(c,d)

if __name__=='__main__':unittest.main()
