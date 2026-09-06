import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_pattern_learning import galaxy_folds, predict, outer_prediction, synthetic_controls


class PatternLearningTests(unittest.TestCase):
    def test_stable_galaxy_partition_and_duplicate_guard(self):
        names=['g'+str(i) for i in range(30)]
        a=dict(zip(names,galaxy_folds(names,19)))
        self.assertEqual(a,dict(zip(names[::-1],galaxy_folds(names[::-1],19))))
        with self.assertRaises(ValueError): galaxy_folds(['x','x'],19)

    def test_outer_response_cannot_change_its_prediction(self):
        rng=np.random.default_rng(66); x=rng.normal(size=(30,3)); y=rng.normal(size=30)
        fold=galaxy_folds([str(i) for i in range(30)],71)
        config=dict(ridge_penalties=[.1,1.],rbf_gamma_multipliers=[.25,1.])
        a,pa=outer_prediction(x,y,fold,2,'rbf_kernel_ridge',config)
        modified=y.copy(); modified[fold==2]+=100000
        b,pb=outer_prediction(x,modified,fold,2,'rbf_kernel_ridge',config)
        np.testing.assert_array_equal(a,b); self.assertEqual(pa,pb)

    def test_constant_response_and_unit_rescaling(self):
        rng=np.random.default_rng(9); a=rng.normal(size=(30,3)); b=rng.normal(size=(8,3))
        for estimator in ['linear_ridge','rbf_kernel_ridge']:
            np.testing.assert_allclose(predict(a,np.full(30,4.),b,estimator,1.),4.)
            y=rng.normal(size=30)
            p=predict(a,y,b,estimator,1.)
            q=predict(a*[1e3,1e-2,7]+[10,-10,1],y,b*[1e3,1e-2,7]+[10,-10,1],estimator,1.)
            np.testing.assert_allclose(p,q,atol=1e-10)

    def test_independent_reference_and_planted_signal(self):
        result=synthetic_controls()
        self.assertLess(result['sklearn_max_abs'],1e-8)
        self.assertLess(result['positive_control_rmse_ratio'],.65)


if __name__=='__main__': unittest.main()
