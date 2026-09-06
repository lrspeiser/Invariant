import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_formula_search import controls, design, outer, paths, replay, library


class FormulaSearchTests(unittest.TestCase):
    def test_independent_and_planted(self):
        c = controls()
        self.assertLess(c['independent_ridge_max_abs'], 1e-8)
        self.assertLess(c['planted_rmse_ratio'], .35)
        self.assertLess(c['replay_max_abs'], 1e-12)

    def test_held_out_labels_cannot_select_formula(self):
        rng = np.random.default_rng(77); x = rng.normal(size=(45,8)); y = rng.normal(size=45)
        folds = np.arange(45)%3; config = dict(maximum_added_terms=2, ridge_penalties=[.1,1.])
        first = outer(x,y,folds,0,config)
        altered = y.copy(); altered[folds==0] += 10000
        second = outer(x,altered,folds,0,config)
        self.assertEqual(first['selection'], second['selection'])
        for k in ('adaptive','baseline'):
            np.testing.assert_array_equal(first[k]['prediction'], second[k]['prediction'])

    def test_training_transform_unaffected_by_test_values(self):
        rng = np.random.default_rng(71); x = rng.normal(size=(30,8)); v = rng.normal(size=(5,8))
        a,_,t = design(x,v); aa,_,tt = design(x,v*100000)
        np.testing.assert_array_equal(a,aa); self.assertEqual(t,tt)

    def test_constant_and_invalid(self):
        p = paths(np.ones((10,8)),np.ones(10)*3,np.ones((2,8)),.1,3)
        for row in p: np.testing.assert_allclose(row['prediction'],3)
        with self.assertRaises(ValueError): design(np.ones((5,7)),np.ones((2,7)))
        with self.assertRaises(ValueError): paths(np.ones((5,8)),np.full(5,np.nan),np.ones((2,8)),.1,1)

    def test_explicit_product_library(self):
        z = np.arange(16).reshape(2,8); values, terms = library(z)
        self.assertEqual(len(terms),30)
        for j,t in enumerate(terms):
            expected = z[:,t[0]] if len(t)==1 else z[:,t[0]]*z[:,t[1]]
            np.testing.assert_array_equal(values[:,j],expected)

if __name__ == '__main__': unittest.main()
