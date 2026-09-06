"""Run with unittest: no third-party test runner or network is required."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mond_atlas_common import canonical_name, fits_primary_header, verify_text_digest, sparc_inputs
from build_mond_atlas_catalog import proximity_candidates
from mond_atlas_cube import project_emission, correlated_score, spatial_beam, gaussian_cdf
from mond_atlas_fields import dst1, poisson, laplacian, solve_fields, validate
from run_mond_atlas_radial import speeds, nested_predictions, fixed_folds


class CatalogAndRadialTests(unittest.TestCase):
    def test_explicit_aliases_and_conservative_names(self):
        self.assertEqual(canonical_name("HO_II"),canonical_name("UGC04305"))
        self.assertEqual(canonical_name("NGC 03198"),"NGC3198")
        self.assertNotEqual(canonical_name("UGCA04305"),canonical_name("UGC04305"))
        self.assertNotEqual(canonical_name("NGC3198A"),canonical_name("NGC3198"))

    def test_hash_detects_data_change_but_explains_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"table.csv"
            expected=hashlib.sha256(b"a,b\n1,2\n").hexdigest()
            p.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(verify_text_digest(p,expected),"CRLF_to_LF_equivalent_only")
            p.write_bytes(b"a,b\r\n1,3\r\n")
            with self.assertRaises(ValueError):verify_text_digest(p,expected)

    def test_fits_inventory_plain_gzip_and_truncation(self):
        cards=["SIMPLE  =                    T","NAXIS   =                    3",
               "CTYPE3  = 'VELO-HEL'", "BUNIT   = 'JY/BEAM'", "END"]
        raw="".join(c.ljust(80) for c in cards).ljust(2880).encode("ascii")
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"image.fits"
            p.write_bytes(raw)
            self.assertEqual(fits_primary_header(p)["BUNIT"],"JY/BEAM")
            p.write_bytes(gzip.compress(raw))
            self.assertEqual(fits_primary_header(p)["CTYPE3"],"VELO-HEL")
            p.write_bytes(b"bad")
            with self.assertRaises(ValueError):fits_primary_header(p)

    def test_sky_wrap_flags_without_merging(self):
        rows=[dict(atlas_id="a",ra_deg=359.999,dec_deg=0),dict(atlas_id="b",ra_deg=.001,dec_deg=0),
              dict(atlas_id="c",ra_deg=100,dec_deg=0)]
        result=proximity_candidates(rows,10)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0]["separation_arcsec"],7.2,places=6)
        self.assertEqual(len(rows),3)

    def test_all_published_sparc_strings_match_archive(self):
        curves,meta,photo,_=sparc_inputs()
        self.assertEqual(len(curves),175)
        self.assertEqual(sum(len(g["rows"]) for g in curves),3391)
        self.assertEqual(set(meta),set(photo))

    def test_signed_gas_hole_is_not_positive_mass_force(self):
        n,m,g,valid=speeds([1,1],[-10,-10],[0,20],[0,0])
        self.assertFalse(valid[0])
        self.assertTrue(valid[1])
        self.assertAlmostEqual(n[1]**2,100)
        self.assertTrue(np.isnan(m[0]))

    def test_spherical_mond_equation_and_distance_scaling(self):
        r=np.array([.1,1,10,100.])
        n,m,g,_=speeds(r,np.ones(4)*20,np.zeros(4),np.zeros(4))
        acceleration=m*m/r*(1e6/3.085677581491367e19)
        mu=acceleration/(acceleration+1.2e-10)
        np.testing.assert_allclose(mu*acceleration,g,rtol=1e-13)
        n2,m2,g2,_=speeds(r,np.ones(4)*20,np.zeros(4),np.zeros(4),distance_scale=1.44)
        np.testing.assert_allclose(m2/m,1.2)
        np.testing.assert_allclose(g2,g)

    def test_entire_held_galaxy_fold_targets_cannot_change_its_prediction(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(35,3)); y=rng.normal(size=35)
        folds=fixed_folds([str(i) for i in range(35)])
        before=nested_predictions(x,y,folds)
        y[folds==0]+=1000
        after=nested_predictions(x,y,folds)
        np.testing.assert_array_equal(before[folds==0],after[folds==0])


class FieldTests(unittest.TestCase):
    def test_dst_self_inverse(self):
        x=np.random.default_rng(7).normal(size=(7,8,9))
        for axis in range(3):np.testing.assert_allclose(dst1(dst1(x,axis),axis),x,atol=2e-15)

    def test_poisson_against_exact_quadratic_with_nonzero_boundary(self):
        a=np.linspace(-2,2,15);h=a[1]-a[0]
        x,y,z=np.meshgrid(a,a,a,indexing="ij")
        phi=x*x+2*y*y+3*z*z+.2*x*y+.3*x
        result=poisson(np.full_like(phi,12.),phi,h)
        np.testing.assert_allclose(result,phi,atol=2e-13)

    def test_solver_refuses_negative_density(self):
        x=np.zeros((9,9,9));rho=x.copy();rho[4,4,4]=-1
        with self.assertRaises(ValueError):solve_fields(rho,1,x,x,gravity_constant=1,a0=1)

    def test_analytic_resolution_and_symmetry_gates(self):
        self.assertTrue(validate()["all_pass"])


class CubeTests(unittest.TestCase):
    def test_gaussian_cdf_against_math_erf(self):
        x=np.linspace(-8,8,201)
        exact=np.array([.5*(1+math.erf(v/np.sqrt(2))) for v in x])
        self.assertLess(np.max(np.abs(exact-gaussian_cdf(x))),8e-8)

    def test_multiple_depth_components_survive_projection(self):
        # Two separated physical layers on the SAME sightline, with opposite velocities.
        emission=np.ones((2,1,1));v=np.array([[[-20]],[[20]]])
        edges=np.arange(-60.,61.,2.)
        cube=project_emission(emission,v,3,edges,1)
        self.assertAlmostEqual(cube.sum(),2,places=10)
        self.assertAlmostEqual(cube[:30].sum(),1,places=10)
        self.assertLess(float(cube[29:31].sum()),1e-7)
        narrow=project_emission(emission,v,3,[-1,1],1)
        self.assertLess(narrow.sum(),1e-7)

    def test_beam_is_linear_and_does_not_wrap(self):
        a=np.zeros((1,15,15));a[0,0,0]=1
        blurred=spatial_beam(a,np.ones((3,3)))
        self.assertLess(np.max(np.abs(blurred[0,-3:,-3:])),1e-14)
        np.testing.assert_allclose(spatial_beam(3*a,np.ones((3,3))),3*blurred,atol=1e-15)

    def test_correlated_score_matches_dense_kronecker_likelihood(self):
        rng=np.random.default_rng(3)
        a=rng.normal(size=(3,3));b=rng.normal(size=(5,5))
        cc=a@a.T+np.eye(3);cs=b@b.T+np.eye(5)
        residual=rng.normal(size=(3,5));result=correlated_score(residual,cc,cs)
        dense=np.kron(cc,cs);r=residual.ravel()
        self.assertAlmostEqual(result["quadratic_form"],float(r@np.linalg.solve(dense,r)),places=12)
        self.assertAlmostEqual(result["log_determinant"],float(np.linalg.slogdet(dense)[1]),places=12)
        self.assertNotAlmostEqual(result["quadratic_form"],float(np.sum(residual**2)))

    def test_irregular_fixed_spatial_mask_subsets_covariance(self):
        cc=np.array([[1,.4],[.4,1]])
        cs=np.full((5,5),.2)+.8*np.eye(5)
        keep=np.array([0,2,4]);r=np.arange(10.).reshape(2,5)
        result=correlated_score(r[:,keep],cc,cs[np.ix_(keep,keep)])
        full=np.kron(cc,cs);indices=np.r_[keep,5+keep];rr=r.ravel()[indices]
        expected=rr@np.linalg.solve(full[np.ix_(indices,indices)],rr)
        self.assertAlmostEqual(result["quadratic_form"],float(expected),places=11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
