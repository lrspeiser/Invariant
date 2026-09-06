import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_smoothing_null import native_to_coarse_operator,normalized_covariance,draw_separable,center_outer_statistic
from mond_atlas_preprocessing import gaussian_plane_float32,block_mean


class SmoothingNullTests(unittest.TestCase):
    def test_operator_matches_native_processing_and_rounding_bound(self):
        rng=np.random.default_rng(603);image=rng.normal(size=(64,80)).astype(np.float32)
        for native_sigma in [0.,1.1]:
            oy=native_to_coarse_operator(64,4,2.3,native_sigma);ox=native_to_coarse_operator(80,4,2.3,native_sigma)
            direct=block_mean(gaussian_plane_float32(gaussian_plane_float32(image,native_sigma),2.3),4)
            matrix=oy@image@ox.T
            self.assertLess(np.linalg.norm(direct-matrix)/np.linalg.norm(direct),1e-6)

    def test_separable_covariance_against_dense_native_operator(self):
        y=native_to_coarse_operator(12,3,1.1,.4);x=native_to_coarse_operator(15,3,1.4,.7)
        transform=np.kron(y,x);expected=transform@transform.T
        actual=np.kron(y@y.T,x@x.T)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-12)

    def test_sampler_covariance_against_independent_dense_solution(self):
        y=native_to_coarse_operator(8,2,.7);x=native_to_coarse_operator(10,2,.8)
        cy,ly=normalized_covariance(y);cx,lx=normalized_covariance(x)
        cc=np.array([[1.,-.3],[-.3,1.]]);lc=np.linalg.cholesky(cc)
        samples=draw_separable(np.random.default_rng(330),30000,lc,ly,lx).reshape(30000,-1)
        expected=np.kron(cc,np.kron(cy,cx));actual=samples.T@samples/len(samples)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),.04)

    def test_ratio_invariant_to_channel_offset_and_scale(self):
        data=np.random.default_rng(950).normal(size=(4,3,10,12));yy,xx=np.indices((10,12));inner=xx<4;outer=xx>7
        baseline=center_outer_statistic(data,inner,outer)
        modified=data*np.array([.3,2.,8.])[None,:,None,None]+np.array([4.,12.,-9.])[None,:,None,None]
        np.testing.assert_allclose(center_outer_statistic(modified,inner,outer),baseline,atol=1e-12)


if __name__=='__main__':unittest.main()
