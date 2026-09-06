import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_preprocessing import finite_convolution_axis,gaussian_plane_float32,block_mean


class PreprocessingTests(unittest.TestCase):
    def test_finite_fft_against_direct_nonperiodic_convolution(self):
        image=np.random.default_rng(642).normal(size=(13,17));kernel=np.array([.1,.2,.3,.4,.5])
        for axis in (0,1):
            expected=np.apply_along_axis(lambda x:np.convolve(x,kernel,mode='same'),axis,image)
            np.testing.assert_allclose(finite_convolution_axis(image,kernel,axis),expected,rtol=0,atol=1e-12)

    def test_gaussian_impulse_and_boundary_loss(self):
        a=np.zeros((81,81));a[40,40]=1
        smoothed=gaussian_plane_float32(a,3.7);yy,xx=np.indices(a.shape)
        self.assertLess(abs(float(smoothed.sum())-1),2e-7)
        self.assertLess(abs(float((smoothed*xx).sum())-40),2e-6)
        self.assertLess(abs(float((smoothed*yy).sum())-40),2e-6)
        a[:]=0;a[0,0]=1;edge=gaussian_plane_float32(a,3.7)
        self.assertLess(float(edge.sum()),.4);self.assertLess(float(np.abs(edge[-10:]).max()),1e-10)

    def test_block_flux_accounting_and_zero_smoothing(self):
        a=np.random.default_rng(445).normal(size=(48,64))
        self.assertLess(abs(block_mean(a,8).sum()*64-a.sum()),1e-12)
        np.testing.assert_array_equal(gaussian_plane_float32(a,0),a.astype(np.float32))
        with self.assertRaises(ValueError):block_mean(a,5)


if __name__=='__main__':unittest.main()
