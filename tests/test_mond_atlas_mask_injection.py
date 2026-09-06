import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_mask_injection import consecutive_mask,spatial_filter_cube,noise_cube,response_kernel


class MaskInjectionTests(unittest.TestCase):
    def test_explicit_triples_gaps_and_boundaries(self):
        a=np.zeros((12,2,2));a[[0,1,2,5,6,9,10,11],0,0]=3
        a[2:7,1,1]=3;mask=consecutive_mask(a)
        np.testing.assert_array_equal(np.flatnonzero(mask[:,0,0]),[0,1,2,9,10,11])
        np.testing.assert_array_equal(np.flatnonzero(mask[:,1,1]),[2,3,4,5,6])
        self.assertFalse(mask[:,0,1].any())
        a[:]=0;a[5,0,0]=100;self.assertFalse(consecutive_mask(a).any())

    def test_cube_filter_matches_independent_direct_convolution(self):
        a=np.random.default_rng(126).normal(size=(4,15,17));kernel=np.array([.1,.2,.4,.2,.1])
        expected=np.apply_along_axis(lambda v:np.convolve(v,kernel,mode='same'),2,np.apply_along_axis(lambda v:np.convolve(v,kernel,mode='same'),1,a))
        np.testing.assert_allclose(spatial_filter_cube(a,kernel),expected,rtol=0,atol=1e-12)

    def test_noise_variance_and_hanning_covariance(self):
        instrument=dict(pixel_arcsec=6.,native_circular_fwhm_arcsec=12.,detection_circular_fwhm_arcsec=30.,spatial_kernel_truncate_sigma=4.,channels=32,spatial_shape=[64,64],hanning_kernel=[.25,.5,.25],spectral_branches=['independent_channels','hanning_channels','decimated_hanning_channels'])
        rng=np.random.default_rng(752);moments=[];lags=[]
        for _ in range(20):
            a=noise_cube(rng,instrument,'hanning_channels');moments.append(np.mean(a*a));lags.append(np.mean(a[1:]*a[:-1]))
        self.assertLess(abs(np.mean(moments)-1),.035)
        self.assertLess(abs(np.mean(lags)-2/3),.035)
        k=response_kernel(instrument);self.assertLess(abs(k.sum()-1),1e-12)
        moments=[];lags=[]
        for _ in range(20):
            a=noise_cube(rng,instrument,'decimated_hanning_channels');moments.append(np.mean(a*a));lags.append(np.mean(a[1:]*a[:-1]))
        self.assertLess(abs(np.mean(moments)-1),.035)
        self.assertLess(abs(np.mean(lags)-1/6),.035)


if __name__=='__main__':unittest.main()
