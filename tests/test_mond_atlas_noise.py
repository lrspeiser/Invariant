"""Noise-model recovery and held-region leakage controls."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mond_atlas_common import ROOT,read_json
from mond_atlas_image_io import gaussian_reflect
from run_mond_atlas_noise import masks,check_packet


class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=read_json(ROOT/"configs/mond_atlas_noise_v2.json")
        yy,xx=np.indices((128,128))
        rng=np.random.default_rng(55)
        cube=np.array([gaussian_reflect(x,2.) for x in rng.normal(size=(24,128,128))])
        for i in range(1,len(cube)):cube[i]=.25*cube[i-1]+np.sqrt(1-.25**2)*cube[i]
        cls.packet=dict(cube=cube,east=(xx-63.5)*12,north=(yy-63.5)*12)
        cls.result,cls.model=check_packet(cls.packet,cls.config)

    def test_balanced_masks_cover_quadrants_and_keep_spatial_guard(self):
        train,test=masks(self.packet["east"],self.packet["north"],self.config)
        self.assertFalse(np.any(train&test))
        self.assertGreaterEqual(self.result["minimum_calibration_validation_separation_arcsec"],108-1e-9)
        self.assertTrue(all(n>=4 for n in self.result["validation_quadrant_counts"]))

    def test_recovers_known_smoothed_noise_scale(self):
        scales=np.array(self.result["spatial_correlation_scales_pixels"])
        # Autocorrelation width is sqrt(2) times the image smoothing sigma.
        self.assertLess(np.max(abs(scales/(2*np.sqrt(2))-1)),.25)
        self.assertLess(abs(self.result["joint_validation_mean_square"]-1),.3)
        self.assertLess(abs(self.result["joint_validation_channel_lag1"]),.15)

    def test_held_region_mutation_cannot_change_calibration_and_detects_nonstationarity(self):
        altered={k:v.copy() for k,v in self.packet.items()}
        held=self.model["background_validation_mask"]
        altered["cube"][:,held]*=4
        changed,model=check_packet(altered,self.config)
        for key in ["channel_covariance","spatial_precision","mean_offset","background_calibration_mask","background_validation_mask"]:
            np.testing.assert_array_equal(self.model[key],model[key])
        self.assertGreater(changed["joint_validation_mean_square"],2)
        self.assertFalse(changed["diagnostic_pass"])


if __name__=="__main__":unittest.main(verbosity=2)
