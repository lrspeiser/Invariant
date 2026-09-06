"""Controls for reversed guarded partitions and reuse of the frozen estimator."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_common import ROOT, read_json
from mond_atlas_image_io import gaussian_reflect
import run_mond_atlas_noise as noise
from run_mond_atlas_noise_robustness import reversed_masks, evaluate


class NoisePartitionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=read_json(ROOT/'configs/mond_atlas_noise_v2.json')
        yy,xx=np.indices((128,128));rng=np.random.default_rng(916)
        cls.packet=dict(cube=np.array([gaussian_reflect(a,2.) for a in rng.normal(size=(24,128,128))]),
            east=(xx-63.5)*12,north=(yy-63.5)*12)

    def test_reverse_blocks_preserve_guard_and_swap_calibration(self):
        east,north=self.packet['east'],self.packet['north']
        train,test=noise.masks(east,north,self.config)
        reverse_train,reverse_test=reversed_masks(east,north,self.config)
        self.assertFalse(np.any(train&reverse_train))
        self.assertTrue(np.all(reverse_train[test]))
        self.assertTrue(np.all(train[reverse_test]))
        for a,b in ((train,test),(reverse_train,reverse_test)):
            xy=np.column_stack((east[a],north[a]));uv=np.column_stack((east[b],north[b]))
            self.assertGreaterEqual(np.sqrt(np.sum((xy[:,None]-uv[None,:])**2,axis=2)).min(),108-1e-9)
            quadrants=(east[b]>=0).astype(int)+2*(north[b]>=0).astype(int)
            self.assertTrue(np.all(np.bincount(quadrants,minlength=4)>=4))

    def test_supplied_forward_partition_replays_frozen_estimator(self):
        before=noise.masks
        reference,model=noise.check_packet(self.packet,self.config)
        train,test=noise.masks(self.packet['east'],self.packet['north'],self.config)
        replay,arrays=evaluate(self.packet,self.config,train,test)
        self.assertEqual(reference,replay)
        for key in model:np.testing.assert_array_equal(model[key],arrays[key])
        self.assertIs(noise.masks,before)

    def test_reverse_ignores_galaxy_values_and_restores_estimator_on_error(self):
        train,test=reversed_masks(self.packet['east'],self.packet['north'],self.config)
        original,model=evaluate(self.packet,self.config,train,test)
        mutated={k:v.copy() for k,v in self.packet.items()}
        galaxy=np.hypot(mutated['east'],mutated['north'])<500
        mutated['cube'][:,galaxy]=1e12
        changed,arrays=evaluate(mutated,self.config,train,test)
        self.assertEqual(original,changed)
        for key in model:np.testing.assert_array_equal(model[key],arrays[key])
        before=noise.masks
        with self.assertRaises(ValueError):evaluate(self.packet,self.config,np.zeros_like(train),test)
        self.assertIs(noise.masks,before)


if __name__=='__main__':unittest.main()
