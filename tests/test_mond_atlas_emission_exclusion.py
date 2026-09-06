import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_emission_exclusion import exclude_support


class FixedExclusionTests(unittest.TestCase):
    def test_only_declared_support_removed(self):
        yy, xx = np.indices((20, 20)); train = xx < 7; test = xx > 12
        expanded = yy < 4; east, north = xx-10, yy-10
        config = dict(minimum_calibration_pixels=1,minimum_validation_pixels=1,diagnostic_gates=dict(minimum_quadrant_pixels=1))
        a,b,result = exclude_support(train,test,expanded,east,north,config)
        np.testing.assert_array_equal(a,train & (yy>=4)); np.testing.assert_array_equal(b,test & (yy>=4))
        self.assertFalse(np.any(a&b)); self.assertFalse(np.any((a|b)&expanded))
        self.assertEqual(result['removed_calibration_pixels'],28)
        self.assertFalse(result['sufficient_support'])  # Validation never covers the western quadrants.

    def test_empty_support_preserves_masks_and_full_support_blocks(self):
        yy,xx = np.indices((16,16)); train = (xx+yy)%2==0; test=~train
        config = dict(minimum_calibration_pixels=100,minimum_validation_pixels=25,diagnostic_gates=dict(minimum_quadrant_pixels=4))
        a,b,result=exclude_support(train,test,np.zeros_like(train),xx-7.5,yy-7.5,config)
        np.testing.assert_array_equal(a,train); np.testing.assert_array_equal(b,test); self.assertTrue(result['sufficient_support'])
        a,b,result=exclude_support(train,test,np.ones_like(train),xx-7.5,yy-7.5,config)
        self.assertFalse(a.any() or b.any()); self.assertFalse(result['sufficient_support'])


if __name__ == '__main__': unittest.main()
