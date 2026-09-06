"""Additional vertical source discretization control, preserving earlier tests."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mond_atlas_reprojected_fields import vertical_cell_density


class VerticalCells(unittest.TestCase):
    def test_mass_and_nested_cell_consistency(self):
        coarse=np.arange(-20,21)*.15;fine=np.arange(-61,62)*.05;height=.23
        c=vertical_cell_density(coarse,.15,height);f=vertical_cell_density(fine,.05,height)
        np.testing.assert_allclose(c,f.reshape(41,3).mean(axis=1),rtol=2e-10,atol=1e-14)
        self.assertAlmostEqual(float(np.sum(c)*.15),1-np.exp(-3.075/height),places=13)
        self.assertGreaterEqual(c.min(),0)


if __name__=='__main__':unittest.main()
