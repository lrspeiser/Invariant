import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_channel_order import increasing_optical_velocity_direction


class ChannelOrderTests(unittest.TestCase):
    def test_frequency_and_radio_agree_with_explicit_optical_definitions(self):
        for delta in [10000.,-10000.]:
            h=dict(CTYPE3='FREQ',CDELT3=delta,CRVAL3=1.420e9,CRPIX3=5.,NAXIS3=12)
            f=h['CRVAL3']+(np.arange(12)+1-h['CRPIX3'])*delta
            optical=299792458*(1.420405752e9/f-1)
            self.assertEqual(increasing_optical_velocity_direction(h,'FREQ'),int(np.sign(np.diff(optical)[0])))
        for delta in [-5000.,5000.]:
            h=dict(CTYPE3='VELO-HEL',CDELT3=delta,VELREF=258)
            radio=300000.+np.arange(12)*delta;optical=radio/(1-radio/299792458)
            self.assertEqual(increasing_optical_velocity_direction(h,'VRAD'),int(np.sign(np.diff(optical)[0])))
            h['CTYPE3']='FELO-HEL'
            self.assertEqual(increasing_optical_velocity_direction(h,'VOPT-F2W'),int(np.sign(delta)))

    def test_inconsistent_headers_fail(self):
        with self.assertRaises(ValueError):increasing_optical_velocity_direction(dict(CTYPE3='VELO-HEL',CDELT3=-1,VELREF=2),'VRAD')
        with self.assertRaises(ValueError):increasing_optical_velocity_direction(dict(CTYPE3='FREQ',CDELT3=1,CRVAL3=-10,CRPIX3=1,NAXIS3=5),'FREQ')
        with self.assertRaises(ValueError):increasing_optical_velocity_direction(dict(CTYPE3='FELO-HEL',CDELT3=-1),'FREQ')


if __name__=='__main__':unittest.main()
