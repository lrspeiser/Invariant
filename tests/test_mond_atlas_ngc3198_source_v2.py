import copy,math,sys,unittest
from decimal import Decimal
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_ngc3198_source_v2 import integer_geometry,squared_annuli,rebin_tracer
from test_mond_atlas_registered_source import header,G,GRID

def grid():return dict(GRID,dimensionless_layout=dict(half_width_cells=24,annulus_width_cells=2,taper_start_cells=16,cutoff_cells=20))
SCALES=[12.355/13.987,15.619/13.987,.1,.7,math.pi,1000.]

class SourceV2Tests(unittest.TestCase):
    def test_integer_reference_and_boundary(self):
        for width in [1,2,3,7]:
            jj,kk=np.mgrid[-50:51,-50:51];sq=jj*jj+kk*kk
            actual=squared_annuli(sq,width)
            expected=np.array([math.isqrt(int(v))//width for v in sq.flat]).reshape(sq.shape)
            np.testing.assert_array_equal(actual,expected)
            for n in [1,2,7,31]:
                boundary=float((n*width)**2)
                values=np.array([np.nextafter(boundary,-np.inf),boundary,np.nextafter(boundary,np.inf)])
                reference=[sum(Decimal.from_float(float(v))>=Decimal((i*width)**2) for i in range(n+2))-1 for v in values]
                np.testing.assert_array_equal(squared_annuli(values,width),reference)

    def test_distance_index_invariance_and_v1_counterexample(self):
        g=grid();base=integer_geometry(g);changes=0
        old=lambda a:np.floor(np.hypot(*np.meshgrid(np.arange(49)*a['spacing_kpc']-a['half_width_kpc'],np.arange(49)*a['spacing_kpc']-a['half_width_kpc']))/a['annulus_width_kpc']).astype(int)
        original=old(g)
        for scale in SCALES:
            modified=copy.deepcopy(g)
            for key in ['half_width_kpc','spacing_kpc','annulus_width_kpc','taper_start_kpc','cutoff_kpc']:modified[key]*=scale
            for a,b in zip(base,integer_geometry(modified)):np.testing.assert_array_equal(a,b)
            changes+=int(np.sum(original!=old(modified)))
        self.assertGreater(changes,0)

    def test_signed_holes_mass_scaling(self):
        yy,xx=np.mgrid[:24,:24];image=np.exp(-((xx-10.2)**2+(yy-12.7)**2)/30)-.03
        good=(xx%7!=0)&(yy%5!=0)
        a,ra,_=rebin_tracer(image,header(),good,G,grid(),subdivisions=2)
        self.assertGreater(np.sum(a['observed']<0),0)
        for scale in SCALES:
            gg=dict(G,distance_mpc=G['distance_mpc']*scale);gr=grid()
            for key in ['half_width_kpc','spacing_kpc','annulus_width_kpc','taper_start_kpc','cutoff_kpc']:gr[key]*=scale
            b,rb,_=rebin_tracer(image,header(),good,gg,gr,subdivisions=2)
            for key in ['signed_measured_integral','conditional_zero_integral','conditional_annular_integral']:
                self.assertLess(abs(rb[key]/ra[key]/scale**2-1),1e-10)
            for key in ['observed','coverage','zero','annular']:np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-10)
            self.assertLess(abs(b['observed'].sum()*gr['spacing_kpc']**2*1e6/rb['untapered_in_field_signed_integral']-1),1e-12)

    def test_reject_bad_metadata(self):
        for bad in [2.5,-1,True]:
            g=grid();g['dimensionless_layout']['annulus_width_cells']=bad
            with self.assertRaises(ValueError):integer_geometry(g)
        g=grid();g['spacing_kpc']*=1.001
        with self.assertRaises(ValueError):integer_geometry(g)

if __name__=='__main__':unittest.main()
