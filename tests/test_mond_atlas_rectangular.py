"""Independent controls for anisotropic fields and source-only mass projection."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mond_atlas_rectangular_fields as rf
from mond_atlas_fields import poisson as isotropic_poisson
from mond_atlas_image_io import plain_tan_pixel_to_world
from build_mond_atlas_ngc2903_source import pixel_geometry,sky_vectors,rebin_source


class RectangularFields(unittest.TestCase):
    def test_manufactured_polynomial_unequal_spacing(self):
        axes=[np.linspace(-2,2,21),np.linspace(-3,3,25),np.linspace(-1,1,33)]
        h=[a[1]-a[0] for a in axes];x,y,z=np.meshgrid(*axes,indexing='ij')
        phi=x*x+2*y*y+3*z*z+0.7*x*y-2*x+z
        solved=rf.poisson(np.full_like(phi,12),phi,h)
        np.testing.assert_allclose(solved,phi,atol=1e-12)

    def test_equal_spacing_replays_independent_milestone(self):
        rng=np.random.default_rng(15);rho=rng.uniform(size=(19,17,21));boundary=rng.normal(size=rho.shape)
        np.testing.assert_allclose(rf.poisson(rho,boundary,[.3]*3),isotropic_poisson(rho,boundary,.3),atol=2e-14)

    def test_uniform_external_field_anisotropic(self):
        axes=[np.linspace(-2,2,21),np.linspace(-3,3,25),np.linspace(-1,1,33)]
        h=[a[1]-a[0] for a in axes];x,y,z=np.meshgrid(*axes,indexing='ij')
        bn=2*x+.3*y-.1*z;bm=bn*(.5+np.sqrt(.25+1/np.sqrt(4+.09+.01)))
        pn,pm,res=rf.solve(np.zeros_like(bn),h,bn,bm,1.,1.)
        np.testing.assert_allclose(pm,bm,atol=1e-12)

    def test_exact_monopole_derivative(self):
        r=np.geomspace(.001,1000,200);GM=3.;a0=.7;dr=r*1e-5
        numerical=(rf.simple_monopole_potential(r+dr,GM,a0)-rf.simple_monopole_potential(r-dr,GM,a0))/(2*dr)
        gn=GM/r**2;expected=.5*gn+np.sqrt(.25*gn**2+a0*gn)
        np.testing.assert_allclose(numerical,expected,rtol=2e-8)

    def test_anisotropic_spherical_plummer(self):
        axes=[np.linspace(-6,6,65),np.linspace(-6,6,49),np.linspace(-6,6,97)]
        h=[a[1]-a[0] for a in axes];xyz=np.meshgrid(*axes,indexing='ij');r=np.sqrt(sum(x*x for x in xyz))
        rho=3/(4*np.pi)*(1+r*r)**(-2.5);bn=-1/np.sqrt(1+r*r)
        t=np.linspace(0,11,100001);gn=t/(1+t*t)**1.5;gm=.5*gn+np.sqrt(.25*gn*gn+gn)
        integ=np.r_[0,np.cumsum(.5*(gm[1:]+gm[:-1])*np.diff(t))];bm=np.interp(r,t,integ)
        pn,pm,res=rf.solve(rho,h,bn,bm,1.,1.)
        use=(r>1.5)&(r<4)
        exactn=r/(1+r*r)**1.5;exactm=.5*exactn+np.sqrt(.25*exactn**2+exactn)
        for p,expected in ((pn,exactn),(pm,exactm)):
            actual=np.sqrt(sum(g*g for g in np.gradient(p,*h,edge_order=2)))
            self.assertLess(np.sqrt(np.mean((actual[use]/expected[use]-1)**2)),.02)
        self.assertLess(max(res.values()),1e-10)


class SourceProjection(unittest.TestCase):
    def header(self,projection='TAN'):
        return dict(CTYPE1='RA---'+projection,CTYPE2='DEC--'+projection,CRVAL1=143.,CRVAL2=21.,CRPIX1=20.,CRPIX2=24.,
                    CDELT1=-.005,CDELT2=.005)
    def geometry(self):return dict(ra_deg=143.01,dec_deg=21.01,distance_mpc=10.,pa_deg=31.,inclination_deg=50.)

    def test_tan_agrees_with_previously_validated_transform(self):
        h=self.header();g=self.geometry();x,y,area,cosi=pixel_geometry(h,(41,43),g)
        h.update(CD1_1=h['CDELT1'],CD1_2=0,CD2_1=0,CD2_2=h['CDELT2'])
        yy,xx=np.indices((41,43));sky=plain_tan_pixel_to_world(np.stack((xx,yy),axis=-1),h)
        ra=np.deg2rad(sky[...,0]);dec=np.deg2rad(sky[...,1]);ra0,dec0=np.deg2rad([g['ra_deg'],g['dec_deg']])
        denom=np.sin(dec)*np.sin(dec0)+np.cos(dec)*np.cos(dec0)*np.cos(ra-ra0)
        east=np.cos(dec)*np.sin(ra-ra0)/denom*10000
        north=(np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0))/denom*10000
        pa=np.deg2rad(g['pa_deg'])
        np.testing.assert_allclose(x,east*np.sin(pa)+north*np.cos(pa),atol=2e-11)
        np.testing.assert_allclose(y,(east*np.cos(pa)-north*np.sin(pa))/cosi,atol=2e-11)

    def test_sin_reduces_to_orthographic_at_reference(self):
        h=self.header('SIN');g=dict(ra_deg=143.,dec_deg=21.,distance_mpc=10.,pa_deg=90.,inclination_deg=0.)
        x,y,area,c=pixel_geometry(h,(41,43),g)
        yy,xx=np.indices(x.shape);l=np.deg2rad((xx+1-h['CRPIX1'])*h['CDELT1']);m=np.deg2rad((yy+1-h['CRPIX2'])*h['CDELT2'])
        root=np.sqrt(1-l*l-m*m)
        np.testing.assert_allclose(x,10000*l/root,atol=1e-11)
        np.testing.assert_allclose(y,-10000*m/root,atol=1e-11)
        np.testing.assert_allclose(area,abs(h['CDELT1']*h['CDELT2'])*(np.pi/180)**2*10000**2/root,rtol=1e-14)

    def test_missingness_and_signed_flux_are_not_zeros(self):
        h=self.header();h['CDELT1']=-.0002;h['CDELT2']=.0002;g=dict(ra_deg=143.,dec_deg=21.,distance_mpc=10.,pa_deg=0.,inclination_deg=0.)
        shape=(81,81);h['CRPIX1']=h['CRPIX2']=41.;a=np.ones(shape)*3;good=np.ones(shape,bool);good[:,::2]=False
        grid=dict(half_width_kpc=2.,spacing_kpc=.2,annulus_width_kpc=.4,minimum_cell_coverage=.3,minimum_annulus_coverage=.2,taper_start_kpc=1.,cutoff_kpc=1.4)
        packet,report,rows=rebin_source(a,h,good,g,grid)
        self.assertGreater(report['conditional_annular_integral'],1.5*report['conditional_zero_integral'])
        a[:]=-3
        packet,report,rows=rebin_source(a,h,good,g,grid)
        self.assertLess(report['signed_measured_integral'],0)
        self.assertGreater(report['negative_projection_added_integral'],0)
        self.assertEqual(report['conditional_annular_integral'],0)


if __name__=='__main__':unittest.main()
