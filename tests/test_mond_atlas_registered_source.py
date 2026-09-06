import copy
import sys
import unittest
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_registered_source import inclination, source_coordinates, transfer_matrix, rebin_tracer
from build_mond_atlas_registered_source import reference_wcs


def header(projection='TAN'):
    return dict(NAXIS=2, NAXIS1=24, NAXIS2=24, CTYPE1='RA---'+projection,
                CTYPE2='DEC--'+projection, CRVAL1=146.8, CRVAL2=67.9,
                CRPIX1=12.5, CRPIX2=12.5, CD1_1=-.002, CD1_2=.00015,
                CD2_1=.0002, CD2_2=.002)


G = dict(ra_deg=146.8, dec_deg=67.9, distance_mpc=3.611, pa_deg=144., inclination_deg=54.)
GRID = dict(half_width_kpc=3., spacing_kpc=.125, annulus_width_kpc=.25,
            taper_start_kpc=2., cutoff_kpc=2.5, minimum_cell_coverage=.5, minimum_annulus_coverage=.2)


def independent_mapping(h, xy, g, reference=None, shift=None):
    sky = WCS(h).wcs_pix2world(xy, 0)
    if reference is not None:
        w = WCS(reference)
        sky = w.wcs_pix2world(w.wcs_world2pix(sky,0)+shift,0)
    ra, dec = np.deg2rad(sky).T
    delta = ra-np.deg2rad(g['ra_deg']); dc=np.deg2rad(g['dec_deg'])
    cosc = np.sin(dc)*np.sin(dec)+np.cos(dc)*np.cos(dec)*np.cos(delta)
    east = np.cos(dec)*np.sin(delta)/cosc*g['distance_mpc']*1000
    north = (np.cos(dc)*np.sin(dec)-np.sin(dc)*np.cos(dec)*np.cos(delta))/cosc*g['distance_mpc']*1000
    pa = np.deg2rad(g['pa_deg']); cosi=np.cos(np.deg2rad(g['inclination_deg']))
    return (east*np.sin(pa)+north*np.cos(pa), (east*np.cos(pa)-north*np.sin(pa))/cosi,
            np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec))))


class RegisteredSourceTests(unittest.TestCase):
    def test_float_header_dimensions_and_spectral_axes_are_not_spatial(self):
        h=header();h.update(NAXIS=4.,NAXIS1=24.,NAXIS2=24.,NAXIS3=1.,NAXIS4=1.,CTYPE3='VELO-HEL',CTYPE4='STOKES')
        xy=np.array([[0.,0.],[10.5,11.4]])
        np.testing.assert_allclose(reference_wcs(h).wcs_pix2world(xy,0),WCS(header()).wcs_pix2world(xy,0),rtol=0,atol=1e-12)

    def test_astropy_tan_sin_and_composed_translation(self):
        xy=np.array([[.1,.5],[10.,15.],[20.1,23.],[7.3,8.2]])
        for projection in ('TAN','SIN'):
            h=header(projection); p1=header();p1['CRVAL1']+=.005;p1['CRVAL2']-=.003
            shift=np.array([-.453,-1.661])
            for registered in (False,True):
                transform=transfer_matrix(p1,shift) if registered else None
                x,y,a,v=source_coordinates(h,xy,G,transform)
                rx,ry,rv=independent_mapping(h,xy,G,p1 if registered else None,shift)
                self.assertLess(np.max(np.linalg.norm(v-rv,axis=1))*206264.806,1e-5)
                np.testing.assert_allclose(x,rx,atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(y,ry,atol=1e-9,rtol=1e-9)

    def test_independent_finite_difference_area(self):
        h=header();p1=header();p1['CRVAL1']+=.005;shift=np.array([-.453,-1.661])
        xy=np.array([[2.,2.],[11.5,11.5],[22.,19.]]);eps=.1
        for projection in ('TAN','SIN'):
            h['CTYPE1']='RA---'+projection;h['CTYPE2']='DEC--'+projection
            area=source_coordinates(h,xy,G,transfer_matrix(p1,shift))[2]
            d=[]
            for step in ([eps,0],[0,eps]):
                a=independent_mapping(h,xy+step,G,p1,shift);b=independent_mapping(h,xy-step,G,p1,shift)
                d.append(np.column_stack(((a[0]-b[0])/(2*eps),(a[1]-b[1])/(2*eps))))
            reference=np.abs(d[0][:,0]*d[1][:,1]-d[0][:,1]*d[1][:,0])*np.cos(np.deg2rad(G['inclination_deg']))
            self.assertLess(np.max(abs(area/reference-1)),1e-6)

    def test_uniform_field_and_conservation(self):
        image=np.full((24,24),2.);good=np.ones_like(image,bool)
        # The inclined image's corners extend beyond the original 3 kpc box.
        # Use an enclosing field for this no-loss control; retain the separate
        # cropped-field test and do not change the conservation tolerance.
        enclosing=dict(GRID,half_width_kpc=6.,taper_start_kpc=4.5,cutoff_kpc=5.)
        maps,r,_=rebin_tracer(image,header(),good,G,enclosing,conversion=3.,subdivisions=4)
        np.testing.assert_allclose(maps['mean'][maps['coverage']>0],6*np.cos(np.deg2rad(54)),rtol=1e-12)
        self.assertLess(abs(r['outside_field_signed_integral']/r['input_signed_integral']),1e-12)
        self.assertAlmostEqual(float(np.sum(maps['observed'])*.125**2*1e6),r['input_signed_integral'],places=6)
        self.assertTrue(good.all())

    def test_distance_area_scaling_and_fixed_angular_aperture(self):
        yy,xx=np.mgrid[:24,:24];image=np.exp(-((xx-11.2)**2+(yy-12.4)**2)/30)
        a,ra,_=rebin_tracer(image,header(),np.ones_like(image,bool),G,GRID,subdivisions=2)
        doubled=dict(G,distance_mpc=2*G['distance_mpc']);grid=copy.deepcopy(GRID)
        for k in ('half_width_kpc','spacing_kpc','annulus_width_kpc','taper_start_kpc','cutoff_kpc'):grid[k]*=2
        b,rb,_=rebin_tracer(image,header(),np.ones_like(image,bool),doubled,grid,subdivisions=2)
        np.testing.assert_allclose(a['observed'],b['observed'],rtol=1e-10,atol=1e-10)
        self.assertLess(abs(rb['conditional_zero_integral']/ra['conditional_zero_integral']-4),1e-10)

    def test_rotation_and_face_on_limit(self):
        xy=np.array([[3.,4.],[12.,10.],[20.,22.]])
        g=dict(G,inclination_deg=0.,pa_deg=0.);a=source_coordinates(header(),xy,g)
        b=source_coordinates(header(),xy,dict(g,pa_deg=90.))
        np.testing.assert_allclose(b[0],a[1],atol=1e-12)
        np.testing.assert_allclose(b[1],-a[0],atol=1e-12)
        self.assertEqual(inclination(dict(ellipticity=0.,intrinsic_axis_ratio=0.)),0.)

    def test_pixel_refinement_against_independent_oversampling(self):
        h=header();h['CD1_1']/=4;h['CD1_2']/=4;h['CD2_1']/=4;h['CD2_2']/=4
        yy,xx=np.mgrid[:24,:24];image=np.exp(-((xx-10.7)**2+(yy-13.2)**2)/28)
        maps,_,_=rebin_tracer(image,h,np.ones_like(image,bool),G,GRID,subdivisions=4)
        # Separate Astropy coordinates and direct histogram; high-order midpoint
        # reference uses the exact constant TAN tangent area at common centers.
        flux=np.zeros_like(maps['observed']);n=32;cd=np.array([[h['CD1_1'],h['CD1_2']],[h['CD2_1'],h['CD2_2']]])
        area=abs(np.linalg.det(cd))*(np.pi/180)**2*(G['distance_mpc']*1000)**2/n**2
        edges=np.r_[maps['axis']-.0625,maps['axis'][-1]+.0625]
        for oy in (np.arange(n)+.5)/n-.5:
            xy=np.column_stack((np.repeat(xx.ravel(),n)+np.tile((np.arange(n)+.5)/n-.5,xx.size),np.repeat(yy.ravel()+oy,n)))
            x,y,_=independent_mapping(h,xy,G)
            flux+=np.histogram2d(x,y,bins=(edges,edges),weights=np.repeat(image.ravel(),n)*area)[0]/.125**2
        self.assertLess(np.sum(abs(maps['observed']-flux))/np.sum(abs(flux)),.03)

    def test_missing_support_signed_flux_and_field_boundary(self):
        image=np.ones((24,24));image[:5,:]=-1.;good=np.ones_like(image,bool);good[:,:3]=False
        maps,r,_=rebin_tracer(image,header(),good,G,GRID,subdivisions=2)
        self.assertGreater(r['negative_projection_added_integral'],0.)
        self.assertFalse(r['missing_flux_is_measured_zero'])
        small=dict(GRID,half_width_kpc=.5,taper_start_kpc=.25,cutoff_kpc=.4)
        _,r,_=rebin_tracer(np.ones_like(image),header(),good,G,small,subdivisions=2)
        self.assertGreater(r['outside_field_signed_integral'],0.)

    def test_invalid_units_geometry_and_support_fail_closed(self):
        with self.assertRaises(ValueError):inclination(dict(ellipticity=.9,intrinsic_axis_ratio=.2))
        with self.assertRaises(ValueError):source_coordinates(header(),np.array([[0,0]]),dict(G,distance_mpc=0))
        h=header();h['CUNIT1']='rad'
        with self.assertRaises(ValueError):source_coordinates(h,np.array([[0,0]]),G)
        with self.assertRaises(ValueError):rebin_tracer(np.ones((2,2)),header(),np.zeros((2,2),bool),G,GRID)


if __name__=='__main__':unittest.main()
