"""Independent coordinate, data-format, footprint and identity controls."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mond_atlas_image_io import (read_primary_image,plain_tan_pixel_to_world,plain_tan_world_to_pixel,
                                 finite_footprint,gaussian_reflect,nearest)
from resolve_mond_atlas_identities import same_nsa_object


def header():
    return dict(CTYPE1="RA---TAN",CTYPE2="DEC--TAN",CRVAL1=359.99,CRVAL2=70.,CRPIX1=80.,CRPIX2=100.,
                CD1_1=-.0002,CD1_2=.00001,CD2_1=.00001,CD2_2=.0002)


class AstrometryTests(unittest.TestCase):
    def test_tan_reference_pixel_and_ra_wrap_roundtrip(self):
        h=header();reference=plain_tan_world_to_pixel([h["CRVAL1"]],[h["CRVAL2"]],h)
        np.testing.assert_allclose(reference,[[79,99]],atol=1e-9)
        xy=np.random.default_rng(6).uniform(-200,1200,size=(100,2))
        sky=plain_tan_pixel_to_world(xy,h)
        replay=plain_tan_world_to_pixel(sky[:,0],sky[:,1],h)
        np.testing.assert_allclose(replay,xy,atol=2e-9)

    def test_tan_matches_independent_orthographic_vector_construction(self):
        h=header();ra0,dec0=np.deg2rad([h["CRVAL1"],h["CRVAL2"]])
        center=np.array([np.cos(dec0)*np.cos(ra0),np.cos(dec0)*np.sin(ra0),np.sin(dec0)])
        east=np.array([-np.sin(ra0),np.cos(ra0),0]);north=np.cross(center,east)
        offset=np.array([[.02,-.03],[-.05,.04]])
        vec=center+np.deg2rad(offset[:,0,None])*east+np.deg2rad(offset[:,1,None])*north
        vec/=np.linalg.norm(vec,axis=1)[:,None]
        ra=np.rad2deg(np.arctan2(vec[:,1],vec[:,0]))%360;dec=np.rad2deg(np.arcsin(vec[:,2]))
        cd=np.array([[h["CD1_1"],h["CD1_2"]],[h["CD2_1"],h["CD2_2"]]])
        expected=offset@np.linalg.inv(cd).T+[79,99]
        np.testing.assert_allclose(plain_tan_world_to_pixel(ra,dec,h),expected,atol=2e-9)

    def test_projection_mismatch_is_rejected(self):
        h=header();h["CTYPE1"]="RA---SIN"
        with self.assertRaises(ValueError):plain_tan_world_to_pixel([0],[70],h)
        h=header();h["PV1_0"]=0
        with self.assertRaises(ValueError):plain_tan_world_to_pixel([0],[70],h)

    def test_scaled_integer_fits_blank_and_gzip(self):
        cards=["SIMPLE  =                    T","BITPIX  =                   16","NAXIS   =                    2",
               "NAXIS1  =                    3","NAXIS2  =                    2","BSCALE  =                    2",
               "BZERO   =                    7","BLANK   =                  -99","END"]
        native=np.array([[1,2,-99],[0,-3,4]],dtype=">i2")
        raw="".join(c.ljust(80) for c in cards).ljust(2880).encode()+native.tobytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"image.fits";p.write_bytes(raw)
            a,_=read_primary_image(p)
            np.testing.assert_allclose(a,[[9,11,np.nan],[7,1,15]],equal_nan=True)
            p.write_bytes(gzip.compress(raw))
            b,_=read_primary_image(p)
            np.testing.assert_array_equal(a,b)
            p.write_bytes(raw[:-2])
            with self.assertRaises(ValueError):read_primary_image(p)

    def test_footprint_uses_finite_exposure_not_rectangular_extent_or_positive_flux(self):
        image=np.ones((40,40));image[:12,:12]=np.nan;image[20,20]=0
        xy=np.array([[5,5],[11,11],[20,20],[30,30],[0,30]])
        np.testing.assert_array_equal(finite_footprint(xy,image),[False,False,True,True,False])

    def test_gaussian_preserves_constant_and_point_flux(self):
        np.testing.assert_allclose(gaussian_reflect(np.ones((41,43))),1,atol=1e-15)
        x=np.zeros((41,43));x[20,21]=1
        y=gaussian_reflect(x)
        self.assertAlmostEqual(y.sum(),1,places=14)
        np.testing.assert_allclose(y,y[::-1,::-1],atol=1e-15)

    def test_nearest_peak_distance_and_empty_case(self):
        d,i=nearest([[1,1],[5,5]],np.array([[0,0],[4,5]]))
        np.testing.assert_allclose(d,[np.sqrt(2),1]);np.testing.assert_array_equal(i,[0,1])
        d,_=nearest([[1,1]],np.empty((0,2)));self.assertTrue(np.isinf(d[0]))

    def test_catalog_id_merging_requires_all_evidence(self):
        a=dict(nsa_nsaid="12",nsa_iauname="J1234+4567")
        self.assertTrue(same_nsa_object(a,a,.1))
        self.assertFalse(same_nsa_object(a,a,3.))
        self.assertFalse(same_nsa_object(a,dict(nsa_nsaid="13",nsa_iauname="J1234+4567"),.1))
        self.assertFalse(same_nsa_object(a,dict(nsa_nsaid="12",nsa_iauname="J9876+5432"),.1))
        self.assertFalse(same_nsa_object(dict(nsa_nsaid="-999",nsa_iauname="J1234"),dict(nsa_nsaid="-999",nsa_iauname="J1234"),.1))


if __name__=="__main__":unittest.main(verbosity=2)
