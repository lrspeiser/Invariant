"""Physical projection, flux and inverse checks for a common source basis."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_nodal_projection import nodal_projection_matrix,project_nodes,adjoint_nodes,fit_nodes
from mond_atlas_source_projection import roughness_gradient


class NodalProjection(unittest.TestCase):
    def test_thin_integrates_triangular_nodes_over_image_cells(self):
        a=nodal_projection_matrix(11,.25,0,60)
        self.assertEqual(a[5,5],.75);self.assertEqual(a[5,4],.125);self.assertEqual(a[5,6],.125)
        self.assertEqual(a[0].sum(),.875)

    def test_matches_physical_linear_interpolation_and_line_of_sight_integral(self):
        n=31;d=.2;axis=(np.arange(n)-n//2)*d;h=.27;inc=55.
        surface=np.exp(-.5*(axis/.8)**2)+.25*np.exp(-.5*((axis-.6)/.15)**2);surface[0]=surface[-1]=0
        expected=nodal_projection_matrix(n,d,h,inc)@surface
        z=(np.arange(32000)+.5)/32000*(32*h)-16*h;dz=z[1]-z[0];vertical=np.exp(-np.abs(z)/h)/(2*h)
        actual=[]
        for center in axis:
            offsets=(np.arange(40)+.5)/40*d-d/2
            coordinates=center+offsets[:,None]-z[None,:]*np.tan(np.deg2rad(inc))
            density=np.interp(coordinates,axis,surface,left=0,right=0)*vertical[None,:]
            actual.append(float(np.mean(density.sum(axis=1)*dz)))
        self.assertLess(np.linalg.norm(np.array(actual)-expected)/np.linalg.norm(expected),.003)

    def test_flux_centroid_and_very_small_height_limit(self):
        a=nodal_projection_matrix(301,.2,.3,50);c=a[:,150];axis=np.arange(301)-150
        self.assertAlmostEqual(float(c.sum()),1,places=10);self.assertAlmostEqual(float(c@axis),0,places=10)
        thin=nodal_projection_matrix(21,.2,1e-7,60)
        np.testing.assert_allclose(thin,nodal_projection_matrix(21,.2,0,60),atol=2e-12)

    def test_two_axis_adjoint_and_objective_derivative(self):
        rng=np.random.default_rng(551);x=rng.normal(size=(15,17));y=rng.normal(size=x.shape);w=rng.uniform(size=x.shape)
        left=nodal_projection_matrix(15,.2,0,60);right=nodal_projection_matrix(17,.2,.3,60)
        self.assertAlmostEqual(float(np.sum(project_nodes(x,left,right)*y)),float(np.sum(x*adjoint_nodes(y,left,right))),places=11)
        penalty=.01;direction=rng.normal(size=x.shape)
        def objective(s):return .5*np.sum(w*(project_nodes(s,left,right)-y)**2)+.5*penalty*sum(np.sum(np.diff(s,axis=a)**2) for a in (0,1))
        eps=1e-5;fd=(objective(x+eps*direction)-objective(x-eps*direction))/(2*eps)
        analytic=adjoint_nodes(w*(project_nodes(x,left,right)-y),left,right)+penalty*roughness_gradient(x)
        self.assertAlmostEqual(fd,float(np.sum(analytic*direction)),places=7)

    def test_omitted_source_measurement_cannot_change_inverse(self):
        target=np.ones((15,17));weight=np.ones_like(target);weight[4:10,5:11]=0;support=np.ones_like(target,bool)
        left=nodal_projection_matrix(15,.2,0,60);right=nodal_projection_matrix(17,.2,.3,60)
        a=fit_nodes(target,weight,left,right,support)[0];target[weight==0]=123456
        b=fit_nodes(target,weight,left,right,support)[0];np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
