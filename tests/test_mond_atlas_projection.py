"""Independent line-of-sight and inverse-source controls, no galaxy velocities."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_source_projection import (projection_matrix,project,adjoint,objective,
    roughness_gradient,fit_nonnegative,weighted_relative_rms)


class ProjectionTests(unittest.TestCase):
    def test_thin_or_face_on_identity(self):
        np.testing.assert_array_equal(projection_matrix(11,.2,0,60),np.eye(11))
        np.testing.assert_array_equal(projection_matrix(11,.2,.6,0),np.eye(11))

    def test_finite_cell_kernel_matches_direct_line_of_sight_quadrature(self):
        n=31;d=.2;height=.27;inc=55.;axis=(np.arange(n)-n//2)*d
        surface=np.exp(-.5*(axis/.8)**2)+.25*np.exp(-.5*((axis-.6)/.15)**2)
        analytic=projection_matrix(n,d,height,inc)@surface
        # Independent midpoint quadrature over observed pixel and physical z.
        z=(np.arange(32000)+.5)/32000*(32*height)-16*height;dz=z[1]-z[0]
        vertical=np.exp(-np.abs(z)/height)/(2*height);tan=np.tan(np.deg2rad(inc));actual=[]
        for center in axis:
            offsets=(np.arange(40)+.5)/40*d-d/2
            coordinates=center+offsets[:,None]-z[None,:]*tan
            indices=np.floor((coordinates-axis[0])/d+.5).astype(int);inside=(indices>=0)&(indices<n)
            rho=np.where(inside,surface[np.clip(indices,0,n-1)],0)*vertical[None,:]
            actual.append(float(np.mean(np.sum(rho,axis=1)*dz)))
        self.assertLess(np.linalg.norm(np.array(actual)-analytic)/np.linalg.norm(analytic),.003)

    def test_flux_centroid_and_finite_field_loss(self):
        a=projection_matrix(301,.2,.3,50);point=np.zeros((1,301));point[0,150]=1
        pred=project(point,a)[0];axis=np.arange(301)-150
        self.assertAlmostEqual(pred.sum(),1,places=10);self.assertAlmostEqual(float(pred@axis),0,places=10)
        point[:]=0;point[0,0]=1
        self.assertLess(project(point,a).sum(),.75) # missing tail must not be renormalized

    def test_adjoint_and_regularization_derivative(self):
        rng=np.random.default_rng(418);s=rng.normal(size=(17,19));v=rng.normal(size=s.shape);a=projection_matrix(19,.2,.4,60)
        self.assertAlmostEqual(float(np.sum(project(s,a)*v)),float(np.sum(s*adjoint(v,a))),places=11)
        weight=rng.uniform(size=s.shape);target=rng.normal(size=s.shape);penalty=.02
        gradient=adjoint(weight*(project(s,a)-target),a)+penalty*roughness_gradient(s);eps=1e-5
        diff=(objective(s+eps*v,target,weight,a,penalty)-objective(s-eps*v,target,weight,a,penalty))/(2*eps)
        self.assertAlmostEqual(diff,float(np.sum(gradient*v)),places=7)

    def test_distinct_depths_can_reproduce_same_positive_image(self):
        axis=np.linspace(-4,4,81);x,y=np.meshgrid(axis,axis,indexing='ij')
        true=np.exp(-.5*((x/1.3)**2+(y/.8)**2));a=projection_matrix(81,.1,.3,60)
        image=project(true,a);weight=np.ones_like(image);support=np.ones_like(image,bool)
        thin=fit_nonnegative(image,weight,projection_matrix(81,.1,0,60),support,regularization=0,tolerance=1e-7)[0]
        thick,result=fit_nonnegative(image,weight,a,support,regularization=1e-5,tolerance=1e-7)
        self.assertTrue(result['converged'])
        self.assertLess(weighted_relative_rms(project(thin,np.eye(81)),image,weight),1e-10)
        self.assertLess(weighted_relative_rms(project(thick,a),image,weight),.01)
        self.assertGreater(np.linalg.norm(thick-thin)/np.linalg.norm(thick),.05)

    def test_omitted_source_pixels_do_not_train_inverse(self):
        target=np.ones((15,17));weight=np.ones_like(target);weight[4:10,5:11]=0;support=np.ones_like(target,bool)
        a=projection_matrix(17,.3,.2,40)
        one=fit_nonnegative(target,weight,a,support)[0];target[weight==0]=123456
        two=fit_nonnegative(target,weight,a,support)[0]
        # Initial guess must not incorporate data excluded by the fit mask.
        np.testing.assert_allclose(one,two,atol=1e-5)


if __name__=='__main__':unittest.main()
