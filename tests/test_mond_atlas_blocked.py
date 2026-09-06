"""Independent and dense-reference checks of global, disk-backed field solves."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mond_atlas_blocked_fields as bf
import mond_atlas_rectangular_fields as dense


class BlockedFields(unittest.TestCase):
    def files(self,path,shape):
        potential=bf.array_file(Path(path)/'phi.npy',shape)
        work=bf.array_file(Path(path)/'work.npy',tuple(n-2 for n in shape))
        return potential,work

    def test_exact_quadratic_with_unequal_grid_and_multiple_pencil_blocks(self):
        axes=[np.linspace(-2,2,19),np.linspace(-3,3,23),np.linspace(-1,1,29)]
        h=[a[1]-a[0] for a in axes];x,y,z=np.meshgrid(*axes,indexing='ij');exact=x*x+2*y*y+3*z*z+x*y-2*x
        with tempfile.TemporaryDirectory() as tmp:
            phi,work=self.files(tmp,exact.shape);phi[:]=exact
            result=bf.poisson_stream(lambda lo,hi:np.full((hi-lo,len(axes[1]),len(axes[2])),12.),phi,work,h,slab_rows=2,max_elements=1200)
            np.testing.assert_allclose(phi,exact,atol=1e-10)
            self.assertTrue(all(t['blocks']>1 for t in result['transform_blocks']))
            del phi,work

    def test_random_source_and_boundaries_match_dense_poisson(self):
        rng=np.random.default_rng(931);shape=(21,19,23);rhs=rng.normal(size=shape);boundary=rng.normal(size=shape);h=[.2,.3,.15]
        reference=dense.poisson(rhs,boundary,h)
        with tempfile.TemporaryDirectory() as tmp:
            phi,work=self.files(tmp,shape);phi[:]=boundary
            bf.poisson_stream(lambda lo,hi:rhs[lo:hi],phi,work,h,slab_rows=3,max_elements=1300)
            np.testing.assert_allclose(phi,reference,rtol=1e-11,atol=1e-12)
            del phi,work

    def test_qumond_halo_seams_and_full_two_stage_solution(self):
        axes=[np.linspace(-4,4,25),np.linspace(-4,4,21),np.linspace(-4,4,31)]
        h=[a[1]-a[0] for a in axes];x,y,z=np.meshgrid(*axes,indexing='ij');r=np.sqrt(x*x+y*y+z*z)
        rho=np.exp(-.5*((x/1.2)**2+(y/.8)**2+(z/.4)**2))
        boundary_n=-1/np.maximum(r,.1);boundary_m=np.log(np.maximum(r,.1))
        pn,pm,_=dense.solve(rho,h,boundary_n,boundary_m,1.,1.)
        expected=dense.qumond_source(pn,h,1.)
        with tempfile.TemporaryDirectory() as tmp:
            phi,work=self.files(tmp,rho.shape);phi[:]=boundary_n
            bf.poisson_stream(lambda lo,hi:4*np.pi*rho[lo:hi],phi,work,h,slab_rows=2,max_elements=1700)
            q=bf.array_file(Path(tmp)/'q.npy',rho.shape);bf.qumond_stream(phi,q,h,1.,slab_rows=2)
            np.testing.assert_allclose(q,expected,rtol=1e-12,atol=1e-12)
            mond=bf.array_file(Path(tmp)/'mond.npy',rho.shape);mond[:]=boundary_m
            bf.poisson_stream(lambda lo,hi:q[lo:hi],mond,work,h,slab_rows=2,max_elements=1700)
            np.testing.assert_allclose(mond,pm,rtol=1e-11,atol=1e-11)
            del phi,work,q,mond

    def test_separable_moments_match_direct_asymmetric_volume_sum(self):
        axes=[np.linspace(-5,5,19),np.linspace(-4,4,21),np.linspace(-3,3,23)]
        h=[a[1]-a[0] for a in axes];x,y=np.meshgrid(axes[0],axes[1],indexing='ij');z=axes[2]
        s=np.exp(-((x-.4)**2/3+(y+.2)**2));v=np.exp(-((z-.15)/.4)**2)
        rho=s[:,:,None]*v[None,None,:]
        bn,bm,reference=dense.multipole_boundary(rho,axes,1.,1.)
        moments=bf.moments_separable([(s,v)],axes,h)
        for key in reference:np.testing.assert_allclose(moments[key],reference[key],rtol=1e-11,atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            a=bf.array_file(Path(tmp)/'n.npy',rho.shape);b=bf.array_file(Path(tmp)/'m.npy',rho.shape)
            bf.fill_boundary(a,axes,moments,1.,1.,'newton');bf.fill_boundary(b,axes,moments,1.,1.,'mond')
            np.testing.assert_allclose(a,bn,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(b,bm,rtol=1e-12,atol=1e-12)
            del a,b


if __name__=='__main__':unittest.main()
