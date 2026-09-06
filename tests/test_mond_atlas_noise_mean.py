"""Independent covariance identities and real-pipeline leakage controls."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_noise_mean import mean_residual_covariance,spatial_mean_model,evaluate_mean_branches
from mond_atlas_common import ROOT,read_json
from mond_atlas_image_io import gaussian_reflect
from run_mond_atlas_noise import masks


class MeanCovarianceControls(unittest.TestCase):
    def test_full_joint_linear_transform_and_white_limit(self):
        rng=np.random.default_rng(902);a=rng.normal(size=(11,11));joint=a@a.T+np.eye(11)
        n=6;operator=np.column_stack((-np.ones((5,n))/n,np.eye(5)))
        result,metadata=mean_residual_covariance(joint[:n,:n],joint[n:,n:],joint[n:,:n])
        np.testing.assert_allclose(result,operator@joint@operator.T,rtol=0,atol=1e-12)
        white,metadata=mean_residual_covariance(np.eye(n),np.eye(5),np.zeros((5,n)))
        np.testing.assert_allclose(white,np.eye(5)+np.ones((5,5))/n,rtol=0,atol=1e-12)
        self.assertAlmostEqual(metadata['expected_calibration_residual_variance_factor'],1-1/n)

    def test_correlated_draws_match_whitening_and_variance_loss(self):
        rng=np.random.default_rng(614);positions=np.array([[0,0],[1,0],[0,1],[1,1],[3,0],[3,1],[4,0],[4,1]],float)
        delta=positions[:,None,:]-positions[None,:,:]
        joint=.9*np.exp(-np.sum(delta**2,axis=2)/3)+.1*np.eye(8)
        residual_cov,metadata=mean_residual_covariance(joint[:4,:4],joint[4:,4:],joint[4:,:4])
        draws=rng.normal(size=(40000,8))@np.linalg.cholesky(joint).T
        residual=draws[:,4:]-draws[:,:4].mean(axis=1)[:,None]
        white=np.linalg.solve(np.linalg.cholesky(residual_cov),residual.T)
        self.assertLess(abs(float(np.mean(white**2))-1),.025)
        observed=float(np.mean((draws[:,:4]-draws[:,:4].mean(axis=1)[:,None])**2))
        self.assertLess(abs(observed/metadata['expected_calibration_residual_variance_factor']-1),.025)

    def test_spatial_translation_invariance_and_disjointness(self):
        cal=np.array([[0.,0],[1,2],[3,1]]);test=np.array([[8.,3],[9,4]])
        precision=np.array([[.2,.04],[.04,.1]])
        a,b,m=spatial_mean_model(cal,test,precision,.01)
        aa,bb,mm=spatial_mean_model(cal+[90,-22],test+[90,-22],precision,.01)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);self.assertEqual(m,mm)
        with self.assertRaises(ValueError):spatial_mean_model(cal,cal[:1],precision,.01)

    def test_held_and_galaxy_mutations_do_not_change_calibration(self):
        config=read_json(ROOT/'configs/mond_atlas_noise_v2.json');yy,xx=np.indices((128,128));rng=np.random.default_rng(195)
        packet=dict(cube=np.array([gaussian_reflect(a,2.) for a in rng.normal(size=(24,128,128))]),east=(xx-63.5)*12,north=(yy-63.5)*12)
        train,test=masks(packet['east'],packet['north'],config)
        initial,calibration=evaluate_mean_branches(packet,config,train,test)
        altered={k:v.copy() for k,v in packet.items()};altered['cube'][:,test]*=4
        changed,second=evaluate_mean_branches(altered,config,train,test)
        for key in calibration:np.testing.assert_array_equal(calibration[key],second[key])
        self.assertFalse(changed['branches']['mean_and_variance_corrected']['diagnostic_pass'])
        altered={k:v.copy() for k,v in packet.items()};altered['cube'][:,np.hypot(packet['east'],packet['north'])<500]=1e12
        outside,third=evaluate_mean_branches(altered,config,train,test)
        self.assertEqual(initial,outside)
        for key in calibration:np.testing.assert_array_equal(calibration[key],third[key])


if __name__=='__main__':unittest.main()
