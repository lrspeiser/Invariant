import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mond_atlas_native_spectral import NativeCube,history_provenance,continuum_operator,spectral_covariance


def history(extra=(),weights='1 1 0 0 0 0 0 0 1 1'):
    lines=["UVLIN RELEASE ='31DEC05'","UVLIN INNAME='OBS' INCLASS='DBCON'","UVLIN INSEQ=1",
        "UVLIN OUTNAME='OBS' OUTCLASS='UVLIN'","UVLIN OUTSEQ=1",'UVLIN / Weights (1/1) '+weights,
        'UVLIN ORDER = 1',"IMAGR RELEASE ='31DEC05'","IMAGR INNAME='OBS' INCLASS='UVLIN'",
        'IMAGR INSEQ=1','IMAGR BCHAN=2','IMAGR ECHAN=9','IMAGR NCHAV=1','IMAGR CHINC=1']+list(extra)
    return [('HISTORY '+line).ljust(80) for line in lines]


class NativeSpectralTests(unittest.TestCase):
    def test_native_integer_scaled_blank_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cube.fits'
            fields=[('SIMPLE','T'),('BITPIX','16'),('NAXIS','4'),('NAXIS1','4'),('NAXIS2','3'),('NAXIS3','2'),('NAXIS4','1'),('BSCALE','2.5'),('BZERO','-3'),('BLANK','-99')]
            text=''.join((key.ljust(8)+'= '+value).ljust(80) for key,value in fields)+'END'.ljust(80)
            header=text.encode().ljust(2880,b' ');data=np.arange(24,dtype='>i2').reshape(2,3,4);data[1,2,2]=-99
            path.write_bytes(header+data.tobytes())
            cube=NativeCube(path);actual=cube.sample_plane(1,2);cube.close()
            expected=data[1,::2,::2].astype(float)*2.5-3;expected[-1,-1]=np.nan
            np.testing.assert_array_equal(actual,expected)
            path.write_bytes(header+data.tobytes()[:-1])
            with self.assertRaises(ValueError):NativeCube(path)

    def test_direct_channel_mapping_retains_only_historical_fit_channels(self):
        result=history_provenance(history(),8)
        self.assertTrue(result['direct_channel_mapping'])
        self.assertEqual(result['parent_channel_indices_zero_based'],list(range(1,9)))
        self.assertEqual(result['retained_continuum_fit_stored_indices'],[0,7])
        self.assertFalse(result['certified_line_free_channels'])

    def test_mapping_rejects_assembly_missing_weights_and_wrong_length(self):
        self.assertFalse(history_provenance(history(['MCUBE NPOINTS=20']),8)['direct_channel_mapping'])
        self.assertFalse(history_provenance(history(),9)['direct_channel_mapping'])
        incomplete=history();incomplete[5]=incomplete[5].replace('(1/1)','(1/2)')
        self.assertFalse(history_provenance(incomplete,8)['direct_channel_mapping'])
        duplicate=history();duplicate.insert(6,duplicate[5])
        with self.assertRaises(ValueError):history_provenance(duplicate,8)

    def test_operator_matches_separate_weighted_lstsq_and_annihilates_polynomial(self):
        n=17;cal=np.array([0,1,2,14,15,16]);out=np.arange(1,16);w=np.array([1,2,3,4,2,1.])
        operator=continuum_operator(n,cal,out,1,w);x=np.arange(n,dtype=float)
        np.testing.assert_allclose(operator@np.column_stack((np.ones(n),x)),0,atol=1e-12)
        data=np.random.default_rng(176).normal(size=(n,7));design=np.column_stack((np.ones(n),x))
        coef=np.linalg.lstsq(design[cal]*np.sqrt(w[:,None]),data[cal]*np.sqrt(w[:,None]),rcond=None)[0]
        np.testing.assert_allclose(operator@data,data[out]-design[out]@coef,rtol=0,atol=1e-12)

    def test_known_constant_subtraction_disjoint_and_shared_channels(self):
        operator=continuum_operator(10,[0,1,2,3],[4,5,6,7,8,9],0)
        np.testing.assert_allclose(operator@operator.T,np.eye(6)+np.ones((6,6))/4,rtol=0,atol=1e-12)
        operator=continuum_operator(10,[0,1,2,3],[0,1,2,3],0)
        np.testing.assert_allclose(operator@operator.T,np.eye(4)-np.ones((4,4))/4,rtol=0,atol=1e-12)

    def test_correlated_monte_carlo_covariance_and_filter_construction(self):
        n=13;kernel=np.array([.25,.5,.25]);filter_matrix=np.zeros((n,n+2))
        for i in range(n):filter_matrix[i,i:i+3]=kernel/np.sqrt(kernel@kernel)
        covariance=spectral_covariance(n,True)
        np.testing.assert_allclose(covariance,filter_matrix@filter_matrix.T,atol=1e-12)
        operator=continuum_operator(n,[0,1,2,10,11,12],np.arange(2,11),1)
        expected=operator@covariance@operator.T
        samples=operator@filter_matrix@np.random.default_rng(9864).normal(size=(n+2,60000))
        observed=samples@samples.T/samples.shape[1]
        self.assertLess(np.linalg.norm(observed-expected)/np.linalg.norm(expected),.025)


if __name__=='__main__':unittest.main()
