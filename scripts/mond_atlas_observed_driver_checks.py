"""Manufactured FITS only. This never generates an observational admission receipt."""
import json,tempfile,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from astropy.io import fits
import mond_atlas_observed_driver as d


def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='manufactured-observed-driver-') as tmp:
        tmp=Path(tmp);rows=[]
        for i in range(15):
            x=2+(i%5)*14;y=2+(i//5)*14
            rows.append(dict(x_start=x,x_stop=x+12,y_start=y,y_stop=y+12,subset='training' if i<10 else 'evaluation'))
        v=np.linspace(-60,60,42)
        class Model:
            def __init__(self,spin): self.spin=spin
            def predict(self,p):
                return np.array([p[2]*(1+i*.04)*np.exp(-.5*((v-p[0]-self.spin*(i-7))/p[1])**2) for i in range(15)])
            def unknown_envelope(self,p): return np.full((15,42),.001)
        signal=Model(1).predict([2.,9.,1.2]);cube=np.zeros((42,46,74))
        for i,a in enumerate(rows):cube[:,a['y_start']:a['y_stop'],a['x_start']:a['x_stop']]=signal[i,:,None,None]/1000
        # Nonuniform spatial structure, exactly zero mean, tests all144 pixel average.
        pattern=(np.arange(144).reshape(12,12)-71.5)*1e-7
        for a in rows:cube[:,a['y_start']:a['y_stop'],a['x_start']:a['x_stop']]+=pattern
        header=fits.Header();header['BUNIT']='JY/BEAM'
        fits.writeto(tmp/'a.fits',cube,header)
        ids,train=d.read_aperture_means(tmp/'a.fits',rows,'training')
        scalar=np.array([[sum(float(cube[ch,y,x]) for y in range(a['y_start'],a['y_stop']) for x in range(a['x_start'],a['x_stop']))*1000/144 for ch in range(42)] for a in rows[:10]])
        checks['scalar_mean_max_error']=float(np.max(abs(scalar-train)));assert checks['scalar_mean_max_error']<1e-14
        for a in rows[10:]:cube[:,a['y_start']:a['y_stop'],a['x_start']:a['x_stop']]=np.nan
        fits.writeto(tmp/'b.fits',cube,header)
        _,train2=d.read_aperture_means(tmp/'b.fits',rows,'training');assert np.array_equal(train,train2)
        checks['nonfinite_held_not_materialized_training']=True
        cases=[dict(id='spin'+str(s),kwargs={'spin':s}) for s in [-1,1]]
        bind={str(Path(d.__file__).resolve()):d.digest(d.__file__)}
        factory=lambda c:Model(c['kwargs']['spin'])
        C=.01*.3**abs(np.arange(42)[:,None]-np.arange(42)[None,:])
        m1=d.fit_and_freeze(tmp/'fit1',rows,train,np.zeros(42),C,cases,factory,bind)
        m2=d.fit_and_freeze(tmp/'fit2',rows,train2,np.zeros(42),C,cases,factory,bind)
        assert m1==m2 and all('prediction' in r for r in m1['cases'])
        checks['both_spins_frozen_identical_with_nan_held']=True
        verified=d.verify_frozen(tmp/'fit1',rows)
        _,held=d.read_aperture_means(tmp/'a.fits',rows,'evaluation')
        result=d.evaluate_saved(tmp/'fit1',verified,held)
        checks['true_spin_parameter_max_error']=float(np.max(abs(np.array(m1['cases'][1]['fit']['parameters'])-[2,9,1.2])))
        assert checks['true_spin_parameter_max_error']<1e-5
        wrong=result[0]['score'];r=np.array(wrong['residuals']);independent=np.einsum('ij,jk,ik->i',r,np.linalg.inv(C),r)/42
        checks['independent_covariance_quadratic_max_error']=float(np.max(abs(independent-wrong['per_aperture_q_per_channel'])));assert checks['independent_covariance_quadratic_max_error']<1e-12
        with np.load(tmp/'fit1'/'prediction-0000.npz') as z: perturb=.001*np.sin(np.arange(210).reshape(5,42))
        shifted_residual=r-perturb;actual_q=np.einsum('ij,jk,ik->i',shifted_residual,np.linalg.inv(C),shifted_residual)/42
        assert np.all(actual_q>=wrong['unknown_emission_q_lower_bound']) and np.all(actual_q<=wrong['unknown_emission_q_upper_bound'])
        checks['unknown_emission_interval_contains_manufactured_perturbation']=True
        checks['true_spin_evaluation_q']=result[1]['score']['mean_q_per_channel'];assert checks['true_spin_evaluation_q']<1e-12
        try:d.verify_frozen(tmp/'notfit',rows)
        except FileNotFoundError:checks['evaluation_without_freeze_rejected']=True
        else:raise AssertionError('Missing freeze accepted')
        try:d.read_aperture_means(tmp/'b.fits',rows,'evaluation')
        except ValueError:checks['evaluation_nonfinite_sentinel_rejected']=True
        else:raise AssertionError('NaN held accepted')
        shifted=d.evaluate_saved(tmp/'fit1',verified,held+.1)
        assert shifted[1]['score']['mean_q_per_channel']>result[1]['score']['mean_q_per_channel']
        checks['held_change_only_changes_evaluation']=True
        with (tmp/'fit1'/'prediction-0000.npz').open('ab') as f:f.write(b'tamper')
        try:d.verify_frozen(tmp/'fit1',rows)
        except ValueError:checks['prediction_tamper_rejected']=True
        else:raise AssertionError('Tamper accepted')
        with (tmp/'fit2'/'frozen-manifest.json').open('a') as f:f.write(' ')
        try:d.verify_frozen(tmp/'fit2',rows)
        except ValueError:checks['manifest_tamper_rejected']=True
        else:raise AssertionError('Manifest tamper accepted')
        # Strong gate ordering: instrument reader is forbidden, not merely monitored.
        with patch.object(d,'read_aperture_means',side_effect=AssertionError('Premature cube access')):
            try:d.run('fit',tmp/'real',tmp/'absent-parent-admission.json')
            except FileNotFoundError:checks['missing_admission_rejected_before_cube_values']=True
            else:raise AssertionError('Missing admission accepted')
        checks['distinct_fits_byte_hashes_expected']=d.digest(tmp/'a.fits')!=d.digest(tmp/'b.fits')
    d.write_json(d.PACKAGE/(sys.argv[1] if len(sys.argv)>1 else 'manufactured-controls.json'),{'status':'MANUFACTURED_CONTROLS_PASS','checks':checks,'observed_response_access':False,'limitations':'Byte hashes include held bytes; interpreted held arrays were excluded from training. No real admission generated.'})


if __name__=='__main__':main()
