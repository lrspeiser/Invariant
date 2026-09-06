"""Propagate the arithmetic calibration mean through background covariance."""
from __future__ import annotations
import numpy as np
from mond_atlas_cube import correlated_score
from run_mond_atlas_noise import spatial_covariance
from run_mond_atlas_noise_robustness import evaluate


def mean_residual_covariance(kcc,ktt,ktc):
    """Cov(e_t - 1 mean(e_c)), with a uniform arithmetic mean.

    This is a marginal linear-transformation identity for supplied covariance
    blocks, not a full conditional predictor or covariance-parameter posterior.
    """
    kcc,ktt,ktc=[np.asarray(a,float) for a in (kcc,ktt,ktc)]
    if kcc.ndim!=2 or ktt.ndim!=2 or kcc.shape[0]!=kcc.shape[1] or ktt.shape[0]!=ktt.shape[1] or ktc.shape!=(len(ktt),len(kcc)) or len(kcc)<2:
        raise ValueError('invalid joint covariance blocks')
    if not all(np.isfinite(a).all() for a in (kcc,ktt,ktc)) or not np.allclose(kcc,kcc.T,rtol=1e-12,atol=1e-14) or not np.allclose(ktt,ktt.T,rtol=1e-12,atol=1e-14):
        raise ValueError('nonfinite or nonsymmetric covariance')
    alpha=float(kcc.mean());cross=ktc.mean(axis=1)
    result=ktt+alpha-cross[:,None]-cross[None,:]
    result=(result+result.T)/2
    return result,dict(calibration_mean_variance_factor=alpha,
        expected_calibration_residual_variance_factor=float(np.trace(kcc)/len(kcc)-alpha),
        maximum_test_mean_cross_covariance=float(np.max(np.abs(cross))))


def spatial_mean_model(calibration_xy,test_xy,precision,nugget):
    calibration_xy,test_xy,precision=[np.asarray(a,float) for a in (calibration_xy,test_xy,precision)]
    if calibration_xy.ndim!=2 or test_xy.ndim!=2 or calibration_xy.shape[1]!=2 or test_xy.shape[1]!=2 or len(test_xy)<1:
        raise ValueError('invalid background positions')
    if not all(np.isfinite(a).all() for a in (calibration_xy,test_xy,precision)) or precision.shape!=(2,2) or not 0<=nugget<=1 or np.linalg.eigvalsh(precision).min()<=0:
        raise ValueError('invalid spatial kernel')
    joined=np.concatenate((calibration_xy,test_xy))
    if len(np.unique(joined,axis=0))!=len(joined):raise ValueError('overlapping or duplicate background positions')
    kcc=spatial_covariance(calibration_xy,precision,nugget)
    ktt=spatial_covariance(test_xy,precision,nugget)
    delta=test_xy[:,None,:]-calibration_xy[None,:,:]
    ktc=(1-nugget)*np.exp(-.5*np.einsum('...i,ij,...j->...',delta,precision,delta))
    corrected,metadata=mean_residual_covariance(kcc,ktt,ktc)
    if not 0<metadata['expected_calibration_residual_variance_factor']<=1:raise ValueError('calibration variance has no valid remaining degrees of freedom')
    np.linalg.cholesky(corrected)
    metadata['effective_independent_calibration_pixels_for_mean']=1/metadata['calibration_mean_variance_factor']
    metadata['calibration_pixels']=len(calibration_xy);metadata['test_pixels']=len(test_xy)
    return ktt,corrected,metadata


def diagnostics(held,cc,cs,quadrants,config):
    score=correlated_score(held,cc,cs)
    white=np.linalg.solve(np.linalg.cholesky(cs),np.linalg.solve(np.linalg.cholesky(cc),held).T).T
    mean_square=float(np.mean(white**2));lag=float(np.mean(white[1:]*white[:-1]))
    rows=[];gates=config['diagnostic_gates']
    for q in range(4):
        use=quadrants==q
        if use.sum()<gates['minimum_quadrant_pixels']:continue
        value=correlated_score(held[:,use],cc,cs[np.ix_(use,use)])['quadratic_form']/held[:,use].size
        rows.append(dict(quadrant=q,pixels=int(use.sum()),mean_square=value))
    a,b=gates['held_whitened_mean_square'];qa,qb=gates['held_quadrant_mean_square']
    passed=dict(held_mean_square=a<mean_square<b,held_channel_lag1=abs(lag)<gates['absolute_held_channel_lag1'],
        spatial_quadrants=len(rows)==4 and all(qa<r['mean_square']<qb for r in rows))
    return dict(mean_square=mean_square,channel_lag1=lag,quadrants=rows,gates=passed,
        diagnostic_pass=all(passed.values()),gaussian_quadratic_form=score['quadratic_form'],
        gaussian_log_determinant=score['log_determinant'],exact_predictive_likelihood_admitted=False)


def evaluate_mean_branches(packet,config,train,test):
    previous,arrays=evaluate(packet,config,train,test)
    yy,xx=np.indices(train.shape)
    calxy=np.column_stack((xx[train],yy[train]));testxy=np.column_stack((xx[test],yy[test]))
    fixed,propagated,metadata=spatial_mean_model(calxy,testxy,arrays['spatial_precision'],config['spatial_nugget'])
    centered=packet['cube'][:,test]-arrays['mean_offset'][:,None]
    cc=arrays['channel_covariance'];corrected_cc=cc/metadata['expected_calibration_residual_variance_factor']
    quadrants=(packet['east'][test]>=0).astype(int)+2*(packet['north'][test]>=0).astype(int)
    branches={}
    for branch,channel,spatial in [('previous_fixed_mean',cc,fixed),('mean_propagated',cc,propagated),('mean_and_variance_corrected',corrected_cc,propagated)]:
        branches[branch]=diagnostics(centered,channel,spatial,quadrants,config)
    replay=branches['previous_fixed_mean']
    if abs(replay['mean_square']-previous['joint_validation_mean_square'])>1e-12 or abs(replay['channel_lag1']-previous['joint_validation_channel_lag1'])>1e-12 or replay['diagnostic_pass']!=previous['diagnostic_pass']:
        raise ArithmeticError('frozen-estimator replay failed')
    return dict(branches=branches,mean_accounting=metadata),dict(**arrays,
        fixed_test_covariance=fixed,mean_propagated_test_covariance=propagated,variance_corrected_channel_covariance=corrected_cc)
