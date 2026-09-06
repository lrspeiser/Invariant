"""Marginal uncertainty for spatially averaged external-background spectra."""
import numpy as np
from mond_atlas_native_covariance import regularized_covariance, gaussian_statistics


def tiles(data, side):
    a=np.asarray(data,float)
    if a.ndim!=4 or not np.isfinite(a).all() or not isinstance(side,int) or side<1:
        raise ValueError('Finite block,y,x,channel array and positive integer side required')
    b,h,w,c=a.shape
    if h%side or w%side: raise ValueError('Tile side must exactly divide both core dimensions')
    return a.reshape(b,h//side,side,w//side,side,c).mean(axis=(2,4))


def fit(training, sides):
    # Validation is deliberately absent from this interface.
    mean=np.asarray(training).mean(axis=(0,1,2)); residual=training-mean
    covariances={}
    for s in sides:
        v=tiles(residual,s).reshape(-1,training.shape[-1])
        covariances[s]=regularized_covariance(v,dict(kind='full',shrinkage=.1))
    return mean,covariances


def scores(data, mean, covariance, side):
    values=tiles(data-mean,side)
    _,q,logp,_=gaussian_statistics(values,covariance)
    n=values.shape[-1]
    return dict(q_over_n=q.mean(axis=(1,2))/n,
                logpdf_per_channel=logp.mean(axis=(1,2))/n,
                trace_second_moment=np.sum(values*values,axis=-1).mean(axis=(1,2)))
