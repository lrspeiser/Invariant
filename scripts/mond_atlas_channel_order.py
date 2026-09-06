"""Restricted monotonic index ordering, not a general spectral WCS transform."""
import numpy as np


def increasing_optical_velocity_direction(header,cached_type):
    ctype=header.get('CTYPE3');increment=float(header['CDELT3'])
    if not np.isfinite(increment) or increment==0:raise ValueError('invalid spectral increment')
    if ctype=='FELO-HEL' and cached_type=='VOPT-F2W':
        return 1 if increment>0 else -1
    if ctype=='VELO-HEL' and cached_type=='VRAD' and float(header.get('VELREF',0))>256:
        return 1 if increment>0 else -1
    if ctype=='FREQ' and cached_type=='FREQ':
        endpoints=float(header['CRVAL3'])+(np.array([1.,float(header['NAXIS3'])])-float(header['CRPIX3']))*increment
        if np.min(endpoints)<=0:raise ValueError('frequency axis crosses nonpositive frequency')
        return -1 if increment>0 else 1
    raise ValueError('unsupported or inconsistent spectral-order contract')
