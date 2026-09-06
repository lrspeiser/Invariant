"""Restricted native-cube/history reader and continuum covariance algebra."""
from __future__ import annotations
import hashlib,re
from pathlib import Path
import numpy as np
from mond_atlas_common import fits_primary_header


def primary_cards(path):
    blocks=[];cards=[]
    with Path(path).open('rb') as stream:
        for _ in range(256):
            block=stream.read(2880)
            if len(block)!=2880:raise ValueError('truncated/uncompressed primary FITS header required')
            blocks.append(block)
            for offset in range(0,2880,80):
                card=block[offset:offset+80].decode('ascii');cards.append(card)
                if card[:8].strip()=='END':
                    return cards,len(blocks)*2880,hashlib.sha256(b''.join(blocks)).hexdigest()
    raise ValueError('primary header limit exceeded')


class NativeCube:
    """Read-only uncompressed primary cube; no generalized WCS or FITS checksum claim."""
    def __init__(self,path):
        self.path=Path(path);self.header=fits_primary_header(path)
        self.cards,self.offset,self.header_sha256=primary_cards(path);h=self.header
        axes=int(h.get('NAXIS',0))
        if axes not in (3,4) or h.get('GROUPS',False) or (axes==4 and h.get('NAXIS4')!=1):
            raise ValueError('expected primary spectral cube with optional singleton fourth axis')
        self.shape=tuple(int(h['NAXIS'+str(i)]) for i in (3,2,1))
        dtype={8:'u1',16:'>i2',32:'>i4',64:'>i8',-32:'>f4',-64:'>f8'}.get(int(h['BITPIX']))
        if dtype is None or min(self.shape)<1:raise ValueError('unsupported primary array')
        size=int(np.prod(self.shape))*np.dtype(dtype).itemsize
        if self.path.stat().st_size<self.offset+size:raise ValueError('truncated primary cube data')
        self._data=np.memmap(path,dtype=dtype,mode='r',offset=self.offset,shape=self.shape)

    def sample_plane(self,index,stride=1):
        if not isinstance(index,(int,np.integer)) or not 0<=index<self.shape[0] or not isinstance(stride,int) or stride<1:
            raise ValueError('invalid stored channel index or stride')
        native=self._data[index,::stride,::stride];result=native.astype(float)
        blank=(native==self.header['BLANK']) if native.dtype.kind in 'iu' and 'BLANK' in self.header else None
        result=result*float(self.header.get('BSCALE',1))+float(self.header.get('BZERO',0))
        if blank is not None:result[blank]=np.nan
        return result

    def close(self):
        self._data._mmap.close()


def task_groups(cards,task):
    groups=[]
    for index,card in enumerate(cards):
        if card[:8].strip()!='HISTORY':continue
        line=card[8:].strip()
        if not re.match(re.escape(task)+r'\s',line,re.I):continue
        start=bool(re.match(re.escape(task)+r'\s+RELEASE\s*=',line,re.I))
        if start or not groups:groups.append(dict(release_record_present=start,card_indexes=[],lines=[]))
        groups[-1]['card_indexes'].append(index);groups[-1]['lines'].append(line)
    for group in groups:
        fields={}
        for line in group['lines']:
            for key,string,numeric in re.findall(r"\b(INNAME|INCLASS|INSEQ|INDISK|OUTNAME|OUTCLASS|OUTSEQ|OUTDISK|BCHAN|ECHAN|NCHAV|CHINC|ORDER)\s*=\s*(?:'([^']*)'|([+-]?\d+))",line):
                value=int(numeric) if numeric else string.strip()
                if key in fields and fields[key]!=value:raise ValueError('conflicting '+task+' '+key)
                fields[key]=value
        group['fields']=fields
        if task=='UVLIN':
            weights={}
            for line in group['lines']:
                match=re.search(r'Weights\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*([01](?:\s+[01])*)\s*$',line)
                if not match:continue
                iff,start=int(match[1]),int(match[2]);current=weights.setdefault(iff,{})
                for j,value in enumerate(map(int,match[3].split()),start):
                    if j in current:raise ValueError('duplicate UVLIN weight channel')
                    current[j]=value
            group['weights_by_if']={str(iff):[current[i] for i in range(1,max(current)+1)] for iff,current in weights.items() if set(current)==set(range(1,max(current)+1))}
            group['weights_complete']=bool(weights) and len(group['weights_by_if'])==len(weights)
    return groups


def history_provenance(cards,channels):
    uv=task_groups(cards,'UVLIN');imagr=task_groups(cards,'IMAGR');reasons=[]
    result=dict(uvlin_groups=uv,imagr_groups=imagr,stored_channels=channels,
        direct_channel_mapping=False,retained_continuum_fit_stored_indices=None)
    if len(uv)!=1:reasons.append('UVLIN_HISTORY_NOT_SINGLE_GROUP')
    if len(imagr)!=1:reasons.append('IMAGR_HISTORY_NOT_SINGLE_GROUP')
    if len(uv)==1 and len(imagr)==1:
        u,m=uv[0],imagr[0];uf,mf=u['fields'],m['fields']
        if not u['release_record_present'] or not m['release_record_present']:reasons.append('MISSING_TASK_START')
        if any(uf.get('OUT'+k)!=mf.get('IN'+k) or uf.get('OUT'+k) is None for k in ('NAME','CLASS','SEQ')):
            reasons.append('UVLIN_TO_IMAGR_DATASET_ID_NOT_MATCHED')
        if mf.get('NCHAV')!=1 or mf.get('CHINC')!=1:reasons.append('CHANNEL_AVERAGING_OR_INCREMENT_UNSUPPORTED')
        first,last=mf.get('BCHAN'),mf.get('ECHAN')
        if first is None or last is None or first<1 or last-first+1!=channels:reasons.append('IMAGR_CHANNEL_COUNT_DIFFERS_FROM_STORED_CUBE')
        if not u['weights_complete'] or set(u['weights_by_if'])!={'1'}:reasons.append('SINGLE_COMPLETE_IF_WEIGHT_VECTOR_MISSING')
        if uf.get('ORDER') not in (0,1):reasons.append('POLYNOMIAL_ORDER_UNSUPPORTED')
        start=u['card_indexes'][-1]
        tasks=sorted({c[8:].strip().split()[0] for c in cards[start+1:] if c[:8].strip()=='HISTORY' and c[8:].strip()})
        result['post_uvlin_task_names']=tasks
        unsupported=set(tasks)-{'IMAGR','AIPS','MOVE','RENAM','TVFLG','UVFLG'}
        if unsupported:reasons.append('POST_UVLIN_PROCESSING_UNRESOLVED:'+','.join(sorted(unsupported)))
        if not reasons:
            weights=u['weights_by_if']['1']
            if last>len(weights):reasons.append('OUTPUT_CHANNELS_NOT_COVERED_BY_WEIGHT_HISTORY')
            else:
                indices=list(range(first-1,last));selected=[j for j,i in enumerate(indices) if weights[i]>0]
                result.update(direct_channel_mapping=True,parent_channel_indices_zero_based=indices,
                    continuum_fit_parent_indices_zero_based=[i for i,w in enumerate(weights) if w>0],
                    retained_continuum_fit_stored_indices=selected,
                    parent_channel_count=len(weights),polynomial_order=uf['ORDER'])
    result['unresolved_reasons']=reasons
    result['certified_line_free_channels']=False
    result['per_visibility_flags_and_imaging_covariance_recovered']=False
    return result


def continuum_operator(parent_channels,calibration_indices,output_indices,order,weights=None):
    """Explicit linear transform for a hypothetical common weighted spectral fit."""
    cal=np.asarray(calibration_indices,int);out=np.asarray(output_indices,int)
    if order not in (0,1) or parent_channels<2 or cal.ndim!=1 or out.ndim!=1 or len(cal)<=order or len(out)<1:
        raise ValueError('invalid continuum fit geometry')
    if len(np.unique(cal))!=len(cal) or len(np.unique(out))!=len(out) or min(cal.min(),out.min())<0 or max(cal.max(),out.max())>=parent_channels:
        raise ValueError('invalid/duplicate channel indices')
    weights=np.ones(len(cal)) if weights is None else np.asarray(weights,float)
    if weights.shape!=cal.shape or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('invalid fit weights')
    x=np.linspace(-1,1,parent_channels);design=np.column_stack([x**k for k in range(order+1)])
    xc=design[cal];mapping=np.linalg.solve(xc.T@(weights[:,None]*xc),xc.T*weights)
    selector=np.eye(parent_channels)[out];correction=np.zeros_like(selector)
    correction[:,cal]=design[out]@mapping
    return selector-correction


def spectral_covariance(parent_channels,hanning=False):
    if not hanning:return np.eye(parent_channels)
    # Stationary covariance from a 3-tap filter on an extended infinite white sequence.
    kernel=np.array([.25,.5,.25]);lags=np.correlate(kernel,kernel,mode='full')[2:];lags=lags/lags[0]
    difference=np.abs(np.subtract.outer(np.arange(parent_channels),np.arange(parent_channels)))
    result=np.zeros_like(difference,float)
    for lag,value in enumerate(lags):result[difference==lag]=value
    return result


def continuum_controls(provenance):
    if not provenance['direct_channel_mapping']:return None
    n=provenance['parent_channel_count'];cal=provenance['continuum_fit_parent_indices_zero_based'];out=provenance['parent_channel_indices_zero_based'];order=provenance['polynomial_order']
    operator=continuum_operator(n,cal,out,order);design=np.column_stack([np.linspace(-1,1,n)**k for k in range(order+1)])
    error=float(np.max(np.abs(operator@design)))
    if error>1e-12:raise ArithmeticError('continuum polynomial not annihilated')
    result=dict(polynomial_annihilation_error=error,parent_channels=n,fit_channels=len(cal),output_channels=len(out),polynomial_order=order,branches={})
    for hanning in [False,True]:
        original=spectral_covariance(n,hanning);covariance=operator@original@operator.T
        std=np.sqrt(np.diag(covariance));corr=covariance/std[:,None]/std[None,:]
        lags={str(lag):float(np.mean(np.diag(corr,lag))) for lag in [1,2,3,6,12] if lag<len(out)}
        result['branches']['hanning_unit_variance' if hanning else 'independent_unit_variance']=dict(
            residual_variance_min=float(np.diag(covariance).min()),residual_variance_max=float(np.diag(covariance).max()),
            correlation_lags=lags,minimum_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
            nonlocal_continuum_covariance_retained=True)
    return result


def robust_region(values):
    values=np.asarray(values,float)
    if len(values)<10 or not np.isfinite(values).all():raise ValueError('insufficient finite region')
    median=float(np.median(values));mad=float(1.482602218505602*np.median(np.abs(values-median)))
    if mad<=0:raise ValueError('zero robust noise scale')
    z=(values-median)/mad;q05,q50,q95=np.quantile(z,[.05,.5,.95])
    return dict(pixels=len(values),median_jy_per_beam=median,mad_scale_jy_per_beam=mad,
        upper_to_lower_90pct_tail_ratio=float((q95-q50)/(q50-q05)),
        above_3_mad_fraction=float(np.mean(z>3)),below_minus3_mad_fraction=float(np.mean(z<-3))),z
