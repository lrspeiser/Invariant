"""Fixed-axis geometry and descriptive nuisance-adjusted associations."""
import numpy as np
from astropy.coordinates import SkyCoord, FK5, BarycentricMeanEcliptic
import astropy.units as u


def vector(lon,lat):
    l,b=np.deg2rad(lon),np.deg2rad(lat)
    return np.stack([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)],axis=-1)


def sky_features(ra,dec,axes):
    c=SkyCoord(ra=np.asarray(ra)*u.deg,dec=np.asarray(dec)*u.deg,frame=FK5(equinox='J2000'))
    g=c.galactic; e=c.transform_to(BarycentricMeanEcliptic(equinox='J2000'))
    directions=vector(g.l.deg,g.b.deg)
    av={k:vector(*v) for k,v in axes.items()}
    bisector=sum(av.values());av['bisector']=bisector/np.linalg.norm(bisector)
    values={}
    for name,a in av.items():
        dot=directions@a
        values[name+'_signed']=dot;values[name+'_axial']=(3*dot**2-1)/2
    values.update(galactic_signed=directions[:,2],galactic_absolute=np.abs(directions[:,2]),
                  ecliptic_axial=(3*np.sin(e.lat.rad)**2-1)/2,equatorial_signed=np.sin(np.deg2rad(dec)))
    return np.column_stack(list(values.values())),list(values),dict(l_deg=g.l.deg,b_deg=g.b.deg,
        ecliptic_b_deg=e.lat.deg,octant=(directions[:,0]>=0).astype(int)+2*(directions[:,1]>=0)+4*(directions[:,2]>=0))


def residualizer(x):
    x=np.asarray(x,float);scale=np.maximum(x.std(axis=0),1e-12)
    design=np.column_stack([np.ones(len(x)),(x-x.mean(axis=0))/scale])
    q,s,_=np.linalg.svd(design,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-12));q=q[:,:rank]
    return np.eye(len(x))-q@q.T,rank


def associations(x,y,sky,names,permutations=1999,seed=9060711):
    m,rank=residualizer(x); yr=m@y; sr=m@sky
    yn=np.linalg.norm(yr);sn=np.linalg.norm(sr,axis=0)
    if yn<1e-12 or np.any(sn<1e-10): raise ValueError('Unidentifiable target or sky column')
    r=yr@sr/(yn*sn); rng=np.random.default_rng(seed); null=[]
    for _ in range(permutations):
        yp=m@rng.permutation(yr)
        null.append(float(np.max(np.abs(yp@sr/(np.linalg.norm(yp)*sn)))))
    rows=[]
    for i,name in enumerate(names):
        rows.append(dict(feature=name,raw_r=float(np.corrcoef(y,sky[:,i])[0,1]),partial_r=float(r[i]),
            partial_slope_dex_per_feature=float(sr[:,i]@yr/(sr[:,i]@sr[:,i])),
            sky_variance_fraction_after_controls=float(np.sum(sr[:,i]**2)/np.sum((sky[:,i]-sky[:,i].mean())**2)),
            maxstat_reference_fraction=float((1+np.sum(np.asarray(null)>=abs(r[i])))/(len(null)+1))))
    return rows,null,yr,sr,rank
