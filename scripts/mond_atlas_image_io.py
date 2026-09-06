"""Restricted primary-FITS image and explicit undistorted TAN geometry.

This is intentionally not a general FITS/WCS replacement. Only simple primary
arrays and the documented TAN/CD contract are supported. Inherited SIP cards
are ignored only through the explicit plain-TAN call; other WCS distortions fail.
Reference: Calabretta & Greisen 2002, https://arxiv.org/abs/astro-ph/0207413.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from mond_atlas_common import fits_primary_header


def read_primary_image(path):
    path=Path(path)
    header=fits_primary_header(path)
    with path.open("rb") as check:compressed=check.read(2)==b"\x1f\x8b"
    opener=gzip.open if compressed else open
    with opener(path,"rb") as stream:
        for _ in range(256):
            block=stream.read(2880)
            if len(block)!=2880:raise ValueError("truncated primary header")
            if any(block[i:i+8].strip()==b"END" for i in range(0,2880,80)):break
        else:raise ValueError("header too large")
        axes=int(header.get("NAXIS",0))
        if axes<2 or axes>4 or header.get("GROUPS",False):raise ValueError("unsupported primary array")
        shape=tuple(int(header["NAXIS"+str(i)]) for i in range(axes,0,-1))
        if any(v<=0 for v in shape) or sum(v>1 for v in shape)!=2:raise ValueError("expected spatial 2D image with optional singleton axes")
        dtype={8:np.dtype("u1"),16:np.dtype(">i2"),32:np.dtype(">i4"),64:np.dtype(">i8"),-32:np.dtype(">f4"),-64:np.dtype(">f8")}.get(int(header["BITPIX"]))
        if dtype is None:raise ValueError("unsupported BITPIX")
        count=int(np.prod(shape))
        raw=stream.read(count*dtype.itemsize)
        if len(raw)!=count*dtype.itemsize:raise ValueError("truncated primary data")
        native=np.frombuffer(raw,dtype=dtype).reshape(shape)
        image=native.astype(float)
        blank=(native==header["BLANK"]) if dtype.kind in "iu" and "BLANK" in header else None
        image=image*float(header.get("BSCALE",1))+float(header.get("BZERO",0))
        if blank is not None:image[blank]=np.nan
    return np.squeeze(image),header


def tan_contract(header):
    if header.get("CTYPE1")!="RA---TAN" or header.get("CTYPE2")!="DEC--TAN":
        raise ValueError("explicit plain TAN/CD contract required")
    if float(header.get("LONPOLE",180))!=180:
        raise ValueError("nondefault celestial pole not supported")
    if float(header.get("LATPOLE",90))!=90:
        raise ValueError("nondefault celestial pole not supported")
    if any(k.startswith(("PV1_","PV2_","CPDIS","D2IM")) for k in header):
        raise ValueError("other distortion conventions not supported")
    if any(header.get("CUNIT"+str(a),"deg").lower() not in ("deg","degree","degrees") for a in (1,2)):
        raise ValueError("degree axes required")
    cd=np.array([[header["CD1_1"],header["CD1_2"]],[header["CD2_1"],header["CD2_2"]]],float)
    if not np.isfinite(cd).all() or np.linalg.det(cd)==0:raise ValueError("singular CD")
    return cd,np.array([header["CRPIX1"],header["CRPIX2"]],float),np.deg2rad([header["CRVAL1"],header["CRVAL2"]])


def plain_tan_world_to_pixel(ra_deg,dec_deg,header):
    cd,crpix,(ra0,dec0)=tan_contract(header)
    ra,dec=np.deg2rad(ra_deg),np.deg2rad(dec_deg)
    dr=ra-ra0
    denominator=np.sin(dec)*np.sin(dec0)+np.cos(dec)*np.cos(dec0)*np.cos(dr)
    if (denominator<=0).any():raise ValueError("sky point outside TAN hemisphere")
    x=np.cos(dec)*np.sin(dr)/denominator
    y=(np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(dr))/denominator
    plane=np.rad2deg(np.stack((x,y),axis=-1))
    return plane@np.linalg.inv(cd).T+crpix-1


def plain_tan_pixel_to_world(xy,header):
    cd,crpix,(ra0,dec0)=tan_contract(header)
    plane=np.deg2rad((np.asarray(xy,float)+1-crpix)@cd.T)
    x,y=plane[...,0],plane[...,1]
    denominator=np.cos(dec0)-y*np.sin(dec0)
    ra=ra0+np.arctan2(x,denominator)
    dec=np.arctan2(np.sin(dec0)+y*np.cos(dec0),np.sqrt(x*x+denominator*denominator))
    return np.stack((np.rad2deg(ra)%360,np.rad2deg(dec)),axis=-1)


def gaussian_reflect(image,sigma=3.,truncate=4.):
    radius=int(truncate*sigma+.5)
    coordinate=np.arange(-radius,radius+1,dtype=float)
    kernel=np.exp(-.5*(coordinate/sigma)**2);kernel/=kernel.sum()
    result=np.asarray(image,float)
    for axis in (0,1):
        padding=[(0,0),(0,0)];padding[axis]=(radius,radius)
        padded=np.pad(result,padding,mode="symmetric")
        out=np.zeros_like(result)
        for i,weight in enumerate(kernel):
            sel=[slice(None),slice(None)];sel[axis]=slice(i,i+result.shape[axis])
            out+=weight*padded[tuple(sel)]
        result=out
    return result


def highpass_peaks(image):
    finite=np.isfinite(image)
    safe=np.where(finite,image,0.)
    high=safe-gaussian_reflect(safe)
    noise=1.4826*np.median(np.abs(high[finite]-np.median(high[finite])))
    padded=np.pad(high,2,mode="symmetric")
    maximum=np.full_like(high,-np.inf)
    for i in range(5):
        for j in range(5):np.maximum(maximum,padded[i:i+image.shape[0],j:j+image.shape[1]],out=maximum)
    yy,xx=np.where(finite&(high==maximum)&(high>8*max(noise,1e-8)))
    peaks=[]
    for y,x in zip(yy,xx):
        if min(y,x)<4 or y>=image.shape[0]-4 or x>=image.shape[1]-4:continue
        patch=np.maximum(high[y-1:y+2,x-1:x+2],0);weight=patch.sum()
        sy,sx=np.mgrid[y-1:y+2,x-1:x+2]
        if weight>0:peaks.append([float(np.sum(sx*patch)/weight),float(np.sum(sy*patch)/weight)])
    return np.asarray(peaks).reshape(-1,2),float(noise)


def finite_footprint(xy,image,radius=3):
    xy=np.asarray(xy,float)
    result=np.zeros(len(xy),bool)
    for i,(x,y) in enumerate(xy):
        if not np.isfinite([x,y]).all():continue
        x,y=int(np.rint(x)),int(np.rint(y))
        if x-radius<0 or y-radius<0 or x+radius>=image.shape[1] or y+radius>=image.shape[0]:continue
        result[i]=np.isfinite(image[y-radius:y+radius+1,x-radius:x+radius+1]).all()
    return result


def nearest(xy,peaks):
    if not len(peaks):return np.full(len(xy),np.inf),np.full(len(xy),-1,dtype=int)
    distance,indices=[],[]
    for start in range(0,len(xy),64):
        d2=np.sum((np.asarray(xy)[start:start+64,None,:]-peaks[None,:,:])**2,axis=2)
        index=np.argmin(d2,axis=1)
        indices.extend(index);distance.extend(np.sqrt(d2[np.arange(len(index)),index]))
    return np.asarray(distance),np.asarray(indices)
