"""Calibrate channel plus spatial covariance on guarded real background patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mond_atlas_common import ROOT,digest,read_json,write_json,write_csv
from mond_atlas_cube import correlated_score

PROTOCOL=ROOT/"configs/mond_atlas_noise_v1.json"


def masks(east,north,config):
    yy,xx=np.indices(east.shape)
    lo,hi=config["sky_annulus_arcsec"]
    sky=np.hypot(east,north)
    background=(sky>lo)&(sky<hi)&(xx>6)&(yy>6)&(xx<east.shape[1]-7)&(yy<east.shape[0]-7)
    side=config["block_side_pixels"];a,b=config["block_interior"]
    interior=(xx%side>=a)&(xx%side<b)&(yy%side>=a)&(yy%side<b)
    parity=(xx//side+yy//side)%2
    train=background&interior&(parity==config["calibration_parity"])
    stride=config["validation_stride_pixels"]
    test=background&interior&(parity==config["validation_parity"])&(xx%stride==0)&(yy%stride==0)
    return train,test


def channel_model(cube,train):
    noise=cube[:,train]
    offset=noise.mean(axis=1)
    residual=noise-offset[:,None]
    std=np.sqrt(np.mean(residual**2,axis=1))
    if (std<=0).any() or not np.isfinite(std).all():raise ValueError("invalid channel variance")
    z=residual/std[:,None]
    nc=len(std)
    lags=np.array([1.]+[float(np.mean(z[k:]*z[:-k]))*(1-k/7) for k in range(1,min(7,nc))]+[0.]*max(0,nc-7))
    corr=lags[np.abs(np.subtract.outer(np.arange(nc),np.arange(nc)))]
    jitter=max(0.,.05-float(np.linalg.eigvalsh(corr).min()))
    corr=(corr+np.eye(nc)*jitter)/(1+jitter)
    cov=std[:,None]*corr*std[None,:]
    white=np.linalg.inv(np.linalg.cholesky(cov))
    return offset,cov,white,jitter


def lag_pairs(shape,dx,dy):
    ny,nx=shape
    a=(slice(max(0,-dy),min(ny,ny-dy)),slice(max(0,-dx),min(nx,nx-dx)))
    b=(slice(max(0,dy),min(ny,ny+dy)),slice(max(0,dx),min(nx,nx+dx)))
    return a,b


def fit_spatial(z,train,min_pairs):
    rows=[]
    for dy in range(0,5):
        for dx in range(-4,5):
            if dy==0 and dx<=0:continue
            a,b=lag_pairs(train.shape,dx,dy)
            support=train[a]&train[b]
            if sum(support.ravel())<min_pairs:continue
            left,right=z[(slice(None),)+a][:,support],z[(slice(None),)+b][:,support]
            denom=np.sqrt(np.mean(left**2)*np.mean(right**2))
            corr=float(np.mean(left*right)/denom)
            rows.append(dict(dx=dx,dy=dy,pairs=int(sum(support.ravel())),correlation=corr))
    chosen=[r for r in rows if .15<r["correlation"]<.98]
    if len(chosen)<5:raise ValueError("insufficient spatial correlation support")
    design=np.array([[r["dx"]**2,2*r["dx"]*r["dy"],r["dy"]**2] for r in chosen])
    y=-2*np.log([r["correlation"] for r in chosen]);w=np.sqrt([r["pairs"] for r in chosen])
    coef=np.linalg.lstsq(design*w[:,None],y*w,rcond=None)[0]
    precision=np.array([[coef[0],coef[1]],[coef[1],coef[2]]])
    if np.linalg.eigvalsh(precision).min()<=0:raise ValueError("spatial Gaussian precision not positive definite")
    return precision,rows


def spatial_covariance(xy,precision,nugget):
    delta=xy[:,None,:]-xy[None,:,:]
    cov=np.exp(-.5*np.einsum("...i,ij,...j->...",delta,precision,delta))
    return (1-nugget)*cov+nugget*np.eye(len(xy))


def check_packet(packet,config):
    cube=np.asarray(packet["cube"],float)
    east,north=packet["east"],packet["north"]
    train,test=masks(east,north,config)
    if sum(train.ravel())<config["minimum_calibration_pixels"] or sum(test.ravel())<config["minimum_validation_pixels"]:
        raise ValueError("insufficient guarded background support")
    offset,cc,white,jitter=channel_model(cube,train)
    centered=cube-offset[:,None,None]
    z=(white@centered.reshape(cube.shape[0],-1)).reshape(cube.shape)
    precision,lags=fit_spatial(z,train,config["minimum_lag_pairs"])
    yy,xx=np.indices(east.shape)
    xy=np.column_stack([xx[test],yy[test]])
    cs=spatial_covariance(xy,precision,config["spatial_nugget"])
    held=centered[:,test]
    score=correlated_score(held,cc,cs)
    joint=np.linalg.solve(np.linalg.cholesky(cs),(white@held).T).T
    # Real-space quadrant scores use each quadrant's own covariance submatrix.
    quadrant=(east[test]>=0).astype(int)+2*(north[test]>=0).astype(int)
    quadrants=[]
    for q in range(4):
        use=quadrant==q
        if sum(use)<config["diagnostic_gates"]["minimum_quadrant_pixels"]:continue
        qs=correlated_score(held[:,use],cc,cs[np.ix_(use,use)])
        quadrants.append(dict(quadrant=q,pixels=int(sum(use)),whitened_mean_square=qs["quadratic_form"]/(cube.shape[0]*int(sum(use)))))
    qmean=score["quadratic_form"]/held.size
    lag=float(np.mean(joint[1:]*joint[:-1]))
    diag=config["diagnostic_gates"]
    a,b=diag["held_whitened_mean_square"];qa,qb=diag["held_quadrant_mean_square"]
    gates=dict(held_mean_square=a<qmean<b,held_channel_lag1=abs(lag)<diag["absolute_held_channel_lag1"],
               spatial_quadrants=len(quadrants)==4 and all(qa<q["whitened_mean_square"]<qb for q in quadrants))
    trainingxy=np.column_stack([east[train],north[train]])
    testxy=np.column_stack([east[test],north[test]])
    separation=float(np.sqrt(np.sum((trainingxy[:,None,:]-testxy[None,:,:])**2,axis=2)).min())
    result=dict(training_background_pixels=int(sum(train.ravel())),validation_background_pixels=int(sum(test.ravel())),
        channels=cube.shape[0],minimum_calibration_validation_separation_arcsec=separation,
        channel_correlation_jitter=jitter,spatial_precision_pixel_minus2=precision.tolist(),
        spatial_correlation_scales_pixels=np.sqrt(1/np.linalg.eigvalsh(precision)).tolist(),
        spatial_nugget=config["spatial_nugget"],calibration_spatial_lags=lags,
        channel_only_validation_mean_square=float(np.mean((white@held)**2)),
        joint_validation_mean_square=float(qmean),joint_validation_channel_lag1=lag,
        quadrants=quadrants,diagnostic_gates=gates,diagnostic_pass=all(gates.values()),
        covariance_exactly_gaussian_and_separable_established=False,galaxy_selection_mask_validated=False,
        cube_gravity_likelihood_admitted=False)
    return result,dict(channel_covariance=cc,spatial_precision=precision,mean_offset=offset,
                        background_calibration_mask=train,background_validation_mask=test)


def run(output,private):
    if output.exists() or private.exists():raise FileExistsError("Use new output and private cache paths")
    output.mkdir(parents=True);private.mkdir(parents=True)
    config=read_json(PROTOCOL)
    source=ROOT/config["source_audit"]
    audits=read_json(source)
    results,failures=[],[]
    for audit in audits:
        name=audit["name"];path=ROOT/config["source_packets"]/(name+".npz")
        try:
            # Only the observed cube and sky coordinates are consumed. In particular,
            # do not load velocity seeds, rotation fits, amplitude or gas descriptors.
            with np.load(path,allow_pickle=False) as data:
                packet={k:data[k] for k in ["cube","east","north"]}
            result,arrays=check_packet(packet,config)
            result.update(galaxy=name,packet_sha256=digest(path),packet_path=path.relative_to(ROOT).as_posix())
            np.savez_compressed(private/(name+".npz"),**arrays)
            write_json(output/(name+".json"),result);results.append(result)
            print(name,"pass",result["diagnostic_pass"],"mean square",round(result["joint_validation_mean_square"],3),
                  "lag1",round(result["joint_validation_channel_lag1"],3),flush=True)
        except Exception as exc:
            failures.append(dict(galaxy=name,error=str(exc)))
            print("FAIL",name,str(exc),flush=True)
    write_csv(output/"galaxies.csv",[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in results])
    summary=dict(status="EXECUTED_BACKGROUND_DIAGNOSTIC" if not failures else "INCOMPLETE_BACKGROUND_DIAGNOSTIC",
        galaxies=len(results),execution_failures=failures,diagnostic_pass=[r["galaxy"] for r in results if r["diagnostic_pass"]],
        diagnostic_fail=[r["galaxy"] for r in results if not r["diagnostic_pass"]],
        input_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in [PROTOCOL,source]},
        code_hashes={p.name:digest(p) for p in [Path(__file__),ROOT/"scripts/mond_atlas_cube.py"]},
        independently_validated_gravity_likelihoods=0,limitations=config["claim_limits"])
    write_json(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--private",type=Path,required=True)
    args=p.parse_args();result=run(args.output,args.private)
    print(json.dumps(result,indent=2));raise SystemExit(bool(result["execution_failures"]))
