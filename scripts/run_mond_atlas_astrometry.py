"""Audit the existing Gaia/image registration results with finite footprint evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mond_atlas_common import ROOT,digest,read_json,write_json,write_csv
from mond_atlas_image_io import read_primary_image,plain_tan_world_to_pixel,highpass_peaks,finite_footprint,nearest

PROTOCOL=ROOT/"configs/mond_atlas_astrometry_v1.json"


def unique_matches(distance,indices,radius):
    good=np.zeros(len(distance),bool);used=set()
    for i in np.argsort(distance):
        if distance[i]<=radius and indices[i] not in used:
            good[i]=True;used.add(indices[i])
    return good


def check(previous,config,rng):
    image_path=ROOT/previous["image_file"]
    catalog_path=ROOT/previous["catalog_file"]
    if digest(image_path)!=previous["image_sha256"] or digest(catalog_path)!=previous["catalog_sha256"]:
        raise ValueError("input image/catalog hash mismatch")
    if previous["selected_wcs"]!="linear_tan":raise ValueError("previous calibration did not choose TAN")
    image,header=read_primary_image(image_path)
    peaks,noise=highpass_peaks(image)
    with catalog_path.open(encoding="utf-8") as stream:catalog={r["source_id"]:r for r in csv.DictReader(stream)}
    matches=previous["matches"]
    source_ids=[r["source_id"] for r in matches]
    ra,dec=[],[]
    for identity in source_ids:
        star=catalog[identity];dt=config["epoch_year"]-float(star["ref_epoch"])
        ra.append(float(star["ra"])+float(star["pmra"])*dt/3.6e6/np.cos(np.deg2rad(float(star["dec"]))))
        dec.append(float(star["dec"])+float(star["pmdec"])*dt/3.6e6)
    xy=plain_tan_world_to_pixel(ra,dec,header)
    scale=np.sqrt(abs(header["CD1_1"]*header["CD2_2"]-header["CD1_2"]*header["CD2_1"]))*3600
    distance,indices=nearest(xy,peaks)
    distance=np.minimum(distance*scale,12.)
    old=np.array([m["linear_distance_arcsec"] for m in matches])
    replay_error=float(np.max(abs(distance-old)))
    if replay_error>config["original_distance_replay_tolerance_arcsec"]:
        raise ValueError("original Astropy/SciPy distance replay failed: "+str(replay_error))
    support=finite_footprint(xy,image)
    calibration=np.array([m["calibration"] for m in matches]);validation=~calibration
    v=support&validation;c=support&calibration
    strict=config["original_strict_gate"]
    passed=(sum(c)>=strict["minimum_calibration"] and sum(v)>=strict["minimum_validation"]
            and np.median(distance[v])<strict["median_arcsec_less_than"]
            and np.quantile(distance[v],.9)<strict["p90_arcsec_less_than"])
    radius=config["association_diagnostic"]["radius_arcsec"]
    matched=unique_matches(distance,indices,radius)&support
    null=[]
    for _ in range(config["association_diagnostic"]["null_shifts"]):
        angle=rng.uniform(0,2*np.pi)
        offset=rng.uniform(*config["association_diagnostic"]["shift_radius_arcsec"])/scale*np.array([np.cos(angle),np.sin(angle)])
        shifted=xy+offset
        supported=finite_footprint(shifted,image)&validation
        d,index=nearest(shifted,peaks)
        hit=unique_matches(d*scale,index,radius)&supported
        null.append(float(sum(hit)/sum(supported)) if sum(supported) else 0.)
    fraction=float(sum(matched&v)/sum(v)) if sum(v) else None
    matched_validation=distance[matched&v]
    rows=[dict(galaxy=previous["name"],source_id=identity,calibration=bool(calibration[i]),
        x_pixel=float(xy[i,0]),y_pixel=float(xy[i,1]),finite_7x7=bool(support[i]),
        original_clipped_distance_arcsec=float(old[i]),replayed_clipped_distance_arcsec=float(distance[i]),
        unique_match_within_1p5arcsec=bool(matched[i]),peak_index=int(indices[i])) for i,identity in enumerate(source_ids)]
    result=dict(galaxy=previous["name"],previous_status=previous["status"],
        input_image=str(image_path.relative_to(ROOT)),image_sha256=previous["image_sha256"],
        input_catalog=str(catalog_path.relative_to(ROOT)),catalog_sha256=previous["catalog_sha256"],
        image_shape=list(image.shape),finite_pixel_fraction=float(np.mean(np.isfinite(image))),
        previous_stars=len(matches),finite_footprint_stars=int(sum(support)),
        unsupported_stars=int(sum(~support)),finite_calibration=int(sum(c)),finite_validation=int(sum(v)),
        original_distance_replay_max_arcsec=replay_error,peak_count=len(peaks),highpass_noise=noise,
        footprint_strict_pass=bool(passed),
        footprint_validation_median_arcsec=float(np.median(distance[v])) if sum(v) else None,
        footprint_validation_p90_arcsec=float(np.quantile(distance[v],.9)) if sum(v) else None,
        matched_validation_stars=int(sum(matched&v)),validation_match_fraction=fraction,
        matched_validation_median_arcsec=float(np.median(matched_validation)) if len(matched_validation) else None,
        matched_validation_p90_arcsec=float(np.quantile(matched_validation,.9)) if len(matched_validation) else None,
        shifted_catalog_null_match_fractions=null,
        chance_association_p=float((1+sum(a>=fraction for a in null))/(len(null)+1)) if fraction is not None else None,
        fitted_coordinate_parameters=0,cleaned_P5_transfer_validated=False,
        association_subset_is_full_astrometry_validation=False,
        note="Original split and WCS preserved; strict gate is recomputed on finite footprint. Matched-subset statistics condition on a 1.5 arcsec match and cannot prove absolute registration tails.")
    return result,rows


def run(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    config=read_json(PROTOCOL)
    source=ROOT/config["source"]
    previous=read_json(source)["objects"]
    rng=np.random.default_rng(config["association_diagnostic"]["seed"])
    results,stars,errors=[],[],[]
    for original in sorted(previous,key=lambda r:r["name"]):
        try:
            result,rows=check(original,config,rng)
            results.append(result);stars.extend(rows)
            write_json(output/(original["name"]+".json"),result)
            print(original["name"],"strict",result["footprint_strict_pass"],"supported",result["finite_footprint_stars"],
                  "validation matches",result["matched_validation_stars"],"/",result["finite_validation"],flush=True)
        except Exception as exc:
            errors.append(dict(galaxy=original["name"],error=str(exc)))
            print("FAIL",original["name"],str(exc),flush=True)
    write_csv(output/"stars.csv",stars)
    write_csv(output/"galaxies.csv",[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results])
    summary=dict(status="FOOTPRINT_AUDIT_EXECUTED" if not errors else "INCOMPLETE_FOOTPRINT_AUDIT",galaxies=len(results),errors=errors,
        input_hashes={source.relative_to(ROOT).as_posix():digest(source),PROTOCOL.relative_to(ROOT).as_posix():digest(PROTOCOL)},
        code_hashes={p.name:digest(p) for p in [Path(__file__),ROOT/"scripts/mond_atlas_image_io.py"]},
        footprint_strict_pass=[r["galaxy"] for r in results if r["footprint_strict_pass"]],
        unsupported_catalog_stars=sum(r["unsupported_stars"] for r in results),
        total_catalog_stars=sum(r["previous_stars"] for r in results),
        maximum_original_distance_replay_arcsec=max([r["original_distance_replay_max_arcsec"] for r in results],default=None),
        new_gravity_predictions=0,cleaned_P5_transfer_validated=False,limitations=config["claim_limits"])
    write_json(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(result["errors"]))
