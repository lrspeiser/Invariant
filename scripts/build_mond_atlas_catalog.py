"""Build the atlas coverage catalog from actual local records, with no downloads."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mond_atlas_common import (PROTOCOL, ROOT, canonical_name, digest, fits_primary_header,
                               number, read_json, sparc_inputs, verify_text_digest, write_csv, write_json)

OUT = ROOT / "work/gravity-first-principles/mond-atlas-catalog-001"


def wallaby_name_position(name):
    match = re.fullmatch(r"WALLABY J(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})", name)
    if not match:
        return None, None
    h, m, s, sign, d, dm, ds = match.groups()
    ra = 15 * (int(h) + int(m) / 60 + int(s) / 3600)
    dec = (int(d) + int(dm) / 60 + int(ds) / 3600) * (-1 if sign == "-" else 1)
    return ra, dec


def proximity_candidates(records, arcsec):
    """Sky-binned neighbor search; proximity flags, never automatic merging."""
    rows = [r for r in records if r["ra_deg"] is not None and r["dec_deg"] is not None]
    rad = np.deg2rad([[r["ra_deg"], r["dec_deg"]] for r in rows])
    xyz = np.column_stack((np.cos(rad[:, 1]) * np.cos(rad[:, 0]),
                           np.cos(rad[:, 1]) * np.sin(rad[:, 0]), np.sin(rad[:, 1])))
    chord = 2 * np.sin(np.deg2rad(arcsec / 3600) / 2)
    bins, result = {}, []
    for i, v in enumerate(xyz):
        cell = tuple(np.floor(v / chord).astype(int))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in bins.get((cell[0]+dx, cell[1]+dy, cell[2]+dz), []):
                        distance = np.linalg.norm(v-xyz[j])
                        if distance <= chord:
                            result.append(dict(id_a=rows[j]["atlas_id"], id_b=rows[i]["atlas_id"],
                                separation_arcsec=float(np.rad2deg(2*np.arcsin(distance/2))*3600),
                                status="REVIEW_REQUIRED_NOT_MERGED"))
        bins.setdefault(cell, []).append(i)
    return result


def build(output):
    if output.exists():
        raise FileExistsError("Use a new run directory: " + str(output))
    output.mkdir(parents=True)
    sources, records, members, assets = [], {}, [], []

    def source(relative, label, url, expected=None):
        p = ROOT / relative
        hashed = digest(p)
        validation = verify_text_digest(p, expected) if expected is not None else "new_snapshot_only"
        sources.append(dict(id=label, path=relative, bytes=p.stat().st_size, sha256=hashed,
                            url=url, prior_hash_verification=validation, expected_prior_sha256=expected))
        return p

    def add(identity, name, survey, row_index, ra=None, dec=None, **values):
        if identity not in records:
            records[identity] = dict(atlas_id=identity, display_name=name,
                ra_deg=ra, dec_deg=dec, surveys=set(), prior_exposure=set(),
                has_local_hi_cube=False, has_local_stellar_image=False,
                has_local_co_image=False, sparc_radial_rows=0,
                wallaby_profile_status="not_in_local_wallaby_extract",
                spatial_3d_baryon_reconstruction="not_demonstrated",
                validated_full_cube_gravity_prediction=False)
        r = records[identity]
        r["surveys"].add(survey)
        if ra is not None and r["ra_deg"] is None:
            r["ra_deg"], r["dec_deg"] = ra, dec
        exposure = values.pop("exposure", "historical_use_not_fully_audited")
        r["prior_exposure"].add(exposure)
        r.update(values)
        members.append(dict(atlas_id=identity, survey=survey, source_row=row_index,
                            source_name=name, exposure=exposure))
        return r

    curves, meta, _, files = sparc_inputs()
    for p in files:
        source(p.relative_to(ROOT).as_posix(), "SPARC_"+p.name,
               "https://astroweb.case.edu/SPARC/")
    for i, g in enumerate(curves):
        name = canonical_name(g["name"])
        add("NAME:"+name, g["name"], "SPARC", i, exposure="development_response_exposed",
            sparc_radial_rows=len(g["rows"]), **meta[g["name"]])

    probes = source("work/item9-probes-schema-audit/main_table.csv", "PROBES_I_catalog",
                    "https://arxiv.org/abs/2209.09912",
                    "885803c3ecf71509eeed804756407cebad9080cc1f1bd258cbd5f8f8fb23b0a2")
    with probes.open(encoding="utf-8") as stream:
        for i, r in enumerate(csv.DictReader(line for line in stream if not line.startswith("#"))):
            name = canonical_name(r["name"])
            add("NAME:"+name, r["name"], "PROBES_I", i, number(r["RA"]), number(r["DEC"]),
                probes_photometry_catalog=True, probes_rotation_survey=r["RC_survey"],
                probes_profile_local_status="not_audited_by_this_ingestion")

    manga_relative = "work/wellnet-2026-09/env-data/clean/manga_env_master.csv"
    manifest = read_json(ROOT / (manga_relative + ".manifest.json"))
    manga = source(manga_relative, "MaNGA_local_derived_master", "https://www.sdss4.org/dr17/manga/",
                   manifest["sha256"])
    source(manga_relative+".manifest.json", "MaNGA_local_manifest", "https://www.sdss4.org/dr17/manga/")
    # Explicit allowlist prevents inherited halo/environment proxies becoming predictors.
    fields = ["plateifu", "mangaid", "nsa_iauname", "nsa_absmag_g", "nsa_absmag_r", "nsa_absmag_i",
              "nsa_elpetro_mass", "nsa_sersic_mass", "nsa_sersic_th50", "nsa_sersic_ba",
              "DAPQUAL", "drp3qual", "SNR_MED_r", "hi_LOGMHI", "hi_LOGHILIM200KMS",
              "hi_SNR", "hi_FHI", "hi_EFHI", "hi_conflag", "hi_conf_prob"]
    with manga.open(encoding="utf-8") as stream:
        manga_rows = list(csv.DictReader(stream))
    if len(manga_rows) != manifest["row_count"]:
        raise ValueError("MaNGA row count mismatch")
    for i, r in enumerate(manga_rows):
        loghi, limit = number(r["hi_LOGMHI"]), number(r["hi_LOGHILIM200KMS"])
        hi_state = ("published_detection" if loghi is not None and loghi > 0
                    else "published_upper_limit" if limit is not None and limit > 0 else "unknown_or_unavailable")
        add("MANGA:"+r["mangaid"], r["mangaid"], "MaNGA_local_master", i,
            number(r["objra"]), number(r["objdec"]), exposure="previously_processed_population_catalog",
            hi_integrated_status=hi_state, manga_source_quality_selection="inherited_custom_10071_row_selection",
            manga_maps_local_status="not_audited_by_this_ingestion",
            **{"manga_"+k: r[k] for k in fields})

    wallaby_relative = "runs/gravity/roadmap/item-10-wallaby-boundaries-v1-source/"
    predictor = read_json(source(wallaby_relative+"predictor-source.json", "WALLABY_local_predictors",
                          "https://doi.org/10.25919/7w8n-9h19"))
    sample = read_json(source(wallaby_relative+"sample-manifest.json", "WALLABY_prior_split",
                       "https://doi.org/10.25919/7w8n-9h19"))
    roles = {r["name"]: r["role"] for r in sample["objects"]}
    for i, r in enumerate(predictor["records"]):
        prior = roles.get(r["name"], "not_admitted_in_item10")
        add("NAME:"+canonical_name(r["name"]), r["name"], "WALLABY_local_predictors", i,
            number(r["ra"]), number(r["dec"]), exposure="item10_"+prior,
            wallaby_profile_status="published_hi_radial_profile_present",
            wallaby_profile_radii=len(r["radius_kpc"]), wallaby_prior_role=prior,
            wallaby_release=r["team_release_kin"], wallaby_stellar_mass_map="not_in_this_extract")
    for i, r in enumerate(predictor["failures"]):
        identity = "NAME:"+canonical_name(r["name"])
        existed = identity in records
        ra, dec = wallaby_name_position(r["name"])
        add(identity, r["name"], "WALLABY_local_predictor_failures", i, ra, dec,
            wallaby_has_failed_release_record=True)
        if not existed:
            records[identity].update(wallaby_profile_status="invalid_in_previous_extractor",
                                     position_source="rounded_catalog_name")

    receipt_paths = ["things-cube-acquisition-001", "stellar-co-acquisition-001",
                     "things-observable-acquisition-003"]
    unique_assets = {}
    for label in receipt_paths:
        relative = "work/gravity-first-principles/"+label+"/receipt.json"
        receipt = read_json(source(relative, label, None))
        for i, item in enumerate(receipt["files"]):
            p = (ROOT / item["file"].replace("\\", "/")).resolve()
            if not p.is_relative_to(ROOT):
                raise ValueError("asset path escaped repo")
            exists = p.is_file()
            if p not in unique_assets:
                unique_assets[p] = (digest(p) if exists else None)
            actual = unique_assets[p]
            size = p.stat().st_size if exists else None
            valid = exists and actual == item["sha256"] and size == item.get("actual_bytes", item["bytes"])
            header = fits_primary_header(p) if valid else {}
            role = item.get("role", "HI_SPECTRAL_CUBE" if "cube" in label
                            else "HI_MOMENT_"+str(item.get("moment")))
            identity = "NAME:"+canonical_name(item["name"])
            r = add(identity, item["name"], "LOCAL_RESOLVED_SEED", label+":"+str(i),
                    number(header.get("CRVAL1")), number(header.get("CRVAL2")),
                    exposure="development_response_exposed")
            if role == "HI_SPECTRAL_CUBE" and valid:
                r["has_local_hi_cube"] = True
            if "STELLAR" in role and "MASK" not in role and valid:
                r["has_local_stellar_image"] = True
            if role.startswith("CO") and valid:
                r["has_local_co_image"] = True
            assets.append(dict(atlas_id=identity, role=role, path=p.relative_to(ROOT).as_posix(),
                receipt=relative, url=item["url"], sha256=actual, expected_sha256=item["sha256"],
                bytes=size, identity_verified=bool(valid), scientific_validation="not_established_by_hash",
                axes=";".join(str(header.get("CTYPE"+str(a), "")) for a in range(1,4)),
                shape="x".join(str(int(header["NAXIS"+str(a)])) for a in range(1,int(header.get("NAXIS",0))+1)),
                bunit=header.get("BUNIT"), beam_major_deg=header.get("BMAJ"),
                beam_minor_deg=header.get("BMIN"), pixel_scale1_deg=header.get("CDELT1"),
                noise_covariance_status="not_validated_by_inventory", mask_status="requires_source_specific_validation"))

    for r in records.values():
        r["surveys"] = ";".join(sorted(r["surveys"]))
        r["prior_exposure"] = ";".join(sorted(r["prior_exposure"]))
    rows = sorted(records.values(), key=lambda r:r["atlas_id"])
    candidates = proximity_candidates(rows, read_json(PROTOCOL)["catalog"]["position_review_arcsec"])
    write_csv(output/"galaxies.csv", rows)
    write_csv(output/"memberships.csv", members)
    write_csv(output/"assets.csv", assets)
    write_csv(output/"identity-review.csv", candidates, ["id_a","id_b","separation_arcsec","status"])
    write_json(output/"sources.json", sources)
    result = dict(status="EXECUTED_LOCAL_COVERAGE_CATALOG_NOT_COMPLETE_ATLAS",
        generated_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(PROTOCOL),
        code_hashes={p.name:digest(p) for p in [Path(__file__), ROOT/"scripts/mond_atlas_common.py"]},
        catalog_identity_groups=len(rows), unique_physical_galaxy_count="not_certified_pending_crossmatch_review",
        memberships=len(members), source_rows_by_survey=dict(Counter(r["survey"] for r in members)),
        crossmatch_review_pairs=len(candidates), identities_without_coordinates=sum(r["ra_deg"] is None for r in rows),
        sparc_rotation_curve_galaxies=len(curves), sparc_published_radii=sum(len(g["rows"]) for g in curves),
        manga_integrated_hi_status=dict(Counter(r["hi_integrated_status"] for r in rows if "hi_integrated_status" in r)),
        resolved_hi_cube_galaxies=sum(r["has_local_hi_cube"] for r in rows),
        asset_entries=len(assets), unique_asset_files=len(unique_assets),
        identity_verified_assets=sum(r["identity_verified"] for r in assets),
        unique_asset_bytes=sum(p.stat().st_size for p in unique_assets if p.exists()),
        validated_full_3d_baryon_galaxies=0, validated_full_cube_gravity_predictions=0,
        new_downloads=0, dark_mass_predictor_columns=0,
        limitations=["Identity groups are not a certified count of distinct physical galaxies. Proximity pairs remain unmerged.",
            "PROBES-I membership is catalog coverage; resolved assets not audited here.",
            "MaNGA input is a prior custom quality selection of 10071 IDs, not the official 10010-galaxy high-quality sample.",
            "WALLABY 303 prior records include repeat releases and failed-profile records; they are not 303 new independent galaxies.",
            "All legacy analyses carry prior-exposure labels; reserved roles do not certify no exposure in other projects.",
            "Finite FITS pixels, hashes and existing cubes do not establish spatial 3D mass, accurate masks or instrument likelihoods.",
            "Imported stellar population masses remain model dependent; integrated HI detections are not total-density maps."])
    write_json(output/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
