"""Merge atlas identities only with an exact common NSA catalog identifier.

Sky proximity alone remains a review candidate. Original catalog rows, survey
memberships, exposure status and contradictory values remain in the source
catalog; this output supplies a physical-object grouping overlay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from mond_atlas_common import ROOT,digest,write_csv,write_json


def read_csv(path):
    with path.open(encoding="utf-8") as stream:return list(csv.DictReader(stream))


def same_nsa_object(a,b,separation_arcsec):
    """Same-release exact ID, same IAU name, and sub-arcsecond positional sanity."""
    try:aid,bid=int(a["nsa_nsaid"]),int(b["nsa_nsaid"])
    except (ValueError,KeyError,TypeError):return False
    iau_a,iau_b=a.get("nsa_iauname",""),b.get("nsa_iauname","")
    return aid>0 and aid==bid and iau_a.startswith("J") and iau_a==iau_b and separation_arcsec<1.


def resolve(catalog,output):
    catalog,output=catalog.resolve(),output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    galaxies=read_csv(catalog/"galaxies.csv")
    candidates=read_csv(catalog/"identity-review.csv")
    source=ROOT/"work/wellnet-2026-09/env-data/clean/manga_env_master.csv"
    master={r["mangaid"]:r for r in read_csv(source)}
    aliases={r["atlas_id"]:r["atlas_id"] for r in galaxies}
    evidence=[]
    for pair in candidates:
        ida,idb=pair["id_a"],pair["id_b"]
        if not (ida.startswith("MANGA:") and idb.startswith("MANGA:")):continue
        a,b=master[ida[6:]],master[idb[6:]]
        if same_nsa_object(a,b,float(pair["separation_arcsec"])):
            root=min(aliases[ida],aliases[idb]);old={aliases[ida],aliases[idb]}
            aliases={identity:root if parent in old else parent for identity,parent in aliases.items()}
            evidence.append(dict(id_a=ida,id_b=idb,nsa_nsaid=a["nsa_nsaid"],nsa_iauname=a["nsa_iauname"],
                separation_arcsec=float(pair["separation_arcsec"]),source_row_mangaids=[a["mangaid"],b["mangaid"]],
                source=source.relative_to(ROOT).as_posix(),status="MERGE_BY_EXACT_SAME_RELEASE_CATALOG_ID",
                motion_values_used=False))
    grouped={}
    for galaxy in galaxies:
        identity=aliases[galaxy["atlas_id"]]
        group=grouped.setdefault(identity,dict(object_group_id=identity,source_identity_ids=[],names=[],surveys=set(),prior_exposures=set()))
        group["source_identity_ids"].append(galaxy["atlas_id"])
        group["names"].append(galaxy["display_name"])
        group["surveys"].update(galaxy["surveys"].split(";"))
        group["prior_exposures"].update(galaxy["prior_exposure"].split(";"))
    rows=[]
    for group in sorted(grouped.values(),key=lambda r:r["object_group_id"]):
        rows.append({**{k:";".join(sorted(v)) for k,v in group.items() if k!="object_group_id"},
            "object_group_id":group["object_group_id"],"source_identities":len(group["source_identity_ids"]),
            "group_holdout_key":hashlib.sha256(("mond-atlas-object-group-v1|"+group["object_group_id"]).encode()).hexdigest()})
    unresolved={}
    for pair in candidates:
        a,b=sorted((aliases[pair["id_a"]],aliases[pair["id_b"]]))
        if a==b:continue
        key=(a,b)
        if key not in unresolved or float(pair["separation_arcsec"])<unresolved[key]["minimum_original_separation_arcsec"]:
            unresolved[key]=dict(id_a=a,id_b=b,minimum_original_separation_arcsec=float(pair["separation_arcsec"]),status="REVIEW_REQUIRED_NOT_MERGED")
    write_csv(output/"object-groups.csv",rows)
    write_csv(output/"identity-to-object.csv",[dict(atlas_identity_id=k,object_group_id=v) for k,v in sorted(aliases.items())])
    write_csv(output/"identity-review.csv",list(unresolved.values()))
    write_json(output/"merge-evidence.json",evidence)
    summary=dict(status="EXACT_IDENTITY_GROUPING_OVERLAY",source_identity_rows=len(galaxies),
        object_groups=len(rows),exact_catalog_id_merges=len(evidence),unresolved_proximity_pairs=len(unresolved),
        certified_distinct_physical_galaxy_count=False,
        original_rows_deleted=0,motion_or_gravity_values_used=False,
        code_sha256=digest(Path(__file__)),
        source_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in
            [source,catalog/"galaxies.csv",catalog/"identity-review.csv"]},
        rule="Same positive NSA ID in same local MaNGA release, same NSA IAU name, and separation below one arcsecond. All other proximity pairs remain unmerged.",
        holdouts="Future galaxy/group splits must use object_group_id, with exposure inherited from every source identity.",
        limitations=["The 90 remaining proximity pairs need independent identity evidence.",
                     "58 source identities still lack coordinates in the base catalog.",
                     "This overlay does not silently combine discrepant photometry or count repeated observations as independent galaxies."])
    write_json(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(resolve(args.catalog,args.output),indent=2))
