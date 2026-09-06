"""Verify published milestone artifacts against the executed inputs and predictions."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import unittest

import numpy as np

from mond_atlas_common import ROOT,PROTOCOL,digest,read_json,write_json


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(catalog,radial,execution):
    catalog,radial,execution=[p.resolve() for p in (catalog,radial,execution)]
    c,r=read_json(catalog/"summary.json"),read_json(radial/"summary.json")
    for summary in [c,r]:
        assert summary["protocol_sha256"]==digest(PROTOCOL)
        for file,expected in summary["code_hashes"].items():
            assert digest(ROOT/"scripts"/file)==expected, file
    for path,expected in read_json(execution/"input-bindings.json").items():
        assert digest(ROOT/path)==expected,path
    for source in read_json(catalog/"sources.json"):
        assert digest(ROOT/source["path"])==source["sha256"],source["path"]
    galaxies=read_csv(catalog/"galaxies.csv")
    ids=[x["atlas_id"] for x in galaxies]
    assert len(ids)==len(set(ids))==c["catalog_identity_groups"]
    for member in read_csv(catalog/"memberships.csv"):
        assert member["atlas_id"] in set(ids)
    assets=read_csv(catalog/"assets.csv")
    assert len(assets)==c["asset_entries"]
    for asset in assets:
        assert asset["identity_verified"]=="True"
        assert digest(ROOT/asset["path"])==asset["sha256"]
    # Fail if prohibited inherited proxies sneak into a later catalog revision.
    columns=set(galaxies[0])
    assert not any(re.search(r"(?:^|_)(?:mnfw|m200|m500|gext_proxy|dynamical_mass)(?:_|$)",field.lower()) for field in columns)
    points=read_csv(radial/"radial-predictions.csv")
    scores=read_csv(radial/"galaxy-residuals.csv")
    selected=[x for x in scores if x["selected"]=="True"]
    assert len(points)==r["published_radii"]
    assert len(scores)==r["published_galaxies"]
    assert len(selected)==r["selected_galaxies"]
    assert sum(x["selected"]=="True" for x in points)==r["selected_radii"]
    for galaxy in selected:
        rows=[p for p in points if p["galaxy"]==galaxy["galaxy"] and p["selected"]=="True"]
        observed=np.array([float(p["observed_speed_km_s"]) for p in rows])
        for label,key in [("newton","newton_speed_km_s"),("mond","algebraic_mond_speed_km_s")]:
            predicted=np.array([float(p[key]) for p in rows])
            actual=float(np.sqrt(np.mean((predicted/observed-1)**2)))
            assert abs(actual-float(galaxy[label+"_rms_fractional_error"]))<1e-13
    holdout=read_csv(radial/"pattern-holdout-predictions.csv")
    for result in r["patterns"]:
        rows=[x for x in holdout if x["feature"]==result["feature"]]
        assert len(rows)==result["galaxy_count"]
        assert len({x["galaxy"] for x in rows})==len(rows)
        y=np.array([float(x["residual_target"]) for x in rows])
        for key,reported in [("base_prediction","baseline_rmse_dex"),("extended_prediction","extended_rmse_dex")]:
            predicted=np.array([float(x[key]) for x in rows])
            assert abs(np.sqrt(np.mean((predicted-y)**2))-result[reported])<1e-14
    viewer=(execution/"atlas.html").read_text(encoding="utf-8")
    data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',viewer,re.S)[1])
    assert len(data["curves"])==r["published_galaxies"]
    assert sum(len(x) for x in data["curves"].values())==r["published_radii"]
    assert "__ATLAS_DATA__" not in viewer
    links=re.findall(r'\]\(([^)]+)\)',(execution/"README.md").read_text(encoding="utf-8"))
    for link in links:
        if not link.startswith("https:"):
            assert (execution/link).is_file(),link
    spec=importlib.util.spec_from_file_location("test_mond_atlas_offline",ROOT/"tests/test_mond_atlas_offline.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (execution/"validation.log").write_text(log.getvalue(),encoding="utf-8",newline="\n")
    assert result.wasSuccessful(),log.getvalue()
    return dict(status="PASS",tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        actual_raw_asset_files_rehashed=len(assets),catalog_rows_verified=len(galaxies),
        radial_points_verified=len(points),pattern_holdout_predictions_verified=len(holdout),
        viewer_payload_and_local_links_verified=True,viewer_visual_render_not_checked=True,
        git_publication="not_possible_under_current_permissions",goal_complete=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--radial",type=Path,required=True)
    p.add_argument("--execution",type=Path,required=True)
    args=p.parse_args()
    output=args.execution/"verification.json"
    if output.exists():raise FileExistsError(output)
    result=verify(args.catalog,args.radial,args.execution)
    write_json(output,result)
    print(json.dumps(result,indent=2))
