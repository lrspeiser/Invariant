from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import hmac
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREDICTORS = ROOT / "runs/gravity/roadmap/item-30-screening-mechanisms-v1-source/predictors.tsv"
ITEM30 = ROOT / "runs/gravity/roadmap/item-30-screening-mechanisms-v1-source/sample-manifest.json"
ITEM31 = ROOT / "runs/gravity/roadmap/item-31-vacuum-permittivity-v1-source/sample-manifest.json"
ROLE_KEY = "invariant-item32-boundary-focus-role-v1-20260828"
BASE = (
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/"
    "v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP"
)


def rank(row: dict[str, str]) -> str:
    return hmac.new(ROLE_KEY.encode(), row["plateifu"].encode(), hashlib.sha256).hexdigest()


def url(row: dict[str, str]) -> str:
    plate, ifu = row["plateifu"].split("-")
    name = f"manga-{row['plateifu']}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    return f"{BASE}/{plate}/{ifu}/{name}"


def check(row: dict[str, str]) -> tuple[str, int, str | None]:
    request = urllib.request.Request(
        url(row), method="HEAD", headers={"User-Agent": "Invariant-Item32-Audit/1.0"}
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return row["plateifu"], int(response.status), response.headers.get("content-length")
        except Exception as error:  # pragma: no cover - audit diagnostic
            last_error = error
            time.sleep(attempt + 1)
    return row["plateifu"], -1, type(last_error).__name__


with PREDICTORS.open(encoding="utf-8", newline="") as handle:
    predictors = list(csv.DictReader(handle, delimiter="\t"))
with ITEM30.open(encoding="utf-8") as handle:
    used30 = {row["plateifu"] for row in json.load(handle)["objects"]}
with ITEM31.open(encoding="utf-8") as handle:
    used31 = {row["plateifu"] for row in json.load(handle)["objects"]}

pool = [
    row
    for row in predictors
    if row["plateifu"] not in used30 | used31
    and float(row["snr_med_g"]) >= 3.0
    and float(row["sersic_index"]) <= 2.5
    and 0.3 <= float(row["axis_ratio"]) <= 0.85
]
mass_median = float(np.median([float(row["log_stellar_mass"]) for row in pool]))
sersic_median = float(np.median([float(row["sersic_index"]) for row in pool]))
cells: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
for row in pool:
    cell = (
        int(float(row["log_stellar_mass"]) > mass_median),
        int(float(row["sersic_index"]) > sersic_median),
    )
    cells[cell].append(row)
selected = [row for cell in sorted(cells) for row in sorted(cells[cell], key=rank)[:12]]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    checks = list(executor.map(check, selected))

print(
    json.dumps(
        {
            "pool": len(pool),
            "mass_median": mass_median,
            "sersic_median": sersic_median,
            "cells": {f"m{key[0]}-n{key[1]}": len(value) for key, value in sorted(cells.items())},
            "selected": len(selected),
            "head_status_counts": dict(sorted(Counter(row[1] for row in checks).items())),
            "head_total_content_length": sum(
                int(row[2]) for row in checks if row[1] == 200 and str(row[2]).isdigit()
            ),
            "checks": sorted(checks),
        },
        indent=2,
        sort_keys=True,
    )
)
