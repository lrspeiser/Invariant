from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sfh"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://califa.caha.es/FTP-PUB/dataproducts/"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def rows(url: str) -> list[list[str]]:
    text = get(url).decode("utf-8")
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(text))
        if row and not row[0].lstrip().startswith("#")
    ]


def normal(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def predecessor_names() -> set[str]:
    values: set[str] = set()
    for path in ROOT.glob("runs/gravity/roadmap/item-*-source/sample-manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                cleaned = normal(value)
                if cleaned and not cleaned.isdigit() and len(cleaned) >= 4:
                    values.add(cleaned)
    return values


obj = {normal(r[0]): r for r in rows(BASE + "DR2_Pipe3D_obj.tab.csv")}
mean = {normal(r[0]): r for r in rows(BASE + "DR2_Pipe3D_mean.tab.csv")}
re_values = {normal(r[0]): r for r in rows(BASE + "DR2_Pipe3D_Re.tab.csv")}
photo_rows = rows(BASE + "photometric_decomposition.csv")
photo: dict[str, list[str]] = {}
for row in photo_rows:
    key = normal(row[1])
    old = photo.get(key)
    if old is None or (float(row[6]) > float(old[6])):
        photo[key] = row

kin_html = get(BASE + "Stellar_Kinematics_V1200/").decode("utf-8")
kin_names = {
    normal(match.group(1)): match.group(1)
    for match in re.finditer(r'href="([^"/]+)\.CALIFA\.V1200\.stekin\.fits"', kin_html)
}
eligible_names = sorted(set(obj) & set(mean) & set(re_values) & set(photo) & set(kin_names))


def download_sfh(key: str) -> tuple[str, str]:
    name = obj[key][0]
    path = OUT / f"{name}.SFH.cube.fits.gz"
    if not path.exists():
        path.write_bytes(get(BASE + f"DR2/Pipe3D/{name}.SFH.cube.fits.gz"))
    return key, str(path)


with ThreadPoolExecutor(max_workers=12) as pool:
    files = dict(pool.map(download_sfh, eligible_names))


def history(path: str) -> dict[str, float] | None:
    with fits.open(path, memmap=False) as hdus:
        data = np.asarray(hdus[0].data[156:195], dtype=float)
        header = hdus[0].header
    ages = np.asarray(
        [float(re.search(r"age\s+0*([0-9.]+)", str(header[f"DESC_{i}"])).group(1)) for i in range(156, 195)]
    )
    finite = np.all(np.isfinite(data), axis=0)
    totals = np.sum(data, axis=0)
    valid = finite & (totals >= 0.5) & (totals <= 1.5)
    if np.count_nonzero(valid) < 100:
        return None
    fractions = np.nanmedian(data[:, valid] / totals[valid], axis=1)
    fractions = np.maximum(fractions, 0.0)
    fractions /= np.sum(fractions)
    return {
        "valid_spaxels": int(np.count_nonzero(valid)),
        "age_mean_gyr": float(np.sum(fractions * ages)),
        "recent_1gyr": float(np.sum(fractions[ages <= 1.0])),
        "old_6gyr": float(np.sum(fractions[ages >= 6.0])),
    }


prior = predecessor_names()
audit = {
    "obj": len(obj),
    "mean": len(mean),
    "re": len(re_values),
    "photo": len(photo),
    "kinematic_names": len(kin_names),
    "intersection": len(eligible_names),
    "predecessor_names": len(prior),
    "rows": [],
}
for key in eligible_names:
    o = obj[key]
    p = photo[key]
    h = history(files[key])
    values = {
        "name": o[0],
        "mass": float(o[5]),
        "z": float(o[4]),
        "mean_age": float(mean[key][1]),
        "metallicity": float(mean[key][3]),
        "disk": int(float(p[6])),
        "bulge": int(float(p[5])),
        "h_r_arcsec": float(p[81]),
        "q_r": float(p[83]),
        "pa_r": float(p[85]),
        "bt_r": float(p[78]),
        "dt_r": float(p[91]),
        "prior": key in prior,
        "history": h,
    }
    quality = (
        h is not None
        and not values["prior"]
        and 8.5 <= values["mass"] <= 11.8
        and 0.003 <= values["z"] <= 0.03
        and values["disk"] == 1
        and 2.0 <= values["h_r_arcsec"] <= 35.0
        and 0.25 <= values["q_r"] <= 0.9
        and 0.0 <= values["dt_r"] <= 1.0
        and 0.0 <= values["bt_r"] <= 1.0
    )
    values["quality"] = quality
    audit["rows"].append(values)

audit["history_valid"] = sum(row["history"] is not None for row in audit["rows"])
audit["prior_overlap"] = sum(row["prior"] for row in audit["rows"])
audit["quality"] = sum(row["quality"] for row in audit["rows"])
(Path(__file__).resolve().parent / "audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({key: value for key, value in audit.items() if key != "rows"}, indent=2))
