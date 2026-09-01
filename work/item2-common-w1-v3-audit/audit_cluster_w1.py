from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import spearmanr

from sigma_theory_compiler import gravity_item2_shape_anisotropy as item2_v1
from sigma_theory_compiler import gravity_item2_wise_multipoles as wise

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
CACHE = AUDIT / "unwise-cluster-raw"
CACHE.mkdir(parents=True, exist_ok=True)

config = wise.load_extraction_config(ROOT)
rows = item2_v1._vizier_rows(AUDIT / "clash-table1.tsv")
morphology_source = item2_v1.parse_donahue_morphology(
    ROOT / config["sources"]["clash_xray_morphology"]["path"]
)
morphology = {
    target: morphology_source[source_name]
    for source_name, target in item2_v1.DONAHUE_TO_TARGET.items()
}
cosmology = FlatLambdaCDM(H0=70.0, Om0=0.27, Tcmb0=2.725)
unwise = config["sources"]["unwise_w1"]
extraction = config["image_extraction"]
session = requests.Session()
session.headers.update({"User-Agent": "Invariant-target-blind-common-W1-audit/1.0"})
results = []

for ordinal, row in enumerate(rows, start=1):
    name = row["AName"].strip()
    coordinate = SkyCoord(row["RAJ2000"], row["DEJ2000"], unit=("hourangle", "deg"))
    redshift = float(row["z"])
    kpc_per_arcsec = float(cosmology.kpc_proper_per_arcmin(redshift).value / 60.0)
    aperture_arcsec = 500.0 / kpc_per_arcsec
    cutout_size_arcsec = max(
        float(extraction["minimum_cutout_diameter_arcsec"]),
        float(extraction["cutout_diameter_aperture_multiple"]) * aperture_arcsec,
    )
    pixel_scale = float(unwise["pixel_scale_arcsec"])
    size_pixels = max(64, math.ceil(cutout_size_arcsec / pixel_scale))
    query = urlencode(
        {
            "bands": "1",
            "dec": format(coordinate.dec.degree, ".10f"),
            "layer": str(unwise["layer"]),
            "pixscale": format(pixel_scale, ".8g"),
            "ra": format(coordinate.ra.degree, ".10f"),
            "size": str(size_pixels),
        }
    )
    url = f"{unwise['cutout_service']}?{query}"
    path = CACHE / f"{name}-unwise-neo11-w1.fits"
    if not path.exists():
        response = session.get(url, timeout=180)
        response.raise_for_status()
        if not response.content.startswith(b"SIMPLE"):
            raise ValueError("unWISE response is not FITS")
        temporary = path.with_suffix(".fits.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    with fits.open(path, memmap=False) as handle:
        image = np.squeeze(np.asarray(handle[0].data, dtype=np.float64))
        measured = wise.measure_w1_multipoles(
            image,
            WCS(handle[0].header),
            ra_deg=coordinate.ra.degree,
            dec_deg=coordinate.dec.degree,
            aperture_arcsec=aperture_arcsec,
            inclination_deg=0.0,
            extraction=extraction,
        )
    xray = morphology[name]
    xray_q2 = (1.0 - float(xray["axis_ratio"]) ** 2) / (
        1.0 + float(xray["axis_ratio"]) ** 2
    )
    results.append(
        {
            "name": name,
            "quality": measured["image_quality_pass"],
            "w1_c": measured["concentration_c20"],
            "w1_q2": measured["quadrupole_amplitude"],
            "w1_shift": measured["centroid_shift"],
            "xray_c": float(xray["concentration"]),
            "xray_q2": xray_q2,
            "xray_shift": float(xray["centroid_shift"]),
        }
    )
    print(
        f"[{ordinal:02d}/20] {name} aperture={aperture_arcsec:.2f} "
        f"q2={measured['quadrupole_amplitude']:.4f} quality={measured['image_quality_pass']}"
    )

quality = [row for row in results if row["quality"]]
print(f"quality={len(quality)}/20")
for w1_field, xray_field in (
    ("w1_c", "xray_c"),
    ("w1_q2", "xray_q2"),
    ("w1_shift", "xray_shift"),
):
    statistic = spearmanr(
        [row[w1_field] for row in quality], [row[xray_field] for row in quality]
    ).statistic
    print(f"rho({w1_field},{xray_field})={float(statistic):.6f}")
