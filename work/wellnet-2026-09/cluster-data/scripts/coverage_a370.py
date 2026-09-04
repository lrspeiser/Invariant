"""Descriptive radial-coverage diagnostics for the Abell 370 lensing data.

This is acquisition QA only: it reports what angular and physical range each
acquired catalogue spans.  It fits nothing and touches no held-out data.
"""
import json
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(LANE, "weaklensing", "buffalo_a370")

# Abell 370 has two comparably bright central galaxies; the adopted reference
# point here is the standard cluster centre used by the HFF/BUFFALO programme.
RA0, DEC0 = 39.9712, -1.5767
ZL = 0.375
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
KPC_PER_ARCMIN = (COSMO.kpc_proper_per_arcmin(ZL)).value  # proper kpc / arcmin


def sep_arcmin(ra, dec):
    dra = (ra - RA0) * np.cos(np.deg2rad(DEC0))
    ddec = dec - DEC0
    return np.hypot(dra, ddec) * 60.0


def describe(name, ra, dec, extra=None):
    r = sep_arcmin(np.asarray(ra, float), np.asarray(dec, float))
    q = np.percentile(r, [0, 5, 50, 95, 100])
    rec = {
        "catalogue": name,
        "n": int(r.size),
        "r_arcmin": {"min": round(float(q[0]), 3), "p5": round(float(q[1]), 3),
                     "median": round(float(q[2]), 3), "p95": round(float(q[3]), 3),
                     "max": round(float(q[4]), 3)},
        "r_kpc_proper": {k: round(v * KPC_PER_ARCMIN, 1) for k, v in
                         zip(["min", "p5", "median", "p95", "max"],
                             [float(x) for x in q])},
    }
    if extra:
        rec.update(extra)
    return rec


out = {"cluster": "Abell 370", "z_lens": ZL,
       "reference_centre_deg": {"ra": RA0, "dec": DEC0},
       "cosmology": "FlatLambdaCDM H0=70, Om0=0.3 (stated for unit conversion only; "
                    "no cosmological inference is made here)",
       "kpc_proper_per_arcmin": round(KPC_PER_ARCMIN, 2),
       "catalogues": []}

# HST-only weak lensing
with fits.open(os.path.join(D, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_wl.fits")) as hl:
    d = hl[1].data
    e = np.hypot(d["e1"], d["e2"])
    out["catalogues"].append(describe(
        "BUFFALO HST weak-lensing sources (wl.fits)", d["ra"], d["dec"],
        {"ellipticity_rms": round(float(np.sqrt(np.mean(e ** 2))), 4),
         "median_shape_SN": round(float(np.median(d["SN"])), 3),
         "per_source_redshift": False}))

# HST + Subaru combined
rows = [l.split() for l in open(
    os.path.join(D, "hlsp_buffalo_hst-subaru_multi_abell370_multi_v1.0_lenstool.cat"))
    if l.strip() and not l.startswith("#")]
a = np.array(rows, dtype=float)
z = a[:, 6]
out["catalogues"].append(describe(
    "BUFFALO HST+Subaru weak-lensing sources (lenstool.cat)", a[:, 1], a[:, 2],
    {"n_with_redshift": int((z > 0).sum()),
     "n_without_redshift": int((z == 0).sum()),
     "redshift_range_of_those_measured": [round(float(z[z > 0].min()), 4),
                                          round(float(z[z > 0].max()), 4)]}))

# Strong lensing images
sl = [l.split() for l in open(
    os.path.join(D, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-final.dat"))
    if l.strip() and not l.startswith("#")]
sra = [float(r[1]) for r in sl]
sdec = [float(r[2]) for r in sl]
sysid = set(r[0].split(".")[0] for r in sl)
classes = {}
for r in sl:
    classes[r[4]] = classes.get(r[4], 0) + 1
out["catalogues"].append(describe(
    "BUFFALO strong-lensing multiple images (sl-final.dat)", sra, sdec,
    {"n_systems": len(sysid), "quality_classes": classes}))

# Cluster members
mem = [l.split() for l in open(
    os.path.join(D, "hlsp_buffalo_hst_multi_abell370_f814w_v1.0_galcat-redseq.cat"))
    if l.strip() and not l.startswith("#")]
out["catalogues"].append(describe(
    "BUFFALO red-sequence cluster members (galcat-redseq.cat)",
    [float(r[1]) for r in mem], [float(r[2]) for r in mem]))

dest = os.path.join(LANE, "weaklensing", "a370_coverage_diagnostics.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
