"""Open each Bolocam tarball, read the FITS headers, and fold the verified
facts back into the manifests.

Checks that the archive really delivered an SZ map for the intended cluster:
the map centre must sit within 5 arcmin of the archive's stated position.
"""
import io
import json
import os
import sys
import tarfile
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import LANE  # noqa: E402

D = os.path.join(LANE, "gas", "bolocam_sz")


def hms_to_deg(h):
    a, b, c = [float(x) for x in h.split(":")]
    return 15.0 * (a + b / 60.0 + c / 3600.0)


def dms_to_deg(d):
    sign = -1.0 if d.strip().startswith("-") else 1.0
    a, b, c = [abs(float(x)) for x in d.replace("+", "").split(":")]
    return sign * (a + b / 60.0 + c / 3600.0)


summary = []
for fn in sorted(os.listdir(D)):
    if not fn.endswith(".tgz"):
        continue
    path = os.path.join(D, fn)
    mpath = path + ".manifest.json"
    man = json.load(open(mpath, encoding="utf-8"))
    info = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".fits"):
                continue
            key = os.path.basename(m.name)
            if key not in ("unfiltered_image.fits", "filtered_image.fits"):
                continue
            with fits.open(io.BytesIO(tf.extractfile(m).read())) as hl:
                h = hl[0]
                d = np.asarray(h.data, dtype=float)
                info[key] = {
                    "shape": list(d.shape),
                    "bunit": h.header.get("BUNIT"),
                    "ctype1": h.header.get("CTYPE1"),
                    "crval1_deg": float(h.header.get("CRVAL1")),
                    "crval2_deg": float(h.header.get("CRVAL2")),
                    "min": round(float(np.nanmin(d)), 3),
                    "max": round(float(np.nanmax(d)), 3),
                }
    ra_arch = hms_to_deg(man["archive_position_ra_hms"])
    dec_arch = dms_to_deg(man["archive_position_dec_dms"])
    u = info["unfiltered_image.fits"]
    dra = (u["crval1_deg"] - ra_arch) * np.cos(np.deg2rad(dec_arch))
    ddec = u["crval2_deg"] - dec_arch
    offset_arcmin = float(np.hypot(dra, ddec) * 60.0)
    ok = offset_arcmin < 5.0

    man["fits_header_verification"] = {
        "images": info,
        "map_centre_vs_archive_position_offset_arcmin": round(offset_arcmin, 3),
        "position_check_passed": bool(ok),
        "units": "uK_CMB -- CMB temperature decrement, proportional to Compton-y",
        "sz_decrement_detected": bool(u["min"] < -50.0),
        "caveat": ("The maps are small: the unfiltered image is %dx%d pixels and the "
                   "filtered image %dx%d. Combined with the 58 arcsec Bolocam beam this "
                   "constrains the integrated electron pressure on arcminute scales over "
                   "roughly 0.1-3.5 R500. It carries NO core structure and must not be "
                   "used as a resolved pressure map."
                   % (info["unfiltered_image.fits"]["shape"][0],
                      info["unfiltered_image.fits"]["shape"][1],
                      info["filtered_image.fits"]["shape"][0],
                      info["filtered_image.fits"]["shape"][1])),
        "filtered_map_warning": ("filtered_image.fits has had large-scale modes removed by "
                                 "the reduction pipeline. It must be compared to a model "
                                 "passed through filtered_image_signal_transfer_function.fits, "
                                 "never treated as sky truth."),
    }
    json.dump(man, open(mpath, "w", encoding="utf-8"), indent=2)
    summary.append({"file": fn, "cluster": man["cluster"],
                    "offset_arcmin": round(offset_arcmin, 3), "passed": ok,
                    "unfiltered_min_uK": u["min"]})
    print("%-28s %-34s offset %.2f'  %s  min %.0f uK" %
          (fn, man["cluster"], offset_arcmin, "PASS" if ok else "FAIL", u["min"]))

print("\nall positions verified:", all(s["passed"] for s in summary))
