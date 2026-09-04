"""Audit the one VizieR CONE search used anywhere in this lane.

A coordinator warning reported that VizieR's `-c` / `-c.rs` cone search can
silently return zero rows with no error.  A search that can silently return
zero can also silently return too few, so the one cone-derived product here --
the Simard+2011 pure-Sersic fits around Abell 2029 -- is checked for radial
truncation: in a complete cone the surface density in equal-AREA annuli is flat
apart from the cluster's own overdensity at small radius.
"""
import json
import os
import numpy as np

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(LANE, "members",
                 "A2029field_Simard2011_ApJS196_11_table3_pureSersic_cone110arcmin.raw.tsv")
RA0, DE0, RS_ARCMIN = 227.7337, 5.7444, 110.0

lines = open(P, encoding="utf-8", errors="replace").read().splitlines()
sep = [i for i, l in enumerate(lines) if l.startswith("---")]
i = sep[-1]
hdr = [h.strip() for h in lines[i - 2].split("\t")]
rows = [l.split("\t") for l in lines[i + 1:] if l.strip() and not l.startswith("#")]


def gc(name):
    j = hdr.index(name)
    out = []
    for r in rows:
        try:
            out.append(float(r[j]))
        except (ValueError, IndexError):
            out.append(np.nan)
    return np.array(out)


mode = None
if "_r" in hdr:
    r = gc("_r")
    mode = "VizieR _r column (arcmin from cone centre)"
else:
    racol = [c for c in hdr if c.startswith("_RA") or c in ("RAJ2000", "RA_ICRS")][0]
    decol = [c for c in hdr if c.startswith("_DE") or c in ("DEJ2000", "DE_ICRS")][0]
    r = np.hypot((gc(racol) - RA0) * np.cos(np.deg2rad(DE0)), gc(decol) - DE0) * 60.0
    mode = "recomputed from %s / %s" % (racol, decol)

r = r[np.isfinite(r)]
edges = np.sqrt(np.linspace(0.0, RS_ARCMIN ** 2, 12))
n, _ = np.histogram(r, bins=edges)
area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
dens = n / area
med_outer = float(np.median(dens[1:]))
ratio = float(dens[-1] / med_outer) if med_outer else float("nan")

out = {
    "audit": "VizieR cone-search truncation check",
    "why": ("A coordinator warning reported that VizieR -c/-c.rs cone searches can return "
            "zero rows with no error, invalidating nulls derived from them. This lane contains "
            "exactly ONE cone-derived product and it is a POSITIVE result (2853 rows), not a "
            "null, so the reported failure mode does not apply to it directly. It is checked "
            "here anyway for silent radial truncation."),
    "file": os.path.relpath(P, LANE).replace("\\", "/"),
    "cone_centre_deg": [RA0, DE0],
    "cone_radius_arcmin": RS_ARCMIN,
    "n_rows": int(len(rows)),
    "radius_source": mode,
    "radial_extent_arcmin": [round(float(r.min()), 2), round(float(r.max()), 2)],
    "equal_area_annuli": [
        {"r_in_arcmin": round(float(a), 1), "r_out_arcmin": round(float(b), 1),
         "n": int(c), "density_per_arcmin2": round(float(d), 5)}
        for a, b, c, d in zip(edges[:-1], edges[1:], n, dens)],
    "outermost_over_median_density": round(ratio, 3),
    "verdict": ("NOT truncated: the outermost equal-area annulus holds %.2f times the median "
                "annular surface density, and rows reach %.1f of the %.0f arcmin requested "
                "radius. The cone returned the full requested area."
                % (ratio, r.max(), RS_ARCMIN)) if (0.5 < ratio < 2.0 and r.max() > 0.95 * RS_ARCMIN)
    else ("SUSPECT: outermost/median density ratio %.2f, max radius %.1f of %.0f arcmin requested. "
          "Re-derive by downloading the full table and matching numerically."
          % (ratio, r.max(), RS_ARCMIN)),
    "other_cone_searches_in_lane": "none - every other VizieR query used -source= only",
    "nulls_derived_from_cone_searches_in_this_lane": ("none. The weak-lensing availability audit, "
                                                      "the strong-lensing negative results and the "
                                                      "member-catalogue NOT_FOUND probes were all "
                                                      "established from directory listings, "
                                                      "-source= probes that return an explicit "
                                                      "'#INFO Error=Table or Catalog not found', "
                                                      "arXiv source-tarball table enumeration, or "
                                                      "GitHub tree listings. No null in this lane "
                                                      "rests on a cone search."),
}
dest = os.path.join(LANE, "cone_search_audit.json")
json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)

print("rows=%d  extent %.1f-%.1f arcmin  outer/median density=%.2f"
      % (len(rows), r.min(), r.max(), ratio))
for a, b, c, d in zip(edges[:-1], edges[1:], n, dens):
    print("  %6.1f-%6.1f  n=%4d  dens=%.4f" % (a, b, c, d))
print(out["verdict"])
