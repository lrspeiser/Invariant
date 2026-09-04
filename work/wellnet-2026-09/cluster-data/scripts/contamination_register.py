"""Build the register of acquired products that are NOT raw observations.

Scans every manifest for explicit flags and for note text describing a
dependence on a dark-matter halo profile, hydrostatic equilibrium, a
mass-follows-light assumption, a scaling relation, or SED/population
modelling.  The result is the quarantine list: columns that may be used as
inputs or context but never scored against as observations.
"""
import glob
import json
import os
import re

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = [
    ("NFW / dark-matter halo profile assumed",
     r"\bNFW\b|Navarro[- ]Frenk[- ]White|gNFW|halo concentration|c200|c_?\{?200"),
    ("Newtonian hydrostatic equilibrium assumed",
     r"hydrostatic|\bHSE\b|Mgrav|hydro_mass|M_?HSE"),
    ("scaling relation, not a direct measurement",
     r"scaling[- ]relation|L\s*-\s*M relation|M\s*-\s*T relation|calibrated on"),
    ("mass model needed to define the aperture (R500 / R200 / M500)",
     r"R500|R_?200|M500|M_?200|r500|virial"),
    ("stellar mass from SED / population synthesis fitting",
     r"SED fit|FAST|Bruzual|Chabrier|population synthesis|stellar[- ]population|BC03"),
    ("parametric lens model output, or redshift optimised by a lens model",
     r"lens model|model[- ]optimis|model[- ]optimiz|LENSTOOL|dPIE|lensmodel"),
    ("placeholder / dummy column, not a measurement",
     r"placeholder|dummy|sentinel|hard-coded"),
    ("projected, not deprojected (or vice versa) - geometry assumption",
     r"deprojec|projected spectroscopic|spherical symmetry"),
]

FLAG_KEYS = ["derived_assumes_newtonian_hse", "contains_hse_mass_column",
             "presupposes_dark_matter", "redshift_column_is_placeholder",
             "contains_model_product", "is_raw_observable"]

entries = []
for m in sorted(glob.glob(os.path.join(LANE, "**", "*.manifest.json"), recursive=True)):
    d = json.load(open(m, encoding="utf-8"))
    rel = os.path.relpath(m, LANE).replace("\\", "/")[: -len(".manifest.json")]
    note = (d.get("note") or "") + " " + (d.get("extraction") or "")
    cols = " ".join(str(c.get("name") or "") + " " + str(c.get("unit") or "")
                    for c in (d.get("columns") or []) if isinstance(c, dict))
    hay = note + " " + cols
    hits = [label for label, pat in PATTERNS if re.search(pat, hay, re.I)]
    flags = {k: d[k] for k in FLAG_KEYS if k in d}
    # is_raw_observable True is a clean bill of health, not a contamination
    explicit = [k for k, v in flags.items()
                if v is True and k != "is_raw_observable"]
    if hits or explicit:
        entries.append({
            "file": rel,
            "explicit_manifest_flags": flags,
            "textual_indicators": hits,
            "declared_raw_observable": d.get("is_raw_observable"),
            "note_excerpt": (d.get("note") or "")[:400],
        })

by_kind = {}
for e in entries:
    for h in e["textual_indicators"]:
        by_kind.setdefault(h, []).append(e["file"])

out = {
    "purpose": ("Quarantine register. Every acquired file whose manifest text or explicit flags "
                "indicate a dependence on a dark-matter halo profile, Newtonian hydrostatic "
                "equilibrium, a mass-follows-light construction, a scaling relation, or "
                "SED/population-synthesis modelling. Files listed here are NOT automatically "
                "unusable -- most are usable as INPUTS (measured baryonic quantities) or as "
                "context. What they must never be is the thing a gravity law is scored against."),
    "how_to_read": ("A hit is a keyword match on the manifest note plus column names, so it is "
                    "deliberately over-inclusive: it flags files that merely DISCUSS a caveat as "
                    "well as files that embody one. Read the note excerpt before acting. The "
                    "explicit_manifest_flags are authoritative; the textual_indicators are a net."),
    "n_files_flagged": len(entries),
    "by_kind": {k: sorted(set(v)) for k, v in sorted(by_kind.items())},
    "entries": entries,
}
dest = os.path.join(LANE, "contamination_register.json")
json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)

print("flagged %d files" % len(entries))
for k, v in sorted(by_kind.items()):
    print("  %-62s %d" % (k, len(set(v))))
