#!/usr/bin/env python3
"""Write manifests for the raw upstream downloads and build fresh_sample_index.json.

ACQUISITION ONLY.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


RAW_FILES = [
    ("arxiv_1803.00020.tar.gz", "https://arxiv.org/e-print/1803.00020",
     "2018ApJ...857...32B",
     "Babyk, McNamara, Nulsen et al. 2018 - X-ray Scaling Relations of "
     "Early-type Galaxies. ACQUIRED: Tables 1,2,3 extracted.",
     "curl -sL https://arxiv.org/e-print/1803.00020"),
    ("arxiv_1802.02589.tar.gz", "https://arxiv.org/e-print/1802.02589",
     "2018ApJ...862...39B",
     "Babyk et al. 2018 - A Universal Entropy Profile for the Hot Atmospheres "
     "of Galaxies and Clusters within R2500. NOT USED: contains no tabulated "
     "gas or total masses; its Table 1 is observation metadata only and its "
     "other tables are broken-power-law entropy fit coefficients. Retained as "
     "provenance for the premise check on ApJ 862, 39.",
     "curl -sL https://arxiv.org/e-print/1802.02589"),
    ("arxiv_1810.11465.tar.gz", "https://arxiv.org/e-print/1810.11465",
     "2019ApJ...887..149B",
     "Babyk et al. 2019 - Origins of Molecular Clouds in Early-type Galaxies "
     "(40 ETGs). NOT USED: the temperature/density/gas-mass/total-mass profiles "
     "are plotted only; no radial profile table exists in the LaTeX source. "
     "Its Table 1 and Table 3 are observation metadata plus kT, M_mol and "
     "1.4 GHz flux.",
     "curl -sL https://arxiv.org/e-print/1810.11465"),
    ("arxiv_astro-ph_0601301.tar.gz", "https://arxiv.org/e-print/astro-ph/0601301",
     "2006ApJ...646..899H",
     "Humphrey, Buote et al. 2006 - A Chandra View of Dark Matter in Early-Type "
     "Galaxies (7 systems). REJECTED UNDER RULE 2: the only tabulated total "
     "masses are M_vir and concentration c from a parametric 'NFW+stars' fit "
     "(table_results). The non-parametric mass profile data-points exist only "
     "as a figure. Retained as provenance for the rejection.",
     "curl -sL https://arxiv.org/e-print/astro-ph/0601301"),
    ("arxiv_0903.2540.tar.gz", "https://arxiv.org/e-print/0903.2540",
     "2009A&A...501..157N",
     "Nagino & Matsushita 2009 - Gravitational potential and X-ray "
     "luminosities of early-type galaxies observed with XMM-Newton and "
     "Chandra. ACQUIRED: sample table + M/L table extracted.",
     "curl -sL https://arxiv.org/e-print/0903.2540"),
]

for fn, url, bib, note, query in RAW_FILES:
    p = os.path.join(RAW, fn)
    if not os.path.exists(p):
        print("MISSING", fn)
        continue
    mf = {
        "file": os.path.join("raw", fn),
        "source_url": url,
        "bibcode": bib,
        "retrieved_utc": NOW,
        "sha256": sha256(p),
        "bytes": os.path.getsize(p),
        "row_count": None,
        "columns_with_units": None,
        "content_type": "application/gzip (arXiv e-print LaTeX source tarball, "
                        "unmodified upstream response)",
        "query_issued": query,
        "notes": note,
    }
    with open(p + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=2)
    print("manifest:", fn)


def load_cols(tsv):
    with open(os.path.join(BASE, tsv), encoding="utf-8") as f:
        return f.readline().rstrip("\n").split("\t")


def nrows(tsv):
    with open(os.path.join(BASE, tsv), encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


index = {
    "acquisition": "sealed-holdout acquisition, lead01-ablation, wellnet-2026-09",
    "built_utc": NOW,
    "scope": ("X-ray-bright early-type galaxies with hydrostatic X-ray mass "
              "determinations. ACQUISITION ONLY - no analysis, no residuals, "
              "no acceleration ratios, no model comparison performed."),
    "sealed_holdouts_untouched": ["KiDS", "wide-binary data"],
    "sources_acquired": [
        {
            "bibcode": "2018ApJ...857...32B",
            "arxiv_id": "1803.00020",
            "reference": ("Babyk I.V., McNamara B.R., Nulsen P.E.J., Hogan M.T., "
                          "Vantyghem A.N., Russell H.R., Pulido F.A., Edge A.C. "
                          "2018, ApJ 857, 32, 'X-ray Scaling Relations of "
                          "Early-type Galaxies'"),
            "instrument": "Chandra ACIS (archival)",
            "n_objects": 94,
            "object_class": ("early-type galaxies: E, S0/SA0/SAB0, plus BCGs and "
                             "cD galaxies; morphological type flagged per object"),
            "radius_definition": ("single radius per object: r = 5 r_e, five "
                                  "optical half-light radii; the value in kpc is "
                                  "tabulated per object as r_5re with its error"),
            "total_mass_is_hydrostatic": True,
            "total_mass_method": ("isothermal beta-model fit to the Chandra X-ray "
                                  "surface brightness combined with the "
                                  "single-temperature spectral fit, inserted into "
                                  "the hydrostatic-equilibrium equation. NOT an "
                                  "NFW fit, NOT a lensing or light-tied "
                                  "parametric model."),
            "provides": ["M_gas(<5re)", "M_tot,hydrostatic(<5re)", "kT",
                         "L_X(0.5-2 keV)", "metallicity Z", "sigma_c",
                         "redshift", "D_A", "D_L", "morphological type",
                         "beta-model beta, r_c, rho_0"],
            "does_not_provide": ["stellar mass", "K-band luminosity",
                                 "resolved radial profiles (single radius only)"],
            "files": {
                "babyk2018_table1_sample.tsv": {
                    "rows": nrows("babyk2018_table1_sample.tsv"),
                    "columns_with_units": load_cols("babyk2018_table1_sample.tsv")},
                "babyk2018_table2_spectra_5re.tsv": {
                    "rows": nrows("babyk2018_table2_spectra_5re.tsv"),
                    "columns_with_units": load_cols("babyk2018_table2_spectra_5re.tsv")},
                "babyk2018_table3_betamodel_masses.tsv": {
                    "rows": nrows("babyk2018_table3_betamodel_masses.tsv"),
                    "columns_with_units": load_cols("babyk2018_table3_betamodel_masses.tsv")},
                "babyk2018_joined_per_object.tsv": {
                    "rows": nrows("babyk2018_joined_per_object.tsv"),
                    "columns_with_units": load_cols("babyk2018_joined_per_object.tsv")},
            },
            "raw": "raw/arxiv_1803.00020.tar.gz",
            "caveats": [
                "Where the X-ray surface-brightness profile does not reach 5 r_e "
                "the authors extrapolated the mass profiles to 5 r_e using the "
                "slope of the last 20 points in log-log space.",
                "M_tot carries two separate errors: statistical then systematic.",
                "Metallicity entries without an error were frozen at 0.5 Zsun.",
                "The Table 1 coordinate columns are headed '(J2000)' but the "
                "paper text states the coordinates were taken as galactic "
                "coordinates from SIMBAD; treat the RA/DEC columns as suspect "
                "and re-resolve positions from the object name if needed.",
                "L_X is the 0.5-2.0 keV band luminosity, not bolometric.",
            ],
        },
        {
            "bibcode": "2009A&A...501..157N",
            "arxiv_id": "0903.2540",
            "reference": ("Nagino R., Matsushita K. 2009, A&A 501, 157, "
                          "'Gravitational potential and X-ray luminosities of "
                          "early-type galaxies observed with XMM-Newton and "
                          "Chandra'"),
            "instrument": "XMM-Newton (MOS) + Chandra (archival)",
            "n_objects": 22,
            "object_class": "early-type galaxies, RC3 T-type -5 (E) and -2 (S0)",
            "radius_definition": ("three radii per object: r < 0.5 r_e, r < 3 r_e, "
                                  "r < 6 r_e. r_e is tabulated in arcmin; the kpc "
                                  "equivalents are reconstructed here from the "
                                  "adopted distance D."),
            "total_mass_is_hydrostatic": True,
            "total_mass_method": ("non-parametric hydrostatic: M(R) = "
                                  "-kT(R)R/(G mu m_p) (dln n/dln R + dln T/dln R) "
                                  "evaluated on the deprojected ISM temperature "
                                  "and density profiles. NOT an NFW fit; NFW "
                                  "appears in the introduction only as literature "
                                  "context."),
            "provides": ["M_tot/L_B at 3 radii", "M_tot/L_K at 3 radii",
                         "log10 L_B", "sigma", "r_e", "distance D",
                         "T-type", "N_H", "X-ray morphology class"],
            "does_not_provide": ["gas mass M_gas(<r) - NOT TABULATED ANYWHERE "
                                 "IN THIS PAPER", "redshift (distances only)",
                                 "tabulated L_K (used by the authors from 2MASS "
                                 "but not printed, so M/L_K cannot be inverted "
                                 "to a mass from this table alone)",
                                 "tabulated kT per object (temperature profiles "
                                 "are plotted, not tabulated)"],
            "files": {
                "nagino2009_etg_masstolight.tsv": {
                    "rows": nrows("nagino2009_etg_masstolight.tsv"),
                    "columns_with_units": load_cols("nagino2009_etg_masstolight.tsv")},
            },
            "raw": "raw/arxiv_0903.2540.tar.gz",
            "caveats": [
                "Columns labelled RECONSTRUCTED are computed here from the "
                "tabulated columns (unit conversion and M/L x L product); they "
                "are not values printed in the paper. The raw tabulated M/L and "
                "log L_B columns are retained alongside.",
                "6 r_e columns are empty for NGC1549, NGC4365, NGC4526 and "
                "NGC4552 (profile did not reach 6 r_e).",
                "For NGC1549, NGC4477 and NGC5322 the mass inside 0.5 r_e rests "
                "on XMM-Newton data only.",
            ],
        },
    ],
    "sources_examined_and_rejected": [
        {
            "bibcode": "2006ApJ...646..899H",
            "arxiv_id": "astro-ph/0601301",
            "reference": "Humphrey, Buote et al. 2006, ApJ 646, 899",
            "reason_rejected": ("RULE 2 VIOLATION: the only tabulated total "
                                "masses are M_vir, r_vir and concentration c from "
                                "a parametric 'NFW+stars' fit. The non-parametric "
                                "mass profile points appear only as a figure, not "
                                "as a table."),
            "raw": "raw/arxiv_astro-ph_0601301.tar.gz",
        },
        {
            "bibcode": "2018ApJ...862...39B",
            "arxiv_id": "1802.02589",
            "reference": ("Babyk et al. 2018, ApJ 862, 39, 'A Universal Entropy "
                          "Profile for the Hot Atmospheres of Galaxies and "
                          "Clusters within R2500'"),
            "reason_rejected": ("Contains no tabulated gas mass or total mass. "
                                "Table 1 is observation metadata for 40 low-mass "
                                "systems; the remaining tables are broken-power-law "
                                "entropy fit coefficients. M2500 is plotted but "
                                "never tabulated."),
            "raw": "raw/arxiv_1802.02589.tar.gz",
        },
        {
            "bibcode": "2019ApJ...887..149B",
            "arxiv_id": "1810.11465",
            "reference": ("Babyk et al. 2019, ApJ 887, 149, 'Origins of Molecular "
                          "Clouds in Early-type Galaxies'"),
            "reason_rejected": ("The 40-galaxy temperature, density, cooling-time, "
                                "gas-mass and total-mass profiles are PLOTTED "
                                "only; no radial profile table exists in the "
                                "LaTeX source. Tables give observation metadata, "
                                "kT, molecular gas mass and 1.4 GHz flux."),
            "raw": "raw/arxiv_1810.11465.tar.gz",
        },
    ],
    "vizier_probe_results": {
        "method": ("https://vizier.cfa.harvard.edu/viz-bin/ReadMe/<ID> with "
                   "curl -sL (the CfA mirror answers https with a 302 to http, "
                   "so redirects MUST be followed or every probe looks like a "
                   "failure)"),
        "J/ApJ/857/32": "ReadMe is not found - NO CATALOGUE",
        "J/ApJ/862/39": "ReadMe is not found - NO CATALOGUE",
        "J/ApJ/646/899": "ReadMe is not found - NO CATALOGUE",
        "J/A+A/501/157": "ReadMe is not found - NO CATALOGUE",
        "J/ApJ/669/158": "ReadMe is not found - NO CATALOGUE",
        "J/A+A/601/A95": ("PRESENT but is a DIFFERENT PAPER: Calabro+ 2017, "
                          "'Star-forming dwarfs at intermediate-z in VUDS'. It is "
                          "NOT O'Sullivan CLoGS."),
        "J/ApJS/174/117": ("PRESENT: Maughan+ 2008, 115 Chandra clusters. NOT "
                           "acquired - it is Priority 3, to be used only if "
                           "Priorities 1 and 2 both failed, and Priority 1 "
                           "succeeded."),
    },
}

out = os.path.join(BASE, "fresh_sample_index.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)
print("wrote", out, os.path.getsize(out), "bytes")
