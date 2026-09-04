"""Write manifests for the BUFFALO Abell 370 lensing DR1 files already downloaded."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import write_manifest, fits_table_info, LANE

D = os.path.join(LANE, "weaklensing", "buffalo_a370")
BASE = "https://archive.stsci.edu/hlsps/buffalo/abell370/catalogs/niemiec-lensing-dr1/"


def ascii_rows(p, comment="#"):
    rows = [l for l in open(p, encoding="utf-8", errors="replace")
            if l.strip() and not l.lstrip().startswith(comment)]
    ncol = len(rows[0].split()) if rows else 0
    return len(rows), ncol


# ---- 1. HST-only weak lensing source catalogue (FITS) ----
p = os.path.join(D, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_wl.fits")
n, nc, cols = fits_table_info(p)
units = {
    "ID": "source id", "ra": "deg (J2000)", "dec": "deg (J2000)",
    "e1": "dimensionless ellipticity component",
    "e2": "dimensionless ellipticity component",
    "gamma1": "dimensionless shear component",
    "gamma2": "dimensionless shear component",
    "var_e1": "error on e1", "var_e2": "error on e2",
    "a": "semi-major axis (arbitrary units)", "b": "semi-minor axis (arbitrary units)",
    "theta": "deg, major-axis position angle, anticlockwise from West",
    "MAG_AUTO_f814w": "AB mag", "MAGERR_AUTO_f814w": "AB mag",
    "MAG_AUTO_f606w": "AB mag", "MAGERR_AUTO_f606w": "AB mag",
    "MAG_AUTO_f160w": "AB mag", "MAGERR_AUTO_f160w": "AB mag",
    "SN": "shape-measurement signal-to-noise",
}
for c in cols:
    c["unit"] = units.get(c["name"], "")
write_manifest(
    p, BASE + os.path.basename(p),
    note=("Abell 370. BUFFALO lensing DR1, HST-only WEAK-LENSING SOURCE CATALOGUE "
          "(Niemiec et al. 2023, arXiv:2307.03778). RAW PER-SOURCE OBSERVABLE: sky position "
          "plus measured ellipticity components e1, e2 with errors, from pyRRG shape measurement "
          "on ACS/WFC F814W in the BUFFALO footprint. gamma1, gamma2 are e1, e2 divided by the "
          "shear polarisability (a calibration factor derived from image simulations) -- still a "
          "per-source measurement, not a mass model. NO PER-SOURCE REDSHIFT in this file: "
          "background selection is by colour-magnitude and the source redshift distribution is "
          "treated statistically. This is the ONLY cluster in the BUFFALO HLSP with a released "
          "lensing catalogue; the other five HFF clusters carry photometric catalogues only."),
    extraction="Verbatim upstream FITS binary table, byte-for-byte unmodified. HDU 1 named 'Joined'.",
    row_count=n, column_count=nc, columns=cols,
    extra={"cluster": "Abell 370", "product": "weak_lensing_raw_shear_catalogue",
           "is_raw_observable": True, "presupposes_dark_matter": False,
           "per_source_redshift": False,
           "instrument": "HST ACS/WFC F814W (BUFFALO, HST PID 15117)",
           "shape_method": "pyRRG (RRG moments, PSF-corrected)",
           "hlsp_doi": "10.17909/t9-w6tj-wp63"})
print("wl.fits", n, nc)

# ---- 2. HST + Subaru combined weak lensing catalogue (LENSTOOL ASCII) ----
p = os.path.join(D, "hlsp_buffalo_hst-subaru_multi_abell370_multi_v1.0_lenstool.cat")
n, nc = ascii_rows(p)
cols = [{"name": x, "unit": u} for x, u in [
    ("ID", "source id"), ("RA", "deg (J2000)"), ("Dec", "deg (J2000)"),
    ("a", "semi-major axis; e = sqrt((a^2-b^2)/(a^2+b^2))"),
    ("b", "semi-minor axis; e = sqrt((a^2-b^2)/(a^2+b^2))"),
    ("theta", "deg, major-axis position angle, anticlockwise from West"),
    ("z", "source redshift where available (phot or spec); 0.0 = not available"),
    ("MAG_AUTO_f814W", "AB mag"),
    ("var_e1", "error on e1"), ("var_e2", "error on e2")]]
write_manifest(
    p, BASE + os.path.basename(p),
    note=("Abell 370. BUFFALO lensing DR1 COMBINED weak-lensing constraint catalogue: HST/BUFFALO "
          "shapes plus ground-based Subaru/Suprime-Cam shapes from Umetsu et al. 2022. LENSTOOL "
          "input format. 18556 sources spanning RA 39.765-40.197, Dec -1.840 to -1.330 "
          "(about 0.43 x 0.51 deg). Ellipticity is carried as the (a, b, theta) ellipse, which is "
          "equivalent information to (e1, e2). ONLY 877 of 18556 sources (4.7%) carry a redshift; "
          "the remaining 17679 have z = 0.0, which is the LENSTOOL sentinel for 'unknown, use the "
          "global source redshift distribution' -- it is NOT a measured redshift of zero. "
          "RAW PER-SOURCE OBSERVABLE. This file is a lens-model INPUT, not an output; it contains "
          "no mass information."),
    extraction=("Verbatim upstream ASCII, byte-for-byte unmodified. Whitespace-delimited with a "
                "single '#REFERENCE 0' header line."),
    row_count=n, column_count=nc, columns=cols,
    extra={"cluster": "Abell 370", "product": "weak_lensing_raw_shear_catalogue",
           "is_raw_observable": True, "presupposes_dark_matter": False,
           "per_source_redshift": "partial: 877 of 18556",
           "instrument": "HST ACS/WFC F814W (BUFFALO) + Subaru/Suprime-Cam (Umetsu+2022)",
           "n_sources_with_redshift": 877, "n_sources_without_redshift": 17679})
print("lenstool.cat", n, nc)

# ---- 3. README ----
p = os.path.join(D, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_readme.txt")
write_manifest(
    p, BASE + os.path.basename(p),
    note=("Abell 370. Upstream README defining every column of the BUFFALO niemiec-lensing-dr1 "
          "catalogues. This is the authoritative column and unit source for the manifests in "
          "this directory."),
    extraction="Verbatim upstream text.",
    row_count=None, column_count=None, columns=None,
    extra={"cluster": "Abell 370", "product": "documentation"})

# ---- 4/5. Strong lensing multiple-image catalogues ----
for fn, desc in [
    ("hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-final.dat",
     "full strong-lensing constraint catalogue, all quality classes (Gold/Silver/Bronze/Platinum)"),
    ("hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
     "gold-class-only strong-lensing constraint catalogue in LENSTOOL format "
     "(columns a, b, theta, mag are dummy placeholders)")]:
    p = os.path.join(D, fn)
    n, nc = ascii_rows(p)
    if "gold" in fn:
        cols = [{"name": x, "unit": u} for x, u in [
            ("ID", "image id, SYSTEM.IMAGE"), ("RA", "deg (J2000)"), ("Dec", "deg (J2000)"),
            ("a", "DUMMY constant"), ("b", "DUMMY constant"), ("theta", "DUMMY constant"),
            ("z", "spectroscopic redshift of the source"), ("mag", "DUMMY constant")]]
    else:
        cols = [{"name": x, "unit": u} for x, u in [
            ("ID", "image id, SYSTEM.IMAGE"),
            ("RA_GAIA", "deg (J2000), Gaia-aligned astrometry"),
            ("DEC_GAIA", "deg (J2000), Gaia-aligned astrometry"),
            ("z_spec", "spectroscopic redshift"),
            ("cat", "quality class: Gold / Silver / Bronze / Platinum")]]
    write_manifest(
        p, BASE + fn,
        note=("Abell 370. BUFFALO lensing DR1 " + desc + ". RAW OBSERVABLE: multiple-image sky "
              "positions with SYSTEM.IMAGE identifiers and redshifts. Astrometry is Gaia-aligned. "
              "PLATINUM = MUSE detection with no clear HST counterpart. See the sibling README for "
              "per-system caveats (system 2 is split into 202 and 102; images 10.3, 42.3, 25.3 and "
              "system 40 and all PLATINUM systems need inflated positional errors; system 58 is "
              "unreliable; system 49 images have inconsistent colours). Contains no mass model."),
        extraction=("Verbatim upstream ASCII, byte-for-byte unmodified. Whitespace-delimited with "
                    "one header line."),
        row_count=n, column_count=nc, columns=cols,
        extra={"cluster": "Abell 370", "product": "strong_lensing_multiple_images",
               "is_raw_observable": True, "presupposes_dark_matter": False})
    print(fn, n, nc)

# ---- 6/7. Cluster-member galaxy catalogues ----
for fn, desc in [
    ("hlsp_buffalo_hst_multi_abell370_multi_v1.0_galcat-full.dat",
     "cluster-member galaxies actually used in the Niemiec+2023 lens models: red-sequence members "
     "plus 20 non-red-sequence galaxies, magF814W < 22.6"),
    ("hlsp_buffalo_hst_multi_abell370_f814w_v1.0_galcat-redseq.cat",
     "red-sequence-selected cluster-member catalogue (no magnitude cut applied)")]:
    p = os.path.join(D, fn)
    n, nc = ascii_rows(p)
    cols = [{"name": x, "unit": u} for x, u in [
        ("ID", "source id"), ("RA", "deg (J2000)"), ("Dec", "deg (J2000)"),
        ("a", "semi-major axis, arbitrary units (SExtractor)"),
        ("b", "semi-minor axis, arbitrary units (SExtractor)"),
        ("theta", "deg, position angle = -THETA_WORLD from SExtractor"),
        ("MAG_AUTO_f814W", "AB mag, SExtractor MAG_AUTO"),
        ("z", "SET TO 0.0 AS A LENSTOOL PLACEHOLDER -- not a measured redshift; the cluster "
              "redshift z = 0.375 is assumed")]]
    write_manifest(
        p, BASE + fn,
        note=("Abell 370. " + desc + ". MEASURED per galaxy: RA, Dec, F814W MAG_AUTO, and the "
              "SExtractor shape moments a, b, theta -- so axis ratio q = b/a and position angle "
              "ARE available. Effective radius R_e and Sersic index n are NOT: no Sersic fit is "
              "provided. TRAP: the z column is a hard-coded 0.0 LENSTOOL placeholder, NOT a "
              "measured redshift. Membership is RED-SEQUENCE PHOTOMETRIC selection, not "
              "spectroscopic."),
        extraction="Verbatim upstream ASCII, byte-for-byte unmodified. Whitespace-delimited.",
        row_count=n, column_count=nc, columns=cols,
        extra={"cluster": "Abell 370", "product": "member_galaxy_catalogue",
               "membership_criterion": "red sequence (photometric)",
               "redshift_column_is_placeholder": True,
               "has_Re": False, "has_sersic_n": False,
               "has_axis_ratio": True, "has_position_angle": True})
    print(fn, n, nc)

# ---- 8. DS9 region file ----
p = os.path.join(D, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl.reg")
write_manifest(
    p, BASE + os.path.basename(p),
    note=("Abell 370. DS9 region file marking the strong-lensing multiple images. Visualisation "
          "aid for the sl-final catalogue; carries no independent information."),
    extraction="Verbatim upstream text.",
    row_count=None, column_count=None, columns=None,
    extra={"cluster": "Abell 370", "product": "auxiliary_region_file"})
print("done")
