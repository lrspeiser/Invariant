"""Transcribe published LaTeX/CSV tables verbatim into cleaned TSVs + manifests.

No unit conversion, no derivation, no cross-table joins. Row counts are asserted
against the sample size stated in each paper.
"""
import os
import re

from _acquire import HERE, write_manifest, utcnow, sha256_bytes


def clean(s):
    """Strip LaTeX markup from a single cell, keeping the numeric content."""
    s = s.strip()
    s = s.replace(r"\\", "")
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\hspace\{[^}]*\}", "", s)
    s = re.sub(r"\\tablenotemark\{[^}]*\}", "", s)
    s = re.sub(r"\\(dots|ldots|nodata)\b", "...", s)
    s = re.sub(r"\\pm\b", " +- ", s)           # keep the +- so errors stay unambiguous
    s = re.sub(r"\\leq\b", "<=", s)            # upper limits must survive
    s = re.sub(r"\\geq\b", ">=", s)
    s = re.sub(r"(\d)\\farcm\s*(\d)", r"\1.\2", s)   # 4\farcm 8 -> 4.8 arcmin
    s = re.sub(r"(\d)\\farcs\s*(\d)", r"\1.\2", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)          # remaining control sequences
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("~", " ").replace("^\\circ", "").replace("\\", "")
    s = re.sub(r"\s*\^\s*$", "", s)            # leftover degree marker
    s = re.sub(r"\s+", " ", s).strip()
    if set(s) <= {"-", " "} and len(s) > 2:
        s = ""                                  # "---------" = no measurement
    return s


def emit(outname, columns, rows, *, expect_rows, source_url, source_file,
         extraction, note, exact_query=None, extra=None):
    assert len(rows) == expect_rows, (
        f"{outname}: transcribed {len(rows)} rows, paper states {expect_rows}")
    ncol = len(columns)
    for i, r in enumerate(rows):
        assert len(r) == ncol, f"{outname}: row {i} has {len(r)} cells, expected {ncol}: {r}"
    body = "\t".join(c["name"] for c in columns) + "\n"
    body += "\n".join("\t".join(r) for r in rows) + "\n"
    path = os.path.join(HERE, outname)
    data = body.encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    m = dict(file=outname, source_url=source_url,
             source_file_within_archive=source_file,
             exact_query=exact_query or source_url,
             retrieved_utc=utcnow(), sha256=sha256_bytes(data), bytes=len(data),
             row_count=len(rows), column_count=ncol, columns=columns,
             extraction=extraction, note=note)
    if extra:
        m.update(extra)
    write_manifest(path, **m)
    print(f"  {outname}: {len(rows)} rows x {ncol} cols")


def ffill(rows, col=0):
    """Forward-fill a leading label column across continuation rows."""
    for k in range(1, len(rows)):
        if not rows[k][col]:
            rows[k][col] = rows[k - 1][col]
    return rows


def _expand_multicolumn(line):
    """Replace \multicolumn{N}{..}{X} with X followed by N-1 empty cells."""
    def rep(m):
        n = int(m.group(1))
        return m.group(2) + " &" * (n - 1)
    return re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{([^{}]*)\}", rep, line)


def latex_rows(tex_path, start_marker, end_marker, start_from=0, include_marker=True):
    """Return the raw '&'-separated body lines between two markers.

    include_marker=True means start_marker itself is the beginning of the first
    data row and is kept.
    """
    txt = open(tex_path, encoding="utf-8", errors="replace").read()
    i = txt.index(start_marker, start_from)
    j = txt.index(end_marker, i)
    block = txt[i:j] if include_marker else txt[i + len(start_marker):j]
    out = []
    for line in block.split("\\\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        line = re.sub(r"\\hline|\\startdata|\\enddata", "", line).strip()
        if not line or "&" not in line:
            continue
        line = _expand_multicolumn(line)
        out.append([clean(c) for c in line.split("&")])
    return out


# ===========================================================================
# 1. Bovy & Rix 2013 Table 3 - Sigma_1.1(R) and K_Z,1.1(R) from vertical kinematics
# ===========================================================================
def bovyrix():
    d = os.path.join(HERE, "arxiv_bovyrix2013_disk_surface_density")
    cols = [
        {"name": "FeH", "unit": "dex", "desc": "central [Fe/H] of the mono-abundance population (MAP)"},
        {"name": "aFe", "unit": "dex", "desc": "central [alpha/Fe] of the MAP"},
        {"name": "R", "unit": "kpc", "desc": "Galactocentric radius at which this MAP best measures the surface density"},
        {"name": "Sigma_1.1", "unit": "Msun/pc2", "desc": "surface density integrated to |Z| = 1.1 kpc"},
        {"name": "e_Sigma_1.1", "unit": "Msun/pc2", "desc": "uncertainty on Sigma_1.1"},
        {"name": "R0_minus_R", "unit": "kpc", "desc": "R_0 - R; hold fixed when rescaling to a different R_0"},
        {"name": "K_Z_1.1", "unit": "2*pi*G*Msun/pc2", "desc": "vertical force at |Z| = 1.1 kpc"},
        {"name": "e_K_Z_1.1", "unit": "2*pi*G*Msun/pc2", "desc": "uncertainty on K_Z_1.1"},
    ]
    rows = [[c.strip() for c in re.sub(r"\\\\", "", l).split("&")]
            for l in open(os.path.join(d, "surf.txt"), encoding="utf-8")
            if "&" in l]
    emit("bovyrix2013_table3_Sigma_Kz_vs_R.tsv", cols, rows, expect_rows=43,
         source_url="https://arxiv.org/e-print/1309.0809",
         source_file="surf.txt (\\input into Table 3 of ms.tex)",
         extraction="Verbatim transcription of the published LaTeX table body. No unit conversion, no derivation, no cross-table join.",
         note=("Bovy & Rix 2013 (ApJ 779, 115), Table 3. Each row is ONE mono-abundance "
               "population (MAP) of SEGUE G-type dwarfs; the quoted R is the radius at which "
               "that MAP best constrains the surface density, so the 43 rows together form a "
               "RESOLVED Sigma_1.1(R) and K_Z,1.1(R) profile of the Milky Way disc over "
               "roughly 4-9 kpc, measured from VERTICAL stellar kinematics. "
               "MODEL DEPENDENCE - this is NOT a raw observable: Sigma_1.1 and K_Z,1.1 are "
               "inferred by fitting a quasi-isothermal distribution function to the vertical "
               "and radial motions inside a parametrised Newtonian potential family "
               "(disk + bulge + power-law halo + gas), with V_c(R_0)=230 km/s, z_h=400 pc and "
               "dlnV_c/dlnR=0 held fixed. Using it to test a modified gravity law is therefore "
               "circular unless the underlying SEGUE kinematics are refitted."),
         extra={"paper_version_note": (
             "The arXiv v3 tarball also contains anc/table3.csv, an ancillary machine-readable "
             "copy. 40 of 43 rows agree exactly; 3 rows differ by up to 4.7% in Sigma_1.1 "
             "(rows with [Fe/H] = -1.25, -1.15 and one other). surf.txt is the file actually "
             "\\input into the v3 manuscript and is used here; anc/table3.csv is kept "
             "unmodified inside the tarball directory.")})


def bovyrix_anc():
    d = os.path.join(HERE, "arxiv_bovyrix2013_disk_surface_density", "anc")
    cols = [
        {"name": "FeH", "unit": "dex"}, {"name": "aFe", "unit": "dex"},
        {"name": "R", "unit": "kpc"}, {"name": "Sigma_1.1", "unit": "Msun/pc2"},
        {"name": "e_Sigma_1.1", "unit": "Msun/pc2"}, {"name": "R0_minus_R", "unit": "kpc"},
        {"name": "K_Z_1.1", "unit": "2*pi*G*Msun/pc2"}, {"name": "e_K_Z_1.1", "unit": "2*pi*G*Msun/pc2"},
    ]
    rows = [l.strip().split(",") for l in open(os.path.join(d, "table3.csv"), encoding="utf-8") if l.strip()]
    emit("bovyrix2013_anc_table3_Sigma_Kz_vs_R.tsv", cols, rows, expect_rows=43,
         source_url="https://arxiv.org/e-print/1309.0809",
         source_file="anc/table3.csv (arXiv ancillary file)",
         extraction="Verbatim copy of the authors' ancillary CSV with a header row added. No unit conversion.",
         note=("Bovy & Rix 2013 ancillary machine-readable copy of Table 3. Differs from the "
               "typeset table (surf.txt) in 3 of 43 rows by up to 4.7% in Sigma_1.1. Kept as a "
               "cross-check; prefer bovyrix2013_table3_Sigma_Kz_vs_R.tsv."))


# ===========================================================================
# 2. Garcia-Ruiz, Sancisi & Kuijken 2002 - HI analysis results (26 edge-ons)
# ===========================================================================
def garciaruiz():
    tex = os.path.join(HERE, "arxiv_garciaruiz2002_edgeon_warps", "artdata12.tex")
    raw = latex_rows(tex, "(1) & \\multicolumn{1}{c}{(2)}", "\\caption{Results from the HI analysis")
    # first body line is the (1)(2)(3)... numbering row remnant; drop non-numeric-leading rows
    rows = [r for r in raw if re.match(r"^\d+$", r[0])]
    # the warp columns are printed as "value +-" & "error" pairs -> 14 cells
    cols = [
        {"name": "UGC", "unit": "UGC number"},
        {"name": "Lop_kin", "unit": "percent", "desc": "kinematical lopsidedness"},
        {"name": "Lop_rho", "unit": "percent", "desc": "density lopsidedness"},
        {"name": "R_HI", "unit": "arcmin", "desc": "HI radius at 1 Msun/pc2"},
        {"name": "M_HI", "unit": "1e8 Msun"},
        {"name": "V_sys", "unit": "km/s"},
        {"name": "W20", "unit": "km/s"},
        {"name": "W50", "unit": "km/s"},
        {"name": "PA", "unit": "deg", "desc": "position angle of the major axis (single global value)"},
        {"name": "warp1", "unit": "deg", "desc": "warp angle on the East side"},
        {"name": "e_warp1", "unit": "deg"},
        {"name": "warp2", "unit": "deg", "desc": "warp angle on the West side"},
        {"name": "e_warp2", "unit": "deg"},
        {"name": "Env", "unit": "class", "desc": "0 = no companion within 100', 1 = companion 50-100', 2 = companion within 50'"},
    ]
    emit("garciaruiz2002_table_hi_analysis_warp_angles.tsv", cols, rows, expect_rows=26,
         source_url="https://arxiv.org/e-print/astro-ph/0207112",
         source_file="artdata12.tex, table 'Results from the HI analysis'",
         extraction="Verbatim transcription of the published LaTeX table. Empty cells were '---------' in the source (no warp measurable on that side).",
         note=("Garcia-Ruiz, Sancisi & Kuijken 2002 (A&A 394, 769), WHISP HI of 26 edge-on spirals. "
               "IMPORTANT PREMISE CORRECTION: this paper does NOT tabulate a tilted-ring "
               "inclination(R) or PA(R) solution. It gives ONE global PA per galaxy and ONE "
               "warp ANGLE per side, measured from the centroid of Gaussian fits perpendicular "
               "to the major axis. The full warp curve z(R), the rotation curve V(R) and the HI "
               "surface-density profile Sigma_HI(R) exist for every galaxy but are published "
               "ONLY as figures (the atlas files u<UGC>-plotwl2.ps and u<UGC>-plotart30.ps in "
               "this same arXiv tarball, IDL vector PostScript)."))


# ===========================================================================
# 3. Herrmann & Ciardullo 2009, Paper III (PN kinematics of 5 face-on spirals)
# ===========================================================================
def herrmann():
    tex = os.path.join(HERE, "arxiv_herrmann2009_paperIII_kinematics", "PIII_astroph.tex")
    src = "https://arxiv.org/e-print/0910.0266"
    f = "PIII_astroph.tex"

    rows = latex_rows(tex, "\\startdata\nIC~342 &Scd", "\\enddata")
    cols = [{"name": "Galaxy", "unit": "---"}, {"name": "Type", "unit": "RC3 Hubble type"},
            {"name": "i", "unit": "deg", "desc": "adopted disk inclination"},
            {"name": "Distance", "unit": "Mpc"}, {"name": "h_R", "unit": "kpc", "desc": "photometric disk scale length"},
            {"name": "mu_0", "unit": "mag/arcsec2", "desc": "central disk surface brightness, inclination corrected"},
            {"name": "E_BV", "unit": "mag"}, {"name": "v_max", "unit": "km/s"},
            {"name": "N_PN_velocities", "unit": "count"}, {"name": "survey_region", "unit": "arcmin"}]
    emit("herrmann2009_III_table1_program_galaxies.tsv", cols, rows, expect_rows=5,
         source_url=src, source_file=f + ", Table 1 (tabBasic)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note="Herrmann & Ciardullo 2009 (ApJ 705, 1686) Table 1. Supplies the baryonic photometry (h_R, mu_0) needed for g_bar.")

    rows = latex_rows(tex, "\\startdata\nIC 342 &505", "\\enddata")
    cols = [{"name": "Galaxy", "unit": "---"},
            {"name": "hz_deGrijs98", "unit": "pc", "desc": "INFERRED from h_R/h_z of de Grijs 1998"},
            {"name": "hz_Kregel02", "unit": "pc", "desc": "INFERRED from h_R/h_z of Kregel+ 2002"},
            {"name": "hz_BM02", "unit": "pc", "desc": "INFERRED from h_R/h_z vs K-band mu_0 of Bizyaev & Mitronova 2002"},
            {"name": "hz_initial_range", "unit": "pc"},
            {"name": "hz_stability_limit", "unit": "pc", "desc": "Toomre stability lower limit"},
            {"name": "hz_rotcurve_limit", "unit": "pc"},
            {"name": "hz_final_range", "unit": "pc", "desc": "ADOPTED range"}]
    emit("herrmann2009_III_table2_scale_heights.tsv", cols, rows, expect_rows=5,
         source_url=src, source_file=f + ", Table 2 (tabhz)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("Herrmann & Ciardullo 2009 Table 2. EVERY scale height here is INFERRED from an "
               "h_R/h_z scaling relation plus stability arguments - none is measured. These "
               "galaxies are face-on, so h_z is not directly observable."))

    rows = latex_rows(tex, "\\startdata\nIC~342\t& 45", "\\enddata")
    if not rows:
        rows = latex_rows(tex, "Radial Range}\\\\\n\t\t\t\t\t\t\t& (in degrees)", "\\enddata")
    ffill(rows)
    cols = [{"name": "Galaxy", "unit": "---"},
            {"name": "PA", "unit": "deg"},
            {"name": "i", "unit": "deg"},
            {"name": "radial_range", "unit": "arcmin"}]
    emit("herrmann2009_III_table3_disk_geometry_vs_radius.tsv", cols, rows, expect_rows=10,
         source_url=src, source_file=f + ", Table 3 (tabParams)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("Herrmann & Ciardullo 2009 Table 3: 'Geometric Parameters for Disk Models'. This IS "
               "a piecewise warp geometry: M83 has five radial zones with PA running 226 -> 172 deg "
               "and i running 24 -> 46 deg (PA and i vary LINEARLY with R inside the last three "
               "zones), and M94 has two zones. IC 342, M74 and M101 use a single constant "
               "(PA, i) over the full range. Geometry is set by the HI velocity field, not the PNe."))

    rows = latex_rows(tex, "\\startdata\nIC 342       &negligible", "\\enddata")
    cols = [{"name": "Galaxy", "unit": "---"},
            {"name": "v_asd_estimate", "unit": "km/s"}, {"name": "v_max", "unit": "km/s"},
            {"name": "0.1_vmax", "unit": "km/s"}, {"name": "0.2_vmax", "unit": "km/s"},
            {"name": "v_asd_best", "unit": "km/s", "desc": "adopted asymmetric drift"}]
    emit("herrmann2009_III_table4_asymmetric_drift.tsv", cols, rows, expect_rows=5,
         source_url=src, source_file=f + ", Table 4 (tabASD)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note="Herrmann & Ciardullo 2009 Table 4. A SINGLE asymmetric-drift value per galaxy, assumed constant with radius.")

    rows = latex_rows(tex, "\\startdata\nIC 342  &$I$", "\\enddata")
    ffill(rows)
    cols = [{"name": "Galaxy", "unit": "---"}, {"name": "Filter", "unit": "---"},
            {"name": "mu_0", "unit": "mag/arcsec2"},
            {"name": "h_z", "unit": "pc", "desc": "ADOPTED (inferred) scale height"},
            {"name": "h_R", "unit": "kpc", "desc": "dynamical scale length of the sigma_z profile"},
            {"name": "sigma_z_0", "unit": "km/s", "desc": "CENTRAL vertical velocity dispersion of the fitted exponential"},
            {"name": "Sigma_0", "unit": "Msun/pc2", "desc": "central disk mass surface density"},
            {"name": "Upsilon_0", "unit": "Msun/Lsun", "desc": "central disk mass-to-light ratio"},
            {"name": "dlog_Upsilon_hz_dr", "unit": "dex/kpc"}]
    emit("herrmann2009_III_table5_disk_mass_models.tsv", cols, rows, expect_rows=7,
         source_url=src, source_file=f + ", Table 5 (massmodel)",
         extraction="Verbatim transcription of the published LaTeX table. M83 and M94 each occupy two rows (inner + outer component).",
         note=("Herrmann & Ciardullo 2009 Table 5. sigma_z_0 is a FITTED CENTRAL VALUE of an "
               "exponential in R (scale length h_R), not a resolved profile. The resolved "
               "sigma_LOS(R) and sigma_z(R) profiles - bins of 15-18 PNe each - exist but are "
               "published ONLY in Figures 4-8. The primary per-PN radial velocities that produce "
               "them ARE machine-readable, in VizieR J/ApJ/703/894/table4."))


# ===========================================================================
# 4. Levine, Blitz & Heiles 2006 - Milky Way HI warp mode fit
# ===========================================================================
def levine():
    tex = os.path.join(HERE, "arxiv_levine2006_mw_hi_vertical", "ms.tex")
    rows = latex_rows(tex, "\\startdata\n0&15&-66", "\\enddata")
    cols = [{"name": "m", "unit": "---", "desc": "azimuthal Fourier mode of the warp"},
            {"name": "R_k", "unit": "kpc", "desc": "reference radius of the polynomial"},
            {"name": "k0", "unit": "pc"}, {"name": "k1", "unit": "pc/kpc"},
            {"name": "k2", "unit": "pc/kpc2"}]
    emit("levine2006_table1_mw_warp_mode_fit.tsv", cols, rows, expect_rows=3,
         source_url="https://arxiv.org/e-print/astro-ph/0601697",
         source_file="ms.tex, Table 1 (tab:warpfit)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("Levine, Blitz & Heiles 2006 (ApJ 643, 881), Table 1: least-squares fit to the "
               "Milky Way HI warp, W_m(R) = k0 + k1 (R - R_k) + k2 (R - R_k)^2 for modes m=0,1,2, "
               "valid for R > R_k. This is the ONLY table in the paper. The maps that matter for "
               "a vertical-field test - Sigma(R,phi), the mean height h(R,phi), the half-thickness "
               "T_h(R,phi), and the mode phases phi_1(R), phi_2(R) - are published ONLY as "
               "contour figures. Underlying data: the LAB HI survey."))


# ===========================================================================
# 5. de Blok et al. 2008 (THINGS) - global tilted-ring geometry
# ===========================================================================
def deblok():
    tex = os.path.join(HERE, "arxiv_deblok2008_things_rotation_curves", "deblok_astroph.tex")
    rows = latex_rows(tex, "\\startdata\nNGC 925  & 02 27 16.5", "\\enddata")
    cols = [{"name": "Name", "unit": "---"}, {"name": "RAJ2000", "unit": "h m s"},
            {"name": "DEJ2000", "unit": "d m s"}, {"name": "D", "unit": "Mpc"},
            {"name": "dR", "unit": "arcsec", "desc": "ring sampling increment of the tilted-ring fit"},
            {"name": "V_sys", "unit": "km/s"},
            {"name": "mean_i", "unit": "deg", "desc": "RADIAL AVERAGE of the tilted-ring inclination"},
            {"name": "mean_PA", "unit": "deg", "desc": "RADIAL AVERAGE of the tilted-ring position angle"}]
    emit("deblok2008_things_table2_tiltedring_means.tsv", cols, rows, expect_rows=19,
         source_url="https://arxiv.org/e-print/0810.2100",
         source_file="deblok_astroph.tex, Table 2 (bigtable)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("de Blok et al. 2008 (AJ 136, 2648), THINGS high-resolution rotation curves. "
               "PREMISE CORRECTION: columns 7 and 8 are the RADIAL MEAN inclination and position "
               "angle only. The per-ring i(R) and PA(R) solutions, and the rotation curves "
               "themselves, are published as per-galaxy FIGURES (figs 3-56 of the same arXiv "
               "tarball), not as tables. Several THINGS galaxies (NGC 925, NGC 2403, NGC 2841, "
               "NGC 3198, NGC 5055, NGC 7331) are strongly warped in those figures."))


if __name__ == "__main__":
    for fn in (bovyrix, bovyrix_anc, garciaruiz, herrmann, levine, deblok):
        print("==", fn.__name__)
        try:
            fn()
        except Exception as e:
            print("   FAILED:", type(e).__name__, e)
