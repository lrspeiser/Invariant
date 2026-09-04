"""Transcribe the Aniyan et al. NGC 6946 resolved sigma_z(R) tables (LaTeX source).

Aniyan, Freeman, Arnaboldi, Gerhard, Coccato, Kuijken, Merrifield, Fabricius 2021,
MNRAS 500, 3579, "Resolving the disc-halo degeneracy II: NGC 6946" (arXiv:2010.03991).
"""
import os
import re

from _acquire import HERE
from _extract_tables import clean, emit

D = os.path.join(HERE, "arxiv_aniyan2021_discHalo_II_ngc6946")
TEX = os.path.join(D, "N6946.tex")
SRC = "https://arxiv.org/e-print/2010.03991"


def body(start, end):
    txt = open(TEX, encoding="utf-8", errors="replace").read()
    i = txt.index(start)
    j = txt.index(end, i)
    out = []
    for line in txt[i:j].split("\\\\"):
        line = re.sub(r"\\hline", "", line).strip()
        line = re.sub(r"%.*", "", line).strip()
        if not line or "&" not in line:
            continue
        out.append([clean(c) for c in line.split("&")])
    return out


def t3():
    rows = body("54          &", "\\hline\n\\end{tabular}")
    cols = [
        {"name": "R_mean", "unit": "arcsec", "desc": "mean radius of the VIRUS-W radial bin"},
        {"name": "sigma_z_cold", "unit": "km/s", "desc": "two-component fit, cold (young) disc"},
        {"name": "sigma_z_hot", "unit": "km/s", "desc": "two-component fit, hot (old) disc - this is the dynamical tracer"},
        {"name": "BIC_2comp", "unit": "---"},
        {"name": "chi2red_2comp", "unit": "---"},
        {"name": "sigma_z_1comp", "unit": "km/s", "desc": "single-Gaussian fit"},
        {"name": "BIC_1comp", "unit": "---"},
        {"name": "chi2red_1comp", "unit": "---"},
    ]
    emit("aniyan2021_ngc6946_table3_sigmaz_virusw.tsv", cols, rows, expect_rows=2,
         source_url=SRC, source_file="N6946.tex, Table 3 (label VW_N6946)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("Aniyan+ 2021 (MNRAS 500, 3579) NGC 6946. RESOLVED sigma_z at 2 radii from "
               "VIRUS-W integrated-light spectra, decomposed by pPXF into a kinematically cold "
               "and a kinematically hot stellar component. sigma_LOS -> sigma_z uses the "
               "epicyclic approximation with sigma_z/sigma_R = 0.6 +- 0.15 (Shapiro+ 2003) and "
               "i = 37 deg. Dispersions corrected for the HI dispersion contribution."))


def t4():
    rows = body("144 & $\\leq 12.1$", "\\hline\n\\end{tabular}\n\\caption[The $\\sigma_z$ values calculated from the PN.S")
    cols = [
        {"name": "R_mean", "unit": "arcsec", "desc": "mean radius of the PN.S radial bin"},
        {"name": "sigma_z_cold", "unit": "km/s", "desc": "90% confidence UPPER LIMIT (values prefixed <=)"},
        {"name": "sigma_z_hot", "unit": "km/s"},
        {"name": "BIC_2comp", "unit": "---"},
        {"name": "sigma_z_1comp", "unit": "km/s"},
        {"name": "BIC_1comp", "unit": "---"},
    ]
    emit("aniyan2021_ngc6946_table4_sigmaz_pns.tsv", cols, rows, expect_rows=3,
         source_url=SRC, source_file="N6946.tex, Table 4 (label PNS_N6946)",
         extraction="Verbatim transcription of the published LaTeX table. '<=' marks 90% confidence upper limits on the cold component.",
         note=("Aniyan+ 2021 NGC 6946. RESOLVED sigma_z at 3 outer radii from Planetary Nebula "
               "Spectrograph (PN.S) velocities, same two-component decomposition. In the "
               "outermost bin the one-component model is preferred by BIC."))


def t6():
    rows = body("1.6  & 65.3 &", "\\hline\n\\end{tabular}\n\\end{adjustbox}")
    cols = [
        {"name": "R", "unit": "kpc"},
        {"name": "sigma_z_hot", "unit": "km/s", "desc": "vertical velocity dispersion of the hot (old) disc"},
        {"name": "Sigma_T", "unit": "Msun/pc2", "desc": "TOTAL surface density from the vertical Jeans equation, sigma_z^2 = 2 pi G h_z Sigma_T"},
        {"name": "Sigma_C_gas", "unit": "Msun/pc2", "desc": "observed cold-gas surface density (THINGS HI + HERACLES CO)"},
        {"name": "LC_over_LD", "unit": "percent/percent", "desc": "luminosity ratio of the cold and hot layers"},
        {"name": "F_C", "unit": "---", "desc": "Sigma_C_star / Sigma_D"},
        {"name": "Sigma_D", "unit": "Msun/pc2", "desc": "stellar surface density of the hot layer"},
        {"name": "Sigma_C_star", "unit": "Msun/pc2", "desc": "stellar surface density of the cold layer"},
        {"name": "Upsilon_B", "unit": "Msun/Lsun"},
        {"name": "Upsilon_V", "unit": "Msun/Lsun"},
        {"name": "Upsilon_I", "unit": "Msun/Lsun"},
        {"name": "Upsilon_3.6um", "unit": "Msun/Lsun"},
    ]
    emit("aniyan2021_ngc6946_table6_sigmaz_and_surface_density.tsv", cols, rows, expect_rows=5,
         source_url=SRC, source_file="N6946.tex, Table 6 (label app_tab1_N6946; printed as Table 5 in the journal version)",
         extraction="Verbatim transcription of the published LaTeX table.",
         note=("Aniyan+ 2021 NGC 6946. THIS IS THE KEY ROW-SET: a genuinely RESOLVED sigma_z(R) "
               "profile at 5 radii (1.6, 2.9, 4.3, 7.2, 9.9 kpc) together with the total dynamical "
               "surface density Sigma_T(R) it implies and the separately measured gas surface "
               "density. Inner two radii from VIRUS-W integrated light, outer three from PN.S. "
               "SCALE HEIGHT IS INFERRED, NOT MEASURED: h_z = 376 +- 75 pc for the I band, taken "
               "from statistical h_R/h_z relations for edge-on galaxies applied to the measured "
               "scale length (NGC 6946 is at i = 37 deg, so h_z is not directly observable). "
               "Sigma_T therefore inherits the h_z uncertainty linearly."))


if __name__ == "__main__":
    for fn in (t3, t4, t6):
        print("==", fn.__name__)
        try:
            fn()
        except Exception as e:
            print("   FAILED:", type(e).__name__, e)
