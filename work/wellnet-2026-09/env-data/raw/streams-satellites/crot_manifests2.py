"""Manifests for the derived subset TSVs and the raw arXiv e-print tarballs."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))


def hdr_of(p):
    with open(p, encoding="utf-8") as fh:
        h = fh.readline().rstrip("\n").split("\t")
        n = sum(1 for _ in fh)
    return h, n


# ------------------------------------------- derived polar / CR subset TSVs ---
p = os.path.join(D, "crot_raimundo2023_POLAR_subset.tsv")
h, n = hdr_of(p)
write_manifest(p,
  source_url="https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/other/NatAs/7.463&-out.all&-out.max=unlimited",
  query=("DERIVED from crot_kinangles_raimundo2023.raw.tsv: rows with "
         "|DPA - 90 deg| <= 30 deg (i.e. 60 <= DPA <= 120), sorted by |DPA-90|. "
         "DPA is the catalogue's own measured stellar-gas kinematic angle difference."),
  columns=[{"name": c, "unit": ("deg" if c.startswith(("PA", "e_PA", "DPA", "e_DPA"))
            else ("deg" if c in ("RAJ2000", "DEJ2000") else
                  ("[Msun]" if c == "logMstar" else "")))} for c in h],
  row_count=n,
  measurement_or_model=("MEASUREMENT. Both PAstellar and PAgas are fitted directly to "
      "the observed SAMI DR3 stellar and ionised-gas velocity fields; DPA is their "
      "arithmetic difference. 3-sigma uncertainties are provided for all three. "
      "logMstar is a stellar-population MODEL quantity. NO DM halo, NO Jeans/JAM."),
  note=("*** HIGHEST-VALUE SUBSET FOR THIS PROGRAMME. *** These are galaxies where the "
        "ionised gas and the stars are BOTH kinematically measured and their angular "
        "momenta are ~90 deg apart: the gas traces the gravitational field in a plane "
        "roughly PERPENDICULAR to the plane the stars trace, in the same baryonic "
        "system. Note DPA is a PROJECTED (on-sky) angle difference, not a "
        "deprojected 3-D angle between the two angular-momentum vectors."),
  extra={"parent_catalog": "J/other/NatAs/7.463 (Raimundo+ 2023, NatAs 7, 463)",
         "parent_row_count": 1310, "selection": "abs(DPA - 90) <= 30 deg",
         "n_polar_60_120": 47, "n_polar_70_110": 31, "n_polar_80_100": 19,
         "projected_not_deprojected": True})

p = os.path.join(D, "crot_atlas3d_II_2sigma_KDC_CRC_subset.tsv")
h, n = hdr_of(p)
write_manifest(p, source_url="https://arxiv.org/e-print/1102.3801",
  query=("DERIVED from crot_atlas3d_II_krajnovic2011_kinclass.tsv: rows whose "
         "KinStruct contains '2s' (2-sigma / counter-rotating disc), 'KDC' "
         "(kinematically decoupled core) or 'CRC' (counter-rotating core)."),
  columns=[{"name": c, "unit": ("deg" if c in ("PAphot", "e_PAphot", "PAkin",
            "e_PAkin", "Psi") else ("km/s" if c == "k1max" else ""))} for c in h],
  row_count=n,
  measurement_or_model=("MEASUREMENT. Classification of the observed SAURON stellar "
      "velocity and dispersion maps by kinemetry. NO DM halo, NO Jeans/JAM model."),
  note=("These are TWO-STELLAR-COMPONENT systems: a 2-sigma galaxy has two "
        "co-spatial counter-rotating STELLAR discs (twin dispersion peaks); a KDC/CRC "
        "has a decoupled core. The two components are ~180 deg apart (anti-parallel "
        "angular momenta), NOT ~90 deg - so they probe the SAME plane in opposite "
        "senses, which is a weaker geometric constraint than a polar configuration. "
        "Counts: 11 2-sigma, 11 KDC, 8 CRC = 30 galaxies out of 260."),
  extra={"parent": "ATLAS3D II (Krajnovic+ 2011), 260 galaxies",
         "n_2sigma": 11, "n_KDC": 11, "n_CRC": 8, "n_total_subset": 30,
         "geometry": "anti-parallel (~180 deg), not polar (~90 deg)"})

# ------------------------------------------------- raw e-print tarballs ------
EPRINTS = [
 ("crot_atlas3d_II_krajnovic2011.eprint.tar.gz", "1102.3801",
  "Krajnovic et al. 2011, MNRAS 414, 2923 (ATLAS3D II)", "krajnovic_A3D_kinmis.tex"),
 ("crot_atlas3d_III_emsellem2011.eprint.tar.gz", "1102.4444",
  "Emsellem et al. 2011, MNRAS 414, 888 (ATLAS3D III)", "PaperIII_ATLAS3D_Final.tex"),
 ("crot_bryant2019_sami_misalign.eprint.tar.gz", "1811.09298",
  "Bryant et al. 2019, MNRAS 483, 458 (SAMI misalignments)",
  "BryantSAMI_MisalignmentPaper_finalsub.tex"),
 ("crot_jin2016_manga_misalign.eprint.tar.gz", "1611.00528",
  "Jin et al. 2016, MNRAS 463, 913 (MaNGA kinematically decoupled)", "mnras_jin.tex"),
 ("crot_bao2023_manga_counterrot_gas.eprint.tar.gz", "2305.13387",
  "Bao et al. 2023 (MaNGA gas-star misalignment / counter-rotating gas origin)",
  "aanda.tex"),
 ("crot_ristea2022_sami_misalign_drivers.eprint.tar.gz", "2210.01147",
  "Ristea et al. 2022, MNRAS 517, 2677 (SAMI misalignment drivers)",
  "MAIN_paper_mnras_template.tex"),
]
NOTE = {
 "1611.00528": ("CONTAINS NO DATA TABLE. The source has zero tabular/deluxetable/"
   "\\input{} table environments. The 66 kinematically misaligned MaNGA galaxies of "
   "this paper are NOT published as a machine-readable list here, and the paper has "
   "no VizieR catalogue. Per-galaxy IDs could NOT be obtained."),
 "2305.13387": ("CONTAINS NO DATA TABLE. Zero tabular environments; the source has "
   "only 3 figure captions. This is an interpretation paper on the origin of "
   "counter-rotating gas, not a catalogue paper."),
 "1811.09298": ("Contains two AGGREGATE statistics tables only (pulled in via "
   "\\input{} from separate archive members 'MisalignmentSummaryTable' and "
   "'MislignmentStatsByMophology'); NO per-galaxy misalignment catalogue."),
 "2210.01147": ("Contains aggregate fraction tables only (Table A1 transcribed); "
   "NO per-galaxy misalignment catalogue."),
 "1102.3801": ("Contains the full 260-row ATLAS3D kinematic classification table as "
   "a single deluxetable - transcribed to crot_atlas3d_II_krajnovic2011_kinclass.tsv."),
 "1102.4444": ("ATLAS3D III text source. The lambda_R table itself was taken from "
   "VizieR J/MNRAS/414/888 (260 rows) rather than from this LaTeX."),
}
for fn, aid, desc, mainfile in EPRINTS:
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        print("MISSING %s" % fn); continue
    write_manifest(p, source_url="https://arxiv.org/e-print/%s" % aid,
      query="HTTP GET https://arxiv.org/e-print/%s (arXiv e-print source tarball)" % aid,
      columns=[], row_count=None,
      measurement_or_model=("RAW SOURCE ARCHIVE - not a data product. Contains LaTeX "
          "and figures. See the transcribed TSVs for the extracted tables and their "
          "own measurement/model labels."),
      note="%s  %s" % (desc, NOTE.get(aid, "")),
      extra={"arxiv": aid, "paper": desc, "main_tex": mainfile,
             "extracted_to": fn.replace(".eprint.tar.gz", "_src"),
             "code_executed": False})

# ---------------------------------------------------- failure record --------
fail = os.path.join(D, "crot_manga_dynpop_VII_zhu2025.raw.tsv")
if os.path.exists(fail):
    write_manifest(fail,
      source_url="https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/280/55&-out.all&-out.max=unlimited",
      query=("HTTP GET the VizieR asu-tsv endpoint for J/ApJS/280/55 in FOUR forms "
             "(-out.all&-out.max=unlimited; -out.all&-out.max=999999; "
             "-source=J/ApJS/280/55/table1; -out.max=unlimited&-out=**). "
             "ALL FOUR returned HTTP 200 with the complete metadata/column header "
             "block and ZERO data rows."),
      columns=[], row_count=0,
      measurement_or_model=("MODEL (would have been). The Vc(Re)/Vc(amaj)/Vcmax/"
          "Vc(rmax) circular-velocity curves are Jeans Anisotropic Modelling (JAM) "
          "outputs - the table carries an explicit 'Qual = JAM model quality (-1 to 3)' "
          "column. Not an observation of the gravitational field."),
      note=("*** ACQUISITION FAILURE - KEPT AS THE RECORD. *** VizieR served header-only "
            "responses for this catalogue in every query form tried. NO data rows were "
            "obtained and NO substitute was used. Low cost to this programme: the "
            "catalogue's headline quantity is a JAM MODEL product, which the hard rule "
            "forbids treating as an observation anyway."),
      extra={"vizier_catalog": "J/ApJS/280/55",
             "vizier_title": "MaNGA DynPop. VII. Circular velocity curves (Zhu+, 2025)",
             "status": "FAILED - header only, 0 data rows",
             "forms_tried": 4})
print("\ndone")
