"""Bulk VizieR TSV acquisition for the counter-rotator / polar-disc lane.
Every fetch is verified with assert_vizier_tsv (VizieR returns HTTP 200 + HTML
for a nonexistent -source=)."""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest, http_get, assert_vizier_tsv

D = os.path.dirname(os.path.abspath(__file__))
BASE = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s"
        "&-out.all&-out.max=unlimited")

TARGETS = [
    # (short name, VizieR catalogue, human description, measurement/model note)
    ("sprc_moiseev2011",       "J/MNRAS/418/244",
     "SDSS-based Polar Ring Catalogue (Moiseev+ 2011)"),
    ("prg_co_combes2013",      "J/A+A/554/A11",
     "CO observations of polar ring galaxies (Combes+ 2013)"),
    ("prg_hi_huchtmeier1997",  "J/A+A/319/401",
     "HI survey of polar ring galaxies II (Huchtmeier 1997)"),
    ("prg_hi_vandriel2002",    "J/A+A/386/140",
     "HI survey of polar ring galaxies IV (van Driel+ 2002)"),
    ("manga_crd_bevacqua2022", "J/MNRAS/511/139",
     "Counter-rotating disc candidates in SDSS-IV MaNGA (Bevacqua+ 2022)"),
    ("manga_counterrot_gasymov2025", "J/ApJS/281/19",
     "Stellar counterrotation in galaxies using MaNGA (Gasymov+ 2025)"),
    ("kinangles_raimundo2023", "J/other/NatAs/7.463",
     "Galaxies stellar and gas kinematic angles (Raimundo+ 2023)"),
    ("manga_kincat_ristea2024","J/MNRAS/527/7438",
     "MaNGA kinematic catalogue (Ristea+ 2024)"),
    ("atlas3d_I_cappellari2011", "J/MNRAS/413/813",
     "ATLAS3D I parent sample (Cappellari+ 2011)"),
    ("atlas3d_III_emsellem2011", "J/MNRAS/414/888",
     "ATLAS3D III lambda_R slow/fast rotators (Emsellem+ 2011)"),
    ("atlas3d_XXIII_krajnovic2013", "J/MNRAS/433/2812",
     "ATLAS3D XXIII (Krajnovic+ 2013)"),
    ("califa_gaskin_garcialorenzo2015", "J/A+A/573/A59",
     "Gas kinematics in the CALIFA survey (Garcia-Lorenzo+ 2015)"),
    ("califa_starkin_falconbarroso2017", "J/A+A/597/A48",
     "Stellar kinematics in the CALIFA survey (Falcon-Barroso+ 2017)"),
    ("manga_atlas3d_etg_zhong2026", "J/A+A/707/A137",
     "MaNGA and ATLAS3D ETGs kinematic data (Zhong+ 2026)"),
    ("s0_morphokin_mendezabreu2018", "J/MNRAS/474/1307",
     "S0 galaxies morpho-kinematic properties (Mendez-Abreu+ 2018)"),
    ("manga_dynpop_VII_zhu2025", "J/ApJS/280/55",
     "MaNGA DynPop VII circular velocity curves (Zhu+ 2025)"),
    # Corsini / Pizzella long-slit gas+stars series (both components measured)
    ("cp_corsini1999",   "J/A+A/342/671", "Early-type spiral galaxies kinematics (Corsini+ 1999)"),
    ("cp_sarzi2000",     "J/A+A/360/439", "Stellar and ionized gas kinematics NGC4672 (Sarzi+ 2000)"),
    ("cp_vegabeltran2001","J/A+A/374/394", "Gas and stars kinematics in disc galaxies (Vega Beltran+ 2001)"),
    ("cp_corsini2002",   "J/A+A/382/488", "Stellar and ionized gas kinematics NGC2855 (Corsini+ 2002)"),
    ("cp_corsini2003",   "J/A+A/408/873", "Gas and stellar kinematics in spirals (Corsini+ 2003)"),
    ("cp_pizzella2004",  "J/A+A/424/447", "Kinematics in 17 nearby spiral galaxies (Pizzella+ 2004)"),
    ("califa_angmom_falconbarroso2019", "J/A+A/632/A59",
     "CALIFA galaxies stellar angular momentum (Falcon-Barroso+ 2019)"),
    ("califa_kinclass_kalinova2017", "J/MNRAS/469/2539",
     "New classification of CALIFA galaxies by circular velocity curve (Kalinova+ 2017)"),
]

results = []
for short, cat, desc in TARGETS:
    url = BASE % cat
    raw = os.path.join(D, "crot_%s.raw.tsv" % short)
    try:
        http_get(url, raw)
        cols, data = assert_vizier_tsv(raw, expect_catalog=cat, min_rows=1)
        ok = True
        err = None
    except Exception as e:
        ok = False; err = "%s: %s" % (type(e).__name__, e)
        print("  !! FAILED %s (%s): %s" % (short, cat, err))
        cols, data = [], []
    results.append({"short": short, "catalog": cat, "desc": desc, "ok": ok,
                    "err": err, "ncols": len(cols), "nrows": len(data),
                    "file": os.path.basename(raw) if ok else None,
                    "url": url})
    time.sleep(1.0)

with open(os.path.join(D, "crot_vizier_fetch_log.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print("\n================ SUMMARY ================")
for r in results:
    print("%-6s %-34s %-22s rows=%-6s cols=%-4s %s"
          % ("OK" if r["ok"] else "FAIL", r["short"], r["catalog"],
             r["nrows"], r["ncols"], r["err"] or ""))
