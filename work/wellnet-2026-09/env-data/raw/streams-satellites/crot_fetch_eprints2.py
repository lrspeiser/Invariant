"""Second round of arXiv e-prints, from IDs resolved by the arXiv API search
(crot_arxiv_search.json). Also CORRECTS a mislabel: arXiv:2305.13387 is
Zinchenko et al. 2023 (A&A 674, L7), NOT Bao."""
import sys, os, tarfile, gzip, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import http_get, write_manifest

D = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
 ("bb2014_califa_kinalign", "1405.5222",
  "Barrera-Ballesteros et al. 2014, A&A 568, A70 - Kinematic alignment of non-interacting CALIFA galaxies"),
 ("xu2022_manga_misaligned", "2202.04937",
  "Xu et al. 2022 - SDSS-IV MaNGA: spatially resolved properties of kinematically misaligned galaxies"),
 ("bao2022_counterrot_stellar_disks", "2202.03848",
  "Bao et al. 2022 - Different Formation Scenarios of Counter-rotating Stellar Disks in Nearby Galaxies"),
 ("moiseev2014_prg_kinematics", "1410.3607",
  "Moiseev et al. 2014 - Structure and kinematics of polar ring galaxies: new observations"),
 ("moiseev2012_inner_polar_rings", "1204.4437",
  "Moiseev 2012 - Inner Polar Rings and Disks: Observed Properties"),
 ("beom2022_manga_counterrot_edgeon", "2206.00682",
  "Beom et al. 2022 - SDSS-IV MaNGA: edge-on galaxies with a counter-rotating gaseous disk"),
 ("moiseev2011_sprc_paper", "1107.1966",
  "Moiseev et al. 2011, MNRAS 418, 244 - A new catalogue of polar-ring galaxies from the SDSS (SPRC)"),
]

log = []
for short, aid, desc in TARGETS:
    dest = os.path.join(D, "crot_%s.eprint.tar.gz" % short)
    try:
        http_get("https://arxiv.org/e-print/%s" % aid, dest)
    except Exception as e:
        print("  !! FAIL %s (%s): %s" % (short, aid, e))
        log.append({"short": short, "arxiv": aid, "ok": False, "error": str(e)})
        time.sleep(3); continue
    outdir = os.path.join(D, "crot_%s_src" % short)
    os.makedirs(outdir, exist_ok=True)
    members = []
    try:
        with tarfile.open(dest, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                nm = m.name.replace("\\", "/")
                if nm.startswith("/") or ".." in nm.split("/"):
                    continue
                tf.extract(m, outdir); members.append(nm)
    except tarfile.ReadError:
        try:
            data = gzip.decompress(open(dest, "rb").read())
            p = os.path.join(outdir, "%s.tex" % short)
            open(p, "wb").write(data); members = ["%s.tex" % short]
        except Exception as e:
            print("  !! not tar/gz: %s" % e)
    tex = [m for m in members if m.lower().endswith(".tex")]
    title = ""
    ntab = 0
    for t in tex:
        try:
            txt = open(os.path.join(outdir, t), encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        ntab += (len(re.findall(r"\\begin\{tabular", txt))
                 + len(re.findall(r"\\begin\{deluxetable", txt))
                 + len(re.findall(r"\\begin\{longtable", txt)))
        m = re.search(r"\\title(?:\[[^\]]*\])?\s*\{", txt)
        if m and not title:
            i = m.end(); depth = 1; buf = []
            while i < len(txt) and depth:
                ch = txt[i]
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: break
                buf.append(ch); i += 1
            title = " ".join("".join(buf).split())[:140]
    print("\n%-34s arXiv:%-11s tex=%d tabular/deluxetable envs=%d" % (short, aid, len(tex), ntab))
    print("   TITLE: %s" % title)
    log.append({"short": short, "arxiv": aid, "ok": True, "n_members": len(members),
                "tex_files": tex, "n_table_envs": ntab, "title_in_source": title,
                "description": desc})
    write_manifest(dest, source_url="https://arxiv.org/e-print/%s" % aid,
        query="HTTP GET https://arxiv.org/e-print/%s (arXiv e-print source tarball)" % aid,
        columns=[], row_count=None,
        measurement_or_model=("RAW SOURCE ARCHIVE - not a data product. See any "
            "transcribed TSV for its own measurement/model label."),
        note="%s | title verified in source: %r | %d tabular/deluxetable environments"
             % (desc, title, ntab),
        extra={"arxiv": aid, "paper": desc, "tex_files": tex,
               "n_table_envs": ntab, "code_executed": False})
    time.sleep(3)

with open(os.path.join(D, "crot_eprint_fetch_log2.json"), "w", encoding="utf-8") as fh:
    json.dump(log, fh, indent=2)

# ------------------- CORRECT the earlier Bao/Zinchenko mislabel --------------
mp = os.path.join(D, "crot_bao2023_manga_counterrot_gas.eprint.tar.gz.manifest.json")
if os.path.exists(mp):
    m = json.load(open(mp, encoding="utf-8"))
    m["paper"] = ("Zinchenko et al. 2023, A&A 674, L7 - 'Gas and stellar kinematic "
                  "misalignment in MaNGA galaxies: what is the origin of counter-rotating gas?'")
    m["note"] = ("CORRECTION: this file was initially named and labelled as 'Bao 2023'. "
                 "arXiv:2305.13387 is in fact ZINCHENKO et al. 2023 (A&A 674, L7); the "
                 "first author is I. A. Zinchenko, not Bao. The FILENAME still says "
                 "'bao2023' and is left unchanged so the SHA-256 and any existing "
                 "reference stay valid - trust this manifest, not the filename. "
                 "CONTAINS NO DATA TABLE: zero tabular environments; it is a Letter "
                 "interpreting the origin of counter-rotating gas, not a catalogue.")
    m["filename_is_misleading"] = True
    m["correct_first_author"] = "I. A. Zinchenko"
    json.dump(m, open(mp, "w", encoding="utf-8"), indent=2)
    print("\nCORRECTED manifest: %s -> Zinchenko et al. 2023 (not Bao)" % os.path.basename(mp))
print("\ndone")
