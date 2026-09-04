"""Fetch arXiv e-print source tarballs for the misalignment papers that are not
in VizieR. IDs resolved by web search + verified against the paper title inside
the fetched source. Nothing downloaded is executed."""
import sys, os, tarfile, gzip, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import http_get

D = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    ("bryant2019_sami_misalign", "1811.09298",
     "The SAMI Galaxy Survey: stellar and gas misalignments and the origin of gas",
     "misalign"),
    ("jin2016_manga_misalign", "1611.00528",
     "SDSS-IV MaNGA: Properties of galaxies with kinematically decoupled stellar and gaseous components",
     "decoupled"),
    ("bao2023_manga_counterrot_gas", "2305.13387",
     "Gas and stellar kinematic misalignment in MaNGA galaxies: origin of counter-rotating gas",
     "counter-rotat"),
    ("ristea2022_sami_misalign_drivers", "2210.01147",
     "The SAMI Galaxy Survey: physical drivers of stellar-gas kinematic misalignments",
     "misalign"),
]

log = []
for short, aid, expect_title, kw in TARGETS:
    dest = os.path.join(D, "crot_%s.eprint.tar.gz" % short)
    try:
        http_get("https://arxiv.org/e-print/%s" % aid, dest)
    except Exception as e:
        print("  !! EPRINT FAIL %s (%s): %s" % (short, aid, e))
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
                    print("  SKIP unsafe %r" % nm); continue
                tf.extract(m, outdir)
                members.append(nm)
    except tarfile.ReadError:
        try:
            data = gzip.decompress(open(dest, "rb").read())
            p = os.path.join(outdir, "%s.tex" % short)
            open(p, "wb").write(data)
            members = ["%s.tex" % short]
        except Exception as e:
            print("  !! not tar/gz: %s" % e)
    tex = [m for m in members if m.lower().endswith(".tex")]
    print("\n%s (arXiv:%s): %d members, %d tex: %s" % (short, aid, len(members), len(tex), tex))
    # verify we got the right paper
    found = False
    for t in tex:
        p = os.path.join(outdir, t)
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if kw.lower() in txt.lower():
            found = True
        m = re.search(r"\\title(?:\[[^\]]*\])?\s*\{", txt)
        if m:
            # brace-match the title
            i = m.end(); depth = 1; buf = []
            while i < len(txt) and depth:
                ch = txt[i]
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: break
                buf.append(ch); i += 1
            print("   TITLE IN SOURCE: %s" % " ".join("".join(buf).split())[:150])
    print("   keyword %r present in source: %s" % (kw, found))
    log.append({"short": short, "arxiv": aid, "ok": True, "n_members": len(members),
                "tex_files": tex, "keyword_verified": found,
                "expected_title": expect_title})
    time.sleep(3)

with open(os.path.join(D, "crot_eprint_fetch_log.json"), "w", encoding="utf-8") as fh:
    json.dump(log, fh, indent=2)
print("\nWROTE crot_eprint_fetch_log.json")
