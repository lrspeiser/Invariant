"""ATLAS3D II (Krajnovic+ 2011, MNRAS 414, 2923) is NOT in VizieR under any
Krajnovic entry (verified by find_catalogs over the whole VizieR description
index). Acquire it from (a) the arXiv e-print source tarball and (b) the ATLAS3D
project page. Downloaded files are READ ONLY -- nothing is executed."""
import sys, os, io, tarfile, gzip, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import http_get

D = os.path.dirname(os.path.abspath(__file__))

# arXiv e-print for ATLAS3D II
SRC = [
    ("atlas3d_II_krajnovic2011", "1102.3801"),
    ("atlas3d_III_emsellem2011", "1102.4444"),
]

for short, aid in SRC:
    dest = os.path.join(D, "crot_%s.eprint.tar.gz" % short)
    try:
        r = http_get("https://arxiv.org/e-print/%s" % aid, dest)
    except Exception as e:
        print("EPRINT FAIL %s: %s" % (aid, e)); continue
    head = open(dest, "rb").read(4)
    print("  magic=%r size=%d" % (head, os.path.getsize(dest)))
    outdir = os.path.join(D, "crot_%s_src" % short)
    os.makedirs(outdir, exist_ok=True)
    try:
        with tarfile.open(dest, "r:*") as tf:
            names = tf.getnames()
            print("  tar members (%d): %s" % (len(names), names[:40]))
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # safety: no absolute paths / traversal
                nm = m.name.replace("\\", "/")
                if nm.startswith("/") or ".." in nm.split("/"):
                    print("  SKIP unsafe member %r" % nm); continue
                tf.extract(m, outdir)
    except tarfile.ReadError:
        # single gzipped .tex
        try:
            data = gzip.decompress(open(dest, "rb").read())
            p = os.path.join(outdir, "%s.tex" % short)
            open(p, "wb").write(data)
            print("  single gz tex -> %s (%d bytes)" % (p, len(data)))
        except Exception as e:
            print("  NOT a tar or gz: %s" % e)

# ATLAS3D project page listing
try:
    http_get("https://www-astro.physics.ox.ac.uk/atlas3d/",
             os.path.join(D, "crot_atlas3d_projectpage.raw.html"))
except Exception as e:
    print("PROJECT PAGE FAIL: %s" % e)

for root, dirs, files in os.walk(D):
    for f in files:
        if "_src" in root:
            p = os.path.join(root, f)
            print("SRC FILE %-70s %d bytes" % (os.path.relpath(p, D), os.path.getsize(p)))
