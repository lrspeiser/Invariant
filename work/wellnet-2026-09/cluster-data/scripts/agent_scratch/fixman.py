import os, json, glob, collections
MEM=r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data\members"
# 1. fix .zout manifests: header is the FIRST '#' line
for p in glob.glob(os.path.join(MEM,"*.zout.manifest.json")):
    dat=p[:-len(".manifest.json")]
    with open(dat,errors="replace") as f: first=f.readline()
    hdr=first[1:].split()
    m=json.load(open(p))
    m["columns"]=[{"name":h,"unit":""} for h in hdr]
    m["column_count"]=len(hdr)
    m["extraction"]=("Byte-range extraction of a single member from the remote uncompressed tar; bytes written verbatim, no reformatting. "
      "Rows counted as non-'#' non-blank lines. Column names taken from the FIRST '#' line (the EAZY column header); the second '#' line is a "
      "version stamp, not a header. Sentinel -99 marks objects with no EAZY solution; z_spec = -1 means no spectroscopic redshift.")
    json.dump(m,open(p,"w",encoding="utf-8"),indent=2)
    print("fixed",os.path.basename(p),len(hdr))
# 2. per-cluster row counts in the VizieR clugal file, col 'Cl'
cl=os.path.join(MEM,"HFF6_Shipley2018_ApJS235_14_clugal_photometry.raw.tsv")
lines=open(cl,errors="replace").read().split("\n")
body=[l for l in lines if not l.startswith("#")]
while body and body[0].strip()=="": body.pop(0)
hdr=body[0].split("\t"); ci=hdr.index("Cl")
cnt=collections.Counter()
for l in body[3:]:
    if l.strip()=="": continue
    cnt[l.split("\t")[ci].strip()]+=1
print("\nVizieR clugal rows per cluster:")
for k,v in sorted(cnt.items()): print("   %-14s %d" % (k,v))
print("\nMAST .zout/.fout row counts:")
for p in sorted(glob.glob(os.path.join(MEM,"HFFDS_*.manifest.json"))):
    m=json.load(open(p)); print("   %-32s rows=%-7d cols=%d" % (m["file"],m["row_count"],m["column_count"]))
