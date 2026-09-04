# -*- coding: utf-8 -*-
import tarrange, os, json, hashlib, datetime, requests
BASE=r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
MEM=os.path.join(BASE,"members")
def utcnow(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def sha256f(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def parse_hdr(path):
    """HFF-DeepSpace .zout/.fout: '#' comment lines; last '#' line before data is the column header."""
    hdr=None; nrow=0
    with open(path,"r",errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                toks=line[1:].split()
                if toks: hdr=toks
            elif line.strip():
                nrow+=1
    return hdr, nrow

FIELDS=[("abell2744","abell2744-clu","abell2744clu","Abell 2744"),
        ("macs0416","macs0416-clu","macs0416clu","MACS J0416.1-2403"),
        ("macs0717","macs0717-clu","macs0717clu","MACS J0717.5+3745"),
        ("macs1149","macs1149-clu","macs1149clu","MACS J1149.5+2223"),
        ("abell1063","abell1063","abell1063clu","Abell S1063 / RXC J2248.7-4431"),
        ("abell370","abell370-clu","abell370clu","Abell 370")]
WANT=[".zout",".fout"]
NOTE={
".zout":("Shipley et al. 2018 ApJS 235,14 HFF-DeepSpace EAZY output (.zout) for the {cl} CLUSTER pointing, catalogue version v3.9. "
  "MODEL-DERIVED: photometric redshifts from EAZY SED fitting -- z_a, z_m1, z_peak (the recommended photo-z), z_m2, and the 68/95/99 per cent "
  "confidence intervals l68/u68/l95/u95/l99/u99, plus chi2 and odds/qz quality statistics. z_spec is the compiled SPECTROSCOPIC redshift where "
  "available (measured; -1 = none). Row order and 'id' match the HFF-DeepSpace photometry catalogue hffds_{root}_v3.9.cat, i.e. the same objects "
  "as VizieR J/ApJS/235/14/clugal restricted to this cluster. NO membership flag: membership must be derived by the user from z_spec/z_peak."),
".fout":("Shipley et al. 2018 ApJS 235,14 HFF-DeepSpace FAST output (.fout) for the {cl} CLUSTER pointing, catalogue version v3.9. "
  "ENTIRELY MODEL-DERIVED (FAST SED fitting at fixed redshift z, Bruzual & Charlot 2003 templates, Chabrier IMF, Calzetti dust): "
  "lmass = log10 stellar mass [Msun], lsfr, lssfr, lage, Av, ltau, metal, chi2. Column 'z' is the redshift the fit was done at (z_spec if "
  "available else EAZY z_peak). Row order and 'id' match hffds_{root}_v3.9.cat / VizieR J/ApJS/235/14/clugal for this cluster. "
  "NOTE: these stellar masses are NOT corrected for lensing magnification.")}
res={}
for dirn, filestem, root, clname in FIELDS:
    url = "https://archive.stsci.edu/hlsps/hff-deepspace/%s/multi/hlsp_hff-deepspace_hst_acs-wfc3_%s_multi_v1_catalogs.tar" % (dirn, filestem)
    print("== %s" % clname); print("   ", url)
    try:
        total, mem = tarrange.index_tar(url, verbose=False)
    except Exception as e:
        print("   INDEX FAIL", e); res[clname]={"status":"FAIL","err":str(e)}; continue
    got=[]
    for ext in WANT:
        cands=[m for m in mem if m["name"].endswith(ext)]
        if not cands:
            print("   MISSING", ext); continue
        m=cands[0]
        base=os.path.basename(m["name"])
        out=os.path.join(MEM, "%s_ShipleyHFFDeepSpace_v3.9_%s" % (clname.split()[0].replace("/","-"), base))
        # normalise name
        out=os.path.join(MEM, "HFFDS_%s_v3.9%s" % (root, ext))
        n=tarrange.fetch_member(url, m, out)
        hdr,nrow = parse_hdr(out)
        man={"file":os.path.basename(out),
             "source_url":url,
             "exact_query":"HTTP GET with Range: bytes=%d-%d on the MAST HLSP tar, extracting tar member '%s' (tar indexed by walking 512-byte ustar headers via range requests; whole tar NOT downloaded)" % (m["offset"], m["offset"]+m["size"]-1, m["name"]),
             "retrieved_utc":utcnow(),
             "sha256":sha256f(out),
             "bytes":os.path.getsize(out),
             "row_count":nrow,
             "column_count":len(hdr) if hdr else 0,
             "columns":[{"name":h,"unit":""} for h in (hdr or [])],
             "extraction":"Byte-range extraction of a single member from the remote uncompressed tar; bytes written verbatim, no reformatting. Rows counted as non-'#' non-blank lines; columns taken from the final '#' header line.",
             "tar_member":m["name"], "tar_member_size":m["size"], "tar_total_bytes":total,
             "note":NOTE[ext].format(cl=clname, root=root),
             "raw_response_file":os.path.basename(out)+"  (this file IS the verbatim upstream bytes)"}
        json.dump(man, open(out+".manifest.json","w",encoding="utf-8"), indent=2)
        print("   OK %-28s rows=%-7d cols=%-4d bytes=%d" % (base, nrow, len(hdr) if hdr else 0, n))
        got.append((ext,nrow,len(hdr) if hdr else 0))
    res[clname]=got
print(json.dumps(res,indent=1))
