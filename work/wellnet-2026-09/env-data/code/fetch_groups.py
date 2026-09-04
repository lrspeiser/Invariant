import requests, hashlib, os, sys, time
BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\groups"
JOBS = {
 "tempel2014_galaxies.tsv": ("J/A+A/566/A1/galaxies", None),
 "tempel2014_groups.tsv":   ("J/A+A/566/A1/groups", None),
 "tempel2017_table1_galaxies.tsv": ("J/A+A/602/A100/table1", None),
 "tempel2017_table2_groups.tsv":   ("J/A+A/602/A100/table2", None),
 "mcxc_piffaretti2011.tsv": ("J/A+A/534/A109/mcxc", None),
}
for name,(src,cols) in JOBS.items():
    out=os.path.join(BASE,name)
    if os.path.exists(out) and os.path.getsize(out)>2000:
        print("exists",name,os.path.getsize(out)); continue
    params={"-source":src,"-out.max":"unlimited","-out.all":""}
    t0=time.time()
    r=requests.get("https://vizier.cds.unistra.fr/viz-bin/asu-tsv",params=params,timeout=900)
    r.raise_for_status()
    txt=r.text
    # ASSERT: real TSV, not the generic VizieR HTML/error page
    assert "#Table" in txt and "#Column" in txt, "NOT A TSV TABLE for %s: %r"%(src, txt[:300])
    assert src.split("/")[-1] in txt or src in txt, "identifier %s not echoed back"%src
    open(out,"w",encoding="utf-8",newline="").write(txt)
    nrow=sum(1 for l in txt.splitlines() if l and not l.startswith("#") and "\t" in l)
    print("%-34s %9d bytes  ~%7d data lines  %.1fs  url=%s"%(name,len(txt.encode()),nrow,time.time()-t0,r.url[:110]))
