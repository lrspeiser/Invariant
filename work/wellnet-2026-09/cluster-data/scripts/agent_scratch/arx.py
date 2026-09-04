import requests, urllib.parse, re, sys
def s(q, n=12):
    u="http://export.arxiv.org/api/query?search_query=%s&max_results=%d&sortBy=submittedDate&sortOrder=descending"%(urllib.parse.quote(q),n)
    r=requests.get(u,timeout=120)
    if r.status_code!=200: print("HTTP",r.status_code); return
    t=r.text
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        idm=re.search(r"<id>(.*?)</id>",e); ti=re.search(r"<title>(.*?)</title>",e,re.S)
        au=re.findall(r"<name>(.*?)</name>",e); jr=re.search(r"journal_ref>(.*?)<",e)
        print("  %-34s %s" % (idm.group(1).split('/abs/')[-1] if idm else "?", " ".join(ti.group(1).split())[:96]))
        print("     %s | %s" % ((au[0] if au else "")+(" et al." if len(au)>1 else ""), (jr.group(1) if jr else "no journal_ref")[:70]))
for qq in ['all:"Bergamini" AND all:"cluster members" AND all:"velocity dispersion"',
           'au:Bergamini_P AND abs:"lens"',
           'all:"MACS J0717" AND all:"Sersic"']:
    print("### ",qq); s(qq); print()
