import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
pats = ["%ApJS/235/14%","%A+A/590/A30%","%A+A/590/A31%","%A+A/607/A30%",
        "%A+A/631/A130%","%A+A/645/A140%","%A+A/674/A79%","%A+A/659/A24%",
        "%ApJS/229/20%","%A+A/667/A117%","%A+A/673/A80%"]
for p in pats:
    q = "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name LIKE '%s'" % p
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":q}, timeout=120)
    if r.status_code!=200: print(p,"HTTP",r.status_code); continue
    rows=r.json().get("data",[])
    print("=== %s -> %d" % (p,len(rows)))
    for row in rows: print("    ", row[0], "|", (row[1] or "")[:120])
