import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=180)
    if r.status_code!=200: print("HTTP",r.status_code,r.text[:300]); return []
    return r.json().get("data",[])
for t in ['"J/ApJS/235/14/clugal"','"J/A+A/590/A31/a2744cl"','"J/A+A/607/A30/m0717clz"','"J/ApJS/229/20/table2"']:
    print("="*20, t)
    rows=q("SELECT column_name, unit, ucd, description FROM TAP_SCHEMA.columns WHERE table_name='%s'" % t.replace("'","''"))
    print("  ncols=",len(rows))
    for r_ in rows: print("   %-16s %-10s %-24s %s" % (r_[0], r_[1] or "", (r_[2] or "")[:24], (r_[3] or "")[:70]))
