import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=180)
    if r.status_code!=200: print("HTTP",r.status_code,r.text[:300]); return []
    return r.json().get("data",[])
for t in ['"J/ApJS/235/14/table1"','"J/A+A/607/A30/m0717clz"','"J/ApJS/229/20/table2"']:
    print("="*20, t)
    rows=q("SELECT column_name, unit, description FROM TAP_SCHEMA.columns WHERE table_name='%s'" % t)
    print("  ncols=",len(rows))
    for r_ in rows: print("   %-16s %-10s %s" % (r_[0], r_[1] or "", (r_[2] or "")[:85]))
print("=== rowcounts")
for t in ['"J/ApJS/235/14/clugal"','"J/A+A/590/A31/a2744cl"','"J/A+A/590/A31/m0416cl"','"J/A+A/607/A30/m0717clz"','"J/A+A/607/A30/m1149clz"','"J/ApJS/229/20/table2"']:
    print(t, q("SELECT COUNT(*) FROM %s" % t))
