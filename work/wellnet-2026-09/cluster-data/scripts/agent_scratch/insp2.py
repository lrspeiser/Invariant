import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=180)
    if r.status_code!=200: return [["HTTP%d"%r.status_code,r.text[:300]]]
    return r.json().get("data",[])
for t in ['"J/ApJ/871/129/table1"','"J/MNRAS/489/99/a370pr"','"J/MNRAS/489/99/a370ph"','"J/ApJ/773/86/table1"']:
    print("="*18,t)
    for r_ in q("SELECT column_name, unit, description FROM TAP_SCHEMA.columns WHERE table_name='%s'"%t):
        print("   %-14s %-12s %s"%(r_[0],r_[1] or "",(r_[2] or "")[:88]))
print("\n### Granata a2744 first 2 rows, selected cols")
print(q('SELECT TOP 2 "ID","RAJ2000","DEJ2000","F160W","ReF160W","nF160W","ARF160W","PAF160W" FROM "J/A+A/709/A254/a2744"'))
