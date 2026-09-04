import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=240)
    if r.status_code!=200: return [["HTTP%d"%r.status_code,r.text[:250]]]
    return r.json().get("data",[])
for t in ['"J/ApJ/812/114/table3"','"J/A+A/646/A83/a2744"','"J/ApJS/224/33/table2"','"J/A+A/671/A146/catalog"']:
    print("="*18,t, q("SELECT COUNT(*) FROM %s"%t))
    for r_ in q("SELECT column_name, unit, description FROM TAP_SCHEMA.columns WHERE table_name='%s'"%t):
        print("   %-16s %-12s %s"%(r_[0],r_[1] or "",(r_[2] or "")[:82]))
print("\n### MUSE Richard+2021 tables")
for r_ in q("SELECT table_name,description FROM TAP_SCHEMA.tables WHERE table_name LIKE '%646/A83%'"):
    print("  ",r_[0],"|",(r_[1] or "")[:80], q("SELECT COUNT(*) FROM %s"%r_[0]))
