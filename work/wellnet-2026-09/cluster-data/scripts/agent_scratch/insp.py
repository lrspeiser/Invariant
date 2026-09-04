import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=180)
    if r.status_code!=200: return [["HTTP%d"%r.status_code,r.text[:200]]]
    return r.json().get("data",[])
print("### all tables in J/A+A/709/A254 and J/MNRAS/489/99 and J/MNRAS/514/497 and J/ApJ/872/192 and J/ApJ/871/129 and J/MNRAS/477/648")
for pat in ["%A+A/709/A254%","%MNRAS/489/99%","%MNRAS/514/497%","%ApJ/872/192%","%ApJ/871/129%","%MNRAS/477/648%","%A+A/671/A146%","%ApJ/773/86%"]:
    for r_ in q("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name LIKE '%s'"%pat):
        n=q("SELECT COUNT(*) FROM %s"%r_[0])
        print("  %-34s nrow=%-8s %s" % (r_[0], n[0][0] if n else "?", (r_[1] or "")[:95]))
print()
for t in ['"J/A+A/709/A254/a2744"','"J/MNRAS/489/99/j2248pr"','"J/MNRAS/514/497/tablea1"','"J/ApJ/872/192/table1"','"J/ApJ/871/129/table1"']:
    print("="*18, t)
    rows=q("SELECT column_name, unit, description FROM TAP_SCHEMA.columns WHERE table_name='%s'"%t)
    for r_ in rows: print("   %-14s %-12s %s" % (r_[0], r_[1] or "", (r_[2] or "")[:88]))
