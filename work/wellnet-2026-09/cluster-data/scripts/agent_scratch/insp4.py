import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=240)
    if r.status_code!=200: return [["HTTP%d"%r.status_code,r.text[:250]]]
    return r.json().get("data",[])
print("### Molino HFF photo-z tables")
for r_ in q("SELECT table_name,description FROM TAP_SCHEMA.tables WHERE table_name LIKE '%MNRAS/470/95%'"):
    print("  ",r_[0], q("SELECT COUNT(*) FROM %s"%r_[0]), "|",(r_[1] or "")[:70])
print("\n### columns of macs0717 photo-z")
for r_ in q("SELECT column_name,unit,description FROM TAP_SCHEMA.columns WHERE table_name='\"J/MNRAS/470/95/macs0717\"'"):
    print("   %-16s %-10s %s"%(r_[0],r_[1] or "",(r_[2] or "")[:80]))
print("\n### MACS-cluster spectroscopy candidates (Ebeling/Ma/Limousin/Connor/Jauzac)")
for pat in ["%Ebeling%","%MACS%spectro%","%MACSJ0717%","%0717%"]:
    rr=q("SELECT table_name,description FROM TAP_SCHEMA.tables WHERE description LIKE '%s'"%pat)
    print("  pattern %s -> %d"%(pat,len(rr)))
    for r_ in rr[:14]: print("     ",r_[0],"|",(r_[1] or "")[:88])
