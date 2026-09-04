import requests
TAP="https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
def q(sql):
    r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":sql}, timeout=180)
    if r.status_code!=200: return [["HTTP%d"%r.status_code, r.text[:200]]]
    return r.json().get("data",[])
terms = ["Abell 2744","MACS J0416","MACS J0717","MACS J1149","Abell 370","S1063","2248.7","Abell 2029","A2029","Frontier Field","BUFFALO","cluster member"]
for t in terms:
    rows=q("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE description LIKE '%%%s%%'" % t)
    print("### '%s' -> %d" % (t,len(rows)))
    for r_ in rows[:28]: print("   ", r_[0], "|", (r_[1] or "")[:115])
