import requests, json
for TAP in ["https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync","https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"]:
    q = "SELECT TOP 5 table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%235/14%'"
    try:
        r = requests.post(TAP, data={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":q}, timeout=60)
        print(TAP, r.status_code, r.text[:600])
    except Exception as e:
        print(TAP,"ERR",e)
