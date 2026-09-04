"""Find VizieR catalogues whose tables contain a given set of column names.

VizieR's TAP_SCHEMA stores column_name values wrapped in literal single quotes,
e.g. 'Vrot', so the ADQL literal must be "'Vrot'".
"""
import sys
import requests

H = {"User-Agent": "wellnet-gravity-acquisition/1.0 (academic research)"}
URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def run(adql):
    r = requests.post(URL, data={"REQUEST": "doQuery", "LANG": "ADQL",
                                 "FORMAT": "tsv", "QUERY": adql},
                      headers=H, timeout=900)
    r.raise_for_status()
    return r.text


if __name__ == "__main__":
    names = sys.argv[1:] or ["Vrot", "Incl", "PA"]
    lit = ",".join("'''%s'''" % n for n in names).replace("'''", "'" + chr(39) + "'")
    # build "'Vrot'" style literals: outer ADQL quotes + inner literal quote chars
    q = chr(39)
    lit = ",".join(f"{q}{q}{n}{q}{q}" for n in names)
    adql = (f"SELECT table_name, COUNT(*) AS n FROM TAP_SCHEMA.columns "
            f"WHERE column_name IN ({lit}) "
            f"GROUP BY table_name HAVING COUNT(*)>={len(names)}")
    print("ADQL:", adql)
    print(run(adql))
