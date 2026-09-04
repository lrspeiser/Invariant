"""SUPPLEMENT: r-band single-Sersic structural parameters for the SAMI DR3
GAMA-region (field/group) targets, from GAMA DR4 SersicCatSDSSv09 (Kelvin
et al. 2012, MNRAS 421, 1007) via the same anonymous Data Central TAP service.

WHY this is here: no SAMI DR3 table releases a Sersic INDEX.
InputCatGAMADR3 and InputCatClustersDR3 give Re, ellipticity and PA that are
Sersic-derived, but not n.  For the GAMA arm, n is recoverable because SAMI's
CATID in the GAMA regions IS the GAMA CATAID (verified: all 5536 SAMI GAMA
targets join to SersicCatSDSSv09).  For the CLUSTER arm it is NOT recoverable:
the fits are from Owers et al. 2019 (arXiv 1901.08185), which describes PROFIT
Sersic fits to SDSS DR9 / VST-ATLAS r-band imaging but publishes no per-galaxy
parameter table (its three deluxetables are spectral-classification summaries),
and the index is not propagated into SAMI DR3.

So DO NOT match cluster against field on Sersic n -- it exists for one arm
only.  The homogeneous structural basis covering both arms is
MGEPhotomUnregDR3 (ReMGE, epsMGE_Re, epsMGE_LW, mMGE).
"""
import hashlib
import io
import json
import os
from datetime import datetime, timezone

import requests
from astropy.io.votable import parse

TAP = "https://datacentral.org.au/vo/tap/sync"
OUT = os.path.dirname(os.path.abspath(__file__))

COLS = ["CATAID", "GALMAG_r", "GALRE_r", "GALINDEX_r", "GALELLIP_r", "GALPA_r",
        "GALREERR_r", "GALINDEXERR_r", "GALELLIPERR_r", "GALPAERR_r",
        "GALMAGERR_r", "GALMU0_r", "GALMUE_r", "GALMUEAVG_r", "GALR90_r",
        "GALCHI2_r", "GALPLAN_r", "PSFFWHM_r"]

QUERY = ("SELECT " + ", ".join("s." + c for c in COLS) +
         " FROM gama_dr4.SersicCatSDSSv09 AS s"
         " JOIN sami_dr3.InputCatGAMADR3 AS i ON s.CATAID = i.CATID")


def sha(b):
    h = hashlib.sha256(); h.update(b); return h.hexdigest()


def main():
    cnt = requests.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY":
                                   "SELECT COUNT(*) AS n FROM gama_dr4.SersicCatSDSSv09 AS s "
                                   "JOIN sami_dr3.InputCatGAMADR3 AS i ON s.CATAID = i.CATID"},
                        timeout=600)
    expected = int(cnt.text.strip().splitlines()[-1])
    assert expected == 5536, "join returns %d rows, expected 5536" % expected

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL",
                                 "FORMAT": "votable", "MAXREC": "1000000",
                                 "QUERY": QUERY}, timeout=900)
    r.raise_for_status()
    raw = r.content
    txt = raw.decode("utf-8", "replace")
    assert "<VOTABLE" in txt[:4000] and "<TABLEDATA>" in txt, "not a VOTable with data"
    assert 'value="OVERFLOW"' not in txt, "result truncated"

    rp = os.path.join(OUT, "gama_dr4_SersicCatSDSSv09_SAMIsubset.vot")
    open(rp, "wb").write(raw)

    tbl = parse(io.BytesIO(raw)).get_first_table()
    arr = tbl.to_table()
    assert len(arr) == expected, "got %d rows, COUNT said %d" % (len(arr), expected)

    tp = os.path.join(OUT, "gama_dr4_SersicCatSDSSv09_SAMIsubset.tsv")
    df = arr.to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    df.to_csv(tp, sep="\t", index=False)
    n = sum(1 for _ in open(tp, encoding="utf-8")) - 1
    assert n == expected

    man = {
        "file": os.path.basename(rp),
        "cleaned_file": os.path.basename(tp),
        "source_url": TAP,
        "service": "Data Central IVOA TAP, anonymous",
        "access_credentials": "none",
        "query": QUERY,
        "count_check_value": expected,
        "retrieved_utc": ts,
        "sha256": sha(raw),
        "bytes": len(raw),
        "cleaned_sha256": sha(open(tp, "rb").read()),
        "cleaned_bytes": os.path.getsize(tp),
        "row_count": n,
        "column_count": len(tbl.fields),
        "columns": [{"name": f.name, "unit": (str(f.unit) if f.unit else None),
                     "datatype": f.datatype, "description": f.description}
                    for f in tbl.fields],
        "reference": "Kelvin et al. 2012, MNRAS 421, 1007 (GAMA SIGMA/GALFIT "
                     "single-Sersic fits); served as gama_dr4.SersicCatSDSSv09.",
        "extraction": "Inner join on CATAID = SAMI CATID, no cut, no unit conversion.",
        "SCOPE_WARNING": (
            "GAMA-region (field and group) SAMI targets ONLY. There is no public "
            "Sersic index for the 896 SAMI CLUSTER-region galaxies: their Re, "
            "ellipticity and PA come from the PROFIT fits of Owers et al. 2019 "
            "(arXiv 1901.08185), which publishes no per-galaxy structural table, "
            "and SAMI DR3 does not carry the index. Matching cluster against field "
            "on Sersic n is therefore NOT possible; use MGEPhotomUnregDR3, which "
            "covers 895/896 cluster and 2100/2100 GAMA galaxies homogeneously."
        ),
        "note": "GALPA_r is in GALFIT image convention (x+, CCW), NOT the "
                "North-through-East convention used by the SAMI input catalogues; "
                "Croom+2021 states the SAMI PA column was corrected to N-through-E "
                "while earlier versions took the GAMA Sersic PA directly.",
    }
    json.dump(man, open(rp + ".manifest.json", "w"), indent=1)
    ok = df.GALINDEX_r.notna().sum()
    print("rows %d, Sersic index finite for %d (%.1f per cent)"
          % (n, ok, 100.0 * ok / n))
    print("wrote", rp, "and", tp)


if __name__ == "__main__":
    main()
