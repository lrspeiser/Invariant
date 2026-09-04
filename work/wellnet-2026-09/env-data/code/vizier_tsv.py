"""Robust reader for VizieR asu-tsv output, with hard assertions.

Known trap (recorded in the programme brief): VizieR returns HTTP 200 with a
generic page for a nonexistent -source=.  Any file that does not carry both
'#Table' and '#Column' comment lines is rejected here rather than silently
parsed into an empty frame.

VizieR asu-tsv layout after the '#' comment block:
    line 0 : column names, tab separated
    line 1 : units          (may be blank fields)
    line 2 : dashed rule    ('----\t----\t...')
    line 3+: data
"""
import io
import pandas as pd


def read_vizier_tsv(path, expect_cols=None, expect_min_rows=None):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    assert "#Table" in text, "%s: no '#Table' line - not a VizieR TSV table" % path
    assert "#Column" in text, "%s: no '#Column' lines - not a VizieR TSV table" % path

    decl = [ln.split("\t")[1] for ln in text.splitlines()
            if ln.startswith("#Column") and len(ln.split("\t")) > 1]

    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    body = [ln for ln in body if ln.strip() != ""]
    assert len(body) >= 3, "%s: fewer than 3 non-comment lines" % path

    names = body[0].split("\t")
    units = body[1].split("\t")
    rule = body[2]
    assert set(rule.replace("\t", "").strip()) <= set("-"), \
        "%s: line 3 is not the dashed rule, got %r" % (path, rule[:80])
    assert len(names) == len(decl), \
        "%s: %d header names vs %d #Column declarations" % (path, len(names), len(decl))
    for a, b in zip(names, decl):
        assert a.strip() == b.strip(), "%s: header %r != declaration %r" % (path, a, b)

    df = pd.read_csv(io.StringIO("\n".join(body[3:])), sep="\t", header=None,
                     names=names, dtype=str, keep_default_na=False,
                     engine="python", quoting=3)

    if expect_cols is not None:
        missing = [c for c in expect_cols if c not in df.columns]
        assert not missing, "%s: missing expected columns %s (have %s)" % (
            path, missing, list(df.columns))
    if expect_min_rows is not None:
        assert len(df) >= expect_min_rows, \
            "%s: %d rows < expected minimum %d" % (path, len(df), expect_min_rows)

    df.attrs["units"] = dict(zip(names, units))
    df.attrs["source_file"] = path
    return df


def num(df, col):
    """Coerce a VizieR string column to float, VizieR blanks -> NaN."""
    return pd.to_numeric(df[col].str.strip().replace("", "nan"), errors="coerce")
