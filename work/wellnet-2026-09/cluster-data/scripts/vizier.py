"""VizieR asu-tsv client with the known HTTP-200-generic-page trap defeated.

VizieR answers a request for a nonexistent -source= with HTTP 200 and a generic
page.  probe() therefore checks that the response really is a TSV payload AND
that the echoed table identifier matches the one we asked for.
"""
import re
import urllib.parse
import urllib.request

BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
UA = "Mozilla/5.0 (compatible; gravity-lane-acquire/1.0; research)"


def build(source, out="**", maxrows="unlimited", extra=None):
    q = [("-source", source), ("-out", out), ("-out.max", str(maxrows))]
    if extra:
        q += list(extra)
    return BASE + "?" + urllib.parse.urlencode(q, safe="*/+")


def fetch(url, timeout=180):
    """curl is markedly faster than urllib in this environment."""
    import subprocess
    p = subprocess.run(["curl", "-s", "--max-time", str(timeout), "-A", UA, url],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("curl exit %d: %s" % (p.returncode, p.stderr[:400]))
    return 200, p.stdout.decode("utf-8", "replace")


def probe(source, maxrows=5, verbose=True):
    ok, url, txt, diag = _probe(source, maxrows)
    if verbose:
        print(("OK   " if ok else "FAIL ") + source + " :: " + diag, flush=True)
    return ok, url, txt, diag


def _probe(source, maxrows=5):
    """Return (ok, url, text, diagnosis)."""
    url = build(source, maxrows=maxrows)
    try:
        code, txt = fetch(url)
    except Exception as e:
        return False, url, "", "REQUEST FAILED: %r" % (e,)

    if "<html" in txt[:800].lower() or "<!DOCTYPE" in txt[:200]:
        return False, url, txt, "HTML page returned, not TSV -> source does not exist"

    # VizieR 7.6 signals a bad -source= inside an otherwise HTTP-200 TSV envelope.
    m = re.search(r"^#INFO\s+Error=(.+)$", txt, re.M)
    if m:
        return False, url, txt, "VizieR error: " + m.group(1).strip()

    # A real asu-tsv payload carries '#Table' / '#Title' comment lines.
    if "#Table" not in txt and "#Title" not in txt:
        return False, url, txt, "no #Table/#Title header -> generic/empty response"

    # Echo the identifier back: VizieR renders J/ApJ/821/116/table3 as J_ApJ_821_116 etc.
    stem = source.split("/")[0:4]
    echoed = "_".join(source.replace("+", "").split("/"))
    tables = re.findall(r"^#Table\s+(\S+)", txt, re.M)
    titles = re.findall(r"^#Title:\s*(.+)$", txt, re.M)

    # count data rows: lines after the '---' separator that are not comments
    lines = txt.splitlines()
    sep = [i for i, l in enumerate(lines) if l.startswith("---")]
    nrows = 0
    header = []
    if sep:
        i = sep[-1]
        header = lines[i - 2].split("\t") if i >= 2 else []
        nrows = sum(1 for l in lines[i + 1:] if l.strip() and not l.startswith("#"))

    diag = "tables=%s titles=%s nrows>=%d ncols=%d" % (tables, titles, nrows, len(header))
    ok = nrows > 0 and bool(tables or titles)
    return ok, url, txt, diag


def parse_tsv(txt):
    """Split a VizieR asu-tsv payload into (meta_lines, colnames, units, rows)."""
    lines = txt.splitlines()
    sep = [i for i, l in enumerate(lines) if l.startswith("---")]
    if not sep:
        raise ValueError("no '---' separator line in VizieR payload")
    i = sep[-1]
    colnames = lines[i - 2].split("\t")
    units = lines[i - 1].split("\t")
    rows = [l.split("\t") for l in lines[i + 1:] if l.strip() and not l.startswith("#")]
    meta = [l for l in lines[:i - 2] if l.startswith("#")]
    return meta, colnames, units, rows
