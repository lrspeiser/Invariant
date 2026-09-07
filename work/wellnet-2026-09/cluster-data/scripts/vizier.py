"""VizieR asu-tsv client under `catalogue_validation` v3.

VizieR answers a request for a nonexistent -source= with HTTP 200 in at least
three different disguises, and only the full detector set below separates them
from a real serve:

  * a generic HTML page;
  * an otherwise well-formed TSV envelope carrying `#INFO Error=...`;
  * a COMPLETELY UNRELATED REAL CATALOGUE, served silently in place of the
    request.  In this lane that fallback was `J/MNRAS/430/1125` (Cooper+2013,
    an RMS near-infrared YSO survey); in the strong-lensing lane it was `I/16`
    with `CatalogsExamined=10213`; in Run AZ it was the wrong PAPER entirely.
    URL-encoding the `+` does not help.

Registry rule `catalogue_validation` v2 -- "#Name: echo OR absence of
CatalogsExamined, either alone sufficient" -- is recorded as BROKEN: neither
detector fires on all three variants.  v3 therefore requires ALL THREE:

  D1  `#Name:` echoes the EXACT identifier requested (or its parent catalogue).
  D2  no `CatalogsExamined=` anywhere in the payload.
  D3  `#Title:` matches the expected author/year, when one is supplied.

A zero-row result is NOT an absence until D1-D3 have been evaluated: a wrong
catalogue served under HTTP 200 reads as a confident, and false, negative.

This is the same rule implemented by `goldcluster/archives.py`
(`vizier_validate` / `vizier_fetch`) and by `clash-audit/ingest.py` (`_vizier`).
It is reimplemented here rather than imported because `archives.py` arms a
provenance guard scoped to the gold-cluster lane and writes into that lane's
own `raw/`; this lane has to stay self-contained.

HISTORY.  Before 2026-09-06 `probe()` computed `stem` and `echoed` and then
never used either: the verdict was `nrows > 0 and bool(tables or titles)`,
which is v1 plus a row count.  D1 was dead code, D2 was absent, D3 was absent.
Every verdict this module produced before that date was made with a dead D1;
the twelve velocity-lane rejections that rest on it are re-validated in
`velocities/REVALIDATION_v3.json`.
"""
import re
import urllib.parse

BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
UA = "Mozilla/5.0 (compatible; gravity-lane-acquire/1.0; research)"

RULE = "catalogue_validation v3"

#: fallback catalogues this programme has actually been served in place of a
#: nonexistent identifier.  Diagnostic only -- D1 does not depend on this list.
KNOWN_FALLBACKS = ("J/MNRAS/430/1125", "I/16")


class ValidationFailure(RuntimeError):
    """The payload is not the catalogue that was requested."""


def build(source, out="**", maxrows="unlimited", extra=None):
    q = [("-source", source), ("-out", out), ("-out.max", str(maxrows))]
    if extra:
        q += list(extra)
    return BASE + "?" + urllib.parse.urlencode(q, safe="*/+")


def fetch(url, timeout=180, max_bytes=1 << 20):
    """Fetch a bounded prefix.  Returns (status, text); 206 means truncated.

    curl is markedly faster than urllib in this environment, but the naive
    `subprocess.run(...); raise if returncode` form has a trap of its own.  For
    a nonexistent `-source=` VizieR does not answer briefly -- it runs an
    all-sky cross-match over ~10k catalogues and STREAMS the result.  The
    stream does not finish inside any sane timeout (7.0 MB and still going at
    100 s, observed 2026-09-07 for J/A+A/656/A147), so curl exits 28 and the
    old code raised and DISCARDED the bytes it already had.

    Those discarded bytes are the evidence.  `CatalogsExamined=`, the `#Name:`
    echo and `#Title:` all appear in the first few kB, so a bounded prefix is
    enough to run every v3 detector -- and a probe that dies with
    REQUEST_FAILED is a probe that cannot tell "absent" from "unreachable".
    We therefore read at most `max_bytes` and stop, and raise only when nothing
    at all arrived.
    """
    import subprocess
    p = subprocess.Popen(["curl", "-s", "--max-time", str(timeout), "-A", UA, url],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf = b""
    try:
        while len(buf) < max_bytes:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        if p.poll() is None:
            p.kill()
        try:
            p.stdout.close()
            err = p.stderr.read()
            p.stderr.close()
        except Exception:                                     # noqa: BLE001
            err = b""
        p.wait()
    rc = p.returncode
    if not buf:
        raise RuntimeError("curl exit %s with an empty body: %s" % (rc, err[:400]))
    truncated = len(buf) >= max_bytes or rc not in (0, None)
    return (206 if truncated else 200), buf.decode("utf-8", "replace")


def validate(source, txt, want_title=()):
    """Apply the v3 three-detector rule.  Return (ok, detectors).

    `want_title` is a sequence of substrings -- typically author surname and
    year -- all of which must appear in the joined `#Title:` lines.  When it is
    empty D3 is not evaluated and is reported as None; the caller is then
    relying on D1+D2 alone, which is v2 and is known to be insufficient for
    asserting that a payload IS the requested catalogue.
    """
    lines = txt.splitlines()
    d = {"rule": RULE, "source": source}

    head = txt[:800].lower()
    d["not_html"] = "<html" not in head and "<!doctype" not in head

    m = re.search(r"^#INFO\s+Error=(.+)$", txt, re.M)
    d["no_vizier_error"] = m is None
    d["vizier_error"] = m.group(1).strip() if m else None

    # ---- D1: the identifier must be echoed back -------------------------
    names = [l.split(":", 1)[1].strip() for l in lines if l.startswith("#Name:")]
    want = source.rstrip("/")
    parent = "/".join(want.split("/")[:4])
    d["names_seen"] = names
    d["D1_name_echo"] = any(n == want or n == parent or n.startswith(parent + "/")
                            for n in names)
    d["served_instead"] = ([n for n in names
                            if not (n == want or n == parent
                                    or n.startswith(parent + "/"))]
                           if not d["D1_name_echo"] else [])
    d["known_fallback"] = [f for f in KNOWN_FALLBACKS
                           if any(n.startswith(f) for n in names)]

    # ---- D2: the fuzzy-fallback signature must be absent -----------------
    d["D2_no_catalogs_examined"] = not any("CatalogsExamined" in l for l in lines)
    mx = re.search(r"CatalogsExamined=(\d+)", txt)
    d["catalogs_examined"] = int(mx.group(1)) if mx else None

    # ---- D3: the title must match the expected author/year ---------------
    titles = [l.split(":", 1)[1].strip() for l in lines if l.startswith("#Title:")]
    d["titles_seen"] = titles
    joined = " ".join(titles).lower()
    d["D3_title_match"] = (all(t.lower() in joined for t in want_title)
                           if want_title else None)
    d["want_title"] = list(want_title)

    # ---- row count, reported but NEVER a detector ------------------------
    sep = [i for i, l in enumerate(lines) if l.startswith("---")]
    d["n_rows"] = (sum(1 for l in lines[sep[-1] + 1:]
                       if l.strip() and not l.startswith("#")) if sep else 0)
    d["tables_seen"] = re.findall(r"^#Table\s+(\S+)", txt, re.M)

    ok = (d["not_html"] and d["no_vizier_error"] and d["D1_name_echo"]
          and d["D2_no_catalogs_examined"] and d["D3_title_match"] is not False)
    d["ok"] = ok
    return ok, d


def classify(source, txt, want_title=()):
    """Return (verdict, detectors) where verdict is one of:

      SERVED           the payload IS the requested catalogue (D1-D3 pass)
      ABSENT_ERROR     VizieR said so explicitly -- `#INFO Error=`
      ABSENT_FALLBACK  D1 failed: a DIFFERENT catalogue came back under HTTP 200
      ABSENT_HTML      a generic HTML page came back
      INDETERMINATE    D1 and D2 pass but the supplied D3 title does not match

    Only an ABSENT_* verdict licenses the word "absent".  A SERVED payload with
    zero rows is an EMPTY catalogue, which is a different claim about the world.
    """
    ok, d = validate(source, txt, want_title)
    if not d["not_html"]:
        v = "ABSENT_HTML"
    elif not d["no_vizier_error"]:
        v = "ABSENT_ERROR"
    elif not d["D1_name_echo"] or not d["D2_no_catalogs_examined"]:
        v = "ABSENT_FALLBACK"
    elif d["D3_title_match"] is False:
        v = "INDETERMINATE"
    else:
        v = "SERVED"
    d["verdict"] = v
    return v, d


def probe(source, want_title=(), maxrows=5, verbose=True):
    """Probe one identifier.  Returns (verdict, url, text, detectors)."""
    url = build(source, maxrows=maxrows)
    try:
        code, txt = fetch(url)
    except Exception as e:                                    # noqa: BLE001
        d = {"rule": RULE, "source": source, "verdict": "REQUEST_FAILED",
             "error": repr(e)}
        if verbose:
            print("REQUEST_FAILED   " + source + " :: " + repr(e), flush=True)
        return "REQUEST_FAILED", url, "", d
    verdict, d = classify(source, txt, want_title)
    d["truncated"] = (code == 206)
    if verbose:
        print("%-16s %-22s D1=%-5s D2=%-5s D3=%-5s rows=%-5s %s"
              % (verdict, source, d["D1_name_echo"], d["D2_no_catalogs_examined"],
                 d["D3_title_match"], d["n_rows"],
                 (",".join(d["served_instead"])[:40] or d["vizier_error"] or "")),
              flush=True)
    return verdict, url, txt, d


def fetch_validated(source, want_title=(), maxrows="unlimited"):
    """Fetch a catalogue for USE.  Raises unless every detector passes."""
    url = build(source, maxrows=maxrows)
    code, txt = fetch(url, max_bytes=1 << 28)
    ok, d = validate(source, txt, want_title)
    d["truncated"] = (code == 206)
    if d["truncated"]:
        raise ValidationFailure(
            "%s: the payload for %s was truncated; refusing to USE a partial "
            "catalogue (probing is fine, ingesting is not)" % (RULE, source))
    if not ok:
        raise ValidationFailure(
            "%s FAILED for %s: D1=%s D2=%s D3=%s error=%s served=%s"
            % (RULE, source, d["D1_name_echo"], d["D2_no_catalogs_examined"],
               d["D3_title_match"], d["vizier_error"], d["served_instead"]))
    return txt, d


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
