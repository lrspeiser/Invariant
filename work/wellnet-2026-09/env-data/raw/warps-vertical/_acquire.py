"""Acquisition helpers: VizieR TSV pull with assertions, arXiv e-print pull, manifests.

Every downloaded artefact gets a sibling <name>.manifest.json.
VizieR responses are asserted to be REAL TSV with the expected catalogue echoed back.
"""
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import datetime as dt

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
HDRS = {"User-Agent": "wellnet-gravity-acquisition/1.0 (academic research)"}


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_manifest(path, **kw):
    mpath = path + ".manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2)
    print("  manifest ->", os.path.basename(mpath))
    return mpath


# --------------------------------------------------------------------------
# VizieR
# --------------------------------------------------------------------------
def vizier_tsv(cat, outname, expect_min_rows=1, extra=""):
    """Pull a VizieR catalogue/table as TSV. Asserts the response is real TSV.

    VizieR returns HTTP 200 with a generic HTML page for a nonexistent -source=,
    so we assert: (1) not HTML, (2) the catalogue identifier is echoed in the
    '#Name:' / '#INFO' header lines, (3) a data block with >= expect_min_rows.
    """
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
           f"?-source={cat}&-out.all&-out.max=unlimited{extra}")
    r = requests.get(url, headers=HDRS, timeout=300)
    r.raise_for_status()
    txt = r.text
    low = txt[:2000].lower()
    if "<html" in low or "<!doctype" in low:
        raise AssertionError(f"VizieR returned HTML for -source={cat} (catalogue does not exist)")
    # echo the identifier back
    base = cat.split("/table")[0]
    if base.lower() not in txt.lower():
        raise AssertionError(f"VizieR response does not echo catalogue id {base!r}; "
                             f"first 300 chars: {txt[:300]!r}")
    lines = txt.splitlines()
    # data block: after the '---\t---' separator line
    sep_idx = [i for i, l in enumerate(lines) if re.match(r"^-+(\t-+)+\s*$", l)]
    if not sep_idx:
        raise AssertionError(f"No TSV separator line found for {cat}; not a TSV table.\n"
                             + "\n".join(lines[:25]))
    hdr_i = sep_idx[0] - 2
    colnames = lines[hdr_i].split("\t")
    units = lines[hdr_i + 1].split("\t")
    data = [l for l in lines[sep_idx[0] + 1:] if l.strip() and not l.startswith("#")]
    if len(data) < expect_min_rows:
        raise AssertionError(f"{cat}: got {len(data)} data rows, expected >= {expect_min_rows}")
    ncol = len(colnames)
    bad = [i for i, l in enumerate(data) if len(l.split("\t")) != ncol]
    print(f"  {cat}: {len(data)} rows x {ncol} cols "
          f"({len(bad)} rows with wrong column count)")
    path = os.path.join(HERE, outname)
    with open(path, "wb") as f:
        f.write(r.content)
    write_manifest(
        path,
        file=outname,
        source_url=url,
        exact_query=url,
        vizier_catalogue=cat,
        retrieved_utc=utcnow(),
        sha256=sha256_bytes(r.content),
        bytes=len(r.content),
        row_count=len(data),
        column_count=ncol,
        columns=[{"name": c, "unit": (u if u.strip() else "---")}
                 for c, u in zip(colnames, units + [""] * ncol)],
        extraction="Raw, unmodified VizieR asu-tsv response. Header/units block retained verbatim.",
        assertions={
            "response_is_tsv_not_html": True,
            "catalogue_id_echoed_in_response": True,
            "rows_with_wrong_column_count": len(bad),
        },
    )
    return path, len(data), colnames


def cds_readme(cat, outname):
    url = f"https://cdsarc.cds.unistra.fr/ftp/{cat}/ReadMe"
    r = requests.get(url, headers=HDRS, timeout=120)
    r.raise_for_status()
    if "<html" in r.text[:400].lower():
        raise AssertionError(f"ReadMe for {cat} is HTML - catalogue absent")
    if cat.lower() not in r.text[:400].lower():
        raise AssertionError(f"ReadMe does not echo {cat}")
    path = os.path.join(HERE, outname)
    with open(path, "wb") as f:
        f.write(r.content)
    write_manifest(path, file=outname, source_url=url, exact_query=url,
                   cds_catalogue=cat, retrieved_utc=utcnow(),
                   sha256=sha256_bytes(r.content), bytes=len(r.content),
                   row_count=len(r.text.splitlines()), column_count=None,
                   columns=[], extraction="Raw CDS ReadMe, unmodified.",
                   note="Byte-by-byte column description for the catalogue.")
    print(f"  ReadMe {cat}: {len(r.content)} bytes")
    return path


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------
def arxiv_eprint(arxiv_id, outdir_name):
    """Download and unpack an arXiv source tarball into <outdir_name>/."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    r = requests.get(url, headers=HDRS, timeout=300)
    r.raise_for_status()
    raw = r.content
    outdir = os.path.join(HERE, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    tarpath = os.path.join(outdir, f"{arxiv_id.replace('/', '_')}.tar.gz")
    with open(tarpath, "wb") as f:
        f.write(raw)
    names = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # path traversal guard
                safe = m.name.replace("\\", "/").lstrip("/")
                if ".." in safe.split("/"):
                    continue
                dest = os.path.join(outdir, *safe.split("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as g:
                    g.write(tf.extractfile(m).read())
                names.append(safe)
    except tarfile.ReadError:
        # single gzipped .tex
        import gzip
        try:
            body = gzip.decompress(raw)
            dest = os.path.join(outdir, f"{arxiv_id.replace('/', '_')}.tex")
            with open(dest, "wb") as g:
                g.write(body)
            names.append(os.path.basename(dest))
        except Exception as e:
            print("  NOT a tar or gzip:", type(e).__name__, e, "first bytes:", raw[:20])
    write_manifest(tarpath, file=os.path.basename(tarpath), source_url=url,
                   exact_query=url, arxiv_id=arxiv_id, retrieved_utc=utcnow(),
                   sha256=sha256_bytes(raw), bytes=len(raw),
                   row_count=len(names), column_count=None, columns=[],
                   extraction="Raw arXiv e-print source tarball, unmodified; unpacked alongside.",
                   files_in_archive=sorted(names))
    print(f"  arXiv {arxiv_id}: {len(raw)} bytes, {len(names)} files -> {outdir_name}/")
    return outdir, names


def arxiv_search(query, maxr=10):
    url = ("http://export.arxiv.org/api/query?search_query="
           + requests.utils.quote(query) + f"&max_results={maxr}")
    r = requests.get(url, headers=HDRS, timeout=120)
    r.raise_for_status()
    ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", r.text)
    titles = [re.sub(r"\s+", " ", t).strip()
              for t in re.findall(r"<title>(.*?)</title>", r.text, re.S)][1:]
    for i, (a, t) in enumerate(zip(ids, titles)):
        print(f"  [{i}] {a:20s} {t[:100]}")
    return list(zip(ids, titles))
