"""Download arXiv e-print source tarballs, keep the raw archive, extract to a
per-paper directory, and write a manifest for the archive.

Programme failure mode guarded against: a LaTeX table split across two
`table*` environments silently returned 59 of 100 rows.  This module only
DOWNLOADS and INVENTORIES; every table transcription that follows must count
rows and cross-check against the paper's stated sample size.
"""
import hashlib
import json
import os
import tarfile
import gzip
import io
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.join(HERE, "eprints")
UA = {"User-Agent": "wellnet-2026-09 polar-ring acquisition (research; contact leonard@horizon3.net)"}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_eprint(arxiv_id, label, note=""):
    os.makedirs(EP, exist_ok=True)
    safe = arxiv_id.replace("/", "_")
    url = "https://arxiv.org/e-print/" + arxiv_id
    dest_dir = os.path.join(EP, "%s_%s" % (safe, label))
    arc_path = os.path.join(EP, "%s.tar.gz" % safe)
    if os.path.exists(arc_path) and os.path.isdir(dest_dir):
        return dest_dir, "cached"
    r = requests.get(url, headers=UA, timeout=300)
    r.raise_for_status()
    blob = r.content
    with open(arc_path, "wb") as f:
        f.write(blob)
    os.makedirs(dest_dir, exist_ok=True)
    kind = "unknown"
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
        names = tf.getnames()
        for m in tf.getmembers():
            if m.isfile() and not m.name.startswith("/") and ".." not in m.name:
                tf.extract(m, dest_dir)
        kind = "tar"
    except tarfile.ReadError:
        try:
            dec = gzip.decompress(blob)
            with open(os.path.join(dest_dir, "%s.tex" % safe), "wb") as f:
                f.write(dec)
            names = ["%s.tex" % safe]
            kind = "gzip-single-tex"
        except Exception:
            with open(os.path.join(dest_dir, "%s.bin" % safe), "wb") as f:
                f.write(blob)
            names = ["%s.bin" % safe]
            kind = "raw"
    with open(arc_path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "file": os.path.basename(arc_path),
            "arxiv_id": arxiv_id,
            "label": label,
            "source_url": url,
            "retrieved_utc": utcnow(),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "archive_kind": kind,
            "members": names,
            "extracted_to": os.path.basename(dest_dir),
            "query": "GET " + url,
            "note": note,
            "extraction": "Raw arXiv e-print source archive, unmodified. Extracted in place for table transcription.",
        }, f, indent=2)
    return dest_dir, kind


if __name__ == "__main__":
    PAPERS = json.load(open(os.path.join(HERE, "_papers.json"), encoding="utf-8"))
    for p in PAPERS:
        try:
            d, k = fetch_eprint(p["id"], p["label"], p.get("note", ""))
            n = sum(len(fs) for _, _, fs in os.walk(d))
            print("OK   %-16s %-34s %-16s files=%d" % (p["id"], p["label"], k, n))
        except Exception as e:
            print("FAIL %-16s %-34s %s: %s" % (p["id"], p["label"], type(e).__name__, e))
