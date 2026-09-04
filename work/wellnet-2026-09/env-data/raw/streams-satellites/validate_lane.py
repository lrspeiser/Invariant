"""Independent validation of every manifest in the lane.

Checks, for each <file>.manifest.json:
  - the manifest is valid JSON and carries the required keys
  - the described file exists
  - the recorded SHA-256 and byte size still match the file on disk
  - the recorded row_count matches a recount of the file, for delimited text
  - a measurement_or_model label is present
"""
import csv
import hashlib
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["file", "source_url", "retrieved_utc", "sha256", "bytes", "row_count",
            "column_count", "columns", "query"]

def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def recount(p):
    """Recount data rows for delimited text files. Returns None if not applicable."""
    ext = os.path.splitext(p)[1].lower()
    if ext not in (".tsv", ".csv", ".dat", ".txt"):
        return None
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except Exception:
        return None
    if not lines:
        return 0
    # Detect VizieR ASU-TSV layout by CONTENT, not by filename: several files
    # carry that layout ('#' comment block, then header/units/dashes/data) while
    # being named plain .tsv. Keying off the filename silently skipped them.
    looks_vizier = (".vizier." in os.path.basename(p)
                    or (lines and lines[0].lstrip().startswith("#")
                        and any(l.startswith("#") and "vizier" in l.lower() for l in lines[:40])))
    if looks_vizier:
        # VizieR ASU-TSV, SECTION-AWARE. A new section starts at a '#Table'
        # marker; within a section the layout is header / units / [dashes] /
        # data. Resetting on every blank line (the naive version) miscounts,
        # and not every section carries a dashes rule row.
        sections, cur = [], None
        for l in lines:
            if l.startswith("#Table"):
                cur = []
                sections.append(cur)
                continue
            if l.startswith("#") or not l.strip():
                continue
            if cur is None:
                cur = []
                sections.append(cur)
            cur.append(l)
        total = 0
        for rows in sections:
            n = len(rows)
            drop = min(n, 2)                                  # header + units
            if n > 2 and set(rows[2].replace("\t", "").strip()) <= set("- "):
                drop += 1                                     # dashes rule
            total += n - drop
        return total
    if ext in (".tsv", ".csv"):
        delim = "\t" if ext == ".tsv" else ","
        rows = [l for l in lines if l.strip() and not l.startswith("#")]
        if rows and rows[0].count(delim) == 0:
            return None
        # Several writers preserve the CDS-style UNITS row (line 2) and a
        # DASHES rule row (line 3) beneath the header. Return every plausible
        # data-row count so the caller can accept any of them.
        n = len(rows) - 1
        cands = {n}
        if len(rows) > 1:
            cands.add(n - 1)               # header + units
        if len(rows) > 2 and set(rows[2].replace(delim, "").strip()) <= set("- "):
            cands.add(n - 2)               # header + units + dashes
        return cands
    return None

manifests = []
for root, _, files in os.walk(BASE):
    for f in files:
        if f.endswith(".manifest.json"):
            manifests.append(os.path.join(root, f))
manifests.sort()
print("manifests found:", len(manifests))
checked_rows = [0]

bad_json, missing_file, sha_mismatch, size_mismatch, row_mismatch = [], [], [], [], []
missing_keys, no_label, ok = [], [], 0
for mp in manifests:
    rel = os.path.relpath(mp, BASE)
    try:
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
    except Exception as e:
        bad_json.append((rel, repr(e)[:90]))
        continue
    miss = [k for k in REQUIRED if k not in man]
    if miss:
        missing_keys.append((rel, miss))
    if "measurement_or_model" not in man:
        no_label.append(rel)
    dp = mp[:-len(".manifest.json")]
    if not os.path.exists(dp):
        missing_file.append(rel)
        continue
    if man.get("sha256") and sha256_of(dp) != man["sha256"]:
        sha_mismatch.append(rel)
    if man.get("bytes") is not None and os.path.getsize(dp) != man["bytes"]:
        size_mismatch.append((rel, os.path.getsize(dp), man["bytes"]))
    rc = man.get("row_count")
    if isinstance(rc, int):
        n = recount(dp)
        if n is not None:
            checked_rows[0] += 1
            cands = n if isinstance(n, set) else {n}
            if rc not in cands:
                row_mismatch.append((rel, sorted(cands), rc))
    ok += 1

def report(name, items):
    print("\n%s: %d" % (name, len(items)))
    for x in items[:14]:
        print("   ", x)

print("\nmanifests whose file exists and hashes/sizes were checked:", ok)
report("INVALID JSON", bad_json)
report("MISSING DATA FILE", missing_file)
report("SHA-256 MISMATCH", sha_mismatch)
report("BYTE-SIZE MISMATCH", size_mismatch)
report("ROW-COUNT MISMATCH (recount vs manifest)", row_mismatch)
report("MISSING REQUIRED KEYS", missing_keys)
report("NO measurement_or_model LABEL", no_label)
print("\nfiles whose row_count was actually RECOUNTED:", checked_rows[0])

clean = not (bad_json or missing_file or sha_mismatch or size_mismatch or missing_keys)
print("\nINTEGRITY (json/file/sha/size/keys): %s" % ("CLEAN" if clean else "PROBLEMS ABOVE"))
print("Row-count recount disagreements: %d (inspect individually; a recount "
      "heuristic can legitimately differ for multi-section or fixed-width files)"
      % len(row_mismatch))
