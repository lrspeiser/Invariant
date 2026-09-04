"""Audit every data file in the lane against its sibling .manifest.json.

Checks, per file:
  * a manifest exists;
  * the recorded sha256 matches the bytes actually on disk;
  * the recorded byte size matches;
  * for delimited text files, the recorded row_count matches a recount.

Row-count re-verification is the specific defence against the silent-truncation
failure mode (a LaTeX table split across two table* environments returning 59 of
100 rows with no error).

Writes scripts/../validation_report.json and prints a summary.
"""
import hashlib
import json
import os
import sys

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {"scripts", "__pycache__", ".git"}
SKIP_EXT = {".py", ".pyc", ".manifest", ".json.tmp"}
TEXT_EXT = {".tsv", ".csv", ".dat", ".cat", ".txt", ".ascii"}
# Files that are documentation / raw upstream envelopes, not row-structured tables.
NO_ROWCOUNT_HINTS = ("readme", "ReadMe", "_raw", "raw/", "\\raw\\", ".reg", "probe_")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_text_rows(path, manifest_columns=None):
    """Count DATA rows: non-blank, non-comment lines, excluding a column-name
    header line if one is present.  Returns (n_rows, mode).

    The header is identified by matching the first payload line against the
    column names recorded in the manifest -- not by guessing -- so a genuine
    first data row is never silently discarded."""
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8", "replace")
    lines = txt.splitlines()

    # VizieR asu-tsv payload: data starts after the last '---' separator line.
    seps = [i for i, l in enumerate(lines) if l.startswith("---")]
    if seps and any(l.startswith("#") for l in lines[:5]):
        i = seps[-1]
        return sum(1 for l in lines[i + 1:]
                   if l.strip() and not l.startswith("#")), "vizier"

    payload = [l for l in lines if l.strip() and not l.lstrip().startswith(("#", "%"))]
    mode = "plain"
    if payload and manifest_columns:
        names = [c.get("name", "") if isinstance(c, dict) else str(c)
                 for c in manifest_columns]
        first = [t.strip() for t in payload[0].replace(",", "\t").split("\t")]
        first = [t for t in first if t != ""]
        norm = lambda s: str(s).strip().lower().lstrip("#").strip()
        if names and first:
            overlap = sum(1 for a, b in zip(map(norm, first), map(norm, names)) if a == b)
            if overlap >= max(2, int(0.6 * min(len(first), len(names)))):
                payload = payload[1:]
                mode = "plain-header-excluded"
    return len(payload), mode


def walk_data_files():
    for root, dirs, files in os.walk(LANE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".manifest.json"):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            if fn in ("INVENTORY.md", "REPORT.md", "validation_report.json"):
                continue
            if fn.endswith(".md"):
                continue
            yield os.path.join(root, fn)


RAW_REFS = {}


def index_raw_references():
    """basename of a raw upstream file -> list of manifests that cite it."""
    import glob
    for m in glob.glob(os.path.join(LANE, "**", "*.manifest.json"), recursive=True):
        try:
            d = json.load(open(m, encoding="utf-8"))
        except Exception:
            continue
        rel = os.path.relpath(m, LANE).replace("\\", "/")
        for k in ("raw_response_file", "raw_file", "source_file_within_archive",
                  "raw", "raw_files", "tar_member"):
            v = d.get(k)
            vs = v if isinstance(v, list) else ([v] if isinstance(v, str) else [])
            for x in vs:
                if isinstance(x, str):
                    RAW_REFS.setdefault(os.path.basename(x), []).append(rel)


def main():
    index_raw_references()
    results = []
    n_ok = n_nomanifest = n_bad = 0
    for path in sorted(walk_data_files()):
        rel = os.path.relpath(path, LANE).replace("\\", "/")
        mpath = path + ".manifest.json"
        rec = {"file": rel, "bytes_on_disk": os.path.getsize(path)}
        if not os.path.exists(mpath):
            # Most of these are raw upstream envelopes kept beside a cleaned
            # file that cites them.  Checksum them anyway so provenance is
            # complete, and record whether some manifest actually references
            # them.
            rec["status"] = "NO_MANIFEST"
            rec["sha256"] = sha256(path)
            rec["referenced_by"] = RAW_REFS.get(os.path.basename(path), [])
            n_nomanifest += 1
            results.append(rec)
            continue
        try:
            m = json.load(open(mpath, encoding="utf-8"))
        except Exception as e:
            rec["status"] = "MANIFEST_UNPARSEABLE"
            rec["error"] = repr(e)
            n_bad += 1
            results.append(rec)
            continue

        problems = []
        actual = sha256(path)
        if m.get("sha256") and m["sha256"] != actual:
            problems.append("sha256 mismatch: manifest %s vs actual %s"
                            % (m["sha256"][:12], actual[:12]))
        if m.get("bytes") is not None and m["bytes"] != rec["bytes_on_disk"]:
            problems.append("byte size mismatch: manifest %s vs actual %s"
                            % (m["bytes"], rec["bytes_on_disk"]))

        ext = os.path.splitext(path)[1].lower()
        skip_rows = any(h.lower() in rel.lower() for h in NO_ROWCOUNT_HINTS)
        if ext in TEXT_EXT and m.get("row_count") is not None and not skip_rows:
            try:
                n, mode = count_text_rows(path, m.get("columns"))
                rec["recounted_rows"] = n
                rec["recount_mode"] = mode
                if n != m["row_count"]:
                    problems.append("row_count mismatch: manifest %s vs recount %s (%s)"
                                    % (m["row_count"], n, mode))
            except Exception as e:
                problems.append("recount failed: %r" % (e,))

        rec["sha256"] = actual
        rec["manifest_row_count"] = m.get("row_count")
        rec["cluster"] = m.get("cluster")
        rec["product"] = m.get("product")
        rec["source_url"] = m.get("source_url")
        if problems:
            rec["status"] = "PROBLEM"
            rec["problems"] = problems
            n_bad += 1
        else:
            rec["status"] = "OK"
            n_ok += 1
        results.append(rec)

    out = {"lane": LANE.replace("\\", "/"),
           "n_files": len(results), "n_ok": n_ok,
           "n_missing_manifest": n_nomanifest, "n_problem": n_bad,
           "files": results}
    dest = os.path.join(LANE, "validation_report.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("files=%d ok=%d no_manifest=%d problem=%d" % (len(results), n_ok, n_nomanifest, n_bad))
    for r in results:
        if r["status"] != "OK":
            print(" ", r["status"], r["file"], r.get("problems", r.get("error", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
