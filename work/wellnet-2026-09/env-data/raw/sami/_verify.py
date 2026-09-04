"""Final integrity pass: for every manifest in this directory, re-hash the file
it describes and re-count its rows.  Run this before trusting anything here.
"""
import hashlib
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    bad, n = [], 0
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".manifest.json"):
            continue
        man = json.load(open(os.path.join(OUT, f)))
        target = f[: -len(".manifest.json")]
        tp = os.path.join(OUT, target)
        if not os.path.exists(tp):
            bad.append("%s: describes missing file %s" % (f, target))
            continue
        n += 1
        if man.get("sha256") and man["sha256"] != sha(tp):
            bad.append("%s: sha256 mismatch" % target)
        if man.get("bytes") and man["bytes"] != os.path.getsize(tp):
            bad.append("%s: byte-size mismatch" % target)
        rc = man.get("row_count")
        if rc and target.endswith((".tsv", ".csv")):
            actual = sum(1 for _ in open(tp, encoding="utf-8", errors="replace")) - 1
            if actual != rc:
                bad.append("%s: %d data lines, manifest says %d" % (target, actual, rc))
        if target.endswith(".vot") and man.get("cleaned_file"):
            cp = os.path.join(OUT, man["cleaned_file"])
            if not os.path.exists(cp):
                bad.append("%s: cleaned file missing" % target)
            elif man.get("cleaned_sha256") and man["cleaned_sha256"] != sha(cp):
                bad.append("%s: cleaned-file sha256 mismatch" % man["cleaned_file"])

    print("checked %d manifests" % n)
    if bad:
        print("FAILURES:")
        for b in bad:
            print("  ", b)
        raise SystemExit(1)
    print("VERIFY PASS: every manifest matches its file on disk (sha256, bytes, rows)")


if __name__ == "__main__":
    main()
