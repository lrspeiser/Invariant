"""Verify every manifest in the lane against the file it describes.

Checks SHA-256 and byte size for every <name>.manifest.json that names a `file`,
plus the MAPS directory manifest which describes a whole directory.
"""
import hashlib
import json
import os

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    problems, checked, seen = [], 0, 0
    for root, dirs, files in os.walk(LANE):
        for f in files:
            if not f.endswith(".manifest.json"):
                continue
            seen += 1
            p = os.path.join(root, f)
            try:
                m = json.load(open(p, encoding="utf-8"))
            except Exception as e:
                problems.append((p, "unreadable: %s" % e))
                continue
            if f == "maps.manifest.json":
                n = len([x for x in os.listdir(root) if x.endswith(".fits.gz")])
                if n != m["n_files"]:
                    problems.append((p, "maps count %d != manifest %d" % (n, m["n_files"])))
                else:
                    checked += 1
                continue
            name = m.get("file")
            if not name:
                continue
            tgt = os.path.join(root, name)
            if not os.path.exists(tgt):
                problems.append((p, "target missing: %s" % name))
                continue
            checked += 1
            if "bytes" in m and m["bytes"] != os.path.getsize(tgt):
                problems.append((p, "SIZE mismatch"))
            elif "sha256" in m and m["sha256"] != sha256(tgt):
                problems.append((p, "SHA-256 mismatch"))
    print("manifests seen: %d; targets verified: %d" % (seen, checked))
    if problems:
        print("PROBLEMS (%d):" % len(problems))
        for p, why in problems:
            print("   %-90s %s" % (os.path.relpath(p, LANE), why))
    else:
        print("PROBLEMS: none")


if __name__ == "__main__":
    main()
