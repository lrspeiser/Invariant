"""
Job 1, precisely: who reaches `_cluster_profile`, and who repeats its bug.

The first pass over-reported. Two corrections, both found by reading the hits:

  * "reads X-COP" was decided by the literal string `d["xcop"]`, which missed
    `p03/p04/p05`: they call `b.confound(...)`, `b.score(...)` or iterate
    `b.d`, all of which consume X-COP.  Any call into the Bench scoring surface
    counts.
  * "re-implements the clamp" was decided by the file mentioning RW_X, T_X and
    np.interp anywhere.  That caught files whose np.interp is on a stellar-mass
    profile with explicit `left=`/`right=`.  Now every `np.interp` call is
    parsed with `ast`, and a call counts only if its own source segment names a
    temperature and carries NEITHER `left` NOR `right`.

This lane's own files are excluded; they are the audit, not a consumer.
"""
from __future__ import annotations

import ast
import json
import os

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "work/gravity-wells-2026-09/invariant_bench.py"
SELF = "work/wellnet-2026-09/tempclamp/"

SCORING_SURFACE = (".score(", ".confound(", ".summary(", ".caveats(",
                   'd["xcop"]', "_xcop(", "b.d[", ".d.items()", ".d.values()")
TEMP_NAMES = ("t_x", "kt", "temperature", "temp", "tx", "kw", "t500")


def interp_calls(src):
    """Every np.interp call, with its source segment and keyword names."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else "")
        if name != "interp":
            continue
        seg = ast.get_source_segment(src, node) or ""
        kws = {k.arg for k in node.keywords}
        args = [ast.get_source_segment(src, a) or "" for a in node.args]
        out.append((seg, kws, node.lineno, args))
    return out


def _has_temp(s):
    return any(t in (s or "").lower() for t in TEMP_NAMES)


def classify(args):
    """Which way round is this interpolation?

    `same_direction`  fp is a temperature: the coarse T profile is being
                      evaluated on a finer grid, exactly the bench's bug.
    `onto_T_grid`     x is a temperature radius: something else is being
                      evaluated ON the T grid.  Same clamping mechanism, but a
                      different exposure, and NOT measured by this audit.
    """
    if len(args) >= 3 and _has_temp(args[2]) and not _has_temp(args[0]):
        return "same_direction"
    if args and _has_temp(args[0]):
        return "onto_T_grid"
    return "other"


def main():
    direct, importers, reimpl, candidates = [], [], [], []
    skip = ("work\\private", "work/private", "site-packages", "__pycache__",
            "node_modules", ".venv")
    for root in ("work", "src", "tests", "scripts", "configs", "docs"):
        base = ROOT + root
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            if any(s in dp for s in skip):
                dn[:] = []
                continue
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                if rel == BENCH or rel.startswith(SELF):
                    continue
                try:
                    src = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if "_cluster_profile(" in src:
                    direct.append(rel)
                if ("from invariant_bench import" in src
                        or "import invariant_bench" in src):
                    hits = [s for s in SCORING_SURFACE if s in src]
                    importers.append(dict(path=rel, reads_xcop=bool(hits),
                                          via=hits))
                # own clamp?
                same, onto, guarded = [], [], []
                for seg, kws, ln, args in interp_calls(src):
                    if not _has_temp(seg):
                        continue
                    if "left" in kws or "right" in kws:
                        guarded.append(ln)
                        continue
                    kind = classify(args)
                    if kind == "same_direction":
                        same.append(ln)
                    elif kind == "onto_T_grid":
                        onto.append(ln)
                if same or onto:
                    reimpl.append(dict(path=rel, same_direction=sorted(same),
                                       onto_T_grid=sorted(onto),
                                       n_guarded=len(guarded)))
                elif guarded:
                    candidates.append(dict(path=rel, lines=guarded))

    direct.sort()
    importers.sort(key=lambda q: q["path"])
    reimpl.sort(key=lambda q: q["path"])
    candidates.sort(key=lambda q: q["path"])

    print("DIRECT callers of _cluster_profile (outside the bench and this lane):",
          len(direct))
    for p in direct:
        print("   ", p)
    print(f"\nIMPORTERS of Bench: {len(importers)}, "
          f"of which {sum(q['reads_xcop'] for q in importers)} consume X-COP")
    for q in importers:
        print(f"   {q['path']:<62}"
              f"{('READS xcop via ' + ', '.join(q['via'][:3])) if q['reads_xcop'] else 'no consumption found'}")
    nsame = sum(1 for q in reimpl if q["same_direction"])
    print(f"\nUNGUARDED temperature interps: {len(reimpl)} files, {nsame} of "
          f"them in the BENCH'S OWN DIRECTION (coarse T on a finer grid)")
    for q in reimpl:
        tag = (("SAME BUG lines " + str(q["same_direction"]))
               if q["same_direction"]
               else "onto-T-grid only, exposure not measured here")
        extra = (f"  [+onto {q['onto_T_grid']}]"
                 if q["same_direction"] and q["onto_T_grid"] else "")
        print(f"   {q['path']:<62} {tag}{extra}")
    print(f"\nGUARDED temperature interps (left=/right= present, NOT the bug):"
          f" {len(candidates)}")
    for q in candidates:
        print(f"   {q['path']:<62} lines {q['lines']}")

    out = dict(direct_callers=direct, importers=importers,
               reimplementations=reimpl, guarded=candidates,
               n_direct=len(direct), n_importers=len(importers),
               n_reads_xcop=sum(q["reads_xcop"] for q in importers),
               n_reimpl=len(reimpl), n_guarded=len(candidates),
               n_same_direction=sum(1 for q in reimpl if q["same_direction"]))
    res_path = os.path.join(HERE, "results.json")
    res = json.load(open(res_path, encoding="utf-8"))
    res["job1"] = dict(res.get("job1", {}), **out)
    json.dump(res, open(res_path, "w", encoding="utf-8"), indent=1, default=float)
    print("\n   results.json job1 replaced")
    return out


if __name__ == "__main__":
    main()
