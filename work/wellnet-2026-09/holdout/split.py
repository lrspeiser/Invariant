"""split.py -- split the gold-cluster pool and seal half of it, before any shear is read.

Run BO established 605 clusters with a public raw weak-lensing channel and read
NOT ONE SHAPE from any of them (see ../goldcluster/spent.json).  That is the only
moment at which a confirmation set can be created: once a pattern search has seen
a cluster, testing the pattern on that cluster proves nothing.  KiDS and the wide
binaries were lost exactly that way in round 1.

So this module splits the pool in half and seals one half.  It is deliberately
boring and deliberately irreversible-by-accident:

  DETERMINISTIC   the assignment is a keyed hash of the cluster's own identity,
                  not a shuffle.  There is no random state to re-roll, and
                  re-running reproduces the same split exactly.

  STRATIFIED      strata are terciles of redshift x X-ray counts x shear depth,
                  so neither half is systematically nearer, brighter or better
                  covered.  Within a stratum the hash decides.

  COMMITTED       the pool digest, the salt and the resulting sealed-set digest
                  are written into the split file and verified on every load.
                  A later edit of the sealed list changes the digest and
                  `verify()` fails.  A seal you can quietly re-roll is not a seal.

WHAT IS AND IS NOT REVEALED.  Sealing is about not SCORING, not about hiding.
The sealed clusters' identities, positions, redshifts and X-ray counts are
public catalogue metadata and stay readable -- a future lane has to know which
clusters to avoid.  What must never be read for a sealed cluster is the OUTCOME:
its shear.  `loader.py` enforces that distinction.

    python split.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
POOL = os.path.join(WELLNET, "goldcluster", "ranking.json")
OUT = os.path.join(HERE, "holdout_split.json")

#: Declared BEFORE the split is computed, and never changed.  Changing it
#: re-rolls the assignment, which is precisely what a seal must prevent, so
#: `verify()` treats a salt change as a broken seal.
SALT = "wellnet-goldcluster-2026-09-07"

#: the fraction sealed
SEAL_FRACTION = 0.5

#: fields that may be read for a SEALED cluster: public catalogue metadata,
#: none of which is the lensing outcome
OPEN_FIELDS = ("name", "ra", "dec", "z", "zType", "cts500", "kt")


def _tercile(values):
    """Return a function mapping a value to 0/1/2 by tercile of `values`."""
    s = sorted(values)
    if not s:
        return lambda v: 0
    a = s[len(s) // 3]
    b = s[2 * len(s) // 3]
    return lambda v: 0 if v < a else (1 if v < b else 2)


def _digest(names):
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _assign(name, stratum):
    """Deterministic half-assignment: the low bit of a keyed hash."""
    h = hashlib.sha256(("%s|%s|%s" % (SALT, stratum, name)).encode("utf-8"))
    return h.digest()[0] & 1          # 0 -> open, 1 -> sealed


def build():
    rank = json.load(io.open(POOL, encoding="utf-8"))
    pool = [c for c in rank["clusters"]
            if c["channels"]["C4_weak_lensing"] == "PUBLIC"]
    if not pool:
        raise SystemExit("no clusters with a public weak-lensing channel in %s" % POOL)

    tz = _tercile([c["z"] for c in pool])
    tc = _tercile([c["cts500"] for c in pool])
    ts = _tercile([c["n_shear_sources"] for c in pool])

    # Assign by keyed hash within each stratum, then repair the balance
    # deterministically so the two halves are within one cluster of each other
    # in EVERY stratum -- a hash alone drifts by a few per cell.
    strata = {}
    for c in pool:
        key = (tz(c["z"]), tc(c["cts500"]), ts(c["n_shear_sources"]))
        strata.setdefault(key, []).append(c)

    sealed, open_ = [], []
    for key, members in sorted(strata.items()):
        members.sort(key=lambda c: c["name"])          # stable order
        want = int(round(len(members) * SEAL_FRACTION))
        # rank inside the stratum by the same keyed hash, take the first `want`
        ordered = sorted(members,
                         key=lambda c: hashlib.sha256(
                             ("%s|%s|%s" % (SALT, key, c["name"])).encode()).hexdigest())
        sealed.extend(ordered[:want])
        open_.extend(ordered[want:])

    return pool, sealed, open_, strata


def _balance(sealed, open_):
    """Report the two halves side by side.  Uses catalogue metadata only."""
    def stats(rows, field):
        v = [r[field] for r in rows if r.get(field) is not None]
        return dict(n=len(v), median=round(statistics.median(v), 4) if v else None,
                    lo=round(min(v), 4) if v else None,
                    hi=round(max(v), 4) if v else None)
    out = {}
    for field in ("z", "cts500", "n_shear_sources", "n_spec_members"):
        out[field] = dict(sealed=stats(sealed, field), open=stats(open_, field))
    out["n_with_spec_z"] = dict(
        sealed=sum(1 for c in sealed if "spec" in (c["zType"] or "")),
        open=sum(1 for c in open_ if "spec" in (c["zType"] or "")))
    out["n_with_kT"] = dict(
        sealed=sum(1 for c in sealed if c["kt"]),
        open=sum(1 for c in open_ if c["kt"]))
    return out


def main():
    pool, sealed, open_, strata = build()
    rec = dict(
        rule="holdout_seal v2",
        created_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                __import__("time").gmtime()),
        source="work/wellnet-2026-09/goldcluster/ranking.json",
        provenance=("Run BO probed these clusters for COVERAGE only and read no "
                    "shape, response or per-source redshift from any of them. The "
                    "split is therefore made before any outcome was observed, "
                    "which is what makes the sealed half a confirmation set "
                    "rather than a validation set."),
        salt=SALT,
        seal_fraction=SEAL_FRACTION,
        stratification="terciles of z x CTS500 x n_shear_sources (27 cells)",
        assignment="keyed sha256 rank within each stratum; no random state",
        n_pool=len(pool),
        n_sealed=len(sealed),
        n_open=len(open_),
        n_strata=len(strata),
        pool_digest=_digest([c["name"] for c in pool]),
        sealed_digest=_digest([c["name"] for c in sealed]),
        open_digest=_digest([c["name"] for c in open_]),
        open_fields=list(OPEN_FIELDS),
        balance=_balance(sealed, open_),
        sealed=sorted(c["name"] for c in sealed),
        open=sorted(c["name"] for c in open_),
    )
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(rec, indent=1) + "\n")
    print("pool=%d  sealed=%d  open=%d  strata=%d"
          % (rec["n_pool"], rec["n_sealed"], rec["n_open"], rec["n_strata"]))
    print("sealed_digest=%s" % rec["sealed_digest"][:16])
    for f, d in rec["balance"].items():
        if isinstance(d.get("sealed"), dict):
            print("  %-16s sealed median %-10s open median %-10s"
                  % (f, d["sealed"]["median"], d["open"]["median"]))
        else:
            print("  %-16s sealed %-10s open %-10s" % (f, d["sealed"], d["open"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
