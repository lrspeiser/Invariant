"""guard.py -- this lane may open shear, but only for the OPEN half.

`goldcluster/guard.py` forbids opening any shape at all; that is why this is a
separate lane. Here the constraint is narrower and sharper: shapes are the whole
point, but a single query against a SEALED cluster spends the programme's only
confirmation set, and that cannot be undone.

So the sealed set is checked twice on every cluster -- once when the target list
is built, once immediately before the archive call -- and every refusal is
ledgered. The check is by IDENTITY, against `holdout/loader.py`, because the
sealed half is rows in a shared archive with no filename to match on.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
if WELLNET not in sys.path:
    sys.path.insert(0, WELLNET)

from holdout import loader                                  # noqa: E402

LEDGER = os.path.join(HERE, "refusals.jsonl")


class SealedClusterQueried(RuntimeError):
    """A sealed cluster was about to be sent to the archive."""


def _ledger(**kw):
    rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rec.update(kw)
    io.open(LEDGER, "a", newline="\n", encoding="utf-8").write(
        json.dumps(rec) + "\n")


def check_target(name):
    """Raise if `name` is in the sealed half.  Called before every query."""
    if loader.is_sealed(name):
        _ledger(event="refused", cluster=name,
                why="sealed half; querying it would spend the confirmation set")
        raise SealedClusterQueried(
            "%s is in the SEALED half. Extracting its shear would turn the "
            "programme's only confirmation set into validation. If this is "
            "deliberate, it needs a one-shot token from holdout.loader and a "
            "committed pre-registration -- not a call from an extraction loop."
            % name)
    return True


def assert_all_open(names, context=""):
    """Check a whole target list at once, before any query is sent."""
    loader.verify()
    sealed = set(loader.sealed_names())
    hit = sorted(set(names) & sealed)
    if hit:
        _ledger(event="refused_batch", n=len(hit), examples=hit[:5],
                context=context)
        raise SealedClusterQueried(
            "%s: %d sealed cluster(s) in the target list, e.g. %s"
            % (context or "target list", len(hit), hit[:3]))
    return True


def summary():
    n = 0
    if os.path.exists(LEDGER):
        n = sum(1 for l in io.open(LEDGER, encoding="utf-8") if l.strip())
    return dict(lane="clustershear", refusals_ledgered=n,
                sealed_clusters=len(loader.sealed_names()),
                open_clusters=len(loader.open_names()),
                seal_intact=loader.status()["intact"])
