"""loader.py -- the only sanctioned route to the gold-cluster pool.

`holdout_seal` v2 requires: no data-loading side effects in constructors; access
needs a one-shot token; every open appends to a ledger; CI fails if an ordinary
test touches a sealed product.  The existing implementation of that rule
(`universes/provenance.py`) seals by PATH TOKEN -- it works for KiDS because KiDS
lives in files whose names contain "kids".  That will not work here.  The sealed
half of the gold-cluster pool is 304 ROWS IN A SHARED PUBLIC ARCHIVE; there is no
filename to match on.  So the seal is enforced by IDENTITY, here.

The distinction this module draws, and it is the whole design:

    METADATA is open for every cluster, sealed or not.  Name, position,
    redshift, X-ray counts and temperature are public catalogue values, and a
    lane has to be able to read them in order to know which clusters to stay
    away from.  Reading them scores nothing.

    OUTCOME is sealed.  For a sealed cluster, anything derived from its shear --
    a shape, a response, a tangential profile -- requires a one-shot token, and
    taking one is a recorded, deliberate act that ends the holdout for that
    cluster.  There is no way to do it by accident.

Typical use by a future extraction lane:

    from holdout import loader
    loader.verify()                       # the split has not drifted
    targets = loader.open_pool()          # the 301 clusters you may work on
    loader.assert_not_sealed(names)       # belt and braces before a query

And, exactly once, at the end of the programme:

    tok = loader.request_token("Run BX: final test of law L, pre-registered in "
                               "runs/.../PREREGISTRATION.md", requester="BX")
    sealed = loader.open_sealed(tok)      # consumes the token, ledgers the open
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "holdout_split.json")
LEDGER = os.path.join(HERE, "access_ledger.jsonl")
TOKENS = os.path.join(HERE, "tokens.jsonl")

_SPLIT_CACHE = None                       # loaded lazily; NOT at import time


class SealBroken(RuntimeError):
    """The split file no longer matches its own committed digests."""


class SealedClusterTouched(RuntimeError):
    """A sealed cluster's outcome was requested without a valid token."""


class TokenInvalid(RuntimeError):
    """The token is unknown, already spent, or for a different purpose."""


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest(names):
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ledger(event, **kw):
    rec = dict(utc=_utc(), event=event)
    rec.update(kw)
    io.open(LEDGER, "a", newline="\n", encoding="utf-8").write(
        json.dumps(rec) + "\n")
    return rec


def load():
    """Read the split.  No side effects beyond the read; safe to call anywhere."""
    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        _SPLIT_CACHE = json.load(io.open(SPLIT, encoding="utf-8"))
    return _SPLIT_CACHE


def verify():
    """Recompute the committed digests.  Raises SealBroken on any drift.

    This is what makes the seal a seal rather than a note. Re-rolling the split,
    moving one cluster from sealed to open, or editing the salt all change a
    digest and are caught here.
    """
    d = load()
    checks = {
        "sealed_digest": (_digest(d["sealed"]), d["sealed_digest"]),
        "open_digest": (_digest(d["open"]), d["open_digest"]),
        "pool_digest": (_digest(list(d["sealed"]) + list(d["open"])),
                        d["pool_digest"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SealBroken(
            "the holdout split does not match its committed digests: %s. The "
            "sealed set must not change after Run BO. If this is a deliberate "
            "re-split, it is a NEW holdout and the old one is spent."
            % {k: "recomputed %s != committed %s" % (a[:12], b[:12])
               for k, (a, b) in bad.items()})
    if len(d["sealed"]) + len(d["open"]) != d["n_pool"]:
        raise SealBroken("sealed + open != pool")
    return True


def sealed_names():
    """The sealed identities.  Open on purpose: a lane must know what to avoid."""
    return list(load()["sealed"])


def open_names():
    return list(load()["open"])


def is_sealed(name):
    return name in set(load()["sealed"])


def open_pool():
    """Full records for the OPEN half -- the clusters you may search on."""
    verify()
    names = set(load()["open"])
    rank = json.load(io.open(os.path.join(os.path.dirname(HERE), "goldcluster",
                                          "ranking.json"), encoding="utf-8"))
    return [c for c in rank["clusters"] if c["name"] in names]


def sealed_metadata():
    """Public catalogue metadata for the sealed half.  No outcome, no token.

    Provided so a lane can build a control sample, check a footprint or make a
    figure without touching the holdout. The fields are exactly those the split
    declared as open.
    """
    verify()
    d = load()
    names = set(d["sealed"])
    fields = d["open_fields"]
    rank = json.load(io.open(os.path.join(os.path.dirname(HERE), "goldcluster",
                                          "ranking.json"), encoding="utf-8"))
    return [{k: c.get(k) for k in fields}
            for c in rank["clusters"] if c["name"] in names]


def assert_not_sealed(names, context=""):
    """Refuse a target list that contains a sealed cluster."""
    sealed = set(load()["sealed"])
    hit = sorted(set(names) & sealed)
    if hit:
        raise SealedClusterTouched(
            "%s%d sealed cluster(s) in this target list, e.g. %s. The sealed "
            "half is the programme's only confirmation set; scoring it here "
            "would turn it into validation, which is how KiDS was lost."
            % (context + ": " if context else "", len(hit), hit[:3]))
    return True


def request_token(reason, requester):
    """Mint a one-shot token.  Recorded before it can be used.

    Taking a token is the deliberate act. It should follow a written
    pre-registration of exactly what will be tested, because the answer can only
    be obtained once.
    """
    if not reason or len(reason) < 40:
        raise TokenInvalid(
            "a token needs a specific written reason (>=40 chars) naming the "
            "run and the pre-registered test; got %r" % (reason,))
    tok = secrets.token_hex(16)
    rec = dict(token=tok, reason=reason, requester=requester,
               issued_utc=_utc(), spent=False,
               sealed_digest=load()["sealed_digest"])
    io.open(TOKENS, "a", newline="\n", encoding="utf-8").write(
        json.dumps(rec) + "\n")
    _ledger("token_issued", token=tok, requester=requester, reason=reason)
    return tok


def _tokens():
    if not os.path.exists(TOKENS):
        return []
    return [json.loads(l) for l in io.open(TOKENS, encoding="utf-8") if l.strip()]


def open_sealed(token):
    """Consume a one-shot token and return the sealed half's full records.

    This ENDS the holdout. After this the sealed clusters are Scored-eligible
    and the programme has no confirmation set again until a new one is created
    from data that has never been searched.
    """
    verify()
    toks = _tokens()
    match = [t for t in toks if t["token"] == token]
    if not match:
        _ledger("token_rejected", token=str(token)[:8], why="unknown")
        raise TokenInvalid("unknown token")
    t = match[-1]
    if t["spent"]:
        _ledger("token_rejected", token=token[:8], why="already spent")
        raise TokenInvalid(
            "this token has already been spent (%s). A one-shot token is one "
            "shot; mint a new one only for a newly pre-registered test."
            % t.get("spent_utc"))
    if t["sealed_digest"] != load()["sealed_digest"]:
        raise SealBroken("token was issued against a different sealed set")

    # rewrite the token file with this one marked spent
    t["spent"] = True
    t["spent_utc"] = _utc()
    out = [x if x["token"] != token else t for x in toks]
    io.open(TOKENS, "w", newline="\n", encoding="utf-8").write(
        "".join(json.dumps(x) + "\n" for x in out))

    names = set(load()["sealed"])
    rank = json.load(io.open(os.path.join(os.path.dirname(HERE), "goldcluster",
                                          "ranking.json"), encoding="utf-8"))
    recs = [c for c in rank["clusters"] if c["name"] in names]
    _ledger("sealed_opened", token=token[:8], requester=t["requester"],
            reason=t["reason"], n_clusters=len(recs))
    return recs


def status():
    d = load()
    toks = _tokens()
    return dict(rule=d["rule"], created_utc=d["created_utc"],
                n_sealed=d["n_sealed"], n_open=d["n_open"],
                sealed_digest=d["sealed_digest"],
                tokens_issued=len(toks),
                tokens_spent=sum(1 for t in toks if t["spent"]),
                intact=all(not t["spent"] for t in toks))
