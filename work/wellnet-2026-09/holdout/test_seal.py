"""test_seal.py -- the seal must actually refuse things.

Run directly, not under pytest (the lane guards raise on .pytest_cache).

    python test_seal.py

T1-T4 are the CI gate: they fail if the split has drifted or if the holdout has
been opened without a recorded pre-registration.  T5-T12 exercise the refusal
paths.  The token machinery is tested END TO END on a temporary fixture, never
on the real seal -- a test that spends the holdout to prove the holdout works
would be a very expensive joke.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import loader

HERE = os.path.dirname(os.path.abspath(__file__))
_results = []


@contextlib.contextmanager
def _fixture():
    """Point the loader at a throwaway split, so no test writes to the real seal.

    Yields the temp directory. Restores the module's real paths and cache on the
    way out, including on failure.
    """
    tmp = tempfile.mkdtemp(prefix="sealtest-")
    saved = (loader.SPLIT, loader.LEDGER, loader.TOKENS, loader._SPLIT_CACHE)
    try:
        names_s = ["FAKE-S-%03d" % i for i in range(4)]
        names_o = ["FAKE-O-%03d" % i for i in range(4)]
        fake = dict(rule="holdout_seal v2", created_utc="1970-01-01T00:00:00Z",
                    n_pool=8, n_sealed=4, n_open=4,
                    sealed=names_s, open=names_o,
                    sealed_digest=loader._digest(names_s),
                    open_digest=loader._digest(names_o),
                    pool_digest=loader._digest(names_s + names_o),
                    open_fields=["name"], balance={})
        loader.SPLIT = os.path.join(tmp, "split.json")
        loader.LEDGER = os.path.join(tmp, "ledger.jsonl")
        loader.TOKENS = os.path.join(tmp, "tokens.jsonl")
        io.open(loader.SPLIT, "w", newline="\n", encoding="utf-8").write(
            json.dumps(fake))
        loader._SPLIT_CACHE = None
        yield tmp
    finally:
        loader.SPLIT, loader.LEDGER, loader.TOKENS, loader._SPLIT_CACHE = saved
        shutil.rmtree(tmp, ignore_errors=True)


def check(name, fn):
    try:
        _results.append((True, name, fn() or ""))
    except AssertionError as e:
        _results.append((False, name, str(e)[:220]))
    except Exception as e:                                      # noqa: BLE001
        _results.append((False, name, "%s: %s" % (type(e).__name__, str(e)[:200])))


# ------------------------------------------------------------------- CI gate
def t1():
    """The committed digests still match the committed lists."""
    loader.verify()
    d = loader.load()
    return "pool=%d sealed=%d open=%d" % (d["n_pool"], d["n_sealed"], d["n_open"])


def t2():
    """THE HOLDOUT IS STILL INTACT -- no token has been spent."""
    s = loader.status()
    assert s["tokens_spent"] == 0, (
        "the sealed half has been OPENED (%d token(s) spent). That is not "
        "necessarily wrong, but it must correspond to a written "
        "pre-registration, and the confirmation set is now used up."
        % s["tokens_spent"])
    return "0 tokens spent; %d sealed clusters still blind" % s["n_sealed"]


def t3():
    """The two halves are disjoint and cover the pool exactly."""
    d = loader.load()
    a, b = set(d["sealed"]), set(d["open"])
    assert not (a & b), "a cluster is in BOTH halves: %s" % sorted(a & b)[:3]
    assert len(a) + len(b) == d["n_pool"], "halves do not cover the pool"
    return "%d + %d disjoint" % (len(a), len(b))


def t4():
    """The split is balanced on every axis it stratified."""
    b = loader.load()["balance"]
    for f in ("z", "cts500", "n_shear_sources"):
        s, o = b[f]["sealed"]["median"], b[f]["open"]["median"]
        assert s and o, "no median for %s" % f
        rel = abs(s - o) / max(abs(s), abs(o))
        assert rel < 0.15, "%s medians differ by %.0f%%: sealed %s open %s" % (
            f, 100 * rel, s, o)
    return "z, cts500 and shear depth all within 15%"


# ---------------------------------------------------------------- refusals
def t5():
    """verify() must FAIL if the sealed list is tampered with."""
    d = loader.load()
    saved = list(d["sealed"])
    try:
        d["sealed"] = saved[:-1]            # drop one cluster
        try:
            loader.verify()
        except loader.SealBroken:
            return "a one-cluster edit is caught"
        raise AssertionError("tampering was NOT caught")
    finally:
        d["sealed"] = saved


def t6():
    """open_pool() must never return a sealed cluster."""
    sealed = set(loader.sealed_names())
    got = {c["name"] for c in loader.open_pool()}
    assert not (got & sealed), "open_pool leaked %d sealed clusters" % len(got & sealed)
    assert len(got) == loader.load()["n_open"], "open_pool returned the wrong count"
    return "%d open clusters, 0 sealed" % len(got)


def t7():
    """sealed_metadata() must expose ONLY the declared open fields."""
    allowed = set(loader.load()["open_fields"])
    rows = loader.sealed_metadata()
    assert rows, "no sealed metadata returned"
    for r in rows[:50]:
        extra = set(r) - allowed
        assert not extra, "sealed metadata leaked %s" % sorted(extra)
    leaky = {"n_shear_sources", "n_spec_members", "channels"}
    assert not (allowed & leaky), "an outcome-bearing field is declared open"
    return "%d rows, fields limited to %s" % (len(rows), sorted(allowed))


def t8():
    """assert_not_sealed() must refuse a target list containing a sealed cluster."""
    one = loader.sealed_names()[0]
    try:
        loader.assert_not_sealed(["something-harmless", one], context="unit test")
    except loader.SealedClusterTouched:
        return "refused a list containing %s" % one
    raise AssertionError("a sealed cluster passed assert_not_sealed")


def t9():
    """assert_not_sealed() must ALLOW a purely open target list."""
    loader.assert_not_sealed(loader.open_names()[:20], context="unit test")
    return "20 open clusters allowed"


def t10():
    """A token needs a specific written reason."""
    for bad in ("", "because", "test"):
        try:
            loader.request_token(bad, requester="test")
        except loader.TokenInvalid:
            continue
        raise AssertionError("a vague reason %r minted a token" % bad)
    return "3 vague reasons refused"


def t11():
    """An unknown token must be refused -- on the FIXTURE, not the real ledger.

    open_sealed() ledgers a rejection before it raises, which is right in
    production and wrong in a test: run against the real paths, every CI run
    would append a token_rejected line to the programme's provenance ledger,
    dirty the working tree, and bury a genuine rejection in test noise.
    """
    with _fixture() as tmp:
        try:
            loader.open_sealed("deadbeef" * 4)
        except loader.TokenInvalid:
            n = sum(1 for _ in io.open(loader.LEDGER, encoding="utf-8")) \
                if os.path.exists(loader.LEDGER) else 0
            assert n == 1, "expected exactly one ledgered rejection, got %d" % n
            del tmp
            return "unknown token refused, and the rejection went to the fixture"
    raise AssertionError("an unknown token opened the seal")


def t12():
    """One-shot really is one shot -- exercised on a FIXTURE, not the real seal."""
    with _fixture():
        loader.verify()
        tok = loader.request_token(
            "fixture test of the one-shot property, exercising mint then "
            "double-open against a temporary split", requester="test_seal")
        toks = loader._tokens()
        assert len(toks) == 1 and not toks[0]["spent"], "token not recorded unspent"
        # mark spent the way open_sealed does, then confirm reuse is refused
        toks[0]["spent"] = True
        toks[0]["spent_utc"] = "1970-01-01T00:00:00Z"
        io.open(loader.TOKENS, "w", newline="\n", encoding="utf-8").write(
            json.dumps(toks[0]) + "\n")
        try:
            loader.open_sealed(tok)
        except loader.TokenInvalid:
            return "a spent token is refused on reuse"
    raise AssertionError("a spent token opened the seal a second time")


def t13():
    """The real seal is untouched by the fixture test."""
    loader.verify()
    s = loader.status()
    assert s["tokens_spent"] == 0, "the fixture test spent the real holdout"
    assert s["n_sealed"] == 304, "sealed count changed: %d" % s["n_sealed"]
    return "real seal intact after the fixture ran"


TESTS = [("T%d" % i, f) for i, f in enumerate(
    [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13], 1)]


def main():
    for name, fn in TESTS:
        check("%s %s" % (name, (fn.__doc__ or "").strip().splitlines()[0]), fn)
    npass = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        print(("PASS  " if ok else "FAIL  ") + name)
        if detail:
            print("        " + detail)
    print("\n%d/%d tests passed" % (npass, len(_results)))
    io.open(os.path.join(HERE, "tests.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(
                dict(n=len(_results), n_passed=npass,
                     failures=[[n, d] for ok, n, d in _results if not ok]),
                indent=1) + "\n")
    return 0 if npass == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
