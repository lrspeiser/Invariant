"""test_lane.py -- tests before results, as the other wellnet lanes do.

Run directly, NOT under pytest: the provenance guard raises ForeignReadError on
pytest's own .pytest_cache write.

    python test_lane.py

The tests that matter here are the ones that check the guard actually refuses
things, because a guard that silently passes is worse than no guard -- it turns
an unchecked claim into a documented one.
"""
from __future__ import annotations

import io
import json
import os
import sys

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))
_results = []


def check(name, fn):
    try:
        detail = fn()
        _results.append((True, name, detail or ""))
    except AssertionError as e:
        _results.append((False, name, str(e)[:200]))
    except Exception as e:                                      # noqa: BLE001
        _results.append((False, name, "%s: %s" % (type(e).__name__, str(e)[:180])))


# ------------------------------------------------------------------ G2 tests
def t1():
    """A pure COUNT projection is inventory and must be allowed."""
    guard.check_query("SELECT COUNT(*) AS n FROM delve_dr3.decade_shear WHERE dec > 0")
    return "COUNT(*) allowed"


def t2():
    """The exact shape query the eFEDS lane legitimately used must be REFUSED."""
    q = ("SELECT ra,dec,mcal_g_1_noshear,mcal_g_2_noshear,mcal_w_noshear,dnf_z,"
         "mcal_g_1_1p,mcal_g_1_1m,mcal_g_2_2p,mcal_g_2_2m "
         "FROM delve_dr3.decade_shear")
    try:
        guard.check_query(q)
    except guard.InventoryViolation:
        return "eFEDS shape projection refused"
    raise AssertionError("the shape projection was NOT refused")


def t3():
    """Refusal must survive case and whitespace games."""
    for q in ("select  ra , DEC , MCAL_G_1_NOSHEAR from delve_dr3.decade_shear",
              "SELECT TOP 10 e1, e2 FROM x",
              "SELECT\n  shear_g1\n  FROM x"):
        try:
            guard.check_query(q)
        except guard.InventoryViolation:
            continue
        raise AssertionError("not refused: %r" % q[:60])
    return "3 variants refused"


def t4():
    """An unparseable projection must fail closed, not open."""
    try:
        guard.check_query("WITH x AS (SELECT 1) SELECT * FROM x")
    except guard.InventoryViolation:
        return "unparseable projection fails closed"
    raise AssertionError("an unparseable query was allowed through")


# ------------------------------------------------------------------ G3 tests
def t5():
    """A gravity statistic must not be writable."""
    for bad in ({"g_t": [1]}, {"cluster": {"delta_sigma": 1}},
                {"a": {"b": {"M500": 3}}}, {"shear_profile": []}):
        try:
            guard.assert_no_statistic(bad)
        except guard.InventoryViolation:
            continue
        raise AssertionError("not refused: %r" % bad)
    return "4 statistic shapes refused"


def t6():
    """An inventory record must be writable."""
    guard.assert_no_statistic(
        {"name": "1eRASS J000000.0-000000", "n_sources_box": 1234,
         "cts500": 900.0, "n_spec_members": 12, "channels": {"C4": "PUBLIC"}})
    return "inventory record allowed"


# --------------------------------------------------------------- v3 detectors
FAKE_GOOD = ("#Name: J/A+A/685/A106\n"
             "#Title: eRASS. Galaxy clusters and groups in WGH (Bulbul+, 2024)\n"
             "#Name: J/A+A/685/A106/emain\n"
             "#Title: Primary galaxy clusters and groups catalog\n"
             "col\n---\nunit\n---\n1\n2\n")


def t7():
    """All three detectors pass on a well-formed payload."""
    ok, d = archives.vizier_validate("J/A+A/685/A106/emain", FAKE_GOOD, ("bulbul",))
    assert ok, "good payload rejected: %r" % d
    assert d["D1_name_echo"] and d["D2_no_catalogs_examined"] and d["D3_title_match"]
    return "D1+D2+D3 pass"


def t8():
    """D1: a payload echoing a DIFFERENT catalogue must be rejected.

    This is the failure the velocities lane hit twelve times -- VizieR answered
    HTTP 200 with Cooper+2013, an unrelated REAL catalogue.
    """
    wrong = FAKE_GOOD.replace("J/A+A/685/A106", "J/MNRAS/430/1125")
    ok, d = archives.vizier_validate("J/A+A/685/A106/emain", wrong, ("bulbul",))
    assert not ok, "a wrong-catalogue serve passed validation"
    assert not d["D1_name_echo"], "D1 did not fire"
    return "wrong-catalogue serve rejected by D1"


def t9():
    """D2: CatalogsExamined means a fuzzy fallback and must be rejected."""
    fuzzy = FAKE_GOOD + "#INFO CatalogsExamined=18000\n"
    ok, d = archives.vizier_validate("J/A+A/685/A106/emain", fuzzy, ("bulbul",))
    assert not ok and not d["D2_no_catalogs_examined"], "D2 did not fire"
    return "fuzzy fallback rejected by D2"


def t10():
    """D3: the right identifier with the WRONG paper must be rejected."""
    ok, d = archives.vizier_validate("J/A+A/685/A106/emain", FAKE_GOOD, ("ettori",))
    assert not ok and d["D3_title_match"] is False, "D3 did not fire"
    return "wrong-paper title rejected by D3"


def t11():
    """A zero-row payload is not an absence until the detectors have passed."""
    empty = "#Name: J/A+A/685/A106\n#Title: eRASS (Bulbul+, 2024)\ncol\n---\nunit\n---\n"
    ok, d = archives.vizier_validate("J/A+A/685/A106", empty, ("bulbul",))
    assert ok and d["n_rows"] == 0, "validated-but-empty was misreported"
    return "0 rows reported separately from validity"


# ------------------------------------------------------------------ fd guard
def t12():
    """The lane must survive shelling out -- a file descriptor is not a path.

    provenance.OpenLedger.check turns the integer 3 into the path "3", finds it
    outside the lane root and raises. This broke the FIRST archive probe.
    """
    import subprocess
    p = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True)
    assert p.returncode == 0 and b"ok" in p.stdout, "subprocess failed under the guard"
    return "subprocess capture_output works with the guard armed"


# ---------------------------------------------------------------- query shape
def t13():
    """The coverage query must be inventory-only and carry the background cut."""
    import coverage
    q = coverage.build_query(150.0, -30.0, 0.25)
    guard.check_query(q)
    assert "COUNT(*)" in q and "dnf_z > 0.4500" in q, q
    return "coverage query is a guarded COUNT with the +0.20 background margin"


def t14():
    """RA wrap must be split, not clamped, or a cluster near RA=0 loses half its sky."""
    import coverage
    q = coverage.build_query(0.2, -30.0, 0.25)
    assert "OR ra BETWEEN" in q, "RA wrap not handled: %s" % q
    guard.check_query(q)
    return "RA wrap split into two ranges"


TESTS = [(f"T{i}", fn) for i, fn in enumerate(
    [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14], 1)]


def main():
    guard.arm()
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
