"""
REGRESSION TESTS for the temperature-extrapolation fix in
`invariant_bench._cluster_profile`.

THE TEST THAT MATTERS is `test_forbid_raises_on_xcop`.  Against the code as it
stood before this patch it FAILS -- the pre-patch `_cluster_profile(d)` took no
`temp_extrapolation` argument at all, so asking it to refuse silent
extrapolation raised TypeError instead of the specific error, and asking it
nothing at all let 93 of 588 X-COP points through with dln kT/dln r forced to
exactly zero and no trace in the output.

`test_default_is_bit_identical` is its counterweight: the default path must
still reproduce the pre-patch arithmetic EXACTLY, so that no recorded number in
the programme moves unless somebody chooses to move it.  The pre-patch body is
inlined below as `legacy_cluster_profile` and is the literal code that was
replaced.

Run:  python -m pytest test_tempclamp.py -q
  or: python test_tempclamp.py
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pytest
from astropy.io import fits

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
BENCH_DIR = ROOT + "work/gravity-wells-2026-09"
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import invariant_bench as IB                                      # noqa: E402
from invariant_bench import (Bench, G, KPC, MSUN, MU, MU_E, MP,   # noqa: E402
                             TemperatureExtrapolationError,
                             TemperatureExtrapolationWarning)

XR = (ROOT + "runs/gravity/roadmap/"
      "item-59-xcop-forward-observable-gate-v1-source/raw/")
CLUSTERS = sorted(n for n in os.listdir(XR) if os.path.isdir(os.path.join(XR, n)))


# ---------------------------------------------------------------------------
#  the pre-patch body, kept verbatim as the bit-identity reference
# ---------------------------------------------------------------------------
def legacy_cluster_profile(d):
    """EXACTLY the code this patch replaced.  Do not 'improve' it."""
    fd = glob.glob(os.path.join(d, "*density*.fits"))
    ft = glob.glob(os.path.join(d, "*temperature*.fits"))
    if not fd or not ft:
        return None
    hd, ht = fits.open(fd[0]), fits.open(ft[0]); H = hd[1].header
    M500 = float(H["M500"]) * 1e14 * MSUN; R500 = float(H["R500"]) * KPC
    kT500 = G * M500 * MU * MP / (2 * R500)
    da = hd[1].data
    r = (0.5 * (da["R_IN"].astype(np.float64) + da["R_OUT"].astype(np.float64))) * KPC
    ne = da["NE"].astype(np.float64) * 1e6
    td = ht[1].data
    kT = np.interp(r, td["RW_X"].astype(np.float64) * R500,
                   td["T_X"].astype(np.float64) * kT500)
    lr = np.log(r)
    go = -(kT / (MU * MP)) * (np.gradient(np.log(ne), lr)
                              + np.gradient(np.log(kT), lr)) / r
    rho = MU_E * ne * MP
    Mg = (4 / 3 * np.pi * r[0] ** 3 * rho[0]
          + np.concatenate([[0.], np.cumsum(4 * np.pi * rho[:-1] * r[:-1] ** 2
                                            * np.diff(r))]))
    fs = glob.glob(os.path.join(d, "*mstar*.fits"))
    Mst = (np.interp(r, fits.open(fs[0])[2].data["RADIUS"].astype(np.float64) * KPC,
                     fits.open(fs[0])[2].data["MSTAR"].astype(np.float64) * MSUN)
           if fs else Mg * 0.10)
    Mb = Mg + Mst
    return r, G * Mb / r ** 2, go, R500


def shell():
    """An uninitialised Bench.  Bench() would load KiDS and the wide binaries,
    both of which are SEALED for this programme."""
    b = Bench.__new__(Bench)
    b.temp_extrapolation = "clamp"
    b.warn_extrapolation = False
    b.extrapolation_report = []
    b.d = {}
    return b


def cdir(name):
    return os.path.join(XR, name)


# ===========================================================================
#  1.  THE REGRESSION TEST: silent extrapolation must no longer be possible
# ===========================================================================
def test_forbid_raises_on_xcop():
    """FAILS against the pre-patch code.

    Pre-patch there was no way to say 'do not extrapolate', so this call
    raised TypeError (unexpected keyword) rather than the specific error, and
    the clamped points went through unannounced.
    """
    b = shell()
    raised = []
    for nm in CLUSTERS:
        try:
            b._cluster_profile(cdir(nm), temp_extrapolation="forbid")
        except TemperatureExtrapolationError as e:
            raised.append((nm, str(e)))
    assert raised, ("no cluster reported out-of-range temperature: either the "
                    "data changed or the guard is not wired up")
    assert len(raised) == len(CLUSTERS), (
        f"only {len(raised)}/{len(CLUSTERS)} clusters flagged; "
        f"the audit measured all 12 to be affected")
    # the message must carry the numbers a reader needs
    for nm, msg in raised:
        assert "outside the measured temperature range" in msg
        assert "R500" in msg and "%" in msg


def test_forbid_is_not_the_default():
    """The default must stay 'clamp', so nothing moves by accident."""
    b = shell()
    assert Bench.__init__.__defaults__[1] == "clamp"
    p = b._cluster_profile(cdir(CLUSTERS[0]))
    assert p.mode == "clamp"


# ===========================================================================
#  2.  bit-identity of the default path
# ===========================================================================
@pytest.mark.parametrize("nm", CLUSTERS)
def test_default_is_bit_identical(nm):
    old = legacy_cluster_profile(cdir(nm))
    new = shell()._cluster_profile(cdir(nm))
    assert len(new) == 4, "the return value must still unpack as (r, gb, go, R500)"
    for i, what in enumerate(("r", "gb", "go")):
        assert np.array_equal(old[i], new[i]), f"{nm}: {what} changed"
    assert old[3] == new[3], f"{nm}: R500 changed"
    # and unpacking still works, which is how every caller uses it
    r, gb, go, R500 = new
    assert np.array_equal(r, old[0])


# ===========================================================================
#  3.  the mask is correct, and it is what the bug actually is
# ===========================================================================
@pytest.mark.parametrize("nm", CLUSTERS)
def test_mask_matches_measured_range(nm):
    b = shell()
    p = b._cluster_profile(cdir(nm))
    ft = glob.glob(os.path.join(cdir(nm), "*temperature*.fits"))[0]
    fd = glob.glob(os.path.join(cdir(nm), "*density*.fits"))[0]
    with fits.open(fd) as h:
        R500 = float(h[1].header["R500"]) * KPC
    with fits.open(ft) as h:
        rw = np.asarray(h[1].data["RW_X"], float) * R500
    r = p[0]
    expect = (r < rw.min()) | (r > rw.max())
    assert np.array_equal(p.extrapolated, expect)
    assert p.frac_extrapolated == pytest.approx(expect.mean())
    assert p.r_tmin == pytest.approx(rw.min())
    assert p.r_tmax == pytest.approx(rw.max())


@pytest.mark.parametrize("nm", CLUSTERS)
def test_clamped_points_have_exactly_zero_temperature_gradient(nm):
    """THE BUG, stated as a test.  Under clamping the temperature-gradient term
    of the hydrostatic equation is not small on these points -- it is zero."""
    b = shell()
    p = b._cluster_profile(cdir(nm))
    lr = np.log(p[0])
    dlnT = np.gradient(np.log(p.kT), lr)
    # A point is "deep" if it and BOTH np.gradient neighbours are clamped, so
    # the whole 3-point stencil sits on the same constant.  Endpoints have a
    # one-sided stencil and are excluded.
    e = p.extrapolated
    deep = np.zeros_like(e)
    deep[1:-1] = e[1:-1] & e[:-2] & e[2:]
    assert deep.any(), f"{nm}: expected clamped interior points, found none"
    # np.gradient's unequal-spacing coefficients sum to zero only to rounding,
    # so a genuinely flat stencil returns ~1e-13, not literal 0.  Measured
    # in-range slopes are O(0.1-1), four orders larger.
    assert np.abs(dlnT[deep]).max() < 1e-11, (
        f"{nm}: clamped interior points should have dlnT/dlnr = 0 to rounding, "
        f"got {np.abs(dlnT[deep]).max():.3e}")
    # and the measured slope is emphatically NOT zero where T IS measured
    inner = ~p.stencil
    assert inner.sum() > 5, f"{nm}: too few fully-measured points"
    assert np.abs(dlnT[inner]).max() > 1e-3, (
        f"{nm}: the temperature-gradient term is what clamping deletes; "
        f"it must be non-trivial where the temperature is measured")


@pytest.mark.parametrize("nm", CLUSTERS)
def test_stencil_is_the_dilated_mask(nm):
    p = shell()._cluster_profile(cdir(nm))
    e = p.extrapolated
    want = e.copy()
    want[:-1] |= e[1:]
    want[1:] |= e[:-1]
    assert np.array_equal(p.stencil, want)
    assert p.stencil.sum() >= p.extrapolated.sum()


# ===========================================================================
#  4.  the alternative modes
# ===========================================================================
@pytest.mark.parametrize("nm", CLUSTERS)
def test_drop_removes_exactly_the_extrapolated_points(nm):
    b = shell()
    full = b._cluster_profile(cdir(nm), temp_extrapolation="clamp")
    dropped = b._cluster_profile(cdir(nm), temp_extrapolation="drop")
    keep = ~full.extrapolated
    assert len(dropped[0]) == int(keep.sum())
    assert dropped.n_dropped == int(full.extrapolated.sum())
    # the retained points keep exactly the values they had -- dropping must not
    # perturb the derivative stencil of the points that survive
    for i in range(3):
        assert np.array_equal(dropped[i], full[i][keep])
    assert not dropped.extrapolated.any()


@pytest.mark.parametrize("nm", CLUSTERS)
def test_loglinear_changes_only_the_extrapolated_points(nm):
    b = shell()
    cl = b._cluster_profile(cdir(nm), temp_extrapolation="clamp")
    ll = b._cluster_profile(cdir(nm), temp_extrapolation="loglinear")
    e = cl.extrapolated
    assert np.array_equal(cl.kT[~e], ll.kT[~e]), "in-range temperature moved"
    if e.any():
        assert not np.array_equal(cl.kT[e], ll.kT[e]), "extrapolation did nothing"
        # X-COP temperatures fall outward, so the continued profile must be
        # COOLER than the clamp beyond the last measured bin
        out = e & (cl[0] > cl.r_tmax)
        if out.any():
            assert np.all(ll.kT[out] <= cl.kT[out] + 1e-30)
        assert np.isfinite(ll.outer_logslope)
        assert ll.outer_logslope < 0.0, "measured outer T slope should be negative"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        shell()._cluster_profile(cdir(CLUSTERS[0]), temp_extrapolation="magic")
    # the constructor must reject it BEFORE loading anything, so a typo can
    # never quietly fall back to the clamped default
    with pytest.raises(ValueError):
        Bench(verbose=False, temp_extrapolation="magic")


# ===========================================================================
#  5.  it can never be invisible again
# ===========================================================================
def test_xcop_reports_the_extrapolated_fraction():
    b = shell()
    xc = b._xcop(temp_extrapolation="clamp")
    assert len(b.extrapolation_report) == len(CLUSTERS)
    n = sum(q["n"] for q in b.extrapolation_report)
    ne = sum(q["n_extrapolated"] for q in b.extrapolation_report)
    assert n == len(xc)
    assert ne == int(xc.extrapolated.sum())
    assert ne > 0, "the audit measured 93 of 588; zero means the wiring broke"
    for q in b.extrapolation_report:
        assert set(("cluster", "mode", "n", "n_extrapolated",
                    "frac_extrapolated", "n_stencil", "r_tmax_over_R500",
                    "r_max_over_R500")) <= set(q)
        # the honest-reach numbers must be present and ordered
        assert q["r_max_over_R500"] > q["r_tmax_over_R500"]


def test_announce_emits_a_structured_warning():
    b = shell()
    b.warn_extrapolation = True
    b.d["xcop"] = b._xcop(temp_extrapolation="clamp")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b._announce_extrapolation(verbose=False)
    assert len(w) == 1
    assert issubclass(w[0].category, TemperatureExtrapolationWarning)
    msg = str(w[0].message)
    assert "TEMPERATURE EXTRAPOLATION" in msg
    assert "%" in msg and "dlnT/dlnr is IDENTICALLY ZERO" in msg
    assert b.extrapolation_summary["n_extrapolated"] > 0
    assert 0.0 < b.extrapolation_summary["frac_extrapolated"] < 1.0


# ===========================================================================
#  6.  ingest assertions (row/column counts on every read)
# ===========================================================================
@pytest.mark.parametrize("nm", CLUSTERS)
def test_row_counts_asserted(nm):
    p = shell()._cluster_profile(cdir(nm))
    fd = glob.glob(os.path.join(cdir(nm), "*density*.fits"))[0]
    with fits.open(fd) as h:
        assert p.n_full == len(h[1].data)
    assert len(p.extrapolated_full) == p.n_full
    assert len(p.stencil_full) == p.n_full


def test_point_count_is_the_recorded_588():
    """Every recorded X-COP number rests on this count."""
    b = shell()
    xc = b._xcop(temp_extrapolation="clamp")
    assert len(xc) == 588, f"X-COP point count moved: {len(xc)} != 588"
    assert int(xc.extrapolated.sum()) == 93, (
        f"extrapolated count moved: {int(xc.extrapolated.sum())} != 93")


def test_drop_mode_still_reports_what_it_removed():
    """BUG FOUND BY THE SMOKE TEST: the first version of this patch counted
    only the extrapolated points still PRESENT, so mode='drop' reported zero
    extrapolation and said nothing about the 93 points it had just deleted --
    exactly as silent as the bug it replaces."""
    b = shell()
    b.warn_extrapolation = True
    b.temp_extrapolation = "drop"
    b.d["xcop"] = b._xcop(temp_extrapolation="drop")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b._announce_extrapolation(verbose=False)
    assert len(w) == 1, "drop mode must still announce what it removed"
    s = b.extrapolation_summary
    assert s["n_dropped"] == 93
    assert s["n_extrapolated"] == 0
    assert s["n_affected"] == 93
    assert s["n_points"] == 495 and s["n_points_full_grid"] == 588
    assert "DROPPED 93" in str(w[0].message)


def test_load_does_not_swallow_the_forbid_error():
    """BUG FOUND BY THE SMOKE TEST: Bench._load wraps every loader in a bare
    `except Exception`, which turned mode='forbid' into a silently missing
    probe instead of a stopped run."""
    import invariant_bench as _IB
    kids, wide = _IB.Bench._kids, _IB.Bench._widebin
    _IB.Bench._kids = lambda self: None          # SEALED
    _IB.Bench._widebin = lambda self: None       # SEALED
    try:
        with pytest.raises(TemperatureExtrapolationError):
            Bench(verbose=False, temp_extrapolation="forbid")
    finally:
        _IB.Bench._kids, _IB.Bench._widebin = kids, wide


def test_sealed_probes_never_loaded_here():
    b = shell()
    b.d["xcop"] = b._xcop(temp_extrapolation="clamp")
    assert "kids" not in b.d and "widebin" not in b.d


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
