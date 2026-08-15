"""Gates for the billion-scale baryonic GPU screen.

The screen exists to answer one question at scale: which universal baryonic laws
produce flat rotation curves and Tully-Fisher without per-galaxy freedom.  The tests
pin the physics controls (Newton-only must fail, known interpolating laws must pass),
the codec, the CPU/GPU agreement, and the claim boundary.  GPU-specific tests skip
cleanly when no CUDA device is present so CI stays green on CPU runners.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigma_theory_compiler.gpu_baryonic_interpolation_screen import (
    CLAIMS,
    FAMILY_SIZE,
    SCREEN_CONFIG,
    BaryonicScreenError,
    build_probe_grid,
    decode_ordinal,
    encode_candidate,
    render_candidate,
    screen_batch,
    verify_candidate_exact,
)


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


@pytest.fixture(scope="module")
def grid():
    return build_probe_grid()


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


def test_family_size_is_the_declared_product():
    assert FAMILY_SIZE == 4 * 7**10 == 1_129_900_996


@pytest.mark.parametrize("ordinal", [0, 1, 7**10 - 1, 7**10, FAMILY_SIZE - 1, 123_456_789])
def test_decode_encode_roundtrip(ordinal):
    candidate = decode_ordinal(ordinal)
    assert (
        encode_candidate(candidate["beta_index"], candidate["a"], candidate["b"]) == ordinal
    )


def test_out_of_range_ordinals_are_refused():
    with pytest.raises(BaryonicScreenError):
        decode_ordinal(-1)
    with pytest.raises(BaryonicScreenError):
        decode_ordinal(FAMILY_SIZE)


def test_render_shows_the_universal_law():
    text = render_candidate(decode_ordinal(encode_candidate(1, [0, 1, 0, 0, 0], [0, 0, 0, 0, 0])))
    assert text == "nu(y) = [(1 + u^2) / (1)]^1/2,  u = y^(-1/2)"


# ---------------------------------------------------------------------------
# Physics known-answer controls (exact mpmath layer)
# ---------------------------------------------------------------------------


def test_newton_alone_fails_flat_curves(grid):
    """nu = 1 is baryons under pure Newton: the dark-matter problem itself."""

    verdict = verify_candidate_exact(encode_candidate(2, [0] * 5, [0] * 5), grid)
    assert verdict["passes"] is False
    assert verdict["checks"]["flat_curves"] is False


@pytest.mark.parametrize(
    ("name", "beta_index", "a"),
    [
        ("sqrt(1+u^2)", 1, [0, 1, 0, 0, 0]),
        ("1+u", 2, [1, 0, 0, 0, 0]),
        ("cbrt(1+u^3)", 0, [0, 0, 1, 0, 0]),
    ],
)
def test_known_interpolating_laws_survive(grid, name, beta_index, a):
    verdict = verify_candidate_exact(encode_candidate(beta_index, a, [0] * 5), grid)
    assert verdict["passes"] is True, name
    slope = float(verdict["btfr_slope"])
    assert abs(slope - 4) <= float(SCREEN_CONFIG["fp64_thresholds"]["btfr_slope"])


def test_overboost_fails_flatness(grid):
    """nu = 1 + u^2 boosts too hard: v rises with radius instead of flattening."""

    verdict = verify_candidate_exact(encode_candidate(2, [0, 1, 0, 0, 0], [0] * 5), grid)
    assert verdict["passes"] is False
    assert verdict["checks"]["flat_curves"] is False


def test_beta_two_family_cannot_recover_newton(grid):
    verdict = verify_candidate_exact(encode_candidate(3, [1, 0, 0, 0, 0], [0] * 5), grid)
    assert verdict["passes"] is False
    assert verdict["checks"]["newton_far"] is False


def test_survivor_slope_is_near_four_not_exactly_assumed(grid):
    """The slope is measured from synthetic curves, not asserted."""

    verdict = verify_candidate_exact(encode_candidate(0, [0, 0, 1, 0, 0], [0] * 5), grid)
    assert verdict["btfr_slope"] is not None
    assert verdict["btfr_slope"] != "4"  # measured decimal, not a planted constant


# ---------------------------------------------------------------------------
# Batched path agrees with the exact layer
# ---------------------------------------------------------------------------


CONTROL_ORDINALS = {
    "newton_only": encode_candidate(2, [0] * 5, [0] * 5),
    "sqrt_family": encode_candidate(1, [0, 1, 0, 0, 0], [0] * 5),
    "linear_u": encode_candidate(2, [1, 0, 0, 0, 0], [0] * 5),
    "cbrt_family": encode_candidate(0, [0, 0, 1, 0, 0], [0] * 5),
    "overboost": encode_candidate(2, [0, 1, 0, 0, 0], [0] * 5),
}
CONTROL_EXPECTED = {
    "newton_only": False,
    "sqrt_family": True,
    "linear_u": True,
    "cbrt_family": True,
    "overboost": False,
}


def test_numpy_batch_decisions_match_exact_layer(grid):
    ordinals = np.array(list(CONTROL_ORDINALS.values()), dtype=np.int64)
    fp64 = screen_batch(
        np, ordinals, grid, dtype=np.float64, thresholds=SCREEN_CONFIG["fp64_thresholds"]
    )
    for name, decision in zip(CONTROL_ORDINALS, fp64, strict=True):
        assert bool(decision) == CONTROL_EXPECTED[name], name


def test_fp32_slack_never_drops_a_strict_survivor(grid):
    """fp32 thresholds are slack: anything fp64 accepts must pass fp32 too."""

    ordinals = np.array(list(CONTROL_ORDINALS.values()), dtype=np.int64)
    fp32 = screen_batch(
        np, ordinals, grid, dtype=np.float32, thresholds=SCREEN_CONFIG["fp32_thresholds"]
    )
    fp64 = screen_batch(
        np, ordinals, grid, dtype=np.float64, thresholds=SCREEN_CONFIG["fp64_thresholds"]
    )
    assert not np.any(fp64 & ~fp32)


def test_gpu_and_cpu_decisions_agree_on_a_sample(grid):
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(FAMILY_SIZE, size=8192, replace=False)).astype(np.int64)
    cpu = screen_batch(
        np, sample, grid, dtype=np.float64, thresholds=SCREEN_CONFIG["fp64_thresholds"]
    )
    gpu = screen_batch(
        cupy,
        cupy.asarray(sample),
        grid,
        dtype=cupy.float64,
        thresholds=SCREEN_CONFIG["fp64_thresholds"],
    ).get()
    assert int((cpu != gpu).sum()) == 0


# ---------------------------------------------------------------------------
# Grammar-level policy: no per-galaxy freedom is expressible
# ---------------------------------------------------------------------------


def test_grammar_has_no_per_galaxy_parameters():
    """Every candidate is one global (beta, a, b) tuple applied to all disks."""

    candidate = decode_ordinal(0)
    assert set(candidate) == {"beta_index", "beta", "a", "b"}
    assert CLAIMS["per_galaxy_free_parameters_expressible"] is False
    assert CLAIMS["invisible_mass_used_as_target_or_rescue"] is False


def test_claims_keep_the_validation_ladder_sealed():
    assert CLAIMS["observational_data_opened"] is False
    assert CLAIMS["survivor_is_validated_theory"] is False
    assert CLAIMS["sealed_validation_ladder_bypassed"] is False
    assert CLAIMS["synthetic_analytic_controls_only"] is True


def test_probe_grid_outer_regions_are_deep(grid):
    """v_flat is only defined in the deep regime; the outer windows must sit there."""

    for disk in grid["disks"]:
        for point in disk["points"]:
            if point["outer"]:
                assert point["gbar"] < 0.05
