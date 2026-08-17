"""Gates for the screened-kernel gravity screen (v3).

The tests pin: the declared family size, the degeneracy-free screening axis, and the
codec; that v2's geometry is *reused* rather than forked (same 46,392 frozen nodes, same
hash); the two new environment fields validated against independent routes (the Poisson
divergence of the frozen Newtonian field and a high-precision central difference) and
against the analytic profiles; the embedded-v2 reproduction to v2's recorded digits --
including that corrupting it aborts the run; the Newton standard control; screening
sanity from both directions; that the *lensing* gate reads the screening (a candidate
whose lensing verdict flips when the screening index moves to ``none``); the batched
path against the exact layer; the fp32/fp64 slack sandwich; the tension-map frontier;
receipt determinism, reseal-tamper behaviour, CLI, and the claim boundary.  GPU tests
skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler import gpu_screened_kernel_screen as sk
from sigma_theory_compiler.gpu_baryonic_interpolation_screen import SCREEN_CONFIG
from sigma_theory_compiler.gpu_baryonic_kernel_screen import (
    KERNEL_CONFIG as V2_CONFIG,
)
from sigma_theory_compiler.gpu_baryonic_kernel_screen import (
    boost_exact,
    build_kernel_geometry,
)
from sigma_theory_compiler.gpu_screened_kernel_screen import (
    AXES,
    AXIS_SIZES,
    CLAIMS,
    CONTROL_ORDINALS,
    FAMILY_SIZE,
    KERNEL_SUBFAMILY_SIZE,
    SCREEN_ENTRIES,
    SCREEN_SCALES,
    SCREEN_SHARPNESS,
    SCREENED_CONFIG,
    TENSION_MAP_BOUNDS,
    V2_RECORDED,
    ScreenedKernelError,
    active_parameter_count,
    build_environment,
    build_exact_context,
    build_precompute_receipt,
    build_sweep_tables,
    decode_ordinal,
    emit_screened_covariant_lift,
    encode_indices,
    encode_named,
    environment_residuals,
    environment_sha256,
    evaluate_batch,
    evaluate_candidate_exact,
    evaluate_values_exact,
    falsifiable_predictions,
    geometry_sha256,
    kernel_values,
    main,
    passer_families,
    render_candidate,
    reproduce_v2_exemplars,
    run_screen,
    screen_factor_exact,
    validate_precompute_receipt,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

#: A hand-verified candidate whose *lensing* verdict is decided by the screening: the
#: bare kernel fails lensing consistency at 0.2490, the same kernel under a
#: density:1e-5:1 screen passes at 0.1456.
LENSING_SCREENING_KERNEL = {
    "local": "sqrt_one_plus_u_squared",
    "w_yukawa": "17/4",
    "L1": "12",
    "w_power": "21/4",
    "L2": "24",
    "p": "1",
    "t": "1",
}


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


@pytest.fixture(scope="module")
def geometry():
    return build_kernel_geometry()


@pytest.fixture(scope="module")
def context():
    return build_exact_context()


@pytest.fixture(scope="module")
def tables(context):
    return build_sweep_tables(context)


@pytest.fixture(scope="module")
def cpu_tables(tables):
    return sk._device_tables(tables, np)


@pytest.fixture(scope="module")
def small_receipt():
    return run_screen(limit=131072, batch_size=32768, use_gpu=False)


@pytest.fixture(scope="module")
def precompute_receipt():
    return build_precompute_receipt()


# ---------------------------------------------------------------------------
# Family declaration and codec
# ---------------------------------------------------------------------------


def test_family_size_is_declared_and_in_band():
    assert FAMILY_SIZE == 145 * 3 * 25 * 8 * 25 * 8 * 12 * 4 == 835_200_000
    assert 10**8 <= FAMILY_SIZE < 10**9
    assert SCREENED_CONFIG["family_size"] == FAMILY_SIZE
    assert tuple(len(AXES[name]) for name in sk.AXIS_ORDER) == AXIS_SIZES
    # The trimmed-axis floor the grammar declares.
    assert len(AXES["local"]) == 3
    assert len(AXES["w_yukawa"]) >= 25 and len(AXES["w_power"]) >= 25
    assert len(AXES["L1"]) >= 8 and len(AXES["L2"]) >= 8
    assert len(AXES["p"]) >= 8 and len(AXES["t"]) >= 4


def test_screening_axis_is_declared_and_degeneracy_free():
    assert len(SCREEN_ENTRIES) == 1 + 3 * 12 * 4 == 145
    assert SCREEN_ENTRIES[0] == ("none", "0", "0")
    assert len(set(SCREEN_ENTRIES)) == len(SCREEN_ENTRIES)
    for family in ("density", "acceleration", "curvature"):
        assert 10 <= len(SCREEN_SCALES[family]) <= 12
        assert len(set(SCREEN_SCALES[family])) == len(SCREEN_SCALES[family])
        # A strictly increasing log ladder.
        values = [float(text) for text in SCREEN_SCALES[family]]
        assert values == sorted(values)
    assert SCREEN_SHARPNESS == ("1", "2", "3", "4")
    # ``none`` appears exactly once, so v2 is embedded without 48-fold duplication.
    assert sum(1 for entry in SCREEN_ENTRIES if entry[0] == "none") == 1


def test_codec_roundtrip_and_embedded_v2_block():
    rng = np.random.default_rng(3)
    for ordinal in rng.integers(0, FAMILY_SIZE, size=64):
        decoded = decode_ordinal(int(ordinal))
        assert encode_indices(decoded["indices"]) == int(ordinal)
    with pytest.raises(ScreenedKernelError):
        decode_ordinal(FAMILY_SIZE)
    with pytest.raises(ScreenedKernelError):
        decode_ordinal(-1)
    with pytest.raises(ScreenedKernelError):
        encode_named(L1="5")  # off the grid
    with pytest.raises(ScreenedKernelError):
        encode_named(screen="density:2e-5:1")  # off the screening grid
    # Screening is the most significant digit: v2 is the leading contiguous block.
    assert KERNEL_SUBFAMILY_SIZE == FAMILY_SIZE // 145 == 5_760_000
    for ordinal in (0, 1, KERNEL_SUBFAMILY_SIZE - 1):
        assert decode_ordinal(ordinal)["values"]["screen"] == "none:0:0"
    assert decode_ordinal(KERNEL_SUBFAMILY_SIZE)["values"]["screen"] != "none:0:0"
    newton = decode_ordinal(encode_named())
    assert newton["values"]["w_yukawa"] == "0" and newton["values"]["w_power"] == "0"
    assert "S(x)*conv(rho_b, K)" in render_candidate(newton)
    assert "S = 1" in render_candidate(newton)


def test_v2_geometry_is_reused_not_forked(geometry, precompute_receipt):
    """The frozen 46,392-node distance tables are v2's, byte for byte."""

    assert precompute_receipt["total_nodes"] == 46_392
    assert precompute_receipt["geometry_sha256"] == geometry_sha256(geometry)
    assert SCREENED_CONFIG["geometry"] is V2_CONFIG["geometry"]
    for tier in ("fp32_thresholds", "fp64_thresholds", "newton_control"):
        assert SCREENED_CONFIG[tier] == V2_CONFIG[tier]
    assert [len(geometry[s]) for s in ("disk", "hernquist", "cluster")] == [5, 165, 5]


# ---------------------------------------------------------------------------
# The new precompute: rho_local and |grad g_N|
# ---------------------------------------------------------------------------


def test_environment_fields_validate_against_independent_routes(geometry):
    """rho_local and |grad g_N| are reproduced by routes that never touch the closed
    forms under test: the Poisson divergence of the frozen Newtonian field (spheres) or
    the frozen enclosed mass (disk), and a high-precision central difference."""

    budget = mp.mpf(SCREENED_CONFIG["environment_validation"]["max_relative_error"])
    residuals = environment_residuals(geometry)
    assert set(residuals) == {"disk", "hernquist", "cluster"}
    for system, block in residuals.items():
        assert set(block) == {"rho_max_relative_error", "grad_max_relative_error"}
        for key, value in block.items():
            assert mp.mpf(value) <= budget, (system, key, value)


def test_environment_fields_match_the_analytic_profiles(geometry):
    """Spot checks of the declared closed forms in the 4*pi*G*rho convention."""

    mp.mp.dps = 50
    environment = build_environment(geometry)
    amplitude = mp.mpf(V2_CONFIG["geometry"]["cluster"]["amplitude_4pi_rho0"])
    for entry in environment["cluster"]:
        radius = entry["radius"]
        assert abs(entry["rho"] / (amplitude / (1 + radius**2)) - 1) < mp.mpf(10) ** -40
        assert abs(entry["g"] / (amplitude * (radius - mp.atan(radius)) / radius**2) - 1) < (
            mp.mpf(10) ** -40
        )
    for entry in environment["hernquist"]:
        radius = entry["radius"]
        assert abs(entry["rho"] / (2 / (radius * (1 + radius) ** 3)) - 1) < mp.mpf(10) ** -40
        assert abs(entry["g"] / (1 / (1 + radius) ** 2) - 1) < mp.mpf(10) ** -40
        assert abs(entry["curv"] / (2 / (1 + radius)) - 1) < mp.mpf(10) ** -40
    thickness = mp.mpf(1) / 10
    for entry in environment["disk"]:
        radius = entry["radius"]
        assert abs(entry["rho"] / (mp.e ** (-radius) / thickness) - 1) < mp.mpf(10) ** -40
    # The declared Solar ambient is the largest control disk at a frozen inner radius.
    ambient_radius = SCREENED_CONFIG["screening"]["solar_ambient_radius"]
    assert ambient_radius in SCREEN_CONFIG["inner_radii"]
    mass = mp.mpf(128) / 125
    expected = mass * mp.e ** (-mp.mpf(ambient_radius)) / thickness
    assert abs(environment["solar_ambient_rho"] / expected - 1) < mp.mpf(10) ** -40


def test_precompute_receipt_validates_and_tampers_closed(precompute_receipt, geometry):
    assert precompute_receipt["schema_version"] == sk.PRECOMPUTE_SCHEMA
    validate_precompute_receipt(precompute_receipt)
    assert precompute_receipt["environment_sha256"] == environment_sha256(
        build_environment(geometry)
    )
    budget = mp.mpf(SCREENED_CONFIG["environment_validation"]["max_relative_error"])
    for system in ("disk", "hernquist", "cluster"):
        block = precompute_receipt["systems"][system]
        assert mp.mpf(block["rho_max_relative_error"]) <= budget
        assert mp.mpf(block["grad_max_relative_error"]) <= budget
    tampered = {k: v for k, v in precompute_receipt.items() if k != "content_sha256"}
    tampered["systems"] = json.loads(json.dumps(precompute_receipt["systems"]))
    tampered["systems"]["disk"]["rho_max_relative_error"] = "1e-90"
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ScreenedKernelError):
        validate_precompute_receipt(tampered)


# ---------------------------------------------------------------------------
# The screening factor itself
# ---------------------------------------------------------------------------


def test_screening_factor_semantics(geometry):
    mp.mp.dps = 50
    environment = build_environment(geometry)
    probe = environment["cluster"][2]
    assert screen_factor_exact(("none", "0", "0"), probe, mp.mpf(1)) == 1
    # S = 1/2 exactly when the screening argument equals 1.
    scale = probe["g"]
    entry = ("acceleration", mp.nstr(scale, 40), "3")
    assert abs(screen_factor_exact(entry, probe, mp.mpf(1)) - mp.mpf(1) / 2) < (
        mp.mpf(10) ** -30
    )
    # Density and acceleration screen harder as the scale shrinks; Lc screens harder as
    # it grows (it multiplies the argument).  Both directions are declared.
    weak = screen_factor_exact(("acceleration", "1000", "2"), probe, mp.mpf(1))
    strong = screen_factor_exact(("acceleration", "1e-2", "2"), probe, mp.mpf(1))
    assert weak > strong
    weak_c = screen_factor_exact(("curvature", "1e-3", "2"), probe, mp.mpf(1))
    strong_c = screen_factor_exact(("curvature", "300", "2"), probe, mp.mpf(1))
    assert weak_c > strong_c
    # Sharper k pushes an already-screened point further down.
    assert screen_factor_exact(("acceleration", "1e-2", "4"), probe, mp.mpf(1)) < strong


# ---------------------------------------------------------------------------
# Embedded v2: the run-aborting reproduction control
# ---------------------------------------------------------------------------


def test_embedded_v2_exemplars_reproduce_recorded_digits(context):
    """v2's four recorded closest approaches, re-run through v3 at screening ``none``."""

    reproduction = reproduce_v2_exemplars(context)
    assert set(reproduction) == set(V2_RECORDED)
    for name, block in reproduction.items():
        assert block["reproduced"] is True, (name, block["mismatches"])
        assert block["mismatches"] == []
    # The exemplar v2 recorded at 50 digits must match verbatim, ratios included.
    exact = reproduction["cluster_solar_safe"]
    assert exact["tier"] == "exact"
    assert exact["observed"]["cluster_dev"] == "4.050292865e-01"
    assert exact["observed"]["safety_margin"] == "2.399880004e-09"
    # Two of the four sit on the trimmed v3 grid and round-trip through the codec.
    on_grid = {name for name, block in reproduction.items() if block["on_v3_grid"]}
    assert on_grid == {"cluster_solar_safe", "lensing_among_galaxy_passers"}
    for name in on_grid:
        ordinal = reproduction[name]["v3_ordinal"]
        assert decode_ordinal(ordinal)["values"]["screen"] == "none:0:0"
        assert ordinal < KERNEL_SUBFAMILY_SIZE
    assert CLAIMS["embedded_v2_family_reproduces_prior_negative"] is True


def test_broken_embedded_v2_reproduction_aborts_the_run(context):
    controls = {
        name: evaluate_candidate_exact(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    good = reproduce_v2_exemplars(context)
    sk._assert_known_answer_controls(controls, good)  # the healthy path
    broken = json.loads(json.dumps(good))
    broken["cluster_solar_safe"]["reproduced"] = False
    broken["cluster_solar_safe"]["mismatches"] = ["cluster_dev drifted"]
    with pytest.raises(ScreenedKernelError, match="embedded v2 reproduction failed"):
        sk._assert_known_answer_controls(controls, broken)


def test_embedded_v2_block_still_fails_the_cluster_gate(cpu_tables):
    """Inside the ``none`` block the v3 pipeline is v2, so the sealed negative holds."""

    rng = np.random.default_rng(11)
    ordinals = rng.integers(0, KERNEL_SUBFAMILY_SIZE, size=200_000).astype(np.int64)
    batch = evaluate_batch(
        np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds"
    )
    assert not batch["all_pass"].any()


# ---------------------------------------------------------------------------
# Standard and screening-sanity controls
# ---------------------------------------------------------------------------


def test_newton_embedding_standard_control(context):
    verdict = evaluate_candidate_exact(CONTROL_ORDINALS["newton_identity"], context)
    assert verdict["newton"]["passes"] is True
    assert verdict["newton"]["safety_passes"] is True
    assert verdict["galaxy"]["passes"] is False
    assert verdict["lensing"]["passes"] is False
    assert verdict["cluster"]["passes"] is False
    assert float(verdict["cluster"]["shortfall_min"]) >= 1.5
    assert all(
        value == "1.000000000e+00" for value in verdict["cluster"]["screen_factor_by_probe"]
    )


def test_screening_sanity_both_directions(context):
    """A strong screen must buy Solar safety at the cost of the galaxy gate; an
    unscreened strong kernel must die at the Solar gates."""

    for name in ("strong_acceleration_screen", "strong_curvature_screen"):
        strong = evaluate_candidate_exact(CONTROL_ORDINALS[name], context)
        assert strong["newton"]["passes"] is True, name
        assert strong["galaxy"]["passes"] is False, name
        # The boost has been screened away, so the law has reverted towards Newton.
        assert float(strong["newton"]["screen_factor_safety"]) < 1e-20
        assert max(
            float(v) for v in strong["cluster"]["screen_factor_by_probe"]
        ) < 1e-5
    unscreened = evaluate_candidate_exact(CONTROL_ORDINALS["unscreened_yukawa"], context)
    assert unscreened["newton"]["safety_passes"] is False
    assert float(unscreened["newton"]["safety_margin"]) > 0.9
    assert float(unscreened["newton"]["solar_ratio"]) > 1e6
    solar_safe = evaluate_candidate_exact(CONTROL_ORDINALS["unscreened_power"], context)
    assert solar_safe["newton"]["passes"] is True


def test_acceleration_screen_frees_solar_without_touching_the_cluster(context):
    """The decisive regime: g_star = 100 kills the Solar boost by 1e-16 while leaving
    every cluster probe unscreened to four digits."""

    verdict = evaluate_candidate_exact(
        CONTROL_ORDINALS["solar_free_acceleration_screen"], context
    )
    assert verdict["newton"]["passes"] is True
    assert float(verdict["newton"]["screen_factor_safety"]) < 1e-15
    assert all(
        float(value) > 0.999 for value in verdict["cluster"]["screen_factor_by_probe"]
    )
    # The same kernel with no screening dies at the Solar gates.
    bare = dict(verdict["values"])
    bare["screen"] = "none:0:0"
    assert evaluate_values_exact(bare, context)["newton"]["passes"] is False


def test_screening_is_applied_in_every_gate(context):
    """Every gate metric moves when the screening index moves -- no gate quietly uses
    the unscreened boost."""

    screened = dict(LENSING_SCREENING_KERNEL, screen="density:1e-5:1")
    bare = dict(LENSING_SCREENING_KERNEL, screen="none:0:0")
    a = evaluate_values_exact(screened, context)
    b = evaluate_values_exact(bare, context)
    assert a["galaxy"]["flat_worst"] != b["galaxy"]["flat_worst"]
    assert a["lensing"]["worst_consistency"] != b["lensing"]["worst_consistency"]
    assert a["cluster"]["max_deviation"] != b["cluster"]["max_deviation"]
    assert a["newton"]["near_deviation"] != b["newton"]["near_deviation"]
    assert a["newton"]["safety_margin"] != b["newton"]["safety_margin"]
    assert SCREENED_CONFIG["screening"]["applied_in_gates"] == [
        "newton", "safety", "galaxy", "lensing", "cluster",
    ]
    assert CLAIMS["screening_applied_in_every_gate_including_lensing"] is True


def test_lensing_verdict_is_decided_by_the_screening(context, cpu_tables):
    """The consistency check the brief asks for: a candidate whose *lensing* verdict
    flips when the screening index is moved to ``none``."""

    screened_ordinal = encode_named(**LENSING_SCREENING_KERNEL, screen="density:1e-5:1")
    bare_ordinal = encode_named(**LENSING_SCREENING_KERNEL, screen="none:0:0")
    assert bare_ordinal < KERNEL_SUBFAMILY_SIZE

    screened = evaluate_candidate_exact(screened_ordinal, context)
    bare = evaluate_candidate_exact(bare_ordinal, context)
    assert bare["lensing"]["passes"] is False
    assert screened["lensing"]["passes"] is True
    assert float(bare["lensing"]["worst_consistency"]) > 0.15
    assert float(screened["lensing"]["worst_consistency"]) <= 0.15

    # The batched path agrees with the exact layer on the same flip.
    batch = evaluate_batch(
        np,
        np.asarray([bare_ordinal, screened_ordinal], dtype=np.int64),
        cpu_tables,
        dtype=np.float64,
        tier="fp64_thresholds",
    )
    assert list(batch["lensing_pass"]) == [False, True]


# ---------------------------------------------------------------------------
# Batched path against the exact layer; fp32 slack is a true superset
# ---------------------------------------------------------------------------


def test_two_ways_screened_boost_agreement(context, tables):
    """The screened basis tables equal direct 50-digit node sums times S.

    The tables carry unit-weight bases (the sweep multiplies by w_Y and w_P per
    candidate), so each side is isolated by zeroing the other amplitude.
    """

    mp.mp.dps = 50
    geometry, environment = context["geometry"], context["environment"]
    entry = ("acceleration", "1", "2")
    screen_index = AXES["screen"].index("acceleration:1:2")
    factors = [
        screen_factor_exact(entry, environment["cluster"][k], mp.mpf(1)) for k in range(5)
    ]

    power_only = dict(
        LENSING_SCREENING_KERNEL, screen="acceleration:1:2", w_yukawa="0", w_power="21/4"
    )
    combo = (
        AXES["L2"].index(power_only["L2"]) * len(AXES["p"])
        + AXES["p"].index(power_only["p"])
    ) * len(AXES["t"]) + AXES["t"].index(power_only["t"])
    weight = mp.mpf(21) / 4
    raw = boost_exact(geometry["cluster"], kernel_values(power_only))
    for k in range(5):
        expected = float(factors[k] * raw[k] / weight)
        got = float(tables["BPC"][(screen_index * 5 + k) * sk.N_POWER_COMBOS + combo])
        assert abs(got / expected - 1) < 1e-12, ("power", k)

    yukawa_only = dict(
        LENSING_SCREENING_KERNEL, screen="acceleration:1:2", w_yukawa="17/4", w_power="0"
    )
    l1_index = AXES["L1"].index(yukawa_only["L1"])
    weight = mp.mpf(17) / 4
    raw = boost_exact(geometry["cluster"], kernel_values(yukawa_only))
    for k in range(5):
        expected = float(factors[k] * raw[k] / weight)
        got = float(tables["BYC"][(screen_index * 5 + k) * len(AXES["L1"]) + l1_index])
        assert abs(got / expected - 1) < 1e-12, ("yukawa", k)


def _sample_ordinals(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, FAMILY_SIZE, size=count)
    return np.unique(
        np.concatenate([sample, np.asarray(list(CONTROL_ORDINALS.values()))])
    ).astype(np.int64)


def test_fp64_batch_decisions_match_exact_layer(context, cpu_tables):
    ordinals = _sample_ordinals(8, seed=11)
    batch = evaluate_batch(
        np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds"
    )
    for row, ordinal in enumerate(ordinals):
        verdict = evaluate_candidate_exact(int(ordinal), context)
        for gate in ("newton", "galaxy", "lensing", "cluster"):
            assert bool(batch[f"{gate}_pass"][row]) == verdict[gate]["passes"], (
                int(ordinal),
                gate,
            )
        assert abs(
            float(batch["cluster_dev"][row]) / float(verdict["cluster"]["max_deviation"]) - 1
        ) < 1e-9


def test_fp32_slack_never_drops_a_strict_survivor(cpu_tables):
    ordinals = _sample_ordinals(200_000, seed=7)
    fp32 = evaluate_batch(np, ordinals, cpu_tables, dtype=np.float32, tier="fp32_thresholds")
    fp64 = evaluate_batch(np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds")
    for gate in ("newton", "safety", "galaxy", "lensing", "cluster", "all"):
        assert not np.any(fp64[f"{gate}_pass"] & ~fp32[f"{gate}_pass"]), gate


def test_gpu_and_cpu_decisions_agree_on_a_sample(tables, cpu_tables):
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    ordinals = _sample_ordinals(20_000, seed=5)
    cpu = evaluate_batch(np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds")
    gpu = evaluate_batch(
        cupy,
        cupy.asarray(ordinals),
        sk._device_tables(tables, cupy),
        dtype=cupy.float64,
        tier="fp64_thresholds",
    )
    for gate in ("newton", "safety", "galaxy", "lensing", "cluster", "all"):
        assert int((cpu[f"{gate}_pass"] != gpu[f"{gate}_pass"].get()).sum()) == 0, gate


# ---------------------------------------------------------------------------
# Survivor structure helpers
# ---------------------------------------------------------------------------


def test_passer_families_group_grid_neighbors_within_a_screening_family():
    base = decode_ordinal(encode_named(screen="acceleration:1:2", w_power="1", L2="2"))[
        "indices"
    ]
    neighbor = list(base)
    neighbor[4] += 1  # w_power, one grid step
    diagonal = list(base)
    diagonal[4] += 1
    diagonal[6] += 1  # w_power and p one step each: Chebyshev distance 1
    far = list(base)
    far[4] += 5
    families = passer_families(
        [encode_indices(base), encode_indices(neighbor), encode_indices(diagonal),
         encode_indices(far)]
    )
    assert sorted(len(f) for f in families) == [1, 3]

    # A screening-family boundary is never crossed even at adjacent screen indices.
    boundary = AXES["screen"].index("density:3:4")
    assert AXES["screen"][boundary + 1].startswith("acceleration")
    left = list(base)
    left[0] = boundary
    right = list(base)
    right[0] = boundary + 1
    split = passer_families([encode_indices(left), encode_indices(right)])
    assert sorted(len(f) for f in split) == [1, 1]


def test_pareto_front_is_exact():
    axes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # dominated by row 3
            [0.5, 2.0, 1.0, 1.0],
            [2.0, 0.5, 1.0, 1.0],
            [0.9, 0.9, 0.9, 0.9],
        ]
    )
    front = {int(v) for v in sk._pareto_front(axes, 64)}
    assert front == {1, 2, 3}
    assert sk._pareto_front(np.empty((0, 4)), 8).size == 0


def test_covariant_lift_carries_the_screening_mechanism():
    values = decode_ordinal(
        encode_named(
            screen="acceleration:100:2",
            local="sqrt_one_plus_u_squared",
            w_yukawa="1/2",
            L1="12",
            w_power="2",
            L2="2",
            p="2",
            t="-1",
        )
    )["values"]
    lift = emit_screened_covariant_lift(values)
    assert lift["claims"] == {"first_principles_derivation_pending": True}
    mechanisms = [component["mechanism"] for component in lift["components"]]
    assert mechanisms == [
        "massive_scalar_exchange",
        "nonlocal_propagator_correction",
        "pointwise_modified_dynamics",
        "vainshtein_kinetic_braiding",
    ]
    # v2's emitter output is reused verbatim for the kernel components.
    assert "m = 1/L1 = 1/12" in lift["components"][0]["field_theory_ansatz"]
    assert "alpha = 1 - t/2 = 3/2" in lift["components"][1]["field_theory_ansatz"]
    screening = lift["components"][3]["field_theory_ansatz"]
    assert "Galileon" in screening and "scale = 100" in screening and "k = 2" in screening
    assert active_parameter_count(values) == 9
    assert active_parameter_count(decode_ordinal(encode_named())["values"]) == 0
    # ``none`` still emits a typed component so the block shape never varies.
    bare = emit_screened_covariant_lift(decode_ordinal(encode_named())["values"])
    assert bare["components"][-1]["mechanism"] == "no_screening"


def test_falsifiable_prediction_block(context):
    values = decode_ordinal(
        encode_named(screen="acceleration:100:2", w_yukawa="6", L1="4")
    )["values"]
    block = falsifiable_predictions(values, context)
    transitions = block["screening_transition_radii"]
    assert set(transitions) == {
        "point_source_separation",
        "cluster_radius",
        "disk_radius_largest_mass",
        "hernquist_radius_largest_mass",
    }
    # g_star = 100 crosses S = 1/2 at a separation s = 1/10 from a unit point mass.
    assert abs(float(transitions["point_source_separation"]) - 0.1) < 1e-6
    # The cluster never reaches g_N = 100, so it is unscreened throughout.
    assert transitions["cluster_radius"] == "unscreened_everywhere"
    deviations = block["predicted_gr_deviations"]
    assert set(deviations) == set(sk.PREDICTION_PROBES)
    assert all(isinstance(value, str) for value in deviations.values())
    assert "kills the family" in block["falsifier"]
    assert block["claims"] == {"first_principles_derivation_pending": True}
    # ``none`` reports no transition anywhere.
    bare = falsifiable_predictions(decode_ordinal(encode_named())["values"], context)
    assert set(bare["screening_transition_radii"].values()) == {"no_screening"}


# ---------------------------------------------------------------------------
# Receipt: end-to-end small run, tension map, determinism, tamper, claims
# ---------------------------------------------------------------------------


def test_small_run_receipt_shape_and_funnel_honesty(small_receipt):
    counts = small_receipt["counts"]
    assert counts["processed"] == 131072
    assert counts["family_size"] == FAMILY_SIZE
    assert counts["embedded_v2_block_size"] == KERNEL_SUBFAMILY_SIZE
    assert counts["exact_confirmed"] + counts["exact_refuted"] == counts["exact_verified"]
    assert counts["all_gate_passers"] <= counts["fp32"]["all"]
    for gate in ("newton", "galaxy", "lensing", "cluster"):
        assert counts["fp64_of_fp32_survivors"][gate] <= counts["fp32"][gate]
    controls = small_receipt["controls"]
    assert controls["newton_identity"]["cluster"]["passes"] is False
    assert controls["sqrt_local"]["lensing"]["passes"] is True
    assert controls["sqrt_local"]["cluster"]["passes"] is False
    assert controls["unscreened_yukawa"]["newton"]["safety_passes"] is False
    assert controls["strong_acceleration_screen"]["newton"]["passes"] is True
    assert controls["strong_acceleration_screen"]["galaxy"]["passes"] is False
    for block in small_receipt["embedded_v2_reproduction"].values():
        assert block["reproduced"] is True
    # This limit covers only the ``none`` block, so v2's sealed negative must reappear.
    assert counts["all_gate_passers"] == 0
    negative = small_receipt["sealed_negative"]
    assert negative["sealed"] is True
    assert small_receipt["decision"].startswith("SCREENED-SEALED-NEGATIVE")
    assert negative["structural_direction"]
    validate_receipt(small_receipt)


def test_screening_family_breakdown_covers_every_declared_family(small_receipt):
    """All four screening families are reported, zeros included -- a family that carries
    no passer is a measured statement about that mechanism, not an omission."""

    breakdown = small_receipt["screening_family_breakdown"]
    assert [row["screening_family"] for row in breakdown] == list(sk.SCREEN_FAMILIES)
    total = sum(row["all_gate_passers"] for row in breakdown)
    assert total == small_receipt["counts"]["all_gate_passers"]
    assert sum(row["equivalence_families"] for row in breakdown) == (
        small_receipt["counts"]["passer_families"]
    )
    for row in breakdown:
        if row["all_gate_passers"] == 0:
            assert row["best_cluster_deviation"] == "none"
            assert row["best_solar_ratio"] == "none"
        else:
            assert isinstance(row["best_cluster_deviation"], str)


def test_exact_verification_is_selected_by_role(small_receipt):
    """The 50-digit set is chosen by role, never by ordinal order, and the number of
    passers left to the fp64 tier is recorded rather than hidden."""

    counts = small_receipt["counts"]
    assert "passers_not_exactly_verified" in counts
    assert "exact_verification_selection_truncated" in counts
    assert small_receipt["exact_verification_policy"]
    allowed = {
        "family_representative",
        "pareto_front",
        "closest_approach_or_frontier_exemplar",
        "bulk_passer_random_sample",
    }
    for entry in small_receipt["exact_verification"]:
        assert entry["selection"] in allowed
        assert entry["exact_confirmed"] is True
    assert counts["exact_refuted"] == 0
    assert counts["passers_not_exactly_verified"] >= 0


def test_tension_map_is_complete_and_monotone(small_receipt):
    tension = small_receipt["tension_map"]
    frontier = tension["frontier"]
    assert [row["solar_ratio_bound"] for row in frontier] == list(TENSION_MAP_BOUNDS)
    assert sum(1 for row in frontier if row["solar_safe"]) == TENSION_MAP_BOUNDS.index("1") + 1
    columns = (
        "best_cluster_deviation",
        "best_cluster_deviation_with_galaxy_and_lensing",
        "best_cluster_deviation_all_gates_fp64",
    )
    previous = dict.fromkeys(columns, float("inf"))
    for row in frontier:
        for key in columns:
            assert isinstance(row[key], str)
            current = row[key]
            numeric = float("inf") if current in ("inf", "none") else float(current)
            assert numeric <= previous[key] + 1e-15, key
            previous[key] = numeric
        # Each added constraint can only make the achievable deviation worse.
        base = previous["best_cluster_deviation"]
        assert previous["best_cluster_deviation_with_galaxy_and_lensing"] >= base - 1e-15
        assert previous["best_cluster_deviation_all_gates_fp64"] >= base - 1e-15
        assert isinstance(row["galaxy_and_lensing_confirmed_fp64"], bool)
        assert row["all_gate_passers_under_bound"] >= 0
    assert "solar_ratio" in tension["definition"]
    # v2's own frontier is carried for comparison, and the penalty factor is measured.
    assert tension["v2_frontier_for_comparison"]["best_cluster_deviation_solar_safe"] == (
        "4.050292865e-01"
    )
    assert isinstance(tension["solar_safety_penalty_factor"], str)
    assert isinstance(tension["strict_all_gate_best_cluster_deviation"], str)
    if small_receipt["counts"]["all_gate_passers"] == 0:
        assert tension["strict_all_gate_best_cluster_deviation"] == "none"
        assert all(
            row["best_cluster_deviation_all_gates_fp64"] == "none" for row in frontier
        )
    # A non-monotone strict column is rejected too.
    broken = json.loads(json.dumps(small_receipt))
    broken["tension_map"]["frontier"][0]["best_cluster_deviation_all_gates_fp64"] = "1e-09"
    broken.pop("content_sha256")
    broken["content_sha256"] = canonical_sha256(broken)
    with pytest.raises(ScreenedKernelError, match="monotone"):
        validate_receipt(broken)
    # A non-monotone frontier is rejected by the validator.
    tampered = json.loads(json.dumps(small_receipt))
    rows = tampered["tension_map"]["frontier"]
    rows[-1]["best_cluster_deviation"] = "9.999999999e+00"
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ScreenedKernelError, match="monotone"):
        validate_receipt(tampered)


def test_receipt_has_no_floating_values(small_receipt):
    def walk(value):
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(small_receipt)


def test_receipt_is_deterministic_modulo_timing():
    def stripped(receipt):
        volatile = {
            "elapsed_seconds",
            "sweep_elapsed_seconds",
            "throughput_candidates_per_second",
            "content_sha256",
        }
        return {key: value for key, value in receipt.items() if key not in volatile}

    first = run_screen(limit=65536, batch_size=16384, use_gpu=False)
    second = run_screen(limit=65536, batch_size=16384, use_gpu=False)
    assert canonical_sha256(stripped(first)) == canonical_sha256(stripped(second))
    third = run_screen(limit=65536, batch_size=8192, use_gpu=False)
    assert third["counts"] == first["counts"]
    assert canonical_sha256(stripped(third)) == canonical_sha256(stripped(first))


def test_reseal_tamper_fails_closed(small_receipt):
    tampered = dict(small_receipt)
    tampered["counts"] = {**small_receipt["counts"], "all_gate_passers": 999}
    with pytest.raises(ScreenedKernelError):
        validate_receipt(tampered)
    for key, value in (
        ("claims", {**CLAIMS, "observational_data_opened": True}),
        ("geometry_sha256", "0" * 64),
        ("environment_sha256", "0" * 64),
        ("v2_config_sha256", "0" * 64),
    ):
        tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
        tampered[key] = value
        tampered["content_sha256"] = canonical_sha256(tampered)
        with pytest.raises(ScreenedKernelError):
            validate_receipt(tampered)
    # Flipping a control verdict is caught by the control replay.
    controls = json.loads(json.dumps(small_receipt["controls"]))
    controls["sqrt_local"]["cluster"]["passes"] = True
    tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
    tampered["controls"] = controls
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ScreenedKernelError):
        validate_receipt(tampered)
    # Rewriting the embedded-v2 reproduction is caught by its own replay.
    reproduction = json.loads(json.dumps(small_receipt["embedded_v2_reproduction"]))
    reproduction["cluster_solar_safe"]["observed"]["cluster_dev"] = "1.000000000e-03"
    tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
    tampered["embedded_v2_reproduction"] = reproduction
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ScreenedKernelError):
        validate_receipt(tampered)


def test_claims_boundary():
    assert CLAIMS["observational_data_opened"] is False
    assert CLAIMS["survivor_is_validated_theory"] is False
    assert CLAIMS["synthetic_controls_only"] is True
    assert CLAIMS["embedded_v2_family_reproduces_prior_negative"] is True
    assert CLAIMS["first_principles_derivation_claimed"] is False
    assert CLAIMS["screening_is_phenomenological_not_derived"] is True
    assert CLAIMS["solar_probe_ambient_density_is_a_declared_assumption"] is True
    assert CLAIMS["invisible_mass_used_as_target_or_rescue"] is False
    assert CLAIMS["per_object_free_parameters_expressible"] is False
    assert CLAIMS["sealed_validation_ladder_bypassed"] is False


def test_system_caps_fail_closed():
    with pytest.raises(ScreenedKernelError):
        run_screen(limit=1024, batch_size=512, use_gpu=False)
    with pytest.raises(ScreenedKernelError):
        run_screen(limit=1024, batch_size=1 << 24, use_gpu=False)
    with pytest.raises(ScreenedKernelError):
        run_screen(limit=0, use_gpu=False)
    with pytest.raises(ScreenedKernelError):
        run_screen(limit=1024, pareto_cap=65, use_gpu=False)


def test_cli_end_to_end(tmp_path):
    output = tmp_path / "receipt.json"
    precompute = tmp_path / "precompute.json"
    assert (
        main(
            [
                "--limit", "16384", "--batch-size", "16384", "--cpu",
                "--output", str(output), "--precompute-output", str(precompute),
            ]
        )
        == 0
    )
    assert main(["--validate-checked", "--output", str(output),
                 "--precompute-output", str(precompute)]) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["decision"] = "everything is fine"
    output.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ScreenedKernelError):
        main(["--validate-checked", "--output", str(output)])
    with pytest.raises(ScreenedKernelError):
        main(["--validate-checked"])
