"""Gates for the P1 lensing + P2 cluster GPU screen.

The tests pin the physics known-answer controls (Newton fails both gates with the
recorded shortfall, the MOND-like sqrt family passes lensing but fails clusters, the
linear-u galaxy survivor fails lensing), the quadrature convergence claim, the frozen
cluster table, the codec reuse (no fork), the fp32/fp64 slack sandwich, receipt
determinism and reseal-tamper behavior, and the claim boundary.  GPU tests skip
cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler import gpu_baryonic_interpolation_screen as screen
from sigma_theory_compiler import gpu_baryonic_lensing_cluster_screen as gates
from sigma_theory_compiler.gpu_baryonic_interpolation_screen import (
    build_probe_grid,
    encode_candidate,
    verify_candidate_exact,
)
from sigma_theory_compiler.gpu_baryonic_lensing_cluster_screen import (
    CLAIMS,
    CONTROL_ORDINALS,
    GATE_CONFIG,
    LensingClusterScreenError,
    _compile_grids,
    _exact_context,
    _fraction,
    build_lensing_grid,
    evaluate_gate_batch,
    main,
    recompute_cluster_table,
    run_screen,
    validate_receipt,
    verify_candidate_exact_gates,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


@pytest.fixture(scope="module")
def lensing_grid():
    return build_lensing_grid()


@pytest.fixture(scope="module")
def compiled(lensing_grid):
    return _compile_grids(lensing_grid)


@pytest.fixture(scope="module")
def exact_context(lensing_grid):
    return _exact_context(lensing_grid, build_probe_grid())


@pytest.fixture(scope="module")
def small_receipt():
    return run_screen(limit=32768, batch_size=8192, use_gpu=False)


EXTRA_ORDINALS = {
    "cbrt_family": encode_candidate(0, [0, 0, 1, 0, 0], [0] * 5),
    "overboost": encode_candidate(2, [0, 1, 0, 0, 0], [0] * 5),
}


# ---------------------------------------------------------------------------
# Codec reuse: the ordinal family is the screen's, not a fork
# ---------------------------------------------------------------------------


def test_codec_is_imported_not_forked():
    assert gates.decode_ordinal is screen.decode_ordinal
    assert gates.encode_candidate is screen.encode_candidate
    assert gates._digits_from_ordinals is screen._digits_from_ordinals
    assert gates._nu_at is screen._nu_at
    assert gates.FAMILY_SIZE is screen.FAMILY_SIZE
    assert GATE_CONFIG["lensing"]["masses"] == screen.SCREEN_CONFIG["disk_masses"]


# ---------------------------------------------------------------------------
# Frozen grids: deep regime, quadrature convergence, cluster table replay
# ---------------------------------------------------------------------------


def test_lensing_paths_stay_in_the_deep_regime(lensing_grid):
    ceiling = float(GATE_CONFIG["lensing"]["deep_y_ceiling"])
    assert float(lensing_grid["max_path_y"]) < ceiling
    assert len(lensing_grid["integrals"]) == 15
    for integral in lensing_grid["integrals"]:
        assert len(integral["nodes"]) == GATE_CONFIG["lensing"]["path_nodes"]
        for node in integral["nodes"]:
            assert float(node["y"]) < ceiling


def _alpha_from_grid(grid, ordinal, mass_index, impact):
    """Deflection from the frozen decimal nodes at 50 digits."""

    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    candidate = gates.decode_ordinal(ordinal)
    for integral in grid["integrals"]:
        if integral["mass_index"] == mass_index and integral["impact_parameter"] == impact:
            total = mp.mpf(0)
            for node in integral["nodes"]:
                nu = gates._nu_exact(candidate, mp.mpf(node["y"]))
                total += mp.mpf(node["weight"]) * nu
            return total
    raise AssertionError("integral not found")


def test_quadrature_converged_against_denser_grid(lensing_grid):
    """33 nodes vs 129 nodes: every integral within the declared 0.5% budget."""

    dense = build_lensing_grid(path_nodes=129)
    for name in ("newton_nu1", "sqrt_family", "cbrt_family"):
        ordinal = CONTROL_ORDINALS.get(name, EXTRA_ORDINALS.get(name))
        for integral in lensing_grid["integrals"]:
            mass_index = integral["mass_index"]
            impact = integral["impact_parameter"]
            coarse = _alpha_from_grid(lensing_grid, ordinal, mass_index, impact)
            fine = _alpha_from_grid(dense, ordinal, mass_index, impact)
            assert abs(coarse / fine - 1) < mp.mpf("5e-3"), (name, mass_index, impact)


def test_quadrature_matches_raw_mpmath_quad(lensing_grid):
    """The frozen Simpson rule reproduces mp.quad of the declared -L..L formula."""

    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    length = mp.mpf(GATE_CONFIG["lensing"]["path_half_length"])
    for name, mass_index, impact in (
        ("sqrt_family", 2, 8),
        ("sqrt_family", 0, 20),
        ("newton_nu1", 0, 8),
    ):
        ordinal = CONTROL_ORDINALS[name]
        candidate = gates.decode_ordinal(ordinal)
        mass = _fraction(GATE_CONFIG["lensing"]["masses"][mass_index])
        b = mp.mpf(impact)

        def integrand(el, _mass=mass, _b=b, _candidate=candidate):
            radius = mp.sqrt(_b**2 + el**2)
            y = _mass / (radius + 1) ** 2
            return gates._nu_exact(_candidate, y) * y * (_b / radius)

        reference = 2 * mp.quad(integrand, [-length, -b, 0, b, length])
        frozen = _alpha_from_grid(lensing_grid, ordinal, mass_index, impact)
        assert abs(frozen / reference - 1) < mp.mpf("5e-3"), (name, mass_index, impact)


def test_cluster_table_replays_at_50_digits():
    table = recompute_cluster_table()
    assert table["gbar_50dps"] == GATE_CONFIG["cluster"]["gbar_50dps"]
    # g_dyn = 2*T0*r/(1+r^2) is exactly rational at the frozen probes.
    t0 = GATE_CONFIG["cluster"]["temperature_T0"]
    from fractions import Fraction

    for radius_text, expected in zip(
        GATE_CONFIG["cluster"]["probe_radii"], GATE_CONFIG["cluster"]["gdyn_exact"], strict=True
    ):
        radius = Fraction(radius_text)
        assert Fraction(expected) == 2 * t0 * radius / (1 + radius**2)


# ---------------------------------------------------------------------------
# Physics known-answer controls (exact 50-digit layer)
# ---------------------------------------------------------------------------


def test_newton_fails_cluster_with_recorded_shortfall(exact_context):
    """nu = 1 is the cluster missing-mass problem itself: shortfall >= 1.5 recorded."""

    verdict = verify_candidate_exact_gates(CONTROL_ORDINALS["newton_nu1"], exact_context)
    assert verdict["cluster"]["passes"] is False
    assert float(verdict["cluster"]["shortfall_min"]) >= 1.5
    assert float(verdict["cluster"]["max_deviation"]) > 0.5
    # Newtonian deflection falls off as 1/b: the profile cannot be flat either.
    assert verdict["lensing"]["passes"] is False
    assert float(verdict["lensing"]["worst_flatness"]) > float(
        GATE_CONFIG["lensing"]["fp64_thresholds"]["flatness"]
    )


def test_sqrt_family_passes_lensing_and_fails_cluster(exact_context):
    """The MOND-like survivor: lensing consistency is nearly automatic, clusters kill it."""

    verdict = verify_candidate_exact_gates(CONTROL_ORDINALS["sqrt_family"], exact_context)
    assert verdict["lensing"]["passes"] is True
    assert float(verdict["lensing"]["worst_consistency"]) <= 0.15
    assert verdict["cluster"]["passes"] is False
    # The documented cluster shortfall: even the closest probe misses by > 2x tolerance.
    assert float(verdict["cluster"]["closest_probe_deviation"]) > 0.3
    assert verdict["both_pass"] is False


def test_linear_u_flattens_curves_but_fails_lensing(exact_context):
    """nu = 1 + u passes the galaxy screen yet fails dynamics-lensing consistency."""

    ordinal = CONTROL_ORDINALS["linear_u"]
    galaxy = verify_candidate_exact(ordinal, build_probe_grid())
    assert galaxy["passes"] is True
    verdict = verify_candidate_exact_gates(ordinal, exact_context)
    assert verdict["lensing"]["passes"] is False
    assert float(verdict["lensing"]["worst_consistency"]) > 0.15
    assert verdict["cluster"]["passes"] is False


def test_cbrt_family_matches_sqrt_family_behavior(exact_context):
    verdict = verify_candidate_exact_gates(EXTRA_ORDINALS["cbrt_family"], exact_context)
    assert verdict["lensing"]["passes"] is True
    assert verdict["cluster"]["passes"] is False


# ---------------------------------------------------------------------------
# Batched path agrees with the exact layer; fp32 slack is a true superset
# ---------------------------------------------------------------------------


ALL_CONTROLS = {**CONTROL_ORDINALS, **EXTRA_ORDINALS}


def test_fp64_batch_decisions_match_exact_layer(compiled, exact_context):
    ordinals = np.array(list(ALL_CONTROLS.values()), dtype=np.int64)
    batch = evaluate_gate_batch(
        np, ordinals, compiled, dtype=np.float64, tier="fp64_thresholds"
    )
    for index, (name, ordinal) in enumerate(ALL_CONTROLS.items()):
        verdict = verify_candidate_exact_gates(ordinal, exact_context)
        assert bool(batch["lensing_pass"][index]) == verdict["lensing"]["passes"], name
        assert bool(batch["cluster_pass"][index]) == verdict["cluster"]["passes"], name


def test_fp32_slack_never_drops_a_strict_survivor(compiled):
    rng = np.random.default_rng(11)
    sample = np.sort(rng.choice(gates.FAMILY_SIZE, size=4096, replace=False))
    ordinals = np.concatenate(
        [np.array(list(ALL_CONTROLS.values()), dtype=np.int64), sample.astype(np.int64)]
    )
    fp32 = evaluate_gate_batch(np, ordinals, compiled, dtype=np.float32, tier="fp32_thresholds")
    fp64 = evaluate_gate_batch(np, ordinals, compiled, dtype=np.float64, tier="fp64_thresholds")
    assert not np.any(fp64["lensing_pass"] & ~fp32["lensing_pass"])
    assert not np.any(fp64["cluster_pass"] & ~fp32["cluster_pass"])


def test_gpu_and_cpu_decisions_agree_on_a_sample(compiled):
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(gates.FAMILY_SIZE, size=2048, replace=False)).astype(np.int64)
    sample = np.concatenate([sample, np.array(list(ALL_CONTROLS.values()), dtype=np.int64)])
    cpu = evaluate_gate_batch(np, sample, compiled, dtype=np.float64, tier="fp64_thresholds")
    gpu = evaluate_gate_batch(
        cupy, cupy.asarray(sample), compiled, dtype=cupy.float64, tier="fp64_thresholds"
    )
    assert int((cpu["lensing_pass"] != gpu["lensing_pass"].get()).sum()) == 0
    assert int((cpu["cluster_pass"] != gpu["cluster_pass"].get()).sum()) == 0


# ---------------------------------------------------------------------------
# Receipt: end-to-end small run, determinism, reseal-tamper, claims
# ---------------------------------------------------------------------------


def test_small_run_receipt_shape_and_controls(small_receipt):
    counts = small_receipt["counts"]
    assert counts["processed"] == 32768
    assert counts["family_size"] == gates.FAMILY_SIZE
    assert counts["exact_confirmed"] + counts["exact_refuted"] == counts["exact_verified"]
    assert counts["both_pass"] <= min(counts["lensing_pass"], counts["cluster_pass"])
    controls = small_receipt["controls"]
    assert controls["newton_nu1"]["cluster"]["passes"] is False
    assert float(controls["newton_nu1"]["cluster"]["shortfall_min"]) >= 1.5
    assert controls["sqrt_family"]["lensing"]["passes"] is True
    assert controls["sqrt_family"]["cluster"]["passes"] is False
    assert controls["linear_u"]["lensing"]["passes"] is False
    # Sealed negative: with zero joint survivors the decision records the margins.
    if counts["both_pass"] == 0:
        assert small_receipt["cluster_negative"] is not None
        assert small_receipt["cluster_negative"]["sealed"] is True
        assert small_receipt["decision"].startswith("SCREENED-SEALED-NEGATIVE")
    validate_receipt(small_receipt)


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
        volatile = {"elapsed_seconds", "throughput_candidates_per_second", "content_sha256"}
        return {key: value for key, value in receipt.items() if key not in volatile}

    first = run_screen(limit=8192, batch_size=4096, use_gpu=False)
    second = run_screen(limit=8192, batch_size=4096, use_gpu=False)
    assert canonical_sha256(stripped(first)) == canonical_sha256(stripped(second))
    # Batch size must not change any scientific count (the closest-cluster exemplar may
    # legitimately refine differently because top-k tracking is per batch).
    third = run_screen(limit=8192, batch_size=2048, use_gpu=False)
    assert third["counts"] == first["counts"]


def test_reseal_tamper_fails_closed(small_receipt):
    # Plain tamper: the seal itself catches it.
    tampered = dict(small_receipt)
    tampered["counts"] = {**small_receipt["counts"], "lensing_pass": 999}
    with pytest.raises(LensingClusterScreenError):
        validate_receipt(tampered)
    # Reseal after tampering with the claims: the claims boundary catches it.
    tampered = {key: value for key, value in small_receipt.items() if key != "content_sha256"}
    tampered["claims"] = {**CLAIMS, "observational_data_opened": True}
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(LensingClusterScreenError):
        validate_receipt(tampered)
    # Reseal after tampering with a frozen lensing node: the grid replay catches it.
    grids = json.loads(json.dumps(small_receipt["frozen_grids"]))
    grids["lensing"]["integrals"][0]["nodes"][0]["y"] = "0.001"
    tampered = {key: value for key, value in small_receipt.items() if key != "content_sha256"}
    tampered["frozen_grids"] = grids
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(LensingClusterScreenError):
        validate_receipt(tampered)
    # Reseal after flipping a control verdict: the control replay catches it.
    controls = json.loads(json.dumps(small_receipt["controls"]))
    controls["sqrt_family"]["lensing"]["passes"] = False
    tampered = {key: value for key, value in small_receipt.items() if key != "content_sha256"}
    tampered["controls"] = controls
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(LensingClusterScreenError):
        validate_receipt(tampered)


def test_claims_boundary():
    assert CLAIMS["observational_data_opened"] is False
    assert CLAIMS["per_object_free_parameters_expressible"] is False
    assert CLAIMS["lensing_prescription_is_an_assumption"] is True
    assert CLAIMS["survivor_is_validated_theory"] is False
    assert CLAIMS["cluster_negative_is_a_valid_deliverable"] is True
    assert CLAIMS["invisible_mass_used_as_target_or_rescue"] is False
    assert CLAIMS["sealed_validation_ladder_bypassed"] is False


def test_system_caps_fail_closed():
    with pytest.raises(LensingClusterScreenError):
        run_screen(limit=1024, batch_size=512, use_gpu=False)
    with pytest.raises(LensingClusterScreenError):
        run_screen(limit=1024, batch_size=1 << 25, use_gpu=False)
    with pytest.raises(LensingClusterScreenError):
        run_screen(limit=0, use_gpu=False)
    with pytest.raises(LensingClusterScreenError):
        run_screen(limit=1024, pareto_cap=65, use_gpu=False)
    with pytest.raises(LensingClusterScreenError):
        build_lensing_grid(path_nodes=34)  # even
    with pytest.raises(LensingClusterScreenError):
        build_lensing_grid(path_nodes=17)  # below the frozen default


def test_gpu_run_with_populated_front_validates_end_to_end():
    """Regression: a survivor-bearing range must yield a front of real ordinals.

    The first campaign died on a row-index/ordinal confusion that only a non-empty
    Pareto front could trigger; the CPU-sized ranges in this suite have empty fronts.
    The limit reaches just past the beta=1/3 all-zero-denominator region (baseline
    ordinal 141,237,624; the cbrt family sits at +49), which contains real P1 passers.
    """

    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    receipt = run_screen(limit=141_238_000, use_gpu=True)
    counts = receipt["counts"]
    assert counts["lensing_pass"] >= 1
    assert counts["pareto_front"] >= 1
    for entry in receipt["pareto_front"]:
        # A row index would be tiny; real survivors here live near 141.2M.
        assert entry["ordinal"] > 100_000_000
        assert entry["formula"] == gates.render_candidate(gates.decode_ordinal(entry["ordinal"]))
    assert receipt["crosscheck"]["performed"] is True
    assert receipt["crosscheck"]["lensing_disagreements"] == 0
    assert receipt["crosscheck"]["cluster_disagreements"] == 0
    validate_receipt(receipt)


def test_cli_end_to_end(tmp_path):
    output = tmp_path / "receipt.json"
    assert main(["--limit", "8192", "--cpu", "--output", str(output)]) == 0
    assert main(["--validate-checked", "--output", str(output)]) == 0
    # Tampered bytes must fail validation.
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["decision"] = "everything is fine"
    output.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(LensingClusterScreenError):
        main(["--validate-checked", "--output", str(output)])
    with pytest.raises(LensingClusterScreenError):
        main(["--validate-checked"])
