"""Gates for the nonlocal-kernel gravity screen (v2).

The tests pin: the declared family size and codec; the geometry validation (the Newton
kernel reproduces the frozen g_bar grids to 1e-10 — the claim all geometry stands on);
quadrature convergence for the two hand-built kernels; the two-ways boost agreement
(basis tables vs direct 50-digit node sums); the embedded-pointwise reproduction of the
prior sealed negatives (pure-local candidates still fail clusters, with the prior
screens' recorded margins); the Newton-embedding standard control; the Yukawa-safety
funnel from both sides; the fp32/fp64 slack sandwich; receipt determinism and
reseal-tamper behavior; the covariant-lift emitter; and the claim boundary.  GPU tests
skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler import gpu_baryonic_kernel_screen as ks
from sigma_theory_compiler.gpu_baryonic_kernel_screen import (
    AXES,
    AXIS_SIZES,
    CLAIMS,
    CONTROL_ORDINALS,
    FAMILY_SIZE,
    KERNEL_CONFIG,
    KernelScreenError,
    active_parameter_count,
    boost_exact,
    build_exact_pack,
    build_kernel_geometry,
    build_precompute_receipt,
    build_sweep_tables,
    decode_ordinal,
    emit_covariant_lift,
    encode_indices,
    encode_named,
    evaluate_batch,
    evaluate_candidate_exact,
    geometry_newton_residuals,
    geometry_sha256,
    main,
    passer_families,
    render_candidate,
    run_screen,
    validate_precompute_receipt,
    validate_receipt,
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
def geometry():
    return build_kernel_geometry()


@pytest.fixture(scope="module")
def pack(geometry):
    return build_exact_pack(geometry)


@pytest.fixture(scope="module")
def context(geometry, pack):
    return {"geometry": geometry, "pack": pack}


@pytest.fixture(scope="module")
def tables(geometry, pack):
    return build_sweep_tables(geometry, pack)


@pytest.fixture(scope="module")
def cpu_tables(tables):
    return ks._device_tables(tables, np)


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
    assert FAMILY_SIZE == 3 * 49 * 12 * 49 * 12 * 16 * 8 == 132_765_696
    assert 10**8 <= FAMILY_SIZE < 10**9
    assert KERNEL_CONFIG["family_size"] == FAMILY_SIZE
    assert tuple(len(AXES[name]) for name in ks.AXIS_ORDER) == AXIS_SIZES


def test_codec_roundtrip():
    rng = np.random.default_rng(3)
    for ordinal in rng.integers(0, FAMILY_SIZE, size=64):
        decoded = decode_ordinal(int(ordinal))
        assert encode_indices(decoded["indices"]) == int(ordinal)
    with pytest.raises(KernelScreenError):
        decode_ordinal(FAMILY_SIZE)
    with pytest.raises(KernelScreenError):
        decode_ordinal(-1)
    with pytest.raises(KernelScreenError):
        encode_named(L1="5")  # off the grid
    # Newton is expressible: identity local, zero kernel.
    newton = decode_ordinal(encode_named())
    assert newton["values"]["w_yukawa"] == "0"
    assert newton["values"]["w_power"] == "0"
    assert "conv(rho_b, K)" in render_candidate(newton)


# ---------------------------------------------------------------------------
# Geometry: the Newton validation the whole build stands on
# ---------------------------------------------------------------------------


def test_newton_kernel_reproduces_frozen_gbar_grids(geometry):
    budget = mp.mpf(KERNEL_CONFIG["geometry"]["newton_validation_max_relative_error"])
    residuals = geometry_newton_residuals(geometry)
    assert set(residuals) == {"disk", "hernquist", "cluster"}
    for system, residual in residuals.items():
        assert mp.mpf(residual) <= budget, (system, residual)


def test_geometry_binds_the_frozen_probe_tables(geometry, pack):
    # The exact pack builds only if the disk g_bar floats, the recovered lensing
    # radii, and the cluster probe table all replay; reaching here is the assertion.
    assert len(pack["lensing"]) == 15
    assert all(len(pair["nodes"]) == 33 for pair in pack["lensing"])
    assert len(pack["cluster"]) == 5
    assert [len(rows) for rows in (geometry["disk"], geometry["cluster"])] == [5, 5]
    assert len(geometry["hernquist"]) == 165


def test_kernel_quadrature_convergence(precompute_receipt):
    budget = mp.mpf(KERNEL_CONFIG["geometry"]["kernel_convergence_budget"])
    convergence = precompute_receipt["kernel_convergence"]
    for name in ("yukawa_hand", "power_hand"):
        assert mp.mpf(convergence[name]) <= budget, (name, convergence[name])


def test_two_ways_boost_agreement(geometry, tables):
    """A hand-built kernel's B computed via basis tables and via direct 50-digit
    node sums must agree; this pins the basis-table compilation."""

    yukawa = decode_ordinal(CONTROL_ORDINALS["yukawa_hand"])["values"]
    index = AXES["L1"].index(yukawa["L1"])
    exact = boost_exact(geometry["cluster"], yukawa)
    for k in range(5):
        assert abs(float(tables["BY_clu"][k][index]) / float(exact[k]) - 1) < 1e-12
    power = decode_ordinal(CONTROL_ORDINALS["power_hand"])["values"]
    combo = (
        AXES["L2"].index(power["L2"]) * len(AXES["p"]) + AXES["p"].index(power["p"])
    ) * len(AXES["t"]) + AXES["t"].index(power["t"])
    for system, table_name in (("disk", "BP_gal"), ("cluster", "BP_clu")):
        exact = boost_exact(geometry[system], power)
        for k in range(5):
            assert abs(float(tables[table_name][k][combo]) / float(exact[k]) - 1) < 1e-12


# ---------------------------------------------------------------------------
# Embedded pointwise family: the prior sealed negatives must replay inside v2
# ---------------------------------------------------------------------------


def test_embedded_pointwise_family_reproduces_prior_negative(context, cpu_tables):
    """Every pure-local candidate still fails the cluster gate, with the prior
    screens' recorded margins; the sqrt family still passes galaxy and lensing."""

    pure_local = [
        encode_named(local=value) for value in AXES["local"]
    ]
    batch = evaluate_batch(
        np,
        np.asarray(pure_local, dtype=np.int64),
        cpu_tables,
        dtype=np.float64,
        tier="fp64_thresholds",
    )
    assert not batch["cluster_pass"].any()
    sqrt_verdict = evaluate_candidate_exact(CONTROL_ORDINALS["sqrt_local"], context)
    assert sqrt_verdict["galaxy"]["passes"] is True
    assert sqrt_verdict["lensing"]["passes"] is True
    assert sqrt_verdict["cluster"]["passes"] is False
    # The prior screen's recorded sqrt-family margins replay inside v2.
    assert abs(float(sqrt_verdict["lensing"]["worst_consistency"]) - 0.118) < 5e-3
    assert float(sqrt_verdict["cluster"]["closest_probe_deviation"]) > 0.3
    linear = evaluate_candidate_exact(CONTROL_ORDINALS["linear_u_local"], context)
    assert linear["galaxy"]["passes"] is True
    assert linear["lensing"]["passes"] is False
    assert abs(float(linear["lensing"]["worst_consistency"]) - 0.158) < 5e-3
    assert CLAIMS["embedded_pointwise_family_reproduces_prior_negative"] is True


def test_newton_embedding_standard_control(context):
    """Zero-weight kernel + identity local = pure Newton: passes the Solar gates,
    fails the galaxy gate — the standard control of every screen in this repo."""

    verdict = evaluate_candidate_exact(CONTROL_ORDINALS["newton_identity"], context)
    assert verdict["newton"]["passes"] is True
    assert verdict["newton"]["safety_passes"] is True
    assert verdict["galaxy"]["passes"] is False
    assert verdict["lensing"]["passes"] is False
    assert verdict["cluster"]["passes"] is False
    assert float(verdict["cluster"]["shortfall_min"]) >= 1.5


def test_yukawa_safety_probe_cuts_both_ways(context):
    """An unscreened O(1) Yukawa dies at the safety probe; the rising power tail
    passes every Solar gate — the funnel is selective, not absolute."""

    yukawa = evaluate_candidate_exact(CONTROL_ORDINALS["yukawa_hand"], context)
    assert yukawa["newton"]["safety_passes"] is False
    assert float(yukawa["newton"]["safety_margin"]) > 0.9
    power = evaluate_candidate_exact(CONTROL_ORDINALS["power_hand"], context)
    assert power["newton"]["passes"] is True
    assert float(power["newton"]["safety_margin"]) < 1e-8
    # And the power kernel has real amplitude at cluster scales: it overshoots.
    assert float(power["cluster"]["shortfall_min"]) < 1


# ---------------------------------------------------------------------------
# Batched path agrees with the exact layer; fp32 slack is a true superset
# ---------------------------------------------------------------------------


def _sample_ordinals(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, FAMILY_SIZE, size=count)
    return np.unique(
        np.concatenate([sample, np.asarray(list(CONTROL_ORDINALS.values()))])
    ).astype(np.int64)


def test_fp64_batch_decisions_match_exact_layer(context, cpu_tables):
    ordinals = _sample_ordinals(12, seed=11)
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


def test_fp32_slack_never_drops_a_strict_survivor(cpu_tables):
    ordinals = _sample_ordinals(4096, seed=7)
    fp32 = evaluate_batch(
        np, ordinals, cpu_tables, dtype=np.float32, tier="fp32_thresholds"
    )
    fp64 = evaluate_batch(
        np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds"
    )
    for gate in ("newton", "galaxy", "lensing", "cluster", "all"):
        assert not np.any(fp64[f"{gate}_pass"] & ~fp32[f"{gate}_pass"]), gate


def test_gpu_and_cpu_decisions_agree_on_a_sample(tables, cpu_tables):
    cupy = _cupy_or_none()
    if cupy is None:
        pytest.skip("no CUDA device")
    ordinals = _sample_ordinals(2048, seed=5)
    cpu = evaluate_batch(
        np, ordinals, cpu_tables, dtype=np.float64, tier="fp64_thresholds"
    )
    gpu = evaluate_batch(
        cupy,
        cupy.asarray(ordinals),
        ks._device_tables(tables, cupy),
        dtype=cupy.float64,
        tier="fp64_thresholds",
    )
    for gate in ("newton", "safety", "galaxy", "lensing", "cluster", "all"):
        assert int((cpu[f"{gate}_pass"] != gpu[f"{gate}_pass"].get()).sum()) == 0, gate


# ---------------------------------------------------------------------------
# Survivor structure helpers
# ---------------------------------------------------------------------------


def test_passer_families_group_grid_neighbors():
    base = decode_ordinal(encode_named(w_power="1", L2="2", p="2", t="-1"))["indices"]
    neighbor = list(base)
    neighbor[3] += 1  # w_power one grid step
    diagonal = list(base)
    diagonal[3] += 1
    diagonal[5] += 1  # w_power and p one step each: Chebyshev distance 1
    far = list(base)
    far[3] += 5
    families = passer_families(
        [
            encode_indices(base),
            encode_indices(neighbor),
            encode_indices(diagonal),
            encode_indices(far),
        ]
    )
    assert sorted(len(f) for f in families) == [1, 3]


def test_covariant_lift_candidate_blocks():
    values = decode_ordinal(
        encode_named(
            local="sqrt_one_plus_u_squared", w_yukawa="1/2", L1="8", w_power="2", L2="2",
            p="2", t="-1",
        )
    )["values"]
    lift = emit_covariant_lift(values)
    assert lift["claims"] == {"first_principles_derivation_pending": True}
    mechanisms = [c["mechanism"] for c in lift["components"]]
    assert mechanisms == [
        "massive_scalar_exchange",
        "nonlocal_propagator_correction",
        "pointwise_modified_dynamics",
    ]
    yukawa = lift["components"][0]["field_theory_ansatz"]
    assert "m = 1/L1 = 1/8" in yukawa
    power = lift["components"][1]["field_theory_ansatz"]
    assert "alpha = 1 - t/2 = 3/2" in power  # t = -1: MOND-like 1/s tail
    assert "s^(-1)" in power
    assert active_parameter_count(values) == 7
    assert active_parameter_count(decode_ordinal(encode_named())["values"]) == 0


# ---------------------------------------------------------------------------
# Receipt: end-to-end small run, determinism, reseal-tamper, claims
# ---------------------------------------------------------------------------


def test_small_run_receipt_shape_and_funnel_honesty(small_receipt):
    counts = small_receipt["counts"]
    assert counts["processed"] == 131072
    assert counts["family_size"] == FAMILY_SIZE
    assert counts["exact_confirmed"] + counts["exact_refuted"] == counts["exact_verified"]
    # Funnel honesty: strict fp64 passers are a subset of the fp32 slack tier.
    assert counts["all_gate_passers"] <= counts["fp32"]["all"]
    for gate in ("newton", "galaxy", "lensing", "cluster"):
        assert counts["fp64_of_fp32_survivors"][gate] <= counts["fp32"][gate]
    controls = small_receipt["controls"]
    assert controls["newton_identity"]["cluster"]["passes"] is False
    assert controls["sqrt_local"]["cluster"]["passes"] is False
    assert controls["sqrt_local"]["lensing"]["passes"] is True
    assert controls["yukawa_hand"]["newton"]["safety_passes"] is False
    if counts["all_gate_passers"] == 0:
        negative = small_receipt["sealed_negative"]
        assert negative["sealed"] is True
        assert small_receipt["decision"].startswith("SCREENED-SEALED-NEGATIVE")
        assert "needed:" in negative["structural_direction"]
    else:
        assert small_receipt["passer_families_reported"]
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
    # Batch size must not change any scientific count or tracked exemplar.
    third = run_screen(limit=65536, batch_size=8192, use_gpu=False)
    assert third["counts"] == first["counts"]
    assert canonical_sha256(stripped(third)) == canonical_sha256(stripped(first))


def test_reseal_tamper_fails_closed(small_receipt):
    # Plain tamper: the seal itself catches it.
    tampered = dict(small_receipt)
    tampered["counts"] = {**small_receipt["counts"], "all_gate_passers": 999}
    with pytest.raises(KernelScreenError):
        validate_receipt(tampered)
    # Reseal after tampering with the claims: the claims boundary catches it.
    tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
    tampered["claims"] = {**CLAIMS, "observational_data_opened": True}
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(KernelScreenError):
        validate_receipt(tampered)
    # Reseal after tampering with the geometry hash: the geometry replay catches it.
    tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
    tampered["geometry_sha256"] = "0" * 64
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(KernelScreenError):
        validate_receipt(tampered)
    # Reseal after flipping a control verdict: the control replay catches it.
    controls = json.loads(json.dumps(small_receipt["controls"]))
    controls["sqrt_local"]["cluster"]["passes"] = True
    tampered = {k: v for k, v in small_receipt.items() if k != "content_sha256"}
    tampered["controls"] = controls
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(KernelScreenError):
        validate_receipt(tampered)


def test_precompute_receipt_validates_and_tampers_closed(precompute_receipt):
    assert precompute_receipt["schema_version"] == ks.PRECOMPUTE_SCHEMA
    validate_precompute_receipt(precompute_receipt)
    for system in ("disk", "hernquist", "cluster"):
        budget = mp.mpf(KERNEL_CONFIG["geometry"]["newton_validation_max_relative_error"])
        recorded = precompute_receipt["systems"][system]["newton_max_relative_error"]
        assert mp.mpf(recorded) <= budget
    tampered = {k: v for k, v in precompute_receipt.items() if k != "content_sha256"}
    tampered["systems"] = json.loads(json.dumps(precompute_receipt["systems"]))
    tampered["systems"]["disk"]["newton_max_relative_error"] = "1e-20"
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(KernelScreenError):
        validate_precompute_receipt(tampered)


def test_geometry_sha_pins_the_frozen_nodes(geometry, precompute_receipt):
    assert precompute_receipt["geometry_sha256"] == geometry_sha256(geometry)


def test_claims_boundary():
    assert CLAIMS["observational_data_opened"] is False
    assert CLAIMS["survivor_is_validated_theory"] is False
    assert CLAIMS["synthetic_controls_only"] is True
    assert CLAIMS["embedded_pointwise_family_reproduces_prior_negative"] is True
    assert CLAIMS["first_principles_derivation_claimed"] is False
    assert CLAIMS["invisible_mass_used_as_target_or_rescue"] is False
    assert CLAIMS["kernel_scales_share_code_units_across_systems"] is True
    assert CLAIMS["per_object_free_parameters_expressible"] is False
    assert CLAIMS["sealed_validation_ladder_bypassed"] is False


def test_system_caps_fail_closed():
    with pytest.raises(KernelScreenError):
        run_screen(limit=1024, batch_size=512, use_gpu=False)
    with pytest.raises(KernelScreenError):
        run_screen(limit=1024, batch_size=1 << 24, use_gpu=False)
    with pytest.raises(KernelScreenError):
        run_screen(limit=0, use_gpu=False)
    with pytest.raises(KernelScreenError):
        run_screen(limit=1024, pareto_cap=65, use_gpu=False)
    with pytest.raises(KernelScreenError):
        build_kernel_geometry(refine=3)


def test_cli_end_to_end(tmp_path):
    output = tmp_path / "receipt.json"
    assert main(["--limit", "16384", "--batch-size", "16384", "--cpu", "--output", str(output)]) == 0
    assert main(["--validate-checked", "--output", str(output)]) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["decision"] = "everything is fine"
    output.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(KernelScreenError):
        main(["--validate-checked", "--output", str(output)])
    with pytest.raises(KernelScreenError):
        main(["--validate-checked"])
