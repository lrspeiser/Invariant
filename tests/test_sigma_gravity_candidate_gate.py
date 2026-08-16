"""Gates for the Sigma-Gravity gauntlet and neighborhood scan.

The tests pin the declared unit mapping (g_dagger = 4/5 exactly, A_cluster ~ 8.446),
the reuse of the frozen synthetic grids (disk grid, Hernquist lensing nodes, cluster
table -- imported, never reinvented), the published point's per-gate verdicts with
their margins, the A0 = 0 Newton control (must fail flat curves, Tully-Fisher, and the
cluster), the impossibility of a C_cluster rescue for the synthetic cluster, the scan
codec, fp64/50-digit agreement, receipt determinism, and reseal-tamper behavior.  GPU
tests skip cleanly when no CUDA device is present.
"""

from __future__ import annotations

import json

import mpmath as mp
import numpy as np
import pytest

from sigma_theory_compiler.gpu_baryonic_interpolation_screen import build_probe_grid
from sigma_theory_compiler.gpu_baryonic_lensing_cluster_screen import GATE_CONFIG
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sigma_gravity_candidate_gate import (
    CANDIDATE_CONFIG,
    CLAIMS_GAUNTLET,
    CLAIMS_SCAN,
    SCAN_CONFIG,
    TOTAL_SCAN_CANDIDATES,
    SigmaGravityGateError,
    build_control_pack,
    build_float_pack,
    decode_scan_ordinal,
    encode_scan_indices,
    evaluate_batch,
    evaluate_candidate_exact,
    evaluate_params_batch,
    main,
    published_exact_parameters,
    published_params_row,
    run_gauntlet,
    scan_neighborhood,
    validate_gauntlet_receipt,
    validate_neighborhood_receipt,
)


def _cupy_or_none():
    try:
        import cupy

        cupy.arange(4).sum()
        return cupy
    except Exception:  # noqa: BLE001 - any CUDA absence means skip
        return None


@pytest.fixture(scope="module")
def pack():
    return build_control_pack()


@pytest.fixture(scope="module")
def floats(pack):
    return build_float_pack(pack)


@pytest.fixture(scope="module")
def published(pack):
    return evaluate_candidate_exact(pack=pack)


@pytest.fixture(scope="module")
def gauntlet_receipt():
    return run_gauntlet()


SMALL_SCAN = 5720  # A0 = 0.8, n = 0.00, g_dagger in {0.4, 0.5}, all C_cluster, p, q


@pytest.fixture(scope="module")
def small_scan():
    return scan_neighborhood(use_gpu=False, limit=SMALL_SCAN)


# ---------------------------------------------------------------------------
# Unit mapping and frozen-grid reuse
# ---------------------------------------------------------------------------


def test_unit_mapping_exact_constants():
    mp.mp.dps = 50
    exact = published_exact_parameters()
    assert exact["g_dagger"] == mp.mpf(4) / 5  # 9.6e-11 / 1.2e-10, exact
    assert abs(exact["A0"] - mp.e ** (1 / (2 * mp.pi))) == 0
    assert mp.mpf("1.172") < exact["A0"] < mp.mpf("1.173")
    a_cluster = exact["A0"] * mp.mpf(1500) ** exact["n"]
    assert mp.mpf("8.44") < a_cluster < mp.mpf("8.45")
    assert exact["p"] == mp.mpf(1) / 2
    assert exact["q"] == 1
    assert exact["C_cluster"] == 1


def test_control_pack_reuses_frozen_grids(pack):
    mp.mp.dps = 50
    assert pack["xi"] == 1 / (2 * mp.pi)
    grid = build_probe_grid()
    frozen = sorted(
        (mp.mpf(point["gbar"]), point["radius"], index)
        for index, disk in enumerate(grid["disks"])
        for point in disk["points"]
    )
    assert len(pack["pooled"]) == 30
    for (gbar, radius, window, disk_index), (g_ref, r_ref, d_ref) in zip(
        pack["pooled"], frozen, strict=True
    ):
        assert gbar == g_ref and radius == r_ref and disk_index == d_ref
        assert 0 < window < 1
    assert len(pack["lensing"]) == 15
    scale = GATE_CONFIG["lensing"]["hernquist_scale"]
    for integral in pack["lensing"]:
        assert len(integral["nodes"]) == GATE_CONFIG["lensing"]["path_nodes"]
        mass_num, mass_den = integral["mass_text"].split("/")
        mass = mp.mpf(mass_num) / mp.mpf(mass_den)
        for y, _, window in integral["nodes"][:3]:
            radius = mp.sqrt(mass / y) - scale
            assert abs(mass / (radius + scale) ** 2 - y) < mp.mpf("1e-45")
            assert 0 < window < 1
    for (gbar, gdyn), frozen_text in zip(
        pack["cluster"], GATE_CONFIG["cluster"]["gbar_50dps"], strict=True
    ):
        assert gbar == mp.mpf(frozen_text)
        assert gdyn > 0
    assert len(pack["monotone_c1_y"]) == len(grid["monotone_y"])


# ---------------------------------------------------------------------------
# Published point: per-gate verdicts with margins
# ---------------------------------------------------------------------------


def test_published_newton_and_solar(published):
    newton = published["galaxy"]["newton"]
    assert newton["near"]["pass"] and newton["far"]["pass"]
    assert float(newton["near"]["error"]) < 1e-5
    assert float(newton["far"]["error"]) < 1e-8
    assert published["solar_proxy"]["report_only"] is True
    assert float(published["solar_proxy"]["sigma_minus_1"]) < 1e-11


def test_published_monotonicity_split(published):
    """C = 1 monotonicity passes; the pooled disk RAR is genuinely two-valued."""

    assert published["galaxy"]["monotone_c1"]["pass"] is True
    rar = published["galaxy"]["monotone_disk_rar"]
    assert rar["pass"] is False
    assert rar["per_disk_pass"] is True
    assert len(rar["violations"]) >= 1
    for violation in rar["violations"]:
        # Every inversion is cross-disk: a lighter disk's low-W inner point against a
        # heavier disk's high-W outer point at nearly equal g_bar.
        assert violation["before"]["disk"] != violation["after"]["disk"]
        assert float(violation["geff_ratio_minus_1"]) < 0
    assert float(rar["min_pair_margin"]) < -0.05


def test_published_flat_curves_and_btfr(published):
    flat = published["galaxy"]["flat_outer_curves"]
    assert flat["pass"] is True
    assert all(entry["pass"] for entry in flat["per_disk"])
    btfr = published["galaxy"]["btfr"]
    assert btfr["pass"] is True
    assert 3.85 < float(btfr["slope"]) < 3.95


def test_published_lensing_passes_with_thin_margin(published):
    lensing = published["lensing"]
    assert lensing["pass"] is True
    assert float(lensing["worst_flatness"]) < 0.08
    assert 0.14 < float(lensing["worst_consistency"]) < 0.15


def test_published_cluster_fails_two_sided(published):
    cluster = published["cluster"]
    assert cluster["pass"] is False
    assert 0.9 < float(cluster["max_deviation"]) < 0.95
    ratios = [float(text) for text in cluster["ratio_by_probe"]]
    assert len(ratios) == 5
    assert ratios[0] < 1 and ratios[1] < 1  # inner probes undershoot
    assert ratios[-1] > 1.5  # outermost probe overshoots
    assert published["verdict"]["all_pass"] is False
    assert published["verdict"]["galaxy_pass"] is False
    assert published["verdict"]["lensing_pass"] is True


def test_newton_a0_zero_control(pack):
    control = evaluate_candidate_exact({"A0": "0"}, pack=pack)
    galaxy = control["galaxy"]
    assert galaxy["newton"]["near"]["pass"] and galaxy["newton"]["far"]["pass"]
    assert galaxy["monotone_disk_rar"]["pass"] is True
    assert galaxy["flat_outer_curves"]["pass"] is False
    assert galaxy["btfr"]["pass"] is False
    assert control["lensing"]["pass"] is False
    assert float(control["lensing"]["worst_flatness"]) > 0.5
    assert control["cluster"]["pass"] is False
    # The recorded missing-mass problem: Newton falls short 2.40x-5.50x.
    shortfalls = [float(text) for text in control["cluster"]["shortfall_by_probe"]]
    assert min(shortfalls) > 2.3 and max(shortfalls) > 5.0


def test_no_c_cluster_value_rescues_the_synthetic_cluster(pack):
    """The needed boost is non-monotone in g_bar (5.50 at g=1.31 vs 2.82 at g=1.50),
    so no constant C_cluster can bring every probe within 15%."""

    for c_text in ("0.2", "0.4", "0.45", "0.5", "0.7", "1"):
        verdict = evaluate_candidate_exact({"C_cluster": c_text}, pack=pack)
        assert verdict["cluster"]["pass"] is False
        assert float(verdict["cluster"]["max_deviation"]) > 0.4


# ---------------------------------------------------------------------------
# Gauntlet receipt: match, determinism, tamper
# ---------------------------------------------------------------------------


def test_gauntlet_receipt_matches_fresh_exact(gauntlet_receipt, pack, published):
    assert gauntlet_receipt["schema_version"] == "invariant-sigma-gravity-gauntlet-1.0"
    assert gauntlet_receipt["decision"].startswith("GAUNTLET-EVALUATED")
    assert canonical_sha256(gauntlet_receipt["published_point"]) == canonical_sha256(published)
    assert gauntlet_receipt["claims"] == CLAIMS_GAUNTLET
    validate_gauntlet_receipt(gauntlet_receipt)


def test_gauntlet_receipt_has_no_floats(gauntlet_receipt):
    def walk(value):
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(gauntlet_receipt)


def test_gauntlet_receipt_is_deterministic(gauntlet_receipt):
    def stripped(receipt):
        volatile = {"elapsed_seconds", "content_sha256"}
        return {key: value for key, value in receipt.items() if key not in volatile}

    second = run_gauntlet()
    assert canonical_sha256(stripped(gauntlet_receipt)) == canonical_sha256(stripped(second))


def test_gauntlet_tamper_fails_closed(gauntlet_receipt):
    tampered = dict(gauntlet_receipt)
    tampered["decision"] = "everything is fine"
    with pytest.raises(SigmaGravityGateError):
        validate_gauntlet_receipt(tampered)
    # Reseal after tampering with the claims: the claims boundary catches it.
    tampered = {k: v for k, v in gauntlet_receipt.items() if k != "content_sha256"}
    tampered["claims"] = {**CLAIMS_GAUNTLET, "real_observational_data_used": True}
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(SigmaGravityGateError):
        validate_gauntlet_receipt(tampered)
    # Reseal after tampering with a frozen grid value: the grid replay catches it.
    grids = json.loads(json.dumps(gauntlet_receipt["frozen_grids"]))
    grids["disk_grid"]["disks"][0]["points"][0]["gbar"] = "1.00000000000000000e-03"
    tampered = {k: v for k, v in gauntlet_receipt.items() if k != "content_sha256"}
    tampered["frozen_grids"] = grids
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(SigmaGravityGateError):
        validate_gauntlet_receipt(tampered)
    # Reseal after flipping the published cluster verdict: the exact replay catches it.
    point = json.loads(json.dumps(gauntlet_receipt["published_point"]))
    point["cluster"]["pass"] = True
    tampered = {k: v for k, v in gauntlet_receipt.items() if k != "content_sha256"}
    tampered["published_point"] = point
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(SigmaGravityGateError):
        validate_gauntlet_receipt(tampered)


# ---------------------------------------------------------------------------
# Scan codec and fp64/exact agreement
# ---------------------------------------------------------------------------


def test_scan_codec_roundtrip():
    sizes = (9, 13, 13, 11, 5, 4)
    assert TOTAL_SCAN_CANDIDATES == int(np.prod(sizes)) == 334620
    assert SCAN_CONFIG["total_candidates"] == TOTAL_SCAN_CANDIDATES
    for ordinal in (0, 1, 4619, 334619):
        decoded = decode_scan_ordinal(ordinal)
        assert encode_scan_indices(decoded["indices"]) == ordinal
    last = decode_scan_ordinal(TOTAL_SCAN_CANDIDATES - 1)
    assert last["values"] == {
        "A0": "1.6", "n": "0.60", "g_dagger": "1.6", "C_cluster": "1.0",
        "p": "0.60", "q": "2.0",
    }
    with pytest.raises(SigmaGravityGateError):
        decode_scan_ordinal(TOTAL_SCAN_CANDIDATES)
    with pytest.raises(SigmaGravityGateError):
        encode_scan_indices([9, 0, 0, 0, 0, 0])


def test_axes_at_published_counts_exact_matches_only(floats):
    indices = [
        0,  # A0 = 0.8 (published A0 is off-grid, never counts)
        5,  # n = 0.25 (published n is off-grid, never counts)
        SCAN_CONFIG["axes"]["g_dagger"].index("0.8"),
        SCAN_CONFIG["axes"]["C_cluster"].index("1.0"),
        SCAN_CONFIG["axes"]["p"].index("0.50"),
        SCAN_CONFIG["axes"]["q"].index("1.0"),
    ]
    ordinal = encode_scan_indices(indices)
    result = evaluate_batch(np, np.array([ordinal], dtype=np.int64), floats, dtype=np.float64)
    assert int(result["axes_at_published"][0]) == 4


def test_published_fp64_pipeline_agrees_with_50_digits(floats, published):
    result = evaluate_params_batch(np, published_params_row(), floats, dtype=np.float64)
    for name in ("galaxy", "lensing", "cluster", "all"):
        assert bool(result[f"{name}_pass"][0]) == published["verdict"][f"{name}_pass"]
    exact_cluster = float(published["cluster"]["max_deviation"])
    assert abs(float(result["cluster_max_deviation"][0]) - exact_cluster) < 1e-9
    exact_cons = float(published["lensing"]["worst_consistency"])
    assert abs(float(result["lensing_worst_consistency"][0]) - exact_cons) < 1e-9


# ---------------------------------------------------------------------------
# Neighborhood scan receipt
# ---------------------------------------------------------------------------


def test_small_scan_counts_and_negative_seal(small_scan):
    counts = small_scan["counts"]
    assert counts["processed"] == SMALL_SCAN
    assert counts["total_candidates"] == TOTAL_SCAN_CANDIDATES
    for key, value in counts.items():
        if key.endswith("_pass") or key == "defined":
            assert 0 <= value <= counts["processed"], key
    assert counts["all_pass"] <= min(
        counts["galaxy_pass"], counts["lensing_pass"], counts["cluster_pass"]
    )
    # The coherence window makes the pooled RAR two-valued across the whole
    # neighborhood: the deep-regime inversion is A0/g_dagger/p/q independent.
    assert counts["disk_rar_pass"] == 0
    assert counts["galaxy_pass"] == 0
    assert counts["disk_rar_per_disk_pass"] == counts["processed"]
    if counts["all_pass"] == 0:
        assert small_scan["decision"].startswith("SCANNED-SEALED-NEGATIVE")
        assert small_scan["best_all_gate_passers"] == []
    assert len(small_scan["closest_candidates"]) == SCAN_CONFIG["closest_report"]["count"]
    for entry in small_scan["closest_candidates"]:
        assert set(entry["metrics"]) >= {
            "cluster_max_deviation",
            "lensing_worst_consistency",
            "newton_far_error",
            "rar_min_margin",
        }
    assert small_scan["published_point"]["passes_all_gates"] is False
    assert small_scan["published_point"]["exact_agrees_with_fp64"] is True
    assert small_scan["pareto_front_total"] >= 1
    assert len(small_scan["pareto_front"]) <= SCAN_CONFIG["pareto"]["reported_cap"]
    validate_neighborhood_receipt(small_scan)


def test_small_scan_receipt_has_no_floats(small_scan):
    def walk(value):
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(small_scan)


def test_small_scan_exact_verification_confirmed(small_scan):
    verified = small_scan["exact_verification"]
    assert verified, "reported candidates must be exactly re-verified"
    assert all(entry["exact_confirmed"] for entry in verified)
    reported = {entry["ordinal"] for entry in small_scan["closest_candidates"]}
    assert reported <= {entry["ordinal"] for entry in verified} | set(range(SMALL_SCAN))


def test_scan_determinism_and_tamper(small_scan):
    def stripped(receipt):
        volatile = {
            "elapsed_seconds",
            "throughput_candidates_per_second",
            "content_sha256",
            "device",
        }
        return {key: value for key, value in receipt.items() if key not in volatile}

    second = scan_neighborhood(use_gpu=False, limit=SMALL_SCAN)
    assert canonical_sha256(stripped(small_scan)) == canonical_sha256(stripped(second))
    tampered = dict(small_scan)
    tampered["counts"] = {**small_scan["counts"], "all_pass": 999}
    with pytest.raises(SigmaGravityGateError):
        validate_neighborhood_receipt(tampered)
    # Reseal after inflating a count: the crosscheck sample replay catches a doctored
    # decision, and the claims boundary catches a doctored claims block.
    tampered = {k: v for k, v in small_scan.items() if k != "content_sha256"}
    tampered["claims"] = {**CLAIMS_SCAN, "scan_is_not_calibration": False}
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(SigmaGravityGateError):
        validate_neighborhood_receipt(tampered)
    crosscheck = dict(small_scan["crosscheck"])
    crosscheck["sample_decisions_sha256"] = "0" * 64
    tampered = {k: v for k, v in small_scan.items() if k != "content_sha256"}
    tampered["crosscheck"] = crosscheck
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(SigmaGravityGateError):
        validate_neighborhood_receipt(tampered)


def test_gpu_scan_crosschecks_cleanly():
    if _cupy_or_none() is None:
        pytest.skip("no CUDA device")
    receipt = scan_neighborhood(use_gpu=True, limit=16384)
    crosscheck = receipt["crosscheck"]
    assert crosscheck["performed"] is True
    for name in ("galaxy_pass", "lensing_pass", "cluster_pass", "all_pass"):
        assert crosscheck[f"{name}_disagreements"] == 0
    validate_neighborhood_receipt(receipt)


# ---------------------------------------------------------------------------
# CLI and claim boundary
# ---------------------------------------------------------------------------


def test_cli_end_to_end(tmp_path):
    gauntlet_path = tmp_path / "gauntlet.json"
    assert main(["--stage", "gauntlet", "--output", str(gauntlet_path)]) == 0
    assert main(["--stage", "gauntlet", "--output", str(gauntlet_path),
                 "--validate-checked"]) == 0
    receipt = json.loads(gauntlet_path.read_text(encoding="utf-8"))
    receipt["decision"] = "everything is fine"
    gauntlet_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SigmaGravityGateError):
        main(["--stage", "gauntlet", "--output", str(gauntlet_path), "--validate-checked"])
    scan_path = tmp_path / "scan.json"
    assert main(["--stage", "scan", "--cpu", "--limit", "440", "--output", str(scan_path)]) == 0
    assert main(["--stage", "scan", "--output", str(scan_path), "--validate-checked"]) == 0
    with pytest.raises(SigmaGravityGateError):
        main(["--stage", "scan", "--validate-checked"])


def test_claims_boundaries():
    for claims in (CLAIMS_GAUNTLET, CLAIMS_SCAN):
        assert claims["external_formula_under_test"] is True
        assert claims["real_observational_data_used"] is False
        assert claims["synthetic_controls_only"] is True
        assert claims["pass_is_not_physical_validation"] is True
        assert claims["scan_is_not_calibration"] is True
    assert CLAIMS_SCAN["scan_is_refutation_of_paper_sparc_fox_fits"] is False
    assert CLAIMS_SCAN["survivor_is_validated_theory"] is False
    assert CANDIDATE_CONFIG["published_parameters"]["g_dagger_code"] == "4/5"
    assert CANDIDATE_CONFIG["published_parameters"]["C_cluster_published"] == "1"
