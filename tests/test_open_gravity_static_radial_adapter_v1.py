from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.open_gravity_static_radial_adapter_v1 as adapter
from sigma_theory_compiler import open_gravity_registry_foundation_v1 as registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return adapter.load_config(ROOT)


@pytest.fixture(scope="module")
def synthetic_sources() -> tuple[dict, dict]:
    return adapter._synthetic_sources()


@pytest.fixture(scope="module")
def gp01_cells(synthetic_sources: tuple[dict, dict]) -> dict:
    return adapter.enumerate_gp01_spherical_cells(synthetic_sources[1])


def _scalar_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if isinstance(value, dict):
        paths: list[tuple[object, ...]] = []
        for key, child in value.items():
            paths.extend(_scalar_paths(child, prefix + (key,)))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_scalar_paths(child, prefix + (index,)))
        return paths
    return [prefix]


def _mutated_scalar(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "_MUTATED"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return math.nextafter(value, math.inf)
    if value is None:
        return "MUTATED"
    raise AssertionError(f"unsupported scalar: {value!r}")


def _mutate_path(value: object, path: tuple[object, ...]) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    key = path[-1]
    target[key] = _mutated_scalar(target[key])  # type: ignore[index]


def test_frozen_config_and_committed_upstream_files_are_exact(config: dict) -> None:
    assert adapter.content_sha256(config) == adapter.EXPECTED_CONFIG_CONTENT_SHA256
    upstream = adapter.verify_committed_upstreams(ROOT, config)
    assert upstream["commits"] == [
        "35f70938f158c81971b2e1b838371b09d9fcee2c",
        "ed2988546fb1165d9efe5e62d52cddebc7b1a79d",
    ]
    assert len(upstream["files"]) == 8
    receipt_roles = [row for row in upstream["files"] if "RECEIPT" in row["role"]]
    assert len(receipt_roles) == 2
    assert all(row["semantic_open"] is False for row in receipt_roles)
    assert upstream["response_bearing_receipts_opened"] == 0


def test_every_nested_config_scalar_is_mutation_sealed(config: dict) -> None:
    paths = _scalar_paths(config)
    assert len(paths) > 250
    for path in paths:
        mutated = copy.deepcopy(config)
        _mutate_path(mutated, path)
        with pytest.raises(adapter.OpenGravityStaticRadialAdapterError):
            adapter.validate_config(mutated)


def test_sparc_source_drivers_have_exact_dimensions_and_values() -> None:
    radius = np.array([1.0, 2.0, 3.0]) * adapter.KPC_M
    gas = np.array([1.0, 2.0, 3.0]) * 1.0e-11
    stars = np.array([1.0, 2.0, 3.0]) * 1.0e-11
    result = adapter.compile_sparc_source_drivers(radius, gas, stars)
    assert result["domain"] == "SPARC"
    assert set(result["normalized"]) == set(adapter.SPARC_DRIVERS)
    assert all(row.shape == (257,) for row in result["normalized"].values())
    assert result["radius_m"][0] == 0.0
    assert result["radius_m"][-1] == 3.0 * adapter.KPC_M
    assert result["normalized"]["D01_ACC"][-1] == pytest.approx(math.tanh(0.6))
    assert result["normalized"]["D03_RAD"][-1] == pytest.approx(math.tanh(3.0))
    assert result["normalized"]["D06_SLOPE"][-1] == pytest.approx(math.tanh(-1.0), abs=1e-12)
    assert result["normalized"]["D13_GASF"][-1] == pytest.approx(math.tanh(0.5))
    assert result["response_inputs"] == result["scores_computed"] == 0


def test_sparc_adapter_signature_has_no_response_surface() -> None:
    names = set(inspect.signature(adapter.compile_sparc_source_drivers).parameters)
    assert names == {
        "radius_m",
        "gas_acceleration_m_s2",
        "stellar_acceleration_m_s2",
        "points",
    }
    radius = np.array([1.0, 2.0, 3.0]) * adapter.KPC_M
    with pytest.raises(TypeError):
        adapter.compile_sparc_source_drivers(  # type: ignore[call-arg]
            radius, np.ones(3), np.ones(3), vobs=np.ones(3)
        )


def test_xcop_spherical_drivers_use_exact_si_normalizations() -> None:
    radius = np.linspace(0.1, 1.0, 11) * 1.0e21
    density = np.full(radius.shape, 2.0e-22)
    result = adapter.compile_xcop_spherical_source_drivers(radius, density)
    assert set(result["normalized"]) == set(adapter.XCOP_DRIVERS)
    assert all(row.shape == (257,) for row in result["normalized"].values())
    assert result["metadata"]["stellar_rule"] == "SHARED_0.1_GAS_MASS"
    assert result["physical"]["D02_POT"][-1] == 0.0
    assert result["physical"]["D01_ACC"][0] == 0.0
    assert result["mass"]["baryonic_enclosed_kg"][0] == 0.0
    for driver_id in adapter.XCOP_DRIVERS:
        expected = np.tanh(
            result["physical"][driver_id]
            / (1.0 if driver_id == "D06_SLOPE" else adapter.DRIVER_REFERENCES[driver_id])
        )
        assert np.array_equal(result["normalized"][driver_id], expected)
    assert result["physical"]["D13_GASF"][1] == pytest.approx(1.0 / 1.1)
    assert result["metadata"]["rho_reference_kg_m3"] > 0.0
    assert result["metadata"]["tidal_reference_s_minus_2"] > 0.0


def test_effective_density_origin_is_exact_frozen_enclosed_mass_rule() -> None:
    radius = np.array([0.0, 1.0, 2.0, 3.0])
    mass = np.array([0.0, 1.0, 100.0, 101.0])
    density = adapter._effective_density(radius, mass)
    expected_origin = 3.0 * mass[1] / (4.0 * math.pi * radius[1] ** 3)
    assert density[0] == expected_origin
    assert density[0] != density[1]


def test_all_static_architecture_parameter_branches_and_gates() -> None:
    source_xi = np.linspace(0.0, 1.0, 65)
    source_u = 0.3 + 0.2 * np.sin(math.pi * source_xi)
    source_g = 1.0e-11 * (0.2 + source_xi)
    count = 0
    for architecture_id in adapter.STATIC_ARCHITECTURES:
        for parameters in adapter.architecture_parameter_cells(architecture_id):
            result = adapter.compile_static_architecture(
                architecture_id, source_xi, source_u, source_g, parameters
            )
            count += 1
            assert result["primary"]["factor"].shape == (257,)
            assert result["convergence"]["factor"].shape == (129,)
            assert all(result["gates"].values())
            assert result["primary"]["diagnostics"]["operator_residual_max_abs"] <= 1e-9
            assert result["primary"]["diagnostics"]["boundary_residual_max_abs"] <= 1e-10
            assert result["convergence_max_abs"] <= 0.02
            if parameters["lambda"] == 0.0:
                assert np.array_equal(result["primary"]["factor"], np.ones(257))
    assert count == 44


def test_a11_convergence_adversary_cannot_override_frozen_gates() -> None:
    xi = np.linspace(0.0, 1.0, 257)
    sharp_driver = np.zeros(257)
    sharp_driver[xi >= 0.033] = 1.0
    g_b = np.ones(257)
    parameters = {"lambda": 0.25, "s_c": 1.0, "n": 2}
    with pytest.raises(
        adapter.OpenGravityStaticRadialAdapterError,
        match="PRIMARY_VS_CONVERGENCE.*False",
    ):
        adapter.compile_static_architecture("A11_DERIV_SCREEN", xi, sharp_driver, g_b, parameters)
    for keyword in (
        {"operator_tolerance": math.inf},
        {"boundary_tolerance": 1.0},
        {"convergence_tolerance": math.inf},
    ):
        with pytest.raises(TypeError):
            adapter.compile_static_architecture(
                "A11_DERIV_SCREEN", xi, sharp_driver, g_b, parameters, **keyword
            )


def test_static_operator_analytic_controls_and_boundaries() -> None:
    xi = np.linspace(0.0, 1.0, 257)
    u = np.full(257, 0.4)
    g = np.ones(257)
    kernel = adapter.apply_static_architecture(
        "A06_SPATIAL_KERNEL", u, g, xi, {"lambda": 0.25, "ell": 0.25}
    )
    assert np.max(np.abs(kernel["state"]["q"] - 0.4)) < 1e-12
    boundary = adapter.apply_static_architecture("A07_BOUNDARY", u, g, xi, {"lambda": 0.25})
    assert np.max(np.abs(boundary["state"]["q"] - 0.4 * (1.0 - xi))) < 1e-14
    massive = adapter.apply_static_architecture(
        "A12_MASSIVE", np.zeros(257), g, xi, {"lambda": 0.25, "mu": 4.0}
    )
    assert np.array_equal(massive["state"]["q"], np.zeros(257))
    feedback = adapter.apply_static_architecture(
        "A19_FEEDBACK", u, g, xi, {"lambda": 0.25, "kappa": 0.5}
    )
    assert feedback["diagnostics"]["iterations"] <= 128
    assert feedback["diagnostics"]["operator_residual_max_abs"] <= 1e-12


@pytest.mark.parametrize("architecture_id", adapter.TIME_SOURCE_BLOCKS)
def test_time_dependent_architectures_are_source_blocked_on_static_real_data(
    architecture_id: str,
) -> None:
    xi = np.linspace(0.0, 1.0, 5)
    with pytest.raises(adapter.StaticSourceBlockedError):
        adapter.apply_static_architecture(architecture_id, xi, xi, xi, {})


def test_relevant_compound_program_values_and_sparc_x01_block() -> None:
    u1 = np.array([0.2, 0.4, 0.8])
    u2 = np.array([-0.5, 0.25, 0.5])
    expected = {
        "X01": np.clip(u1 * (1.0 + u2) / 2.0, -1.0, 1.0),
        "X05": (u1 - u2) / 2.0,
        "X10": u1 / (1.0 + np.abs(u2)),
        "X13": (u1 + 2.0 * u2) / 3.0,
        "X17": u1 / (1.0 + np.abs(u2)),
        "X18": (u1 + u2) / 2.0,
    }
    required = {
        "X01": ("D01_ACC", "D06_SLOPE"),
        "X05": ("D02_POT", "D06_SLOPE"),
        "X10": ("D01_ACC", "D07_TIDE"),
        "X13": ("D02_POT", "D13_GASF"),
        "X17": ("D04_RHO", "D03_RAD"),
        "X18": ("D01_ACC", "D02_POT"),
    }
    for compound_id, names in required.items():
        actual = adapter.combine_compound_drivers(
            "XCOP_SPHERICAL", compound_id, {names[0]: u1, names[1]: u2}
        )
        assert np.array_equal(actual, expected[compound_id])
    with pytest.raises(adapter.StaticSourceBlockedError):
        adapter.combine_compound_drivers("SPARC", "X01", {"D01_ACC": u1, "D06_SLOPE": u2})


def test_gp01_local_values_dimensions_and_analytic_limits() -> None:
    for n in (1, 2, 4):
        assert adapter.gp01_nu_n(1.0, n) == pytest.approx(2.0 ** (1.0 / (2.0 * n)))
        deep_y = 1.0e-12
        high_y = 1.0e12
        assert adapter.gp01_nu_n(deep_y, n) * math.sqrt(deep_y) == pytest.approx(1.0, rel=1e-6)
        assert adapter.gp01_nu_n(high_y, n) == pytest.approx(1.0, rel=1e-6)
        g_b = np.array([0.0, deep_y, 1.0, high_y]) * 1.2e-10
        result = adapter.gp01_l_acceleration(g_b, n=n)
        assert result.shape == g_b.shape
        assert result[0] == 0.0
        assert np.all(np.isfinite(result))
    weight = np.array([0.25, 0.5])
    target = adapter.gp01_bounded_target(np.array([0.0, 1.2e-10]), weight, n=2, A_max=4.0)
    assert target[0] == pytest.approx(0.25 * math.log(4.0))
    assert np.all((target >= 0.0) & (target <= math.log(4.0)))


def test_spherical_solver_residual_boundaries_convergence_and_analytic_solution() -> None:
    radius_257 = np.linspace(0.0, 1.0, 257)
    target_257 = np.full(257, 0.4)
    solved_257 = adapter.solve_spherical_gamma(radius_257, target_257, L_g_m=0.25)
    analytic = adapter.spherical_constant_target_solution(radius_257, target=0.4, L_g_m=0.25)
    assert solved_257["operator_residual_max_abs"] < 1e-10
    assert solved_257["inner_regularity_residual"] == 0.0
    assert solved_257["outer_dirichlet_residual"] == 0.0
    assert np.max(np.abs(solved_257["gamma"] - analytic)) < 1e-5
    radius_129 = np.linspace(0.0, 1.0, 129)
    solved_129 = adapter.solve_spherical_gamma(radius_129, np.full(129, 0.4), L_g_m=0.25)
    assert np.max(np.abs(solved_257["gamma"][::2] - solved_129["gamma"])) < 2e-5
    zero = adapter.solve_spherical_gamma(radius_257, target_257, L_g_m=0.0)
    assert zero["zero_length_recovers_interior_target"] is True
    assert zero["gamma"][-1] == 0.0
    assert zero["gamma"][0] == zero["gamma"][1]


def test_integrated_spherical_flux_is_exact() -> None:
    radius = np.linspace(0.0, 3.0e20, 257)
    mass = 2.0e30 * (radius / radius[-1]) ** 3
    gamma = 0.2 * (1.0 - radius / radius[-1])
    result = adapter.integrated_spherical_flux(radius, mass, gamma)
    assert result["integrated_flux_relative_residual"] < 1e-14
    assert np.array_equal(result["g_eff_m_s2"], result["factor"] * result["g_b_m_s2"])
    assert np.all(result["factor"] > 0.0)


def test_gp01_exact_1296_cell_enumeration_and_gates(gp01_cells: dict) -> None:
    assert gp01_cells["cell_count"] == 1296
    assert len(gp01_cells["cells"]) == 1296
    assert len({row["cell_id"] for row in gp01_cells["cells"]}) == 1296
    assert gp01_cells["all_gates_pass"] is True
    assert gp01_cells["maximum_operator_residual"] <= 1e-9
    assert gp01_cells["maximum_boundary_residual"] <= 1e-10
    assert gp01_cells["maximum_primary_vs_convergence"] <= 0.02
    assert gp01_cells["maximum_integrated_flux_relative_residual"] <= 1e-12
    assert sum(row["parameters"]["L_ratio"] == 0.0 for row in gp01_cells["cells"]) == 324
    assert all(
        row["zero_length_recovers_interior_target"] is True
        for row in gp01_cells["cells"]
        if row["parameters"]["L_ratio"] == 0.0
    )
    assert {row["parameters"]["n"] for row in gp01_cells["cells"]} == {1, 2, 4}


def test_prediction_equivalence_and_degeneracy_hashes_are_explicit() -> None:
    report = adapter._prediction_equivalence_report()
    assert len(report["groups"]) == 2
    assert all(len(row["shared_prediction_sha256"]) == 64 for row in report["groups"])
    assert all(len(row["degeneracy_sha256"]) == 64 for row in report["groups"])
    for row in report["gp01_spherical_equivalence_links"]:
        assert row["GP01-L_prediction_sha256"] == row["AQUAL_spherical_prediction_sha256"]
        assert row["score_once"] is True
    assert report["matching_synthetic_predictions_do_not_imply_formula_identity"] is True


def test_target_free_wrong_controls_and_program_hashes(config: dict) -> None:
    radius = np.linspace(0.0, 1.0, 9)
    factor = np.linspace(1.0, 2.0, 9)
    identity = adapter.apply_wrong_control("IDENTITY", factor, radius)
    reversal = adapter.apply_wrong_control("RADIAL_FACTOR_REVERSAL", factor, radius)
    assert np.array_equal(identity, factor)
    assert np.array_equal(reversal, factor[::-1])
    hashes = adapter.wrong_control_program_hashes(config)
    assert set(hashes) == {"IDENTITY", "RADIAL_FACTOR_REVERSAL"}
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes == adapter.wrong_control_program_hashes(config)


def test_twell_rebind_stays_deferred_and_requires_exact_final_formula_root(config: dict) -> None:
    deferred = adapter.evaluate_twell_rebind_gate(config, None)
    assert deferred["status"] == "DEFERRED_NOT_REBOUND"
    assert deferred["authoritative"] is False
    assert deferred["repair_paths_opened"] == 0
    candidate = {
        "independent_audit_status": "PASS_ADAPTER_OPERATOR_EQUIVALENCE_ONLY",
        "final_config_sha256": "1" * 64,
        "final_module_sha256": "2" * 64,
        "final_test_sha256": "3" * 64,
        "final_card_stream_sha256": "4" * 64,
        "final_receipt_sha256": "5" * 64,
        "final_formula_program_root_sha256": adapter.program_hash_report(config)[
            "full_formula_program_root_sha256"
        ],
    }
    rebound = adapter.evaluate_twell_rebind_gate(config, candidate)
    assert rebound["authoritative"] is True
    assert rebound["campaign_authority"] is False
    forged = copy.deepcopy(candidate)
    forged["final_formula_program_root_sha256"] = "0" * 64
    with pytest.raises(adapter.OpenGravityStaticRadialAdapterError):
        adapter.evaluate_twell_rebind_gate(config, forged)


def test_complete_typed_card_catalog_is_registry_valid_and_hash_exact(config: dict) -> None:
    catalog = adapter.typed_mechanism_card_catalog(ROOT, config)
    assert catalog["card_count"] == 133
    assert catalog["admission_counts"] == {
        "READY_FOR_THEORY_GATES": 128,
        "SOURCE_BLOCKED": 3,
        "KNOWN_REWRITE_NONINDEPENDENT": 1,
        "QUARANTINED_REVISION_REQUIRED": 1,
    }
    assert catalog["gp01_live_card_count"] == 7
    assert catalog["provisional_twell_adapter_card_count"] == 126
    assert catalog["provisional_twell_cards_manifest_authority"] is False
    assert catalog["controls_are_candidate_cards"] is False
    assert {row["transformation_id"] for row in catalog["control_comparators"]} == {
        "IDENTITY",
        "RADIAL_FACTOR_REVERSAL",
    }
    assert catalog["lane_assignment_authority"] is False
    assert catalog["orthogonal_or_wildcard_fillers_invented"] is False
    assert catalog["lane_hints_present"] == ["ADJACENT", "CORE"]
    cards = [row["card"] for row in catalog["cards"]]
    assert len({card["card_id"] for card in cards}) == 133
    schema = registry.load_schemas(ROOT)["mechanism_card"]
    for row in catalog["cards"]:
        card = row["card"]
        admission = registry.mechanism_card_admission(card, schema)
        assert admission["status"] == row["registry_admission_status"]
        assert registry.content_sha256(card) == row["card_sha256"]
        assert registry.mechanism_formula_sha256(card) == row["formula_sha256"]
        assert (
            registry.equivalence_fingerprint_sha256(card) == row["equivalence_fingerprint_sha256"]
        )
        assert set(row["domain_execution"]) == set(registry.DOMAINS)
        assert all(domain["scored"] is False for domain in row["domain_execution"].values())
        assert row["lane_assignment_authority"] is False
    assert registry.mechanism_card_set_sha256(cards) == catalog["ordered_card_set_sha256"]


def test_typed_card_domain_dispositions_preserve_blocks_and_theory_only(config: dict) -> None:
    catalog = adapter.typed_mechanism_card_catalog(ROOT, config)
    rows = {row["card"]["stable_concept_id"]: row for row in catalog["cards"]}
    x01 = rows["X01"]
    assert x01["manifest_authority_after_required_bindings"] is False
    assert x01["domain_source_status"]["SPARC"].startswith("SOURCE_BLOCKED_")
    assert x01["domain_execution"]["GALAXIES"]["execution_disposition"] == "NOT_APPLICABLE"
    assert x01["domain_execution"]["CLUSTERS"]["execution_disposition"] == "THEORY_ONLY"
    gp01 = rows["GP01-L"]
    assert gp01["manifest_authority_after_required_bindings"] is True
    assert gp01["candidate_status_hint"] == "REGISTERED_THEORY_ONLY"
    assert gp01["domain_execution"]["GALAXIES"]["execution_disposition"] == "THEORY_ONLY"
    assert len(gp01["card"]["parameter_cells"]) == 3
    aqual = rows["GP01-AQUAL"]
    assert aqual["candidate_status_hint"] == "KNOWN_REWRITE_NONINDEPENDENT"
    blocked = rows["GP01-T1"]
    assert blocked["candidate_status_hint"] == "SOURCE_BLOCKED"
    assert blocked["domain_execution"]["GALAXIES"]["execution_disposition"] == "SOURCE_BLOCKED"
    assert blocked["domain_execution"]["CLUSTERS"]["execution_disposition"] == "SOURCE_BLOCKED"
    action = rows["GP01-ACTION-PLACEHOLDER"]
    assert action["candidate_status_hint"] == "QUARANTINED_REVISION_REQUIRED"
    assert action["domain_execution"]["GALAXIES"]["execution_disposition"] == "QUARANTINED"
    assert all(
        row["card"]["stable_concept_id"]
        not in {"CONTROL-IDENTITY", "CONTROL-RADIAL-FACTOR-REVERSAL"}
        for row in catalog["cards"]
    )


def test_atomic_no_clobber_preserves_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = b'{"safe":true}\n'
    assert adapter._atomic_no_clobber(target, payload) == "CREATED"
    assert adapter._atomic_no_clobber(target, payload) == "EXISTING_IDENTICAL"
    target.write_bytes(b"different\n")
    with pytest.raises(adapter.OpenGravityStaticRadialAdapterError):
        adapter._atomic_no_clobber(target, payload)
    assert target.read_bytes() == b"different\n"


def test_receipt_is_deterministic_zero_access_and_mechanics_only() -> None:
    first = adapter.build_receipt(ROOT)
    second = adapter.build_receipt(ROOT)
    assert first == second
    assert first["content_sha256"] == adapter.receipt_content_sha256(first)
    assert all(value == 0 for value in first["access_ledger"].values())
    assert first["synthetic_verification"]["gp01_elliptic"]["cell_count"] == 1296
    assert first["twell_rebind"]["status"] == "DEFERRED_NOT_REBOUND"
    assert first["claim_boundary"]["scientific_pass_claimed"] is False
    assert first["claim_boundary"]["campaign_manifest_created"] is False
    assert first["claim_boundary"]["campaign_scoring_authorized"] is False
    assert first["synthetic_verification"]["wrong_controls"]["target_inputs"] == 0
    assert (
        first["synthetic_verification"]["wrong_controls"][
            "environment_or_object_shuffles_in_adapter"
        ]
        == 0
    )
    assert first["bindings"]["audit_blocked_predecessor"] == {
        "path": (
            "work/open-gravity-static-radial-adapter-v1-audit-blocked/receipt-pre-repair.json"
        ),
        "sha256": "8162b16fd78afc81d105a2ac7394c01af2d4665efce2dbef70887b8dc9a5eabc",
        "status": "BLOCKED_SUPERSEDED_PRESERVED_AS_COUNTEREVIDENCE",
        "semantic_open": False,
    }


def test_stored_receipt_is_exact_and_in_memory_forgery_fails_closed() -> None:
    stored = adapter.validate_receipt(ROOT)
    assert stored == adapter.build_receipt(ROOT)
    forged = copy.deepcopy(stored)
    forged["claim_boundary"]["scientific_pass_claimed"] = True
    forged["content_sha256"] = adapter.receipt_content_sha256(forged)
    with pytest.raises(adapter.OpenGravityStaticRadialAdapterError):
        adapter.validate_receipt_payload(ROOT, forged)


def test_validate_receipt_hard_binds_canonical_path_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []

    def reject_open(path: Path) -> dict:
        opened.append(path)
        raise AssertionError("a noncanonical path was opened")

    monkeypatch.setattr(adapter, "_read_json", reject_open)
    invalid_paths = (
        Path("receipt.json"),
        ROOT / "forged-receipt.json",
        Path("runs/gravity/open-gravity-static-radial-adapter-v1/../other.json"),
    )
    for invalid_path in invalid_paths:
        with pytest.raises(
            adapter.OpenGravityStaticRadialAdapterError,
            match="canonical frozen OUTPUT_PATH",
        ):
            adapter.validate_receipt(ROOT, invalid_path)
    assert opened == []
