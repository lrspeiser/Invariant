from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_full3d_source_screen_v1 as screen,
)


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = screen.load_config(verify_package=False)
    receipt = screen.build_receipt(config)
    return config, receipt


def test_config_is_exact_real_source_zero_response_contract(packet: tuple[dict, dict]) -> None:
    config, _receipt = packet
    screen.validate_config(config)
    assert config["objects"] == ["NGC2903", "NGC3351", "NGC3627"]
    assert config["real_source_contract"]["source_files_opened_per_build"] == 21
    assert config["real_source_contract"]["source_bytes_opened_per_build"] == 74030400
    assert config["operator_contract"]["response_parameter_fitting"] is False


def test_both_predecessors_are_exactly_bound_and_passed(packet: tuple[dict, dict]) -> None:
    config, _receipt = packet
    receipts = screen.validate_predecessors(config)
    assert set(receipts) == {
        "REAL_SOURCE_225_CELL_SYSTEMATICS",
        "REFRACTED_GRAVITY_PRIMARY_PAPER_BENCHMARK",
    }
    assert receipts["REAL_SOURCE_225_CELL_SYSTEMATICS"]["cell_count"] == 225
    assert receipts["REFRACTED_GRAVITY_PRIMARY_PAPER_BENCHMARK"]["benchmark_suite"]["passed"] == 14


def test_physical_density_conversion_is_exact(packet: tuple[dict, dict]) -> None:
    config, _receipt = packet
    units = config["unit_contract"]
    expected = units["solar_mass_kg"] / units["parsec_m"] ** 3 / 1000.0
    assert units["msun_pc3_to_g_cm3"] == expected
    assert units["same_mass_field_for_rhs_and_permittivity"] is True


def test_full_225_by_9_ledger_and_exact_solve_accounting(packet: tuple[dict, dict]) -> None:
    _config, receipt = packet
    assert receipt["source_cell_count"] == 225
    assert receipt["registered_source_parameter_pairs"] == 2025
    assert receipt["unique_linear_solves"] == 1350
    assert sum(len(row["parameter_results"]) for row in receipt["source_rows"]) == 2025
    assert all(row["unique_coefficient_fields"] == 6 for row in receipt["source_rows"])


def test_all_nine_published_parameter_cells_are_retained(packet: tuple[dict, dict]) -> None:
    _config, receipt = packet
    cells = receipt["published_parameter_cells"]
    assert len(cells) == 9
    assert cells[0]["id"] == "DISKMASS_UNIVERSAL_MEDIAN"
    assert sum(row["id"].startswith("PRIOR_CORNER_") for row in cells) == 8
    assert all(len(row["parameter_results"]) == 9 for row in receipt["source_rows"])


def test_epsilon_one_cells_are_exact_newton_equivalents(packet: tuple[dict, dict]) -> None:
    _config, receipt = packet
    for source_row in receipt["source_rows"]:
        epsilon_one = [row for row in source_row["parameter_results"] if row["epsilon_0"] == 1.0]
        assert len(epsilon_one) == 4
        assert len({row["coefficient_hash"] for row in epsilon_one}) == 1
        assert len({row["potential_hash"] for row in epsilon_one}) == 1
        assert sum(row["equivalence_reused"] for row in epsilon_one) == 3
        assert max(row["epsilon_one_relative_newton_error"] for row in epsilon_one) < 1.0e-12


def test_only_inherited_source_counterexample_is_retained_for_all_parameters(
    packet: tuple[dict, dict],
) -> None:
    _config, receipt = packet
    counterexamples = receipt["retained_counterexamples"]
    assert receipt["eligible_pair_count"] == 2016
    assert receipt["retained_counterexample_count"] == 9
    assert {(row["object_id"], row["source_cell_id"]) for row in counterexamples} == {
        (
            "NGC2903",
            "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200",
        )
    }
    assert {tuple(row["failed_source_gates"]) for row in counterexamples} == {
        ("parent_source_screen",)
    }
    assert all(row["failed_operator_gates"] == [] for row in counterexamples)


def test_every_operator_solve_passes_without_family_pruning(packet: tuple[dict, dict]) -> None:
    config, receipt = packet
    assert config["gate_contract"]["one_failure_cannot_prune_theory_family"] is True
    assert all(
        parameter["operator_numerical_pass"] is True
        for source_row in receipt["source_rows"]
        for parameter in source_row["parameter_results"]
    )
    assert receipt["status"].endswith("WITH_RETAINED_COUNTEREXAMPLES")


def test_source_and_parameter_envelopes_cover_every_object_and_radius(
    packet: tuple[dict, dict],
) -> None:
    _config, receipt = packet
    median = receipt["fixed_median_source_envelopes"]
    full = receipt["all_parameter_and_source_envelopes"]
    assert len(median) == len(full) == 9
    assert all(row["maximum_to_minimum_ratio"] >= 1.0 for row in median + full)
    assert all(row["value_count"] in {71, 72} for row in median)
    assert all(row["value_count"] in {639, 648} for row in full)


def test_same_source_comparisons_to_newton_aqual_and_qumond_are_explicit(
    packet: tuple[dict, dict],
) -> None:
    _config, receipt = packet
    rows = receipt["same_source_theory_comparisons"]
    assert len(rows) == 9
    expected = {
        "RG_OVER_NEWTON",
        "ABS_RG_MINUS_AQUAL_OVER_AQUAL",
        "ABS_RG_MINUS_QUMOND_OVER_QUMOND",
    }
    assert all(set(row["metrics"]) == expected for row in rows)
    assert all(
        metric["minimum"] <= metric["median"] <= metric["maximum"]
        for row in rows
        for metric in row["metrics"].values()
    )


def test_receipt_roots_are_exact_recomputations(packet: tuple[dict, dict]) -> None:
    _config, receipt = packet
    roots = receipt["roots"]
    assert roots["source_parameter_ledger_sha256"] == screen.content_sha256(receipt["source_rows"])
    assert roots["counterexample_ledger_sha256"] == screen.content_sha256(
        receipt["retained_counterexamples"]
    )
    assert roots["equivalence_ledger_sha256"] == screen.content_sha256(
        receipt["equivalence_groups"]
    )


def test_no_response_scoring_or_observational_claim(packet: tuple[dict, dict]) -> None:
    config, receipt = packet
    access = receipt["access_accounting"]
    assert access["scientific_response_files_opened"] == 0
    assert access["scientific_response_rows_opened"] == 0
    assert access["scores_computed"] == 0
    assert access["parameters_fit"] == 0
    claims = config["claim_boundary"]
    assert claims["real_kinematic_response_tested"] is False
    assert claims["observational_preference_established"] is False
    assert claims["publication_ready"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("status",), "CONFIRMED"),
        (("operator_contract", "parameter_cells"), 1),
        (("operator_contract", "response_parameter_fitting"), True),
        (("gate_contract", "one_failure_cannot_prune_theory_family"), False),
        (("access_contract", "scientific_response_files_opened"), 1),
        (("claim_boundary", "observational_preference_established"), True),
    ),
)
def test_config_mutations_fail_closed(
    packet: tuple[dict, dict], path: tuple[str, ...], replacement: object
) -> None:
    config, _receipt = packet
    forged = copy.deepcopy(config)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(screen.RefractedGravitySourceScreenError):
        screen.validate_config(forged)


def test_coherently_rehashed_receipt_forgery_is_rejected(
    packet: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, receipt = packet
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["observational_preference_established"] = True
    forged["content_sha256"] = screen.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(screen, "build_receipt", lambda _config: receipt)
    with pytest.raises(screen.RefractedGravitySourceScreenError):
        screen.validate_receipt_payload(config, forged)
