from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_quantum_entity_wave_atlas_v2 as atlas


def _copy_package(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / atlas.CONFIG_PATH).read_text(encoding="utf-8"))
    relatives = (
        atlas.CONFIG_PATH,
        atlas.MODULE_PATH,
        atlas.TEST_PATH,
        Path(config["predecessor"]["path"]),
    )
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return tmp_path


def test_predecessor_is_preserved_and_all_cards_are_typed() -> None:
    config = atlas.load_config()
    assert config["predecessor"]["status"] == "PRESERVED_SUPERSEDED_BY_TYPED_AUDIT_REPAIR"
    assert tuple(card["id"] for card in config["cards"]) == atlas.CARD_IDS
    assert len(config["cards"]) == 16
    for card in config["cards"]:
        assert card["state_space"]
        assert card["parameters"]
        assert card["equations"]
        assert card["probability_law"]["normalization"]
        assert {row["channel"] for row in card["observables"]} == {
            "matter",
            "photon",
            "tensor",
        }
        assert card["limits"]
        assert set(card["fixture"]["equation_ids"]) <= {row["id"] for row in card["equations"]}


def test_general_gaussian_pushforward_is_exact_in_multiple_dimensions() -> None:
    first = atlas.finite_gaussian_pushforward(
        [[1, 2, 0], [0, 1, -1]],
        [1, -1, 2],
        [[2, 1, 0], [1, 3, 1], [0, 1, 2]],
        [1, -1],
        [[1, 0], [0, 2]],
    )
    assert first["input_dimension"] == 3
    assert first["output_dimension"] == 2
    assert first["output_mean"] == [0, -4]
    assert first["output_covariance"] == [[19, 5], [5, 5]]
    second = atlas.finite_gaussian_pushforward(
        [[1, 0], [0, 1], [1, 1]],
        [2, 3],
        [[1, 0], [0, 1]],
        [0, 0, 0],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert second["input_dimension"] == 2
    assert second["output_dimension"] == 3
    assert second["output_mean"] == [2, 3, 5]
    assert second["output_covariance"] == [[2, 0, 1], [0, 2, 1], [1, 1, 3]]
    assert "back-action" in second["assumptions"]


def test_q13_is_true_immigration_death_and_stationary_cumulants_are_derived() -> None:
    config = atlas.load_config()
    card = next(
        card for card in config["cards"] if card["id"] == "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY"
    )
    result = atlas.generate_fixture(card)["result"]
    assert result["jump_law"] == "dN=dJ_plus-dJ_minus"
    assert result["conditional_intensities"] == {"J_plus": "lambda", "J_minus": "N/tau"}
    assert result["stationary_family"] == "Poisson(lambda tau)"
    assert result["stationary_mean"] == result["stationary_variance"] == 12
    assert result["ordinary_cumulants_1_to_4"] == [12, 12, 12, 12]
    assert result["max_detailed_balance_residual_n0_to5"] < 1e-15
    assert result["autocovariance"][1]["value"] == pytest.approx(12 / math.e)
    assert "log PGF" in result["derivation"]


def test_fixtures_are_generated_from_every_card_definition() -> None:
    config = atlas.load_config()
    fixtures = atlas.generate_fixtures(config)
    assert [row["card_id"] for row in fixtures] == list(atlas.CARD_IDS)
    assert [row["evaluator"] for row in fixtures] == list(atlas.EVALUATORS)
    assert all(len(row["card_definition_sha256"]) == 64 for row in fixtures)
    assert all(row["equation_ids"] for row in fixtures)
    bmv = next(
        row["result"] for row in fixtures if row["card_id"] == "Q07_ENTANGLEMENT_MEDIATED_GRAVITY"
    )
    massive = next(row["result"] for row in fixtures if row["card_id"] == "Q01_MASSIVE_SPIN2")
    assert bmv["concurrence"] > 0
    assert bmv["entangled"] is True
    assert massive["lower_frequency_later"] is True


def test_cq_cp_counterexample_and_ktm_bound_are_retained() -> None:
    config = atlas.load_config()
    fixtures = {row["card_id"]: row["result"] for row in atlas.generate_fixtures(config)}
    cq = fixtures["Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY"]
    assert cq["determinant"] == 0
    assert cq["positive_semidefinite_scalar_block"] is True
    assert cq["minimum_noise_boundary"] is True
    assert cq["counterexample_determinant"] < 0
    assert cq["counterexample_rejected"] is True
    ktm = fixtures["Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY"]
    assert ktm["all_bound_residuals_nonnegative"] is True
    assert ktm["minimum_grid_gamma"] == pytest.approx(0.25)
    assert ktm["minimum_grid_cost"] == pytest.approx(0.5)
    assert ktm["entangling_capacity_under_LOCC"] == 0


def test_parameter_limit_families_replace_coarse_binary_claims() -> None:
    config = atlas.load_config()
    fixtures = atlas.generate_fixtures(config)
    families = atlas.equivalence_results(config, fixtures)
    assert len(families) == 5
    assert all(row["parameter_map"] and row["not_equivalent"] for row in families)
    assert families[0]["executable_check"]["verified"] is True
    assert families[1]["executable_check"]["verified"] is True
    gaussian = next(row for row in families if row["id"] == "EF03_FIXED_GAUSSIAN_OUTPUT")
    assert gaussian["status"] == "EXACT_MEASUREMENT_CLASS_EQUIVALENCE"
    assert "back-action" in " ".join(gaussian["executable_check"]["scope_assumptions"])
    memory = next(row for row in families if row["id"] == "EF04_MEMORY_MEAN")
    assert memory["executable_check"]["verified_exact_equivalence"] is False


def test_source_manifests_are_strict_and_holometer_is_demoted() -> None:
    config = atlas.load_config()
    manifests = {row["id"]: row for row in config["source_manifests"]}
    assert manifests["M01_GWOSC_DISPERSION"]["status"] == (
        "EXECUTABLE_MANIFEST_FROZEN_PAYLOAD_UNOPENED"
    )
    assert [row["uid"] for row in manifests["M01_GWOSC_DISPERSION"]["events"]] == [
        "GW150914-v3",
        "GW170814-v3",
        "GW170817-v3",
    ]
    assert manifests["M01_GWOSC_DISPERSION"]["model_grid"]
    assert manifests["M01_GWOSC_DISPERSION"]["nuisance_priors"]
    assert manifests["M01_GWOSC_DISPERSION"]["likelihood"]
    assert manifests["M01_GWOSC_DISPERSION"]["tolerances"]
    assert manifests["M09_HOLOMETER_UNSUITABLE_TIME_CUMULANTS"]["status"] == ("SOURCE_BLOCKED")
    assert "do not support" in manifests["M09_HOLOMETER_UNSUITABLE_TIME_CUMULANTS"]["reason"]
    assert manifests["M04_BMV_SOURCE_BLOCKED"]["response_status"] == ("NO_GRAVITY_RESPONSE_EXISTS")


def test_deterministic_build_check_replay_and_tamper_gate(tmp_path: Path) -> None:
    base = _copy_package(tmp_path)
    assert atlas.build(base) == "CREATED"
    assert atlas.check(base) == "VALID"
    assert atlas.build(base) == "EXISTING_IDENTICAL"
    receipt = json.loads((base / atlas.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert receipt["counts"]["typed_cards"] == 16
    assert receipt["counts"]["generated_fixtures"] == 16
    assert receipt["counts"]["real_observational_rows"] == 0
    assert set(receipt["audit_repairs"].values()) == {True}
    artifact = base / atlas.ARTIFACT_DIR / "source-manifests.json"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(atlas.QuantumAtlasV2Error, match="artifact differs"):
        atlas.check(base)


def test_claim_and_access_boundaries_remain_closed() -> None:
    config = atlas.load_config()
    assert set(config["access_contract"].values()) == {0}
    assert config["claim_boundary"]["real_observational_rows_scored"] is False
    assert config["claim_boundary"]["any_branch_empirically_supported"] is False
    assert config["claim_boundary"]["historical_novelty_established"] is False
    assert config["claim_boundary"]["publication_ready"] is False
    cards = {card["id"]: card for card in config["cards"]}
    assert cards["Q03_DISCRETE_GRAVITY_IMPULSES"]["scores"]["data_readiness"] == 1
    assert cards["Q11_FINITE_OCCUPATION_COHERENCE"]["scores"]["data_readiness"] == 1
    assert cards["Q07_ENTANGLEMENT_MEDIATED_GRAVITY"]["scores"]["data_readiness"] == 0
