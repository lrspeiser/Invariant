from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_quantum_entity_wave_atlas_v3 as atlas


def _card(config: dict[str, object], card_id: str) -> dict[str, object]:
    cards = config["cards"]
    assert isinstance(cards, list)
    return next(card for card in cards if card["id"] == card_id)


def _copy_package(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / atlas.CONFIG_PATH).read_text(encoding="utf-8"))
    relatives = [atlas.CONFIG_PATH, atlas.MODULE_PATH, atlas.TEST_PATH]
    relatives.extend(Path(row["path"]) for row in config["predecessors"])
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return tmp_path


def _output(execution: dict[str, object], name: str) -> object:
    outputs = execution["outputs"]
    assert isinstance(outputs, list)
    return next(row["value"] for row in outputs if row["name"] == name)


def test_all_cards_execute_stored_ast_with_units_domains_and_conditions() -> None:
    config = atlas.load_config()
    assert tuple(card["id"] for card in config["cards"]) == atlas.CARD_IDS
    assert len(config["cards"]) == 16
    assert config["program_contract"]["card_specific_evaluators"] is False
    for card in config["cards"]:
        execution = atlas.validate_and_execute_card(card)
        assert execution["dimension_check"] == "PASS"
        assert execution["domain_shape_check"] == "PASS"
        assert len(execution["program_sha256"]) == 64
        assert card["conditions"]["initial"]["status"]
        assert card["conditions"]["boundary"]["status"]
        assert all(len(output["ast_sha256"]) == 64 for output in execution["outputs"])


def test_representative_ast_outputs_and_psd_gates() -> None:
    config = atlas.load_config()
    q04 = atlas.validate_and_execute_card(_card(config, "Q04_CLASSICAL_STOCHASTIC_METRIC"))
    assert _output(q04, "output_mean") == [0, -4]
    assert _output(q04, "output_covariance") == [[19, 5], [5, 5]]
    assert {row["id"] for row in q04["assertions"]} == {
        "C_symmetric",
        "C_PSD",
        "N_symmetric",
        "N_PSD",
    }
    q13 = atlas.validate_and_execute_card(_card(config, "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY"))
    assert _output(q13, "mean") == 12
    assert _output(q13, "ordinary_cumulants") == [12, 12, 12, 12]
    assert _output(q13, "balance_max_abs") < 1e-12
    q07 = atlas.validate_and_execute_card(_card(config, "Q07_ENTANGLEMENT_MEDIATED_GRAVITY"))
    assert _output(q07, "concurrence") > 0


def test_q01_program_mutation_is_rejected_dimensionally() -> None:
    config = atlas.load_config()
    mutated = copy.deepcopy(_card(config, "Q01_MASSIVE_SPIN2"))
    omega_ast = mutated["program"]["outputs"][0]["ast"]
    omega_ast["args"][1]["arg"]["args"][1]["exponent"] = 4
    with pytest.raises(atlas.QuantumAtlasV3Error, match="unit mismatch in add"):
        atlas.validate_and_execute_card(mutated)


def test_q14_and_q15_program_mutations_change_outputs() -> None:
    config = atlas.load_config()
    original_q14 = _card(config, "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY")
    baseline_q14 = atlas.validate_and_execute_card(original_q14)
    mutated_q14 = copy.deepcopy(original_q14)
    determinant_ast = mutated_q14["program"]["outputs"][1]["ast"]
    mutated_q14["program"]["outputs"][1]["ast"] = {
        "op": "add",
        "args": [determinant_ast["left"], determinant_ast["right"]],
    }
    changed_q14 = atlas.validate_and_execute_card(mutated_q14)
    assert _output(baseline_q14, "determinant") == 0
    assert _output(changed_q14, "determinant") == 2

    original_q15 = _card(config, "Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY")
    baseline_q15 = atlas.validate_and_execute_card(original_q15)
    mutated_q15 = copy.deepcopy(original_q15)
    denominator_constant = mutated_q15["program"]["outputs"][0]["ast"]["args"][1]["den"]["args"][0]
    denominator_constant["value"] = 2.0
    changed_q15 = atlas.validate_and_execute_card(mutated_q15)
    assert _output(baseline_q15, "minimum_grid_cost") == pytest.approx(0.5)
    assert _output(changed_q15, "minimum_grid_cost") != pytest.approx(0.5)


def test_undeclared_variables_and_bad_psd_inputs_reject() -> None:
    config = atlas.load_config()
    unknown = copy.deepcopy(_card(config, "Q00_GR_COHERENT_SPIN2_CONTROL"))
    unknown["program"]["outputs"][0]["ast"]["args"][0]["name"] = "unwritten_c"
    with pytest.raises(atlas.QuantumAtlasV3Error, match="forward or undeclared"):
        atlas.validate_and_execute_card(unknown)

    bad_covariance = copy.deepcopy(_card(config, "Q04_CLASSICAL_STOCHASTIC_METRIC"))
    c_decl = next(row for row in bad_covariance["variables"] if row["name"] == "C")
    c_decl["fixture"][0][1] = 9
    with pytest.raises(atlas.QuantumAtlasV3Error, match="domain violation for C"):
        atlas.validate_and_execute_card(bad_covariance)


def test_exact_parameter_mappings_cover_keys_types_units_domains() -> None:
    config = atlas.load_config()
    results = [atlas.execute_parameter_mapping(config, row) for row in config["parameter_mappings"]]
    assert len(results) == 2
    assert all(row["max_abs_residual"] == 0 for row in results)
    assert all(row["source_key_coverage"] == "EXACT" for row in results)
    assert all(row["target_key_coverage"] == "EXACT" for row in results)
    assert all(row["type_shape_unit_domain_checks"] == "PASS" for row in results)


def test_arbitrary_ef03_mapping_rejects_exact_coverage_gate() -> None:
    config = atlas.load_config()
    q11 = _card(config, "Q11_FINITE_OCCUPATION_COHERENCE")
    arbitrary = {
        "id": "BAD_Q11_TO_Q04",
        "source_card": q11["id"],
        "target_card": "Q04_CLASSICAL_STOCHASTIC_METRIC",
        "status": "UNJUSTIFIED",
        "compare_output": "output_mean",
        "source_classification": {row["name"]: "mapped" for row in q11["variables"]},
        "target_assignments": {},
        "scope": "invalid",
    }
    with pytest.raises(atlas.QuantumAtlasV3Error, match="target assignment coverage mismatch"):
        atlas.execute_parameter_mapping(config, arbitrary)
    assert config["equivalence_boundaries"][0]["status"] == "SCOPED_EXISTENCE_NOT_INSTANTIATED"
    assert "Q14 is not included" in config["equivalence_boundaries"][0]["reason"]
    assert config["equivalence_boundaries"][1]["status"] == "NO_PARAMETER_EQUIVALENCE_CLAIMED"


def test_all_empirical_sources_are_blocked_and_readiness_inherits() -> None:
    config = atlas.load_config()
    assert all(row["status"] == "SOURCE_BLOCKED" for row in config["source_manifests"])
    assert all(row["missing"] for row in config["source_manifests"])
    assert all(card["data_readiness"] <= 2 for card in config["cards"])
    m01 = next(
        row for row in config["source_manifests"] if row["id"] == "M01_GWOSC_DISPERSION_BLOCKED"
    )
    assert "exact resolved per-detector product URLs" in m01["missing"]
    assert "frozen waveform implementation/version and source-generation model" in m01["missing"]
    q14 = _card(config, "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY")
    q15 = _card(config, "Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY")
    assert q14["implementation_level"] == "REDUCED_DIAGNOSTIC_NOT_CQ_GENERATOR"
    assert q15["implementation_level"] == "REDUCED_DIAGNOSTIC_NOT_FEEDBACK_MASTER_EQUATION"


def test_deterministic_build_check_replay_and_tamper_gate(tmp_path: Path) -> None:
    base = _copy_package(tmp_path)
    assert atlas.build(base) == "CREATED"
    assert atlas.check(base) == "VALID"
    assert atlas.build(base) == "EXISTING_IDENTICAL"
    receipt = json.loads((base / atlas.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert receipt["counts"]["reduced_AST_cards"] == 16
    assert receipt["counts"]["source_blocked_manifests"] == 11
    assert receipt["counts"]["observational_rows"] == 0
    assert receipt["claim_scope"]["submission_status"] == "READY_FOR_STRICT_REAUDIT"
    proof = (base / atlas.ARTIFACT_DIR / "gaussian-fixed-measurement-proof.md").read_text(
        encoding="utf-8"
    )
    assert "arbitrary finite-dimensional Gaussian random vector" in proof
    assert "not a novelty claim" in proof
    artifact = base / atlas.ARTIFACT_DIR / "AST-program-cards.json"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(atlas.QuantumAtlasV3Error, match="artifact differs"):
        atlas.check(base)


def test_uncommitted_provenance_and_claim_boundaries_are_explicit() -> None:
    config = atlas.load_config()
    assert config["provenance"]["repository_state"] == "UNCOMMITTED_WORKTREE_FILES"
    assert config["provenance"]["commit_sha"] is None
    assert set(config["access_contract"].values()) == {0}
    assert config["claim_boundary"]["source_execution_ready"] is False
    assert config["claim_boundary"]["real_observational_rows_scored"] is False
    assert config["claim_boundary"]["any_branch_empirically_supported"] is False
    assert config["claim_boundary"]["publication_ready"] is False
    assert config["claim_boundary"]["strict_reaudit_submission_ready"] is True
