from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.generated_candidate_metric_variation_execution_campaign import (
    build_generated_candidate_metric_variation_execution_campaign,
    validate_generated_candidate_metric_variation_execution_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generated_candidate_metric_variation_execution_campaign.json"
ARTIFACT = ROOT / "runs/engine/generated-candidate-metric-variation-execution.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_generated_candidate_metric_variation_execution_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    validate_generated_candidate_metric_variation_execution_campaign(committed)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "8adcfbd3846eb60c55c837f972ebe82507d91def2224777f65c3cda69d2afb4e"
    )
    assert committed["content_sha256"] == (
        "bbd5ec183d7710141959361b555a218f7702c095023c06b158d35246569184d8"
    )


def test_all_current_action_hashes_receive_candidate_receipts(rebuilt: dict) -> None:
    records = rebuilt["candidate_records"]
    assert len(records) == 163
    assert len({record["candidate_id"] for record in records}) == 163
    assert len({record["action_sha256"] for record in records}) == 163
    assert Counter(record["family_id"] for record in records) == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
    }
    assert all(record["generic_metric_variation_theorem_bound"] is True for record in records)
    assert all(
        record["candidate_specialized_euler_expression_materialized"] is True for record in records
    )
    assert all(record["candidate_formula_domain_validated"] is True for record in records)
    assert all(record["candidate_action_hash_replayed"] is True for record in records)
    assert all(
        record["candidate_backend_metric_variation_executed"] is False for record in records
    )
    assert all(record["formal_pass_inferred"] is False for record in records)


def test_family_specializers_are_explicit_and_action_bound(rebuilt: dict) -> None:
    from sigma_theory_compiler import (
        generated_candidate_metric_variation_execution_campaign as module,
    )

    adapters = Counter(
        record["metric_variation_execution"]["adapter"] for record in rebuilt["candidate_records"]
    )
    assert adapters == {
        "candidate_specialized_fixed_covector_K1_K2_K3_K4_metric_variation": 128,
        "candidate_specialized_arbitrary_G2_G3_metric_variation": 32,
        "candidate_specialized_arbitrary_G2_metric_variation": 2,
        "candidate_specialized_G2_plus_F_phi_R_metric_variation": 1,
    }
    for record in rebuilt["candidate_records"]:
        execution = record["metric_variation_execution"]
        assert execution["generic_control_status"] == "pass"
        assert execution["specialization"]["specialization_residual"] == "0"
        domain = execution["specialization"]["exact_formula_domain_certificate"]
        assert execution["specialization"]["exact_formula_domain_certificate_sha256"] == (
            module._sha(domain)
        )
        binding = execution["candidate_specialization_binding"]
        binding_body = {key: value for key, value in binding.items() if key != "content_sha256"}
        assert binding["content_sha256"] == module._sha(binding_body)
        assert binding["action_sha256"] == record["action_sha256"]
        assert binding["formula_inputs_sha256"] == record["formula_inputs_sha256"]
        assert execution["candidate_backend_metric_variation_executed"] is False
        assert execution["negative_control"]["rejected"] is True
        assert len(record["action_sha256"]) == 64
        assert len(record["formula_inputs_sha256"]) == 64
        assert len(record["generated_action_record_sha256"]) == 64
    aether = next(
        record
        for record in rebuilt["candidate_records"]
        if record["family_id"] == "AETHER_K1234_PARAMETER_CELL"
    )
    control = aether["metric_variation_execution"]["specialization"]["generic_basis_formal_control"]
    assert control["formal_control_name"] == "cadabra_einstein_aether_metric_variation"
    assert control["backend_return_code"] == 0
    assert control["executed_script_path"] == (
        "formal/cadabra/einstein_aether_metric_variation.cdb"
    )
    assert len(control["script_formal_control_binding_sha256"]) == 64


def test_rational_parameter_substitutions_are_candidate_specific(rebuilt: dict) -> None:
    g2 = [
        record
        for record in rebuilt["candidate_records"]
        if record["family_id"] == "KESSENCE_G2_CONVEX"
    ]
    g3 = [
        record
        for record in rebuilt["candidate_records"]
        if record["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL"
    ]
    assert {
        record["metric_variation_execution"]["specialization"][
            "exact_rational_parameter_substitutions"
        ]["q"]
        for record in g2
    } == {"1/4", "1/8"}
    assert (
        len(
            {
                record["metric_variation_execution"]["specialization"][
                    "exact_rational_parameter_substitutions"
                ]["beta"]
                for record in g3
            }
        )
        == 32
    )
    assert all(
        record["metric_variation_execution"]["specialization"][
            "exact_formula_domain_certificate"
        ]["status"]
        == "pass_exact_linear_G3_jet_domain"
        for record in g3
    )
    g4 = next(
        record
        for record in rebuilt["candidate_records"]
        if record["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR"
    )
    g4_domain = g4["metric_variation_execution"]["specialization"][
        "exact_formula_domain_certificate"
    ]
    assert g4_domain["G4_X"] == "0"
    assert g4_domain["G4_XX"] == "0"
    assert g4_domain["G4_phi"] == "phi/50"
    assert g4_domain["G4_phiphi"] == "1/50"
    assert g4_domain["X_independence_exact"] is True


def test_hash_parameter_and_atom_tampering_fail_closed() -> None:
    config = _load(CONFIG)
    bad_hash = copy.deepcopy(config)
    bad_hash["generated_action_export"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generated action export file hash mismatch"):
        build_generated_candidate_metric_variation_execution_campaign(bad_hash, ROOT)

    bad_controls = copy.deepcopy(config)
    bad_controls["formal_controls_artifact"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="formal controls file hash mismatch"):
        build_generated_candidate_metric_variation_execution_campaign(bad_controls, ROOT)

    bad_compiler = copy.deepcopy(config)
    bad_compiler["compilation_config"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="compilation config file hash mismatch"):
        build_generated_candidate_metric_variation_execution_campaign(bad_compiler, ROOT)

    export_path = ROOT / config["generated_action_export"]["path"]
    export = _load(export_path)
    g4 = next(
        record
        for record in export["candidate_records"]
        if record["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR"
    )
    g4["theory_formula_inputs"]["parameters"]["G4"] = "1/2+X_phi"
    formula = g4["theory_formula_inputs"]
    formula_body = {key: value for key, value in formula.items() if key != "formula_inputs_sha256"}
    module = pytest.importorskip(
        "sigma_theory_compiler.generated_candidate_metric_variation_execution_campaign"
    )
    formula["formula_inputs_sha256"] = module._sha(formula_body)
    g4["formula_inputs_sha256"] = formula["formula_inputs_sha256"]
    g4_body = {key: value for key, value in g4.items() if key != "content_sha256"}
    g4["content_sha256"] = module._sha(g4_body)
    export["candidate_record_registry_root_sha256"] = module._sha(
        [item["content_sha256"] for item in export["candidate_records"]]
    )
    original_loader = module._load_bound
    try:
        module._load_bound = lambda root, binding, label: (
            export if label == "generated action export" else original_loader(root, binding, label)
        )
        with pytest.raises(ValueError, match="candidate action lineage changed"):
            build_generated_candidate_metric_variation_execution_campaign(config, ROOT)
    finally:
        module._load_bound = original_loader

    export_path = ROOT / config["generated_action_export"]["path"]
    export = _load(export_path)
    original = export["candidate_records"][0]["theory_formula_inputs"][
        "ordered_operator_densities"
    ][0]["atom"]
    export["candidate_records"][0]["theory_formula_inputs"]["ordered_operator_densities"][0][
        "atom"
    ] = "FORBIDDEN"
    config_with_memory = copy.deepcopy(config)
    from sigma_theory_compiler import (
        generated_candidate_metric_variation_execution_campaign as module,
    )

    original_loader = module._load_bound
    try:
        module._load_bound = lambda root, binding, label: (
            export if label == "generated action export" else original_loader(root, binding, label)
        )
        with pytest.raises(ValueError, match="ordered action atoms changed"):
            build_generated_candidate_metric_variation_execution_campaign(config_with_memory, ROOT)
    finally:
        module._load_bound = original_loader
        export["candidate_records"][0]["theory_formula_inputs"]["ordered_operator_densities"][0][
            "atom"
        ] = original


def test_density_action_and_aether_script_coherent_tampering_fail_closed() -> None:
    from sigma_theory_compiler import (
        generated_candidate_metric_variation_execution_campaign as module,
    )

    config = _load(CONFIG)
    export = _load(ROOT / config["generated_action_export"]["path"])
    density_tamper = copy.deepcopy(export)
    target = density_tamper["candidate_records"][0]
    target["theory_formula_inputs"]["ordered_operator_densities"][0]["density"] += "+0"
    formula = target["theory_formula_inputs"]
    formula_body = {key: value for key, value in formula.items() if key != "formula_inputs_sha256"}
    formula["formula_inputs_sha256"] = module._sha(formula_body)
    target["formula_inputs_sha256"] = formula["formula_inputs_sha256"]
    target_body = {key: value for key, value in target.items() if key != "content_sha256"}
    target["content_sha256"] = module._sha(target_body)
    density_tamper["candidate_record_registry_root_sha256"] = module._sha(
        [item["content_sha256"] for item in density_tamper["candidate_records"]]
    )

    original_loader = module._load_bound
    try:
        module._load_bound = lambda root, binding, label: (
            density_tamper if label == "generated action export" else original_loader(root, binding, label)
        )
        with pytest.raises(ValueError, match="candidate action lineage changed"):
            build_generated_candidate_metric_variation_execution_campaign(config, ROOT)
    finally:
        module._load_bound = original_loader

    action_tamper = copy.deepcopy(export)
    target = action_tamper["candidate_records"][0]
    target["action_sha256"] = "0" * 64
    target["theory_formula_inputs"]["action_content_sha256"] = "0" * 64
    formula = target["theory_formula_inputs"]
    formula_body = {key: value for key, value in formula.items() if key != "formula_inputs_sha256"}
    formula["formula_inputs_sha256"] = module._sha(formula_body)
    target["formula_inputs_sha256"] = formula["formula_inputs_sha256"]
    target_body = {key: value for key, value in target.items() if key != "content_sha256"}
    target["content_sha256"] = module._sha(target_body)
    action_tamper["candidate_record_registry_root_sha256"] = module._sha(
        [item["content_sha256"] for item in action_tamper["candidate_records"]]
    )
    try:
        module._load_bound = lambda root, binding, label: (
            action_tamper if label == "generated action export" else original_loader(root, binding, label)
        )
        with pytest.raises(ValueError, match="candidate action lineage changed"):
            build_generated_candidate_metric_variation_execution_campaign(config, ROOT)
    finally:
        module._load_bound = original_loader

    receipt = _load(ROOT / config["aether_execution_receipt"]["path"])
    receipt["script_path"] = "formal/cadabra/not-the-bound-script.cdb"
    receipt_body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = module._sha(receipt_body)
    try:
        module._load_bound = lambda root, binding, label: (
            receipt if label == "Aether execution receipt" else original_loader(root, binding, label)
        )
        with pytest.raises(ValueError, match="Aether metric-variation formal control"):
            build_generated_candidate_metric_variation_execution_campaign(config, ROOT)
    finally:
        module._load_bound = original_loader


def test_scope_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["current_operator_families_complete"] is True
    assert rebuilt["future_unregistered_operator_families_complete"] is False
    assert rebuilt["first_missing_premise"] == (
        "metric_variation_exporters_for_future_unregistered_nonminimal_operator_families"
    )
    assert rebuilt["metric_variation_execution_counts"]["formal_passes_inferred"] == 0
    counts = rebuilt["metric_variation_execution_counts"]
    assert counts["aether_formal_control_bound"] == 128
    assert counts["candidate_action_hashes_specialized"] == 163
    assert counts["candidate_euler_expressions_materialized"] == 163
    assert counts["typed_action_hashes_replayed"] == 163
    assert counts["exact_formula_domains_validated"] == 163
    assert counts["candidate_specializations_symbolically_verified"] == 163
    assert counts["candidate_backend_variations_executed"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
