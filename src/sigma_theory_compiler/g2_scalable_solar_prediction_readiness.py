from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .promotion_orchestrator import ELIGIBILITY
from .solar_promotion_evaluator import (
    GR_SOLAR_BUNDLE,
    bundle_binding,
    solar_known_answer_evaluator,
)

SCHEMA_VERSION = "sigma-g2-scalable-solar-prediction-readiness-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _load_bound(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = json.loads(_bound_path(root, binding, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding["content_sha256"]
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _compact_records(export: dict[str, Any]) -> list[dict[str, Any]]:
    columns = export.get("candidate_record_columns")
    rows = export.get("candidate_records")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise TypeError("scalable export compact candidate table missing")
    if len(columns) != len(set(columns)):
        raise ValueError("scalable export candidate columns are not unique")
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError("scalable export candidate row shape changed")
        records.append(dict(zip(columns, row, strict=True)))
    return records


def _replay_exact_action(
    root: Path,
    compiler_config: dict[str, Any],
    candidate_id: str,
    action_sha256: str,
) -> dict[str, Any]:
    manifest = _load_bound(
        root,
        compiler_config["parameter_cell_manifest"],
        "parameter-cell manifest",
        content=True,
    )
    source = _load_bound(
        root,
        compiler_config["source_seed_manifest"],
        "source seed manifest",
        content=True,
    )
    _bound_path(root, compiler_config["compiler_semantics"], "compiler semantics")
    family = next(
        item for item in source["typed_family_seeds"] if item["family_id"] == "KESSENCE_G2_CONVEX"
    )
    manifest_binding = {
        "parameter_cell_manifest_content_sha256": manifest["content_sha256"],
        "parameter_cell_registry_root_sha256": manifest["parameter_cell_registry_root_sha256"],
    }
    candidates = []
    for cell in iter_parameter_cells(manifest, source):
        if cell["family_id"] != "KESSENCE_G2_CONVEX":
            continue
        equivalence = _sha(_action_density_key(cell))
        if "G3A-" + equivalence[:24] != candidate_id:
            continue
        pseudo_seed = {
            "seed_id": cell["parameter_cell_id"],
            "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"],
            "family_lineage_sha256": cell["family_lineage_sha256"],
            "theory_contract": cell["theory_contract"],
            "operator_atoms": cell["operator_atoms"],
            "parameters": cell["parameters"],
        }
        action = _compile_action_ir(pseudo_seed, family, manifest_binding)
        if action["content_sha256"] == action_sha256:
            candidates.append((cell, action, equivalence))
    if len(candidates) != 1:
        raise ValueError("exact target action did not replay uniquely from typed compiler inputs")
    cell, action, equivalence = candidates[0]
    if (
        action.get("family_id") != "KESSENCE_G2_CONVEX"
        or action.get("fields") != ["g_mu_nu", "phi"]
        or [item["atom"] for item in action.get("operators", [])] != ["EH_R", "G2_PHI_X"]
        or action.get("matter_coupling") != {"metric": "g_mu_nu", "universal": True}
        or action.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("replayed target action field/operator/matter contract changed")
    body = {
        "candidate_id": candidate_id,
        "action_sha256": action_sha256,
        "action_density_equivalence_sha256": equivalence,
        "parameter_cell_id": cell["parameter_cell_id"],
        "parameter_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
        "fields": action["fields"],
        "operators": action["operators"],
        "parameters": action["parameters"],
        "matter_coupling": action["matter_coupling"],
        "data_eligibility": action["data_eligibility"],
    }
    return {**body, "content_sha256": _sha(body)}


def _scalar_free_prediction_certificate(coefficient: Fraction) -> dict[str, Any]:
    if coefficient not in {Fraction(1, 4), Fraction(1, 8)}:
        raise ValueError("unsupported G2 Solar-readiness coefficient")
    x = sp.Symbol("X", real=True)
    g2 = x + sp.Rational(coefficient.numerator, coefficient.denominator) * x**2
    residuals = {
        "G2_at_X_zero": sp.simplify(g2.subs(x, 0)),
        "G2_X_at_X_zero_minus_one": sp.simplify(sp.diff(g2, x).subs(x, 0) - 1),
        "scalar_current_at_constant_phi": sp.Integer(0),
        "scalar_stress_at_constant_phi": sp.Integer(0),
    }
    if any(value != 0 for value in residuals.values()):
        raise ValueError("G2 scalar-free branch residual failed")
    body = {
        "G2": str(g2),
        "background_and_boundary": "phi=phi_infinity_constant_and_X=0",
        "exact_residuals": {key: str(value) for key, value in residuals.items()},
        "scalar_equation": "nabla_mu(G2_X*nabla^mu(phi))=0",
        "linear_scalar_equation": "box(delta_phi)=0_with_no_matter_source",
        "metric_backreaction_order": "T_phi_is_quadratic_or_higher_in_delta_phi",
        "branch_result": (
            "every Einstein solution with universally g_mu_nu-coupled matter is an exact "
            "candidate solution for constant phi"
        ),
        "Newtonian_prediction": {
            "G_cav_over_G_star": "1",
            "Poisson_equation": "laplacian(U)=4*pi*G_star*rho",
            "exterior_potential": "U=G_star*M/r",
            "status": "pass_on_exact_constant_phi_branch",
        },
        "PPN_prediction": {
            "gamma": "1",
            "beta": "1",
            "preferred_frame_parameters": "0_on_scalar_free_Lorentz_invariant_background",
            "status": "pass_on_exact_constant_phi_branch",
        },
        "vacuum_exterior": "Schwarzschild_is_exact_on_this_branch",
        "known_answer_formulas": {
            "perihelion_per_orbit": "6*pi*G_N*M/(a*(1-e^2)*c^2)",
            "light_deflection": "4*G_N*M/(b*c^2)",
            "one_way_Shapiro": "2*G_N*M/c^3*log(4*r_E*r_R/b^2)",
        },
        "scope": (
            "candidate-derived analytic prediction on the exact constant-scalar branch; not a "
            "measurement, fitted PPN result, or claim about nonzero-gradient initial data"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _static_source_class_certificate(coefficient: Fraction) -> dict[str, Any]:
    two_a = 2 * coefficient
    body = {
        "source_class": (
            "static_connected_regular_horizonless_weak_compact_source_with_universal_minimal_"
            "metric_coupling"
        ),
        "required_domain": {
            "lapse": "N>=N_min>0",
            "inner_boundary": "none",
            "scalar_symmetry": "Lie_t(phi)=0",
            "asymptotic_boundary": "phi->phi_infinity_constant",
            "boundary_flux": ("lim_r_to_infinity integral_Sr N*(phi-phi_infinity)*G2_X*D_r(phi)=0"),
            "ellipticity_sign": f"G2_X=1+({two_a})*X>0_everywhere",
        },
        "static_equation": "D_i(N*G2_X*D^i(phi))=0",
        "integrated_identity": ("integral_Sigma N*G2_X*D_i(phi)*D^i(phi)=zero_boundary_flux"),
        "conditional_conclusion": "D_i(phi)=0_and_phi=phi_infinity",
        "formal_cell_shortcut": (
            "on_Lie_t_phi_zero_data_X=-D_i(phi)D^i(phi)/2_and_registered_X>=0_already_"
            "forces_D_i(phi)=0"
        ),
        "status": "pass_as_conditional_source_class_theorem",
        "real_sun_instantiated": False,
        "missing_real_source_facts": [
            "registered_staticity_and_positive_lapse_domain",
            "registered_regular_horizonless_topology",
            "registered_constant_scalar_asymptotic_boundary_and_zero_flux",
            "registered_candidate_use_authorization",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def _calibration_certificate(reference: dict[str, Any]) -> dict[str, Any]:
    checks = reference.get("golden_checks", [])
    statuses = {str(item.get("name")): str(item.get("status")) for item in checks}
    required = {
        "schwarzschild_vacuum",
        "gr_ppn_recovery",
        "mercury_perihelion",
        "solar_limb_light_deflection",
        "shapiro_delay_geometry_control",
    }
    if set(statuses) != required or set(statuses.values()) != {"pass"}:
        raise ValueError("GR Solar calibration known-answer checks changed")
    body = {
        "role": "solver_calibration_control_not_candidate_evidence",
        "reference_action_sha256": reference["formal_prerequisite"]["input_action_sha256"],
        "statuses": dict(sorted(statuses.items())),
        "candidate_inference_rule": (
            "candidate predictions follow only from its exact action replay and scalar-free "
            "certificate; the GR control action or outputs are never substituted"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _solar_readiness(
    protocol: dict[str, Any],
    audit: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or audit.get("status") != "pass"
        or audit.get("observational_dataset_opened") is not False
        or source.get("status") != "metadata_registered_data_sealed"
        or source.get("data_opened") is not False
        or source.get("candidate_use_authorized") is not False
    ):
        raise ValueError("Solar protocol/source observation seal changed")
    candidate = {
        "candidate_id": target["candidate_id"],
        "ordinal": target["ordinal"],
        "correction_expression": target["G2"],
        "data_eligibility": dict(ELIGIBILITY),
    }
    context = {
        "stage_name": "solar_known_answer_controls",
        "category": "observational",
        "attempt": 1,
        "input_lineage_sha256": target["formal_record_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
    }
    evaluator_result = solar_known_answer_evaluator(candidate, context)
    if (
        evaluator_result.get("decision") != "blocked"
        or evaluator_result.get("blocker") != "missing_exact_action_bound_solar_control_bundle"
    ):
        raise ValueError("Solar evaluator unexpectedly authorized a G2 candidate")
    reused = {
        **candidate,
        "solar_control_provenance": {
            "bundle_id": GR_SOLAR_BUNDLE["bundle_id"],
            "bundle_binding_sha256": bundle_binding(GR_SOLAR_BUNDLE),
            "input_action_sha256": GR_SOLAR_BUNDLE["input_action_sha256"],
        },
    }
    try:
        solar_known_answer_evaluator(reused, context)
    except ValueError as error:
        reuse_error = str(error)
    else:
        raise ValueError("GR Solar control bundle was reusable by a G2 candidate")
    if reuse_error != "Solar known-answer bundle cannot be attached to a discovery candidate":
        raise ValueError("GR Solar control reuse negative changed")
    missing = [
        "candidate_specific_real_source_contract_sha256",
        "source_branch_domain_instantiation_sha256",
        "candidate_specific_evaluator_descriptor_sha256",
        "training_only_initial_state_sha256",
        "frozen_nuisance_likelihood_stopping_rule_sha256",
        "held_out_split_commitment_sha256",
        "action_bound_prediction_bundle_descriptor_sha256",
        "action_bound_prediction_bundle_file_sha256",
        "selected_primary_record_roots_sha256",
        "observation_opening_authorization_sha256",
    ]
    body = {
        "decision": "blocked",
        "first_missing_premise": (
            "registered_candidate_specific_real_source_and_action_bound_Solar_prediction_bundle"
        ),
        "missing_registration_fields": missing,
        "filled_registration_fields": {
            "frozen_protocol_file_sha256": target["protocol_file_sha256"],
            "analytic_prediction_certificate_sha256": target[
                "analytic_prediction_certificate_sha256"
            ],
            "formal_pass_record_sha256": target["formal_record_sha256"],
        },
        "current_evaluator_result": evaluator_result,
        "GR_control_bundle_reuse_negative": {"rejected": True, "reason": reuse_error},
        "source_metadata_status": source["status"],
        "candidate_use_authorized": False,
        "observation_opening_authorization_present": False,
        "observational_inputs_opened_by_this_audit": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_target(
    target: dict[str, Any],
    formal_record: dict[str, Any],
    export_record: dict[str, Any],
) -> Fraction:
    if (
        formal_record.get("candidate_id") != target["candidate_id"]
        or formal_record.get("typed_action_ir_sha256") != target["action_sha256"]
        or formal_record.get("G2") != target["G2"]
        or formal_record.get("decision") != "pass"
        or formal_record.get("content_sha256") != target["formal_record_sha256"]
        or formal_record.get("previous_blocker_closed")
        != "hash_bound_general_nonmaximal_positive_mass_theorem"
    ):
        raise ValueError("G2 formal-pass target binding changed")
    formula = export_record.get("theory_formula_inputs", {})
    if (
        export_record.get("candidate_id") != target["candidate_id"]
        or export_record.get("action_sha256") != target["action_sha256"]
        or formula.get("action_content_sha256") != target["action_sha256"]
        or formula.get("formula_inputs_sha256") != target["formula_inputs_sha256"]
        or formula.get("parameters") != {"G2": target["G2"], "X_domain": "0<=X_phi<=1/32"}
    ):
        raise ValueError("G2 exact action/formula export binding changed")
    return Fraction(target["quadratic_coefficient"])


def build_g2_scalable_solar_prediction_readiness(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("schema_version") != "sigma-g2-scalable-solar-prediction-readiness-config-1.0":
        raise ValueError("G2 Solar readiness config schema changed")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G2 Solar readiness eligibility is not fail-closed")
    _bound_path(root, config["adapter_source"], "G2 Solar readiness adapter source")
    formal = _load_bound(root, config["formal_pass"], "G2 formal pass", content=True)
    export = _load_bound(root, config["scalable_export"], "scalable export", content=True)
    compiler_config = _load_bound(root, config["compiler_config"], "compiler config")
    compilation = _load_bound(root, config["compilation_campaign"], "compilation", content=True)
    reference = _load_bound(root, config["calibration_reference"], "GR calibration")
    solar = config["solar_contract"]
    protocol = _load_bound(root, solar["protocol"], "Solar protocol")
    protocol_audit = _load_bound(root, solar["protocol_audit"], "Solar protocol audit")
    source = _load_bound(root, solar["source_registration"], "Solar source registration")
    _bound_path(root, solar["evaluator_source"], "Solar evaluator source")
    descriptor = _load_bound(root, solar["evaluator_descriptor"], "Solar evaluator descriptor")
    if (
        descriptor.get("artifact_sha256") != solar["evaluator_source"]["file_sha256"]
        or descriptor.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("Solar evaluator descriptor binding changed")
    if (
        compilation.get("structural_gate_pass_counts", {}).get("universal_matter_coupling") != 256
        or compilation.get("negative_control_counts") != {"reject": 5}
        or compilation.get("observational_data_opened") is not False
        or compilation.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("typed compilation universal-coupling evidence changed")
    formal_by_id = {item["candidate_id"]: item for item in formal["candidate_records"]}
    export_by_id = {item["candidate_id"]: item for item in _compact_records(export)}
    calibration = _calibration_certificate(reference)
    records = []
    for target_config in config["targets"]:
        formal_record = formal_by_id.get(target_config["candidate_id"])
        export_record = export_by_id.get(target_config["candidate_id"])
        if formal_record is None or export_record is None:
            raise ValueError("G2 Solar target missing from bound predecessor")
        coefficient = _validate_target(target_config, formal_record, export_record)
        action = _replay_exact_action(
            root, compiler_config, target_config["candidate_id"], target_config["action_sha256"]
        )
        prediction = _scalar_free_prediction_certificate(coefficient)
        source_class = _static_source_class_certificate(coefficient)
        target = {
            **target_config,
            "protocol_file_sha256": solar["protocol"]["file_sha256"],
            "analytic_prediction_certificate_sha256": prediction["content_sha256"],
        }
        readiness = _solar_readiness(protocol, protocol_audit, source, target)
        provenance_body = {
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "formal_pass_content_sha256": config["formal_pass"]["content_sha256"],
            "formal_record_sha256": target["formal_record_sha256"],
            "formula_inputs_sha256": target["formula_inputs_sha256"],
            "action_replay_sha256": action["content_sha256"],
            "analytic_prediction_sha256": prediction["content_sha256"],
            "source_class_sha256": source_class["content_sha256"],
            "calibration_sha256": calibration["content_sha256"],
            "readiness_sha256": readiness["content_sha256"],
            "data_eligibility": ELIGIBILITY,
        }
        record_body = {
            "candidate_id": target["candidate_id"],
            "action_sha256": target["action_sha256"],
            "G2": target["G2"],
            "decision": "blocked",
            "candidate_analytic_prediction_status": "pass_on_exact_constant_phi_branch",
            "exact_action_replay": action,
            "scalar_free_prediction_certificate": prediction,
            "static_source_class_certificate": source_class,
            "GR_calibration_control": calibration,
            "real_solar_readiness": readiness,
            "gate_ledger": {
                "exact_action_and_formal_pass_binding": "pass",
                "universal_minimal_matter_coupling": "pass",
                "exact_constant_scalar_GR_branch": "pass",
                "Newtonian_limit": "pass",
                "PPN_gamma_beta": "pass",
                "conditional_static_source_class_uniqueness": "pass",
                "candidate_specific_real_source_registration": "blocked",
                "candidate_specific_evaluator_and_prediction_bundle": "blocked",
                "observation_opening_authorization": "blocked",
            },
            "first_missing_premise": readiness["first_missing_premise"],
            "necessary_condition_rejection_found": False,
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
            "solar_bundle": {
                "analytic_prediction_certificate_generated": True,
                "candidate_specific_evaluator_bundle_generated": False,
                "real_observational_bundle_generated": False,
                "real_observational_bundle_admissible": False,
                "status": "blocked_before_data_opening",
            },
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    records.sort(key=lambda item: item["candidate_id"])
    if len(records) != 2 or len({item["candidate_id"] for item in records}) != 2:
        raise ValueError("G2 Solar readiness requires exactly two unique candidates")
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "adapter_source": config["adapter_source"],
            "formal_pass": config["formal_pass"],
            "scalable_export": config["scalable_export"],
            "compiler_config": config["compiler_config"],
            "compilation_campaign": config["compilation_campaign"],
            "calibration_reference": config["calibration_reference"],
            "solar_contract": solar,
        },
        "candidate_count": 2,
        "decision_counts": {"blocked": 2},
        "candidate_analytic_prediction_pass_count": 2,
        "conditional_static_source_class_pass_count": 2,
        "real_source_registration_pass_count": 0,
        "real_solar_bundle_count": 0,
        "real_solar_bundle_admissible_count": 0,
        "candidate_records": records,
        "candidate_registry_root_sha256": _sha(
            [[item["candidate_id"], item["action_sha256"]] for item in records]
        ),
        "result_registry_root_sha256": _sha(
            [[item["candidate_id"], item["content_sha256"]] for item in records]
        ),
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Both exact scalable G2 actions have an exact constant-scalar GR branch and therefore "
            "predict the GR Newtonian potential and PPN gamma=beta=1 on that branch. A conditional "
            "static source-class identity also excludes nonconstant regular scalar profiles when "
            "its boundary and G2_X>0 premises hold. Neither real Solar readiness result passes: "
            "candidate-specific source/domain registration, evaluator and prediction-bundle "
            "lineage, selected primary roots, and explicit observation authorization are absent."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
