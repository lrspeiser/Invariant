"""Candidate-specific formal follow-up for future Aether preflight survivors."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .aether_parameter_cell_formal_gate_campaign import _specialize
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_future_parameter_formal_preflight_campaign import (
    build_reviewed_future_parameter_formal_preflight,
)

CONFIG_SCHEMA = "sigma-future-aether-candidate-formal-followup-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-candidate-formal-followup-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
BLOCKER = "full_constraint_embedding_of_negative_static_twist_jet"
EXPECTED_ADAPTER_IDS = {
    "einstein_aether_generic_dh_covariance",
    "einstein_aether_generic_hh_deformation_kinematics",
    "einstein_aether_linearized_physical_energy",
    "einstein_aether_restricted_nonlinear_total_energy",
    "einstein_aether_global_tilt_legendre_strata",
    "maxwell_unit_aether_nonlinear_hamiltonian",
}
EXPECTED_APPLICABILITY = {
    "einstein_aether_generic_dh_covariance": "generic_family_identity_specialized_by_exact_action",
    "einstein_aether_generic_hh_deformation_kinematics": "regular_patch_kinematics_specialized_by_exact_action",
    "einstein_aether_linearized_physical_energy": "candidate_rational_formula_specialized",
    "einstein_aether_restricted_nonlinear_total_energy": "restricted_subsector_scope_exclusion_only",
    "einstein_aether_global_tilt_legendre_strata": "candidate_rational_formula_specialized",
    "maxwell_unit_aether_nonlinear_hamiltonian": "control_only_not_action_equivalent",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("content_sha256") != binding.get("content_sha256")
            or _sha(body) != binding["content_sha256"]
        ):
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_preflight_artifact",
        "source_preflight_config",
        "source_preflight_implementation",
        "aether_formal_control_config",
        "aether_formal_campaign_implementation",
        "formal_report",
        "prior_twist_audit",
        "required_arbitrary_background_noether",
        "reviewed_adapters",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether formal-followup config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether formal-followup eligibility is open")
    if config.get("observational_authorization") is not False:
        raise ValueError("future Aether formal-followup opened observations")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("future Aether formal-followup enabled paid LLM calls")
    budget = config["budget"]
    if set(budget) != {
        "maximum_candidates",
        "maximum_formal_adapter_replays",
        "maximum_bound_cadabra_controls",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_candidates"]) != 14
        or int(budget["maximum_formal_adapter_replays"]) != 6
        or int(budget["maximum_bound_cadabra_controls"]) != 1
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("future Aether formal-followup budget is not exact")
    descriptors = config["reviewed_adapters"]
    if (
        len(descriptors) != 6
        or {item.get("formal_check_id") for item in descriptors} != EXPECTED_ADAPTER_IDS
        or {item.get("formal_check_id"): item.get("applicability") for item in descriptors}
        != EXPECTED_APPLICABILITY
    ):
        raise ValueError("future Aether reviewed adapter registry is incomplete")


def _validate_bound_noether_control(
    config: dict[str, Any], report: dict[str, Any], root: Path
) -> dict[str, Any]:
    descriptor = config["required_arbitrary_background_noether"]
    if (
        set(descriptor) != {"formal_check_id", "evidence_sha256", "scripts"}
        or descriptor["formal_check_id"] != "einstein_aether_arbitrary_background_4d_noether"
    ):
        raise ValueError("arbitrary-background Aether Noether descriptor is invalid")
    scripts = descriptor["scripts"]
    expected_labels = {
        "action_diffeomorphism_covariance",
        "euler_noether_coefficient",
        "metric_euler",
        "vector_multiplier_euler",
    }
    if set(scripts) != expected_labels:
        raise ValueError("arbitrary-background Aether Noether script registry is incomplete")
    check = next(
        (
            item
            for item in report.get("checks", [])
            if item.get("name") == descriptor["formal_check_id"]
        ),
        None,
    )
    evidence = check.get("evidence") if isinstance(check, dict) else None
    if (
        not isinstance(evidence, dict)
        or check.get("status") != "pass"
        or evidence.get("passed") is not True
        or _sha(evidence) != descriptor["evidence_sha256"]
        or evidence.get("scope")
        != "exact off-shell arbitrary-background abstract-tensor identity for the standard Einstein-Aether K1..K4 plus unit action in the fixed-covector convention"
    ):
        raise ValueError("arbitrary-background Aether Noether control changed")
    stable_scripts = {}
    for label in sorted(expected_labels):
        binding = scripts[label]
        path = _bound_path(root, binding, "arbitrary-background Aether Noether script")
        report_script = evidence.get("scripts", {}).get(label, {})
        if (
            report_script.get("passed") is not True
            or report_script.get("sha256") != binding["file_sha256"]
            or Path(report_script.get("script", "")).name != path.name
        ):
            raise ValueError("arbitrary-background Aether Noether script evidence changed")
        stable_scripts[label] = {
            "path": binding["path"],
            "file_sha256": binding["file_sha256"],
        }
    body = {
        "formal_check_id": descriptor["formal_check_id"],
        "evidence_sha256": descriptor["evidence_sha256"],
        "script_registry_root_sha256": _sha(stable_scripts),
        "scope": evidence["scope"],
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _resolve_adapter(root: Path, descriptor: dict[str, Any]) -> Any:
    source = _bound_path(
        root,
        {
            "path": descriptor["source_path"],
            "file_sha256": descriptor["source_file_sha256"],
        },
        "reviewed Aether adapter source",
    )
    module_name, separator, attribute = descriptor["entrypoint"].partition(":")
    if not separator:
        raise ValueError("reviewed Aether adapter entrypoint must use module:function")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    callback_source = (
        inspect.getsourcefile(inspect.unwrap(callback)) if callable(callback) else None
    )
    if (
        not callable(callback)
        or callback_source is None
        or Path(callback_source).resolve() != source
    ):
        raise ValueError("reviewed Aether adapter is not defined by its bound source")
    return callback


def _replay_adapters(
    config: dict[str, Any],
    formal_config: dict[str, Any],
    report: dict[str, Any],
    root: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    formal_descriptors = {item["id"]: item for item in formal_config.get("formal_adapters", [])}
    checks = {item["name"]: item for item in report.get("checks", [])}
    stable = {}
    for descriptor in config["reviewed_adapters"]:
        if set(descriptor) != {
            "formal_check_id",
            "entrypoint",
            "source_path",
            "source_file_sha256",
            "evidence_sha256",
            "applicability",
        }:
            raise ValueError("reviewed Aether adapter descriptor fields are invalid")
        check_id = descriptor["formal_check_id"]
        formal_descriptor = formal_descriptors.get(check_id)
        if formal_descriptor != {
            "id": check_id,
            "source_path": descriptor["source_path"],
            "source_file_sha256": descriptor["source_file_sha256"],
            "evidence_sha256": descriptor["evidence_sha256"],
        }:
            raise ValueError(f"reviewed Aether formal descriptor changed: {check_id}")
        callback = _resolve_adapter(root, descriptor)
        evidence = callback()
        check = checks.get(check_id)
        if (
            not isinstance(evidence, dict)
            or evidence.get("passed") is not True
            or not isinstance(check, dict)
            or check.get("status") != "pass"
            or evidence != check.get("evidence")
            or _sha(evidence) != descriptor["evidence_sha256"]
        ):
            raise ValueError(f"reviewed Aether adapter replay mismatch: {check_id}")
        body = {
            "formal_check_id": check_id,
            "entrypoint": descriptor["entrypoint"],
            "source_file_sha256": descriptor["source_file_sha256"],
            "evidence_sha256": descriptor["evidence_sha256"],
            "applicability": descriptor["applicability"],
            "scope": evidence.get("scope"),
            "status": "pass",
        }
        stable[check_id] = {**body, "content_sha256": _sha(body)}
    if len(stable) != int(config["budget"]["maximum_formal_adapter_replays"]):
        raise ValueError("reviewed Aether adapter replay count changed")
    nonlinear = checks["einstein_aether_restricted_nonlinear_total_energy"]["evidence"]
    maxwell = checks["maxwell_unit_aether_nonlinear_hamiltonian"]["evidence"]
    if (
        nonlinear.get("generic_status") != "unresolved"
        or nonlinear.get("energy_not_local_density") is not True
        or nonlinear.get("out_of_domain_controls", {})
        .get("aether_with_twist", {})
        .get("theorem_premise_rejected")
        is not True
        or maxwell.get("subclass") != "Maxwell-form unit Aether: c3=-c1, c2=c4=0 up to convention"
    ):
        raise ValueError("reviewed nonlinear Aether adapter scopes changed")
    return stable, _sha(stable)


def _validate_prior_twist(prior: dict[str, Any]) -> str:
    formulas = {
        record["twist_sector_certificate"]["nonlinear_static_pure_twist_sector"][
            "exact_hamiltonian"
        ]
        for record in prior.get("candidate_records", [])
    }
    expected = "H_twist=(M_Pl^2/2)*[(c1-c3)*W2-(c1/(1+y)+c4)*WA2]"
    if formulas != {expected}:
        raise ValueError("reviewed Aether static twist Hamiltonian changed")
    return expected


def build_future_aether_candidate_formal_followup(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    """Replay all inputs and evaluate the 14 future Aether preflight survivors."""
    _validate_config(config)
    root = Path(root).resolve()
    _bound_path(root, config["campaign_implementation"], "campaign implementation")
    preflight = _bound_json(
        root, config["source_preflight_artifact"], "source preflight", content=True
    )
    preflight_config = _bound_json(
        root, config["source_preflight_config"], "source preflight config"
    )
    _bound_path(root, config["source_preflight_implementation"], "source preflight implementation")
    if preflight_config.get("campaign_implementation") != config["source_preflight_implementation"]:
        raise ValueError("source preflight implementation binding changed")
    rebuilt_preflight = build_reviewed_future_parameter_formal_preflight(preflight_config, root)
    if rebuilt_preflight != preflight:
        raise ValueError("source preflight does not replay exactly")
    formal_config = _bound_json(
        root, config["aether_formal_control_config"], "Aether formal control config"
    )
    _bound_path(
        root,
        config["aether_formal_campaign_implementation"],
        "Aether formal campaign implementation",
    )
    if (
        formal_config.get("source_bindings", {}).get("campaign_source")
        != config["aether_formal_campaign_implementation"]
    ):
        raise ValueError("Aether formal campaign implementation binding changed")
    report = _bound_json(root, config["formal_report"], "formal report")
    prior_twist = _bound_json(root, config["prior_twist_audit"], "prior twist audit", content=True)
    if (
        formal_config["source_bindings"]["formal_report"] != config["formal_report"]
        or formal_config["source_bindings"]["prior_twist_audit"] != config["prior_twist_audit"]
    ):
        raise ValueError("Aether formal source bindings changed")
    twist_hamiltonian = _validate_prior_twist(prior_twist)
    noether_control = _validate_bound_noether_control(config, report, root)
    adapter_evidence, adapter_root = _replay_adapters(config, formal_config, report, root)
    formal_evidence_root = _sha(
        {
            "reviewed_adapter_evidence_root_sha256": adapter_root,
            "arbitrary_background_noether_control_sha256": noether_control["content_sha256"],
        }
    )
    targets = [
        record
        for record in preflight["candidate_records"]
        if record["family_id"] == TARGET_FAMILY and record["decision"] == "pass"
    ]
    if (
        len(targets) != int(config["budget"]["maximum_candidates"])
        or preflight.get("family_decision_counts", {}).get(TARGET_FAMILY)
        != {"pass": 14, "reject": 2}
        or preflight.get("promotion", {}).get("eligible_for_candidate_specific_formal_queue") != 14
    ):
        raise ValueError("future Aether preflight survivor set changed")
    records = []
    witness_tilts: Counter[str] = Counter()
    tilt_strata: Counter[str] = Counter()
    for ordinal, source in enumerate(targets):
        source_body = {key: value for key, value in source.items() if key != "content_sha256"}
        if (
            source.get("content_sha256") != _sha(source_body)
            or source.get("preflight_pass_scope")
            != "candidate_specific_Aether_formal_queue_only; no global energy or theory pass"
            or source.get("next_required_formal_stage")
            != "candidate_specific_Aether_ADM_twist_constraint_and_global_energy_campaign"
            or source.get("expensive_candidate_specific_formal_run") is not False
            or source.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("future Aether source preflight record changed")
        specialization = _specialize(source["parameters"])
        if specialization != source["exact_specialization"]:
            raise ValueError("future Aether exact specialization does not replay")
        witness = specialization.get("finite_negative_twist_witness")
        if (
            specialization.get("adm_aligned_regular") is not True
            or specialization.get("principal_and_linear_mode_domain_pass") is not True
            or specialization.get("restricted_positive_energy_coupling_domain") is not True
            or specialization.get("static_twist_domain_status")
            != "finite_negative_local_density_witness"
            or not isinstance(witness, dict)
            or witness.get("local_hamiltonian_density_negative") is not True
            or witness.get("full_gravitational_constraint_embedding_proven") is not False
            or witness.get("candidate_rejection_authorized_by_this_witness_alone") is not False
        ):
            raise ValueError("future Aether candidate formal specialization changed")
        witness_tilts[witness["tilt_squared_y"]] += 1
        global_tilt = specialization["global_unit_tilt_legendre_strata"]
        tilt_key = (
            "globally_noncharacteristic_for_finite_unit_tilt"
            if global_tilt["globally_noncharacteristic_for_finite_unit_tilt"]
            else "finite_characteristic_foliation_present"
        )
        tilt_strata[tilt_key] += 1
        gates = {
            "source_preflight_candidate_action_lineage": {"status": "pass"},
            "arbitrary_background_covariant_Noether_identity": {
                "status": "pass",
                "evidence_sha256": noether_control["evidence_sha256"],
                "scope": noether_control["scope"],
            },
            "generic_spatial_diffeomorphism_and_DH_covariance": {
                "status": "pass",
                "evidence_sha256": adapter_evidence["einstein_aether_generic_dh_covariance"][
                    "evidence_sha256"
                ],
            },
            "regular_patch_HH_deformation_and_five_mode_count": {
                "status": "pass",
                "scope": "regular positive-unit-branch patch; boundary completion remains separate",
            },
            "aligned_principal_and_linear_modes": {"status": "pass"},
            "candidate_specific_linear_energy": {"status": "pass"},
            "global_unit_tilt_legendre_strata": {
                "status": "pass" if tilt_key.startswith("globally") else "conditional",
                "finding": tilt_key,
                "finite_characteristic_tilt_squared": global_tilt[
                    "finite_characteristic_tilt_squared"
                ],
            },
            "restricted_nonlinear_positive_energy_theorem": {
                "status": "not_applicable_to_generic_twisting_sector",
                "reason": "the reviewed theorem requires hypersurface-orthogonal Aether on a maximal asymptotically-flat slice",
            },
            "static_unit_reduced_pure_twist_local_energy": {
                "status": "blocked",
                "exact_hamiltonian": twist_hamiltonian,
                "witness": witness,
                "reason": "the exact negative local density is not yet embedded in the gravitational constraint surface and completed boundary energy",
            },
            "maxwell_unit_aether_control": {
                "status": "not_applicable_different_action_subclass",
                "reason": "the control requires c3=-c1 and c2=c4=0; every target has c4=1/32",
            },
            "generic_twisting_constraint_reduced_hamiltonian": {"status": "blocked"},
            "global_positive_energy": {"status": "blocked"},
            "observational_data_seal": {"status": "pass"},
        }
        provenance_body = {
            "candidate_id": source["candidate_id"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "source_preflight_record_sha256": source["content_sha256"],
            "compilation_receipt_sha256": source["compilation_receipt_sha256"],
            "parameter_cell_lineage_sha256": source["parameter_cell_lineage_sha256"],
            "exact_specialization_sha256": specialization["content_sha256"],
            "reviewed_formal_evidence_root_sha256": formal_evidence_root,
            "data_eligibility": ELIGIBILITY,
        }
        record_body = {
            "ordinal": ordinal,
            "candidate_id": source["candidate_id"],
            "family_id": TARGET_FAMILY,
            "parameter_cell_id": source["parameter_cell_id"],
            "parameter_cell_lineage_sha256": source["parameter_cell_lineage_sha256"],
            "typed_action_ir_sha256": source["typed_action_ir_sha256"],
            "action_density_equivalence_sha256": source["action_density_equivalence_sha256"],
            "compilation_receipt_sha256": source["compilation_receipt_sha256"],
            "source_preflight_record_sha256": source["content_sha256"],
            "parameters": source["parameters"],
            "exact_specialization": specialization,
            "gate_ledger": gates,
            "decision": "blocked",
            "first_blocker": BLOCKER,
            "formal_pass": False,
            "full_formal_completion_claimed": False,
            "candidate_rejection_authorized": False,
            "solar_bundle_generated": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {
                **provenance_body,
                "binding_sha256": _sha(provenance_body),
            },
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if witness_tilts != Counter({"1": 8, "2": 4, "8": 2}) or tilt_strata != Counter(
        {
            "finite_characteristic_foliation_present": 13,
            "globally_noncharacteristic_for_finite_unit_tilt": 1,
        }
    ):
        raise ValueError("future Aether exact witness partition changed")
    record_root = _sha(
        [
            [
                item["candidate_id"],
                item["typed_action_ir_sha256"],
                item["source_preflight_record_sha256"],
                item["content_sha256"],
            ]
            for item in records
        ]
    )
    provenance_body = {
        "source_preflight_content_sha256": preflight["content_sha256"],
        "source_preflight_candidate_registry_root_sha256": preflight[
            "candidate_record_registry_root_sha256"
        ],
        "formal_report_file_sha256": config["formal_report"]["file_sha256"],
        "prior_twist_audit_content_sha256": prior_twist["content_sha256"],
        "reviewed_adapter_evidence_root_sha256": adapter_root,
        "arbitrary_background_noether_control_sha256": noether_control["content_sha256"],
        "reviewed_formal_evidence_root_sha256": formal_evidence_root,
        "candidate_record_registry_root_sha256": record_root,
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "immutable_config_sha256": _sha(config),
        "source_preflight_binding": config["source_preflight_artifact"],
        "input_preflight_pass_count": len(targets),
        "candidate_count": len(records),
        "decision_counts": {"blocked": 14},
        "first_blocker_counts": {BLOCKER: 14},
        "formal_pass_count": 0,
        "exact_negative_local_twist_witness_count": 14,
        "candidate_rejection_authorized_count": 0,
        "witness_tilt_squared_counts": dict(sorted(witness_tilts.items())),
        "global_tilt_strata_counts": dict(sorted(tilt_strata.items())),
        "reviewed_adapter_replay_count": len(adapter_evidence),
        "reviewed_bound_cadabra_control_count": 1,
        "reviewed_adapter_applicability_counts": dict(
            sorted(Counter(EXPECTED_APPLICABILITY.values()).items())
        ),
        "reviewed_adapter_evidence": adapter_evidence,
        "reviewed_adapter_evidence_root_sha256": adapter_root,
        "reviewed_arbitrary_background_noether_control": noether_control,
        "reviewed_formal_evidence_root_sha256": formal_evidence_root,
        "candidate_record_registry_root_sha256": record_root,
        "candidate_records": records,
        "candidate_specific_formal_followup_completed": True,
        "full_candidate_specific_formal_completion_claimed": False,
        "automatic_downstream_enqueue_performed": False,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "All fourteen future Aether preflight survivors retain healthy exact aligned "
            "principal and linear-energy coefficients and replay the reviewed regular-patch "
            "constraint controls. Every action also has an exact finite negative static "
            "pure-twist local-density witness. Because no reviewed adapter embeds that "
            "witness in the full gravitational constraint surface and completed boundary "
            "energy, all fourteen are blocked rather than rejected or passed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_aether_candidate_formal_followup(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    """Atomically publish once and refuse divergent replacement."""
    artifact = build_future_aether_candidate_formal_followup(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != artifact:
            raise ValueError("refusing to replace divergent future Aether formal artifact")
        return artifact
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write((_canonical(artifact) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return artifact
