from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FORMAL_GATES = (
    "field_contract",
    "static_dictionary_derivation",
    "higher_jet_regularity",
    "nonlinear_x_regularity",
    "adm_decomposition",
    "legendre_map",
    "generated_dirac_closure",
    "parameter_domain",
    "covariant_variation",
    "covariant_identity",
    "adm_dirac",
    "hamiltonian_stability",
    "principal_symbol",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolved_artifact_path(health_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (health_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def audit_galaxy_observable_protocol(
    protocol_path: str | Path, evidence_policy_path: str | Path
) -> dict[str, Any]:
    """Validate the sealed observable-to-observable galaxy discovery contract.

    A pass validates only the protocol. It does not open data: an exact candidate
    action and a hash-committed dataset manifest must still pass separate audits.
    """

    protocol_path = Path(protocol_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-galaxy-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load galaxy observable protocol: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-galaxy-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load observational evidence policy: {error}"],
            "observational_dataset_opened": False,
        }

    if protocol.get("schema_version") != "sigma-galaxy-observable-protocol-1.0":
        errors.append("unsupported galaxy observable protocol schema")
    if protocol.get("status") != "sealed":
        errors.append("galaxy observable protocol must remain sealed")
    if protocol.get("data_opened") is not False:
        errors.append("galaxy observable protocol cannot claim opened data")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("unsupported observational evidence policy schema")
    if policy.get("status") != "frozen":
        errors.append("observational evidence policy is not frozen")

    discovery = protocol.get("discovery_channel", {})
    inputs = discovery.get("inputs", [])
    target = discovery.get("prediction_target", "")
    allowed_formula_inputs = discovery.get("allowed_formula_inputs", [])
    forbidden_formula_inputs = set(discovery.get("forbidden_formula_inputs", []))
    if not isinstance(inputs, list) or len(inputs) < 4:
        errors.append("discovery channel lacks the required measured input classes")
    if not isinstance(target, str) or not all(
        token in target.casefold() for token in ("held-out", "doppler")
    ):
        errors.append("prediction target must be held-out Doppler/spectral kinematics")
    contaminated_allowed_text = " ".join(
        [*(str(item) for item in inputs), *(str(item) for item in allowed_formula_inputs), target]
    ).casefold()
    for prohibited in (
        "halo mass",
        "halo concentration",
        "nfw",
        "redshift-derived distance",
        "lensing-derived mass",
        "per-galaxy acceleration",
    ):
        if prohibited in contaminated_allowed_text:
            errors.append(f"allowed discovery channel contains prohibited input: {prohibited}")
    required_forbidden_inputs = {
        "dark-matter halo mass, concentration, radius, profile, or abundance-matching label",
        "mass discrepancy inferred from the target rotation curve",
        "redshift-derived distance or physical size",
        "lensing-derived mass, convergence, or acceleration map",
        "per-galaxy acceleration scale or gravitational coupling",
    }
    missing_forbidden_inputs = sorted(required_forbidden_inputs - forbidden_formula_inputs)
    if missing_forbidden_inputs:
        errors.append(
            "missing forbidden discovery inputs: " + ", ".join(missing_forbidden_inputs)
        )
    conversion_rule = str(discovery.get("baryonic_conversion_rule", "")).casefold()
    if "not direct observations" not in conversion_rule or "each galaxy" not in conversion_rule:
        errors.append("baryonic conversion nuisances are not constrained against target fitting")

    split = protocol.get("split_contract", {})
    if split.get("unit") != "whole galaxy":
        errors.append("train/validation/test splitting must occur by whole galaxy")
    if split.get("group_leakage_forbidden") is not True:
        errors.append("galaxy group leakage is not forbidden")
    roles = set(split.get("roles", []))
    if "untouched target-blind test galaxies" not in roles:
        errors.append("target-blind whole-galaxy test role is missing")
    sealed_rule = str(split.get("sealed_test_rule", "")).casefold()
    if not all(token in sealed_rule for token in ("formula hash", "stopping rule", "inaccessible")):
        errors.append("sealed test rule does not freeze formula and stopping decisions")

    scoring = protocol.get("scoring_contract", {})
    if scoring.get("object_specific_gravity_parameters") != 0:
        errors.append("object-specific gravitational parameters must equal zero")
    selection_rule = str(scoring.get("selection_rule", "")).casefold()
    if not all(token in selection_rule for token in ("validation", "no test", "lensing")):
        errors.append("selection rule does not isolate test and lensing channels")
    if "covariance" not in str(scoring.get("fit_term", "")).casefold():
        errors.append("fit score must carry measurement and calibration covariance")
    if not scoring.get("required_baselines"):
        errors.append("baryons-only and empirical baselines are missing")

    lensing = protocol.get("independent_lensing_falsification", {})
    if lensing.get("status") != "sealed_until_after_kinematic_formula_freeze":
        errors.append("lensing channel is not sealed until after formula freeze")
    if lensing.get("formula_selection_use") != "prohibited":
        errors.append("lensing is not prohibited from formula selection")
    same_law = str(lensing.get("same_law_rule", "")).casefold()
    if not all(token in same_law for token in ("frozen covariant action", "without any lensing-only")):
        errors.append("lensing does not require the same frozen law without private parameters")
    lensing_forbidden = " ".join(lensing.get("forbidden_targets", [])).casefold()
    if "dark-matter map" not in lensing_forbidden or "gr-derived" not in lensing_forbidden:
        errors.append("lensing-derived invisible-mass targets are not fully prohibited")

    prohibited_truth = {
        str(item).casefold() for item in protocol.get("prohibited_truth_or_rescue", [])
    }
    for required in (
        "dark matter",
        "halo mass",
        "halo concentration",
        "redshift-derived distance",
        "supernova distance modulus",
    ):
        if required not in prohibited_truth:
            errors.append(f"missing prohibited truth/rescue quantity: {required}")
    policy_uses = set(policy.get("unobserved_components", {}).get("prohibited_uses", []))
    if not {"target labels", "formula selection", "object-specific halo fitting"} <= policy_uses:
        errors.append("evidence policy no longer prohibits inferred-component target leakage")
    if policy.get("supernovae", {}).get("default_status") != "excluded":
        errors.append("supernova distances are not excluded by the evidence policy")
    if "treating redshift as a distance" not in policy.get("redshift", {}).get(
        "not_allowed_by_default", []
    ):
        errors.append("redshift-distance inference is not excluded by default")

    opening_requirements = protocol.get("opening_requirements", [])
    if not isinstance(opening_requirements, list) or len(opening_requirements) < 5:
        errors.append("dataset-opening requirements are incomplete")
    passed = not errors
    return {
        "schema_version": "sigma-galaxy-observable-audit-1.0",
        "status": "pass" if passed else "fail",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "policy": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "errors": errors,
        "discovery_target": target,
        "split_unit": split.get("unit"),
        "object_specific_gravity_parameters": scoring.get(
            "object_specific_gravity_parameters"
        ),
        "lensing_formula_selection_use": lensing.get("formula_selection_use"),
        "redshift_distance_allowed_by_default": False,
        "supernova_default_status": policy.get("supernovae", {}).get("default_status"),
        "observational_dataset_opened": False,
        "formula_search_authorized": False,
        "next_required_binding": (
            "an eligible exact action hash and an independently audited, hash-committed dataset manifest"
        ),
    }


def audit_theory_observation_eligibility(
    health_report_path: str | Path,
    evidence_policy_path: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Fail closed before Solar references or any candidate dataset can be opened."""

    if mode not in {"known_answer_reference", "candidate_data"}:
        raise ValueError("mode must be known_answer_reference or candidate_data")
    health_path = Path(health_report_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        health = _load_json(health_path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-observation-eligibility-1.0",
            "status": "ineligible",
            "mode": mode,
            "errors": [f"cannot load action-health report: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-observation-eligibility-1.0",
            "status": "ineligible",
            "mode": mode,
            "errors": [f"cannot load observational evidence policy: {error}"],
            "observational_dataset_opened": False,
        }

    if health.get("schema_version") != "sigma-action-health-1.0":
        errors.append("unsupported action-health schema")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("unsupported observational evidence policy schema")
    if policy.get("status") != "frozen":
        errors.append("observational evidence policy is not frozen")
    gates = health.get("gates", {})
    missing_gates = [name for name in REQUIRED_FORMAL_GATES if name not in gates]
    failed_gates = [
        name for name in REQUIRED_FORMAL_GATES if gates.get(name, {}).get("status") != "pass"
    ]
    if missing_gates:
        errors.append("missing formal gates: " + ", ".join(missing_gates))
    if failed_gates:
        errors.append("formal gates not passed: " + ", ".join(failed_gates))

    action_path_value = health.get("action_ir")
    action_ir: dict[str, Any] = {}
    if not action_path_value:
        errors.append("action-health report does not name its action IR")
    else:
        try:
            action_ir = _load_json(_resolved_artifact_path(health_path, action_path_value))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load bound action IR: {error}")
    action_hash = action_ir.get("content_sha256")
    if not action_ir.get("valid"):
        errors.append("bound covariant action IR is invalid")
    if health.get("input_action_sha256") != action_hash:
        errors.append("action-health report belongs to a different action hash")

    artifact_records: dict[str, Any] = {}
    for key, expected_schema, hash_field in (
        ("generated_q_operator_ir", "sigma-q-operator-ir-1.0", "input_action_sha256"),
        ("generated_q_variation_ir", "sigma-q-variation-ir-1.0", "input_action_sha256"),
        ("generated_x_operator_ir", "sigma-x-operator-ir-1.0", "input_action_sha256"),
        ("generated_principal_ir", "sigma-physical-principal-ir-1.0", "input_action_sha256"),
        (
            "generated_hamiltonian_ir",
            "sigma-physical-hamiltonian-ir-1.0",
            "input_action_sha256",
        ),
    ):
        record = health.get(key, {})
        if not record.get("path"):
            errors.append(f"action-health report does not name {key}")
            continue
        try:
            artifact = _load_json(_resolved_artifact_path(health_path, record["path"]))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load {key}: {error}")
            continue
        artifact_records[key] = artifact
        if artifact.get("schema_version") != expected_schema:
            errors.append(f"unsupported {key} schema")
        if artifact.get("status") != "pass":
            errors.append(f"{key} has not passed")
        if artifact.get(hash_field) != action_hash:
            errors.append(f"{key} belongs to a different action hash")
        if record.get("content_sha256") != artifact.get("content_sha256"):
            errors.append(f"{key} content hash differs from the health report")

    source_role = action_ir.get("canonical", {}).get("source_role")
    family = health.get("family")
    if mode == "known_answer_reference":
        if source_role != "known_answer_control":
            errors.append("Solar reference mode requires a known-answer control action")
        if family != "einstein_hilbert":
            errors.append("Solar reference mode requires the Einstein-Hilbert family")
        if not health.get("promotion_allowed") or health.get("status") != "pass":
            errors.append("Einstein-Hilbert control health packet is not a full pass")
    else:
        if source_role != "candidate":
            errors.append("candidate-data mode requires a generated candidate action")
        if not health.get("promotion_allowed") or health.get("status") != "pass":
            errors.append("candidate has not passed every formal promotion gate")
        if health.get("discovery_blockers"):
            errors.append("candidate still has discovery blockers")

    eligible = not errors
    return {
        "schema_version": "sigma-observation-eligibility-1.0",
        "status": "eligible" if eligible else "ineligible",
        "mode": mode,
        "health_report": str(health_path),
        "input_action_sha256": action_hash,
        "source_role": source_role,
        "family": family,
        "formal_gate_statuses": {
            name: gates.get(name, {}).get("status") for name in REQUIRED_FORMAL_GATES
        },
        "policy": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "policy_status": policy.get("status"),
        "errors": errors,
        "reference_controls_allowed": eligible and mode == "known_answer_reference",
        "candidate_dataset_manifest_may_be_audited": eligible and mode == "candidate_data",
        "observational_dataset_opened": False,
        "dataset_opening_blocker": (
            "an independently audited dataset manifest is still required"
            if eligible and mode == "candidate_data"
            else "theory-side eligibility has not passed"
            if not eligible
            else "known-answer reference mode does not authorize opening candidate datasets"
        ),
        "prohibited_targets_preserved": policy.get("unobserved_components", {}).get(
            "prohibited_uses", []
        ),
        "supernova_default_status": policy.get("supernovae", {}).get("default_status"),
        "redshift_distance_allowed_by_default": False,
    }


def write_observation_eligibility(
    eligibility: dict[str, Any], output: str | Path
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
