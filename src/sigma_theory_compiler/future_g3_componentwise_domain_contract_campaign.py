"""Reviewed fail-closed domain contracts for blocked future cubic-G3 actions."""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from .cubic_bssn_domain import generic_cubic_scalar_effective_metric_control
from .g3_full_lapse_dirac_operator_audit import _derive_full_delta
from .g3_seed_weak_cell_formal_audit import _candidate_certificate
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-g3-componentwise-domain-contract-config-1.0"
ARTIFACT_SCHEMA = "sigma-future-g3-componentwise-domain-contract-campaign-1.0"
FAMILY_ID = "CUBIC_HORNDESKI_G3_WEAK_CELL"
FIRST_BLOCKER = "candidate_bound_nonzero_componentwise_normalized_local_jet_box_values"
REQUIRED_DOMAIN_FIELDS = [
    "normalization_binding",
    "frame_binding",
    "phi_interval",
    "anchor_X_interval",
    "normal_gradient_padding",
    "spatial_gradient_component_abs",
    "symmetric_hessian_component_abs",
    "riemann_tetrad_component_abs",
    "lapse_interval",
    "lapse_log_spatial_gradient_component_abs",
    "extrinsic_curvature_component_abs",
    "slicing_BSSN_sigma_m",
]
_G3 = re.compile(r"\((\d+/\d+)\)\*X_phi")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(
    root: Path, binding: dict[str, Any], label: str, *, content: bool = False
) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
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


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "adapter_source",
            "bindings",
            "targets",
            "center_calibration",
            "required_domain_contract",
            "output_path",
            "data_eligibility",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("future G3 domain-contract config is invalid")
    if set(config.get("bindings", {})) != {
        "preflight",
        "componentwise_method_control",
        "lapse_dirac_method_control",
        "effective_metric_source",
        "specializer_source",
        "lapse_source",
    }:
        raise ValueError("future G3 domain-contract bindings changed")
    if not isinstance(config.get("output_path"), str) or not config["output_path"].startswith(
        "runs/engine/"
    ):
        raise ValueError("future G3 domain-contract output path changed")
    center = config.get("center_calibration", {})
    if center != {
        "X_phi": "1/2",
        "hessian_covariant": "zero",
        "G4": "1/2",
        "G4_X": "0",
        "BSSN_m": "1",
        "BSSN_sigma": "1",
    }:
        raise ValueError("future G3 center calibration changed")
    contract = config.get("required_domain_contract", {})
    if (
        contract.get("contract_kind")
        != "candidate_bound_pointwise_local_jet_box_not_evolution_invariant"
        or contract.get("required_fields") != REQUIRED_DOMAIN_FIELDS
        or contract.get("direction_sphere") != "all xi_i with delta^ij*xi_i*xi_j=1; no sampling"
        or contract.get("registered_values") != {field: None for field in REQUIRED_DOMAIN_FIELDS}
    ):
        raise ValueError("future G3 required domain contract changed")


def _record_by_target(preflight: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in preflight.get("candidate_records", [])
        if item.get("candidate_id") == target["candidate_id"]
    ]
    if len(matches) != 1:
        raise ValueError("future G3 target is not unique")
    record = matches[0]
    expected = {
        "ordinal": target["ordinal"],
        "candidate_id": target["candidate_id"],
        "family_id": FAMILY_ID,
        "parameter_cell_id": target["parameter_cell_id"],
        "parameter_cell_lineage_sha256": target["parameter_cell_lineage_sha256"],
        "compilation_receipt_sha256": target["compilation_receipt_sha256"],
        "typed_action_ir_sha256": target["action_sha256"],
        "action_density_equivalence_sha256": target["action_density_equivalence_sha256"],
        "content_sha256": target["preflight_record_content_sha256"],
        "decision": "blocked",
        "first_blocker": (
            "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing"
        ),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("future G3 target binding changed")
    match = _G3.fullmatch(record.get("parameters", {}).get("G3", ""))
    if (
        match is None
        or match.group(1) != target["beta"]
        or record["parameters"].get("G2") != "X_phi"
        or record["parameters"].get("jet_domain")
        != f"dimensionless derivative ratios<={target['beta']}"
        or record.get("domain_contract")
        != "weak derivative cell only; common-cone proof remains unresolved"
        or record.get("observational_data_opened") is not False
        or record.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("future G3 action or qualitative domain changed")
    return record


def _all_direction_center_certificate(record: dict[str, Any]) -> dict[str, Any]:
    center = record["exact_specialization"]["center_principal_calibration"]
    p00 = Fraction(center["effective_P00"])
    spatial = Fraction(center["effective_spatial_eigenvalue"])
    speed = Fraction(center["scalar_speed_squared"])
    slicing = Fraction(center["slicing_speed_squared"])
    gap = Fraction(center["scalar_slicing_cone_gap_squared"])
    if (
        p00 >= 0
        or spatial <= 0
        or speed != spatial / (-p00)
        or slicing != 2
        or gap != slicing - speed
        or center.get("status") != "pass_at_center_only"
    ):
        raise ValueError("future G3 exact center cone identities changed")
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["typed_action_ir_sha256"],
        "point": {
            "X_phi": "1/2",
            "hessian_covariant": "zero",
            "BSSN_m": "1",
            "BSSN_sigma": "1",
        },
        "effective_metric": {
            "P00": str(p00),
            "isotropic_spatial_eigenvalue": str(spatial),
            "time_covector_margin": str(-p00),
            "all_unit_spatial_direction_quadratic_form_lower": str(spatial),
        },
        "scalar_speed_squared": str(speed),
        "slicing_speed_squared": str(slicing),
        "slicing_minus_scalar_speed_squared_gap": str(gap),
        "direction_method": (
            "isotropic spatial block gives the same exact quadratic form for every unit "
            "spatial covector; no sampling"
        ),
        "status": "pass_all_directions_at_single_center_only",
        "scope": (
            "exact point certificate only; zero-width center is not a nonzero local jet box "
            "and supplies no uniform neighborhood or evolution invariant"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _reviewed_domain_contract(record: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    registered = dict(template["registered_values"])
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["typed_action_ir_sha256"],
        "contract_kind": template["contract_kind"],
        "normalization_required_by_adapter": (
            "Lambda_phi=1 and M_Pl=1 in explicitly bound dimensionless adapter variables"
        ),
        "frame_required_by_adapter": (
            "local orthonormal tetrad with e0 equal to the chosen BSSN foliation normal"
        ),
        "direction_sphere": template["direction_sphere"],
        "registered_values": registered,
        "unfilled_fields": REQUIRED_DOMAIN_FIELDS,
        "unfilled_field_count": len(REQUIRED_DOMAIN_FIELDS),
        "qualitative_source_label": record["parameters"]["jet_domain"],
        "qualitative_label_classification": (
            "insufficient_no_coordinate_map_norm_definition_frame_normalization_or_anchor"
        ),
        "nonempty_box_registered": False,
        "interval_adapter_invocation_authorized": False,
        "status": "reviewed_requirement_contract_registered_values_missing",
        "scope": (
            "registers the exact fields needed for a future candidate-specific interval run; "
            "it does not assign those fields or strengthen the compiled candidate domain"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _interval_attempt(contract: dict[str, Any]) -> dict[str, Any]:
    if contract["unfilled_field_count"] != len(REQUIRED_DOMAIN_FIELDS):
        raise ValueError("future G3 interval attempt received an unexpected filled field")
    body = {
        "candidate_id": contract["candidate_id"],
        "action_sha256": contract["action_sha256"],
        "required_domain_contract_sha256": contract["content_sha256"],
        "certify_cubic_bssn_domain_invoked": False,
        "reason_not_invoked": (
            "numeric componentwise input contract is incomplete; inventing bounds would be a "
            "new assumption rather than a reviewed candidate-domain specialization"
        ),
        "uniform_effective_metric_interval": None,
        "uniform_common_time_covector_margin": None,
        "uniform_spatial_eigenvalue_lower": None,
        "uniform_characteristic_discriminant_lower": None,
        "uniform_slicing_cone_separation": None,
        "all_direction_uniform_certificate": None,
        "status": "blocked_before_interval_adapter_invocation",
    }
    return {**body, "content_sha256": _sha(body)}


def _lapse_scope(record: dict[str, Any], beta: Fraction) -> dict[str, Any]:
    derivation = _derive_full_delta(beta)
    if (
        derivation["full_Delta_N"]["beta_specialization"] != str(beta)
        or set(derivation["exact_residuals"].values()) != {"0"}
        or derivation["full_Delta_N"]["differential_order"] != 0
        or derivation["full_Delta_N"]["operator_type"] != "real_multiplication_operator"
    ):
        raise ValueError("future G3 full lapse derivation changed")
    center_delta = 1 + Fraction(3, 2) * beta**2
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["typed_action_ir_sha256"],
        "beta": str(beta),
        "full_Delta_N_derivation": derivation,
        "single_center": {
            "N": "1",
            "K": "0",
            "Delta_N": str(center_delta),
            "strictly_positive": center_delta > 0,
            "status": "pass_at_center_only",
        },
        "uniform_coercivity_lower_bound": None,
        "inverse_norm_upper": None,
        "periodic_function_space_domain_registered": False,
        "periodic_distributed_Dirac": "blocked",
        "asymptotically_flat_Dirac": "blocked",
        "global_energy": "blocked",
        "first_missing_lapse_premise": (
            "candidate_bound_N_K_and_jet_bounds_plus_function_space_boundary_domain"
        ),
        "scope": (
            "the complete algebraic multiplication operator is derived for the exact beta; "
            "center positivity is not a uniform coercivity or Dirac certificate"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _nontransfer_control(
    record: dict[str, Any], beta: Fraction, method: dict[str, Any], lapse: dict[str, Any]
) -> dict[str, Any]:
    method_record = method["candidate_records"][0]
    lapse_record = lapse["candidate_records"][0]
    control_beta = Fraction(1, 100)
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["typed_action_ir_sha256"],
        "candidate_beta": str(beta),
        "method_control_beta": str(control_beta),
        "candidate_beta_is_smaller": beta < control_beta,
        "method_control_action_sha256": method_record["action_sha256"],
        "lapse_control_action_sha256": lapse_record["action_sha256"],
        "action_identity_match": (
            record["typed_action_ir_sha256"] == method_record["action_sha256"]
        ),
        "componentwise_domain_identity_match": False,
        "monotonicity_transfer_theorem_registered": False,
        "prior_interval_or_coercivity_bound_reused": False,
        "decision": "pass_negative_control",
        "rule": (
            "a smaller beta or shared family label cannot transfer another action's local box, "
            "interval certificate, lapse coercivity bound, or Dirac result"
        ),
    }
    if body["action_identity_match"] or body["prior_interval_or_coercivity_bound_reused"]:
        raise ValueError("future G3 non-transfer control failed")
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_componentwise_domain_contract_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Build exact center/lapse evidence and keep missing componentwise domains blocked."""

    _validate_config(config)
    root = Path(project_root).resolve()
    adapter_path = (root / config["adapter_source"]["path"]).resolve()
    if _file_sha(adapter_path) != config["adapter_source"]["file_sha256"]:
        raise ValueError("future G3 campaign source hash mismatch")
    bindings = config["bindings"]
    preflight = _load_bound(root, bindings["preflight"], "preflight", content=True)
    method = _load_bound(
        root,
        bindings["componentwise_method_control"],
        "componentwise method control",
        content=True,
    )
    lapse_control = _load_bound(
        root,
        bindings["lapse_dirac_method_control"],
        "lapse/Dirac method control",
        content=True,
    )
    for label in ("effective_metric_source", "specializer_source", "lapse_source"):
        path = (root / bindings[label]["path"]).resolve()
        if _file_sha(path) != bindings[label]["file_sha256"]:
            raise ValueError(f"{label} file hash mismatch")
    if (
        preflight.get("family_counts", {}).get(FAMILY_ID) != 3
        or preflight.get("first_blocker_counts", {}).get(
            "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing"
        )
        != 3
        or method.get("uniform_principal_common_cone_pass_count") != 1
        or lapse_control.get("full_lapse_dirac_pass_count") != 1
    ):
        raise ValueError("future G3 method-control boundary changed")
    passed, generic_metric = generic_cubic_scalar_effective_metric_control()
    first_g3 = next(
        item for item in preflight["candidate_records"] if item.get("family_id") == FAMILY_ID
    )
    metric_evidence = next(
        item
        for item in first_g3["reviewed_adapter_evidence"]
        if item["formal_check_id"] == "generic_cubic_horndeski_scalar_effective_metric"
    )
    if not passed or _sha(generic_metric) != metric_evidence["evidence_sha256"]:
        raise ValueError("future G3 effective-metric adapter replay mismatch")
    records = []
    for target in config["targets"]:
        record = _record_by_target(preflight, target)
        replayed = _candidate_certificate(
            {
                "parameters": record["parameters"],
                "g3_linear_x_coefficient": target["beta"],
            },
            config["center_calibration"],
        )
        if replayed != record["exact_specialization"]:
            raise ValueError("future G3 exact specialization replay mismatch")
        beta = Fraction(target["beta"])
        center = _all_direction_center_certificate(record)
        domain = _reviewed_domain_contract(record, config["required_domain_contract"])
        interval = _interval_attempt(domain)
        lapse = _lapse_scope(record, beta)
        nontransfer = _nontransfer_control(record, beta, method, lapse_control)
        gates = {
            "receipt_action_and_lineage_binding": "pass",
            "generic_effective_metric_identity_replay": "pass",
            "all_direction_single_center_principal_and_cone": "pass_at_center_only",
            "reviewed_componentwise_domain_requirement_contract": "pass",
            "nonzero_componentwise_domain_values": "blocked",
            "uniform_local_jet_box_principal_common_cone": "blocked",
            "exact_full_Delta_N_derivation": "pass",
            "uniform_Delta_N_coercivity": "blocked",
            "periodic_distributed_Dirac": "blocked",
            "asymptotically_flat_Dirac_and_global_energy": "blocked",
            "smaller_beta_or_family_label_transfer": "rejected_as_inference",
            "observational_data_seal": "pass",
        }
        provenance_body = {
            "preflight_content_sha256": preflight["content_sha256"],
            "preflight_record_content_sha256": record["content_sha256"],
            "candidate_id": record["candidate_id"],
            "action_sha256": record["typed_action_ir_sha256"],
            "parameter_cell_lineage_sha256": record["parameter_cell_lineage_sha256"],
            "center_certificate_sha256": center["content_sha256"],
            "domain_contract_sha256": domain["content_sha256"],
            "interval_attempt_sha256": interval["content_sha256"],
            "lapse_scope_sha256": lapse["content_sha256"],
            "nontransfer_control_sha256": nontransfer["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "ordinal": record["ordinal"],
            "candidate_id": record["candidate_id"],
            "action_sha256": record["typed_action_ir_sha256"],
            "parameter_cell_id": record["parameter_cell_id"],
            "beta": str(beta),
            "qualitative_jet_domain": record["parameters"]["jet_domain"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "all_direction_center_certificate": center,
            "reviewed_domain_contract": domain,
            "uniform_interval_attempt": interval,
            "lapse_and_dirac_scope": lapse,
            "nontransfer_control": nontransfer,
            "gate_ledger": gates,
            "necessary_condition_rejection_found": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {
                **provenance_body,
                "binding_sha256": _sha(provenance_body),
            },
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if len(records) != 3:
        raise ValueError("future G3 target count changed")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "candidate_count": len(records),
        "decision_counts": {"blocked": len(records)},
        "candidate_records": records,
        "all_direction_single_center_pass_count": len(records),
        "nonzero_componentwise_box_pass_count": 0,
        "uniform_principal_common_cone_pass_count": 0,
        "full_Delta_N_derivation_pass_count": len(records),
        "uniform_Delta_N_coercivity_pass_count": 0,
        "periodic_distributed_Dirac_pass_count": 0,
        "asymptotically_flat_Dirac_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: len(records)},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "All three actions have exact all-direction principal/cone checks at the single "
            "off-shell center and exact action-specific full lapse multiplication operators. "
            "Their scalar qualitative derivative labels do not define nonzero componentwise "
            "boxes, so no uniform interval, coercivity, Dirac, or full-formal pass is claimed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_componentwise_domain_contract_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_componentwise_domain_contract_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
