from __future__ import annotations

import hashlib
import importlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g3-seed-weak-cell-formal-audit-1.0"
TARGET_FAMILY = "CUBIC_HORNDESKI_G3_WEAK_CELL"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any], *, content: bool = False) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {descriptor['path']}")
    if content:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor["content_sha256"]:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _resolve(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator:
        raise ValueError("formal adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"formal adapter is not callable: {entrypoint}")
    return callback


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("family_id") != TARGET_FAMILY:
        raise ValueError("G3 target identity or family mismatch")
    if record["typed_action_ir"].get("content_sha256") != target["action_sha256"]:
        raise ValueError("G3 target action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G3 predecessor provenance mismatch")
    if record["typed_action_ir"].get("parameters") != target["parameters"]:
        raise ValueError("G3 parameter binding mismatch")
    atoms = [item["atom"] for item in record["typed_action_ir"]["operators"]]
    if atoms != ["EH_R", "G2_PHI_X", "G3_PHI_X_BOX_PHI"]:
        raise ValueError("G3 target operator basis mismatch")


def _candidate_certificate(target: dict[str, Any], center: dict[str, str]) -> dict[str, Any]:
    match = re.fullmatch(r"\((\d+/\d+)\)\*X_phi", target["parameters"]["G3"])
    if match is None:
        raise ValueError("unsupported G3 seed function")
    beta = Fraction(match.group(1))
    if beta != Fraction(target["g3_linear_x_coefficient"]):
        raise ValueError("G3 coefficient binding mismatch")
    x = Fraction(center["X_phi"])
    if x <= 0 or center["hessian_covariant"] != "zero":
        raise ValueError("center calibration must have timelike gradient and zero Hessian")
    beta_squared_x_squared = beta**2 * x**2
    p00 = -(1 + 3 * beta_squared_x_squared)
    spatial = 1 - beta_squared_x_squared
    speed_squared = spatial / (-p00)
    slicing_speed_squared = 2 * Fraction(center["BSSN_sigma"])
    cone_gap = slicing_speed_squared - speed_squared
    body = {
        "method": "exact_rational_specialization_of_source_trace_reversed_cubic_effective_metric",
        "functions": {
            "canonical_G2": "X_phi",
            "canonical_G3": target["parameters"]["G3"],
            "source_normalized_G2": "0",
            "source_normalized_G3": f"-({beta})*X_phi",
            "source_G3_X": str(-beta),
            "source_G3_XX": "0",
            "G4": center["G4"],
            "G4_X": center["G4_X"],
        },
        "adm_primary": {
            "G4_minus_2X_G4_X": "1/2",
            "velocity_hessian_rank": 6,
            "velocity_hessian_nullity": 1,
            "primary_constraint": "p_V_star=0",
            "status": "pass",
        },
        "distributed_dirac": {
            "generic_secondary_chain": "pass_if_candidate_Delta_N_invertible",
            "candidate_Delta_N_operator": "not_derived_for_G3_braiding_action",
            "boundary_function_space": "not_declared",
            "status": "blocked",
        },
        "center_principal_calibration": {
            "role": "off-shell algebraic center calibration, not a uniform-cell or on-shell theorem",
            "X_phi": str(x),
            "hessian_covariant": "zero",
            "effective_P00": str(p00),
            "effective_spatial_eigenvalue": str(spatial),
            "scalar_speed_squared": str(speed_squared),
            "BSSN_m": center["BSSN_m"],
            "BSSN_sigma": center["BSSN_sigma"],
            "slicing_speed_squared": str(slicing_speed_squared),
            "scalar_slicing_cone_gap_squared": str(cone_gap),
            "two_distinct_real_scalar_roots": p00 < 0 and spatial > 0,
            "common_time_covector": p00 < 0,
            "status": "pass_at_center_only",
        },
        "declared_weak_cell_audit": {
            "input": target["parameters"]["jet_domain"],
            "componentwise_gradient_bounds": None,
            "componentwise_hessian_bounds": None,
            "curvature_bounds": None,
            "frame_and_normalization_binding": None,
            "source_numeric_threshold_for_much_less_than_one": None,
            "uniform_effective_metric_interval": None,
            "uniform_direction_sphere_cone_gap": None,
            "status": "blocked",
            "reason": "a scalar qualitative ratio label cannot instantiate the source-defined componentwise principal and cone inequalities",
        },
        "global_energy": {
            "status": "blocked",
            "reason": "G3 braiding has no candidate-specific reduced/global Hamiltonian or boundary positive-energy theorem",
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_g3_seed_weak_cell_formal_audit(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    for descriptor in config["adapter_sources"]:
        if _file_sha(root / descriptor["path"]) != descriptor["file_sha256"]:
            raise ValueError(f"adapter source hash mismatch: {descriptor['path']}")
    expected_ids = {item["id"] for item in config["formal_adapters"]}
    formal_checks = {item["name"]: item for item in formal_report["checks"] if item["name"] in expected_ids}
    if set(formal_checks) != expected_ids:
        raise ValueError("required G3 formal checks missing from bound report")
    invoked = []
    adapter_results = {}
    for descriptor in config["formal_adapters"]:
        returned = _resolve(descriptor["entrypoint"])()
        if descriptor["return_kind"] == "tuple":
            passed, evidence = returned
        elif descriptor["return_kind"] == "dict":
            evidence = returned
            passed = evidence.get("passed") is True
        else:
            raise ValueError("unsupported adapter return kind")
        bound = formal_checks[descriptor["id"]]
        if not passed or bound["status"] != "pass" or evidence != bound["evidence"]:
            raise ValueError(f"G3 formal adapter replay mismatch: {descriptor['id']}")
        invoked.append(descriptor["entrypoint"])
        adapter_results[descriptor["id"]] = {
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": _sha(evidence),
            "status": "pass",
            "scope": evidence.get("scope", bound["scope"]),
        }
    target = config["target_seed"]
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(record, target)
    certificate = _candidate_certificate(target, config["center_calibration"])
    gates = {
        "typed_action_and_provenance": {"status": "pass"},
        "covariant_G2_G3_variation_noether": {"status": "pass"},
        "adm_primary_degeneracy": {"status": "pass"},
        "candidate_specific_distributed_dirac": {
            "status": "blocked",
            "reason": "the G3 action-specific Delta_N operator and its boundary-domain invertibility are not derived",
        },
        "principal_common_cone_center": {"status": "pass_at_center_only"},
        "uniform_weak_cell_principal_common_cone": {
            "status": "blocked",
            "reason": "the seed supplies no componentwise jet box and the source supplies no universal weak-field threshold",
        },
        "global_hamiltonian_energy": {"status": "blocked"},
        "formal_prerequisite_completion": {"status": "blocked"},
    }
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
        "certificate_sha256": certificate["content_sha256"],
        "adapter_evidence_sha256": {
            key: value["evidence_sha256"] for key, value in sorted(adapter_results.items())
        },
        "data_eligibility": ELIGIBILITY,
    }
    candidate_record = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "candidate_certificate": certificate,
        "gate_ledger": gates,
        "first_missing_adm_dirac_premise": "candidate_specific_Delta_N_boundary_operator",
        "first_missing_uniform_principal_premise": "componentwise_normalized_local_jet_box",
        "necessary_condition_rejection_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "solar_bundle": {"generated": False, "status": "blocked"},
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "predecessor": config["predecessor"],
            "formal_report": config["formal_report"],
            "adapter_sources": config["adapter_sources"],
        },
        "invoked_adapter_entrypoints": invoked,
        "adapter_results": adapter_results,
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "candidate_records": [candidate_record],
        "formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": "The exact G3 seed is regular at the ADM-primary level and has a healthy algebraic zero-Hessian principal center. The current qualitative jet-domain label cannot instantiate the source-defined componentwise interval proof, so neither uniform BSSN/common-cone coverage nor complete G3 Dirac/global-energy closure is claimed.",
    }
    return {**body, "content_sha256": _sha(body)}
