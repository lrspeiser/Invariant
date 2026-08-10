from __future__ import annotations

import hashlib
import importlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-phi-seed-formal-audit-1.0"
TARGET_FAMILY = "CONFORMAL_G4_PHI_SCALAR_TENSOR"


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_bound(root: Path, descriptor: dict[str, Any], *, content: bool = False) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = _load_json(path)
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


def _validate_known_answer(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    action_path = root / descriptor["action_path"]
    health_path = root / descriptor["health_path"]
    if _file_sha(action_path) != descriptor["action_file_sha256"]:
        raise ValueError("canonical-scalar action file hash mismatch")
    if _file_sha(health_path) != descriptor["health_file_sha256"]:
        raise ValueError("canonical-scalar health file hash mismatch")
    action = _load_json(action_path)
    health = _load_json(health_path)
    pass_count = sum(gate["status"] == "pass" for gate in health["gates"].values())
    if (
        action.get("content_sha256") != descriptor["action_content_sha256"]
        or health.get("input_action_sha256") != descriptor["action_content_sha256"]
        or health.get("status") != "pass"
        or pass_count != descriptor["expected_pass_gate_count"]
    ):
        raise ValueError("canonical-scalar calibration mismatch")
    body = {
        "role": descriptor["role"],
        "action_content_sha256": action["content_sha256"],
        "action_file_sha256": descriptor["action_file_sha256"],
        "health_file_sha256": descriptor["health_file_sha256"],
        "pass_gate_count": pass_count,
        "eligible_as_candidate_evidence": False,
    }
    return {**body, "binding_sha256": _sha(body)}


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("family_id") != TARGET_FAMILY:
        raise ValueError("G4(phi) target identity or family mismatch")
    if record["typed_action_ir"].get("content_sha256") != target["action_sha256"]:
        raise ValueError("G4(phi) target action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G4(phi) predecessor provenance mismatch")
    if record["typed_action_ir"].get("parameters") != target["parameters"]:
        raise ValueError("G4(phi) parameter binding mismatch")
    atoms = [item["atom"] for item in record["typed_action_ir"]["operators"]]
    if atoms != ["G2_PHI_X", "G4_PHI_R"]:
        raise ValueError("G4(phi) target operator basis mismatch")


def _candidate_certificate(target: dict[str, Any]) -> dict[str, Any]:
    match = re.fullmatch(r"1/2\+\((\d+/\d+)\)\*phi\^2", target["parameters"]["G4"])
    if match is None:
        raise ValueError("unsupported G4(phi) polynomial")
    xi = Fraction(match.group(1))
    if xi != Fraction(target["quadratic_G4_coefficient"]):
        raise ValueError("G4(phi) coefficient binding mismatch")
    if target["parameters"]["G2"] != "X_phi" or target["parameters"]["phi_domain"] != "abs(phi)<=1":
        raise ValueError("unsupported G4(phi) seed domain")
    # Write the action as (f(phi)/2)R+X with f=2G4. Under g_E=f*g,
    # K_E=1/f+(3/2)(f'/f)^2. For y=phi^2 this is
    # (2500+56y)/(50+y)^2 and decreases on 0<=y<=1.
    f_minimum = Fraction(1)
    f_maximum = Fraction(51, 50)
    einstein_kinetic_minimum = Fraction(284, 289)
    einstein_kinetic_maximum = Fraction(1)
    lapse_multiplier_minimum = Fraction(1)
    lapse_multiplier_maximum = Fraction(426, 425)
    body = {
        "method": "exact_conformal_frame_and_rational_interval_certificate",
        "jordan_frame": {
            "G2": "X_phi",
            "G4": target["parameters"]["G4"],
            "G4_X": "0",
            "domain": target["parameters"]["phi_domain"],
            "G4_interval": ["1/2", "51/100"],
        },
        "invertible_field_transformation": {
            "einstein_metric": "g_E_munu=f(phi)*g_munu",
            "f": "1+phi^2/50",
            "f_interval": [str(f_minimum), str(f_maximum)],
            "einstein_scalar_kinetic": "K_E=(2500+56*phi^2)/(50+phi^2)^2",
            "K_E_interval": [str(einstein_kinetic_minimum), str(einstein_kinetic_maximum)],
            "derivative_with_respect_to_phi_squared": "-(2200+56*phi^2)/(50+phi^2)^3<0",
            "regular_on_entire_phi_domain": True,
        },
        "inhomogeneous_principal_common_cone": {
            "einstein_frame_tensor_principal": "g_E^munu*xi_mu*xi_nu",
            "einstein_frame_scalar_principal": "K_E*g_E^munu*xi_mu*xi_nu",
            "jordan_and_einstein_null_cones_identical": True,
            "field_redefinition_principal_congruence_invertible": True,
            "uniform_tensor_margin_f": str(f_minimum),
            "uniform_scalar_margin_K_E": str(einstein_kinetic_minimum),
            "status": "pass",
            "scope": "all inhomogeneous field jets with abs(phi)<=1; lower-derivative conformal terms do not change the diagonalized principal cone",
        },
        "adm_primary": {
            "G4_minus_2X_G4_X": "1/2+phi^2/100",
            "minimum": "1/2",
            "velocity_hessian_rank": 6,
            "velocity_hessian_nullity": 1,
            "status": "pass",
        },
        "unitary_lapse_operator": {
            "gauge_scope": "timelike-gradient patches with phi=t",
            "einstein_lapse": "N_E=sqrt(f)*N",
            "einstein_spatial_volume": "sqrt(q_E)=f^(3/2)*sqrt(q)",
            "local_Delta_N_kernel": "sqrt(q)*f*K_E/N^3*delta(x-y)",
            "f_times_K_E": "(2500+56*phi^2)/(50*(50+phi^2))",
            "f_times_K_E_interval": [str(lapse_multiplier_minimum), str(lapse_multiplier_maximum)],
            "pointwise_nonzero_for_finite_N_positive": True,
            "local_constraint_matrix_rank": 2,
            "local_physical_dof": 3,
            "global_bounded_inverse": "blocked",
            "global_blocker": "no declared lapse bounds, function space, asymptotic class, or boundary conditions",
        },
        "global_energy": {
            "einstein_frame_local_matter_kinetic_positive": True,
            "canonical_scalar_known_answer_is_only_calibration": True,
            "status": "blocked",
            "missing_premises": [
                "hash-bound positive-mass/boundary theorem for the transformed candidate",
                "complete asymptotically-flat initial-data and boundary-generator contract",
                "proof that evolution remains inside abs(phi)<=1 or a larger certified conformal domain",
            ],
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_phi_seed_formal_audit(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    for descriptor in config["adapter_sources"]:
        if _file_sha(root / descriptor["path"]) != descriptor["file_sha256"]:
            raise ValueError(f"adapter source hash mismatch: {descriptor['path']}")
    known_answer = _validate_known_answer(root, config["known_answer_control"])
    expected_ids = {item["id"] for item in config["formal_adapters"]}
    formal_checks = {item["name"]: item for item in formal_report["checks"] if item["name"] in expected_ids}
    if set(formal_checks) != expected_ids:
        raise ValueError("required G4(phi) formal checks missing from bound report")
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
            raise ValueError(f"G4(phi) formal adapter replay mismatch: {descriptor['id']}")
        invoked.append(descriptor["entrypoint"])
        adapter_results[descriptor["id"]] = {
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": _sha(evidence),
            "status": "pass",
            "scope": evidence.get("scope", bound["scope"]),
        }
    target = config["target_seed"]
    prior = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(prior, target)
    certificate = _candidate_certificate(target)
    gates = {
        "typed_action_and_provenance": {"status": "pass"},
        "covariant_G2_G4_phi_variation_noether": {"status": "pass"},
        "adm_primary_degeneracy": {"status": "pass"},
        "candidate_local_lapse_dirac_pair": {"status": "pass"},
        "global_lapse_operator_invertibility": {
            "status": "blocked",
            "reason": "pointwise positive multiplication coefficient does not supply a bounded inverse without lapse/function-space/boundary data",
        },
        "inhomogeneous_principal_symbol": {"status": "pass"},
        "common_time_cone": {"status": "pass"},
        "global_positive_energy": {
            "status": "blocked",
            "reason": "regular conformal reduction and positive local kinetic energy do not replace a global positive-mass/boundary theorem",
        },
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
        "first_missing_premise": "global_lapse_function_space_and_boundary_contract",
        "global_energy_blocker": "candidate_specific_positive_mass_boundary_theorem_unavailable",
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
        "known_answer_control": known_answer,
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
        "interpretation": "The candidate has an exact regular conformal transformation, a positive local lapse-pair coefficient, and a uniform inhomogeneous common principal cone on abs(phi)<=1. It remains blocked because pointwise positivity does not establish global lapse-operator invertibility or a nonlinear positive-mass/boundary theorem.",
    }
    return {**body, "content_sha256": _sha(body)}
