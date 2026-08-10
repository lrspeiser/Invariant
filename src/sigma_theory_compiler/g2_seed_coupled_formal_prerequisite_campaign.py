from __future__ import annotations

import hashlib
import importlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g2-seed-coupled-formal-prerequisite-campaign-1.0"
TARGET_FAMILY = "KESSENCE_G2_CONVEX"


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
        raise ValueError("canonical-scalar known-answer control mismatch")
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
        raise ValueError("G2 target identity or family mismatch")
    if record["typed_action_ir"].get("content_sha256") != target["action_sha256"]:
        raise ValueError("G2 target action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G2 predecessor provenance mismatch")
    if record["typed_action_ir"].get("parameters") != target["parameters"]:
        raise ValueError("G2 rational parameter point mismatch")
    atoms = [item["atom"] for item in record["typed_action_ir"]["operators"]]
    if atoms != ["EH_R", "G2_PHI_X"]:
        raise ValueError("G2 target is not the exact minimally coupled typed action")


def _extract_coefficient(target: dict[str, Any]) -> Fraction:
    match = re.fullmatch(r"X_phi\+\((\d+/\d+)\)\*X_phi\^2", target["parameters"]["G2"])
    if match is None:
        raise ValueError("unsupported G2 polynomial shape")
    parsed = Fraction(match.group(1))
    if parsed != Fraction(target["quadratic_coefficient"]):
        raise ValueError("G2 quadratic coefficient binding mismatch")
    return parsed


def _candidate_certificate(target: dict[str, Any]) -> dict[str, Any]:
    a = _extract_coefficient(target)
    if target["parameters"]["X_domain"] != "0<=X_phi<=1" or a <= 0:
        raise ValueError("unsupported G2 parameter domain")
    speed_minimum = (1 + 2 * a) / (1 + 6 * a)
    pointwise_hamiltonian = f"X_phi+({3 * a})*X_phi^2+(1+({2 * a})*X_phi)*s_squared"
    lapse_numerator = f"N^2+({3 * a})"
    body = {
        "method": "exact_rational_polynomial_interval_and_covariant_cone_algebra",
        "G2": target["parameters"]["G2"],
        "domain": target["parameters"]["X_domain"],
        "quadratic_coefficient": str(a),
        "derivatives": {
            "G2_X": f"1+({2 * a})*X_phi",
            "G2_XX": str(2 * a),
            "G2_X_plus_2X_G2_XX": f"1+({6 * a})*X_phi",
        },
        "uniform_margins": {
            "G2_X_minimum": "1",
            "G2_X_plus_2X_G2_XX_minimum": "1",
            "scalar_speed_squared_interval": [str(speed_minimum), "1"],
            "constant_G4_minus_2X_G4_X": "1/2",
        },
        "common_time_cone": {
            "effective_inverse_metric": "P^munu=G2_X*g^munu-G2_XX*p^mu*p^nu",
            "shared_timelike_covector_evaluation": "P^munu*tau_mu*tau_nu=-G2_X*T^2-G2_XX*(p_dot_tau)^2<0",
            "assumptions": ["g^munu*tau_mu*tau_nu=-T^2 with T>0", "G2_X>0", "G2_XX>0"],
            "covers_declared_causal_gradient_cell": True,
            "status": "pass",
        },
        "unitary_lapse_pair": {
            "X_of_N": "1/(2*N^2)",
            "Delta_N_multiplication_factor": f"({lapse_numerator})/N^5",
            "strictly_positive_for_finite_N_positive": True,
            "local_poisson_rank": 2,
            "local_physical_dof": 3,
            "global_function_space_and_boundary_conditions_declared": False,
        },
        "pointwise_scalar_hamiltonian": {
            "expression": pointwise_hamiltonian,
            "derivation": "-G2+G2_X*v_n^2 with v_n^2=2*X_phi+s_squared",
            "nonnegative_on_domain": True,
            "strict_velocity_convexity": True,
        },
        "dominant_energy_condition": {
            "pressure": f"X_phi+({a})*X_phi^2",
            "energy_density": f"X_phi+({3 * a})*X_phi^2",
            "rho_minus_pressure": f"({2 * a})*X_phi^2",
            "rho_plus_pressure": f"2*X_phi+({4 * a})*X_phi^2",
            "status_on_causal_gradient_domain": "pass",
            "scope": "candidate matter stress algebra only; not a global positive-mass theorem",
        },
        "global_positive_energy": {
            "status": "blocked",
            "missing_premises": [
                "hash-bound minimally-coupled-k-essence positive-mass theorem adapter",
                "complete asymptotically-flat initial-data and boundary-generator contract",
                "global function-space control of the coupled constraints",
            ],
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_g2_seed_coupled_formal_prerequisite_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
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
        raise ValueError("required formal checks missing from bound report")
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
        if not passed or formal_checks[descriptor["id"]]["status"] != "pass":
            raise ValueError(f"formal adapter failed: {descriptor['id']}")
        if evidence != formal_checks[descriptor["id"]]["evidence"]:
            raise ValueError(f"formal adapter replay differs from bound report: {descriptor['id']}")
        invoked.append(descriptor["entrypoint"])
        adapter_results[descriptor["id"]] = {
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": _sha(evidence),
            "status": "pass",
            "scope": evidence["scope"],
        }
    predecessor_by_id = {item["seed_id"]: item for item in predecessor["candidate_records"]}
    records = []
    for target in config["target_seeds"]:
        prior = predecessor_by_id[target["seed_id"]]
        _validate_target(prior, target)
        certificate = _candidate_certificate(target)
        gates = {
            "typed_action_and_provenance": {"status": "pass"},
            "covariant_variation_noether": {"status": "pass"},
            "coupled_adm_primary_and_legendre": {"status": "pass"},
            "candidate_local_dirac_pair": {"status": "pass"},
            "complete_distributed_dirac_boundary_contract": {
                "status": "blocked",
                "reason": "pointwise Delta_N is positive, but no global function space or boundary contract is declared",
            },
            "principal_symbol": {"status": "pass"},
            "common_time_cone": {"status": "pass"},
            "pointwise_hamiltonian": {"status": "pass"},
            "dominant_energy_condition": {"status": "pass"},
            "global_positive_energy": {
                "status": "blocked",
                "reason": "DEC and pointwise positivity do not replace a hash-bound global positive-mass and boundary theorem",
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
        records.append(
            {
                "seed_id": target["seed_id"],
                "action_sha256": target["action_sha256"],
                "decision": "blocked",
                "candidate_certificate": certificate,
                "gate_ledger": gates,
                "first_missing_premise": "complete_distributed_dirac_boundary_contract",
                "global_energy_blocker": "global_positive_mass_and_boundary_adapter_unavailable",
                "negative_energy_counterexample_found": False,
                "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
                "solar_bundle": {"generated": False, "status": "blocked"},
            }
        )
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
        "target_seed_count": len(records),
        "decision_counts": {"blocked": len(records)},
        "candidate_records": records,
        "formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": "Both G2 polynomials pass exact local ADM, local Dirac-rank, principal, common-cone, pointwise-energy, and DEC necessary conditions. They remain blocked because local algebra and a canonical-scalar calibration do not supply the candidate-specific global boundary/function-space and positive-mass theorems.",
    }
    return {**body, "content_sha256": _sha(body)}
