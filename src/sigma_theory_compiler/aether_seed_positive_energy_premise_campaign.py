from __future__ import annotations

import hashlib
import importlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-aether-seed-positive-energy-premise-campaign-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"
REQUIRED_FORMAL_CHECKS = {
    "einstein_aether_linearized_physical_energy",
    "einstein_aether_restricted_nonlinear_total_energy",
}


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


def _twisting_unit_aether_witness() -> dict[str, Any]:
    """Return an exact normalized covector with nonzero Frobenius three-form."""

    x = sp.symbols("x", real=True)
    amplitude = sp.sqrt(1 + x**2)
    u_t, u_x, u_y = -amplitude, sp.Integer(0), x
    norm_residual = sp.factor(-u_t**2 + u_x**2 + u_y**2 + 1)
    frobenius_txy = sp.factor(
        u_t * sp.diff(u_y, x) + u_y * (-sp.diff(u_t, x))
    )
    origin_value = sp.factor(frobenius_txy.subs(x, 0))
    body = {
        "background": "Minkowski metric diag(-1,1,1,1)",
        "covector": ["-sqrt(1+x^2)", "0", "x", "0"],
        "unit_constraint_residual": str(norm_residual),
        "frobenius_component": "(u wedge du)_txy=" + str(frobenius_txy),
        "frobenius_component_at_x_zero": str(origin_value),
        "unit_constraint_pass": norm_residual == 0,
        "hypersurface_orthogonality_rejected_for_witness": origin_value != 0,
        "scope": "kinematic field-space witness, not an equation-of-motion solution or negative-energy counterexample",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_specialization(parameters: dict[str, str]) -> dict[str, Any]:
    c1 = Fraction(parameters["c1"])
    c2 = Fraction(parameters["c2"])
    c3 = Fraction(parameters["c3"])
    c4 = Fraction(parameters["c4"])
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    vector_numerator = 2 * c1 - c1**2 + c3**2
    scalar_trace = 2 + c13 + 3 * c2
    speeds = {
        "spin_2": 1 / (1 - c13),
        "spin_1": vector_numerator / (2 * c14 * (1 - c13)),
        "spin_0": c123 * (2 - c14) / (c14 * (1 - c13) * scalar_trace),
    }
    energies = {
        "spin_2": Fraction(1),
        "spin_1": vector_numerator / (1 - c13),
        "spin_0": c14 * (2 - c14),
    }
    acceleration_coefficient = c14 * (1 - c14 / 2)
    shear_coefficient = 1 - c13
    body = {
        "method": "exact_fraction_arithmetic",
        "parameters": parameters,
        "coupling_combinations": {
            "c13": str(c13),
            "c14": str(c14),
            "c123": str(c123),
            "twist_sector_c1_minus_c3": str(c1 - c3),
        },
        "linearized_speed_squared": {name: str(value) for name, value in speeds.items()},
        "linearized_energy_coefficients": {name: str(value) for name, value in energies.items()},
        "restricted_theorem_coefficients": {
            "c14_times_one_minus_c14_over_two": str(acceleration_coefficient),
            "one_minus_c13": str(shear_coefficient),
        },
        "coupling_domain": {
            "zero_le_c14_le_two": 0 <= c14 <= 2,
            "c13_le_one": c13 <= 1,
            "all_linearized_speeds_positive": all(value > 0 for value in speeds.values()),
            "all_linearized_energies_positive": all(value > 0 for value in energies.values()),
            "restricted_curvature_coefficients_positive": acceleration_coefficient > 0 and shear_coefficient > 0,
        },
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_predecessor_record(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("family_id") != TARGET_FAMILY:
        raise ValueError("target seed identity or family mismatch")
    if record["typed_action_ir"].get("content_sha256") != target["action_sha256"]:
        raise ValueError("target action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("target predecessor provenance mismatch")
    if record["typed_action_ir"].get("parameters") != target["parameters"]:
        raise ValueError("target rational parameter point mismatch")
    nonpass = {
        name: gate["status"]
        for name, gate in record["gate_ledger"].items()
        if gate["status"] != "pass"
    }
    if nonpass != {
        "formal_prerequisite_completion": "blocked",
        "hamiltonian_stability": "unresolved",
    }:
        raise ValueError("target does not have the expected sole substantive unresolved gate")


def build_aether_seed_positive_energy_premise_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    adapter_source = root / config["adapter_source"]["path"]
    if _file_sha(adapter_source) != config["adapter_source"]["file_sha256"]:
        raise ValueError("formal adapter source hash mismatch")
    formal_checks = {
        item["name"]: item for item in formal_report["checks"] if item["name"] in REQUIRED_FORMAL_CHECKS
    }
    if set(formal_checks) != REQUIRED_FORMAL_CHECKS or any(
        item["status"] != "pass" for item in formal_checks.values()
    ):
        raise ValueError("required hash-bound Aether energy controls are absent or failed")
    adapter_results = {}
    adapter_evidence = {}
    invoked_entrypoints = []
    for adapter in config["formal_adapters"]:
        result = _resolve(adapter["entrypoint"])()
        if result.get("passed") is not True:
            raise ValueError(f"formal adapter failed: {adapter['id']}")
        if result != formal_checks[adapter["id"]]["evidence"]:
            raise ValueError(f"formal adapter replay differs from bound report: {adapter['id']}")
        adapter_results[adapter["id"]] = {
            "entrypoint": adapter["entrypoint"],
            "result_sha256": _sha(result),
            "status": "pass",
            "scope": result["scope"],
        }
        adapter_evidence[adapter["id"]] = result
        invoked_entrypoints.append(adapter["entrypoint"])
    nonlinear = adapter_evidence["einstein_aether_restricted_nonlinear_total_energy"]
    if (
        nonlinear.get("theorem_status") != "pass_in_restricted_subsector"
        or nonlinear.get("generic_status") != "unresolved"
        or nonlinear["out_of_domain_controls"]["aether_with_twist"]["theorem_premise_rejected"] is not True
    ):
        raise ValueError("restricted positive-energy theorem boundary changed")
    witness = _twisting_unit_aether_witness()
    records_by_id = {item["seed_id"]: item for item in predecessor["candidate_records"]}
    records = []
    for target in config["target_seeds"]:
        record = records_by_id[target["seed_id"]]
        _validate_predecessor_record(record, target)
        specialization = _candidate_specialization(target["parameters"])
        if not all(specialization["coupling_domain"].values()):
            decision = "reject"
            first_missing = "coupling_domain"
        else:
            action_atoms = [item["atom"] for item in record["typed_action_ir"]["operators"]]
            if "FROBENIUS_CONSTRAINT" in action_atoms:
                raise ValueError("unexpected hypersurface-orthogonality constraint requires review")
            decision = "blocked"
            first_missing = "hypersurface_orthogonal_aether"
        provenance_body = {
            "predecessor_content_sha256": config["predecessor"]["content_sha256"],
            "seed_id": target["seed_id"],
            "action_sha256": target["action_sha256"],
            "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
            "specialization_sha256": specialization["content_sha256"],
            "twist_witness_sha256": witness["content_sha256"],
            "adapter_result_sha256": {
                key: value["result_sha256"] for key, value in sorted(adapter_results.items())
            },
            "data_eligibility": ELIGIBILITY,
        }
        records.append(
            {
                "seed_id": target["seed_id"],
                "action_sha256": target["action_sha256"],
                "decision": decision,
                "exact_specialization": specialization,
                "premise_ledger": {
                    "linearized_physical_energy": {
                        "status": "pass",
                        "reason": "all five candidate-specific mode speeds and energy coefficients are positive",
                    },
                    "restricted_theorem_coupling_domain": {
                        "status": "pass" if decision == "blocked" else "reject",
                        "reason": "exact c13/c14 coefficients satisfy the restricted theorem inequalities",
                    },
                    "hypersurface_orthogonal_aether": {
                        "status": "blocked",
                        "reason": "the typed action has no Frobenius constraint and admits an exact normalized nonzero-twist field-space witness",
                    },
                    "remaining_restricted_theorem_premises": {
                        "status": "not_evaluated_after_first_blocker",
                        "premises": [
                            "asymptotically_flat_complete_orientable_slice",
                            "maximal_slice_K_equals_zero",
                            "matter_energy_density_nonnegative",
                        ],
                    },
                    "generic_nonlinear_hamiltonian_stability": {
                        "status": "blocked",
                        "reason": "a restricted positive-energy theorem cannot certify the full twisting-Aether phase space",
                    },
                },
                "first_missing_premise": first_missing,
                "negative_energy_counterexample_found": False,
                "restricted_subsector_theorem_status": "pass_in_restricted_subsector",
                "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["decision"]] = counts.get(record["decision"], 0) + 1
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "predecessor": config["predecessor"],
            "formal_report": config["formal_report"],
            "adapter_source": config["adapter_source"],
            "formal_check_sha256": {
                name: _sha(item) for name, item in sorted(formal_checks.items())
            },
        },
        "invoked_adapter_entrypoints": invoked_entrypoints,
        "adapter_results": adapter_results,
        "twisting_unit_aether_witness": witness,
        "target_seed_count": len(records),
        "decision_counts": dict(sorted(counts.items())),
        "candidate_records": records,
        "formal_pass_count": 0,
        "solar_bundles_generated": 0,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": "Both rational points satisfy the known restricted theorem's coupling inequalities, but neither action restricts its phase space to hypersurface-orthogonal Aether. The exact twist witness blocks generic positive-energy promotion without rejecting either theory.",
    }
    return {**body, "content_sha256": _sha(body)}
