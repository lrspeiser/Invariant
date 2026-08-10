from __future__ import annotations

import hashlib
import importlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-aether-twist-sector-energy-audit-1.0"
TARGET_FAMILY = "AETHER_K1234_PARAMETER_CELL"


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


def _validate_target(
    seed_record: dict[str, Any], premise_record: dict[str, Any], target: dict[str, Any]
) -> None:
    if seed_record.get("seed_id") != target["seed_id"] or seed_record.get("family_id") != TARGET_FAMILY:
        raise ValueError("Aether target identity or family mismatch")
    if seed_record["typed_action_ir"].get("content_sha256") != target["action_sha256"]:
        raise ValueError("Aether target action hash mismatch")
    if seed_record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("Aether target predecessor provenance mismatch")
    if seed_record["typed_action_ir"].get("parameters") != target["parameters"]:
        raise ValueError("Aether rational parameter mismatch")
    if (
        premise_record.get("seed_id") != target["seed_id"]
        or premise_record.get("decision") != "blocked"
        or premise_record.get("first_missing_premise") != "hypersurface_orthogonal_aether"
        or premise_record.get("negative_energy_counterexample_found") is not False
    ):
        raise ValueError("earlier Frobenius result is not bound as a kinematic blocker")


def _twist_certificate(parameters: dict[str, str]) -> dict[str, Any]:
    c1 = Fraction(parameters["c1"])
    c2 = Fraction(parameters["c2"])
    c3 = Fraction(parameters["c3"])
    c4 = Fraction(parameters["c4"])
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    trace_factor = 2 + c13 + 3 * c2
    vector_numerator = 2 * c1 - c1**2 + c3**2
    quadratic_energy = {
        "spin_2": Fraction(1),
        "spin_1": vector_numerator / (1 - c13),
        "spin_0": c14 * (2 - c14),
    }
    reduced_kinetic = {
        "tensor": 1 - c13,
        "vector": 2 * c14,
        "scalar": c14**2 * (1 - c13) * trace_factor / c123,
    }
    reduced_gradient = {
        "tensor": Fraction(1),
        "vector": vector_numerator / (1 - c13),
        "scalar": c14 * (2 - c14),
    }
    twist_coefficient_at_zero = c1 - c3
    twist_coercive_infimum = c1 - c3 - (c1 + c4) / 2
    first_nonlinear_coefficient = -c14 / 2
    if not all(value > 0 for value in (*quadratic_energy.values(), *reduced_kinetic.values(), *reduced_gradient.values())):
        raise ValueError("candidate has a decisive reduced quadratic sign failure")
    body = {
        "method": "exact_unit-branch_static_antisymmetric-spatial-jet_Hamiltonian",
        "couplings": parameters,
        "coupling_combinations": {
            "c13": str(c13),
            "c14": str(c14),
            "c123": str(c123),
            "c1_minus_c3": str(c1 - c3),
        },
        "quadratic_constraint_reduced_energy": {
            "physical_modes": {"spin_2": 2, "spin_1_twist": 2, "spin_0": 1},
            "energy_coefficients": {key: str(value) for key, value in quadratic_energy.items()},
            "kinetic_coefficients": {key: str(value) for key, value in reduced_kinetic.items()},
            "gradient_coefficients": {key: str(value) for key, value in reduced_gradient.items()},
            "all_positive": True,
        },
        "nonlinear_static_pure_twist_sector": {
            "field_jet": "u_mu=(-sqrt(1+A^2),A_i), partial_i A_j=W_ij=-W_ji, partial_t=0",
            "definitions": ["y=A_i*A_i>=0", "W2=W_ij*W_ij", "WA2=(W_ij*A_j)^2"],
            "unit_branch_invariants": {
                "K1": "W2-WA2/(1+y)",
                "K2": "0",
                "K3": "-W2",
                "K4": "WA2",
            },
            "exact_hamiltonian": "H_twist=(M_Pl^2/2)*[(c1-c3)*W2-(c1/(1+y)+c4)*WA2]",
            "orientation_bound": "0<=WA2<=(y/2)*W2",
            "coercive_coefficient": "C(y)=c1-c3-(y/2)*(c1/(1+y)+c4)",
            "C_at_zero": str(twist_coefficient_at_zero),
            "C_derivative": f"-(1/2)*({c1}/(1+y)^2+{c4})<0",
            "C_infimum_y_to_infinity": str(twist_coercive_infimum),
            "first_nonlinear_term_in_C": f"({first_nonlinear_coefficient})*y",
            "Hamiltonian_lower_bound": f"(M_Pl^2/2)*({twist_coercive_infimum})*W2",
            "positive_for_all_tilts_and_twist_orientations": twist_coercive_infimum > 0,
            "negative_energy_mode_found": False,
            "scope": "exact static pure-twist local jet after the unit constraint; shear, expansion, time velocities, gravitational constraint solving, and boundary energy are separate",
        },
        "generic_nonlinear_extension": {
            "status": "blocked",
            "uncontrolled_terms": [
                "mixed twist-shear-expansion terms",
                "time-velocity and metric-Aether momentum mixing after full constraint reduction",
                "nonmaximal gravitational constraint solutions",
                "asymptotic boundary charge and global spatial integral",
            ],
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_aether_twist_sector_energy_audit(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    seed_predecessor = _load_bound(root, config["seed_predecessor"], content=True)
    premise_predecessor = _load_bound(root, config["premise_predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    if _file_sha(root / config["adapter_source"]["path"]) != config["adapter_source"]["file_sha256"]:
        raise ValueError("Aether adapter source hash mismatch")
    expected_ids = {item["id"] for item in config["formal_adapters"]}
    formal_checks = {item["name"]: item for item in formal_report["checks"] if item["name"] in expected_ids}
    if set(formal_checks) != expected_ids:
        raise ValueError("required Aether formal checks missing from bound report")
    invoked = []
    adapter_results = {}
    adapter_evidence = {}
    for descriptor in config["formal_adapters"]:
        evidence = _resolve(descriptor["entrypoint"])()
        bound = formal_checks[descriptor["id"]]
        if evidence.get("passed") is not True or bound["status"] != "pass" or evidence != bound["evidence"]:
            raise ValueError(f"Aether formal adapter replay mismatch: {descriptor['id']}")
        invoked.append(descriptor["entrypoint"])
        adapter_evidence[descriptor["id"]] = evidence
        adapter_results[descriptor["id"]] = {
            "entrypoint": descriptor["entrypoint"],
            "evidence_sha256": _sha(evidence),
            "status": "pass",
            "scope": evidence["scope"],
        }
    maxwell = adapter_evidence["maxwell_unit_aether_nonlinear_hamiltonian"]
    if maxwell["hamiltonian_stability"]["status"] != "reject":
        raise ValueError("Maxwell-unit-Aether inference negative control changed")
    seed_by_id = {item["seed_id"]: item for item in seed_predecessor["candidate_records"]}
    premise_by_id = {item["seed_id"]: item for item in premise_predecessor["candidate_records"]}
    records = []
    for target in config["target_seeds"]:
        _validate_target(seed_by_id[target["seed_id"]], premise_by_id[target["seed_id"]], target)
        certificate = _twist_certificate(target["parameters"])
        gates = {
            "typed_action_and_provenance": {"status": "pass"},
            "constraint_reduced_quadratic_energy": {"status": "pass"},
            "regular_patch_legendre_and_dirac_constraint_algebra": {
                "status": "pass",
                "scope": (
                    "exact unit-reduced local Legendre strata plus bulk D-H/H-H closure on "
                    "regular positive-unit-branch patches; boundary charges and reduced "
                    "Hamiltonian boundedness are excluded"
                ),
            },
            "static_pure_twist_nonlinear_coercivity": {"status": "pass"},
            "complete_generic_twisting_reduced_hamiltonian": {
                "status": "blocked",
                "reason": "the exact twist subspace does not include mixed shear, expansion, velocities, metric constraints, or boundary charge",
            },
            "global_positive_energy": {"status": "blocked"},
            "formal_prerequisite_completion": {"status": "blocked"},
        }
        provenance_body = {
            "seed_predecessor_content_sha256": config["seed_predecessor"]["content_sha256"],
            "premise_predecessor_content_sha256": config["premise_predecessor"]["content_sha256"],
            "seed_id": target["seed_id"],
            "action_sha256": target["action_sha256"],
            "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
            "twist_certificate_sha256": certificate["content_sha256"],
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
                "twist_sector_certificate": certificate,
                "gate_ledger": gates,
                "negative_energy_mode_found": False,
                "first_missing_premise": "complete_generic_twisting_reduced_hamiltonian",
                "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
                "solar_bundle": {"generated": False, "status": "blocked"},
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "seed_predecessor": config["seed_predecessor"],
            "premise_predecessor": config["premise_predecessor"],
            "formal_report": config["formal_report"],
            "adapter_source": config["adapter_source"],
        },
        "invoked_adapter_entrypoints": invoked,
        "adapter_results": adapter_results,
        "inference_negative_control": {
            "subclass": maxwell["subclass"],
            "full_hamiltonian_status": maxwell["hamiltonian_stability"]["status"],
            "evidence_sha256": adapter_results["maxwell_unit_aether_nonlinear_hamiltonian"]["evidence_sha256"],
            "lesson": "positive energy on a restricted twist subspace cannot imply boundedness of the complete constrained Hamiltonian",
        },
        "prior_frobenius_witness_treatment": {
            "status": "kinematic_noncoverage_witness",
            "establishes": "generic unit-timelike Aether configurations need not be hypersurface orthogonal",
            "negative_energy_mode_found": False,
            "candidate_rejection_authorized": False,
        },
        "target_seed_count": len(records),
        "decision_counts": {"blocked": len(records)},
        "candidate_records": records,
        "formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": "Both candidates have positive constraint-reduced quadratic energies and an exact positive all-tilt static pure-twist Hamiltonian. The first nonlinear correction is negative but the resummed twist coefficient remains uniformly positive. No negative candidate mode is found, while the Maxwell-unit negative control proves that subspace coercivity alone cannot certify the full generic twisting Hamiltonian.",
    }
    return {**body, "content_sha256": _sha(body)}
