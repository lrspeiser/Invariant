from __future__ import annotations

import hashlib
import importlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g2-global-positive-mass-prerequisite-audit-1.0"


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
        if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor[
            "content_sha256"
        ]:
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


def _validate_contract(contract: dict[str, Any]) -> None:
    required = {
        "contract_kind": "conditional_theorem_domain_not_claim_of_data_existence",
        "spatial_dimension": 3,
        "initial_slice": {
            "topology": "smooth_connected_complete_orientable_without_inner_boundary",
            "asymptotic_end": "one_asymptotically_euclidean_end",
            "weighted_regularities": {
                "p": "p>3",
                "q": "1/2<q<1",
                "h_minus_delta": "W^{2,p}_{-q}",
                "K_ij": "W^{1,p}_{-1-q}",
            },
        },
        "scalar_phase_space": {
            "phi_minus_phi_infinity": "O_2(r^-1)",
            "D_i_phi": "O_1(r^-2)",
            "v_equals_nabla_n_phi": "O_1(r^-2)",
            "allowed_variation_delta_phi": "O_1(r^-1)",
            "causal_gradient_cell": "X=(v^2-D_i_phi*D^i_phi)/2 in [0,1]",
        },
        "hamiltonian_smearing": {
            "lapse": "N>=N_min>0 and N=1+O_2(r^-q)",
            "shift": "beta^i=O_2(r^-q)",
            "asymptotic_generator": "unit_time_translation",
        },
        "constraint_domain": {
            "hamiltonian": "R(h)+K^2-K_ij*K^ij=16*pi*G*rho",
            "momentum": "D_j(K^ij-h^ij*K)=8*pi*G*j^i",
            "restricted_core_slice": "K=0",
            "matter_condition": "rho>=sqrt(j_i*j^i)",
        },
        "boundary_contract": {
            "gravitational_charge": "ADM_four_momentum_of_the_asymptotically_euclidean_end",
            "energy_smearing": "N_to_1_beta_to_0",
            "scalar_surface_variation": (
                "lim_r_to_infinity integral_Sr r^2*G2_X*D_r_phi*delta_phi=0"
            ),
            "extra_scalar_boundary_charge": (
                "none_for_minimal_first_derivative_G2_under_declared_falloff"
            ),
        },
    }
    if contract != required:
        raise ValueError("global function-space or boundary contract changed")


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G2 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G2 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G2 predecessor provenance mismatch")
    if (
        record["candidate_certificate"].get("content_sha256")
        != target["prior_candidate_certificate_sha256"]
    ):
        raise ValueError("G2 local certificate hash mismatch")
    if record.get("global_energy_blocker") != "global_positive_mass_and_boundary_adapter_unavailable":
        raise ValueError("G2 predecessor global blocker changed")


def _dec_and_boundary_certificate(target: dict[str, Any]) -> dict[str, Any]:
    a = Fraction(target["quadratic_coefficient"])
    if a <= 0:
        raise ValueError("G2 coefficient must be positive")
    body = {
        "G2": f"X+({a})*X^2",
        "G2_X": f"1+({2 * a})*X",
        "slice_variables": {
            "v": "n^mu*nabla_mu(phi)",
            "s_squared": "D_i(phi)*D^i(phi)>=0",
            "X": "(v^2-s_squared)/2",
        },
        "stress_projections": {
            "rho": f"X+({3 * a})*X^2+(1+({2 * a})*X)*s_squared",
            "j_squared": f"(1+({2 * a})*X)^2*(2*X+s_squared)*s_squared",
            "rho_squared_minus_j_squared": (
                f"(X+({3 * a})*X^2)^2+({2 * a})*X^2*(1+({2 * a})*X)*s_squared"
            ),
        },
        "dominant_energy_condition": {
            "domain": "X in [0,1], s_squared>=0",
            "rho_nonnegative": True,
            "rho_squared_minus_j_squared_nonnegative": True,
            "future_causal_energy_flux": True,
            "status": "pass",
        },
        "falloff_consequences": {
            "X": "O(r^-4)",
            "rho_and_j": "O(r^-4)",
            "matter_sources_integrable": True,
            "scalar_surface_variation": "O(r^-1)->0",
            "extra_scalar_boundary_charge": False,
        },
        "maximal_slice_reduction": {
            "constraint_with_K_zero": "R(h)=16*pi*G*rho+K_ij*K^ij",
            "spatial_scalar_curvature_nonnegative": True,
            "boundary_charge": "E_ADM",
            "restricted_positive_mass_core_applicable": True,
            "scope": "complete asymptotically Euclidean maximal initial data satisfying the bound contract",
        },
    }
    return {**body, "content_sha256": _sha(body)}


def build_g2_global_positive_mass_prerequisite_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_contract(config["global_contract"])
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    core = config["positive_mass_core"]
    if _file_sha(root / core["source_path"]) != core["source_file_sha256"]:
        raise ValueError("positive-mass core source hash mismatch")
    formal_check = next(
        (item for item in formal_report["checks"] if item["name"] == core["id"]), None
    )
    if formal_check is None or formal_check["status"] != "pass":
        raise ValueError("restricted positive-mass core missing from formal report")
    core_evidence = _resolve(core["entrypoint"])()
    if core_evidence != formal_check["evidence"] or core_evidence.get("passed") is not True:
        raise ValueError("restricted positive-mass core replay mismatch")
    if (
        core_evidence.get("generic_status") != "unresolved"
        or core_evidence["theorem_domain"].get("maximal_slice_K_equals_zero") is not True
    ):
        raise ValueError("restricted positive-mass theorem boundary changed")
    predecessor_by_id = {item["seed_id"]: item for item in predecessor["candidate_records"]}
    records = []
    for target in config["target_seeds"]:
        prior = predecessor_by_id[target["seed_id"]]
        _validate_target(prior, target)
        certificate = _dec_and_boundary_certificate(target)
        gates = {
            "typed_action_local_formal_predecessor": {"status": "pass"},
            "explicit_global_contract": {
                "status": "pass",
                "scope": "conditional domain definition, not proof that arbitrary candidate data satisfy it",
            },
            "candidate_DEC_on_contract_domain": {"status": "pass"},
            "scalar_boundary_flux": {"status": "pass"},
            "adm_boundary_charge_identification": {
                "status": "pass",
                "scope": "minimal first-derivative matter under the declared falloff",
            },
            "restricted_maximal_slice_positive_mass": {
                "status": "pass",
                "scope": "Riemannian positive-mass core only on K=0 contract data",
            },
            "general_nonmaximal_positive_mass": {
                "status": "blocked",
                "reason": (
                    "no hash-bound candidate adapter proves E_ADM>=sqrt(P_i*P^i) for complete "
                    "nonmaximal Einstein-G2 constraint data satisfying DEC"
                ),
            },
            "global_positive_energy": {
                "status": "blocked",
                "reason": "a restricted maximal-slice theorem cannot certify the full nonmaximal phase space",
            },
            "formal_prerequisite_completion": {"status": "blocked"},
        }
        provenance_body = {
            "predecessor_content_sha256": config["predecessor"]["content_sha256"],
            "seed_id": target["seed_id"],
            "action_sha256": target["action_sha256"],
            "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
            "prior_candidate_certificate_sha256": target["prior_candidate_certificate_sha256"],
            "global_contract_sha256": _sha(config["global_contract"]),
            "dec_boundary_certificate_sha256": certificate["content_sha256"],
            "positive_mass_core_evidence_sha256": _sha(core_evidence),
            "data_eligibility": ELIGIBILITY,
        }
        records.append(
            {
                "seed_id": target["seed_id"],
                "action_sha256": target["action_sha256"],
                "decision": "blocked",
                "dec_and_boundary_certificate": certificate,
                "gate_ledger": gates,
                "restricted_theorem_result": "pass_on_explicit_maximal_slice_contract",
                "first_missing_premise": "hash_bound_general_nonmaximal_positive_mass_theorem",
                "negative_total_energy_counterexample_found": False,
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
            "positive_mass_core": core,
        },
        "global_contract": config["global_contract"],
        "global_contract_sha256": _sha(config["global_contract"]),
        "positive_mass_core_replay": {
            "entrypoint": core["entrypoint"],
            "role": core["role"],
            "evidence_sha256": _sha(core_evidence),
            "generic_status": core_evidence["generic_status"],
            "eligible_as_direct_candidate_evidence": False,
        },
        "target_seed_count": len(records),
        "decision_counts": {"blocked": len(records)},
        "candidate_records": records,
        "restricted_maximal_slice_pass_count": len(records),
        "full_formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Both exact G2 seeds satisfy DEC and have no extra scalar boundary charge under the "
            "sealed asymptotic contract. Existing repository machinery supports a conditional "
            "maximal-slice Riemannian positive-mass reduction. It does not contain a hash-bound "
            "general nonmaximal Einstein-matter positive-mass adapter, so neither candidate is "
            "promoted from the restricted result."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
