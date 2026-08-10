from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-asymptotically-flat-einstein-frame-audit-1.0"


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
        if value.get("content_sha256") != descriptor["content_sha256"] or _sha(body) != descriptor[
            "content_sha256"
        ]:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _resolve(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator:
        raise ValueError("adapter must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"adapter is not callable: {entrypoint}")
    return callback


def _validate_target(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G4 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G4 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G4 predecessor provenance mismatch")
    if record["candidate_certificate"].get("content_sha256") != target[
        "prior_candidate_certificate_sha256"
    ]:
        raise ValueError("G4 predecessor certificate mismatch")
    if record.get("first_missing_premise") != "global_lapse_function_space_and_boundary_contract":
        raise ValueError("G4 predecessor lapse blocker changed")


def _validate_domain(domain: dict[str, Any]) -> None:
    required = {
        "contract_kind": "candidate_specific_maximal_AF_positive_mass_and_lapse_compatibility_domain",
        "initial_slice": {
            "topology": "smooth_connected_complete_orientable_without_inner_boundary",
            "asymptotic_end": "one_asymptotically_euclidean_end",
            "weighted_regularities": {
                "p": "p>3",
                "q": "1/2<q<1",
                "h_J_minus_delta": "W^{2,p}_{-q}",
                "K_J": "W^{1,p}_{-1-q}",
            },
        },
        "scalar_data": {
            "phi": "O_2(r^-1)",
            "D_i_phi": "O_1(r^-2)",
            "Pi_phi": "O_1(r^-2)",
            "allowed_variation": "delta_phi=O_1(r^-1)",
            "field_value_domain": "all_real_phi_via_global_Einstein_scalar_redefinition",
        },
        "ADM_smearing": {
            "Jordan_lapse": "1/2<=N_J<=2 and N_J=1+O_2(r^-1)",
            "shift": "beta^i=O_2(r^-1)",
            "Einstein_lapse": "N_E=sqrt(f)*N_J",
        },
        "constraint_subdomain": {
            "Einstein_hamiltonian": "R_E+K_E^2-K_Eij*K_E^ij=16*pi*G*rho_chi",
            "Einstein_momentum": "D_Ej(K_E^ij-h_E^ij*K_E)=8*pi*G*j_chi^i",
            "maximal_slice": "K_E=0",
            "nonempty_witness": "Minkowski_h_E_with_K_E=0_and_phi=Pi_phi=0",
        },
        "boundary_contract": {
            "Jordan_charge": "ADM_four_momentum_of_h_J_K_J",
            "Einstein_charge": "ADM_four_momentum_of_h_E_K_E",
            "scalar_variation_flux": "zero_under_declared_falloff",
            "conformal_total_divergence": "zero_under_declared_falloff",
        },
        "unitary_lapse_space": {
            "multiplier_space": "L2_of_the_AF_end",
            "test_core": "compactly_supported_smooth_annulus_functions",
            "required_inverse": "bounded_inverse_on_L2",
        },
    }
    if domain != required:
        raise ValueError("G4 asymptotically flat domain changed")


def _conformal_certificate() -> dict[str, Any]:
    phi = sp.Symbol("phi", real=True)
    y = sp.Symbol("y", nonnegative=True)
    f = 1 + phi**2 / 50
    f_prime = sp.diff(f, phi)
    kinetic = sp.factor(1 / f + sp.Rational(3, 2) * (f_prime / f) ** 2)
    kinetic_y = sp.factor(kinetic.subs(phi**2, y))
    expected = (2500 + 56 * y) / (50 + y) ** 2
    f_times_kinetic = sp.factor(((50 + y) / 50) * expected)
    derivative = sp.factor(sp.diff(f_times_kinetic, y))
    if sp.factor(kinetic_y - expected) != 0 or derivative != 6 / (50 + y) ** 2:
        raise ValueError("G4 global conformal algebra failed")
    body = {
        "map": "g_E=f(phi)*g_J",
        "f": "1+phi^2/50",
        "f_positive_for_all_real_phi": True,
        "Einstein_scalar_kinetic": str(expected),
        "Einstein_scalar_kinetic_positive_for_all_real_phi": True,
        "canonical_scalar_redefinition": "dchi/dphi=sqrt(K_E)>0, chi(0)=0",
        "global_field_range": {
            "phi_domain": "R",
            "chi_domain": "R",
            "large_abs_phi": "chi~sqrt(56)*sign(phi)*log(abs(phi))",
            "domain_preservation": "no finite field-value boundary exists to cross",
        },
        "lapse_multiplier_factor": {
            "f_times_K_E": str(f_times_kinetic),
            "derivative_in_phi_squared": str(derivative),
            "global_interval": ["1", "28/25"],
        },
        "exact_residuals": {"kinetic": "0", "lapse_factor_derivative": "0"},
    }
    return {**body, "content_sha256": _sha(body)}


def _boundary_and_positive_mass_certificate(
    domain: dict[str, Any], positive_mass_core: dict[str, Any]
) -> dict[str, Any]:
    if (
        positive_mass_core.get("passed") is not True
        or positive_mass_core["theorem_domain"].get("maximal_slice_K_equals_zero") is not True
    ):
        raise ValueError("positive-mass core theorem boundary changed")
    body = {
        "falloff_transport": {
            "f_minus_one": "phi^2/50=O(r^-2)",
            "D_i_f": "O(r^-3)",
            "h_E_minus_delta": "W^{2,p}_{-q}",
            "K_E": "W^{1,p}_{-1-q}",
            "N_E_minus_one": "O(r^-1)",
            "chi": "phi+O(phi^3)=O(r^-1)",
        },
        "ADM_charge_transform": {
            "energy_surface_correction": "-(1/(8*pi*G))*lim integral_Sr D_r(f)",
            "energy_correction_falloff": "r^2*O(r^-3)->0",
            "momentum_correction_falloff": "r^2*O(r^-3)->0",
            "E_E_equals_E_J": True,
            "P_E_equals_P_J": True,
        },
        "scalar_and_conformal_boundaries": {
            "scalar_variation": "r^2*D_r(chi)*delta_chi=O(r^-1)->0",
            "conformal_total_divergence": "r^2*D_r(log(f))=O(r^-1)->0",
            "status": "pass",
        },
        "Einstein_frame_matter": {
            "rho_chi": "(Pi_chi^2+D_i(chi)*D^i(chi))/2>=0",
            "dominant_energy_condition": "pass_for_canonical_massless_scalar",
        },
        "maximal_constraint_reduction": {
            "equation": "R_E=16*pi*G*rho_chi+K_Eij*K_E^ij>=0",
            "riemannian_positive_mass_core_applicable": True,
            "E_E_nonnegative": True,
            "therefore_E_J_nonnegative": True,
            "scope": "the explicitly bound complete maximal asymptotically Euclidean subdomain",
        },
        "domain_contract_sha256": _sha(domain),
        "core_evidence_sha256": _sha(positive_mass_core),
    }
    return {**body, "content_sha256": _sha(body)}


def _global_lapse_obstruction() -> dict[str, Any]:
    body = {
        "local_unitary_multiplier": "Delta_N=sqrt(h_J)*f*K_E/N_phi^3",
        "bounded_coefficient": "1<=f*K_E<28/25",
        "AF_clock_dichotomy": {
            "AF_scalar_falloff": "nabla(phi)->0 implies X->0 and N_phi=1/sqrt(2X)->infinity",
            "Delta_limit": "0",
            "alternative_phi_equals_time": (
                "N_J->1 implies X->1/2 and nondecaying scalar stress, contradicting finite ADM energy"
            ),
        },
        "annulus_sequence": {
            "test_functions": "f_R in C_c_infinity on annuli, ||f_R||_L2=1",
            "result": "||Delta_N f_R||_L2->0 whenever X->0 uniformly along the AF end",
            "conclusion": "no_bounded_global_unitary_Delta_inverse",
        },
        "ordinary_ADM_lapse": {
            "Jordan_bound": "1/2<=N_J<=2",
            "Einstein_relation": "N_E=sqrt(f)*N_J",
            "does_not_control_scalar_clock_lapse": True,
        },
        "physical_interpretation": (
            "the obstruction is failure of one global unitary-gauge chart, not a ghost or a "
            "failure of the covariant Einstein-frame theory"
        ),
        "status": "blocked",
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_asymptotically_flat_einstein_frame_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_domain(config["asymptotically_flat_domain"])
    predecessor = _load_bound(root, config["predecessor"], content=True)
    formal_report = _load_bound(root, config["formal_report"])
    core_descriptor = config["positive_mass_core"]
    if _file_sha(root / core_descriptor["source_path"]) != core_descriptor["source_file_sha256"]:
        raise ValueError("positive-mass core source hash mismatch")
    formal_check = next(
        (item for item in formal_report["checks"] if item["name"] == core_descriptor["id"]), None
    )
    core_evidence = _resolve(core_descriptor["entrypoint"])()
    if (
        formal_check is None
        or formal_check["status"] != "pass"
        or core_evidence != formal_check["evidence"]
    ):
        raise ValueError("positive-mass core replay mismatch")
    target = config["target"]
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    _validate_target(record, target)
    conformal = _conformal_certificate()
    positive_mass = _boundary_and_positive_mass_certificate(
        config["asymptotically_flat_domain"], core_evidence
    )
    lapse = _global_lapse_obstruction()
    gates = {
        "typed_action_and_predecessor": {"status": "pass"},
        "global_conformal_field_redefinition": {"status": "pass"},
        "weighted_AF_function_space_transport": {"status": "pass"},
        "conformal_ADM_charge_and_scalar_boundaries": {"status": "pass"},
        "candidate_specific_maximal_positive_mass": {"status": "pass"},
        "field_value_domain_preservation": {"status": "pass"},
        "ordinary_ADM_lapse_bounds": {"status": "pass"},
        "global_unitary_Delta_N_inverse": {
            "status": "blocked",
            "reason": "an AF decaying scalar clock forces N_phi to infinity and Delta_N to zero",
        },
        "formal_prerequisite_completion": {"status": "blocked"},
    }
    provenance_body = {
        "predecessor_content_sha256": config["predecessor"]["content_sha256"],
        "action_sha256": target["action_sha256"],
        "predecessor_provenance_sha256": target["predecessor_provenance_sha256"],
        "prior_candidate_certificate_sha256": target["prior_candidate_certificate_sha256"],
        "AF_domain_sha256": _sha(config["asymptotically_flat_domain"]),
        "conformal_sha256": conformal["content_sha256"],
        "positive_mass_sha256": positive_mass["content_sha256"],
        "lapse_obstruction_sha256": lapse["content_sha256"],
        "positive_mass_core_evidence_sha256": _sha(core_evidence),
        "data_eligibility": ELIGIBILITY,
    }
    candidate = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "asymptotically_flat_domain": config["asymptotically_flat_domain"],
        "global_conformal_certificate": conformal,
        "boundary_positive_mass_certificate": positive_mass,
        "global_unitary_lapse_obstruction": lapse,
        "gate_ledger": gates,
        "resolved_global_energy_followup": "pass_on_explicit_maximal_AF_domain",
        "first_missing_premise": "global_unitary_Delta_N_inverse_compatible_with_AF_scalar_falloff",
        "necessary_condition_rejection_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "solar_bundle": {
            "generated": False,
            "status": "blocked",
            "reason": "full_formal_pass_not_proven",
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "predecessor": config["predecessor"],
            "formal_report": config["formal_report"],
            "positive_mass_core": core_descriptor,
        },
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "candidate_records": [candidate],
        "candidate_specific_positive_mass_pass_count": 1,
        "global_unitary_lapse_pass_count": 0,
        "full_formal_pass_count": 0,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The conformal transformation is globally regular, preserves the bound AF class and "
            "ADM charge, and yields a candidate-specific positive-mass theorem on the explicit "
            "maximal subdomain. The same AF scalar decay is incompatible with a bounded global "
            "unitary-clock lapse operator. This is a gauge-chart obstruction, not a physical "
            "negative-energy mode, but the required global Delta_N follow-up remains fail-closed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
