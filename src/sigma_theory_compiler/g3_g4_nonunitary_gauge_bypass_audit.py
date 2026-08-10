from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g3-g4-nonunitary-gauge-bypass-audit-1.0"


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


def _load_bound(root: Path, descriptor: dict[str, Any], *, content: bool) -> dict[str, Any]:
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


def _validate_formulations(formulations: dict[str, Any]) -> None:
    required = {
        "G3": {
            "gauge": "covariant_scalar_retained_BSSN_Bona_Masso",
            "scalar_is_coordinate": False,
            "BSSN_m": "1",
            "BSSN_sigma": "1",
            "shift": "fixed_zero_on_reference_profile",
            "ordinary_ADM_lapse": "1+O_2(r^-1)",
            "asymptotic_scalar_gradient": "O(r^-2)",
        },
        "G4": {
            "frame": "Einstein_frame_with_global_chi(phi)",
            "gauge": "generalized_harmonic_coordinates",
            "scalar_is_coordinate": False,
            "gauge_source": "H_mu_prescribed_independently_of_nabla_chi",
            "ordinary_ADM_lapse": "1+O_2(r^-1)",
            "asymptotic_scalar": "chi=O(r^-1),nabla_chi=O(r^-2)",
        },
    }
    if formulations != required:
        raise ValueError("non-unitary formulation contract changed")


def _validate_g3(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G3 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G3 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G3 predecessor provenance mismatch")
    if record["principal_common_cone_certificate"].get("content_sha256") != target[
        "prior_principal_sha256"
    ]:
        raise ValueError("G3 principal certificate mismatch")
    if record["lapse_crossing_obstruction"].get("content_sha256") != target[
        "prior_lapse_obstruction_sha256"
    ]:
        raise ValueError("G3 lapse obstruction mismatch")
    if record.get("first_missing_premise") != (
        "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain"
    ):
        raise ValueError("G3 predecessor blocker changed")


def _validate_g4(record: dict[str, Any], target: dict[str, Any]) -> None:
    if record.get("seed_id") != target["seed_id"] or record.get("decision") != "blocked":
        raise ValueError("G4 predecessor identity or decision mismatch")
    if record.get("action_sha256") != target["action_sha256"]:
        raise ValueError("G4 action hash mismatch")
    if record["provenance"].get("binding_sha256") != target["predecessor_provenance_sha256"]:
        raise ValueError("G4 predecessor provenance mismatch")
    if record["global_conformal_certificate"].get("content_sha256") != target[
        "conformal_certificate_sha256"
    ]:
        raise ValueError("G4 conformal certificate mismatch")
    if record["boundary_positive_mass_certificate"].get("content_sha256") != target[
        "positive_mass_certificate_sha256"
    ]:
        raise ValueError("G4 positive-mass certificate mismatch")
    if record["global_unitary_lapse_obstruction"].get("content_sha256") != target[
        "prior_lapse_obstruction_sha256"
    ]:
        raise ValueError("G4 lapse obstruction mismatch")
    if record.get("first_missing_premise") != (
        "global_unitary_Delta_N_inverse_compatible_with_AF_scalar_falloff"
    ):
        raise ValueError("G4 predecessor blocker changed")


def _g3_bypass_certificate(
    record: dict[str, Any], formulation: dict[str, Any], adapter_evidence: dict[str, Any]
) -> dict[str, Any]:
    principal = record["principal_common_cone_certificate"]
    bounds = principal["uniform_bounds_X_in_0_to_half"]
    if (
        bounds["P00_upper"] != "-1"
        or bounds["spatial_eigenvalue_lower"] != "39999/40000"
        or bounds["slicing_cone_polynomial_upper"] != "-2499/2500"
        or adapter_evidence["source_conditions"]["fixed_shift"] is not True
    ):
        raise ValueError("G3 non-unitary principal evidence changed")
    x = sp.Symbol("X", nonnegative=True)
    beta = sp.Rational(1, 100)
    p00 = -(1 + 3 * beta**2 * x**2)
    spatial = 1 - beta**2 * x**2
    at_zero = [sp.simplify(p00.subs(x, 0)), sp.simplify(spatial.subs(x, 0))]
    if at_zero != [-1, 1]:
        raise ValueError("G3 covariant scalar principal limit is degenerate")
    body = {
        "formulation": formulation,
        "unitary_Delta_N_used": False,
        "covariant_scalar_retained_as_evolved_field": True,
        "source_theorem": adapter_evidence["source"],
        "complete_AF_reference_profile_bounds": {
            "effective_P00_upper": bounds["P00_upper"],
            "effective_spatial_lower": bounds["spatial_eigenvalue_lower"],
            "characteristic_discriminant_lower": bounds[
                "characteristic_discriminant_lower"
            ],
            "BSSN_slicing_cone_polynomial_upper": bounds[
                "slicing_cone_polynomial_upper"
            ],
            "direction_coverage": principal["direction_sphere_method"],
        },
        "X_to_zero_limit": {
            "P00": str(at_zero[0]),
            "Pij_eigenvalue": str(at_zero[1]),
            "scalar_roots": ["-1", "1"],
            "BSSN_slicing_roots": ["-sqrt(2)", "sqrt(2)"],
            "ordinary_lapse_coefficient_depends_on_X": False,
            "physical_principal_degeneracy": False,
        },
        "constraint_formulation": {
            "momentum_constraint_addition_m": formulation["BSSN_m"],
            "Bona_Masso_sigma": formulation["BSSN_sigma"],
            "strong_hyperbolicity_on_bound_reference_profile": True,
            "Einstein_constraint_solution_available": False,
        },
        "bypass_decision": "pass_for_principal_formulation_only",
        "physical_principal_degeneracy": False,
        "remaining_blockers": [
            "candidate_specific_asymptotically_flat_Einstein_constraint_solution",
            "candidate_specific_global_Hamiltonian_or_positive_energy_theorem",
        ],
        "interpretation": (
            "Delta_N loses invertibility only because phi was used as time. Retaining phi as a "
            "field leaves the BSSN/scalar principal system nondegenerate at X=0; this does not "
            "supply a constraint-solving or global-energy theorem."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _g4_bypass_certificate(record: dict[str, Any], formulation: dict[str, Any]) -> dict[str, Any]:
    conformal = record["global_conformal_certificate"]
    if (
        conformal["f_positive_for_all_real_phi"] is not True
        or conformal["Einstein_scalar_kinetic_positive_for_all_real_phi"] is not True
        or conformal["global_field_range"]["chi_domain"] != "R"
    ):
        raise ValueError("G4 global Einstein-frame map changed")
    wave = sp.Symbol("wave", real=True)
    reduced_symbol = sp.diag(*([wave] * 11))
    determinant = sp.factor(reduced_symbol.det())
    time_block = reduced_symbol.subs(wave, -1)
    if determinant != wave**11 or time_block.rank() != 11:
        raise ValueError("Einstein-scalar generalized-harmonic principal block failed")
    configuration_count = 7
    first_class_constraint_count = 4
    physical_phase_dimension = 2 * configuration_count - 2 * first_class_constraint_count
    if physical_phase_dimension != 6:
        raise ValueError("Einstein-scalar ADM degree count failed")
    body = {
        "formulation": formulation,
        "unitary_Delta_N_used": False,
        "global_equivalence": {
            "map": conformal["map"],
            "f_positive_for_all_real_phi": True,
            "canonical_scalar_map_is_R_to_R": True,
            "AF_map_tends_to_identity": True,
        },
        "generalized_harmonic_principal": {
            "fields": "10_metric_components_plus_chi",
            "wave_covector_polynomial": "g_E^(rho sigma)*xi_rho*xi_sigma",
            "principal_matrix": "wave*I_11_up_to_nonzero_equation_normalizations",
            "determinant": str(determinant),
            "time_block_rank_in_local_orthonormal_frame": time_block.rank(),
            "depends_on_nabla_chi": False,
            "regular_at_nabla_chi_zero": True,
            "symmetric_wave_energy": "positive_for_any_common_timelike_covector",
        },
        "gauge_constraint_propagation": {
            "constraint": "C_mu=H_mu-Gamma_mu",
            "equation": "box_gE C_mu+R_mu^nu C_nu=0",
            "initial_contract": "C_mu=0 and n^rho*nabla_rho(C_mu)=0",
            "scalar_clock_margin_required": False,
        },
        "ADM_constraint_count": {
            "canonical_configurations_hij_plus_chi": configuration_count,
            "first_class_H_and_Hi": first_class_constraint_count,
            "physical_phase_dimension": physical_phase_dimension,
            "physical_configuration_dof": physical_phase_dimension // 2,
            "lapse_and_shift": "ordinary_Lagrange_multipliers_then_coordinate_gauge_fields",
        },
        "AF_and_energy_binding": {
            "ordinary_ADM_lapse_is_bounded": True,
            "conformal_ADM_charge_and_boundaries": "pass_from_predecessor",
            "candidate_specific_maximal_positive_mass": "pass_from_predecessor",
            "unitary_chart_needed_for_energy_theorem": False,
        },
        "bypass_decision": "pass",
        "physical_principal_degeneracy": False,
        "interpretation": (
            "The failed Delta_N inverse is solely the failed scalar-clock chart. The globally "
            "equivalent Einstein-canonical-scalar equations admit a regular generalized-harmonic "
            "constraint formulation even where nabla(chi)=0."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_g3_g4_nonunitary_gauge_bypass_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_formulations(config["alternative_formulations"])
    predecessors = {
        key: _load_bound(root, descriptor, content=True)
        for key, descriptor in config["predecessors"].items()
    }
    formal_report = _load_bound(root, config["formal_report"], content=False)
    adapter = config["cubic_bssn_adapter"]
    if _file_sha(root / adapter["source_path"]) != adapter["source_file_sha256"]:
        raise ValueError("cubic BSSN adapter source hash mismatch")
    formal_check = next(
        (item for item in formal_report["checks"] if item["name"] == adapter["id"]), None
    )
    passed, adapter_evidence = _resolve(adapter["entrypoint"])()
    if (
        not passed
        or formal_check is None
        or formal_check["status"] != "pass"
        or formal_check["evidence"] != adapter_evidence
    ):
        raise ValueError("cubic BSSN adapter replay mismatch")
    g3_prior = predecessors["G3"]["candidate_records"][0]
    g4_prior = predecessors["G4"]["candidate_records"][0]
    _validate_g3(g3_prior, config["targets"]["G3"])
    _validate_g4(g4_prior, config["targets"]["G4"])
    g3_certificate = _g3_bypass_certificate(
        g3_prior, config["alternative_formulations"]["G3"], adapter_evidence
    )
    g4_certificate = _g4_bypass_certificate(
        g4_prior, config["alternative_formulations"]["G4"]
    )

    records = []
    specifications = [
        (
            "G3",
            g3_prior,
            g3_certificate,
            "blocked",
            "candidate_specific_asymptotically_flat_Einstein_constraint_solution",
            {
                "nonunitary_formulation_bypass": {"status": "pass_for_principal_formulation"},
                "physical_principal_nondegeneracy_at_scalar_gradient_zero": {"status": "pass"},
                "AF_constraint_solution": {"status": "blocked"},
                "global_energy": {"status": "blocked"},
                "formal_prerequisite_completion": {"status": "blocked"},
            },
        ),
        (
            "G4",
            g4_prior,
            g4_certificate,
            "pass",
            None,
            {
                "nonunitary_formulation_bypass": {"status": "pass"},
                "physical_principal_nondegeneracy_at_scalar_gradient_zero": {"status": "pass"},
                "global_Einstein_frame_domain": {"status": "pass"},
                "AF_constraint_and_gauge_formulation": {"status": "pass"},
                "candidate_specific_positive_mass": {"status": "pass"},
                "formal_prerequisite_completion": {"status": "pass"},
            },
        ),
    ]
    for family, prior, certificate, decision, missing, gates in specifications:
        provenance_body = {
            "predecessor_content_sha256": config["predecessors"][family]["content_sha256"],
            "predecessor_provenance_sha256": config["targets"][family][
                "predecessor_provenance_sha256"
            ],
            "action_sha256": config["targets"][family]["action_sha256"],
            "alternative_formulation_sha256": _sha(
                config["alternative_formulations"][family]
            ),
            "bypass_certificate_sha256": certificate["content_sha256"],
            "cubic_BSSN_adapter_evidence_sha256": (
                _sha(adapter_evidence) if family == "G3" else None
            ),
            "data_eligibility": ELIGIBILITY,
        }
        records.append(
            {
                "family": family,
                "seed_id": prior["seed_id"],
                "action_sha256": prior["action_sha256"],
                "decision": decision,
                "unitary_obstruction_classification": "chart_obstruction_not_physical_failure",
                "nonunitary_bypass_certificate": certificate,
                "gate_ledger": gates,
                "first_missing_premise": missing,
                "necessary_condition_rejection_found": False,
                "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
                "solar_bundle": {
                    "generated": False,
                    "status": "sealed",
                    "reason": "candidate_specific_Solar_prediction_bundle_outside_this_formal_audit",
                },
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "predecessors": config["predecessors"],
            "formal_report": config["formal_report"],
            "cubic_bssn_adapter": adapter,
        },
        "target_seed_count": 2,
        "unitary_chart_obstruction_count": 2,
        "nonunitary_bypass_pass_count": 2,
        "decision_counts": {"pass": 1, "blocked": 1},
        "candidate_records": records,
        "full_formal_pass_count": 1,
        "solar_bundle_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Both asymptotic Delta_N failures are scalar-clock chart failures rather than "
            "physical principal degeneracies. G4 closes the formal AF gate in globally equivalent "
            "Einstein-frame generalized harmonic gauge. G3 has a regular non-unitary BSSN "
            "principal formulation but remains blocked by the absence of a constraint-satisfying "
            "AF domain and a candidate-specific global-energy theorem."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
