from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY
from .solar_promotion_evaluator import (
    GR_SOLAR_BUNDLE,
    bundle_binding,
    solar_known_answer_evaluator,
)

SCHEMA_VERSION = "sigma-g4-conformal-solar-prediction-audit-1.0"


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


def _load_bound(
    root: Path, descriptor: dict[str, Any], *, content: bool = False
) -> dict[str, Any]:
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


def _validate_target(
    formal_record: dict[str, Any], action_record: dict[str, Any], target: dict[str, Any]
) -> None:
    expected_parameters = {
        "G2": "X_phi",
        "G4": "1/2+(1/100)*phi^2",
        "matter_metric": "g_J",
        "phi_infinity": "0",
    }
    if target.get("parameters") != expected_parameters:
        raise ValueError("G4 weak-field parameter or background contract changed")
    if (
        formal_record.get("family") != "G4"
        or formal_record.get("seed_id") != target["seed_id"]
        or formal_record.get("decision") != "pass"
        or formal_record.get("action_sha256") != target["action_sha256"]
        or formal_record.get("first_missing_premise") is not None
    ):
        raise ValueError("formal G4 target identity or pass decision mismatch")
    if formal_record["provenance"].get("binding_sha256") != target[
        "formal_provenance_sha256"
    ]:
        raise ValueError("formal G4 provenance mismatch")
    if formal_record["nonunitary_bypass_certificate"].get("content_sha256") != target[
        "formal_certificate_sha256"
    ]:
        raise ValueError("formal G4 certificate mismatch")
    action = action_record.get("typed_action_ir", {})
    if (
        action_record.get("seed_id") != target["seed_id"]
        or action.get("content_sha256") != target["action_sha256"]
        or action_record["provenance"].get("binding_sha256") != target[
            "action_provenance_sha256"
        ]
        or action.get("matter_coupling") != {"metric": "g_mu_nu", "universal": True}
        or action.get("parameters", {}).get("G2") != target["parameters"]["G2"]
        or action.get("parameters", {}).get("G4") != target["parameters"]["G4"]
    ):
        raise ValueError("typed G4 action or matter coupling mismatch")


def _validate_synthetic_source(contract: dict[str, Any]) -> None:
    required = {
        "role": "known_answer_calibration_not_real_Solar_evidence",
        "shape": "static_uniform_density_sphere",
        "dimensionless_compactness_upper": "1/1000",
        "trace_approximation": "T=-rho",
        "exterior_boundary": "chi->0_at_spatial_infinity",
    }
    if contract != required:
        raise ValueError("synthetic source contract changed")


def _coupling_and_ppn_certificate() -> dict[str, Any]:
    phi = sp.Symbol("phi", real=True)
    f = 1 + phi**2 / 50
    f_phi = sp.diff(f, phi)
    kinetic = sp.factor(1 / f + sp.Rational(3, 2) * (f_phi / f) ** 2)
    coupling = sp.factor((-f_phi / (2 * f)) / sp.sqrt(kinetic))
    expected_coupling = -phi / sp.sqrt(2500 + 56 * phi**2)
    if sp.simplify(coupling - expected_coupling) != 0:
        raise ValueError("Einstein-frame matter coupling derivation failed")
    alpha_zero = sp.simplify(coupling.subs(phi, 0))
    d_chi_d_phi_zero = sp.sqrt(kinetic).subs(phi, 0)
    beta_zero = sp.simplify(sp.diff(coupling, phi).subs(phi, 0) / d_chi_d_phi_zero)
    gamma_minus_one = sp.factor(-2 * alpha_zero**2 / (1 + alpha_zero**2))
    beta_minus_one = sp.factor(
        sp.Rational(1, 2) * beta_zero * alpha_zero**2 / (1 + alpha_zero**2) ** 2
    )
    newton_factor = sp.factor((1 / sp.sqrt(f.subs(phi, 0))) ** 2 * (1 + alpha_zero**2))
    if (
        alpha_zero != 0
        or beta_zero != -sp.Rational(1, 50)
        or gamma_minus_one != 0
        or beta_minus_one != 0
        or newton_factor != 1
    ):
        raise ValueError("candidate PPN specialization failed")
    body = {
        "Jordan_to_Einstein": {
            "f": str(f),
            "g_E": "f*g_J",
            "matter_conformal_factor_A": "f^(-1/2)",
            "Einstein_scalar_kinetic": str(kinetic),
        },
        "matter_coupling": {
            "alpha(phi)=d_ln_A/d_chi": str(coupling),
            "alpha_0_at_phi_infinity_zero": str(alpha_zero),
            "beta_0=d_alpha/d_chi": str(beta_zero),
        },
        "Newtonian_prediction": {
            "G_cav_over_G_star": str(newton_factor),
            "linear_scalar_source_alpha_0_T": "0",
            "Poisson_equation": "laplacian(U)=4*pi*G_star*rho",
            "exterior_potential": "U=G_star*M/r",
            "status": "pass_on_declared_phi_infinity_zero_background",
        },
        "PPN_prediction": {
            "gamma_minus_one": str(gamma_minus_one),
            "gamma": "1",
            "beta_minus_one": str(beta_minus_one),
            "beta": "1",
            "preferred_frame_parameters": "0_for_covariant_massless_scalar_tensor_action",
            "status": "pass_on_declared_phi_infinity_zero_background",
        },
        "scope": (
            "massless scalar-tensor weak-field expansion about the exact phi_infinity=0 branch; "
            "it is not evidence that every compact-source boundary-value problem selects that branch"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _exact_scalar_free_branch_certificate() -> dict[str, Any]:
    phi = sp.Symbol("phi", real=True)
    g4 = sp.Rational(1, 2) + phi**2 / 100
    residuals = {
        "G4_at_zero_minus_EH_coefficient": sp.simplify(g4.subs(phi, 0) - sp.Rational(1, 2)),
        "G4_phi_at_zero": sp.diff(g4, phi).subs(phi, 0),
        "G2_at_constant_phi_zero": sp.Integer(0),
        "scalar_stress_at_constant_phi_zero": sp.Integer(0),
    }
    if any(value != 0 for value in residuals.values()):
        raise ValueError("exact scalar-free GR branch residual failed")
    body = {
        "background_and_boundary": "phi=0_everywhere_with_phi_infinity=0",
        "exact_field_equation_residuals": {key: str(value) for key, value in residuals.items()},
        "branch_result": (
            "every Einstein solution with universally Jordan-coupled matter is an exact candidate "
            "solution on phi=0"
        ),
        "vacuum_exterior": "Schwarzschild_is_exact_on_this_branch",
        "known_answer_formulas": {
            "perihelion_per_orbit": "6*pi*G_N*M/(a*(1-e^2)*c^2)",
            "light_deflection": "4*G_N*M/(b*c^2)",
            "one_way_Shapiro": "2*G_N*M/c^3*log(4*r_E*r_R/b^2)",
        },
        "branch_selection_warning": (
            "existence of the GR branch is not uniqueness for an arbitrary material source; "
            "nontrivial scalar branches require a source-specific boundary audit"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _synthetic_uniform_sphere_certificate(contract: dict[str, Any]) -> dict[str, Any]:
    compactness = Fraction(contract["dimensionless_compactness_upper"])
    beta_abs = Fraction(1, 50)
    radial_mode_squared = 3 * beta_abs * compactness
    if radial_mode_squared != Fraction(3, 50_000) or radial_mode_squared >= 1:
        raise ValueError("synthetic weak-source uniqueness bound failed")
    body = {
        "source_contract": contract,
        "linearized_static_scalar_equation_inside": (
            "laplacian(delta_chi)+4*pi*G_star*abs(beta_0)*rho*delta_chi=0"
        ),
        "uniform_sphere_radial_parameter": {
            "z_squared=3*abs(beta_0)*G_star*M/(R*c^2)": str(radial_mode_squared),
            "first_zero_mode_threshold": "z=pi/2",
            "certificate": "z^2<=3/50000<1<pi^2/4",
        },
        "linear_scalar_free_branch_unique": True,
        "status": "pass_synthetic_known_answer_only",
        "non_extension": (
            "does not certify the real Sun, nonspherical bodies, pressure/composition effects, "
            "strong-field bodies, nonlinear scalarization, or time-dependent initial data"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _calibration_certificate(reference: dict[str, Any]) -> dict[str, Any]:
    checks = reference.get("golden_checks", [])
    statuses = {str(item.get("name")): str(item.get("status")) for item in checks}
    required = {
        "schwarzschild_vacuum",
        "gr_ppn_recovery",
        "mercury_perihelion",
        "solar_limb_light_deflection",
        "shapiro_delay_geometry_control",
    }
    if set(statuses) != required or set(statuses.values()) != {"pass"}:
        raise ValueError("GR calibration known-answer checks changed")
    body = {
        "role": "solver_calibration_control_not_candidate_evidence",
        "reference_action_sha256": reference["formal_prerequisite"]["input_action_sha256"],
        "statuses": dict(sorted(statuses.items())),
        "numeric_outputs": {
            item["name"]: item["evidence"]
            for item in checks
            if item["name"] in {
                "mercury_perihelion",
                "solar_limb_light_deflection",
                "shapiro_delay_geometry_control",
            }
        },
        "candidate_inference_rule": (
            "candidate agreement follows only from its independently derived exact scalar-free "
            "branch and PPN certificate, never from reusing the GR control action hash"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _real_solar_admissibility(
    protocol: dict[str, Any], audit: dict[str, Any], source: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or audit.get("observational_dataset_opened") is not False
        or source.get("data_opened") is not False
    ):
        raise ValueError("Solar observational seal changed")
    evaluator_result = solar_known_answer_evaluator(
        {
            "candidate_id": target["seed_id"],
            "ordinal": 0,
            "correction_expression": "candidate_specific_conformal_G4",
            "data_eligibility": dict(ELIGIBILITY),
        },
        {
            "stage_name": "solar_known_answer_controls",
            "category": "observational",
            "attempt": 1,
            "input_lineage_sha256": target["formal_provenance_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        },
    )
    if (
        evaluator_result.get("decision") != "blocked"
        or evaluator_result.get("blocker") != "missing_exact_action_bound_solar_control_bundle"
    ):
        raise ValueError("current Solar evaluator unexpectedly authorized the candidate")
    reused_control = {
        "candidate_id": target["seed_id"],
        "ordinal": 0,
        "correction_expression": "candidate_specific_conformal_G4",
        "solar_control_provenance": {
            "bundle_id": GR_SOLAR_BUNDLE["bundle_id"],
            "bundle_binding_sha256": bundle_binding(GR_SOLAR_BUNDLE),
            "input_action_sha256": GR_SOLAR_BUNDLE["input_action_sha256"],
        },
        "data_eligibility": dict(ELIGIBILITY),
    }
    try:
        solar_known_answer_evaluator(
            reused_control,
            {
                "stage_name": "solar_known_answer_controls",
                "category": "observational",
                "attempt": 1,
                "input_lineage_sha256": target["formal_provenance_sha256"],
                "data_eligibility": dict(ELIGIBILITY),
            },
        )
    except ValueError as exc:
        gr_reuse_negative = str(exc)
    else:
        raise ValueError("GR known-answer control was reusable by a discovery candidate")
    if gr_reuse_negative != "Solar known-answer bundle cannot be attached to a discovery candidate":
        raise ValueError("GR known-answer reuse negative control changed")
    readiness = source.get("readiness", {})
    blockers = [
        "current Solar evaluator has no registered bundle for the discovery action hash",
        "real source stress/pressure/composition and scalar-branch uniqueness are not bound",
        "training-only initial state, nuisance, likelihood, and stopping rule are not frozen",
    ]
    blockers.extend(str(item) for item in readiness.get("blockers", []))
    body = {
        "admissible": False,
        "decision": "blocked",
        "first_missing_premise": "registered_candidate_specific_action_bound_Solar_bundle",
        "current_evaluator_result": evaluator_result,
        "GR_control_bundle_reuse_negative": {
            "rejected": True,
            "reason": gr_reuse_negative,
        },
        "protocol_status": protocol["status"],
        "protocol_audit_status": audit["status"],
        "source_registration_status": source["status"],
        "candidate_use_authorized": source.get("candidate_use_authorized"),
        "dataset_ready": readiness.get("dataset_ready"),
        "primary_files_downloaded": readiness.get("primary_files_downloaded"),
        "blockers": blockers,
        "observational_inputs_opened_by_this_audit": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_conformal_solar_prediction_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_synthetic_source(config["synthetic_source_contract"])
    formal = _load_bound(root, config["formal_predecessor"], content=True)
    action = _load_bound(root, config["action_predecessor"], content=True)
    calibration_reference = _load_bound(root, config["calibration_reference"])
    contract = config["solar_contract"]
    protocol = _load_bound(root, contract["protocol"])
    protocol_audit = _load_bound(root, contract["protocol_audit"])
    source_registration = _load_bound(root, contract["source_registration"])
    if _file_sha(root / contract["evaluator_source"]["path"]) != contract[
        "evaluator_source"
    ]["file_sha256"]:
        raise ValueError("Solar evaluator source hash mismatch")
    evaluator_descriptor = _load_bound(root, contract["evaluator_descriptor"])
    if (
        evaluator_descriptor.get("artifact_sha256")
        != contract["evaluator_source"]["file_sha256"]
        or evaluator_descriptor.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("Solar evaluator descriptor binding mismatch")
    target = config["target"]
    formal_records = [
        item for item in formal["candidate_records"] if item.get("seed_id") == target["seed_id"]
    ]
    action_records = [
        item for item in action["candidate_records"] if item.get("seed_id") == target["seed_id"]
    ]
    if len(formal_records) != 1 or len(action_records) != 1:
        raise ValueError("G4 target is not unique in bound predecessors")
    _validate_target(formal_records[0], action_records[0], target)
    coupling = _coupling_and_ppn_certificate()
    branch = _exact_scalar_free_branch_certificate()
    synthetic = _synthetic_uniform_sphere_certificate(config["synthetic_source_contract"])
    calibration = _calibration_certificate(calibration_reference)
    admissibility = _real_solar_admissibility(
        protocol, protocol_audit, source_registration, target
    )
    provenance_body = {
        "formal_predecessor_content_sha256": config["formal_predecessor"]["content_sha256"],
        "formal_provenance_sha256": target["formal_provenance_sha256"],
        "formal_certificate_sha256": target["formal_certificate_sha256"],
        "action_predecessor_content_sha256": config["action_predecessor"]["content_sha256"],
        "action_provenance_sha256": target["action_provenance_sha256"],
        "action_sha256": target["action_sha256"],
        "coupling_PPN_sha256": coupling["content_sha256"],
        "scalar_free_branch_sha256": branch["content_sha256"],
        "synthetic_source_sha256": synthetic["content_sha256"],
        "calibration_sha256": calibration["content_sha256"],
        "real_solar_admissibility_sha256": admissibility["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    record = {
        "seed_id": target["seed_id"],
        "action_sha256": target["action_sha256"],
        "decision": "blocked",
        "candidate_analytic_prediction_status": "pass_on_declared_scalar_free_background",
        "coupling_and_PPN_certificate": coupling,
        "exact_scalar_free_branch_certificate": branch,
        "synthetic_uniform_sphere_certificate": synthetic,
        "GR_calibration_control": calibration,
        "real_solar_admissibility": admissibility,
        "gate_ledger": {
            "exact_action_and_formal_provenance": {"status": "pass"},
            "declared_phi_infinity_zero_background": {"status": "pass"},
            "Newtonian_limit": {"status": "pass"},
            "PPN_gamma_beta": {"status": "pass"},
            "exact_GR_scalar_free_branch": {"status": "pass"},
            "synthetic_known_answer_calibration": {"status": "pass"},
            "real_source_branch_uniqueness": {"status": "blocked"},
            "registered_direct_observable_Solar_bundle": {"status": "blocked"},
        },
        "first_missing_premise": "registered_candidate_specific_action_bound_Solar_bundle",
        "necessary_condition_rejection_found": False,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "solar_bundle": {
            "analytic_known_answer_bundle_generated": True,
            "real_observational_bundle_generated": False,
            "real_observational_bundle_admissible": False,
            "status": "blocked_before_data_opening",
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "formal_predecessor": config["formal_predecessor"],
            "action_predecessor": config["action_predecessor"],
            "calibration_reference": config["calibration_reference"],
            "solar_contract": contract,
        },
        "target_seed_count": 1,
        "decision_counts": {"blocked": 1},
        "gate_status_counts": {"pass": 6, "blocked": 2},
        "calibration_control_status_counts": {"pass": 5},
        "candidate_records": [record],
        "analytic_known_answer_bundle_count": 1,
        "real_solar_bundle_count": 0,
        "real_solar_bundle_admissible_count": 0,
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The exact candidate predicts the GR Newtonian and PPN values on its declared "
            "phi_infinity=0 scalar-free branch, independently of the GR solver calibration. "
            "A real Solar bundle is not admissible because the discovery action has no registered "
            "Solar bundle, real-source branch uniqueness is unproved, and direct records remain sealed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
