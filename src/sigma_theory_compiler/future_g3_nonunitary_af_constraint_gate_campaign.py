from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski import generic_cubic_horndeski_bssn_hyperbolicity_control
from .promotion_orchestrator import ELIGIBILITY
from .scalar_tensor_pack import generic_g3_variation_noether_control

ARTIFACT_SCHEMA = "sigma-future-g3-nonunitary-af-constraint-gate-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_"
    "on_decaying_gradient_domain_in_nonunitary_formulation"
)


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
        if (
            value.get("content_sha256") != descriptor["content_sha256"]
            or _sha(body) != descriptor["content_sha256"]
        ):
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_source_bindings(root: Path, config: dict[str, Any]) -> None:
    if _file_sha(root / config["adapter_source"]["path"]) != config["adapter_source"][
        "file_sha256"
    ]:
        raise ValueError("campaign source hash mismatch")
    for label, descriptor in config["source_bindings"].items():
        if _file_sha(root / descriptor["path"]) != descriptor["file_sha256"]:
            raise ValueError(f"{label} source hash mismatch")


def _validate_formulation(formulation: dict[str, Any]) -> None:
    expected = {
        "gauge": "covariant_scalar_retained_BSSN_Bona_Masso",
        "scalar_is_coordinate": False,
        "unitary_Delta_N_used": False,
        "BSSN_m": "1",
        "BSSN_sigma": "1",
        "shift": "fixed_zero_on_reference_profile",
        "ordinary_ADM_lapse": "1+O_2(r^-1)",
        "evolved_scalar_variables": ["phi", "Pi", "D_i(phi)"],
    }
    if formulation != expected:
        raise ValueError("nonunitary formulation contract changed")


def _validate_method_control(method: dict[str, Any], config: dict[str, Any]) -> None:
    record = next(
        (item for item in method["candidate_records"] if item.get("family") == "G3"), None
    )
    expected = config["method_control_expectations"]
    if (
        record is None
        or record.get("decision") != "blocked"
        or record.get("provenance", {}).get("binding_sha256")
        != expected["record_binding_sha256"]
        or record.get("nonunitary_bypass_certificate", {}).get("content_sha256")
        != expected["certificate_content_sha256"]
        or record.get("nonunitary_bypass_certificate", {}).get("bypass_decision")
        != "pass_for_principal_formulation_only"
        or record.get("first_missing_premise")
        != "candidate_specific_asymptotically_flat_Einstein_constraint_solution"
    ):
        raise ValueError("nonunitary method control changed")


def _principal_certificate(
    record: dict[str, Any], beta: Fraction, formulation: dict[str, Any]
) -> dict[str, Any]:
    prior = record["principal_common_cone_certificate"]
    bounds = prior["uniform_bounds_on_zero_less_equal_X_less_equal_one_half"]
    length = Fraction(100)
    expected = {
        "P00_upper": "-1",
        "common_time_covector_margin": "1",
        "spatial_eigenvalue_lower": str(1 - beta**2 / 4),
        "time_space_norm_upper_squared": str(2 * beta**2 / length**2),
        "characteristic_discriminant_lower": str(1 - beta**2 / 4),
        "BSSN_sigma": "1",
        "slicing_cone_polynomial_upper": bounds["slicing_cone_polynomial_upper"],
        "slicing_cone_separation": bounds["slicing_cone_separation"],
    }
    if bounds != expected:
        raise ValueError("candidate principal bounds changed")
    x = sp.Symbol("X", nonnegative=True)
    p00 = -(1 + 3 * sp.Rational(beta.numerator, beta.denominator) ** 2 * x**2)
    spatial = 1 - sp.Rational(beta.numerator, beta.denominator) ** 2 * x**2
    if (sp.simplify(p00.subs(x, 0)), sp.simplify(spatial.subs(x, 0))) != (-1, 1):
        raise ValueError("nonunitary scalar principal limit is degenerate")
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["action_sha256"],
        "beta": str(beta),
        "formulation": formulation,
        "direct_candidate_specialization": True,
        "family_label_used_as_equivalence_proof": False,
        "unitary_lapse_multiplier_or_inverse_used": False,
        "AF_profile_domain_sha256": record["AF_transition_domain_certificate"][
            "content_sha256"
        ],
        "effective_scalar_principal": {
            "P00": "-(1+3*beta^2*X^2)",
            "P0i": "-2*beta*D_i(Pi)",
            "Pij": "(1-beta^2*X^2)*delta_ij",
            "X_domain": "0<=X<=1/2",
            "direction_domain": "all_unit_spatial_covectors_no_sampling",
        },
        "uniform_exact_bounds": bounds,
        "X_to_zero_limit": {
            "P00": "-1",
            "Pij_eigenvalue": "1",
            "physical_principal_degeneracy": False,
            "ordinary_lapse_coefficient_depends_on_X": False,
        },
        "BSSN_gauge_roots_squared": {
            "transverse": "1",
            "momentum": "1",
            "slicing": "2",
            "longitudinal": "1",
        },
        "status": "pass_candidate_bound_nonunitary_AF_principal_formulation",
        "scope": (
            "The scalar is retained as an evolved field and ordinary ADM lapse is used. The "
            "candidate-specific effective scalar cone and BSSN gauge cone stay separated on the "
            "complete decaying-gradient reference profile, including X->0. This is a principal-"
            "formulation result, not an Einstein-constraint solution or global theorem."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _constraint_gate(record: dict[str, Any], beta: Fraction) -> dict[str, Any]:
    v, d_v, h_00 = sp.symbols("v d_v H_00", real=True)
    beta_sp = sp.Rational(beta.numerator, beta.denominator)
    theta = -h_00
    x = v**2 / 2
    q_0 = v * h_00
    q_r = v * d_v
    grad_g_0 = beta_sp * q_0
    grad_g_r = beta_sp * q_r
    grad_g_dot_p = -grad_g_0 * v
    cubic_t_00 = sp.factor(
        -beta_sp * theta * v**2 - 2 * grad_g_0 * v - grad_g_dot_p
    )
    cubic_t_0r = sp.factor(-grad_g_r * v)
    canonical_t_00 = sp.factor(v**2 + x * (-1))
    hamiltonian_residual = sp.factor(-2 * (canonical_t_00 + cubic_t_00))
    expected_flux = -beta_sp * v**2 * d_v
    if (
        cubic_t_00 != 0
        or cubic_t_0r != expected_flux
        or canonical_t_00 != v**2 / 2
        or hamiltonian_residual != -v**2
    ):
        raise ValueError("exact flat-reference constraint reduction failed")
    body = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["action_sha256"],
        "beta": str(beta),
        "stress_formula_source": "exact_bound_generic_G3_Hilbert_stress_control",
        "nontrivial_decaying_gradient_reference_ansatz": {
            "role": "principal_reference_and_rejected_constraint_ansatz_only",
            "h_ij": "delta_ij",
            "K_ij": "0",
            "phi_at_t0": "0",
            "Pi": "v(r)=1/sqrt(1+(r/L)^4)",
            "X": "v(r)^2/2",
            "canonical_energy_density": "v(r)^2/2",
            "cubic_G3_energy_density": str(cubic_t_00),
            "cubic_G3_matter_flux_T_n_r": "-beta*v(r)^2*d_v_d_r",
            "cubic_G3_matter_flux_for_profile": (
                "2*beta*r^3/L^4/(1+(r/L)^4)^(5/2)"
            ),
            "Hamiltonian_constraint_residual_LHS_minus_2rho": str(
                hamiltonian_residual
            ),
            "Hamiltonian_constraint_residual_for_profile": (
                "-1/(1+(r/L)^4)"
            ),
            "zero_extrinsic_curvature_momentum_LHS": "0",
            "matter_flux_nonzero_for_every_r_greater_than_zero": True,
            "status": "reject_reference_ansatz_as_Einstein_constraint_solution",
            "theory_rejected": False,
        },
        "actual_AF_vacuum_constraint_solution": {
            "candidate_action_bound": True,
            "h_ij": "delta_ij",
            "K_ij": "0",
            "phi": "constant",
            "Pi": "0",
            "X": "0",
            "ADM_lapse": "1",
            "shift": "0",
            "G2": "0",
            "G3": "0",
            "scalar_stress_tensor": "0",
            "Hamiltonian_constraint_residual": "0",
            "momentum_constraint_residual": "0",
            "asymptotically_flat": True,
            "full_field_equation_reference": "Minkowski_plus_constant_scalar",
            "status": "pass_actual_AF_vacuum_constraint_solution",
            "overlap_with_nontrivial_transition_profile": "X=0_asymptotic_endpoint_only",
        },
        "candidate_nontrivial_AF_constraint_solution_available": False,
        "constraint_solution_counting_contract": (
            "The exact vacuum is counted only as a reference AF constraint solution. It does not "
            "solve, approximate, or continue the registered nonzero decaying-gradient datum."
        ),
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "The non-unitary formulation removes the scalar-clock chart obstruction but does not "
            "repair the rejected flat nontrivial datum. A coupled conformal/extrinsic-curvature "
            "constraint solution for that candidate-bound decaying-gradient domain is absent."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_nonunitary_af_constraint_gate_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source_bindings(root, config)
    _validate_formulation(config["alternative_formulation"])
    predecessor = _load_bound(root, config["bindings"]["predecessor"], content=True)
    method = _load_bound(root, config["bindings"]["method_control"], content=True)
    _validate_method_control(method, config)
    bssn_passed, bssn_evidence = generic_cubic_horndeski_bssn_hyperbolicity_control()
    stress_passed, stress_evidence = generic_g3_variation_noether_control()
    if (
        not bssn_passed
        or bssn_evidence["source_conditions"]["fixed_shift"] is not True
        or bssn_evidence["healthy_parameter_witness"]["assignment"]
        != {"m": "1", "sigma": "1", "c_phi_squared": "1"}
        or not stress_passed
        or stress_evidence["metric_stress_tensor"]
        != (
            "T_mu_nu=-G3_X theta p_mu p_nu-2 nabla_(mu(G3) p_nu)"
            "+g_mu_nu nabla_rho(G3)p^rho"
        )
    ):
        raise ValueError("reviewed nonunitary dependency control failed")
    records_by_id = {item["candidate_id"]: item for item in predecessor["candidate_records"]}
    records = []
    for target in config["targets"]:
        prior = records_by_id.get(target["candidate_id"])
        if (
            prior is None
            or prior["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or prior["AF_transition_domain_certificate"]["content_sha256"]
            != target["AF_transition_domain_content_sha256"]
            or prior["principal_common_cone_certificate"]["content_sha256"]
            != target["principal_content_sha256"]
            or prior["radial_profile_certificate"]["content_sha256"]
            != target["profile_content_sha256"]
            or prior["unitary_lapse_obstruction"]["content_sha256"]
            != target["unitary_obstruction_content_sha256"]
            or prior["decision"] != "blocked"
            or prior["theory_rejected"] is not False
        ):
            raise ValueError("target binding changed")
        beta = Fraction(target["beta"])
        principal = _principal_certificate(prior, beta, config["alternative_formulation"])
        constraint = _constraint_gate(prior, beta)
        gates = {
            "nonunitary_formulation_registration": {"status": "pass"},
            "candidate_bound_nonunitary_AF_principal": {"status": "pass"},
            "flat_nontrivial_reference_constraint_ansatz": {"status": "reject_ansatz"},
            "actual_AF_vacuum_constraint_reference": {"status": "pass_reference_only"},
            "candidate_nontrivial_AF_Einstein_constraint_solution": {"status": "blocked"},
            "global_hamiltonian_energy": {"status": "blocked"},
            "full_formal": {"status": "blocked"},
        }
        provenance_body = {
            "predecessor_content_sha256": config["bindings"]["predecessor"][
                "content_sha256"
            ],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "method_control_content_sha256": config["bindings"]["method_control"][
                "content_sha256"
            ],
            "action_sha256": prior["action_sha256"],
            "alternative_formulation_sha256": _sha(config["alternative_formulation"]),
            "principal_certificate_sha256": principal["content_sha256"],
            "constraint_gate_sha256": constraint["content_sha256"],
            "bssn_control_evidence_sha256": _sha(bssn_evidence),
            "g3_stress_control_evidence_sha256": _sha(stress_evidence),
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "unitary_obstruction_classification": "chart_obstruction_not_physical_principal_failure",
            "nonunitary_AF_principal_certificate": principal,
            "Einstein_constraint_gate": constraint,
            "gate_ledger": gates,
            "theory_rejected": False,
            "global_energy_pass": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if len(records) != 3:
        raise ValueError("expected exactly three future G3 targets")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": {
            **config["bindings"],
            **config["source_bindings"],
        },
        "dependency_control_evidence": {
            "generic_cubic_BSSN_sha256": _sha(bssn_evidence),
            "generic_G3_variation_stress_sha256": _sha(stress_evidence),
        },
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "nonunitary_formulation_registration_pass_count": 3,
        "nonunitary_AF_principal_pass_count": 3,
        "flat_nontrivial_reference_constraint_ansatz_reject_count": 3,
        "actual_AF_vacuum_constraint_reference_pass_count": 3,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "global_hamiltonian_energy_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: 3},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "synthetic_fixture_role": "none_used",
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "All three actions have a candidate-bound non-unitary BSSN/scalar principal "
            "formulation that stays regular through X->0, so the prior unitary-lapse inverse "
            "obstruction is a chart obstruction at the principal level. The registered nonzero "
            "flat decaying-gradient datum is still not an Einstein-constraint solution. Each "
            "action also admits the exact AF Minkowski/constant-scalar vacuum constraint "
            "reference, but that vacuum is not a continuation of the nontrivial datum and does "
            "not promote candidate-specific global energy or full formal completion."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_nonunitary_af_constraint_gate_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_nonunitary_af_constraint_gate_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
