from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .action_ir import compile_action_file
from .adm_ir import compile_adm_ir, write_adm_ir
from .covariant_variation import vary_proca_action_file, vary_scalar_action_file
from .dirac_ir import compile_dirac_ir, write_dirac_ir
from .formal_backend import run_formal_control_suite
from .hamiltonian_ir import compile_physical_hamiltonian_ir, write_physical_hamiltonian_ir
from .higher_jet_ir import compile_higher_jet_auxiliary_ir, write_higher_jet_auxiliary_ir
from .legendre_ir import compile_legendre_ir, write_legendre_ir
from .principal_ir import compile_physical_principal_ir, write_physical_principal_ir
from .q_operator_ir import compile_q_operator_ir, write_q_operator_ir
from .q_variation_ir import compile_q_variation_ir, write_q_variation_ir
from .stability_ir import compile_stability_ir, write_stability_ir
from .static_dictionary import compile_static_dictionary_ir, write_static_artifact
from .x_operator_ir import compile_x_operator_ir, write_x_operator_ir


def _gate(status: str, evidence: list[str], scope: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "evidence": evidence, "scope": scope, **extra}


def _family(term_ids: set[str]) -> str:
    if term_ids == {"EH_R"}:
        return "einstein_hilbert"
    if term_ids == {"EH_R", "SCALAR_X", "SCALAR_MASS"}:
        return "canonical_scalar_gravity"
    if term_ids == {"EH_R", "PROCA_F2", "PROCA_MASS"}:
        return "proca_gravity"
    if term_ids == {"EH_R", "SCALAR_X", "HORNDESKI_L4_LINEAR_X"}:
        return "quartic_horndeski"
    aether = {
        "EH_R",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "UNIT_VECTOR_CONSTRAINT",
    }
    if term_ids == aether:
        return "einstein_aether"
    return "unsupported"


def analyze_action_health(
    spec_path: str | Path,
    grammar_path: str | Path,
    contract_path: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path,
    formal_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_ir = compile_action_file(spec_path, grammar_path, contract_path)
    ir_path = output / "action-ir.json"
    ir_path.write_text(json.dumps(action_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not action_ir["valid"]:
        report = {
            "schema_version": "sigma-action-health-1.0",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "reject",
            "promotion_allowed": False,
            "reason": "action IR failed before formal execution",
            "action_ir": str(ir_path),
            "errors": action_ir["errors"],
        }
        path = output / "action-health.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**report, "report_path": str(path)}

    terms = {item["id"] for item in action_ir["canonical"]["terms"]}
    family = _family(terms)
    generated_static_dictionary_ir = compile_static_dictionary_ir(action_ir)
    static_dictionary_ir_path = write_static_artifact(
        generated_static_dictionary_ir, output / "static-dictionary-ir.json"
    )
    generated_q_operator_ir = compile_q_operator_ir(action_ir)
    q_operator_ir_path = write_q_operator_ir(
        generated_q_operator_ir, output / "q-operator-ir.json"
    )
    generated_q_variation_ir = compile_q_variation_ir(action_ir)
    q_variation_ir_path = write_q_variation_ir(
        generated_q_variation_ir, output / "q-variation-ir.json"
    )
    generated_higher_jet_ir = compile_higher_jet_auxiliary_ir(action_ir)
    higher_jet_ir_path = write_higher_jet_auxiliary_ir(
        generated_higher_jet_ir, output / "higher-jet-auxiliary-ir.json"
    )
    generated_x_operator_ir = compile_x_operator_ir(action_ir)
    x_operator_ir_path = write_x_operator_ir(
        generated_x_operator_ir, output / "x-operator-ir.json"
    )
    formal = formal_report or run_formal_control_suite(contract_path, project_root)
    checks = {item["name"]: item for item in formal["checks"]}
    control_status = {name: item.get("status") == "pass" for name, item in checks.items()}
    generated_adm_ir = compile_adm_ir(action_ir, control_status)
    adm_ir_path = write_adm_ir(generated_adm_ir, output / "adm-ir.json")
    generated_legendre_ir = compile_legendre_ir(action_ir, generated_adm_ir)
    legendre_ir_path = write_legendre_ir(generated_legendre_ir, output / "legendre-ir.json")
    generated_dirac_ir = compile_dirac_ir(
        action_ir, generated_adm_ir, generated_legendre_ir, control_status
    )
    dirac_ir_path = write_dirac_ir(generated_dirac_ir, output / "dirac-ir.json")
    generated_stability_ir = compile_stability_ir(
        action_ir, generated_dirac_ir, control_status
    )
    stability_ir_path = write_stability_ir(
        generated_stability_ir, output / "stability-ir.json"
    )
    generated_principal_ir = compile_physical_principal_ir(
        action_ir, generated_dirac_ir, generated_stability_ir
    )
    principal_ir_path = write_physical_principal_ir(
        generated_principal_ir, output / "principal-ir.json"
    )
    generated_hamiltonian_ir = compile_physical_hamiltonian_ir(
        action_ir,
        generated_dirac_ir,
        generated_stability_ir,
        generated_principal_ir,
    )
    hamiltonian_ir_path = write_physical_hamiltonian_ir(
        generated_hamiltonian_ir, output / "hamiltonian-ir.json"
    )

    def checks_pass(*names: str) -> bool:
        return all(checks.get(name, {}).get("status") == "pass" for name in names)

    common_variation = [
        "cadabra_einstein_hilbert_metric_variation",
        "cadabra_nonlinear_contracted_bianchi",
    ]
    gravity_adm = [
        "einstein_hilbert_linearized_adm",
        "canonical_metric_diffeomorphism_algebra",
        "canonical_metric_dewitt_kinetic_covariance",
        "spatial_curvature_density_diffeomorphism_covariance",
        "cadabra_adm_spatial_curvature_variation",
        "nonlinear_adm_hamiltonian_constraint_algebra",
    ]
    gates: dict[str, dict[str, Any]] = {
        "field_contract": _gate(
            "pass",
            ["covariant_field_contract"],
            "exact action-IR and universal-matter-coupling validation",
        ),
        "static_dictionary_derivation": _gate(
            "pass" if generated_static_dictionary_ir["status"] == "pass" else "unresolved",
            [str(static_dictionary_ir_path)],
            "action-hash-bound exact static zero-shift tensor reductions, including the unit-Aether acceleration dictionary and diagnostic-only baryonic z",
            static_dictionary_ir_sha256=generated_static_dictionary_ir.get("content_sha256"),
            legacy_generator_dictionary=generated_static_dictionary_ir.get(
                "legacy_generator_dictionary", {}
            ),
        ),
        "higher_jet_regularity": _gate(
            generated_q_operator_ir["status"],
            [
                str(q_operator_ir_path),
                str(higher_jet_ir_path),
                "projected_aether_q_aligned_auxiliary_dirac",
                "projected_aether_q_constant_tilt_root_audit",
            ],
            generated_q_operator_ir.get("proof_scope", "Q_a_u is not present"),
            applicable=generated_q_operator_ir.get("applicable", False),
            q_operator_ir_sha256=generated_q_operator_ir.get("content_sha256"),
            conclusion=generated_q_operator_ir.get("conclusion"),
            auxiliary_lift_status=generated_higher_jet_ir.get("status"),
            auxiliary_lift_equivalence_status=generated_higher_jet_ir.get(
                "equivalence_certificate", {}
            ).get("status"),
            higher_jet_auxiliary_ir_sha256=generated_higher_jet_ir.get("content_sha256"),
            aligned_auxiliary_dirac_status=checks.get(
                "projected_aether_q_aligned_auxiliary_dirac", {}
            ).get("status"),
            constant_tilt_root_audit_status=checks.get(
                "projected_aether_q_constant_tilt_root_audit", {}
            ).get("status"),
        ),
        "nonlinear_x_regularity": _gate(
            generated_x_operator_ir["status"],
            [str(x_operator_ir_path), "nonlinear_aether_acceleration_global_convexity"],
            generated_x_operator_ir.get("proof_scope", "nonlinear X_a is not present"),
            applicable=generated_x_operator_ir.get("applicable", False),
            x_operator_ir_sha256=generated_x_operator_ir.get("content_sha256"),
            unbounded_characteristic_speed=generated_x_operator_ir.get(
                "unbounded_characteristic_speed"
            ),
            conclusion=generated_x_operator_ir.get("conclusion"),
        ),
        "adm_decomposition": _gate(
            "pass" if generated_adm_ir["status"] == "pass" else "unresolved",
            sorted(
                {
                    control
                    for term in generated_adm_ir.get("terms", [])
                    for control in term["required_controls"]
                }
            ),
            "action-hash-bound termwise 3+1 decomposition, boundary contract, velocity channels, and lapse/shift plus field-specific primary/secondary constraint seeds",
            adm_ir=str(adm_ir_path),
            adm_ir_sha256=generated_adm_ir.get("content_sha256"),
            templates_complete=generated_adm_ir.get("term_templates_complete", False),
        ),
        "legendre_map": _gate(
            "pass" if generated_legendre_ir["status"] == "pass" else "unresolved",
            [str(legendre_ir_path)],
            "action-hash-bound exact local kinetic Hessian, generic rank, singular strata, and kinetic primary-constraint seeds in the bounded grammar",
            legendre_ir=str(legendre_ir_path),
            legendre_ir_sha256=generated_legendre_ir.get("content_sha256"),
            generic_rank=generated_legendre_ir.get("generic_hessian_rank"),
            generic_nullity=generated_legendre_ir.get("generic_hessian_nullity"),
            regularity_factors=generated_legendre_ir.get("regularity_factors", []),
            proof_scope=generated_legendre_ir.get("proof_scope"),
        ),
        "generated_dirac_closure": _gate(
            "pass" if generated_dirac_ir["status"] == "pass" else "unresolved",
            [str(dirac_ir_path)],
            "hash-bound exact local canonical transform plus distributed D-D/D-H/H-H closure and constraint-surface degree count only where executable controls cover the same action specialization",
            dirac_ir=str(dirac_ir_path),
            dirac_ir_sha256=generated_dirac_ir.get("content_sha256"),
            family=generated_dirac_ir.get("distributed_constraint_closure", {}).get("family"),
            physical_dof=generated_dirac_ir.get("distributed_constraint_closure", {})
            .get("constraint_surface_rank", {})
            .get("physical_dof"),
            proof_scope=generated_dirac_ir.get("proof_scope"),
        ),
    }
    generated_variation: dict[str, Any] | None = None
    total_dof: int | None = None

    if family == "einstein_hilbert":
        variation_names = common_variation
        gates["covariant_variation"] = _gate(
            "pass" if checks_pass(*variation_names) else "reject",
            variation_names,
            "nonlinear Einstein-Hilbert bulk variation with boundary term tracked",
        )
        gates["covariant_identity"] = _gate(
            "pass" if checks_pass("cadabra_nonlinear_contracted_bianchi") else "reject",
            ["cadabra_nonlinear_contracted_bianchi"],
            "nonlinear contracted Bianchi identity",
        )
        total_dof = 2
        gates["adm_dirac"] = _gate(
            "pass" if checks_pass(*gravity_adm) else "reject",
            gravity_adm,
            "nonlinear pure-GR D-D and H-H closure plus kinetic D-H covariance, curvature variation, primary constraints, and complete linearized two-mode count",
            physical_dof=total_dof,
        )
    elif family == "canonical_scalar_gravity":
        generated_variation = vary_scalar_action_file(
            spec_path,
            grammar_path,
            contract_path,
            output / "field-variation",
            project_root=project_root,
        )
        variation_names = common_variation + [
            "cadabra_canonical_scalar_metric_variation",
            "cadabra_canonical_scalar_variation",
        ]
        variation_pass = generated_variation["status"] == "pass" and checks_pass(*variation_names)
        gates["covariant_variation"] = _gate(
            "pass" if variation_pass else "reject",
            variation_names + ["generated_scalar_action_ir_variation"],
            "combined EH metric, scalar metric, and generated scalar-field variations",
        )
        identity_names = [
            "cadabra_nonlinear_contracted_bianchi",
            "canonical_scalar_noether_identity",
        ]
        gates["covariant_identity"] = _gate(
            "pass" if checks_pass(*identity_names) else "reject",
            identity_names,
            "combined diffeomorphism identity from div(G)=0 and div(T_phi)=E_phi grad(phi)",
        )
        total_dof = 3
        gates["adm_dirac"] = _gate(
            "pass" if checks_pass(*gravity_adm, "canonical_scalar") else "reject",
            [*gravity_adm, "canonical_scalar"],
            "nonlinear pure-GR constraint algebra plus linearized minimally coupled scalar direct-sum control",
            physical_dof=total_dof,
        )
    elif family == "proca_gravity":
        generated_variation = vary_proca_action_file(
            spec_path,
            grammar_path,
            contract_path,
            output / "field-variation",
            project_root=project_root,
        )
        variation_names = common_variation + [
            "cadabra_proca_metric_variation",
            "cadabra_proca_variation",
        ]
        variation_pass = generated_variation["status"] == "pass" and checks_pass(*variation_names)
        gates["covariant_variation"] = _gate(
            "pass" if variation_pass else "reject",
            variation_names + ["generated_proca_action_ir_variation"],
            "combined EH metric, Proca metric, and generated vector-field variations",
        )
        identity_names = [
            "cadabra_nonlinear_contracted_bianchi",
            "proca_divergence_identity",
            "proca_stress_noether_identity",
            "proca_curved_background_noether_identity",
        ]
        gates["covariant_identity"] = _gate(
            "pass" if checks_pass(*identity_names) else "reject",
            identity_names,
            "Einstein Bianchi identity plus Proca divergence constraint and exact stress Noether residuals on flat, FLRW, and static-spherical controls",
        )
        total_dof = 5
        gates["adm_dirac"] = _gate(
            "pass" if checks_pass(*gravity_adm, "proca_adm_dirac") else "reject",
            [*gravity_adm, "proca_adm_dirac"],
            "nonlinear pure-GR constraint algebra plus exact flat-background Proca Dirac closure",
            physical_dof=total_dof,
        )
    elif family == "einstein_aether":
        variation_names = common_variation + [
            "cadabra_einstein_aether_vector_variation",
            "cadabra_einstein_aether_metric_variation",
        ]
        gates["covariant_variation"] = _gate(
            "pass" if checks_pass(*variation_names) else "reject",
            variation_names,
            "EH metric plus complete K1..K4 vector, multiplier, and connection-dependent metric variations",
        )
        identity_names = [
            "cadabra_nonlinear_contracted_bianchi",
            "einstein_aether_arbitrary_background_4d_noether",
            "einstein_aether_flrw_variation_noether",
            "einstein_aether_inhomogeneous_2d_noether",
            "einstein_aether_inhomogeneous_4d_numeric_noether",
        ]
        gates["covariant_identity"] = _gate(
            "pass" if checks_pass(*identity_names) else "unresolved",
            identity_names,
            "exact arbitrary-background fixed-covector metric-vector-multiplier identity, independently corroborated by nonlinear homogeneous 4D, exact arbitrary-jet 2D, and unrestricted numerical arbitrary-jet 4D controls",
        )
        aether_adm_names = [
            "einstein_aether_modes",
            "einstein_aether_arbitrary_background_4d_noether",
            "einstein_aether_flrw_variation_noether",
            "einstein_aether_adm_kinetic_hessian",
            "einstein_aether_generic_3plus1_legendre",
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "einstein_aether_generic_dh_covariance",
            "einstein_aether_generic_hh_deformation_kinematics",
            "einstein_aether_global_tilt_legendre_strata",
            "einstein_aether_covariant_arbitrary_background_hyperbolicity",
            "einstein_aether_coupled_unit_normal",
            "einstein_aether_spatial_diffeomorphism_algebra",
            "unit_timelike_vector_dirac_chain",
            "regular_holonomic_multiplier_dirac_theorem",
            "maxwell_unit_aether_nonlinear_hamiltonian",
        ]
        total_dof = 5
        gates["adm_dirac"] = _gate(
            "pass" if checks_pass(*aether_adm_names) else "unresolved",
            aether_adm_names,
            "exact generic K1..K4 D-D/D-H/H-H hypersurface-deformation algebra and five-mode Dirac count on regular positive-unit-branch patches, independently anchored by arbitrary-background covariance; the global unit-timelike tilt determinant is spin-factorized and restricts evolution to common noncharacteristic slices, while coupling-boundary rank, boundary charges, and reduced Hamiltonian stability remain separate",
            physical_dof=total_dof,
            regularity_scope="aligned kinetic factors nonzero and, at arbitrary tilt x=A_i A^i, F_s(x)=-K_s+(N_s-K_s)x nonzero for spin 2, 1, and 0",
        )
    elif family == "quartic_horndeski":
        scalar_variation_names = [
            "cadabra_canonical_scalar_metric_variation",
            "cadabra_canonical_scalar_variation",
            "quartic_horndeski_scalar_covariant_variation",
        ]
        scalar_variation_pass = checks_pass(*scalar_variation_names)
        metric_noether_names = ["quartic_horndeski_metric_variation_and_noether"]
        metric_noether_pass = checks_pass(*metric_noether_names)
        gates["covariant_variation"] = _gate(
            "pass" if scalar_variation_pass and metric_noether_pass else "unresolved",
            [*common_variation, *scalar_variation_names, *metric_noether_names],
            "action-hash-bound scalar and metric Euler variations, including the full Palatini metric contribution and second-order scalar reduction",
            scalar_variation_status="pass" if scalar_variation_pass else "reject",
            metric_variation_status="pass" if metric_noether_pass else "unresolved",
        )
        gates["covariant_identity"] = _gate(
            "pass"
            if checks_pass(
                "cadabra_nonlinear_contracted_bianchi",
                "canonical_scalar_noether_identity",
                "quartic_horndeski_scalar_covariant_variation",
                "quartic_horndeski_metric_variation_and_noether",
            )
            else "unresolved",
            [
                "cadabra_nonlinear_contracted_bianchi",
                "canonical_scalar_noether_identity",
                "quartic_horndeski_scalar_covariant_variation",
                "quartic_horndeski_metric_variation_and_noether",
            ],
            "source-bound arbitrary-background metric-scalar Euler Noether coefficient, covariant action-density divergence, and nonlinear lapse-FLRW corroboration",
        )
        horndeski_dirac_names = [
            "quartic_horndeski_covariant_adm_degeneracy",
            "quartic_horndeski_unitary_flrw_dirac_chain",
            "quartic_horndeski_unitary_distributed_dirac_closure",
        ]
        horndeski_dirac_pass = (
            checks_pass(*horndeski_dirac_names)
            and generated_legendre_ir["status"] == "pass"
            and generated_dirac_ir["status"] == "pass"
        )
        total_dof = 3
        gates["adm_dirac"] = _gate(
            "pass" if horndeski_dirac_pass else "unresolved",
            [
                *horndeski_dirac_names,
                str(legendre_ir_path),
                str(dirac_ir_path),
            ],
            "the action-hash-bound covariant-to-ADM cancellation, seven-channel Legendre Hessian, exact 3D metric+lapse spatial cotangent lift, secondary-density covariance, regular second-class lapse-pair theorem, and curved-FLRW nonempty-patch witness give three physical modes; global lapse-Hessian invertibility and boundary zero modes remain excluded from the regular patch",
            physical_dof=total_dof if horndeski_dirac_pass else None,
            regularity_scope=(
                "unitary-gauge patches where the boundary-condition-dependent distributed lapse "
                "Hessian Delta_N is invertible and the metric Legendre factor is nonzero"
            ),
        )
    else:
        for gate_name in ("covariant_variation", "covariant_identity", "adm_dirac"):
            gates[gate_name] = _gate("unresolved", [], "term family has no formal backend adapter")
        gates["covariant_variation"]["evidence"] = [str(q_variation_ir_path)]
        gates["covariant_variation"]["fixed_metric_vector_status"] = (
            generated_q_variation_ir.get("fixed_metric_vector_variation", {}).get("status")
        )
        gates["covariant_variation"]["metric_variation_status"] = (
            generated_q_variation_ir.get("metric_variation", {}).get("status")
        )

    condition_certificate = generated_stability_ir.get("condition_certificate", {})
    gates["parameter_domain"] = _gate(
        condition_certificate.get("status", "reject"),
        [str(stability_ir_path)],
        "hash-bound implication from the frozen parameter domain to every required effective coefficient sign and nondegeneracy condition",
        stability_ir_sha256=generated_stability_ir.get("content_sha256"),
        conditions=condition_certificate.get("conditions", []),
        pointwise_status=condition_certificate.get("pointwise_status"),
        background_domain=generated_stability_ir.get("background_domain"),
        background_domain_preservation=condition_certificate.get(
            "background_domain_preservation"
        ),
    )
    if family == "quartic_horndeski":
        gates["parameter_domain"]["evidence"].extend(
            [
                "quartic_horndeski_global_timelike_gradient_no_go",
                "quartic_horndeski_flrw_background_domain_crossing",
            ]
        )
        gates["parameter_domain"]["background_domain_required"] = (
            "A_star^2 < M_Pl^2/abs(alpha); unrestricted nonlinear FLRW evolution does not "
            "preserve this regular EFT patch, so any restricted solution class or stopping "
            "boundary requires separate justification"
        )
        gates["parameter_domain"]["global_all_timelike_amplitudes"] = "reject"
    generated_hamiltonian = generated_stability_ir.get("physical_hamiltonian", {})
    gates["hamiltonian_stability"] = _gate(
        generated_hamiltonian_ir.get("status", "reject"),
        generated_hamiltonian.get("required_controls", []) + [str(hamiltonian_ir_path)],
        generated_hamiltonian_ir.get(
            "proof_scope", "physical reduced Hamiltonian not derived"
        ),
        missing_or_failed_controls=generated_hamiltonian.get(
            "missing_or_failed_controls", []
        ),
        hamiltonian_ir_sha256=generated_hamiltonian_ir.get("content_sha256"),
        generic_nonlinear_total_energy=generated_hamiltonian_ir.get(
            "generic_nonlinear_total_energy", {}
        ),
    )
    generated_principal = generated_stability_ir.get("principal_symbol", {})
    gates["principal_symbol"] = _gate(
        generated_principal_ir.get("status", "reject"),
        generated_principal.get("required_controls", []) + [str(principal_ir_path)],
        generated_principal_ir.get(
            "proof_scope", "candidate/background principal extraction incomplete"
        ),
        missing_or_failed_controls=generated_principal.get("missing_or_failed_controls", []),
        characteristic_speed_squared=generated_principal_ir.get(
            "characteristic_speed_squared", {}
        ),
        principal_ir_sha256=generated_principal_ir.get("content_sha256"),
    )

    required = [
        "field_contract",
        "static_dictionary_derivation",
        "higher_jet_regularity",
        "nonlinear_x_regularity",
        "adm_decomposition",
        "legendre_map",
        "generated_dirac_closure",
        "parameter_domain",
        "covariant_variation",
        "covariant_identity",
        "adm_dirac",
        "hamiltonian_stability",
        "principal_symbol",
    ]
    all_pass = all(gates[name]["status"] == "pass" for name in required)
    discovery_blockers: list[str] = []
    if action_ir["canonical"]["source_role"] != "known_answer_control":
        discovery_blockers.append(
            "candidate-specific background solution and generator-expression static matching"
        )
    promotion_allowed = all_pass and not discovery_blockers
    any_reject = any(gates[name]["status"] == "reject" for name in required)
    status = (
        "pass"
        if promotion_allowed
        else ("control_pass" if all_pass else ("reject" if any_reject else "unresolved"))
    )
    report = {
        "schema_version": "sigma-action-health-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "promotion_allowed": promotion_allowed,
        "family": family,
        "input_action_sha256": action_ir["content_sha256"],
        "action_ir": str(ir_path),
        "physical_dof": total_dof,
        "gates": gates,
        "discovery_blockers": discovery_blockers,
        "generated_variation": generated_variation,
        "generated_static_dictionary_ir": {
            "path": str(static_dictionary_ir_path),
            "status": generated_static_dictionary_ir["status"],
            "content_sha256": generated_static_dictionary_ir.get("content_sha256"),
            "aether_x_status": generated_static_dictionary_ir.get(
                "legacy_generator_dictionary", {}
            ).get("x", {}).get("status"),
            "q_status": generated_static_dictionary_ir.get(
                "legacy_generator_dictionary", {}
            ).get("q", {}).get("status"),
            "z_status": generated_static_dictionary_ir.get(
                "legacy_generator_dictionary", {}
            ).get("z", {}).get("generator_status"),
            "proof_scope": generated_static_dictionary_ir.get("proof_scope"),
        },
        "generated_q_operator_ir": {
            "path": str(q_operator_ir_path),
            "status": generated_q_operator_ir["status"],
            "content_sha256": generated_q_operator_ir.get("content_sha256"),
            "applicable": generated_q_operator_ir.get("applicable", False),
            "rank_certificate": generated_q_operator_ir.get("rank_certificate"),
            "conclusion": generated_q_operator_ir.get("conclusion"),
            "proof_scope": generated_q_operator_ir.get("proof_scope"),
        },
        "generated_q_variation_ir": {
            "path": str(q_variation_ir_path),
            "status": generated_q_variation_ir["status"],
            "content_sha256": generated_q_variation_ir.get("content_sha256"),
            "fixed_metric_vector_status": generated_q_variation_ir.get(
                "fixed_metric_vector_variation", {}
            ).get("status"),
            "metric_variation_status": generated_q_variation_ir.get(
                "metric_variation", {}
            ).get("status"),
            "proof_scope": generated_q_variation_ir.get("proof_scope"),
        },
        "generated_higher_jet_auxiliary_ir": {
            "path": str(higher_jet_ir_path),
            "status": generated_higher_jet_ir["status"],
            "content_sha256": generated_higher_jet_ir.get("content_sha256"),
            "applicable": generated_higher_jet_ir.get("applicable", False),
            "equivalence_status": generated_higher_jet_ir.get(
                "equivalence_certificate", {}
            ).get("status"),
            "proof_scope": generated_higher_jet_ir.get("proof_scope"),
        },
        "generated_x_operator_ir": {
            "path": str(x_operator_ir_path),
            "status": generated_x_operator_ir["status"],
            "content_sha256": generated_x_operator_ir.get("content_sha256"),
            "applicable": generated_x_operator_ir.get("applicable", False),
            "unbounded_characteristic_speed": generated_x_operator_ir.get(
                "unbounded_characteristic_speed"
            ),
            "proof_scope": generated_x_operator_ir.get("proof_scope"),
        },
        "generated_adm_ir": {
            "path": str(adm_ir_path),
            "status": generated_adm_ir["status"],
            "content_sha256": generated_adm_ir.get("content_sha256"),
            "proof_scope": generated_adm_ir.get("proof_scope"),
        },
        "generated_legendre_ir": {
            "path": str(legendre_ir_path),
            "status": generated_legendre_ir["status"],
            "content_sha256": generated_legendre_ir.get("content_sha256"),
            "generic_hessian_rank": generated_legendre_ir.get("generic_hessian_rank"),
            "generic_hessian_nullity": generated_legendre_ir.get("generic_hessian_nullity"),
            "proof_scope": generated_legendre_ir.get("proof_scope"),
        },
        "generated_dirac_ir": {
            "path": str(dirac_ir_path),
            "status": generated_dirac_ir["status"],
            "content_sha256": generated_dirac_ir.get("content_sha256"),
            "family": generated_dirac_ir.get("distributed_constraint_closure", {}).get("family"),
            "physical_dof": generated_dirac_ir.get("distributed_constraint_closure", {})
            .get("constraint_surface_rank", {})
            .get("physical_dof"),
            "proof_scope": generated_dirac_ir.get("proof_scope"),
        },
        "generated_stability_ir": {
            "path": str(stability_ir_path),
            "status": generated_stability_ir["status"],
            "content_sha256": generated_stability_ir.get("content_sha256"),
            "family": generated_stability_ir.get("family"),
            "condition_status": condition_certificate.get("status"),
            "physical_hamiltonian_status": generated_hamiltonian.get("status"),
            "principal_symbol_status": generated_principal.get("status"),
            "proof_scope": generated_stability_ir.get("proof_scope"),
        },
        "generated_principal_ir": {
            "path": str(principal_ir_path),
            "status": generated_principal_ir["status"],
            "content_sha256": generated_principal_ir.get("content_sha256"),
            "family": generated_principal_ir.get("family"),
            "retained_mode_count": generated_principal_ir.get(
                "gauge_reduction_certificate", {}
            ).get("retained_mode_count"),
            "characteristic_speed_squared": generated_principal_ir.get(
                "characteristic_speed_squared", {}
            ),
            "proof_scope": generated_principal_ir.get("proof_scope"),
        },
        "generated_hamiltonian_ir": {
            "path": str(hamiltonian_ir_path),
            "status": generated_hamiltonian_ir["status"],
            "content_sha256": generated_hamiltonian_ir.get("content_sha256"),
            "family": generated_hamiltonian_ir.get("family"),
            "physical_mode_count": generated_hamiltonian_ir.get("physical_mode_count"),
            "positivity_status": generated_hamiltonian_ir.get(
                "positivity_certificate", {}
            ).get("status"),
            "generic_nonlinear_total_energy": generated_hamiltonian_ir.get(
                "generic_nonlinear_total_energy", {}
            ),
            "proof_scope": generated_hamiltonian_ir.get("proof_scope"),
        },
        "observational_gates_unsealed": False,
        "interpretation": "A control pass validates only the declared action family and backgrounds. It is not observational evidence or a truth probability.",
    }
    path = output / "action-health.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_path": str(path)}
