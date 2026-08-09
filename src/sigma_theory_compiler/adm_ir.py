from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-adm-ir-1.0"


_TERM_TEMPLATES: dict[str, dict[str, Any]] = {
    "EH_R": {
        "bulk_basis": "N sqrt(h) (R3 + K_ij K^ij - K^2)",
        "velocity_channels": ["K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)"],
        "spatial_jets": ["R3[h]"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [
            "Einstein-Hilbert normal divergence removed by GHY completion or fixed-boundary/compact-support contract"
        ],
        "required_controls": [
            "cadabra_adm_spatial_curvature_variation",
            "nonlinear_adm_hamiltonian_constraint_algebra",
        ],
        "proof_scope": "exact ADM Einstein-Hilbert bulk decomposition after the declared boundary completion",
    },
    "SCALAR_X": {
        "bulk_basis": "N sqrt(h) (Pi_phi^2 - D_i(phi) D^i(phi))/2",
        "definitions": [
            "Pi_phi=n^mu nabla_mu phi=(dot(phi)-N^i D_i phi)/N"
        ],
        "velocity_channels": ["Pi_phi"],
        "spatial_jets": ["D_i phi"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["canonical_scalar"],
        "proof_scope": "exact first-derivative scalar 3+1 split for the contract definition of X_phi",
    },
    "SCALAR_MASS": {
        "bulk_basis": "N sqrt(h) m_phi^2 phi^2",
        "velocity_channels": [],
        "spatial_jets": [],
        "nondynamical_channels": ["N"],
        "boundary_terms": [],
        "required_controls": ["canonical_scalar"],
        "proof_scope": "algebraic scalar potential density",
    },
    "HORNDESKI_L4_LINEAR_X": {
        "bulk_basis": "N sqrt(h) [-X_c (K_ij K^ij-K^2) + X_c R3 + spatial A_star-gradient terms]",
        "definitions": [
            "X_c=Lambda_phi^4 X_phi=A_star^2/2 in unitary gauge",
            "V_star=L_n A_star cancels between the G4(X_c)R boundary term and the Horndeski second-derivative completion",
        ],
        "velocity_channels": ["K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)"],
        "spatial_jets": ["R3[h]", "D_i A_star", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i", "A_star"],
        "boundary_terms": [
            "Gauss-Codazzi normal divergence integrated against G4(X_c); its V_star K bulk term is retained and exactly cancelled"
        ],
        "required_controls": ["quartic_horndeski_covariant_adm_degeneracy"],
        "proof_scope": "exact named quartic-Horndeski unitary-gauge kinetic ADM identity for G4=M_Pl^2/2+alpha X_c; distributed closure and stability remain separate",
    },
    "PROCA_F2": {
        "bulk_basis": "2 N sqrt(h) (B_ij B^ij/2 - E_i E^i)",
        "definitions": [
            "E_i=n^mu F_mu_i",
            "B_ij=h_i^mu h_j^nu F_mu_nu=2 D_[i A_j]",
        ],
        "velocity_channels": ["E_i"],
        "spatial_jets": ["B_ij", "D_i A_perp"],
        "nondynamical_channels": ["N", "N^i", "A_perp=-n^mu A_mu"],
        "boundary_terms": [],
        "required_controls": ["proca_adm_dirac"],
        "proof_scope": "exact antisymmetric-field-strength 3+1 split",
    },
    "PROCA_MASS": {
        "bulk_basis": "N sqrt(h) m_A^2 (-A_perp^2 + A_i A^i)",
        "velocity_channels": [],
        "spatial_jets": [],
        "nondynamical_channels": ["N", "A_perp=-n^mu A_mu"],
        "boundary_terms": [],
        "required_controls": ["proca_adm_dirac"],
        "proof_scope": "exact normal/spatial covector norm split",
    },
    "AETHER_K1": {
        "bulk_basis": "N sqrt(h) [Q^2-E_i E^i-P_i P^i+S_ij S^ij]",
        "velocity_channels": [
            "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)",
            "V_i=L_n A_i",
            "L_n chi",
        ],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre"],
        "proof_scope": "exact K1_u block before positive-unit-branch velocity reduction",
    },
    "AETHER_K2": {
        "bulk_basis": "N sqrt(h) (Q + S_i^i)^2",
        "velocity_channels": [
            "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)",
            "V_i=L_n A_i",
            "L_n chi",
        ],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre"],
        "proof_scope": "exact K2_u block before positive-unit-branch velocity reduction",
    },
    "AETHER_K3": {
        "bulk_basis": "N sqrt(h) [Q^2+2 E_i P^i+S_ij S^ji]",
        "velocity_channels": [
            "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)",
            "V_i=L_n A_i",
            "L_n chi",
        ],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre"],
        "proof_scope": "exact K3_u block before positive-unit-branch velocity reduction",
    },
    "AETHER_K4": {
        "bulk_basis": "N sqrt(h) [-a_perp^2+a_i a^i]",
        "definitions": [
            "a_perp=-chi Q-A^i P_i",
            "a_i=chi E_i+S_ji A^j",
        ],
        "velocity_channels": [
            "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2N)",
            "V_i=L_n A_i",
            "L_n chi",
        ],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre"],
        "proof_scope": "exact Aether acceleration K4_u block before positive-unit-branch reduction",
    },
    "AETHER_Q1": {
        "bulk_basis": "N sqrt(h) Q3, Q3=(L_sigma^2/a_sigma^2) P3^{AC} P3^{BD} B_AB B_CD",
        "definitions": [
            "P3^{nn}=A_i A^i, P3^{ni}=chi A^i, P3^{ij}=h^{ij}+A^i A^j",
            "B_AB=e_A^mu e_B^nu nabla_mu(a_nu)",
            "a_perp=-chi Q-A^i P_i, a_i=chi E_i+S_ji A^j",
        ],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "higher_time_derivative_channels": ["L_n a_perp", "L_n a_i"],
        "spatial_jets": ["D_i a_perp", "D_i a_j", "D_i K_jk", "D_i V_j"],
        "nondynamical_channels": [],
        "boundary_terms": [],
        "required_controls": ["projected_aether_q_generic_3plus1_decomposition"],
        "proof_scope": "exact generic-tilt projector block decomposition; higher-jet Dirac degeneracy is not inferred",
    },
    "AETHER_Q2": {
        "bulk_basis": "N sqrt(h) Q3^2, Q3=(L_sigma^2/a_sigma^2) P3^{AC} P3^{BD} B_AB B_CD",
        "definitions": [
            "P3^{nn}=A_i A^i, P3^{ni}=chi A^i, P3^{ij}=h^{ij}+A^i A^j",
            "B_AB=e_A^mu e_B^nu nabla_mu(a_nu)",
        ],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "higher_time_derivative_channels": ["L_n a_perp", "L_n a_i"],
        "spatial_jets": ["D_i a_perp", "D_i a_j", "D_i K_jk", "D_i V_j"],
        "nondynamical_channels": [],
        "boundary_terms": [],
        "required_controls": ["projected_aether_q_generic_3plus1_decomposition"],
        "proof_scope": "exact square of the generic-tilt projector block; nonlinear higher-jet Dirac degeneracy is not inferred",
    },
    "AETHER_Q3": {
        "bulk_basis": "N sqrt(h) Q3^3, Q3=(L_sigma^2/a_sigma^2) P3^{AC} P3^{BD} B_AB B_CD",
        "definitions": [
            "P3^{nn}=A_i A^i, P3^{ni}=chi A^i, P3^{ij}=h^{ij}+A^i A^j",
            "B_AB=e_A^mu e_B^nu nabla_mu(a_nu)",
        ],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "higher_time_derivative_channels": ["L_n a_perp", "L_n a_i"],
        "spatial_jets": ["D_i a_perp", "D_i a_j", "D_i K_jk", "D_i V_j"],
        "nondynamical_channels": [],
        "boundary_terms": [],
        "required_controls": ["projected_aether_q_generic_3plus1_decomposition"],
        "proof_scope": "exact cube of the generic-tilt projector block; nonlinear higher-jet Dirac degeneracy is not inferred",
    },
    "AETHER_X_SQRT1P": {
        "bulk_basis": "N sqrt(h) [sqrt(1+(-a_perp^2+a_i a^i)/a_sigma^2)-1]",
        "definitions": [
            "a_perp=-chi Q-A^i P_i",
            "a_i=chi E_i+S_ji A^j",
        ],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre"],
        "proof_scope": "exact nonlinear function of the Aether-acceleration K4_u block",
    },
    "AETHER_X_P2_3": {
        "bulk_basis": "N sqrt(h) [(1+(-a_perp^2+a_i a^i)/a_sigma^2)^(2/3)-1]",
        "definitions": ["a_perp=-chi Q-A^i P_i", "a_i=chi E_i+S_ji A^j"],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": [
            "einstein_aether_generic_3plus1_legendre",
            "nonlinear_aether_acceleration_global_convexity",
        ],
        "proof_scope": "exact nonlinear function of the Aether-acceleration K4_u block; nonlinear Legendre closure remains separate",
    },
    "AETHER_X_P3_4": {
        "bulk_basis": "N sqrt(h) [(1+(-a_perp^2+a_i a^i)/a_sigma^2)^(3/4)-1]",
        "definitions": ["a_perp=-chi Q-A^i P_i", "a_i=chi E_i+S_ji A^j"],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": [
            "einstein_aether_generic_3plus1_legendre",
            "nonlinear_aether_acceleration_global_convexity",
        ],
        "proof_scope": "exact nonlinear function of the Aether-acceleration K4_u block; nonlinear Legendre closure remains separate",
    },
    "AETHER_MATCHED_K14_P1_2": {
        "bulk_basis": "N sqrt(h) (1+X_a)^(-3/2) [K1_ADM+K4_ADM]",
        "definitions": ["X_a=(-a_perp^2+a_i a^i)/a_sigma^2", "K1_ADM and K4_ADM are the exact registered Aether blocks"],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre", "nonlinear_aether_acceleration_global_convexity"],
        "proof_scope": "exact product of first-derivative scalar blocks; nonlinear constraint closure remains separate",
    },
    "AETHER_MATCHED_K14_P2_3": {
        "bulk_basis": "N sqrt(h) (1+X_a)^(-4/3)(1+X_a/3) [K1_ADM+K4_ADM]",
        "definitions": ["X_a=(-a_perp^2+a_i a^i)/a_sigma^2", "K1_ADM and K4_ADM are the exact registered Aether blocks"],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre", "nonlinear_aether_acceleration_global_convexity"],
        "proof_scope": "exact product of first-derivative scalar blocks; nonlinear constraint closure remains separate",
    },
    "AETHER_MATCHED_K14_P3_4": {
        "bulk_basis": "N sqrt(h) (1+X_a)^(-5/4)(1+X_a/2) [K1_ADM+K4_ADM]",
        "definitions": ["X_a=(-a_perp^2+a_i a^i)/a_sigma^2", "K1_ADM and K4_ADM are the exact registered Aether blocks"],
        "velocity_channels": ["K_ij", "V_i=L_n A_i", "L_n chi"],
        "spatial_jets": ["D_i A_j", "D_i chi", "D_i ln(N)"],
        "nondynamical_channels": ["N", "N^i"],
        "boundary_terms": [],
        "required_controls": ["einstein_aether_generic_3plus1_legendre", "nonlinear_aether_acceleration_global_convexity"],
        "proof_scope": "exact product of first-derivative scalar blocks; nonlinear constraint closure remains separate",
    },
    "UNIT_VECTOR_CONSTRAINT": {
        "bulk_basis": "N sqrt(h) lambda_u (-chi^2 + A_i A^i + 1)",
        "definitions": ["u^a=chi n^a+A^a", "chi>0"],
        "velocity_channels": [],
        "spatial_jets": [],
        "nondynamical_channels": ["N", "lambda_u"],
        "boundary_terms": [],
        "required_controls": [
            "einstein_aether_generic_3plus1_legendre",
            "unit_timelike_vector_dirac_chain",
        ],
        "proof_scope": "exact positive-unit-branch holonomic constraint split",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _unique(items: list[str]) -> list[str]:
    return sorted(set(items))


def compile_adm_ir(
    action_ir: dict[str, Any],
    control_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Compile a valid covariant action IR into a fail-closed, termwise 3+1 IR."""

    if not action_ir.get("valid"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["covariant action IR is invalid"],
            "promotion_allowed": False,
        }
    canonical = action_ir["canonical"]
    term_records: list[dict[str, Any]] = []
    missing_templates: list[str] = []
    missing_or_failed_controls: list[str] = []
    velocity_channels: list[str] = []
    spatial_jets: list[str] = []
    nondynamical_channels: list[str] = []
    boundaries: list[str] = []
    definitions: list[str] = []
    higher_time_derivative_channels: list[str] = []
    for term in canonical["terms"]:
        template = _TERM_TEMPLATES.get(term["id"])
        if template is None:
            missing_templates.append(term["id"])
            continue
        required_controls = list(template["required_controls"])
        if control_status is None:
            verification = "unresolved_not_executed"
            missing_or_failed_controls.extend(required_controls)
        else:
            failed = [name for name in required_controls if not control_status.get(name, False)]
            verification = "pass" if not failed else "unresolved_control_failure"
            missing_or_failed_controls.extend(failed)
        record = {
            "term_id": term["id"],
            "covariant_density": term["density"],
            "coefficient": term["coefficient"],
            "adm_bulk_basis": template["bulk_basis"],
            "weighted_adm_bulk": f"({term['coefficient']}) * ({template['bulk_basis']})",
            "definitions": list(template.get("definitions", [])),
            "velocity_channels": list(template["velocity_channels"]),
            "higher_time_derivative_channels": list(template.get("higher_time_derivative_channels", [])),
            "spatial_jets": list(template["spatial_jets"]),
            "nondynamical_channels": list(template["nondynamical_channels"]),
            "boundary_terms": list(template["boundary_terms"]),
            "required_controls": required_controls,
            "verification": verification,
            "proof_scope": template["proof_scope"],
        }
        term_records.append(record)
        velocity_channels.extend(record["velocity_channels"])
        higher_time_derivative_channels.extend(record["higher_time_derivative_channels"])
        spatial_jets.extend(record["spatial_jets"])
        nondynamical_channels.extend(record["nondynamical_channels"])
        boundaries.extend(record["boundary_terms"])
        definitions.extend(record["definitions"])

    term_ids = {term["id"] for term in canonical["terms"]}
    has_q_higher_jets = any(term_id.startswith("AETHER_Q") for term_id in term_ids)
    primary_seeds = (
        ["lapse/shift primary constraints unresolved for generic-tilt higher-jet Q_a_u"]
        if has_q_higher_jets
        else ["p_N=0", "p_(N^i)=0 (three components)"]
    )
    secondary_seeds = (
        ["Hamiltonian/momentum secondary constraints require higher-jet Dirac reduction"]
        if has_q_higher_jets
        else ["H_perp=0", "H_i=0 (three components)"]
    )
    if "PROCA_F2" in term_ids:
        primary_seeds.append("p_(A_perp)=0")
        secondary_seeds.append("Proca Gauss/mass constraint from preservation of p_(A_perp)")
    if "UNIT_VECTOR_CONSTRAINT" in term_ids:
        primary_seeds.append("p_(lambda_u)=0")
        secondary_seeds.append("-chi^2+A_i A^i+1=0")

    templates_complete = not missing_templates and len(term_records) == len(canonical["terms"])
    controls_pass = control_status is not None and not missing_or_failed_controls
    status = "pass" if templates_complete and controls_pass else "unresolved"
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "source_role": canonical["source_role"],
        "matter_metric": canonical["matter_metric"],
        "static_dictionary_status": canonical["static_dictionary_status"],
        "term_templates_complete": templates_complete,
        "missing_templates": missing_templates,
        "missing_or_failed_controls": _unique(missing_or_failed_controls),
        "terms": term_records,
        "combined_adm_bulk_density": " + ".join(
            record["weighted_adm_bulk"] for record in term_records
        ),
        "definitions": _unique(definitions),
        "velocity_channels": _unique(velocity_channels),
        "higher_time_derivative_channels": _unique(higher_time_derivative_channels),
        "spatial_jets": _unique(spatial_jets),
        "nondynamical_channels": _unique(nondynamical_channels),
        "primary_constraint_seeds": primary_seeds,
        "secondary_constraint_seeds": secondary_seeds,
        "boundary_contract": _unique(boundaries),
        "lapse_shift_statement": (
            "Generic tilt makes P3^{nn}=A_i A^i nonzero, so Q_a_u contains normal "
            "derivatives of Aether acceleration. No lapse/shift primary or secondary "
            "constraint is asserted before an auxiliary-field higher-jet Dirac analysis."
            if has_q_higher_jets
            else "No dot(N) or dot(N^i) channel occurs after the declared boundary completion; "
            "their momenta are primary constraints. Secondary H_perp/H_i entries are seeds "
            "until the action-specific Poisson algebra is executed."
        ),
        "unit_aether_reduction": (
            {
                "branch": "chi=sqrt(1+A_i A^i)>0",
                "normal_velocity": "L_n chi=(A^i L_n A_i-A^i K_ij A^j)/chi",
                "spatial_gradient": "D_i chi=A^j D_i A_j/chi",
                "electric_velocity": "W_i=V_i-K_ij A^j+chi D_i ln(N)",
            }
            if "UNIT_VECTOR_CONSTRAINT" in term_ids
            else None
        ),
        "proof_scope": (
            "termwise exact 3+1 template instantiation and primary/secondary seed extraction for "
            "the bounded action grammar; it does not by itself prove action-specific Poisson "
            "closure, global Hessian rank, physical degree count, or reduced boundedness"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_adm_ir(adm_ir: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adm_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
