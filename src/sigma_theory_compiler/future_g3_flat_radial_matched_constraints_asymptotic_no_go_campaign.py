from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = "sigma-future-g3-flat-radial-matched-constraints-asymptotic-no-go-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_AF_metric_York_data_beyond_flat_radial_r_minus_2_asymptotic_class"
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


def _load_bound(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != descriptor["content_sha256"]
        or _sha(body) != descriptor["content_sha256"]
    ):
        raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_source(root: Path, config: dict[str, Any]) -> None:
    source = config["adapter_source"]
    if _file_sha(root / source["path"]) != source["file_sha256"]:
        raise ValueError("campaign source hash mismatch")


def _validate_asymptotic_contract(contract: dict[str, Any]) -> None:
    expected = {
        "spatial_metric": "h_ij=delta_ij_on_R3",
        "scalar_curvature": "R3=0",
        "scalar_profile": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "transition_length_L": "100",
        "York_decomposition": "K_ij=A_ij+(K/3)*delta_ij",
        "radial_tracefree_tensor": "A^i_j=a(r)*(n^i*n_j-delta^i_j/3)",
        "tracefree_norm": "A_ij*A^ij=(2/3)*a(r)^2",
        "real_data": True,
        "asymptotic_expansions": [
            "v(r)=L^2/r^2+O(r^-6)",
            "K(r)=k*L^2/r^2+o_1(r^-2)",
            "a(r)=alpha*L^2/r^2+o_1(r^-2)",
        ],
        "momentum_equation": ("a_prime+3*a/r-K_prime=(3/2)*beta*v^2*v_prime"),
        "Hamiltonian_equation": ("0=v^2+2*beta*K*v^3-(2/3)*K^2+(2/3)*a^2"),
        "no_compensation_inequality_assumed": True,
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("flat radial matched-constraint asymptotic contract changed")


def _symbolic_leading_control() -> dict[str, Any]:
    k, alpha = sp.symbols("k alpha", real=True)
    momentum = alpha + 2 * k
    hamiltonian = 1 - sp.Rational(2, 3) * k**2 + sp.Rational(2, 3) * alpha**2
    reduced = sp.factor(hamiltonian.subs(alpha, -2 * k))
    formal_k = sp.Symbol("formal_k")
    complex_roots = sp.solve(2 * formal_k**2 + 1, formal_k)
    if reduced != 2 * k**2 + 1 or complex_roots != [
        -sp.sqrt(2) * sp.I / 2,
        sp.sqrt(2) * sp.I / 2,
    ]:
        raise ValueError("flat radial leading obstruction changed")
    body = {
        "momentum_r_minus_3_coefficient": str(momentum),
        "Hamiltonian_r_minus_4_coefficient": str(hamiltonian),
        "momentum_condition": "alpha=-2*k",
        "joint_reduced_Hamiltonian_coefficient": str(reduced),
        "joint_real_solution_exists": False,
        "formal_complex_k_roots": [str(item) for item in complex_roots],
        "formal_complex_alpha_roots": [str(sp.simplify(-2 * item)) for item in complex_roots],
        "status": "pass_exact_real_asymptotic_obstruction",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_no_go(
    prior: dict[str, Any], contract: dict[str, Any], symbolic: dict[str, Any]
) -> dict[str, Any]:
    predecessor = prior["general_geometry_surplus_mismatch_certificate"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or predecessor["matched_surplus_control"]["status"] != "pointwise_Hamiltonian_match_only"
        or predecessor["matched_surplus_control"]["momentum_constraint_solved"] is not False
    ):
        raise ValueError("surplus-matching predecessor changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": prior["beta"],
        "predecessor_surplus_mismatch_sha256": predecessor["content_sha256"],
        "asymptotic_contract": contract,
        "symbolic_leading_control_sha256": symbolic["content_sha256"],
        "direct_action_binding": True,
        "family_label_used_as_constraint_evidence": False,
        "exact_candidate_asymptotics": {
            "scalar_profile_leading": "v=L^2*r^-2+O(r^-6)",
            "matter_momentum_source_leading": (f"(3/2)*({prior['beta']})*v^2*v_prime=O(r^-7)"),
            "cubic_Hamiltonian_term_leading": (f"2*({prior['beta']})*K*v^3=O(r^-8)"),
            "candidate_beta_absent_from_obstructing_orders": True,
        },
        "joint_constraint_obstruction": {
            "momentum_r_minus_3_condition": "alpha+2*k=0",
            "Hamiltonian_r_minus_4_condition": ("1-(2/3)*k^2+(2/3)*alpha^2=0"),
            "after_momentum_substitution": "1+2*k^2=0",
            "real_asymptotic_coefficients_exist": False,
            "flat_radial_AF_matched_constraint_datum_exists": False,
        },
        "momentum_only_negative_control": {
            "k": "0",
            "alpha": "0",
            "momentum_coefficient": "0",
            "Hamiltonian_coefficient": "1",
            "joint_constraints_pass": False,
        },
        "Hamiltonian_only_negative_control": {
            "k": "sqrt(6)/2",
            "alpha": "0",
            "Hamiltonian_coefficient": "0",
            "momentum_coefficient": "sqrt(6)",
            "joint_constraints_pass": False,
        },
        "complex_root_control": {
            "role": "algebraic_negative_control_only",
            "formal_roots": symbolic["formal_complex_k_roots"],
            "rejected_by_real_initial_data_contract": True,
        },
        "decision": "reject_flat_radial_r_minus_2_matched_constraint_class",
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "This exact obstruction jointly uses Hamiltonian matching and the momentum constraint "
            "for real flat spherical AF data with standard r^-2 York falloff. It assumes neither "
            "the prior compensation inequality nor pointwise surplus mismatch. Nonflat metrics, "
            "nonradial tensors, different leading falloff, and polyhomogeneous/logarithmic data "
            "remain open; the covariant action is not rejected."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_asymptotic_contract(config["asymptotic_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    curvature_shortfall = predecessors["curvature_shortfall_source"]
    if immediate.get("source_bindings", {}).get("predecessor", {}).get(
        "content_sha256"
    ) != curvature_shortfall.get("content_sha256"):
        raise ValueError("predecessor chain changed")
    symbolic = _symbolic_leading_control()
    records_by_id = {item["candidate_id"]: item for item in immediate["candidate_records"]}
    records = []
    for target in config["targets"]:
        prior = records_by_id.get(target["candidate_id"])
        if (
            prior is None
            or prior["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or prior["general_geometry_surplus_mismatch_certificate"]["content_sha256"]
            != target["predecessor_theorem_content_sha256"]
        ):
            raise ValueError("target binding changed")
        certificate = _candidate_no_go(prior, config["asymptotic_contract"], symbolic)
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "curvature_shortfall_source_content_sha256": curvature_shortfall["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "asymptotic_contract_sha256": _sha(config["asymptotic_contract"]),
            "symbolic_leading_control_sha256": symbolic["content_sha256"],
            "candidate_no_go_sha256": certificate["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "flat_radial_matched_constraint_asymptotic_certificate": certificate,
            "gate_ledger": {
                "surplus_matching_predecessor": {"status": "pass"},
                "radial_momentum_leading_order": {"status": "pass"},
                "flat_Hamiltonian_leading_order": {"status": "pass"},
                "joint_real_asymptotic_coefficients": {"status": "reject_constraint_class"},
                "candidate_AF_metric_York_datum_beyond_class": {"status": "blocked"},
                "global_hamiltonian_energy": {"status": "blocked"},
                "full_formal": {"status": "blocked"},
            },
            "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
            "theory_rejected": False,
            "global_energy_pass": False,
            "full_formal_pass": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
            "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    if len(records) != 3:
        raise ValueError("expected exactly three candidate flat-radial records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "symbolic_leading_order_control": symbolic,
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "radial_momentum_leading_order_pass_count": 3,
        "flat_Hamiltonian_leading_order_pass_count": 3,
        "joint_real_asymptotic_coefficient_solution_count": 0,
        "flat_radial_matched_constraint_class_reject_count": 3,
        "registered_AF_metric_York_datum_pass_count": 0,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "theory_reject_count": 0,
        "global_hamiltonian_energy_pass_count": 0,
        "full_formal_pass_count": 0,
        "first_blocker_counts": {FIRST_BLOCKER: 3},
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "synthetic_fixture_role": "deterministic_symbolic_negative_controls_only",
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "For every candidate, flat spherical AF data with real r^-2 York coefficients cannot "
            "jointly satisfy the momentum and Hamiltonian constraints. Momentum fixes "
            "alpha=-2k, after which the leading Hamiltonian coefficient is 1+2k^2. This is a "
            "constraint-class obstruction, not a theory rejection; nonflat, nonradial, or "
            "different-asymptotic data and global energy remain open."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_flat_radial_matched_constraints_asymptotic_no_go_campaign(
        config, root
    )
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
