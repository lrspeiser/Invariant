from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = (
    "sigma-future-g3-nonradial-york-bounded-mean-curvature-no-go-campaign-1.0"
)
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_"
    "conformally_flat_bounded_mean_curvature_York_class"
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
    if _file_sha(root / config["adapter_source"]["path"]) != config["adapter_source"][
        "file_sha256"
    ]:
        raise ValueError("campaign source hash mismatch")


def _validate_domain_contract(contract: dict[str, Any]) -> None:
    expected = {
        "initial_slice": "R3",
        "spatial_metric": "h_ij=psi(x)^4*delta_ij",
        "conformal_factor_class": "C2_positive_nonradial_allowed",
        "conformal_boundary_and_falloff": [
            "psi(x)>0",
            "psi(x)-1=O_2(|x|^-1)",
            "psi(x)->1_as_|x|->infinity",
        ],
        "scalar_position": "phi_at_t0=0",
        "scalar_normal_gradient": "Pi=v(r)=1/sqrt(1+(r/L)^4)",
        "scalar_spatial_gradient": "D_i(phi)=0",
        "transition_length_L": "100",
        "extrinsic_curvature_decomposition": (
            "K_ij=A_ij+(K/3)*h_ij_with_h_trace(A)=0"
        ),
        "tracefree_York_tensor_scope": (
            "arbitrary_smooth_nonradial_TT_or_longitudinal_or_mixed_A_ij"
        ),
        "mean_curvature_bound": "abs(K(x))<=(6/5)*v(r)",
        "extrinsic_curvature_falloff": "K_ij=O_1(|x|^-2)",
        "no_momentum_solution_assumed": True,
        "no_observation_or_numerical_solver_used": True,
    }
    if contract != expected:
        raise ValueError("nonradial York domain contract changed")


def _symbolic_york_hamiltonian_control() -> dict[str, Any]:
    v, beta = sp.symbols("v beta", positive=True, real=True)
    mean_k = sp.Symbol("K", real=True)
    a_sq = sp.Symbol("A_squared", nonnegative=True, real=True)
    rho = v**2 / 2 + beta * mean_k * v**3
    source = sp.factor(2 * rho - sp.Rational(2, 3) * mean_k**2 + a_sq)
    expected = a_sq - sp.Rational(2, 3) * mean_k**2 + 2 * beta * mean_k * v**3 + v**2
    if sp.expand(source - expected) != 0:
        raise ValueError("York Hamiltonian reduction changed")
    body = {
        "constraint_normalization": "R3+K^2-K_ij*K^ij=2*rho",
        "York_decomposition": "K_ij=A_ij+(K/3)*h_ij",
        "tracefree_norm_identity": "K_ij*K^ij=A_ij*A^ij+K^2/3",
        "candidate_matter_density": "rho=v^2/2+beta*K*v^3",
        "required_scalar_curvature": (
            "R3=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
        ),
        "conformal_scalar_curvature": "R3=-8*psi^(-5)*Delta_delta(psi)",
        "reduced_equation": "-Delta_delta(psi)=q(x)*psi(x)^5",
        "q_definition": (
            "q=(v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij)/8"
        ),
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _universal_nonradial_green_control() -> dict[str, Any]:
    y = sp.Symbol("y", positive=True)
    ratio = (y - 1) / y**5
    maximum = sp.factor(ratio.subs(y, sp.Rational(5, 4)))
    if maximum != sp.Rational(256, 3125):
        raise ValueError("universal nonradial comparison maximum changed")
    body = {
        "AF_Newton_representation": (
            "psi(x)-1=(1/(4*pi))*integral_R3_"
            "q(z)*psi(z)^5/|x-z|_d3z"
        ),
        "positivity_consequence": "psi(x)>=1",
        "ball": "B_L={x:|x|<=L}",
        "ball_kernel_bound": "1/|x-z|>=1/(2*L)_for_x,z_in_B_L",
        "ball_volume": "4*pi*L^3/3",
        "ball_minimum": "m=min_B_L(psi)>=1",
        "necessary_inequality": "m-1>=B_L*m^5",
        "B_L_definition": "B_L=q_lower_on_B_L*L^2/6",
        "universal_allowed_B_L_upper": "256/3125",
        "unique_maximizer": "m=5/4",
        "scope": (
            "The Newton representation follows from the declared O_2 AF falloff and the "
            "nonnegative Hamiltonian source. No radial symmetry of psi, K, or A_ij is used."
        ),
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_no_go(
    prior: dict[str, Any], beta: Fraction, contract: dict[str, Any]
) -> dict[str, Any]:
    kappa = Fraction(6, 5)
    length = Fraction(contract["transition_length_L"])
    source_factor = 1 - 2 * beta * kappa - Fraction(2, 3) * kappa**2
    q_lower = source_factor / 16
    green_coefficient = q_lower * length**2 / 6
    universal_maximum = Fraction(256, 3125)
    excess = green_coefficient - universal_maximum
    if source_factor <= 0 or q_lower <= 0 or excess <= 0:
        raise ValueError("candidate nonradial York comparison obstruction did not close")
    certificate = prior["radial_Lichnerowicz_no_go_certificate"]
    if (
        prior["decision"] != "blocked"
        or prior["theory_rejected"] is not False
        or certificate["positive_global_solution_exists_in_declared_class"] is not False
    ):
        raise ValueError("predecessor no-go decision changed")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_radial_no_go_sha256": certificate["content_sha256"],
        "domain_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_constraint_or_no_go_evidence": False,
        "exact_source_bound": {
            "mean_curvature_cap_kappa": str(kappa),
            "starting_identity": (
                "8*q=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
            ),
            "bound_steps": [
                "2*beta*K*v^3>=-2*beta*kappa*v^4>=-2*beta*kappa*v^2",
                "-(2/3)*K^2>=-(2/3)*kappa^2*v^2",
                "A_ij*A^ij>=0",
            ],
            "global_source_factor": str(source_factor),
            "global_q_lower_as_factor_times_v_squared": (
                f"q>={source_factor}/8*v^2"
            ),
            "v_squared_lower_on_B_L": "1/2",
            "q_lower_on_B_L": str(q_lower),
        },
        "exact_nonradial_green_comparison": {
            "green_ball_coefficient_B_L_lower": str(green_coefficient),
            "universal_allowed_B_L_upper": str(universal_maximum),
            "strict_excess": str(excess),
            "B_L_lower_exceeds_universal_upper": True,
            "positive_AF_conformal_factor_exists_in_declared_class": False,
        },
        "decision": "reject_conformally_flat_bounded_mean_curvature_York_class",
        "momentum_constraint_status": (
            "not_reached_because_Hamiltonian_constraint_has_no_solution_in_class"
        ),
        "candidate_nontrivial_AF_Einstein_constraint_solution_available": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "This exact comparison excludes nonradial conformal factors and arbitrary smooth "
            "trace-free York tensors, including TT and longitudinal pieces, whenever the "
            "candidate-bound mean curvature obeys |K|<=(6/5)*v. It does not exclude "
            "non-conformally-flat metrics, mean curvature outside that cap, a different scalar "
            "profile, or the covariant action."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_domain_contract(config["domain_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    radial_reduction = predecessors["radial_reduction_source"]
    nonunitary = predecessors["nonunitary_predecessor"]
    af_profile = predecessors["AF_profile_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != radial_reduction.get("content_sha256")
        or radial_reduction.get("source_bindings", {}).get("predecessor", {}).get(
            "content_sha256"
        )
        != nonunitary.get("content_sha256")
        or radial_reduction.get("source_bindings", {}).get("AF_profile_source", {}).get(
            "content_sha256"
        )
        != af_profile.get("content_sha256")
    ):
        raise ValueError("predecessor chain changed")
    symbolic = _symbolic_york_hamiltonian_control()
    universal = _universal_nonradial_green_control()
    records_by_id = {item["candidate_id"]: item for item in immediate["candidate_records"]}
    records = []
    for target in config["targets"]:
        prior = records_by_id.get(target["candidate_id"])
        if (
            prior is None
            or prior["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or prior["radial_Lichnerowicz_no_go_certificate"]["content_sha256"]
            != target["radial_no_go_content_sha256"]
        ):
            raise ValueError("target binding changed")
        no_go = _candidate_no_go(
            prior, Fraction(target["beta"]), config["domain_contract"]
        )
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "radial_reduction_content_sha256": radial_reduction["content_sha256"],
            "nonunitary_predecessor_content_sha256": nonunitary["content_sha256"],
            "AF_profile_source_content_sha256": af_profile["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "domain_contract_sha256": _sha(config["domain_contract"]),
            "symbolic_control_sha256": symbolic["content_sha256"],
            "universal_green_control_sha256": universal["content_sha256"],
            "candidate_no_go_sha256": no_go["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "nonradial_York_no_go_certificate": no_go,
            "gate_ledger": {
                "radial_Lichnerowicz_no_go_predecessor": {"status": "pass"},
                "nonradial_York_Hamiltonian_reduction": {"status": "pass"},
                "bounded_mean_curvature_green_comparison": {"status": "pass"},
                "conformally_flat_bounded_mean_curvature_York_class": {
                    "status": "reject_ansatz_class"
                },
                "candidate_nontrivial_AF_Einstein_constraint_solution_beyond_class": {
                    "status": "blocked"
                },
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
        raise ValueError("expected exactly three candidate-bound nonradial York records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "symbolic_York_Hamiltonian_control": symbolic,
        "universal_nonradial_green_control": universal,
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "nonradial_York_Hamiltonian_reduction_pass_count": 3,
        "bounded_mean_curvature_green_comparison_pass_count": 3,
        "conformally_flat_bounded_mean_curvature_York_class_reject_count": 3,
        "momentum_constraint_solution_pass_count": 0,
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
        "theory_reject_count": 0,
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
            "The radial no-go now extends to a genuinely nonradial conformally flat York class. "
            "For each action, arbitrary trace-free TT/longitudinal data only increase the "
            "Hamiltonian source, while |K|<=(6/5)*v leaves a strictly positive candidate-bound "
            "source margin. The AF Newton representation forces a ball-minimum inequality whose "
            "exact coefficient exceeds 256/3125, excluding every positive conformal factor in "
            "the class. This is an ansatz-class obstruction, not an action rejection; "
            "non-conformally-flat or larger-mean-curvature constraint data, global energy, and "
            "full formal completion remain fail-closed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
        config, root
    )
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
