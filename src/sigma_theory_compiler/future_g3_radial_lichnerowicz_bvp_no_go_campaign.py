from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .promotion_orchestrator import ELIGIBILITY

ARTIFACT_SCHEMA = "sigma-future-g3-radial-lichnerowicz-bvp-no-go-campaign-1.0"
FIRST_BLOCKER = (
    "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_"
    "radial_conformal_pure_trace_ansatz"
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


def _validate_comparison_contract(contract: dict[str, Any]) -> None:
    expected = {
        "comparison_radius": "L",
        "transition_length_L": "100",
        "solution_class": "C2_positive_radial_regular_center_and_psi_limit_one",
        "equation": "psi_double_prime+2*psi_prime/r+Q_beta*psi^5=0",
        "Q_beta": "(1/8)*v^2*(1-(7/6)*beta^2*v^4)",
        "v_squared": "1/(1+(r/L)^4)",
        "boundary_conditions": ["psi_prime(0)=0", "psi(r)>0", "psi(infinity)=1"],
        "no_observation_or_numerical_shooting_used": True,
    }
    if contract != expected:
        raise ValueError("comparison proof contract changed")


def _universal_scalar_inequality_control() -> dict[str, Any]:
    y = sp.Symbol("y", positive=True)
    ratio = (y - 1) / y**5
    derivative = sp.factor(sp.diff(ratio, y))
    critical = sp.Rational(5, 4)
    maximum = sp.factor(ratio.subs(y, critical))
    if sp.simplify(derivative + (4 * y - 5) / y**6) != 0 or maximum != sp.Rational(
        256, 3125
    ):
        raise ValueError("universal comparison maximum changed")
    body = {
        "domain": "y>=1",
        "function": "f(y)=(y-1)/y^5",
        "derivative": str(derivative),
        "unique_interior_maximizer": str(critical),
        "endpoint_values": {"f(1)": "0", "limit_y_to_infinity": "0"},
        "exact_global_maximum": str(maximum),
        "status": "pass",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_no_go(
    prior: dict[str, Any], beta: Fraction, contract: dict[str, Any]
) -> dict[str, Any]:
    certificate = prior["constraint_reduction_certificate"]
    bvp = certificate["radial_Lichnerowicz_BVP"]
    if (
        bvp["equation"]
        != "psi_double_prime(r)+2*psi_prime(r)/r+Q_beta(r)*psi(r)^5=0"
        or bvp["Q_beta"] != "(1/8)*v(r)^2*(1-(7/6)*beta^2*v(r)^4)"
        or bvp["boundary_conditions"]
        != ["psi(r)>0", "psi_prime(0)=0", "psi(infinity)=1"]
        or bvp["positive_global_solution_proved"] is not False
    ):
        raise ValueError("predecessor radial BVP changed")
    length = Fraction(contract["transition_length_L"])
    margin = 1 - Fraction(7, 6) * beta**2
    q_lower_on_ball = margin / 16
    a_lower = length**2 * margin / 48
    universal_maximum = Fraction(256, 3125)
    excess = a_lower - universal_maximum
    if margin <= 0 or q_lower_on_ball <= 0 or excess <= 0:
        raise ValueError("candidate comparison obstruction did not close")
    body = {
        "candidate_id": prior["candidate_id"],
        "action_sha256": prior["action_sha256"],
        "beta": str(beta),
        "predecessor_constraint_reduction_sha256": certificate["content_sha256"],
        "comparison_contract": contract,
        "direct_action_binding": True,
        "family_label_used_as_existence_or_no_go_evidence": False,
        "radial_monotonicity_identity": {
            "mass_function": "M(r)=integral_0^r s^2*Q_beta(s)*psi(s)^5 ds",
            "identity": "-r^2*psi_prime(r)=M(r)",
            "consequences": ["psi_prime(r)<=0", "psi(r)>=psi(infinity)=1"],
        },
        "comparison_at_R_equals_L": {
            "exact_tail_identity": (
                "psi(L)-1=integral_L^infinity M(t)/t^2 dt"
            ),
            "tail_lower_bound": "psi(L)-1>=M(L)/L",
            "monotonic_source_lower_bound": (
                "M(L)>=psi(L)^5*integral_0^L s^2*Q_beta(s) ds"
            ),
            "combined_necessary_inequality": "y-1>=A_L*y^5_for_y=psi(L)>=1",
            "A_L_definition": "A_L=(1/L)*integral_0^L s^2*Q_beta(s) ds",
        },
        "exact_candidate_bounds": {
            "v_squared_on_0_to_L": "1/2<=v^2<=1",
            "one_minus_braiding_factor_lower": str(margin),
            "Q_beta_lower_on_0_to_L": str(q_lower_on_ball),
            "A_L_lower": str(a_lower),
            "universal_allowed_A_L_upper": str(universal_maximum),
            "strict_excess": str(excess),
            "A_L_lower_exceeds_universal_upper": True,
        },
        "decision": "reject_radial_conformal_pure_trace_ansatz",
        "positive_global_solution_exists_in_declared_class": False,
        "theory_rejected": False,
        "first_remaining_blocker": FIRST_BLOCKER,
        "scope": (
            "The comparison inequality exactly excludes a positive C2 radial solution with a "
            "regular center and psi->1 for this candidate-bound pure-trace conformal reduction. "
            "It does not exclude anisotropic extrinsic curvature, non-conformally-flat spatial "
            "metrics, a different scalar profile, or the underlying covariant action."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    _validate_source(root, config)
    _validate_comparison_contract(config["comparison_contract"])
    predecessors = {
        key: _load_bound(root, descriptor) for key, descriptor in config["bindings"].items()
    }
    immediate = predecessors["predecessor"]
    nonunitary = predecessors["nonunitary_predecessor"]
    af_profile = predecessors["AF_profile_source"]
    if (
        immediate.get("source_bindings", {}).get("predecessor", {}).get("content_sha256")
        != nonunitary.get("content_sha256")
        or immediate.get("source_bindings", {}).get("AF_profile_source", {}).get(
            "content_sha256"
        )
        != af_profile.get("content_sha256")
    ):
        raise ValueError("predecessor chain changed")
    universal = _universal_scalar_inequality_control()
    records_by_id = {item["candidate_id"]: item for item in immediate["candidate_records"]}
    records = []
    for target in config["targets"]:
        prior = records_by_id.get(target["candidate_id"])
        if (
            prior is None
            or prior["action_sha256"] != target["action_sha256"]
            or prior["beta"] != target["beta"]
            or prior["content_sha256"] != target["predecessor_record_content_sha256"]
            or prior["constraint_reduction_certificate"]["content_sha256"]
            != target["constraint_reduction_content_sha256"]
            or prior["decision"] != "blocked"
            or prior["theory_rejected"] is not False
        ):
            raise ValueError("target binding changed")
        no_go = _candidate_no_go(
            prior, Fraction(target["beta"]), config["comparison_contract"]
        )
        gates = {
            "radial_Lichnerowicz_BVP_registration_predecessor": {"status": "pass"},
            "exact_comparison_inequality": {"status": "pass"},
            "positive_global_solution_in_declared_radial_class": {"status": "reject"},
            "radial_conformal_pure_trace_constraint_ansatz": {"status": "reject_ansatz"},
            "candidate_nontrivial_AF_Einstein_constraint_solution_beyond_ansatz": {
                "status": "blocked"
            },
            "global_hamiltonian_energy": {"status": "blocked"},
            "full_formal": {"status": "blocked"},
        }
        provenance_body = {
            "predecessor_content_sha256": immediate["content_sha256"],
            "predecessor_record_content_sha256": prior["content_sha256"],
            "nonunitary_predecessor_content_sha256": nonunitary["content_sha256"],
            "AF_profile_source_content_sha256": af_profile["content_sha256"],
            "action_sha256": prior["action_sha256"],
            "comparison_contract_sha256": _sha(config["comparison_contract"]),
            "universal_inequality_sha256": universal["content_sha256"],
            "candidate_no_go_sha256": no_go["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        record_body = {
            "candidate_id": prior["candidate_id"],
            "action_sha256": prior["action_sha256"],
            "beta": prior["beta"],
            "decision": "blocked",
            "first_blocker": FIRST_BLOCKER,
            "radial_Lichnerowicz_no_go_certificate": no_go,
            "gate_ledger": gates,
            "radial_conformal_pure_trace_ansatz_rejected": True,
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
        raise ValueError("expected exactly three candidate-bound BVP no-go records")
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_adapter_source": config["adapter_source"],
        "config_content_sha256": _sha(config),
        "source_bindings": config["bindings"],
        "universal_scalar_inequality_control": universal,
        "candidate_count": 3,
        "decision_counts": {"blocked": 3},
        "candidate_records": records,
        "exact_comparison_inequality_pass_count": 3,
        "positive_global_radial_Lichnerowicz_solution_pass_count": 0,
        "positive_global_radial_Lichnerowicz_solution_nonexistence_count": 3,
        "radial_conformal_pure_trace_ansatz_reject_count": 3,
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
            "The explicit radial Lichnerowicz BVP has no positive regular-center solution with "
            "psi->1 for any of the three candidate beta values: its exact candidate source lower "
            "bound violates the universal maximum of (y-1)/y^5. This rejects the radial "
            "conformal/pure-trace constraint ansatz, not the actions. An AF constraint solution "
            "with anisotropic extrinsic curvature, a non-conformally-flat spatial metric, or a "
            "different candidate-bound construction remains open; global energy and full formal "
            "completion remain fail-closed."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def write_future_g3_radial_lichnerowicz_bvp_no_go_campaign(
    config_path: str | Path, project_root: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact = build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(config, root)
    output = root / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical(artifact) + "\n", encoding="utf-8")
    return output
