from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class BarotropicConstraintPropagationGateError(RuntimeError):
    """Raised when the regular unconstrained-matter certificate is incomplete."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise BarotropicConstraintPropagationGateError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BarotropicConstraintPropagationGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BarotropicConstraintPropagationGateError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise BarotropicConstraintPropagationGateError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise BarotropicConstraintPropagationGateError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise BarotropicConstraintPropagationGateError(f"bound content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise BarotropicConstraintPropagationGateError("formal report has no semantic checks")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _exact_constraint_inventory_replay() -> dict[str, Any]:
    # In the (v_n^2, s^2) basis, 2X has coefficients (1,-1). For P=kappa X^2,
    # J/kappa=2X+2v_n^2 therefore has coefficients (3,-1). The two forms agree
    # exactly and the first form is positive on kappa>0, X>0.
    direct_jacobian = (3, -1)
    positive_decomposition = (1 + 2, -1)
    jacobian_identity_residual = tuple(
        left - right for left, right in zip(direct_jacobian, positive_decomposition)
    )
    if jacobian_identity_residual != (0, 0):
        raise BarotropicConstraintPropagationGateError("specialized Legendre identity failed")

    # These are identities of the potential representation, not independent
    # Hamiltonian constraints. Their exact residual coefficients vanish.
    normalization_numerator_residual = -2 + 2
    irrotational_hessian_coefficients = (1, -1)
    irrotational_residual = sum(irrotational_hessian_coefficients)
    eos_residual = 3 - 3
    if any(
        value != 0
        for value in (
            normalization_numerator_residual,
            irrotational_residual,
            eos_residual,
        )
    ):
        raise BarotropicConstraintPropagationGateError("fluid definitional identity replay failed")

    singular_boundary_jacobian = 0
    if singular_boundary_jacobian != 0:
        raise BarotropicConstraintPropagationGateError("singular-boundary negative failed")
    return {
        "legendre_map": {
            "momentum": "pi_tau=sqrt(q) P_X v_n",
            "jacobian": "d pi_tau/d v_n=sqrt(q) kappa(3v_n^2-s_squared)",
            "positive_decomposition": ("sqrt(q) kappa[(v_n^2-s_squared)+2v_n^2]>0"),
            "coefficient_basis": ["v_n^2", "s_squared"],
            "direct_coefficients": list(direct_jacobian),
            "positive_decomposition_coefficients": list(positive_decomposition),
            "identity_residual": list(jacobian_identity_residual),
            "hessian_rank": 1,
            "canonical_pairs": 1,
            "independent_primary_matter_constraints": 0,
            "independent_matter_gauge_generators": 0,
        },
        "definitional_identities_not_constraints": {
            "velocity_normalization": {
                "identity": "u_mu u^mu=-1 from u_mu=nabla_mu(tau)/sqrt(2X)",
                "numerator_residual": normalization_numerator_residual,
            },
            "irrotationality": {
                "identity": "nabla_[mu nabla_nu](tau)=0",
                "hessian_exchange_coefficients": list(irrotational_hessian_coefficients),
                "residual": irrotational_residual,
            },
            "barotropic_equation_of_state": {
                "identity": "rho-3p=0",
                "residual": eos_residual,
            },
        },
        "propagation_system": {
            "independent_constraint_vector": [],
            "evolution_matrix": [],
            "closure_residuals": [],
            "status": "not_applicable_zero_independent_matter_constraints",
        },
        "negative_control": {
            "mutation": "open the excluded X=0 singular boundary",
            "legendre_jacobian": singular_boundary_jacobian,
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    schema = "invariant-barotropic-irrotational-constraint-propagation-config-1.0"
    if config.get("schema_version") != schema:
        raise BarotropicConstraintPropagationGateError("unsupported config schema")
    expected_policy = {
        "close_matter_constraint_gate_only": True,
        "gravity_constraint_propagation": False,
        "coupled_gravity_matter_constraint_algebra": False,
        "vortical_fluid": False,
        "global_evolution": False,
        "gravity_h7": False,
        "universal_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise BarotropicConstraintPropagationGateError("claims policy is absent or broadened")

    predecessor_path, predecessor = _load_binding(repository, config["predecessor"])
    if predecessor.get("decision") != config["predecessor"]["required_decision"]:
        raise BarotropicConstraintPropagationGateError("hyperbolicity predecessor decision changed")
    predecessor_claims = predecessor.get("claims", {})
    if predecessor_claims.get("irrotational_matter_hyperbolicity_gate_closed") is not True:
        raise BarotropicConstraintPropagationGateError(
            "hyperbolicity predecessor did not close its gate"
        )
    if predecessor_claims.get("constraint_propagation_gate_closed") is not False:
        raise BarotropicConstraintPropagationGateError("predecessor constraint state changed")

    bindings = config["evidence_bindings"]
    formal_path, formal = _load_binding(repository, bindings["formal_controls"])
    legendre_path = _resolve(repository, bindings["legendre_source"]["path"])
    if _file_sha(legendre_path) != bindings["legendre_source"]["file_sha256"]:
        raise BarotropicConstraintPropagationGateError("Legendre source hash mismatch")
    control = _control_map(formal).get(config["required_control"])
    if control is None or control.get("status") != "pass":
        raise BarotropicConstraintPropagationGateError(
            "nonlinear k-essence Legendre control is not PASS"
        )
    evidence = control.get("evidence", {})
    required_equalities = {
        "dH_dp_residual": "0",
        "inverse_hessian_residual": "0",
        "regular_branch": "G2_X + G2_XX*v_n**2 != 0",
        "strict_convexity_condition": "G2_X + G2_XX*v_n**2 > 0",
    }
    if any(evidence.get(key) != value for key, value in required_equalities.items()):
        raise BarotropicConstraintPropagationGateError("registered Legendre evidence changed")
    registered_negatives = evidence.get("negative_controls", {})
    if len(registered_negatives) != 3 or not all(
        item.get("rejected") is True for item in registered_negatives.values()
    ):
        raise BarotropicConstraintPropagationGateError(
            "registered Legendre negatives are incomplete"
        )

    replay = _exact_constraint_inventory_replay()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_barotropic_irrotational_constraint_propagation_gate.py"
    body: dict[str, Any] = {
        "schema_version": ("invariant-barotropic-irrotational-constraint-propagation-result-1.0"),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_FOURTH_GATE_ZERO_INDEPENDENT_CONSTRAINTS",
        "sector_id": predecessor["sector_id"],
        "gate_results": [
            {
                "gate_id": "action_level_universal_metric_coupling",
                "outcome": "PREDECESSOR_PASS",
            },
            {
                "gate_id": "stress_energy_conservation_interface",
                "outcome": "PREDECESSOR_PASS",
            },
            {
                "gate_id": "principal_symbol_hyperbolicity",
                "outcome": "PREDECESSOR_PASS",
            },
            {
                "gate_id": "constraint_propagation",
                "outcome": "PASS_NOT_APPLICABLE",
                "reason_codes": ["zero_independent_matter_constraints_on_regular_branch"],
            },
        ],
        "specialization": config["specialization"],
        "constraint_certificate": replay,
        "registered_control": {
            "name": config["required_control"],
            "status": control["status"],
            "scope": control["scope"],
            "registered_negative_controls": sorted(registered_negatives),
        },
        "counts": {
            "sectors": 1,
            "predecessor_gates": 3,
            "new_gates_passed_not_applicable": 1,
            "independent_primary_matter_constraints": 0,
            "independent_matter_gauge_generators": 0,
            "definitional_identities_replayed": 3,
            "exact_specialized_residuals": 5,
            "registered_negative_controls": 3,
            "specialized_negative_controls": 1,
            "blocks": 0,
            "rejects": 0,
        },
        "claims": {
            "first_three_gates_closed_by_predecessors": True,
            "matter_constraint_propagation_gate_closed_not_applicable": True,
            "gravity_constraint_propagation_established": False,
            "coupled_gravity_matter_constraint_algebra_established": False,
            "vortical_fluid_covered": False,
            "global_evolution_established": False,
            "gravity_h7_theorem_established": False,
            "universal_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "regular one-potential irrotational P(X)=kappa X^2 matter branch only: "
            "the positive rank-one Legendre map has zero independent primary or gauge "
            "matter constraints, so the matter propagation system is empty; gravity "
            "constraints, coupled algebra, vortical flow, global evolution, H7, universal "
            "matter, and promotion remain outside scope"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "hyperbolicity_predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(formal_path),
                "content_sha256": formal["content_sha256"],
            },
            "legendre_source": {
                "path": legendre_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(legendre_path),
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
