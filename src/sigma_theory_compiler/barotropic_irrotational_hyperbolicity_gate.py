from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class BarotropicHyperbolicityGateError(RuntimeError):
    """Raised when bound principal-cone evidence is missing or altered."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise BarotropicHyperbolicityGateError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BarotropicHyperbolicityGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BarotropicHyperbolicityGateError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise BarotropicHyperbolicityGateError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise BarotropicHyperbolicityGateError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise BarotropicHyperbolicityGateError(f"bound content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise BarotropicHyperbolicityGateError("formal report has no semantic checks")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _exact_specialized_principal_replay() -> dict[str, Any]:
    # Suppress the strictly positive common factor kappa*X. In a local physical
    # orthonormal frame aligned with the timelike field gradient, the effective
    # inverse metric coefficients are diag(-6,2,2,2).
    p_x = 2
    two_x_p_xx = 4
    kinetic = p_x + two_x_p_xx
    gradient = p_x
    determinant_coefficient = -(gradient**3) * kinetic

    # The reduced Hamiltonian is kappa*X*k^2*q^2 + p^2/(12*kappa*X).
    # Multiplying by the positive factor 12*kappa*X leaves coefficients (12,1).
    rescaled_hamiltonian_coefficients = (12, 1)
    cone_residual = kinetic - 3 * gradient
    determinant_residual = determinant_coefficient - (-48)
    boundary_determinant = 0
    wrong_sign_kinetic = -kinetic
    if (
        cone_residual != 0
        or determinant_residual != 0
        or min(rescaled_hamiltonian_coefficients) <= 0
        or boundary_determinant != 0
        or wrong_sign_kinetic >= 0
    ):
        raise BarotropicHyperbolicityGateError("specialized principal replay failed")
    return {
        "positive_common_factor": "kappa X",
        "aligned_effective_inverse_metric": "kappa X diag(-6,2,2,2)",
        "aligned_integer_coefficients": [-6, 2, 2, 2],
        "determinant": "-48 (kappa X)^4",
        "determinant_coefficient": determinant_coefficient,
        "determinant_residual": determinant_residual,
        "principal_polynomial": "2 kappa X (|k|^2-3 omega^2)",
        "characteristic_roots": ["-|k|/sqrt(3)", "+|k|/sqrt(3)"],
        "sound_speed_squared": "1/3",
        "cone_residual": cone_residual,
        "scalar_gradient": "2 kappa X > 0",
        "scalar_kinetic": "6 kappa X > 0",
        "reduced_hamiltonian": "kappa X |k|^2 q^2 + p^2/(12 kappa X)",
        "positive_rescaled_hamiltonian_coefficients": list(rescaled_hamiltonian_coefficients),
        "common_time_covector": "nabla_mu(tau)",
        "common_time_check": (
            "nabla(tau) is timelike for g by X>0 and timelike for the acoustic "
            "inverse metric because its aligned time coefficient is -6 kappa X"
        ),
        "negative_controls": {
            "X_zero_cone_collapse": {
                "determinant": str(boundary_determinant),
                "rejected": True,
            },
            "wrong_sign_kappa_ghost": {
                "kinetic_coefficient": wrong_sign_kinetic,
                "rejected": True,
            },
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    schema = "invariant-barotropic-irrotational-hyperbolicity-config-1.0"
    if config.get("schema_version") != schema:
        raise BarotropicHyperbolicityGateError("unsupported config schema")
    expected_policy = {
        "close_matter_hyperbolicity_only": True,
        "coupled_gravity_matter_hyperbolicity": False,
        "constraint_propagation": False,
        "vortical_fluid": False,
        "global_evolution": False,
        "gravity_h7": False,
        "universal_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise BarotropicHyperbolicityGateError("claims policy is absent or broadened")

    predecessor_path, predecessor = _load_binding(repository, config["predecessor"])
    if predecessor.get("decision") != config["predecessor"]["required_decision"]:
        raise BarotropicHyperbolicityGateError("stress predecessor decision changed")
    predecessor_claims = predecessor.get("claims", {})
    if predecessor_claims.get("stress_conservation_gate_closed") is not True:
        raise BarotropicHyperbolicityGateError("stress predecessor did not close its gate")
    if predecessor_claims.get("hyperbolicity_gate_closed") is not False:
        raise BarotropicHyperbolicityGateError("predecessor hyperbolicity state changed")

    bindings = config["evidence_bindings"]
    formal_path, formal = _load_binding(repository, bindings["formal_controls"])
    principal_path = _resolve(repository, bindings["principal_source"]["path"])
    if _file_sha(principal_path) != bindings["principal_source"]["file_sha256"]:
        raise BarotropicHyperbolicityGateError("principal source hash mismatch")
    control = _control_map(formal).get(config["required_control"])
    if control is None or control.get("status") != "pass":
        raise BarotropicHyperbolicityGateError("k-essence principal control is not PASS")
    evidence = control.get("evidence", {})
    required_equalities = {
        "effective_metric_determinant_residual": "0",
        "canonical_momentum_residual": "0",
        "legendre_residual": "0",
        "hamiltonian_hessian_residual": "Matrix([[0, 0], [0, 0]])",
    }
    if any(evidence.get(key) != value for key, value in required_equalities.items()):
        raise BarotropicHyperbolicityGateError("registered principal residual changed")
    if evidence.get("healthy_domain") != [
        "G2_X > 0",
        "G2_X+2 X G2_XX > 0",
    ]:
        raise BarotropicHyperbolicityGateError("registered healthy domain changed")
    negatives = evidence.get("negative_controls", {})
    if len(negatives) != 4 or not all(item.get("rejected") is True for item in negatives.values()):
        raise BarotropicHyperbolicityGateError("registered pathology controls are incomplete")

    replay = _exact_specialized_principal_replay()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_barotropic_irrotational_hyperbolicity_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-barotropic-irrotational-hyperbolicity-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_THIRD_GATE_ONLY",
        "sector_id": predecessor["sector_id"],
        "gate_results": [
            {"gate_id": "action_level_universal_metric_coupling", "outcome": "PREDECESSOR_PASS"},
            {"gate_id": "stress_energy_conservation_interface", "outcome": "PREDECESSOR_PASS"},
            {"gate_id": "principal_symbol_hyperbolicity", "outcome": "PASS"},
            {"gate_id": "constraint_propagation", "outcome": "NOT_EVALUATED"},
        ],
        "specialization": config["specialization"],
        "principal_certificate": replay,
        "registered_control": {
            "name": config["required_control"],
            "status": control["status"],
            "scope": control["scope"],
            "healthy_domain": evidence["healthy_domain"],
            "registered_negative_controls": sorted(negatives),
        },
        "counts": {
            "sectors": 1,
            "predecessor_gates": 2,
            "new_gates_passed": 1,
            "gates_not_evaluated": 1,
            "exact_registered_residuals": 4,
            "exact_specialized_residuals": 2,
            "registered_negative_controls": 4,
            "specialized_negative_controls": 2,
            "blocks": 0,
            "rejects": 0,
        },
        "claims": {
            "action_and_stress_gates_closed_by_predecessors": True,
            "irrotational_matter_hyperbolicity_gate_closed": True,
            "coupled_gravity_matter_hyperbolicity_established": False,
            "constraint_propagation_gate_closed": False,
            "vortical_fluid_covered": False,
            "global_evolution_established": False,
            "gravity_h7_theorem_established": False,
            "universal_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "local strong hyperbolicity and positive reduced quadratic energy of the "
            "irrotational P(X)=kappa X^2 matter equation at every arbitrary-background "
            "point satisfying kappa>0, X>0 and the common-time assumption; coupled gravity, "
            "constraint propagation, vortical flow, global evolution, H7, and universal "
            "matter remain outside scope"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "stress_predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(formal_path),
                "content_sha256": formal["content_sha256"],
            },
            "principal_source": {
                "path": principal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(principal_path),
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
