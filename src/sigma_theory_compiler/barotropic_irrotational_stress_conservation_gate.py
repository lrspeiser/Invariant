from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class BarotropicStressConservationError(RuntimeError):
    """Raised when the registered stress-conservation chain is incomplete or altered."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise BarotropicStressConservationError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BarotropicStressConservationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BarotropicStressConservationError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise BarotropicStressConservationError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise BarotropicStressConservationError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise BarotropicStressConservationError(f"bound content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise BarotropicStressConservationError("formal report has no semantic checks")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _specialized_noether_replay() -> dict[str, Any]:
    # Basis for each free covector component nu after differentiating T_mu_nu:
    #   B0=(p^a p^b H_ab)p_nu, B1=X theta p_nu, B2=p^a H_a_nu.
    # Coefficients suppress the common factor kappa. P_X=2*kappa*X and
    # nabla_a X=-p^b H_ab. Hessian symmetry cancels the B2 terms exactly.
    divergence_terms = {
        "nabla_P_X_times_pp": (-2, 0, 0),
        "P_X_box_times_p": (0, 2, 0),
        "P_X_p_times_H": (0, 0, 2),
        "nabla_pressure": (0, 0, -2),
    }
    divergence = tuple(sum(term[index] for term in divergence_terms.values()) for index in range(3))
    euler_times_gradient = (-2, 2, 0)
    residual = tuple(left - right for left, right in zip(divergence, euler_times_gradient))

    corrupted_pressure = (0, 0, 2)
    corrupted_divergence = tuple(
        divergence[index] - divergence_terms["nabla_pressure"][index] + corrupted_pressure[index]
        for index in range(3)
    )
    corrupted_residual = tuple(
        left - right for left, right in zip(corrupted_divergence, euler_times_gradient)
    )
    if residual != (0, 0, 0) or corrupted_residual == (0, 0, 0):
        raise BarotropicStressConservationError("specialized Noether replay failed")
    return {
        "basis": ["(pHp)p_nu", "X box(tau) p_nu", "p^mu H_mu_nu"],
        "divergence_term_coefficients": {
            name: list(coefficients) for name, coefficients in divergence_terms.items()
        },
        "divergence_coefficients": list(divergence),
        "euler_times_gradient_coefficients": list(euler_times_gradient),
        "off_shell_identity_residual": list(residual),
        "on_shell_conclusion": "nabla^mu T_mu_nu=0 when E_tau=0",
        "negative_control": {
            "mutation": "reverse the pressure-gradient sign",
            "residual": list(corrupted_residual),
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    schema = "invariant-barotropic-irrotational-stress-conservation-config-1.0"
    if config.get("schema_version") != schema:
        raise BarotropicStressConservationError("unsupported config schema")
    expected_policy = {
        "close_stress_conservation_only": True,
        "hyperbolicity": False,
        "constraint_propagation": False,
        "vortical_fluid": False,
        "gravity_h7": False,
        "universal_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise BarotropicStressConservationError("claims policy is absent or broadened")

    predecessor_path, predecessor = _load_binding(repository, config["predecessor"])
    if predecessor.get("decision") != config["predecessor"]["required_decision"]:
        raise BarotropicStressConservationError("action predecessor decision changed")
    if predecessor.get("claims", {}).get("earliest_action_level_gate_closed") is not True:
        raise BarotropicStressConservationError("action predecessor did not close its gate")
    if predecessor.get("claims", {}).get("stress_conservation_gate_closed") is not False:
        raise BarotropicStressConservationError("predecessor conservation state changed")

    bindings = config["evidence_bindings"]
    formal_path, formal = _load_binding(repository, bindings["formal_controls"])
    variation_path = _resolve(repository, bindings["variation_source"]["path"])
    if _file_sha(variation_path) != bindings["variation_source"]["file_sha256"]:
        raise BarotropicStressConservationError("variation source hash mismatch")
    control = _control_map(formal).get(config["required_control"])
    if control is None or control.get("status") != "pass":
        raise BarotropicStressConservationError("generic G2 Noether control is not PASS")
    evidence = control.get("evidence", {})
    if evidence.get("local_jet_residuals") != ["0"] * 4:
        raise BarotropicStressConservationError("generic G2 Noether residuals are not zero")
    if evidence.get("corrupted_sign_rejected") is not True:
        raise BarotropicStressConservationError("generic G2 negative control is absent")
    expected_scope_fragment = "arbitrary local scalar-gradient/Hessian jet"
    if expected_scope_fragment not in str(evidence.get("scope", "")):
        raise BarotropicStressConservationError("generic G2 arbitrary-point scope is absent")

    replay = _specialized_noether_replay()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_barotropic_irrotational_stress_conservation_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-barotropic-irrotational-stress-conservation-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_SECOND_GATE_ONLY",
        "sector_id": predecessor["sector_id"],
        "gate_results": [
            {
                "gate_id": "action_level_universal_metric_coupling",
                "outcome": "PREDECESSOR_PASS",
            },
            {
                "gate_id": "stress_energy_conservation_interface",
                "outcome": "PASS",
            },
            {
                "gate_id": "principal_symbol_hyperbolicity",
                "outcome": "NOT_EVALUATED",
            },
            {
                "gate_id": "constraint_propagation",
                "outcome": "NOT_EVALUATED",
            },
        ],
        "specialization": config["specialization"],
        "covariant_identity": {
            "off_shell": "nabla^mu T_mu_nu=E_tau nabla_nu(tau)",
            "on_shell": "nabla^mu T_mu_nu=0 when E_tau=0",
            "arbitrary_background_basis": (
                "tensorial identity evaluated in Riemann normal coordinates at an arbitrary "
                "point; curvature is unrestricted and does not enter this first-derivative sector"
            ),
            "registered_local_jet_residuals": evidence["local_jet_residuals"],
            "specialized_replay": replay,
        },
        "counts": {
            "sectors": 1,
            "predecessor_gates": 1,
            "new_gates_passed": 1,
            "gates_not_evaluated": 2,
            "registered_exact_residuals": 4,
            "specialized_exact_residual_coefficients": 3,
            "negative_controls": 2,
            "blocks": 0,
            "rejects": 0,
        },
        "claims": {
            "action_level_gate_closed_by_predecessor": True,
            "stress_conservation_gate_closed": True,
            "hyperbolicity_gate_closed": False,
            "constraint_propagation_gate_closed": False,
            "vortical_fluid_covered": False,
            "gravity_h7_theorem_established": False,
            "universal_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact covariant on-shell Hilbert stress conservation for the admitted "
            "irrotational P(X)=kappa X^2 matter patch only; later PDE/constraint gates, "
            "vortical fluids, H7, universal matter, and promotion remain outside scope"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "action_predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(formal_path),
                "content_sha256": formal["content_sha256"],
            },
            "variation_source": {
                "path": variation_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(variation_path),
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
