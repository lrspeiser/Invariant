from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class BarotropicActionGateError(RuntimeError):
    """Raised when the bounded fluid action evidence is missing or altered."""


_RECEIPT_KEYS = {
    "admitted_action",
    "campaign_id",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "exact_replay",
    "gate_results",
    "registered_variation_control",
    "schema_version",
    "scope",
    "sector_id",
    "source_bindings",
}


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise BarotropicActionGateError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BarotropicActionGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BarotropicActionGateError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise BarotropicActionGateError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise BarotropicActionGateError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    content_sha = binding.get("content_sha256")
    if content_sha is not None and value.get("content_sha256") != content_sha:
        raise BarotropicActionGateError(f"bound content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise BarotropicActionGateError("formal report has no semantic checks")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _fluid_replay() -> dict[str, Any]:
    # Coefficient arithmetic for P=kappa*X^n at n=2. Keeping kappa*X^2 as one
    # formal unit makes every check an exact integer coefficient identity.
    exponent = 2
    pressure_coefficient = 1
    p_x_times_x_coefficient = exponent
    density_coefficient = 2 * p_x_times_x_coefficient - pressure_coefficient
    enthalpy_coefficient = density_coefficient + pressure_coefficient
    perfect_fluid_uu_coefficient = 2 * p_x_times_x_coefficient
    eos_residual = density_coefficient - 3 * pressure_coefficient
    stress_residual = enthalpy_coefficient - perfect_fluid_uu_coefficient

    wrong_density_coefficient = p_x_times_x_coefficient - pressure_coefficient
    wrong_eos_residual = wrong_density_coefficient - 3 * pressure_coefficient
    if eos_residual != 0 or stress_residual != 0 or wrong_eos_residual == 0:
        raise BarotropicActionGateError("exact fluid action replay failed")
    return {
        "formal_unit": "kappa X^2",
        "P_coefficient": pressure_coefficient,
        "X_P_X_coefficient": p_x_times_x_coefficient,
        "rho_coefficient_from_2X_P_X_minus_P": density_coefficient,
        "rho": "3 kappa X^2",
        "p": "kappa X^2",
        "equation_of_state": "p=rho/3",
        "equation_of_state_residual": str(eos_residual),
        "normalized_velocity": "u_mu=nabla_mu(tau)/sqrt(2X)",
        "hilbert_stress": "T_mu_nu=P_X nabla_mu(tau)nabla_nu(tau)+P g_mu_nu",
        "perfect_fluid_stress": "T_mu_nu=(rho+p)u_mu u_nu+p g_mu_nu",
        "stress_decomposition_residual": str(stress_residual),
        "negative_control": {
            "mutation": "omit one factor of X P_X in rho",
            "equation_of_state_residual": str(wrong_eos_residual),
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    schema = "invariant-barotropic-irrotational-action-gate-config-1.0"
    if config.get("schema_version") != schema:
        raise BarotropicActionGateError("unsupported config schema")

    expected_policy = {
        "close_only_action_level_gate": True,
        "stress_conservation_gate": "NOT_EVALUATED",
        "hyperbolicity_gate": "NOT_EVALUATED",
        "constraint_propagation_gate": "NOT_EVALUATED",
        "arbitrary_background_closure": False,
        "gravity_h7": False,
        "universal_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise BarotropicActionGateError("claims policy is absent or broadened")

    universal_path, universal = _load_binding(
        repository, config["predecessors"]["universal_matter"]
    )
    maxwell_path, maxwell = _load_binding(repository, config["predecessors"]["maxwell_followup"])
    fluid = next(
        (
            item
            for item in universal.get("sector_results", [])
            if item.get("sector_id") == "barotropic_perfect_fluid"
        ),
        None,
    )
    if not isinstance(fluid, dict):
        raise BarotropicActionGateError("predecessor fluid sector is absent")
    reason_codes = fluid.get("first_blocker", {}).get("reason_codes", [])
    if "missing_admitted_variational_matter_action" not in reason_codes:
        raise BarotropicActionGateError("predecessor action-level fluid blocker is absent")
    if maxwell.get("claims", {}).get("universal_matter_closure_established") is not False:
        raise BarotropicActionGateError("Maxwell predecessor scope is broader than registered")

    formal_path, formal = _load_binding(repository, config["evidence_bindings"]["formal_controls"])
    controls = _control_map(formal)
    control_name = config["required_control"]
    control = controls.get(control_name)
    if control is None or control.get("status") != "pass":
        raise BarotropicActionGateError("required generic G2 variation control is not PASS")
    evidence = control.get("evidence", {})
    if evidence.get("x_definition") != "x=-nabla_mu(phi)nabla^mu(phi)/2 in Lambda_phi=1 units":
        raise BarotropicActionGateError("registered G2 kinetic convention changed")
    if evidence.get("local_jet_residuals") != ["0"] * 4:
        raise BarotropicActionGateError("registered G2 variation residuals are not zero")
    if evidence.get("corrupted_sign_rejected") is not True:
        raise BarotropicActionGateError("registered G2 variation negative is absent")

    sector = config["sector"]
    required_dependencies = {"g_mu_nu", "tau", "kappa"}
    if set(sector.get("dependencies", [])) != required_dependencies:
        raise BarotropicActionGateError("fluid action dependency manifest changed")
    replay = _fluid_replay()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_barotropic_irrotational_action_gate.py"

    body: dict[str, Any] = {
        "schema_version": "invariant-barotropic-irrotational-action-gate-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EARLIEST_GATE_ONLY",
        "sector_id": sector["sector_id"],
        "gate_results": [
            {
                "gate_id": "action_level_universal_metric_coupling",
                "outcome": "PASS",
                "reason_codes": [],
            },
            {
                "gate_id": "stress_energy_conservation_interface",
                "outcome": "NOT_EVALUATED",
                "reason_codes": ["outside_single_gate_scope"],
            },
            {
                "gate_id": "principal_symbol_hyperbolicity",
                "outcome": "NOT_EVALUATED",
                "reason_codes": ["outside_single_gate_scope"],
            },
            {
                "gate_id": "constraint_propagation",
                "outcome": "NOT_EVALUATED",
                "reason_codes": ["outside_single_gate_scope"],
            },
        ],
        "admitted_action": {
            "density": sector["action_density"],
            "kinetic_scalar": sector["kinetic_scalar"],
            "pressure_function": sector["pressure_function"],
            "physical_metric": sector["physical_metric"],
            "dependencies": sorted(required_dependencies),
            "forbidden_gravitational_or_species_dependencies": [],
            "maximum_derivatives_per_matter_field": 1,
            "domain": sector["domain"],
            "scope_restriction": sector["scope_restriction"],
        },
        "exact_replay": replay,
        "registered_variation_control": {
            "name": control_name,
            "status": control["status"],
            "scope": control["scope"],
            "metric_stress_tensor": evidence.get("metric_stress_tensor"),
            "local_jet_residuals": evidence["local_jet_residuals"],
            "corrupted_sign_rejected": evidence["corrupted_sign_rejected"],
        },
        "counts": {
            "sectors": 1,
            "gates_passed": 1,
            "gates_not_evaluated": 3,
            "exact_integer_residuals": 2,
            "exact_registered_variation_residuals": 4,
            "negative_controls": 1,
            "registered_formal_controls": 1,
            "blocks": 0,
            "rejects": 0,
        },
        "claims": {
            "earliest_action_level_gate_closed": True,
            "all_barotropic_flows_covered": False,
            "vortical_flows_covered": False,
            "stress_conservation_gate_closed": False,
            "hyperbolicity_gate_closed": False,
            "constraint_propagation_gate_closed": False,
            "arbitrary_background_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "standard first-derivative shift-symmetric velocity-potential action for an "
            "irrotational isentropic radiation-like barotrope on kappa>0, X>0 only; "
            "no later fluid gate, vortical sector, H7, or universal-matter claim"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "universal_predecessor": {
                "path": universal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(universal_path),
                "content_sha256": universal["content_sha256"],
            },
            "maxwell_predecessor": {
                "path": maxwell_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(maxwell_path),
                "content_sha256": maxwell["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(formal_path),
                "content_sha256": formal["content_sha256"],
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


def validate_receipt(
    receipt: dict[str, Any], config_path: Path, *, root: Path | None = None
) -> None:
    """Validate the closed receipt seal and exact live replay."""
    if set(receipt) != _RECEIPT_KEYS:
        raise BarotropicActionGateError("barotropic receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != _canonical_sha(body):
        raise BarotropicActionGateError("barotropic receipt content seal changed")
    if receipt != build_receipt(config_path, root=root):
        raise BarotropicActionGateError("barotropic receipt immutable replay changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
