from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class MaxwellArbitraryBackgroundStressError(RuntimeError):
    """Raised when the arbitrary-background Maxwell identity cannot be certified."""


Linear = dict[str, Fraction]
Monomial = tuple[str, str]
Polynomial = dict[Monomial, Fraction]


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise MaxwellArbitraryBackgroundStressError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MaxwellArbitraryBackgroundStressError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MaxwellArbitraryBackgroundStressError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise MaxwellArbitraryBackgroundStressError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise MaxwellArbitraryBackgroundStressError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise MaxwellArbitraryBackgroundStressError(f"bound content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise MaxwellArbitraryBackgroundStressError("formal report has no semantic checks")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _linear_add(*terms: Linear) -> Linear:
    result: Linear = {}
    for term in terms:
        for variable, coefficient in term.items():
            result[variable] = result.get(variable, Fraction(0)) + coefficient
            if result[variable] == 0:
                del result[variable]
    return result


def _linear_scale(term: Linear, coefficient: Fraction | int) -> Linear:
    scale = Fraction(coefficient)
    return {variable: scale * value for variable, value in term.items() if scale * value != 0}


def _polynomial_add(target: Polynomial, term: Polynomial, scale: Fraction | int = 1) -> None:
    factor = Fraction(scale)
    for monomial, coefficient in term.items():
        target[monomial] = target.get(monomial, Fraction(0)) + factor * coefficient
        if target[monomial] == 0:
            del target[monomial]


def _multiply(left: Linear, right: Linear) -> Polynomial:
    result: Polynomial = {}
    for left_name, left_coefficient in left.items():
        for right_name, right_coefficient in right.items():
            monomial = tuple(sorted((left_name, right_name)))
            result[monomial] = result.get(monomial, Fraction(0)) + (
                left_coefficient * right_coefficient
            )
            if result[monomial] == 0:
                del result[monomial]
    return result


def _field_strength(first: int, second: int) -> Linear:
    if first == second:
        return {}
    low, high = sorted((first, second))
    sign = 1 if first < second else -1
    return {f"F_{low}{high}": Fraction(sign)}


def _potential_second(component: int, first: int, second: int) -> Linear:
    low, high = sorted((first, second))
    return {f"S_{component}_{low}{high}": Fraction(1)}


def _field_strength_derivative(derivative: int, first: int, second: int) -> Linear:
    # At an arbitrary Riemann-normal-coordinate point,
    # D_a F_bc = A_c,ab - A_b,ac. The potential second jet is symmetric in a,b,
    # which builds dF=0 into an otherwise arbitrary closed two-form first jet.
    return _linear_add(
        _potential_second(second, derivative, first),
        _linear_scale(_potential_second(first, derivative, second), -1),
    )


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _first_polynomial_term(polynomial: Polynomial) -> dict[str, Any] | None:
    if not polynomial:
        return None
    monomial = min(polynomial)
    return {
        "monomial": "*".join(monomial),
        "coefficient": _format_fraction(polynomial[monomial]),
    }


def _exact_arbitrary_local_jet_replay() -> dict[str, Any]:
    dimension = 4
    signature = (-1, 1, 1, 1)

    antisymmetry_residuals = [
        _linear_add(_field_strength(mu, nu), _field_strength(nu, mu))
        for mu in range(dimension)
        for nu in range(dimension)
    ]
    bianchi_residuals = [
        _linear_add(
            _field_strength_derivative(alpha, beta, gamma),
            _field_strength_derivative(beta, gamma, alpha),
            _field_strength_derivative(gamma, alpha, beta),
        )
        for alpha in range(dimension)
        for beta in range(dimension)
        for gamma in range(dimension)
    ]
    if any(antisymmetry_residuals) or any(bianchi_residuals):
        raise MaxwellArbitraryBackgroundStressError(
            "field-strength identity parameterization failed"
        )

    identity_residuals: list[Polynomial] = []
    corrupted_trace_sign_residuals: list[Polynomial] = []
    for nu in range(dimension):
        divergence_without_trace: Polynomial = {}
        for mu in range(dimension):
            for rho in range(dimension):
                coefficient = signature[mu] * signature[rho]
                _polynomial_add(
                    divergence_without_trace,
                    _multiply(
                        _field_strength_derivative(mu, mu, rho),
                        _field_strength(nu, rho),
                    ),
                    coefficient,
                )
                _polynomial_add(
                    divergence_without_trace,
                    _multiply(
                        _field_strength(mu, rho),
                        _field_strength_derivative(mu, nu, rho),
                    ),
                    coefficient,
                )

        trace_derivative: Polynomial = {}
        for alpha in range(dimension):
            for beta in range(dimension):
                coefficient = signature[alpha] * signature[beta]
                _polynomial_add(
                    trace_derivative,
                    _multiply(
                        _field_strength_derivative(nu, alpha, beta),
                        _field_strength(alpha, beta),
                    ),
                    coefficient,
                )
                _polynomial_add(
                    trace_derivative,
                    _multiply(
                        _field_strength(alpha, beta),
                        _field_strength_derivative(nu, alpha, beta),
                    ),
                    coefficient,
                )

        force: Polynomial = {}
        for rho in range(dimension):
            euler_up: Linear = {}
            for mu in range(dimension):
                euler_up = _linear_add(
                    euler_up,
                    _linear_scale(
                        _field_strength_derivative(mu, mu, rho),
                        signature[mu] * signature[rho],
                    ),
                )
            _polynomial_add(force, _multiply(_field_strength(nu, rho), euler_up))

        correct = dict(divergence_without_trace)
        _polynomial_add(correct, trace_derivative, Fraction(-1, 4))
        _polynomial_add(correct, force, -1)
        identity_residuals.append(correct)

        corrupted = dict(divergence_without_trace)
        _polynomial_add(corrupted, trace_derivative, Fraction(1, 4))
        _polynomial_add(corrupted, force, -1)
        corrupted_trace_sign_residuals.append(corrupted)

    if any(identity_residuals):
        raise MaxwellArbitraryBackgroundStressError(
            "arbitrary local-jet Maxwell stress identity failed"
        )
    if not any(corrupted_trace_sign_residuals):
        raise MaxwellArbitraryBackgroundStressError(
            "corrupted trace-sign negative did not leave a residual"
        )

    negative_nonzero_components = sum(bool(residual) for residual in corrupted_trace_sign_residuals)
    negative_monomials = sum(len(residual) for residual in corrupted_trace_sign_residuals)
    negative_witness = next(
        _first_polynomial_term(residual) for residual in corrupted_trace_sign_residuals if residual
    )
    return {
        "dimension": dimension,
        "signature": list(signature),
        "coordinate_device": (
            "Riemann normal coordinates at an arbitrary point; tensorial vanishing then "
            "holds in every coordinate system on every smooth metric background"
        ),
        "independent_field_strength_components": 6,
        "independent_symmetric_potential_second_jets": 40,
        "field_strength_antisymmetry_residuals": 16,
        "field_strength_antisymmetry_all_zero": True,
        "differential_bianchi_residuals": 64,
        "differential_bianchi_all_zero": True,
        "identity": "nabla^mu T_mu_nu-F_nu_rho E^rho=0",
        "identity_component_residual_monomials": [len(residual) for residual in identity_residuals],
        "identity_components_all_zero": True,
        "metric_compatibility_use": (
            "nabla g=0 removes metric-derivative terms and permits index movement"
        ),
        "curvature_commutators_required": False,
        "negative_control": {
            "mutation": "reverse the Hilbert stress trace-term sign",
            "nonzero_components": negative_nonzero_components,
            "nonzero_monomials": negative_monomials,
            "first_witness": negative_witness,
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    schema = "invariant-maxwell-arbitrary-background-stress-divergence-config-1.0"
    if config.get("schema_version") != schema:
        raise MaxwellArbitraryBackgroundStressError("unsupported config schema")
    expected_policy = {
        "close_arbitrary_background_maxwell_stress_divergence": True,
        "external_current": False,
        "boundary_or_charge_control": False,
        "coupled_gravity_matter_pde": False,
        "gravity_h7": False,
        "universal_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise MaxwellArbitraryBackgroundStressError("claims policy is absent or broadened")

    predecessor_path, predecessor = _load_binding(repository, config["predecessor"])
    block = predecessor.get("arbitrary_background_block", {})
    if (
        block.get("outcome") != "BLOCK"
        or block.get("reason_code") != config["predecessor"]["required_blocker"]
    ):
        raise MaxwellArbitraryBackgroundStressError(
            "predecessor arbitrary-background blocker changed"
        )

    bindings = config["evidence_bindings"]
    formal_path, formal = _load_binding(repository, bindings["formal_controls"])
    action_path, action_ir = _load_binding(repository, bindings["proca_action_ir"])
    registered_source_path = _resolve(
        repository, bindings["registered_covariant_identity_source"]["path"]
    )
    if (
        _file_sha(registered_source_path)
        != bindings["registered_covariant_identity_source"]["file_sha256"]
    ):
        raise MaxwellArbitraryBackgroundStressError(
            "registered covariant identity source hash mismatch"
        )

    controls = _control_map(formal)
    selected: dict[str, dict[str, Any]] = {}
    for name in config["required_controls"]:
        control = controls.get(name)
        if control is None or control.get("status") != "pass":
            raise MaxwellArbitraryBackgroundStressError(f"required PASS control absent: {name}")
        selected[name] = control
    terms = action_ir.get("canonical", {}).get("terms", [])
    term_ids = {item.get("id") for item in terms if isinstance(item, dict)}
    specialization = config["specialization"]
    required_terms = {
        specialization["retained_action_term"],
        specialization["removed_action_term"],
    }
    if not required_terms <= term_ids:
        raise MaxwellArbitraryBackgroundStressError(
            "Proca action lacks massless specialization terms"
        )

    replay = _exact_arbitrary_local_jet_replay()
    source_path = Path(__file__).resolve()
    test_path = (
        repository / "tests/test_maxwell_arbitrary_background_hilbert_stress_divergence_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": ("invariant-maxwell-arbitrary-background-stress-divergence-result-1.0"),
        "campaign_id": config["campaign_id"],
        "decision": "PASS_ARBITRARY_BACKGROUND_MAXWELL_STRESS_DIVERGENCE",
        "gate_id": "dedicated_maxwell_hilbert_stress_noether_identity",
        "massless_specialization": specialization,
        "arbitrary_background_certificate": replay,
        "registered_controls": [
            {
                "name": name,
                "status": selected[name]["status"],
                "scope": selected[name]["scope"],
            }
            for name in config["required_controls"]
        ],
        "on_shell_conclusion": ("nabla^mu T_mu_nu=0 for source-free Maxwell solutions E^rho=0"),
        "counts": {
            "dimensions": 4,
            "independent_field_strength_components": 6,
            "independent_potential_second_jets": 40,
            "antisymmetry_residuals": 16,
            "bianchi_residuals": 64,
            "stress_identity_components": 4,
            "stress_identity_residual_monomials": 0,
            "registered_controls": 4,
            "action_terms_specialized": 2,
            "negative_controls": 1,
            "negative_residual_components": replay["negative_control"]["nonzero_components"],
            "blocks": 0,
            "rejects": 0,
        },
        "claims": {
            "arbitrary_background_maxwell_hilbert_stress_divergence_closed": True,
            "registered_profile_controls_remain_corroboration_only": True,
            "external_current_interface_closed": False,
            "boundary_or_charge_control_established": False,
            "coupled_gravity_matter_pde_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact source-free Maxwell Hilbert-stress divergence identity on every smooth "
            "four-dimensional metric background and arbitrary local Maxwell potential jet; "
            "external currents, boundary charges, coupled gravity-matter PDE closure, H7, "
            "universal matter, and promotion remain outside scope"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(formal_path),
                "content_sha256": formal["content_sha256"],
            },
            "proca_action_ir": {
                "path": action_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(action_path),
                "content_sha256": action_ir["content_sha256"],
            },
            "registered_covariant_identity_source": {
                "path": registered_source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(registered_source_path),
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
