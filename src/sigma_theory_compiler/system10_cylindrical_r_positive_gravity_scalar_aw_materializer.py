from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
)
from .system10_cylindrical_r_positive_domain_lift import R, _local_geometry
from .system10_cylindrical_sourced_constraint_row_materializer import (
    _atoms,
    _first,
    _second,
    _zero_tensor,
)


class System10GravityScalarAWMaterializerError(RuntimeError):
    """Raised when a fixed-r coordinate A/W row cannot be checkpointed exactly."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10GravityScalarAWMaterializerError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10GravityScalarAWMaterializerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10GravityScalarAWMaterializerError("JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10GravityScalarAWMaterializerError("path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWMaterializerError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10GravityScalarAWMaterializerError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWMaterializerError(f"bound source hash mismatch: {path}")
    return path


def _with_sha(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _canonical_sha(body)}


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-gravity-scalar-aw-materializer-config-1.0"
    ):
        raise System10GravityScalarAWMaterializerError("unsupported config schema")
    expected_caps = {
        "candidate_id": "quartic-symbol-06e267a9215345b6",
        "rows": list(range(11)),
        "wall_seconds_per_row": 120,
        "rss_bytes_per_row": 4294967296,
        "maximum_row_bytes": 8388608,
    }
    if config.get("caps") != expected_caps:
        raise System10GravityScalarAWMaterializerError("caps changed")
    bound = {
        name: _load_binding(root, binding) for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {"aw_readiness", "total_matter_action", "r_positive_domain"}:
        raise System10GravityScalarAWMaterializerError("binding manifest changed")
    readiness = bound["aw_readiness"][1]
    action = bound["total_matter_action"][1]
    domain = bound["r_positive_domain"][1]
    representative = readiness.get("materialization", {}).get("representative_candidate", {})
    if (
        readiness.get("decision") != "BLOCK_COORDINATE_ARITHMETIC_A_W_MATERIALIZER_MISSING"
        or representative.get("candidate_id") != expected_caps["candidate_id"]
        or representative.get("coefficients", {}).get("a10") != "-1/2"
        or representative.get("coefficients", {}).get("c20") != "-1"
        or representative.get("coefficients", {}).get("m2") != "1"
    ):
        raise System10GravityScalarAWMaterializerError("representative authority changed")
    if action.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY":
        raise System10GravityScalarAWMaterializerError("matter action authority changed")
    sectors = {
        item.get("sector_id")
        for item in action.get("shared_matter_action", {}).get("components", [])
    }
    if (
        not {
            "canonical_minimally_coupled_scalar",
            "source_free_maxwell",
            "barotropic_irrotational_fluid",
        }
        <= sectors
    ):
        raise System10GravityScalarAWMaterializerError("matter action sectors changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
    ):
        raise System10GravityScalarAWMaterializerError("r-positive authority changed")
    sources = {
        name: _load_source(root, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    expected_test = root / (
        "tests/test_system10_cylindrical_r_positive_gravity_scalar_aw_materializer.py"
    )
    if (
        set(sources) != {"source", "test", "nonlinear_source"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
        or sources["nonlinear_source"]
        != root / "src/sigma_theory_compiler/quartic_nonlinear_evolution_campaign.py"
    ):
        raise System10GravityScalarAWMaterializerError("source evidence changed")
    return config, representative


@cache
def _gauge_geometry() -> dict[str, Any]:
    atoms = _atoms()
    metric = sp.diag(-1, 1, R**2, 1)
    inverse = metric.inv()
    metric_first = _zero_tensor((4, 4, 4))
    metric_second = _zero_tensor((4, 4, 4, 4))
    for field, (left, right) in enumerate(SYMMETRIC_METRIC_PAIRS):
        weight = SYMMETRIC_METRIC_WEIGHTS[field]
        for derivative in range(4):
            first = _first(atoms, derivative, field) / weight
            metric_first[derivative][left][right] = first
            metric_first[derivative][right][left] = first
            for second_derivative in range(4):
                second = _second(atoms, derivative, second_derivative, field) / weight
                metric_second[derivative][second_derivative][left][right] = second
                metric_second[derivative][second_derivative][right][left] = second
    inverse_first = _zero_tensor((4, 4, 4))
    for derivative in range(4):
        for upper in range(4):
            for right in range(4):
                inverse_first[derivative][upper][right] = -sum(
                    inverse[upper, left]
                    * metric_first[derivative][left][lower]
                    * inverse[lower, right]
                    for left in range(4)
                    for lower in range(4)
                )
    connection = _zero_tensor((4, 4, 4))
    connection_first = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                bracket = [
                    metric_first[left][contracted][right]
                    + metric_first[right][contracted][left]
                    - metric_first[contracted][left][right]
                    for contracted in range(4)
                ]
                connection[upper][left][right] = (
                    sum(inverse[upper, contracted] * bracket[contracted] for contracted in range(4))
                    / 2
                )
                for derivative in range(4):
                    bracket_first = [
                        metric_second[derivative][left][contracted][right]
                        + metric_second[derivative][right][contracted][left]
                        - metric_second[derivative][contracted][left][right]
                        for contracted in range(4)
                    ]
                    connection_first[derivative][upper][left][right] = (
                        sum(
                            inverse_first[derivative][upper][contracted] * bracket[contracted]
                            + inverse[upper, contracted] * bracket_first[contracted]
                            for contracted in range(4)
                        )
                        / 2
                    )
    return {
        "metric": metric,
        "inverse": inverse,
        "metric_first": metric_first,
        "connection": connection,
        "connection_first": connection_first,
    }


@cache
def _constraint_covariant_first() -> list[list[sp.Expr]]:
    geometry = _gauge_geometry()
    metric = geometry["metric"]
    metric_first = geometry["metric_first"]
    connection = geometry["connection"]
    connection_first = geometry["connection_first"]
    tilde = sp.diag(-4, 1, R**-2, 1)
    tilde_first = _zero_tensor((4, 4, 4))
    tilde_first[1][2][2] = -2 / R**3
    reference = _zero_tensor((4, 4, 4))
    reference[1][2][2] = -R
    reference[2][1][2] = reference[2][2][1] = 1 / R
    reference_first = _zero_tensor((4, 4, 4, 4))
    reference_first[1][1][2][2] = -1
    reference_first[1][2][1][2] = reference_first[1][2][2][1] = -1 / R**2
    delta = _zero_tensor((4, 4, 4))
    delta_lower = _zero_tensor((4, 4, 4))
    delta_lower_first = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                delta[upper][left][right] = (
                    connection[upper][left][right] - reference[upper][left][right]
                )
    for lower in range(4):
        for left in range(4):
            for right in range(4):
                delta_lower[lower][left][right] = sum(
                    metric[lower, upper] * delta[upper][left][right] for upper in range(4)
                )
                for derivative in range(4):
                    delta_lower_first[derivative][lower][left][right] = sum(
                        metric_first[derivative][lower][upper] * delta[upper][left][right]
                        + metric[lower, upper]
                        * (
                            connection_first[derivative][upper][left][right]
                            - reference_first[derivative][upper][left][right]
                        )
                        for upper in range(4)
                    )
    constraint = [
        sum(
            tilde[left, right] * delta_lower[lower][left][right]
            for left in range(4)
            for right in range(4)
        )
        for lower in range(4)
    ]
    result = _zero_tensor((4, 4))
    for derivative in range(4):
        for lower in range(4):
            ordinary = sum(
                tilde_first[derivative][left][right] * delta_lower[lower][left][right]
                + tilde[left, right] * delta_lower_first[derivative][lower][left][right]
                for left in range(4)
                for right in range(4)
            )
            result[derivative][lower] = ordinary - sum(
                connection[upper][derivative][lower] * constraint[upper] for upper in range(4)
            )
    return result


def _gauge_upper(mu: int, nu: int) -> sp.Expr:
    inverse = _gauge_geometry()["inverse"]
    hat = sp.diag(-9, 1, R**-2, 1)
    constraint_first = _constraint_covariant_first()
    return -sp.Rational(1, 2) * sum(
        (
            int(lower == mu) * hat[nu, derivative]
            + int(lower == nu) * hat[mu, derivative]
            - int(mu == nu) * hat[lower, derivative]
        )
        * inverse[lower, beta]
        * constraint_first[derivative][beta]
        for lower in range(4)
        for derivative in range(4)
        for beta in range(4)
    )


@cache
def _matter_upper_unfactored() -> sp.Matrix:
    atoms = _atoms()
    inverse = _local_geometry()["inverse_metric"]
    metric = inverse.inv()
    chi = sp.Matrix([_first(atoms, derivative, 11) for derivative in range(4)])
    chi_up = inverse * chi
    x_chi = -sum(chi[index] * chi_up[index] for index in range(4)) / 2
    scalar = chi * chi.T + x_chi * metric
    field_strength = sp.Matrix(
        4,
        4,
        lambda left, right: _first(atoms, left, 12 + right) - _first(atoms, right, 12 + left),
    )
    raised_second = field_strength * inverse
    field_square = sum(
        field_strength[left, right]
        * sum(
            inverse[left, upper] * inverse[right, lower] * field_strength[upper, lower]
            for upper in range(4)
            for lower in range(4)
        )
        for left in range(4)
        for right in range(4)
    )
    maxwell = sp.Matrix(
        4,
        4,
        lambda left, right: (
            sum(field_strength[left, index] * raised_second[right, index] for index in range(4))
            - metric[left, right] * field_square / 4
        ),
    )
    tau = sp.Matrix([_first(atoms, derivative, 16) for derivative in range(4)])
    tau_up = inverse * tau
    x_fluid = -sum(tau[index] * tau_up[index] for index in range(4)) / 2
    fluid = 2 * atoms["kappa"] * x_fluid * (tau * tau.T) + atoms["kappa"] * x_fluid**2 * metric
    return inverse * (scalar + maxwell + fluid) * inverse


def _action_common() -> dict[str, Any]:
    geometry = _local_geometry()
    inverse = geometry["inverse_metric"]
    metric = geometry["metric"]
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = geometry["scalar_hessian"]
    ricci = geometry["ricci"]
    einstein = geometry["einstein"]
    curvature = geometry["scalar_curvature"]
    riemann = geometry["riemann_up"]
    x_scalar = -sum(p_down[index] * p_up[index] for index in range(4)) / 2
    theta = sum(
        inverse[left, right] * hessian[left][right] for left in range(4) for right in range(4)
    )
    hessian_squared = sum(
        inverse[left, upper] * inverse[right, lower] * hessian[left][right] * hessian[upper][lower]
        for left in range(4)
        for right in range(4)
        for upper in range(4)
        for lower in range(4)
    )
    ricci_pp = sum(
        p_up[left] * p_up[right] * ricci[left][right] for left in range(4) for right in range(4)
    )
    return locals()


def _metric_equation(row: int) -> sp.Expr:
    common = _action_common()
    mu, nu = SYMMETRIC_METRIC_PAIRS[row]
    inverse = common["inverse"]
    metric = common["metric"]
    p_down = common["p_down"]
    p_up = common["p_up"]
    hessian = common["hessian"]
    ricci = common["ricci"]
    einstein = common["einstein"]
    curvature = common["curvature"]
    riemann = common["riemann"]
    x_scalar = common["x_scalar"]
    theta = common["theta"]
    hessian_squared = common["hessian_squared"]
    ricci_pp = common["ricci_pp"]
    alpha = -sp.Rational(1, 2)
    c20 = -sp.Integer(1)
    function = sp.Rational(1, 2) + alpha * x_scalar
    g2 = x_scalar + c20 * x_scalar**2
    g2_x = 1 + 2 * c20 * x_scalar
    hessian_product = sum(
        inverse[left, right] * hessian[left][mu] * hessian[right][nu]
        for left in range(4)
        for right in range(4)
    )
    ricci_gradient = sum(
        p_up[index] * (ricci[index][mu] * p_down[nu] + ricci[index][nu] * p_down[mu])
        for index in range(4)
    )
    riemann_gradient = sum(
        p_up[first]
        * p_up[second]
        * sum(metric[mu, raised] * riemann[raised][first][nu][second] for raised in range(4))
        for first in range(4)
        for second in range(4)
    )
    lower = (
        function * einstein[mu][nu]
        - alpha * curvature * p_down[mu] * p_down[nu] / 2
        - alpha * theta * hessian[mu][nu]
        + alpha * hessian_product
        + metric[mu, nu] * alpha * (theta**2 - hessian_squared) / 2
        + alpha * ricci_gradient
        - metric[mu, nu] * alpha * ricci_pp
        + alpha * riemann_gradient
        - (metric[mu, nu] * g2 + g2_x * p_down[mu] * p_down[nu]) / 2
    )
    upper = sum(
        inverse[mu, left] * inverse[nu, right] * (lower if (left, right) == (mu, nu) else 0)
        for left in range(4)
        for right in range(4)
    )
    weight = sp.sqrt(2) if mu != nu else sp.Integer(1)
    return weight * (upper + _gauge_upper(mu, nu) - _matter_upper_unfactored()[mu, nu] / 2)


def _scalar_equation() -> sp.Expr:
    geometry = _local_geometry()
    inverse = geometry["inverse_metric"]
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = sp.Matrix(geometry["scalar_hessian"])
    einstein_upper = inverse * sp.Matrix(geometry["einstein"]) * inverse
    x_scalar = -sum(p_down[index] * p_up[index] for index in range(4)) / 2
    g2_x = 1 - 2 * x_scalar
    return -sum(
        (g2_x * inverse[mu, nu] + 2 * p_up[mu] * p_up[nu] + einstein_upper[mu, nu])
        * hessian[mu, nu]
        for mu in range(4)
        for nu in range(4)
    )


def _expression_text(expression: sp.Expr) -> str:
    return sp.sstr(expression, order="lex")


def _radial_poles(expressions: list[sp.Expr]) -> list[str]:
    forbidden = set()
    for expression in expressions:
        for power in expression.atoms(sp.Pow):
            if power.exp.is_number and power.exp.is_negative and power.base != R:
                forbidden.add(sp.sstr(power.base))
    if forbidden:
        raise System10GravityScalarAWMaterializerError(
            f"non-radial symbolic denominator detected: {sorted(forbidden)}"
        )
    return ["r=0"]


def build_row_checkpoint(
    config_path: Path, row: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, representative = _validate_config(config_path, repository)
    if row not in config["caps"]["rows"]:
        raise System10GravityScalarAWMaterializerError("row outside frozen cap")
    atoms = _atoms()
    accelerations = tuple(atoms["acceleration"][:11])
    equation = _scalar_equation() if row == 10 else _metric_equation(row)
    zero = {symbol: 0 for symbol in accelerations}
    raw_a_values = [sp.expand(sp.diff(equation, symbol)) for symbol in accelerations]
    raw_w_value = sp.expand(equation.xreplace(zero))
    affine_residual = sp.expand(
        equation
        - sum(value * symbol for value, symbol in zip(raw_a_values, accelerations, strict=True))
        - raw_w_value
    )
    if affine_residual != 0:
        raise System10GravityScalarAWMaterializerError("Euler row is not affine in accelerations")
    a_values = [value.xreplace(atoms["replacement"]) for value in raw_a_values]
    w_value = raw_w_value.xreplace(atoms["replacement"])
    if any(value.free_symbols & set(accelerations) for value in (*a_values, w_value)):
        raise System10GravityScalarAWMaterializerError("acceleration survived affine split")
    poles = _radial_poles([*a_values, w_value])
    a_entries = []
    for column, value in enumerate(a_values):
        text = _expression_text(value)
        a_entries.append(
            _with_sha(
                {"column": column, "label": f"A[{row},{column}]", "expression": text},
                "entry_sha256",
            )
        )
    w_text = _expression_text(w_value)
    w_entry = _with_sha({"label": f"W[{row}]", "expression": w_text}, "entry_sha256")
    body = {
        "schema_version": "invariant-system10-fixed-r-positive-coordinate-aw-row-1.0",
        "candidate_id": representative["candidate_id"],
        "coefficients": representative["coefficients"],
        "row": row,
        "field_pair": list(SYMMETRIC_METRIC_PAIRS[row]) if row < 10 else "gravity_scalar",
        "domain": "fixed cylindrical r>0 generic registered 85-state jet",
        "A_entries": a_entries,
        "W_entry": w_entry,
        "certificates": {
            "affine_residual": "0",
            "acceleration_free_A_entries": 11,
            "acceleration_free_W_entries": 1,
            "integrability_replacement": "partial_0 w_i[A] -> partial_i v[A]",
            "coordinate_poles": poles,
            "domain_excludes_all_poles": True,
            "global_factorization_used": False,
            "row_checkpoint_before_other_rows": True,
        },
        "source_bindings": {
            "config_sha256": _canonical_sha(config),
            "nonlinear_source_sha256": config["source_evidence"]["nonlinear_source"][
                "canonical_lf_sha256"
            ],
            "readiness_content_sha256": config["bindings"]["aw_readiness"]["content_sha256"],
        },
        "claims": {
            "single_A_W_row_materialized": True,
            "solved_acceleration_row": False,
            "all_11_rows_materialized": False,
            "full_rhs": False,
            "propagation": False,
            "hyperbolicity": False,
        },
    }
    checkpoint = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(checkpoint).encode("utf-8")) > config["caps"]["maximum_row_bytes"]:
        raise System10GravityScalarAWMaterializerError("row output cap exceeded")
    return checkpoint


def _verify_checkpoint(value: dict[str, Any], row: int, config_sha: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != _canonical_sha(body)
        or value.get("row") != row
        or value.get("source_bindings", {}).get("config_sha256") != config_sha
    ):
        raise System10GravityScalarAWMaterializerError("checkpoint seal mismatch")


def _process_rss_bytes(pid: int) -> int | None:
    if os.name != "nt":
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
        return None
    process = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
    if not process:
        return None
    counters = (ctypes.c_ulong * 10)()
    counters[0] = ctypes.sizeof(counters)
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), ctypes.sizeof(counters)
        ):
            return None
        return int(counters[2])
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def _worker(config: str, row: int, root: str, temporary: str) -> None:
    value = build_row_checkpoint(Path(config), row, root=Path(root))
    Path(temporary).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_rows(
    config_path: Path,
    output_dir: Path,
    rows: list[int],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, _ = _validate_config(config_path, repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    config_sha = _canonical_sha(config)
    context = mp.get_context("spawn")
    for row in rows:
        final = output_dir / f"row-{row:02d}.json"
        if final.exists():
            value = _load_json(final)
            _verify_checkpoint(value, row, config_sha)
            results.append(value)
            continue
        temporary = output_dir / f".row-{row:02d}.{os.getpid()}.tmp"
        if temporary.exists():
            temporary.unlink()
        process = context.Process(
            target=_worker,
            args=(str(config_path), row, str(repository), str(temporary)),
        )
        process.start()
        started = time.monotonic()
        failure = None
        while process.is_alive():
            process.join(0.1)
            if time.monotonic() - started > config["caps"]["wall_seconds_per_row"]:
                failure = "wall cap exceeded"
                break
            rss = _process_rss_bytes(process.pid)
            if rss is not None and rss > config["caps"]["rss_bytes_per_row"]:
                failure = "RSS cap exceeded"
                break
        if failure is not None:
            process.terminate()
            process.join(5)
            if temporary.exists():
                temporary.unlink()
            raise System10GravityScalarAWMaterializerError(f"row {row}: {failure}")
        if process.exitcode != 0 or not temporary.exists():
            raise System10GravityScalarAWMaterializerError(f"row {row}: worker failed")
        value = _load_json(temporary)
        _verify_checkpoint(value, row, config_sha)
        os.replace(temporary, final)
        results.append(value)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", required=True)
    arguments = parser.parse_args()
    rows = [int(item) for item in arguments.rows.split(",")]
    run_rows(arguments.config.resolve(), arguments.output_dir.resolve(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
