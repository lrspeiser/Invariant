from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
)
from .system10_cylindrical_r_positive_domain_lift import _local_geometry
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _action_common,
    _atoms,
    _canonical_lf_sha,
    _canonical_sha,
    _expression_text,
    _gauge_upper,
    _load_binding,
    _load_json,
    _matter_upper_unfactored,
    _radial_poles,
    _resolve,
    _with_sha,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _validate_config as _validate_representative_config,
)


class System10TwelveCandidateAWError(RuntimeError):
    """Raised when the twelve-candidate fixed-r A/W packet cannot be certified."""


PACKET_SCHEMA = "invariant-system10-fixed-r-positive-candidate-coordinate-aw-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-fixed-r-positive-twelve-candidate-aw-census-1.0"


def _packet_authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10TwelveCandidateAWError(f"bound source hash mismatch: {path}")
    return path


def _candidate_authority(domain: dict[str, Any]) -> list[dict[str, Any]]:
    results = domain.get("materialization", {}).get("candidate_results", [])
    candidates = [
        {
            "candidate_id": item.get("candidate_id"),
            "coefficients": {
                key: item.get("coefficients", {}).get(key) for key in ("a10", "c20", "m2")
            },
        }
        for item in results
    ]
    if len(candidates) != 12 or any(None in item["coefficients"].values() for item in candidates):
        raise System10TwelveCandidateAWError("candidate authority is incomplete")
    return candidates


def _validate_config(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10TwelveCandidateAWError("unsupported config schema")
    expected_caps = {
        "candidate_count": 12,
        "rows_per_candidate": 11,
        "columns_per_row": 11,
        "maximum_candidate_packet_bytes": 8388608,
        "maximum_candidate_wall_seconds": 180,
        "slice_r": "1",
        "tube_abs_v_10_max": "1/4",
        "real_state_variables": True,
        "maximum_receipt_bytes": 262144,
    }
    if config.get("caps") != expected_caps:
        raise System10TwelveCandidateAWError("caps changed")
    predecessor = config.get("predecessor", {})
    parent_config_path = _resolve(root, str(predecessor.get("config_path", "")))
    if _canonical_lf_sha(parent_config_path) != predecessor.get(
        "config_canonical_lf_sha256"
    ) or _canonical_sha(_load_json(parent_config_path)) != predecessor.get("config_content_sha256"):
        raise System10TwelveCandidateAWError("representative predecessor mismatch")
    parent_config, _ = _validate_representative_config(parent_config_path, root)
    _, domain = _load_binding(root, parent_config["bindings"]["r_positive_domain"])
    if config.get("candidates") != _candidate_authority(domain):
        raise System10TwelveCandidateAWError("candidate manifest changed")
    if len(config.get("expected_slice_determinants", [])) != 12:
        raise System10TwelveCandidateAWError("determinant manifest changed")
    sources = {
        name: _load_source(root, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    expected_test = root / "tests/test_system10_cylindrical_r_positive_twelve_candidate_aw.py"
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
    ):
        raise System10TwelveCandidateAWError("source evidence changed")
    return config


def _candidate_metric_equation(row: int, *, m2: sp.Expr, alpha: sp.Expr, c20: sp.Expr) -> sp.Expr:
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
    function = m2 / 2 + alpha * x_scalar
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
    gauge = m2 * _gauge_upper(mu, nu)
    matter = _matter_upper_unfactored()[mu, nu] / 2
    return weight * (upper + gauge - matter)


def _candidate_scalar_equation(*, alpha: sp.Expr, c20: sp.Expr) -> sp.Expr:
    geometry = _local_geometry()
    inverse = geometry["inverse_metric"]
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = sp.Matrix(geometry["scalar_hessian"])
    einstein_upper = inverse * sp.Matrix(geometry["einstein"]) * inverse
    x_scalar = -sum(p_down[index] * p_up[index] for index in range(4)) / 2
    g2_x = 1 + 2 * c20 * x_scalar
    return -sum(
        (
            g2_x * inverse[mu, nu]
            - 2 * c20 * p_up[mu] * p_up[nu]
            - 2 * alpha * einstein_upper[mu, nu]
        )
        * hessian[mu, nu]
        for mu in range(4)
        for nu in range(4)
    )


def _build_row(row: int, candidate: dict[str, Any]) -> dict[str, Any]:
    coefficients = candidate["coefficients"]
    m2 = sp.sympify(coefficients["m2"])
    alpha = sp.sympify(coefficients["a10"])
    c20 = sp.sympify(coefficients["c20"])
    equation = (
        _candidate_scalar_equation(alpha=alpha, c20=c20)
        if row == 10
        else _candidate_metric_equation(row, m2=m2, alpha=alpha, c20=c20)
    )
    atoms = _atoms()
    accelerations = tuple(atoms["acceleration"][:11])
    zero = {symbol: 0 for symbol in accelerations}
    raw_a = [sp.expand(sp.diff(equation, symbol)) for symbol in accelerations]
    raw_w = sp.expand(equation.xreplace(zero))
    residual = sp.expand(
        equation
        - sum(value * symbol for value, symbol in zip(raw_a, accelerations, strict=True))
        - raw_w
    )
    if residual != 0:
        raise System10TwelveCandidateAWError("Euler row is not affine in accelerations")
    a_values = [value.xreplace(atoms["replacement"]) for value in raw_a]
    w_value = raw_w.xreplace(atoms["replacement"])
    if any(value.free_symbols & set(accelerations) for value in (*a_values, w_value)):
        raise System10TwelveCandidateAWError("acceleration survived affine split")
    poles = _radial_poles([*a_values, w_value])
    a_entries = [
        _with_sha({"column": column, "expression": _expression_text(value)}, "entry_sha256")
        for column, value in enumerate(a_values)
    ]
    w_entry = _with_sha({"expression": _expression_text(w_value)}, "entry_sha256")
    body = {
        "row": row,
        "field_pair": list(SYMMETRIC_METRIC_PAIRS[row]) if row < 10 else "gravity_scalar",
        "A_entries": a_entries,
        "W_entry": w_entry,
        "certificates": {
            "affine_residual": "0",
            "coordinate_poles": poles,
            "domain_excludes_all_poles": poles == ["r=0"],
            "global_factorization_used": False,
        },
    }
    return {**body, "row_content_sha256": _canonical_sha(body)}


def build_candidate_packet(
    config_path: Path, candidate_index: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _validate_config(config_path.resolve(), repository)
    if not 0 <= candidate_index < 12:
        raise System10TwelveCandidateAWError("candidate index outside frozen cap")
    candidate = config["candidates"][candidate_index]
    started = time.monotonic()
    rows = [_build_row(row, candidate) for row in range(11)]
    elapsed = time.monotonic() - started
    if elapsed > config["caps"]["maximum_candidate_wall_seconds"]:
        raise System10TwelveCandidateAWError("candidate wall cap exceeded")
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": candidate["candidate_id"],
        "coefficients": candidate["coefficients"],
        "domain": "fixed cylindrical r>0 generic registered 85-state jet",
        "source_bindings": {"packet_authority_sha256": _packet_authority_sha(config)},
        "rows": rows,
        "row_count": 11,
        "A_entry_count": 121,
        "W_entry_count": 11,
        "claims": {
            "candidate_A_W_materialized": True,
            "candidate_invertible": False,
            "all_candidates_invertible": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_candidate_packet_bytes"]:
        raise System10TwelveCandidateAWError("candidate packet output cap exceeded")
    return packet


def _verify_packet(packet: dict[str, Any], index: int, config: dict[str, Any]) -> None:
    body = {key: value for key, value in packet.items() if key != "content_sha256"}
    if (
        packet.get("content_sha256") != _canonical_sha(body)
        or packet.get("candidate_index") != index
        or packet.get("candidate_id") != config["candidates"][index]["candidate_id"]
        or packet.get("source_bindings", {}).get("packet_authority_sha256")
        != _packet_authority_sha(config)
    ):
        raise System10TwelveCandidateAWError("candidate packet seal mismatch")
    rows = packet.get("rows", [])
    if len(rows) != 11:
        raise System10TwelveCandidateAWError("candidate row count mismatch")
    for row, value in enumerate(rows):
        row_body = {key: item for key, item in value.items() if key != "row_content_sha256"}
        if value.get("row") != row or value.get("row_content_sha256") != _canonical_sha(row_body):
            raise System10TwelveCandidateAWError("candidate row seal mismatch")


def run_candidates(
    config_path: Path,
    output_dir: Path,
    candidate_indices: list[int],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in candidate_indices:
        if not 0 <= index < 12:
            raise System10TwelveCandidateAWError("candidate index outside frozen cap")
        final = output_dir / f"candidate-{index:02d}.json"
        if final.exists():
            packet = _load_json(final)
            _verify_packet(packet, index, config)
            results.append(packet)
            continue
        packet = build_candidate_packet(config_path, index, root=repository)
        temporary = output_dir / f".candidate-{index:02d}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, final)
        results.append(packet)
    return results


def analyze_candidate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    matrix = sp.Matrix(
        [[sp.sympify(entry["expression"]) for entry in row["A_entries"]] for row in packet["rows"]]
    )
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    zero_symbols = sorted(matrix.free_symbols - {r, v_10}, key=str)
    sliced = matrix.xreplace({symbol: 0 for symbol in zero_symbols}).subs(r, 1)
    determinant = sp.factor(sliced.det(method="domain-ge"))
    x = sp.Symbol("x", real=True)
    determinant_polynomial = sp.Poly(determinant, v_10)
    if any(power[0] % 2 for power, _ in determinant_polynomial.terms()):
        raise System10TwelveCandidateAWError("candidate determinant is not even in v_10")
    determinant_x = sp.factor(
        sum(
            coefficient * x ** (power[0] // 2)
            for power, coefficient in determinant_polynomial.terms()
        )
    )
    sign_at_zero = sp.sign(determinant_x.subs(x, 0))
    if sign_at_zero not in (-1, 1):
        raise System10TwelveCandidateAWError("candidate determinant vanishes at tube center")
    absolute_branch = sp.factor(sign_at_zero * determinant_x)
    derivative = sp.factor(sp.diff(absolute_branch, x))
    reduced_derivative = derivative
    while reduced_derivative.subs(x, 0) == 0:
        reduced_derivative = sp.cancel(reduced_derivative / x)
    interior_root_count = sp.Poly(reduced_derivative, x).count_roots(
        sp.Integer(0), sp.Rational(1, 16)
    )
    midpoint_sign = sp.sign(derivative.subs(x, sp.Rational(1, 32)))
    if interior_root_count != 0 or midpoint_sign not in (-1, 1):
        raise System10TwelveCandidateAWError("tube monotonicity is not certified")
    endpoint_values = [
        sp.factor(absolute_branch.subs(x, endpoint))
        for endpoint in (sp.Integer(0), sp.Rational(1, 16))
    ]
    lower_bound = endpoint_values[1] if midpoint_sign == -1 else endpoint_values[0]
    return {
        "zero_symbols": zero_symbols,
        "determinant": determinant,
        "determinant_text": sp.sstr(determinant, order="lex"),
        "absolute_branch_x": absolute_branch,
        "absolute_branch_x_text": sp.sstr(absolute_branch, order="lex"),
        "absolute_branch_derivative_x": derivative,
        "absolute_branch_derivative_x_text": sp.sstr(derivative, order="lex"),
        "monotonicity": "nonincreasing" if midpoint_sign == -1 else "nondecreasing",
        "interior_derivative_root_count": interior_root_count,
        "endpoint_values": endpoint_values,
        "lower_bound": lower_bound,
    }


def build_census_receipt(
    config_path: Path, packets_dir: Path, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _validate_config(config_path.resolve(), repository)
    candidates = []
    common_tube_pass = True
    for index in range(12):
        packet_path = packets_dir / f"candidate-{index:02d}.json"
        packet = _load_json(packet_path)
        _verify_packet(packet, index, config)
        analysis = analyze_candidate_packet(packet)
        zero_symbols = analysis["zero_symbols"]
        determinant_text = analysis["determinant_text"]
        expected = config["expected_slice_determinants"][index]
        if determinant_text != expected["determinant"]:
            raise System10TwelveCandidateAWError("candidate determinant changed")
        lower_bound = analysis["lower_bound"]
        tube_pass = (
            lower_bound > 0
            and str(lower_bound) == expected["absolute_lower_bound"]
            and analysis["absolute_branch_derivative_x_text"]
            == expected["absolute_branch_derivative_x"]
            and analysis["monotonicity"] == expected["monotonicity"]
            and analysis["interior_derivative_root_count"] == 0
            and expected["tube_monotone_certificate"]
            == "sturm_no_interior_roots_and_exact_midpoint_sign"
        )
        common_tube_pass &= bool(tube_pass)
        candidates.append(
            {
                "candidate_index": index,
                "candidate_id": packet["candidate_id"],
                "coefficients": packet["coefficients"],
                "packet_content_sha256": packet["content_sha256"],
                "zeroed_A_symbol_count": len(zero_symbols),
                "zeroed_A_symbol_set_sha256": _canonical_sha([str(item) for item in zero_symbols]),
                "slice_determinant": determinant_text,
                "absolute_determinant_branch_x": analysis["absolute_branch_x_text"],
                "absolute_determinant_branch_derivative_x": analysis[
                    "absolute_branch_derivative_x_text"
                ],
                "absolute_determinant_monotonicity": analysis["monotonicity"],
                "interior_derivative_root_count": int(analysis["interior_derivative_root_count"]),
                "tube_abs_v_10_max": "1/4",
                "exact_absolute_determinant_lower_bound": str(lower_bound),
                "tube_admitted": bool(tube_pass),
            }
        )
    decision = (
        "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
        if common_tube_pass
        else "BLOCK_COMMON_TUBE_ONE_OR_MORE_CANDIDATES_NOT_ADMITTED"
    )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": decision,
        "source_bindings": {"config_sha256": _canonical_sha(config)},
        "counts": {
            "candidate_packets": 12,
            "rows": 132,
            "A_entries": 1452,
            "W_entries": 132,
            "tube_admitted_candidates": sum(item["tube_admitted"] for item in candidates),
        },
        "common_preregistered_tube": {
            "r": "1",
            "real_v_10_interval": ["-1/4", "1/4"],
            "all_other_candidate_A_symbols": "0",
            "all_candidates_admitted": common_tube_pass,
        },
        "candidate_results": candidates,
        "claims": {
            "all_twelve_candidate_A_W_packets_materialized": True,
            "common_local_tube_admitted": common_tube_pass,
            "global_candidate_domains_invertible": False,
            "accelerations_solved_on_common_tube": False,
            "full_rhs": False,
            "propagation": False,
            "hyperbolicity": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10TwelveCandidateAWError("census receipt output cap exceeded")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize twelve candidate fixed-r A/W packets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--candidate-indices", required=True)
    census = subparsers.add_parser("census")
    census.add_argument("--config", type=Path, required=True)
    census.add_argument("--packets-dir", type=Path, required=True)
    census.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        indices = [int(value) for value in args.candidate_indices.split(",")]
        run_candidates(args.config, args.output_dir, indices)
        return
    if args.output.exists():
        raise System10TwelveCandidateAWError("refusing to overwrite census receipt")
    receipt = build_census_receipt(args.config, args.packets_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
