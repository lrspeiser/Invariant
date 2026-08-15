from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class System10GravityScalarReadinessError(RuntimeError):
    """Raised when the candidate gravity-scalar readiness contract fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _canonical_lf_sha(path: Path) -> str:
    try:
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as exc:
        raise System10GravityScalarReadinessError(f"cannot read bound file: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10GravityScalarReadinessError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10GravityScalarReadinessError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10GravityScalarReadinessError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarReadinessError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10GravityScalarReadinessError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarReadinessError(f"bound source hash mismatch: {path}")
    return path


def _with_sha(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _canonical_sha(body)}


def _fraction(value: Any) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise System10GravityScalarReadinessError(f"invalid exact coefficient: {value}") from exc


def _text(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def _candidate_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = value.get("certificates", value.get("candidate_results", []))
    try:
        result = {str(item["candidate_id"]): item for item in items}
    except (KeyError, TypeError) as exc:
        raise System10GravityScalarReadinessError("candidate manifest malformed") from exc
    if len(result) != 12 or len(items) != 12:
        raise System10GravityScalarReadinessError("candidate census changed")
    return result


def _independent_source_replay() -> dict[str, Any]:
    import sympy as sp

    from .quartic_scalar_row_lowering_campaign import _universal_scalar_row_data

    data = _universal_scalar_row_data()
    substitution = dict(
        zip(
            data["inverse_symbols"],
            (-1, 0, 0, 0, 1, 0, 0, 1, 0, 1),
            strict=True,
        )
    )
    substitution.update({symbol: 0 for symbol in data["connection_symbols"]})
    substitution[data["connection_symbols"][17]] = -1
    substitution[data["connection_symbols"][25]] = 1
    substitution.update({symbol: 0 for symbol in data["einstein_bar_symbols"]})
    substitution.update(dict(zip(data["scalar_gradient"], (1, 0, 0, 0), strict=True)))
    substitution.update({symbol: 0 for symbol in data["scalar_second_symbols"]})
    substitution[data["scalar_second_symbols"][4]] = 1
    coefficients = [sp.factor(item.subs(substitution)) for item in data["time_row_A"]]
    remainder = sp.factor(data["remainder_W"].subs(substitution))
    expected = [
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        -data["alpha"],
        sp.Integer(0),
        -data["alpha"],
        1 + 3 * data["c20"],
    ]
    if coefficients != expected or remainder != -data["c20"] - 1:
        raise System10GravityScalarReadinessError("independent scalar-row replay changed")
    body = {
        "constructor": "quartic_scalar_row_lowering_campaign._universal_scalar_row_data",
        "profile": "cylindrical_r=1_with_registered_connection",
        "connection_substitution": {
            "Gamma^1_22": "-1",
            "Gamma^2_12=Gamma^2_21": "1",
        },
        "ordered_acceleration_coefficients": [sp.sstr(item) for item in coefficients],
        "W_phi": sp.sstr(remainder),
        "affine_residual": sp.sstr(sp.factor(data["affine_residual"].subs(substitution))),
        "exact_match": True,
    }
    return _with_sha(body, "replay_sha256")


def _validate_predecessors(
    bound: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    maxwell = bound["maxwell_rhs"][1]
    scalar = bound["scalar_row_lowering"][1]
    local_solve = bound["local_acceleration_solve"][1]
    tensor_dag = bound["metric_tensor_dag"][1]
    arithmetic = bound["lower_row_arithmetic"][1]
    domain = bound["r_positive_domain"][1]
    if (
        maxwell.get("decision") != "BOUNDED_PASS_4_MAXWELL_DYNAMIC_ROWS_BLOCK_11_GRAVITY_ROWS"
        or maxwell.get("counts", {}).get("total_rhs_rows_closed_per_candidate") != 74
        or maxwell.get("counts", {}).get("candidate_dynamic_rows_remaining") != 132
    ):
        raise System10GravityScalarReadinessError("74-row predecessor changed")
    if (
        scalar.get("status")
        != "pass_all_12_universal_scalar_row_affinity_partial_mixed_checkpoints"
        or scalar.get("generic_scalar_row_affinity_control", {}).get("identity")
        != "E_phi=A_phi,B*a_B+W_phi"
        or scalar.get("counts", {}).get("solved_source_component_derivatives") != 0
    ):
        raise System10GravityScalarReadinessError("scalar-row lowering boundary changed")
    if local_solve.get(
        "status"
    ) != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations" or any(
        item.get("time_block_determinant_nonzero") is not True
        or item.get("acceleration_solution_residual_zero") is not True
        for item in local_solve.get("certificates", [])
    ):
        raise System10GravityScalarReadinessError("local acceleration solve changed")
    if tensor_dag.get(
        "status"
    ) != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed" or any(
        item.get("coverage", {}).get("entrywise_arithmetic_materialized") is not False
        for item in tensor_dag.get("certificates", [])
    ):
        raise System10GravityScalarReadinessError("tensor-DAG materialization boundary changed")
    if arithmetic.get(
        "status"
    ) != "pass_all_12_all_lower_rows_arithmetic_mixed_tensor_fail_closed" or any(
        item.get("full_11x153_source_Jacobian_entrywise_materialized") is not False
        for item in arithmetic.get("certificates", [])
    ):
        raise System10GravityScalarReadinessError("lower-row arithmetic boundary changed")
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("materialization", {}).get("domain_certificate", {}).get("domain") != "r>0"
    ):
        raise System10GravityScalarReadinessError("r-positive domain authority changed")

    maxwell_map = _candidate_map(maxwell["materialization"])
    scalar_map = _candidate_map(scalar)
    local_map = _candidate_map(local_solve)
    tensor_map = _candidate_map(tensor_dag)
    arithmetic_map = _candidate_map(arithmetic)
    candidates = sorted(maxwell_map)
    if any(
        set(candidate_map) != set(candidates)
        for candidate_map in (
            scalar_map,
            local_map,
            tensor_map,
            arithmetic_map,
        )
    ):
        raise System10GravityScalarReadinessError("candidate identity join changed")
    coefficients: dict[str, dict[str, Any]] = {}
    for candidate_id in candidates:
        scalar_coefficients = scalar_map[candidate_id].get("coefficients", {})
        local_coefficients = local_map[candidate_id].get("coefficients", {})
        if scalar_coefficients != local_coefficients:
            raise System10GravityScalarReadinessError("candidate coefficient join changed")
        coefficients[candidate_id] = scalar_coefficients
    origins = {
        "scalar_root_packet_sha256": str(
            scalar["universal_arithmetic_tensor_packet"]["root_packet"]["content_sha256"]
        ),
        "tensor_dag_content_sha256": str(tensor_dag["content_sha256"]),
        "lower_row_arithmetic_content_sha256": str(arithmetic["content_sha256"]),
        "local_solve_content_sha256": str(local_solve["content_sha256"]),
    }
    return candidates, coefficients, origins


def _ambiguity_witness(
    candidate_id: str, coefficients: dict[str, Any], origins: dict[str, str]
) -> dict[str, Any]:
    alpha = _fraction(coefficients["a10"])
    c20 = _fraction(coefficients["c20"])
    scalar_coefficient = 1 + 3 * c20
    if alpha == 0 or scalar_coefficient == 0:
        raise System10GravityScalarReadinessError("registered ambiguity witness degenerated")
    baseline_scalar = (c20 + 1) / scalar_coefficient
    alternate_metric_22 = scalar_coefficient
    alternate_scalar = baseline_scalar + alpha

    def residual(metric_22: Fraction, metric_33: Fraction, scalar: Fraction) -> Fraction:
        return -alpha * metric_22 - alpha * metric_33 + scalar_coefficient * scalar - (c20 + 1)

    baseline_residual = residual(Fraction(0), Fraction(0), baseline_scalar)
    alternate_residual = residual(alternate_metric_22, Fraction(0), alternate_scalar)
    if baseline_residual or alternate_residual or alternate_scalar == baseline_scalar:
        raise System10GravityScalarReadinessError("ambiguity witness failed exact replay")
    body = {
        "candidate_id": candidate_id,
        "candidate_coefficients": {"alpha": _text(alpha), "c20": _text(c20)},
        "source_equation_origin": {
            "action": "G2=X+c20*X**2; G4=M2/2+alpha*X; G3=G5=0",
            "lowered_row": 10,
            "identity": "E_phi=A_phi,B*a_B+W_phi",
            **origins,
        },
        "witness_profile": {
            "coordinate_domain": "fixed cylindrical r>0",
            "radius": "1",
            "inverse_metric": "diag(-1,1,1,1)",
            "nonzero_connection": ["Gamma^1_22=-1", "Gamma^2_12=Gamma^2_21=1"],
            "scalar_gradient": ["1", "0", "0", "0"],
            "scalar_coordinate_second": {"partial_1_partial_1_phi": "1", "all_other": "0"},
            "acceleration_free_Einstein_upper": "0",
        },
        "specialized_scalar_euler": {
            "equation": ("-alpha*ag22-alpha*ag33+(1+3*c20)*aphi-(1+c20)=0"),
            "ordered_acceleration_coefficients": [
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                _text(-alpha),
                "0",
                _text(-alpha),
                _text(scalar_coefficient),
            ],
            "acceleration_order": [
                "ag00",
                "ag01",
                "ag02",
                "ag03",
                "ag11",
                "ag12",
                "ag13",
                "ag22",
                "ag23",
                "ag33",
                "aphi",
            ],
            "W_phi": _text(-(c20 + 1)),
        },
        "solution_A": {
            "ag22": "0",
            "ag33": "0",
            "aphi": _text(baseline_scalar),
            "exact_scalar_euler_residual": _text(baseline_residual),
        },
        "solution_B": {
            "ag22": _text(alternate_metric_22),
            "ag33": "0",
            "aphi": _text(alternate_scalar),
            "exact_scalar_euler_residual": _text(alternate_residual),
        },
        "scalar_acceleration_difference": _text(alpha),
        "conclusion": "SCALAR_ROW_ALONE_DOES_NOT_IDENTIFY_APHI",
        "required_missing_primitive": (
            "fixed_r_positive_candidate_11x11_A_W_materialization_and_component_10_elimination"
        ),
        "status": "BLOCK_COUPLED_METRIC_ACCELERATIONS_UNMATERIALIZED",
    }
    return _with_sha(body, "witness_sha256")


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], frozen: dict[str, Any]
) -> dict[str, Any]:
    candidates, coefficients, origins = _validate_predecessors(bound)
    replay = _independent_source_replay()
    witnesses = [
        _ambiguity_witness(candidate_id, coefficients[candidate_id], origins)
        for candidate_id in candidates
    ]
    measured = {
        "candidate_blocks": len(witnesses),
        "exact_ambiguity_witnesses": sum(
            item["solution_A"]["exact_scalar_euler_residual"] == "0"
            and item["solution_B"]["exact_scalar_euler_residual"] == "0"
            and item["scalar_acceleration_difference"] != "0"
            for item in witnesses
        ),
        "witness_set_sha256": _canonical_sha([item["witness_sha256"] for item in witnesses]),
        "rhs_rows_closed_per_candidate": 74,
        "remaining_dynamic_rows_per_candidate": 11,
        "source_replay_sha256": replay["replay_sha256"],
    }
    if measured != frozen:
        raise System10GravityScalarReadinessError("frozen gravity-scalar expectations changed")
    negatives = {
        "zero_metric_coupling": {
            "mutation": "replace both -alpha metric-acceleration coefficients by zero",
            "all_candidates_have_nonzero_alpha": all(
                item["candidate_coefficients"]["alpha"] != "0" for item in witnesses
            ),
            "rejected": True,
        },
        "promote_scalar_only_solution": {
            "mutation": "promote solution_A while leaving ag22 and ag33 unspecified",
            "distinct_exact_solution_per_candidate": True,
            "rejected": True,
        },
        "reuse_local_point_solve_as_fixed_r_row": {
            "mutation": "reinterpret one local-jet solve as a symbolic fixed-r>0 RHS row",
            "local_solve_has_serialized_fixed_r_component_10": False,
            "rejected": True,
        },
        "claim_75_rows": {
            "mutation": "count the blocked gravity-scalar row as closed",
            "registered_solved_row_instances": 0,
            "rejected": True,
        },
    }
    return {
        "target_row": {
            "row_id": "evolution_v[10]",
            "field_index": 10,
            "lhs_state_index": 27,
            "sector": "candidate_gravity_scalar",
            "candidate_instances_required": 12,
            "candidate_instances_closed": 0,
        },
        "equation_origin_status": {
            "lowered_scalar_Euler_row": True,
            "affine_in_all_11_accelerations": True,
            "abstract_coupled_solve_present": "a=-A^-1*W",
            "fixed_r_positive_component_10_materialized": False,
        },
        "independent_source_replay": replay,
        "candidate_blocks": witnesses,
        "witness_set_sha256": measured["witness_set_sha256"],
        "negative_controls": negatives,
        "stop_condition": {
            "reason": "the requested scalar acceleration is coupled to unresolved metric accelerations",
            "next_exact_primitive": (
                "materialize the complete candidate-specific 11x11 fixed-r>0 A and W, prove "
                "det(A) nonzero on a declared state domain, solve component 10, and replay "
                "the scalar plus ten metric Euler residuals"
            ),
            "metric_rows_attempted_after_block": 0,
            "status": "BLOCK_BEFORE_METRIC_ROW_EXTENSION",
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-r-positive-gravity-scalar-readiness-config-1.0"
    ):
        raise System10GravityScalarReadinessError("unsupported config schema")
    expected_caps = {
        "candidates": 12,
        "state_dimension": 85,
        "predecessor_rows": 74,
        "target_candidate_rows": 12,
        "metric_rows_after_target": 0,
        "maximum_output_bytes": 262144,
    }
    if config.get("caps") != expected_caps:
        raise System10GravityScalarReadinessError("caps changed")
    expected_claims = {
        "gravity_scalar_dynamic_row": False,
        "candidate_specific_block": True,
        "fixed_cylindrical_r_positive": True,
        "metric_dynamic_rows": False,
        "full_85_state_rhs": False,
        "constraint_propagation": False,
        "hyperbolicity": False,
        "global_theorem": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10GravityScalarReadinessError("claims policy broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "maxwell_rhs",
        "scalar_row_lowering",
        "local_acceleration_solve",
        "metric_tensor_dag",
        "lower_row_arithmetic",
        "r_positive_domain",
    }:
        raise System10GravityScalarReadinessError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"source", "test"}:
        raise System10GravityScalarReadinessError("source evidence manifest changed")
    expected_test = repository / (
        "tests/test_system10_cylindrical_r_positive_gravity_scalar_dynamic_rhs_readiness.py"
    )
    if sources["source"] != Path(__file__).resolve() or sources["test"] != expected_test:
        raise System10GravityScalarReadinessError("self evidence path changed")
    materialization = _materialize(bound, config.get("frozen_expectations", {}))
    body = {
        "schema_version": (
            "invariant-system10-cylindrical-r-positive-gravity-scalar-readiness-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BLOCK_GRAVITY_SCALAR_REQUIRES_COUPLED_11X11_FIXED_R_SOLVE",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_dimension": 85,
            "predecessor_rhs_rows_per_candidate": 74,
            "gravity_scalar_rows_registered": 0,
            "exact_candidate_ambiguity_witnesses": 12,
            "total_rhs_rows_closed_per_candidate": 74,
            "dynamic_rows_remaining_per_candidate": 11,
            "candidate_dynamic_rows_remaining": 132,
            "metric_rows_attempted_after_block": 0,
            "negative_controls": 4,
        },
        "claims": {
            "gravity_scalar_dynamic_row_closed": False,
            "candidate_specific_missing_primitive_sealed": True,
            "fixed_cylindrical_r_positive_witness_closed": True,
            "metric_dynamic_rows_closed": False,
            "full_85_state_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Fail-closed readiness result for candidate gravity-scalar row evolution_v[10]. "
            "The bound authorities prove an exact scalar Euler row affine in eleven "
            "accelerations and an abstract coupled solve, but do not serialize the fixed-r>0 "
            "component-10 elimination. At the admissible cylindrical r=1 profile, every one "
            "of the twelve candidates has two exact scalar-Euler solutions with different "
            "scalar accelerations and compensating metric accelerations. Therefore no 75th "
            "RHS row is registered and metric-row extension stops. No full RHS, propagation, "
            "hyperbolicity, global, or promotion claim is made."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "canonical_lf_sha256": _canonical_lf_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10GravityScalarReadinessError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists() and output_path.read_bytes() != data:
        raise System10GravityScalarReadinessError("immutable output conflict")
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
