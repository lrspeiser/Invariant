from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_gravity_scalar_aw_invertibility import (
    _load_matrix,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_invertibility import (
    _validate_config as _validate_invertibility_config,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_invertibility import (
    build_receipt as build_invertibility_receipt,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10GravityScalarAWNonsingularTubeError(RuntimeError):
    """Raised when the representative nonsingular-tube solve cannot be replayed."""


SCHEMA = "invariant-system10-cylindrical-r-positive-gravity-scalar-aw-tube-solve-1.0"
DECISION = "BOUNDED_PASS_REPRESENTATIVE_A_W_SOLVE_ON_PREREGISTERED_NONSINGULAR_TUBE"


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWNonsingularTubeError(f"bound source hash mismatch: {path}")
    return path


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{SCHEMA}-config":
        raise System10GravityScalarAWNonsingularTubeError("unsupported config schema")
    expected_caps = {
        "candidate_id": "quartic-symbol-06e267a9215345b6",
        "r": "1",
        "v_10_interval": ["-1/2", "1/2"],
        "real_state_variables": True,
        "zero_symbol_count": 57,
        "zero_symbol_set_sha256": (
            "a340f5bb46ca7e39cdb102bf4c7090e099f81101dabff507b1b4bdadd234c96f"
        ),
        "determinant_absolute_lower_bound": "3486784401/268435456",
        "acceleration_count": 11,
        "residual_count": 11,
        "maximum_receipt_bytes": 65536,
    }
    if config.get("caps") != expected_caps:
        raise System10GravityScalarAWNonsingularTubeError("caps changed")
    if config.get("ordered_w_entry_set_sha256") != (
        "0c4f4f277d82205c143766cca903a8d1150c3012cbe25c34223d471871eedf57"
    ):
        raise System10GravityScalarAWNonsingularTubeError("ordered W cap changed")
    predecessor = config.get("predecessor", {})
    prior_config_path = _resolve(root, str(predecessor.get("config_path", "")))
    if _canonical_lf_sha(prior_config_path) != predecessor.get(
        "config_canonical_lf_sha256"
    ) or _canonical_sha(_load_json(prior_config_path)) != predecessor.get("config_content_sha256"):
        raise System10GravityScalarAWNonsingularTubeError("predecessor config mismatch")
    prior_config = _validate_invertibility_config(prior_config_path, root)
    prior_receipt_path = _resolve(root, str(predecessor.get("receipt_path", "")))
    prior_receipt = _load_json(prior_receipt_path)
    if (
        _canonical_lf_sha(prior_receipt_path) != predecessor.get("receipt_canonical_lf_sha256")
        or prior_receipt.get("content_sha256") != predecessor.get("receipt_content_sha256")
        or build_invertibility_receipt(prior_config_path, root=root) != prior_receipt
    ):
        raise System10GravityScalarAWNonsingularTubeError("predecessor receipt mismatch")
    if prior_receipt.get("decision") != "BLOCK_GLOBAL_ACCELERATION_SOLVE_EXACT_SINGULAR_A_WITNESS":
        raise System10GravityScalarAWNonsingularTubeError("predecessor decision changed")
    sources = {
        name: _load_source(root, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    expected_test = root / (
        "tests/test_system10_cylindrical_r_positive_gravity_scalar_aw_nonsingular_tube.py"
    )
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
    ):
        raise System10GravityScalarAWNonsingularTubeError("source evidence changed")
    return config, prior_config


def _sealed_w_inputs(
    prior_config: dict[str, Any], root: Path, expected_ordered_sha: str
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    entry_seals: list[str] = []
    for row, binding in enumerate(prior_config["aw_packet"]["rows"]):
        checkpoint = _load_json(_resolve(root, binding["path"]))
        w_entry = checkpoint["W_entry"]
        if checkpoint["certificates"]["acceleration_free_W_entries"] != 1:
            raise System10GravityScalarAWNonsingularTubeError("W acceleration-free seal changed")
        entry_seal = str(w_entry["entry_sha256"])
        entry_seals.append(entry_seal)
        bindings.append(
            {
                "symbol": f"W_{row}",
                "meaning": f"sealed W[{row}] evaluated on the preregistered tube slice",
                "source_row": row,
                "source_row_content_sha256": checkpoint["content_sha256"],
                "source_W_entry_sha256": entry_seal,
            }
        )
    ordered_sha = hashlib.sha256("".join(entry_seals).encode("ascii")).hexdigest()
    if ordered_sha != expected_ordered_sha:
        raise System10GravityScalarAWNonsingularTubeError("ordered W binding changed")
    return bindings


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = root or Path(__file__).resolve().parents[2]
    config, prior_config = _validate_config(config_path.resolve(), repository.resolve())
    matrix = _load_matrix(prior_config, repository)
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10", real=True)
    source_v_10 = sp.Symbol("v_10")
    zero_symbols = sorted(matrix.free_symbols - {r, source_v_10}, key=str)
    zero_names = [str(symbol) for symbol in zero_symbols]
    if (
        len(zero_names) != config["caps"]["zero_symbol_count"]
        or _canonical_sha(zero_names) != config["caps"]["zero_symbol_set_sha256"]
    ):
        raise System10GravityScalarAWNonsingularTubeError("tube symbol set changed")
    tube_matrix = matrix.xreplace({symbol: sp.Integer(0) for symbol in zero_symbols}).subs(r, 1)
    tube_matrix = tube_matrix.xreplace({source_v_10: v_10})
    determinant = sp.factor(tube_matrix.det(method="domain-ge"))
    determinant_text = sp.sstr(determinant, order="lex")
    if determinant_text != prior_config["caps"]["expected_slice_determinant"]:
        raise System10GravityScalarAWNonsingularTubeError("tube determinant changed")
    x = sp.Symbol("x", real=True)
    absolute_determinant_x = sp.Rational(6561, 16384) * (x + 2) ** 6 * (1 - 3 * x)
    derivative = sp.factor(sp.diff(absolute_determinant_x, x))
    lower_bound = sp.factor(absolute_determinant_x.subs(x, sp.Rational(1, 4)))
    if derivative != -sp.Rational(137781, 16384) * x * (x + 2) ** 5 or lower_bound != sp.Rational(
        3486784401, 268435456
    ):
        raise System10GravityScalarAWNonsingularTubeError("determinant margin replay failed")
    w_symbols = sp.Matrix(sp.symbols("W_0:11"))
    accelerations = [sp.factor(value) for value in tube_matrix.inv() * (-w_symbols)]
    acceleration_text = [sp.sstr(value, order="lex") for value in accelerations]
    if acceleration_text != config["expected_accelerations"]:
        raise System10GravityScalarAWNonsingularTubeError("acceleration solve changed")
    residuals = [sp.factor(value) for value in tube_matrix * sp.Matrix(accelerations) + w_symbols]
    if residuals != [sp.Integer(0)] * 11:
        raise System10GravityScalarAWNonsingularTubeError("residual replay failed")
    w_bindings = _sealed_w_inputs(prior_config, repository, config["ordered_w_entry_set_sha256"])
    acceleration_entries = [
        {
            "row": row,
            "label": f"partial_0_v_{row}",
            "expression": expression,
            "entry_sha256": _canonical_sha({"row": row, "expression": expression}),
        }
        for row, expression in enumerate(acceleration_text)
    ]
    residual_entries = [
        {
            "row": row,
            "expression": "0",
            "entry_sha256": _canonical_sha({"row": row, "expression": "0"}),
        }
        for row in range(11)
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "candidate": {
            "candidate_id": config["caps"]["candidate_id"],
            "scope": "representative_candidate_on_preregistered_tube_only",
        },
        "source_bindings": {
            "config_sha256": _canonical_sha(config),
            "predecessor_receipt_content_sha256": config["predecessor"]["receipt_content_sha256"],
            "ordered_aw_row_set_sha256": prior_config["aw_packet"]["ordered_row_set_sha256"],
            "ordered_w_entry_set_sha256": config["ordered_w_entry_set_sha256"],
        },
        "preregistered_tube": {
            "r": "1",
            "real_v_10_interval": ["-1/2", "1/2"],
            "zeroed_symbol_count": len(zero_names),
            "zeroed_symbols": zero_names,
            "zeroed_symbol_set_sha256": _canonical_sha(zero_names),
            "W_semantics": (
                "W_i is the explicitly sealed acceleration-free W[i] expression after the same "
                "tube specialization; elimination treats its value algebraically"
            ),
        },
        "invertibility_certificate": {
            "determinant": determinant_text,
            "real_substitution": "x=v_10**2 in [0,1/4]",
            "absolute_determinant_as_x": sp.sstr(absolute_determinant_x, order="lex"),
            "derivative_as_x": sp.sstr(derivative, order="lex"),
            "monotonicity": "nonincreasing because -137781*x*(x+2)**5/16384 <= 0",
            "exact_absolute_lower_bound": str(lower_bound),
            "lower_bound_attained_at": "v_10**2=1/4",
            "nonzero_denominators": {
                "v_10**2+2": ">=2",
                "1-3*v_10**2": ">=1/4",
                "18": "nonzero",
            },
        },
        "sealed_W_inputs": w_bindings,
        "accelerations": acceleration_entries,
        "residual_replay": {
            "equation": "A * partial_0_v + W",
            "entries": residual_entries,
            "all_zero": True,
            "count": 11,
        },
        "claims": {
            "representative_tube_invertible": True,
            "representative_tube_all_11_accelerations_solved": True,
            "representative_tube_all_11_residuals_replayed": True,
            "global_representative_domain_invertible": False,
            "other_candidates_solved": False,
            "full_rhs": False,
            "propagation": False,
            "hyperbolicity": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10GravityScalarAWNonsingularTubeError("receipt output cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise System10GravityScalarAWNonsingularTubeError("refusing to overwrite receipt")
    receipt = build_receipt(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve representative A/W on nonsingular tube")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_receipt(args.config, args.output)


if __name__ == "__main__":
    main()
