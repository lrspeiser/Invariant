from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
    _verify_checkpoint,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _validate_config as _validate_aw_config,
)


class System10GravityScalarAWInvertibilityError(RuntimeError):
    """Raised when the representative A/W invertibility receipt cannot be replayed."""


SCHEMA = "invariant-system10-cylindrical-r-positive-gravity-scalar-aw-invertibility-1.0"
DECISION = "BLOCK_GLOBAL_ACCELERATION_SOLVE_EXACT_SINGULAR_A_WITNESS"


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10GravityScalarAWInvertibilityError(f"bound source hash mismatch: {path}")
    return path


def _validate_config(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{SCHEMA}-config":
        raise System10GravityScalarAWInvertibilityError("unsupported config schema")
    expected_caps = {
        "candidate_id": "quartic-symbol-06e267a9215345b6",
        "row_count": 11,
        "column_count": 11,
        "slice_r": "1",
        "slice_free_symbol": "v_10",
        "witness_v_10": "sqrt(3)/3",
        "zero_symbol_count": 57,
        "zero_symbol_set_sha256": (
            "a340f5bb46ca7e39cdb102bf4c7090e099f81101dabff507b1b4bdadd234c96f"
        ),
        "expected_slice_determinant": ("-6561*(v_10**2 + 2)**6*(3*v_10**2 - 1)/16384"),
        "maximum_receipt_bytes": 65536,
    }
    if config.get("caps") != expected_caps:
        raise System10GravityScalarAWInvertibilityError("caps changed")
    packet = config.get("aw_packet", {})
    packet_config = _resolve(root, str(packet.get("config_path", "")))
    if _canonical_lf_sha(packet_config) != packet.get(
        "config_canonical_lf_sha256"
    ) or _canonical_sha(_load_json(packet_config)) != packet.get("config_content_sha256"):
        raise System10GravityScalarAWInvertibilityError("A/W config binding mismatch")
    _validate_aw_config(packet_config, root)
    row_bindings = packet.get("rows")
    if not isinstance(row_bindings, list) or len(row_bindings) != 11:
        raise System10GravityScalarAWInvertibilityError("A/W row manifest changed")
    config_sha = _canonical_sha(_load_json(packet_config))
    content_seals: list[str] = []
    for row, binding in enumerate(row_bindings):
        path = _resolve(root, str(binding.get("path", "")))
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10GravityScalarAWInvertibilityError(f"A/W row hash mismatch: {row}")
        checkpoint = _load_json(path)
        _verify_checkpoint(checkpoint, row, config_sha)
        if checkpoint.get("content_sha256") != binding.get("content_sha256"):
            raise System10GravityScalarAWInvertibilityError(f"A/W row seal mismatch: {row}")
        content_seals.append(str(checkpoint["content_sha256"]))
    row_set_sha = hashlib.sha256("".join(content_seals).encode("ascii")).hexdigest()
    if row_set_sha != packet.get("ordered_row_set_sha256"):
        raise System10GravityScalarAWInvertibilityError("A/W row-set seal mismatch")
    sources = {
        name: _load_source(root, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    expected_test = root / (
        "tests/test_system10_cylindrical_r_positive_gravity_scalar_aw_invertibility.py"
    )
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
    ):
        raise System10GravityScalarAWInvertibilityError("source evidence changed")
    return config


def _load_matrix(config: dict[str, Any], root: Path) -> sp.Matrix:
    rows: list[list[sp.Expr]] = []
    for binding in config["aw_packet"]["rows"]:
        checkpoint = _load_json(_resolve(root, binding["path"]))
        entries = checkpoint.get("A_entries", [])
        if len(entries) != 11 or [entry.get("column") for entry in entries] != list(range(11)):
            raise System10GravityScalarAWInvertibilityError("A/W matrix shape changed")
        rows.append([sp.sympify(str(entry["expression"])) for entry in entries])
    return sp.Matrix(rows)


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = root or Path(__file__).resolve().parents[2]
    config = _validate_config(config_path.resolve(), repository.resolve())
    matrix = _load_matrix(config, repository)
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    zero_symbols = sorted(matrix.free_symbols - {r, v_10}, key=str)
    zero_names = [str(symbol) for symbol in zero_symbols]
    if (
        len(zero_symbols) != config["caps"]["zero_symbol_count"]
        or _canonical_sha(zero_names) != config["caps"]["zero_symbol_set_sha256"]
    ):
        raise System10GravityScalarAWInvertibilityError("specialization symbol set changed")
    slice_matrix = matrix.xreplace({symbol: sp.Integer(0) for symbol in zero_symbols}).subs(r, 1)
    determinant = sp.factor(slice_matrix.det(method="domain-ge"))
    determinant_text = sp.sstr(determinant, order="lex")
    if determinant_text != config["caps"]["expected_slice_determinant"]:
        raise System10GravityScalarAWInvertibilityError("slice determinant changed")
    witness = sp.sqrt(3) / 3
    witness_matrix = slice_matrix.subs(v_10, witness)
    determinant_at_witness = sp.simplify(determinant.subs(v_10, witness))
    rank = witness_matrix.rank()
    right_null = witness_matrix.nullspace()
    left_null = witness_matrix.T.nullspace()
    expected_null = sp.Matrix([0] * 10 + [1])
    if (
        determinant_at_witness != 0
        or rank != 10
        or right_null != [expected_null]
        or left_null != [expected_null]
    ):
        raise System10GravityScalarAWInvertibilityError("singular witness replay failed")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "candidate": {
            "candidate_id": config["caps"]["candidate_id"],
            "scope": "representative_candidate_only",
        },
        "source_bindings": {
            "config_sha256": _canonical_sha(config),
            "aw_packet_config_sha256": config["aw_packet"]["config_content_sha256"],
            "ordered_aw_row_set_sha256": config["aw_packet"]["ordered_row_set_sha256"],
        },
        "specialized_slice": {
            "r": "1",
            "free_symbol": "v_10",
            "zeroed_symbol_count": len(zero_names),
            "zeroed_symbols": zero_names,
            "zeroed_symbol_set_sha256": _canonical_sha(zero_names),
            "matrix": [
                [sp.sstr(slice_matrix[row, column], order="lex") for column in range(11)]
                for row in range(11)
            ],
            "determinant": determinant_text,
        },
        "exact_singular_witness": {
            "r": "1",
            "v_10": "sqrt(3)/3",
            "determinant": "0",
            "rank": rank,
            "right_null_vector": ["0"] * 10 + ["1"],
            "left_null_vector": ["0"] * 10 + ["1"],
            "domain_certificate": "r=1>0",
        },
        "conclusion": {
            "global_invertibility_over_fixed_r_positive_state_domain": False,
            "unique_global_acceleration_solve": False,
            "all_11_accelerations_solved": False,
            "residual_replay_after_solve": False,
            "reason": (
                "one exact admissible state has rank(A)=10, so A is not invertible over the "
                "declared r>0 state domain"
            ),
        },
        "nonclaims": [
            "no assertion that A is singular away from the sealed witness",
            "no assertion about the other eleven candidates",
            "no full-RHS, propagation, or hyperbolicity claim",
        ],
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10GravityScalarAWInvertibilityError("receipt output cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise System10GravityScalarAWInvertibilityError("refusing to overwrite receipt")
    receipt = build_receipt(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal representative fixed-r A/W singularity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_receipt(args.config, args.output)


if __name__ == "__main__":
    main()
