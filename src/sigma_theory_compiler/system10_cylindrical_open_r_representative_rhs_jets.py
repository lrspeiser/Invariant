from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw import (
    _validate_config as _validate_aw_config,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw import (
    _verify_packet as _verify_aw_packet,
)


class System10OpenRRepresentativeRHSJetsError(RuntimeError):
    """Raised when the representative open-r RHS-jet packet fails closed."""


PACKET_SCHEMA = "invariant-system10-open-r-representative-rhs-jets-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-open-r-representative-rhs-jets-receipt-1.0"
DECISION = "BOUNDED_PASS_REPRESENTATIVE_11_OPEN_R_RHS_ROWS_AND_RADIAL_JETS"
_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_Q = re.compile(r"q_(\d+)")
_V = re.compile(r"v_(\d+)")
_W = re.compile(r"w_([123])_(\d+)")
_PARTIAL_V = re.compile(r"partial_([123])_v_(\d+)")
_PARTIAL_W = re.compile(r"partial_([123])_w_([123])_(\d+)")


def _body_sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_bound_json(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10OpenRRepresentativeRHSJetsError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _body_sealed(
        document
    ):
        raise System10OpenRRepresentativeRHSJetsError(f"bound content mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10OpenRRepresentativeRHSJetsError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": [0],
        "rhs_rows": 11,
        "radial_rhs_jets": 11,
        "radial_interval": ["1/2", "3/2"],
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_packet_bytes": 262144,
        "maximum_receipt_bytes": 131072,
    }:
        raise System10OpenRRepresentativeRHSJetsError("caps changed")
    aw = config["bindings"]["aw_authority"]
    aw_config_path = _resolve(root, aw["config_path"])
    if (
        _canonical_lf_sha(aw_config_path) != aw["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(aw_config_path)) != aw["config_content_sha256"]
    ):
        raise System10OpenRRepresentativeRHSJetsError("A/W config mismatch")
    aw_config = _validate_aw_config(aw_config_path, root)
    _, aw_receipt = _load_bound_json(root, aw["receipt"])
    if (
        aw_receipt.get("decision") != "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
        or aw_receipt.get("counts", {}).get("candidate_packets") != 12
    ):
        raise System10OpenRRepresentativeRHSJetsError("A/W receipt mismatch")
    aw_packet_path, aw_packet = _load_bound_json(root, aw["representative_packet"])
    _verify_aw_packet(aw_packet, 0, aw_config)
    if (
        aw_packet_path.name != "candidate-00.json"
        or aw_packet["content_sha256"]
        != aw_receipt["candidate_results"][0]["packet_content_sha256"]
    ):
        raise System10OpenRRepresentativeRHSJetsError("representative A/W packet mismatch")
    _, tube_solution = _load_bound_json(root, config["bindings"]["r1_solution"])
    if (
        tube_solution.get("candidate_index") != 0
        or len(tube_solution.get("accelerations", [])) != 11
        or not tube_solution.get("residual_replay", {}).get("all_zero")
    ):
        raise System10OpenRRepresentativeRHSJetsError("r=1 solution mismatch")
    _, blocker = _load_bound_json(root, config["bindings"]["radial_jet_blocker"])
    missing = blocker.get("materialization", {}).get("first_missing_primitive", {})
    if (
        blocker.get("decision")
        != "BLOCK_COMMON_TUBE_RHS_HAS_NO_RADIAL_JET_FOR_CONSTRAINT_PROPAGATION"
        or missing.get("status") != "BLOCK_RADIAL_RHS_JET_UNREGISTERED"
    ):
        raise System10OpenRRepresentativeRHSJetsError("radial-jet predecessor changed")
    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10OpenRRepresentativeRHSJetsError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"]
        != root / "tests/test_system10_cylindrical_open_r_representative_rhs_jets.py"
    ):
        raise System10OpenRRepresentativeRHSJetsError("source evidence paths changed")
    return config, {"aw_packet": aw_packet, "tube_solution": tube_solution}


def _radial_derivative_atom(token: str) -> str | None:
    if match := _Q.fullmatch(token):
        return f"w_1_{match.group(1)}"
    if match := _V.fullmatch(token):
        return f"partial_1_v_{match.group(1)}"
    if match := _W.fullmatch(token):
        return f"partial_1_w_{match.group(1)}_{match.group(2)}"
    if match := _PARTIAL_V.fullmatch(token):
        return f"partial_1_partial_{match.group(1)}_v_{match.group(2)}"
    if match := _PARTIAL_W.fullmatch(token):
        return f"partial_1_partial_{match.group(1)}_w_{match.group(2)}_{match.group(3)}"
    return None


def _w_derivative_node(row: dict[str, Any]) -> dict[str, Any]:
    expression = row["W_entry"]["expression"]
    tokens = sorted(set(_TOKEN.findall(expression)))
    ignored = {"sqrt", "r", "kappa"}
    classified = []
    unknown = []
    for token in tokens:
        if token in ignored:
            continue
        derivative = _radial_derivative_atom(token)
        if derivative is None:
            unknown.append(token)
        else:
            classified.append({"source_atom": token, "radial_derivative_atom": derivative})
    if unknown:
        raise System10OpenRRepresentativeRHSJetsError(
            f"unclassified W radial derivative atoms: {unknown[:5]}"
        )
    operator = {
        "coordinate": 1,
        "explicit_coordinate_symbol": "r",
        "chain_rule": "D_1 f=partial f/partial r+sum_x (partial f/partial x) D_1 x",
        "coordinate_partials_commute": True,
        "classified_source_atoms": classified,
    }
    body = {
        "row": row["row"],
        "symbol": f"D1_W_{row['row']}",
        "source_W_entry_sha256": row["W_entry"]["entry_sha256"],
        "source_W_expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "operator": operator,
        "operator_sha256": _canonical_sha(operator),
        "classified_atom_count": len(classified),
        "unclassified_atom_count": 0,
    }
    return {**body, "node_sha256": _canonical_sha(body)}


def _open_r_matrix(packet: dict[str, Any]) -> tuple[sp.Matrix, list[str]]:
    matrix = sp.Matrix(
        [[sp.sympify(entry["expression"]) for entry in row["A_entries"]] for row in packet["rows"]]
    )
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    zero_symbols = sorted(matrix.free_symbols - {r, v_10}, key=str)
    return matrix.xreplace({symbol: sp.Integer(0) for symbol in zero_symbols}), [
        str(symbol) for symbol in zero_symbols
    ]


def _total_radial_derivative(expression: sp.Expr) -> sp.Expr:
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    derivative = sp.diff(expression, r) + sp.diff(expression, v_10) * sp.Symbol("partial_1_v_10")
    for row in range(11):
        derivative += sp.diff(expression, sp.Symbol(f"W_{row}")) * sp.Symbol(f"D1_W_{row}")
    return sp.factor(derivative)


def build_packet(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    aw_packet = bound["aw_packet"]
    matrix, zero_names = _open_r_matrix(aw_packet)
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    determinant = sp.factor(matrix.det(method="domain-ge"))
    expected_determinant = (
        -sp.Rational(6561, 16384) * (v_10**2 + 2) ** 6 * (3 * v_10**2 - 1) / r**10
    )
    if sp.factor(determinant - expected_determinant) != 0:
        raise System10OpenRRepresentativeRHSJetsError("open-r determinant changed")
    w_symbols = sp.Matrix(sp.symbols("W_0:11"))
    rhs = [sp.factor(value) for value in matrix.inv() * (-w_symbols)]
    residual = [sp.factor(value) for value in matrix * sp.Matrix(rhs) + w_symbols]
    radial_rhs = [_total_radial_derivative(value) for value in rhs]
    radial_matrix = matrix.applyfunc(_total_radial_derivative)
    radial_residual = [
        sp.factor(value)
        for value in radial_matrix * sp.Matrix(rhs)
        + matrix * sp.Matrix(radial_rhs)
        + sp.Matrix(sp.symbols("D1_W_0:11"))
    ]
    if residual != [sp.Integer(0)] * 11 or radial_residual != [sp.Integer(0)] * 11:
        raise System10OpenRRepresentativeRHSJetsError("open-r residual replay failed")
    r1_expected = {
        entry["row"]: sp.sympify(entry["expression"])
        for entry in bound["tube_solution"]["accelerations"]
    }
    if any(sp.factor(rhs[row].subs(r, 1) - r1_expected[row]) != 0 for row in range(11)):
        raise System10OpenRRepresentativeRHSJetsError("r=1 solution replay failed")

    w_nodes = [_w_derivative_node(row) for row in aw_packet["rows"]]
    rows = []
    for row, (rhs_value, radial_value) in enumerate(zip(rhs, radial_rhs, strict=True)):
        row_body = {
            "row": row,
            "rhs_row_id": f"evolution_v[{row}]",
            "rhs_expression": sp.sstr(rhs_value, order="lex"),
            "rhs_denominator": sp.sstr(sp.factor(sp.denom(sp.together(rhs_value))), order="lex"),
            "radial_rhs_jet_id": f"partial_1_F_{row}",
            "radial_rhs_expression": sp.sstr(radial_value, order="lex"),
            "radial_rhs_denominator": sp.sstr(
                sp.factor(sp.denom(sp.together(radial_value))), order="lex"
            ),
            "source_A_W_row_content_sha256": aw_packet["rows"][row]["row_content_sha256"],
            "source_W_radial_derivative_node_sha256s": [
                node["node_sha256"]
                for node in w_nodes
                if sp.Symbol(f"D1_W_{node['row']}") in radial_value.free_symbols
            ],
            "open_r_residual": "0",
            "radial_residual": "0",
            "r1_replay": sp.sstr(rhs_value.subs(r, 1), order="lex"),
        }
        rows.append({**row_body, "row_sha256": _canonical_sha(row_body)})
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": 0,
        "candidate_id": aw_packet["candidate_id"],
        "coefficients": aw_packet["coefficients"],
        "source_bindings": {
            "packet_authority_sha256": _authority_sha(config),
            "source_A_W_packet_content_sha256": aw_packet["content_sha256"],
            "source_r1_solution_content_sha256": bound["tube_solution"]["content_sha256"],
        },
        "open_r_neighborhood": {
            "radial_interval": ["1/2", "3/2"],
            "real_v_10_interval": ["-1/4", "1/4"],
            "zeroed_A_symbols": zero_names,
            "determinant": sp.sstr(determinant, order="lex"),
            "coordinate_pole_set": ["r=0"],
            "exact_absolute_determinant_lower_bound": "13/36",
            "denominators_nonzero": True,
        },
        "W_radial_derivative_nodes": w_nodes,
        "rows": rows,
        "replay": {
            "open_r_zero_residuals": 11,
            "radially_differentiated_zero_residuals": 11,
            "exact_r1_rhs_replays": 11,
        },
        "claims": {
            "representative_open_r_all_11_rhs_rows_closed": True,
            "representative_open_r_all_11_radial_rhs_jets_closed": True,
            "representative_r1_solution_replayed": True,
            "all_twelve_candidates_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_packet_bytes"]:
        raise System10OpenRRepresentativeRHSJetsError("packet cap exceeded")
    return packet


def _verify_packet(packet: dict[str, Any], config: dict[str, Any]) -> None:
    if (
        not _body_sealed(packet)
        or packet.get("source_bindings", {}).get("packet_authority_sha256")
        != _authority_sha(config)
        or len(packet.get("rows", [])) != 11
        or len(packet.get("W_radial_derivative_nodes", [])) != 11
        or packet.get("replay", {}).get("radially_differentiated_zero_residuals") != 11
        or not packet.get("claims", {}).get("representative_open_r_all_11_radial_rhs_jets_closed")
    ):
        raise System10OpenRRepresentativeRHSJetsError("packet seal mismatch")


def build_receipt(
    config_path: Path, packet_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, _ = _validate_config(config_path.resolve(), repository)
    packet = _load_json(packet_path)
    _verify_packet(packet, config)
    if packet != build_packet(config_path, root=repository):
        raise System10OpenRRepresentativeRHSJetsError("packet replay mismatch")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact checkpointable linked-DAG open-r solve for representative candidate 0 on "
            "1/2<=r<=3/2 and real |v_10|<=1/4. Eleven RHS rows and their eleven total "
            "radial derivative rows replay the Euler residuals and the r=1 authority. This "
            "closes the prior radial-jet primitive for one candidate only; it does not prove "
            "all-twelve closure, constraint propagation, or hyperbolicity."
        ),
        "source_bindings": {
            "receipt_authority_sha256": _authority_sha(config),
            "representative_packet_content_sha256": packet["content_sha256"],
        },
        "counts": {
            "candidate_packets": 1,
            "candidate_passes": 1,
            "candidate_blocks": 0,
            "open_r_rhs_rows": 11,
            "open_r_radial_rhs_jets": 11,
            "open_r_zero_residuals": 11,
            "radially_differentiated_zero_residuals": 11,
            "r1_rhs_replays": 11,
            "remaining_candidate_packets": 11,
            "constraint_propagation_proofs": 0,
        },
        "scaling_plan": {
            "candidate_order": list(range(1, 12)),
            "atomic_outputs_per_candidate": {
                "A_entries": 121,
                "W_entries": 11,
                "rhs_rows": 11,
                "radial_rhs_jets": 11,
            },
            "checkpoint_boundaries": [
                "load_and_seal_candidate_A_W",
                "slice_and_factor_determinant",
                "solve_and_replay_11_rhs_rows",
                "classify_11_W_radial_derivative_DAGs",
                "differentiate_and_replay_11_radial_rhs_rows",
                "atomic_candidate_packet",
            ],
            "stop_conditions": [
                "determinant_zero_or_unbounded_below_on_registered_neighborhood",
                "unclassified_W_atom",
                "open_r_Euler_residual_nonzero",
                "radially_differentiated_Euler_residual_nonzero",
                "r1_replay_mismatch",
                "output_cap_exceeded",
            ],
            "all_twelve_completion_claim_deferred": True,
        },
        "claims": {
            "representative_open_r_radial_rhs_jet_primitive_closed": True,
            "all_twelve_open_r_radial_rhs_jet_primitive_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10OpenRRepresentativeRHSJetsError("receipt cap exceeded")
    return receipt


def write_outputs(
    config_path: Path, output_dir: Path, *, root: Path | None = None
) -> tuple[Path, Path]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, _ = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "candidate-00.json"
    packet = build_packet(config_path, root=repository)
    if packet_path.exists():
        checked = _load_json(packet_path)
        _verify_packet(checked, config)
        if checked != packet:
            raise System10OpenRRepresentativeRHSJetsError("existing packet replay mismatch")
    else:
        temporary = packet_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, packet_path)
    receipt = build_receipt(config_path, packet_path, root=repository)
    receipt_path = output_dir / "receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    return packet_path, receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize representative open-r RHS jets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_outputs(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
