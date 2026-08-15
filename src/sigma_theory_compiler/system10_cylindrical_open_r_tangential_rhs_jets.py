from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_open_r_representative_rhs_jets import _open_r_matrix
from .system10_cylindrical_open_r_twelve_candidate_rhs_jets import (
    _validate_config as _validate_radial_config,
)
from .system10_cylindrical_open_r_twelve_candidate_rhs_jets import _verify_candidate
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


class System10OpenRTangentialRHSJetsError(RuntimeError):
    """Raised when an atomic tangential RHS-jet packet fails."""


PACKET_SCHEMA = "invariant-system10-open-r-candidate-tangential-rhs-jets-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-open-r-tangential-rhs-jets-receipt-1.0"
DECISION = "BOUNDED_PASS_ALL_TWELVE_264_TANGENTIAL_RHS_JETS"
_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_Q = re.compile(r"q_(\d+)")
_V = re.compile(r"v_(\d+)")
_W = re.compile(r"w_([123])_(\d+)")
_PARTIAL_V = re.compile(r"partial_([123])_v_(\d+)")
_PARTIAL_W = re.compile(r"partial_([123])_w_([123])_(\d+)")


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10OpenRTangentialRHSJetsError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10OpenRTangentialRHSJetsError(f"bound content mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10OpenRTangentialRHSJetsError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "directions": [2, 3],
        "rhs_rows_per_direction_per_candidate": 11,
        "tangential_rhs_jets": 264,
        "radial_interval": ["1/2", "3/2"],
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_packet_bytes": 1048576,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10OpenRTangentialRHSJetsError("caps changed")

    radial = config["bindings"]["radial_jets"]
    radial_config_path = _resolve(root, radial["config_path"])
    if (
        _canonical_lf_sha(radial_config_path) != radial["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(radial_config_path)) != radial["config_content_sha256"]
    ):
        raise System10OpenRTangentialRHSJetsError("radial config mismatch")
    radial_config, _ = _validate_radial_config(radial_config_path, root)
    _, radial_receipt = _load_binding(root, radial["receipt"])
    if (
        radial_receipt.get("decision") != "BOUNDED_PASS_ALL_TWELVE_OPEN_R_RHS_ROWS_AND_RADIAL_JETS"
        or radial_receipt.get("counts", {}).get("open_r_radial_rhs_jets") != 132
    ):
        raise System10OpenRTangentialRHSJetsError("radial receipt mismatch")
    radial_dir = _resolve(root, radial["packet_dir"])
    for index in range(12):
        packet = _load_json(radial_dir / f"candidate-{index:02d}.json")
        _verify_candidate(packet, radial_config, index)
        if (
            packet["content_sha256"]
            != radial_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10OpenRTangentialRHSJetsError("radial packet mismatch")

    aw = config["bindings"]["aw_authority"]
    aw_config_path = _resolve(root, aw["config_path"])
    if (
        _canonical_lf_sha(aw_config_path) != aw["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(aw_config_path)) != aw["config_content_sha256"]
    ):
        raise System10OpenRTangentialRHSJetsError("A/W config mismatch")
    aw_config = _validate_aw_config(aw_config_path, root)
    _, aw_receipt = _load_binding(root, aw["receipt"])
    aw_dir = _resolve(root, aw["packet_dir"])
    if aw_receipt.get("counts", {}).get("candidate_packets") != 12:
        raise System10OpenRTangentialRHSJetsError("A/W receipt mismatch")
    for index in range(12):
        packet = _load_json(aw_dir / f"candidate-{index:02d}.json")
        _verify_aw_packet(packet, index, aw_config)
        if (
            packet["content_sha256"]
            != aw_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10OpenRTangentialRHSJetsError("A/W packet mismatch")

    _, predecessor = _load_binding(root, config["bindings"]["propagation_reaudit"])
    if (
        predecessor.get("decision")
        != "BLOCK_TANGENTIAL_RHS_JETS_UNREGISTERED_AFTER_RADIAL_JET_CLOSURE"
        or predecessor.get("counts", {}).get("tangential_rhs_jets_required") != 264
    ):
        raise System10OpenRTangentialRHSJetsError("tangential predecessor changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10OpenRTangentialRHSJetsError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != root / "tests/test_system10_cylindrical_open_r_tangential_rhs_jets.py"
    ):
        raise System10OpenRTangentialRHSJetsError("source evidence paths changed")
    return config, {
        "radial_config": radial_config,
        "radial_receipt": radial_receipt,
        "radial_dir": radial_dir,
        "aw_config": aw_config,
        "aw_receipt": aw_receipt,
        "aw_dir": aw_dir,
    }


def _directional_derivative_atom(token: str, direction: int) -> str | None:
    if match := _Q.fullmatch(token):
        return f"w_{direction}_{match.group(1)}"
    if match := _V.fullmatch(token):
        return f"partial_{direction}_v_{match.group(1)}"
    if match := _W.fullmatch(token):
        return f"partial_{direction}_w_{match.group(1)}_{match.group(2)}"
    if match := _PARTIAL_V.fullmatch(token):
        return f"partial_{direction}_partial_{match.group(1)}_v_{match.group(2)}"
    if match := _PARTIAL_W.fullmatch(token):
        return f"partial_{direction}_partial_{match.group(1)}_w_{match.group(2)}_{match.group(3)}"
    return None


def _w_directional_node(row: dict[str, Any], direction: int) -> dict[str, Any]:
    expression = row["W_entry"]["expression"]
    tokens = sorted(set(_TOKEN.findall(expression)))
    classified = []
    unknown = []
    for token in tokens:
        if token in {"sqrt", "r", "kappa"}:
            continue
        derivative = _directional_derivative_atom(token, direction)
        if derivative is None:
            unknown.append(token)
        else:
            classified.append({"source_atom": token, "directional_derivative_atom": derivative})
    if unknown:
        raise System10OpenRTangentialRHSJetsError(
            f"unclassified direction-{direction} W atoms: {unknown[:5]}"
        )
    operator = {
        "coordinate": direction,
        "explicit_coordinate_dependence": "none",
        "chain_rule": f"D_{direction} f=sum_x (partial f/partial x) D_{direction} x",
        "coordinate_partials_commute": True,
        "classified_source_atoms": classified,
    }
    body = {
        "row": row["row"],
        "direction": direction,
        "symbol": f"D{direction}_W_{row['row']}",
        "source_W_entry_sha256": row["W_entry"]["entry_sha256"],
        "source_W_expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "operator": operator,
        "operator_sha256": _canonical_sha(operator),
        "classified_atom_count": len(classified),
        "unclassified_atom_count": 0,
    }
    return {**body, "node_sha256": _canonical_sha(body)}


def _total_directional_derivative(expression: sp.Expr, direction: int) -> sp.Expr:
    v_10 = sp.Symbol("v_10")
    derivative = sp.diff(expression, v_10) * sp.Symbol(f"partial_{direction}_v_10")
    for row in range(11):
        derivative += sp.diff(expression, sp.Symbol(f"W_{row}")) * sp.Symbol(
            f"D{direction}_W_{row}"
        )
    return sp.factor(derivative)


def _build_candidate(
    config: dict[str, Any], bound: dict[str, Any], candidate_index: int
) -> dict[str, Any]:
    if candidate_index not in config["caps"]["candidate_indices"]:
        raise System10OpenRTangentialRHSJetsError("candidate outside frozen cap")
    radial_packet = _load_json(bound["radial_dir"] / f"candidate-{candidate_index:02d}.json")
    _verify_candidate(radial_packet, bound["radial_config"], candidate_index)
    aw_packet = _load_json(bound["aw_dir"] / f"candidate-{candidate_index:02d}.json")
    _verify_aw_packet(aw_packet, candidate_index, bound["aw_config"])
    matrix, zero_names = _open_r_matrix(aw_packet)
    rhs = [sp.sympify(row["rhs_expression"]) for row in radial_packet["rows"]]
    w_symbols = sp.Matrix(sp.symbols("W_0:11"))
    if [sp.factor(value) for value in matrix * sp.Matrix(rhs) + w_symbols] != [sp.Integer(0)] * 11:
        raise System10OpenRTangentialRHSJetsError("base RHS residual mismatch")

    nodes = [
        _w_directional_node(row, direction) for direction in (2, 3) for row in aw_packet["rows"]
    ]
    jets = []
    differentiated_residuals = 0
    for direction in (2, 3):
        directional_rhs = [_total_directional_derivative(value, direction) for value in rhs]
        directional_matrix = matrix.applyfunc(
            lambda value, selected=direction: _total_directional_derivative(value, selected)
        )
        residuals = [
            sp.factor(value)
            for value in directional_matrix * sp.Matrix(rhs)
            + matrix * sp.Matrix(directional_rhs)
            + sp.Matrix(sp.symbols(f"D{direction}_W_0:11"))
        ]
        if residuals != [sp.Integer(0)] * 11:
            raise System10OpenRTangentialRHSJetsError(
                f"direction-{direction} differentiated residual failed"
            )
        differentiated_residuals += len(residuals)
        for row, expression in enumerate(directional_rhs):
            body = {
                "row": row,
                "direction": direction,
                "jet_id": f"partial_{direction}_F_{row}",
                "expression": sp.sstr(expression, order="lex"),
                "denominator": sp.sstr(sp.factor(sp.denom(sp.together(expression))), order="lex"),
                "source_rhs_row_sha256": radial_packet["rows"][row]["row_sha256"],
                "source_W_directional_node_sha256s": [
                    node["node_sha256"]
                    for node in nodes
                    if node["direction"] == direction
                    and sp.Symbol(f"D{direction}_W_{node['row']}") in expression.free_symbols
                ],
                "differentiated_Euler_residual": "0",
            }
            jets.append({**body, "jet_sha256": _canonical_sha(body)})
    if len(jets) != 22 or differentiated_residuals != 22:
        raise System10OpenRTangentialRHSJetsError("tangential jet census failed")
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": aw_packet["candidate_id"],
        "coefficients": aw_packet["coefficients"],
        "source_bindings": {
            "packet_authority_sha256": _authority_sha(config),
            "source_radial_packet_content_sha256": radial_packet["content_sha256"],
            "source_A_W_packet_content_sha256": aw_packet["content_sha256"],
        },
        "open_r_neighborhood": radial_packet["open_r_neighborhood"],
        "zeroed_A_symbol_replay": zero_names,
        "W_directional_derivative_nodes": nodes,
        "tangential_rhs_jets": jets,
        "replay": {
            "base_rhs_rows_replayed": 11,
            "tangential_rhs_jets": 22,
            "differentiated_zero_residuals": 22,
            "directions": [2, 3],
        },
        "claims": {
            "candidate_all_22_tangential_rhs_jets_closed": True,
            "candidate_direction_2_residuals_replayed": True,
            "candidate_direction_3_residuals_replayed": True,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_packet_bytes"]:
        raise System10OpenRTangentialRHSJetsError("candidate packet cap exceeded")
    return packet


def build_candidate_packet(
    config_path: Path, candidate_index: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    return _build_candidate(config, bound, candidate_index)


def _verify_packet(packet: dict[str, Any], config: dict[str, Any], index: int) -> None:
    if (
        not _sealed(packet)
        or packet.get("candidate_index") != index
        or packet.get("source_bindings", {}).get("packet_authority_sha256")
        != _authority_sha(config)
        or len(packet.get("W_directional_derivative_nodes", [])) != 22
        or len(packet.get("tangential_rhs_jets", [])) != 22
        or packet.get("replay", {}).get("differentiated_zero_residuals") != 22
    ):
        raise System10OpenRTangentialRHSJetsError("candidate packet seal mismatch")


def run_candidates(
    config_path: Path, output_dir: Path, *, root: Path | None = None
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for index in range(12):
        packet = _build_candidate(config, bound, index)
        path = output_dir / f"candidate-{index:02d}.json"
        if path.exists():
            checked = _load_json(path)
            _verify_packet(checked, config, index)
            if checked != packet:
                raise System10OpenRTangentialRHSJetsError("existing packet replay mismatch")
        else:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        packets.append(packet)
    return packets


def build_receipt(
    config_path: Path, output_dir: Path, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    packets = []
    for index in range(12):
        packet = _load_json(output_dir / f"candidate-{index:02d}.json")
        _verify_packet(packet, config, index)
        if packet != _build_candidate(config, bound, index):
            raise System10OpenRTangentialRHSJetsError("candidate replay mismatch")
        packets.append(packet)
    ordered_set = hashlib.sha256(
        "".join(packet["content_sha256"] for packet in packets).encode("ascii")
    ).hexdigest()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact atomic linked-DAG total-directional derivatives partial_2 F and partial_3 F "
            "for all twelve candidates on the registered open-r tube. All 264 jets replay "
            "their differentiated Euler residuals. Together with the predecessor radial jets, "
            "the first spatial RHS jet is complete. Constraint propagation is not inferred."
        ),
        "source_bindings": {
            "receipt_authority_sha256": _authority_sha(config),
            "ordered_candidate_packet_set_sha256": ordered_set,
        },
        "counts": {
            "candidate_packets": 12,
            "candidate_passes": 12,
            "candidate_blocks": 0,
            "direction_2_rhs_jets": 132,
            "direction_3_rhs_jets": 132,
            "tangential_rhs_jets": 264,
            "differentiated_zero_residuals": 264,
            "base_rhs_row_replays": 132,
            "unclassified_W_atoms": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": packet["candidate_index"],
                "candidate_id": packet["candidate_id"],
                "outcome": "PASS_22_TANGENTIAL_RHS_JETS",
                "packet_content_sha256": packet["content_sha256"],
                "exact_absolute_determinant_lower_bound": packet["open_r_neighborhood"][
                    "exact_absolute_determinant_lower_bound"
                ],
            }
            for packet in packets
        ],
        "claims": {
            "all_twelve_tangential_rhs_jets_closed": True,
            "all_twelve_first_spatial_rhs_jets_closed_with_radial_predecessor": True,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10OpenRTangentialRHSJetsError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, output_dir, root=root)
    path = output_dir / "receipt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize all tangential RHS jets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.config.resolve().parents[1]
    run_candidates(args.config, args.output, root=root)
    write_receipt(args.config, args.output, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
