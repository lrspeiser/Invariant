from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_open_r_representative_rhs_jets import (
    _open_r_matrix,
    _total_radial_derivative,
    _w_derivative_node,
)
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
from .system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    _validate_config as _validate_solve_config,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    _verify_solution_packet,
)


class System10OpenRTwelveCandidateRHSJetsError(RuntimeError):
    """Raised when an atomic all-candidate open-r jet unit fails."""


PACKET_SCHEMA = "invariant-system10-open-r-candidate-rhs-jets-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-open-r-twelve-candidate-rhs-jets-receipt-1.0"
DECISION = "BOUNDED_PASS_ALL_TWELVE_OPEN_R_RHS_ROWS_AND_RADIAL_JETS"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10OpenRTwelveCandidateRHSJetsError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10OpenRTwelveCandidateRHSJetsError(f"bound content mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10OpenRTwelveCandidateRHSJetsError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "rhs_rows_per_candidate": 11,
        "radial_rhs_jets_per_candidate": 11,
        "radial_interval": ["1/2", "3/2"],
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_packet_bytes": 524288,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10OpenRTwelveCandidateRHSJetsError("caps changed")

    aw = config["bindings"]["aw_authority"]
    aw_config_path = _resolve(root, aw["config_path"])
    if (
        _canonical_lf_sha(aw_config_path) != aw["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(aw_config_path)) != aw["config_content_sha256"]
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("A/W config mismatch")
    aw_config = _validate_aw_config(aw_config_path, root)
    _, aw_receipt = _load_binding(root, aw["receipt"])
    if (
        aw_receipt.get("decision") != "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
        or aw_receipt.get("counts", {}).get("candidate_packets") != 12
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("A/W receipt mismatch")
    aw_dir = _resolve(root, aw["packet_dir"])
    for index in range(12):
        packet = _load_json(aw_dir / f"candidate-{index:02d}.json")
        _verify_aw_packet(packet, index, aw_config)
        if (
            packet["content_sha256"]
            != aw_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10OpenRTwelveCandidateRHSJetsError("A/W packet receipt mismatch")

    solve = config["bindings"]["r1_solutions"]
    solve_config_path = _resolve(root, solve["config_path"])
    if (
        _canonical_lf_sha(solve_config_path) != solve["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(solve_config_path)) != solve["config_content_sha256"]
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("r=1 solve config mismatch")
    solve_config, _ = _validate_solve_config(solve_config_path, root)
    _, solve_receipt = _load_binding(root, solve["receipt"])
    solve_dir = _resolve(root, solve["solution_dir"])
    if (
        solve_receipt.get("decision")
        != "BOUNDED_PASS_ALL_TWELVE_COMMON_TUBE_ACCELERATION_SOLVES_AND_RESIDUALS"
        or solve_receipt.get("counts", {}).get("candidate_passes") != 12
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("r=1 solve receipt mismatch")
    for index in range(12):
        packet = _load_json(solve_dir / f"solution-{index:02d}.json")
        _verify_solution_packet(packet, index, solve_config)
        if (
            packet["content_sha256"]
            != solve_receipt["candidate_results"][index]["solution_packet_content_sha256"]
        ):
            raise System10OpenRTwelveCandidateRHSJetsError("r=1 solution receipt mismatch")

    _, representative = _load_binding(root, config["bindings"]["representative_receipt"])
    if (
        representative.get("decision")
        != "BOUNDED_PASS_REPRESENTATIVE_11_OPEN_R_RHS_ROWS_AND_RADIAL_JETS"
        or representative.get("counts", {}).get("remaining_candidate_packets") != 11
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("representative predecessor changed")
    _, blocker = _load_binding(root, config["bindings"]["propagation_blocker"])
    if (
        blocker.get("decision")
        != "BLOCK_COMMON_TUBE_RHS_HAS_NO_RADIAL_JET_FOR_CONSTRAINT_PROPAGATION"
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("propagation blocker changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10OpenRTwelveCandidateRHSJetsError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"]
        != root / "tests/test_system10_cylindrical_open_r_twelve_candidate_rhs_jets.py"
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("source evidence paths changed")
    return config, {
        "aw_config": aw_config,
        "aw_receipt": aw_receipt,
        "aw_dir": aw_dir,
        "solve_config": solve_config,
        "solve_receipt": solve_receipt,
        "solve_dir": solve_dir,
    }


def _build_candidate(
    config: dict[str, Any], bound: dict[str, Any], candidate_index: int
) -> dict[str, Any]:
    if candidate_index not in config["caps"]["candidate_indices"]:
        raise System10OpenRTwelveCandidateRHSJetsError("candidate outside frozen order")
    aw_packet = _load_json(bound["aw_dir"] / f"candidate-{candidate_index:02d}.json")
    _verify_aw_packet(aw_packet, candidate_index, bound["aw_config"])
    r1_packet = _load_json(bound["solve_dir"] / f"solution-{candidate_index:02d}.json")
    _verify_solution_packet(r1_packet, candidate_index, bound["solve_config"])
    admitted = bound["aw_receipt"]["candidate_results"][candidate_index]

    matrix, zero_names = _open_r_matrix(aw_packet)
    r = sp.Symbol("r")
    determinant = sp.factor(matrix.det(method="domain-ge"))
    r1_determinant = sp.factor(determinant.subs(r, 1))
    if (
        sp.factor(determinant / r1_determinant - r**-10) != 0
        or sp.factor(r1_determinant - sp.sympify(admitted["slice_determinant"])) != 0
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("open-r determinant replay failed")
    exact_lower_bound = sp.factor(
        sp.Rational(admitted["exact_absolute_determinant_lower_bound"]) / sp.Rational(3, 2) ** 10
    )
    if exact_lower_bound <= 0:
        raise System10OpenRTwelveCandidateRHSJetsError("open-r determinant margin failed")

    w_symbols = sp.Matrix(sp.symbols("W_0:11"))
    rhs = [sp.factor(value) for value in matrix.inv() * (-w_symbols)]
    radial_rhs = [_total_radial_derivative(value) for value in rhs]
    residual = [sp.factor(value) for value in matrix * sp.Matrix(rhs) + w_symbols]
    radial_matrix = matrix.applyfunc(_total_radial_derivative)
    radial_residual = [
        sp.factor(value)
        for value in radial_matrix * sp.Matrix(rhs)
        + matrix * sp.Matrix(radial_rhs)
        + sp.Matrix(sp.symbols("D1_W_0:11"))
    ]
    if residual != [sp.Integer(0)] * 11 or radial_residual != [sp.Integer(0)] * 11:
        raise System10OpenRTwelveCandidateRHSJetsError("Euler residual replay failed")
    r1_expected = {
        entry["row"]: sp.sympify(entry["expression"]) for entry in r1_packet["accelerations"]
    }
    if any(sp.factor(rhs[row].subs(r, 1) - r1_expected[row]) != 0 for row in range(11)):
        raise System10OpenRTwelveCandidateRHSJetsError("r=1 RHS replay failed")

    w_nodes = [_w_derivative_node(row) for row in aw_packet["rows"]]
    rows = []
    for row, (rhs_value, radial_value) in enumerate(zip(rhs, radial_rhs, strict=True)):
        body = {
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
        rows.append({**body, "row_sha256": _canonical_sha(body)})
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": aw_packet["candidate_id"],
        "coefficients": aw_packet["coefficients"],
        "source_bindings": {
            "packet_authority_sha256": _authority_sha(config),
            "source_A_W_packet_content_sha256": aw_packet["content_sha256"],
            "source_r1_solution_content_sha256": r1_packet["content_sha256"],
        },
        "open_r_neighborhood": {
            "radial_interval": ["1/2", "3/2"],
            "real_v_10_interval": ["-1/4", "1/4"],
            "zeroed_A_symbols": zero_names,
            "determinant": sp.sstr(determinant, order="lex"),
            "determinant_r_scaling": "r**(-10)",
            "coordinate_pole_set": ["r=0"],
            "exact_absolute_determinant_lower_bound": sp.sstr(exact_lower_bound),
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
            "candidate_open_r_all_11_rhs_rows_closed": True,
            "candidate_open_r_all_11_radial_rhs_jets_closed": True,
            "candidate_r1_solution_replayed": True,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_packet_bytes"]:
        raise System10OpenRTwelveCandidateRHSJetsError("candidate packet cap exceeded")
    return packet


def build_candidate_packet(
    config_path: Path, candidate_index: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    return _build_candidate(config, bound, candidate_index)


def _verify_candidate(packet: dict[str, Any], config: dict[str, Any], index: int) -> None:
    if (
        not _sealed(packet)
        or packet.get("candidate_index") != index
        or packet.get("source_bindings", {}).get("packet_authority_sha256")
        != _authority_sha(config)
        or len(packet.get("rows", [])) != 11
        or len(packet.get("W_radial_derivative_nodes", [])) != 11
        or packet.get("replay", {}).get("radially_differentiated_zero_residuals") != 11
    ):
        raise System10OpenRTwelveCandidateRHSJetsError("candidate packet seal mismatch")


def run_candidates(
    config_path: Path, output_dir: Path, *, root: Path | None = None
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for index in config["caps"]["candidate_indices"]:
        packet = _build_candidate(config, bound, index)
        path = output_dir / f"candidate-{index:02d}.json"
        if path.exists():
            checked = _load_json(path)
            _verify_candidate(checked, config, index)
            if checked != packet:
                raise System10OpenRTwelveCandidateRHSJetsError("existing packet replay mismatch")
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
        _verify_candidate(packet, config, index)
        if packet != _build_candidate(config, bound, index):
            raise System10OpenRTwelveCandidateRHSJetsError("candidate replay mismatch")
        packets.append(packet)
    ordered_set = hashlib.sha256(
        "".join(packet["content_sha256"] for packet in packets).encode("ascii")
    ).hexdigest()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact atomic linked-DAG open-r acceleration and total-radial-derivative packets "
            "for all twelve candidates on 1/2<=r<=3/2, real |v_10|<=1/4. Every packet "
            "replays eleven Euler residuals, eleven differentiated residuals, and eleven r=1 "
            "solutions. This closes the radial RHS-jet input only; constraint propagation and "
            "hyperbolicity remain separate questions."
        ),
        "source_bindings": {
            "receipt_authority_sha256": _authority_sha(config),
            "ordered_candidate_packet_set_sha256": ordered_set,
        },
        "counts": {
            "candidate_packets": 12,
            "candidate_passes": 12,
            "candidate_blocks": 0,
            "open_r_rhs_rows": 132,
            "open_r_radial_rhs_jets": 132,
            "open_r_zero_residuals": 132,
            "radially_differentiated_zero_residuals": 132,
            "r1_rhs_replays": 132,
            "remaining_candidate_packets": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": packet["candidate_index"],
                "candidate_id": packet["candidate_id"],
                "outcome": "PASS_11_OPEN_R_RHS_ROWS_AND_RADIAL_JETS",
                "exact_absolute_determinant_lower_bound": packet["open_r_neighborhood"][
                    "exact_absolute_determinant_lower_bound"
                ],
                "packet_content_sha256": packet["content_sha256"],
            }
            for packet in packets
        ],
        "claims": {
            "all_twelve_open_r_radial_rhs_jet_primitive_closed": True,
            "all_twelve_r1_solutions_replayed": True,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10OpenRTwelveCandidateRHSJetsError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, output_dir, root=root)
    path = output_dir / "receipt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize all twelve open-r RHS jet packets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.config.resolve().parents[1]
    run_candidates(args.config, args.output, root=root)
    write_receipt(args.config, args.output, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
