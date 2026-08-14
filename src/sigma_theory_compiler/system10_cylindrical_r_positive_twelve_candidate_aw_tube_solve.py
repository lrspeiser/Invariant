from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from .system10_cylindrical_r_positive_twelve_candidate_aw import (
    analyze_candidate_packet,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw import (
    build_census_receipt as build_aw_census_receipt,
)


class System10TwelveCandidateAWTubeSolveError(RuntimeError):
    """Raised when a common-tube acceleration solve cannot be sealed exactly."""


PACKET_SCHEMA = "invariant-system10-fixed-r-positive-candidate-aw-tube-solve-1.0"
RECEIPT_SCHEMA = "invariant-system10-fixed-r-positive-twelve-candidate-aw-tube-solve-1.0"
DECISION = "BOUNDED_PASS_ALL_TWELVE_COMMON_TUBE_ACCELERATION_SOLVES_AND_RESIDUALS"


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10TwelveCandidateAWTubeSolveError(f"bound source hash mismatch: {path}")
    return path


def _solution_authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10TwelveCandidateAWTubeSolveError("unsupported config schema")
    expected_caps = {
        "candidate_count": 12,
        "accelerations_per_candidate": 11,
        "residuals_per_candidate": 11,
        "r": "1",
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_solution_packet_bytes": 131072,
        "maximum_receipt_bytes": 131072,
    }
    if config.get("caps") != expected_caps:
        raise System10TwelveCandidateAWTubeSolveError("caps changed")
    predecessor = config.get("predecessor", {})
    prior_config_path = _resolve(root, str(predecessor.get("config_path", "")))
    if _canonical_lf_sha(prior_config_path) != predecessor.get(
        "config_canonical_lf_sha256"
    ) or _canonical_sha(_load_json(prior_config_path)) != predecessor.get("config_content_sha256"):
        raise System10TwelveCandidateAWTubeSolveError("predecessor config mismatch")
    prior_config = _validate_aw_config(prior_config_path, root)
    packet_dir = _resolve(root, str(predecessor.get("packet_dir", "")))
    prior_receipt_path = _resolve(root, str(predecessor.get("receipt_path", "")))
    prior_receipt = _load_json(prior_receipt_path)
    if (
        _canonical_lf_sha(prior_receipt_path) != predecessor.get("receipt_canonical_lf_sha256")
        or prior_receipt.get("content_sha256") != predecessor.get("receipt_content_sha256")
        or build_aw_census_receipt(prior_config_path, packet_dir, root=root) != prior_receipt
        or prior_receipt.get("decision")
        != "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
        or prior_receipt.get("counts", {}).get("tube_admitted_candidates") != 12
    ):
        raise System10TwelveCandidateAWTubeSolveError("predecessor receipt mismatch")
    packet_seals = []
    for index in range(12):
        packet = _load_json(packet_dir / f"candidate-{index:02d}.json")
        _verify_aw_packet(packet, index, prior_config)
        if (
            packet["content_sha256"]
            != prior_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10TwelveCandidateAWTubeSolveError("predecessor packet receipt mismatch")
        packet_seals.append(packet["content_sha256"])
    packet_set_sha = hashlib.sha256("".join(packet_seals).encode("ascii")).hexdigest()
    if packet_set_sha != predecessor.get("ordered_packet_set_sha256"):
        raise System10TwelveCandidateAWTubeSolveError("predecessor packet-set mismatch")
    sources = {
        name: _load_source(root, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    expected_test = root / (
        "tests/test_system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve.py"
    )
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
    ):
        raise System10TwelveCandidateAWTubeSolveError("source evidence changed")
    return config, prior_config


def _tube_matrix(packet: dict[str, Any]) -> tuple[sp.Matrix, list[str]]:
    matrix = sp.Matrix(
        [[sp.sympify(entry["expression"]) for entry in row["A_entries"]] for row in packet["rows"]]
    )
    r = sp.Symbol("r")
    v_10 = sp.Symbol("v_10")
    zero_symbols = sorted(matrix.free_symbols - {r, v_10}, key=str)
    return (
        matrix.xreplace({symbol: sp.Integer(0) for symbol in zero_symbols}).subs(r, 1),
        [str(symbol) for symbol in zero_symbols],
    )


def build_solution_packet(
    config_path: Path, candidate_index: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, prior_config = _validate_config(config_path.resolve(), repository)
    if not 0 <= candidate_index < 12:
        raise System10TwelveCandidateAWTubeSolveError("candidate index outside frozen cap")
    packet_dir = _resolve(repository, config["predecessor"]["packet_dir"])
    aw_packet = _load_json(packet_dir / f"candidate-{candidate_index:02d}.json")
    _verify_aw_packet(aw_packet, candidate_index, prior_config)
    analysis = analyze_candidate_packet(aw_packet)
    prior_receipt = _load_json(_resolve(repository, config["predecessor"]["receipt_path"]))
    admitted = prior_receipt["candidate_results"][candidate_index]
    if (
        not admitted["tube_admitted"]
        or admitted["slice_determinant"] != analysis["determinant_text"]
        or admitted["exact_absolute_determinant_lower_bound"] == "0"
    ):
        raise System10TwelveCandidateAWTubeSolveError("candidate tube is not admitted")
    matrix, zero_names = _tube_matrix(aw_packet)
    w_symbols = sp.Matrix(sp.symbols("W_0:11"))
    accelerations = [sp.factor(value) for value in matrix.inv() * (-w_symbols)]
    residuals = [sp.factor(value) for value in matrix * sp.Matrix(accelerations) + w_symbols]
    if residuals != [sp.Integer(0)] * 11:
        raise System10TwelveCandidateAWTubeSolveError("candidate residual replay failed")
    w_bindings = []
    for row, aw_row in enumerate(aw_packet["rows"]):
        w_bindings.append(
            {
                "symbol": f"W_{row}",
                "meaning": f"sealed candidate W[{row}] evaluated on the common tube slice",
                "source_row": row,
                "source_row_content_sha256": aw_row["row_content_sha256"],
                "source_W_entry_sha256": aw_row["W_entry"]["entry_sha256"],
            }
        )
    acceleration_entries = []
    for row, acceleration in enumerate(accelerations):
        expression = sp.sstr(acceleration, order="lex")
        denominator = sp.sstr(sp.factor(sp.denom(sp.together(acceleration))), order="lex")
        acceleration_entries.append(
            {
                "row": row,
                "label": f"partial_0_v_{row}",
                "expression": expression,
                "denominator": denominator,
                "entry_sha256": _canonical_sha(
                    {"row": row, "expression": expression, "denominator": denominator}
                ),
            }
        )
    residual_entries = [
        {
            "row": row,
            "expression": "0",
            "entry_sha256": _canonical_sha({"row": row, "expression": "0"}),
        }
        for row in range(11)
    ]
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": aw_packet["candidate_id"],
        "coefficients": aw_packet["coefficients"],
        "source_bindings": {
            "solution_authority_sha256": _solution_authority_sha(config),
            "source_aw_packet_content_sha256": aw_packet["content_sha256"],
            "source_aw_receipt_content_sha256": config["predecessor"]["receipt_content_sha256"],
        },
        "common_tube": {
            "r": "1",
            "real_v_10_interval": ["-1/4", "1/4"],
            "zeroed_A_symbols": zero_names,
            "zeroed_A_symbol_set_sha256": _canonical_sha(zero_names),
            "slice_determinant": analysis["determinant_text"],
            "exact_absolute_determinant_lower_bound": admitted[
                "exact_absolute_determinant_lower_bound"
            ],
        },
        "sealed_W_inputs": w_bindings,
        "accelerations": acceleration_entries,
        "residual_replay": {"entries": residual_entries, "all_zero": True, "count": 11},
        "claims": {
            "candidate_common_tube_all_11_accelerations_solved": True,
            "candidate_common_tube_all_11_residuals_replayed": True,
            "candidate_global_domain_solved": False,
            "full_rhs": False,
            "propagation": False,
            "hyperbolicity": False,
        },
    }
    solution_packet = {**body, "content_sha256": _canonical_sha(body)}
    if (
        len(json.dumps(solution_packet).encode("utf-8"))
        > config["caps"]["maximum_solution_packet_bytes"]
    ):
        raise System10TwelveCandidateAWTubeSolveError("solution packet output cap exceeded")
    return solution_packet


def _verify_solution_packet(packet: dict[str, Any], index: int, config: dict[str, Any]) -> None:
    body = {key: value for key, value in packet.items() if key != "content_sha256"}
    if (
        packet.get("content_sha256") != _canonical_sha(body)
        or packet.get("candidate_index") != index
        or packet.get("source_bindings", {}).get("solution_authority_sha256")
        != _solution_authority_sha(config)
        or len(packet.get("accelerations", [])) != 11
        or packet.get("residual_replay", {}).get("count") != 11
        or not packet.get("residual_replay", {}).get("all_zero")
    ):
        raise System10TwelveCandidateAWTubeSolveError("solution packet seal mismatch")


def run_solutions(
    config_path: Path,
    output_dir: Path,
    candidate_indices: list[int],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, _ = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in candidate_indices:
        if not 0 <= index < 12:
            raise System10TwelveCandidateAWTubeSolveError("candidate index outside frozen cap")
        final = output_dir / f"solution-{index:02d}.json"
        if final.exists():
            packet = _load_json(final)
            _verify_solution_packet(packet, index, config)
            results.append(packet)
            continue
        packet = build_solution_packet(config_path, index, root=repository)
        temporary = output_dir / f".solution-{index:02d}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, final)
        results.append(packet)
    return results


def build_census_receipt(
    config_path: Path, solutions_dir: Path, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, _ = _validate_config(config_path.resolve(), repository)
    results = []
    for index in range(12):
        packet = _load_json(solutions_dir / f"solution-{index:02d}.json")
        _verify_solution_packet(packet, index, config)
        results.append(
            {
                "candidate_index": index,
                "candidate_id": packet["candidate_id"],
                "solution_packet_content_sha256": packet["content_sha256"],
                "exact_absolute_determinant_lower_bound": packet["common_tube"][
                    "exact_absolute_determinant_lower_bound"
                ],
                "acceleration_count": 11,
                "zero_residual_count": 11,
                "outcome": "PASS_EXACT_COMMON_TUBE_SOLVE",
            }
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "source_bindings": {
            "config_sha256": _canonical_sha(config),
            "source_aw_receipt_content_sha256": config["predecessor"]["receipt_content_sha256"],
        },
        "common_tube": {"r": "1", "real_v_10_interval": ["-1/4", "1/4"]},
        "counts": {
            "candidate_solution_packets": 12,
            "accelerations_solved": 132,
            "zero_residuals_replayed": 132,
            "candidate_passes": 12,
            "candidate_blocks": 0,
        },
        "candidate_results": results,
        "claims": {
            "all_twelve_common_tube_accelerations_solved": True,
            "all_twelve_common_tube_residuals_replayed": True,
            "all_twelve_global_domains_solved": False,
            "full_rhs": False,
            "propagation": False,
            "hyperbolicity": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10TwelveCandidateAWTubeSolveError("census receipt output cap exceeded")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve twelve candidate A/W systems on common tube"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("--config", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    solve.add_argument("--candidate-indices", required=True)
    census = subparsers.add_parser("census")
    census.add_argument("--config", type=Path, required=True)
    census.add_argument("--solutions-dir", type=Path, required=True)
    census.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "solve":
        run_solutions(
            args.config,
            args.output_dir,
            [int(value) for value in args.candidate_indices.split(",")],
        )
        return
    if args.output.exists():
        raise System10TwelveCandidateAWTubeSolveError("refusing to overwrite census receipt")
    receipt = build_census_receipt(args.config, args.solutions_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
