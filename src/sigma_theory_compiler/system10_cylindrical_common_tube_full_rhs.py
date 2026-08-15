from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    _validate_config as _validate_solve_config,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    _verify_solution_packet,
    build_solution_packet,
)
from .system10_cylindrical_r_positive_twelve_candidate_aw_tube_solve import (
    build_census_receipt as build_solve_census_receipt,
)


class System10CommonTubeFullRHSError(RuntimeError):
    """Raised when the linked exact 85-row RHS cannot be sealed."""


PACKET_SCHEMA = "invariant-system10-cylindrical-common-tube-full-rhs-packet-1.0"
RECEIPT_SCHEMA = "invariant-system10-cylindrical-common-tube-full-rhs-receipt-1.0"
DECISION = "BOUNDED_PASS_12_CANDIDATES_EXACT_85_OF_85_LINKED_RHS_ON_COMMON_TUBE"
_W_RE = re.compile(r"\bW_(\d+)\b")


def _body_seal(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_bound_json(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10CommonTubeFullRHSError(f"bound file hash mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _body_seal(document):
        raise System10CommonTubeFullRHSError(f"bound content seal mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_base_receipt(name: str, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "kinematic": (68, 68),
        "matter": (2, 2),
        "maxwell": (4, 4),
    }
    rows = receipt.get("materialization", {}).get("rows", [])
    if name not in expected or len(rows) != expected[name][0]:
        raise System10CommonTubeFullRHSError(f"{name} row count mismatch")
    if any(
        not row.get("row_sha256") or not row.get("equation_origin", {}).get("origin_sha256")
        for row in rows
    ):
        raise System10CommonTubeFullRHSError(f"{name} row/origin seal missing")
    if len({row["lhs_state_index"] for row in rows}) != expected[name][1]:
        raise System10CommonTubeFullRHSError(f"{name} state-row collision")
    return rows


def _validate_config(
    config_path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10CommonTubeFullRHSError("unsupported config schema")
    expected_caps = {
        "candidate_count": 12,
        "state_dimension": 85,
        "predecessor_rows_per_candidate": 74,
        "new_dynamic_rows_per_candidate": 11,
        "total_rows_per_candidate": 85,
        "r": "1",
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_candidate_packet_bytes": 524288,
        "maximum_receipt_bytes": 262144,
    }
    if config.get("caps") != expected_caps:
        raise System10CommonTubeFullRHSError("caps changed")

    base_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("kinematic", "matter", "maxwell"):
        _, receipt = _load_bound_json(root, config.get("base_authorities", {}).get(name, {}))
        base_rows[name] = _validate_base_receipt(name, receipt)
    if {row["lhs_state_index"] for rows in base_rows.values() for row in rows} != (
        set(range(17)) | set(range(28, 85))
    ):
        raise System10CommonTubeFullRHSError("base 74-row partition mismatch")

    solve = config.get("solve_authority", {})
    solve_config_path = _resolve(root, str(solve.get("config_path", "")))
    if _canonical_lf_sha(solve_config_path) != solve.get(
        "config_canonical_lf_sha256"
    ) or _canonical_sha(_load_json(solve_config_path)) != solve.get("config_content_sha256"):
        raise System10CommonTubeFullRHSError("solve config mismatch")
    solve_config, _ = _validate_solve_config(solve_config_path, root)
    solution_dir = _resolve(root, str(solve.get("solution_dir", "")))
    receipt_path, solve_receipt = _load_bound_json(root, solve.get("receipt", {}))
    rebuilt = build_solve_census_receipt(solve_config_path, solution_dir, root=root)
    if (
        rebuilt != solve_receipt
        or solve_receipt.get("decision")
        != "BOUNDED_PASS_ALL_TWELVE_COMMON_TUBE_ACCELERATION_SOLVES_AND_RESIDUALS"
        or solve_receipt.get("counts", {}).get("candidate_passes") != 12
        or receipt_path != _resolve(root, solve["receipt"]["path"])
    ):
        raise System10CommonTubeFullRHSError("solve receipt mismatch")
    seals = []
    for index in range(12):
        packet = _load_json(solution_dir / f"solution-{index:02d}.json")
        _verify_solution_packet(packet, index, solve_config)
        if packet != build_solution_packet(solve_config_path, index, root=root):
            raise System10CommonTubeFullRHSError("solution replay mismatch")
        seals.append(packet["content_sha256"])
    if hashlib.sha256("".join(seals).encode("ascii")).hexdigest() != solve.get(
        "ordered_solution_set_sha256"
    ):
        raise System10CommonTubeFullRHSError("solution set mismatch")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, str(binding.get("path", "")))
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10CommonTubeFullRHSError(f"source evidence mismatch: {name}")
        sources[name] = path
    expected_test = root / "tests/test_system10_cylindrical_common_tube_full_rhs.py"
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != expected_test
    ):
        raise System10CommonTubeFullRHSError("source evidence changed")
    return config, base_rows


def _base_row_references(base_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    references = []
    for authority in ("kinematic", "matter", "maxwell"):
        for row in base_rows[authority]:
            references.append(
                {
                    "authority": authority,
                    "row_id": row["row_id"],
                    "lhs_state_index": row["lhs_state_index"],
                    "row_sha256": row["row_sha256"],
                    "equation_origin_sha256": row["equation_origin"]["origin_sha256"],
                }
            )
    return sorted(references, key=lambda item: item["lhs_state_index"])


def build_candidate_packet(
    config_path: Path, candidate_index: int, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, base_rows = _validate_config(config_path.resolve(), repository)
    return _build_candidate_packet(config, base_rows, candidate_index, repository)


def _build_candidate_packet(
    config: dict[str, Any],
    base_rows: dict[str, list[dict[str, Any]]],
    candidate_index: int,
    repository: Path,
) -> dict[str, Any]:
    if not 0 <= candidate_index < 12:
        raise System10CommonTubeFullRHSError("candidate index outside frozen cap")
    solve = config["solve_authority"]
    solve_packet = _load_json(
        _resolve(repository, solve["solution_dir"]) / f"solution-{candidate_index:02d}.json"
    )
    acceleration_by_row = {entry["row"]: entry for entry in solve_packet["accelerations"]}
    residual_by_row = {entry["row"]: entry for entry in solve_packet["residual_replay"]["entries"]}
    w_by_row = {entry["source_row"]: entry for entry in solve_packet["sealed_W_inputs"]}
    dynamic_rows = []
    for row in range(11):
        acceleration = acceleration_by_row[row]
        residual = residual_by_row[row]
        used_w_rows = sorted({int(value) for value in _W_RE.findall(acceleration["expression"])})
        w_nodes = [w_by_row[index] for index in used_w_rows]
        origin = {
            "origin_type": "candidate_fixed_r_positive_unfactored_euler_component",
            "source_equation_row": row,
            "source_field_pair": "gravity_scalar"
            if row == 10
            else [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 2],
                [2, 3],
                [3, 3],
            ][row],
            "source_solution_packet_content_sha256": solve_packet["content_sha256"],
            "source_acceleration_entry_sha256": acceleration["entry_sha256"],
            "source_residual_entry_sha256": residual["entry_sha256"],
            "source_W_entry_sha256s": [node["source_W_entry_sha256"] for node in w_nodes],
            "source_W_row_content_sha256s": [node["source_row_content_sha256"] for node in w_nodes],
        }
        origin["origin_sha256"] = _canonical_sha(origin)
        row_body = {
            "row_id": f"evolution_v[{row}]",
            "field_index": row,
            "lhs": f"partial_0 state[{17 + row}]",
            "lhs_state_index": 17 + row,
            "domain": "r=1, real |v_10|<=1/4",
            "rhs_representation": "sealed_linked_exact_dag",
            "rhs_acceleration_expression": acceleration["expression"],
            "rhs_denominator": acceleration["denominator"],
            "rhs_W_nodes": w_nodes,
            "equation_origin": origin,
            "exact_residual_replay": {
                "expression": residual["expression"],
                "source_entry_sha256": residual["entry_sha256"],
                "zero": residual["expression"] == "0",
            },
        }
        dynamic_rows.append({**row_body, "row_sha256": _canonical_sha(row_body)})

    base_refs = _base_row_references(base_rows)
    state_indices = [row["lhs_state_index"] for row in base_refs + dynamic_rows]
    if len(state_indices) != 85 or sorted(state_indices) != list(range(85)):
        raise System10CommonTubeFullRHSError("combined 85-row partition mismatch")
    if any(not row["exact_residual_replay"]["zero"] for row in dynamic_rows):
        raise System10CommonTubeFullRHSError("dynamic residual replay failed")
    body = {
        "schema_version": PACKET_SCHEMA,
        "campaign_id": config["campaign_id"],
        "candidate_index": candidate_index,
        "candidate_id": solve_packet["candidate_id"],
        "coefficients": solve_packet["coefficients"],
        "source_bindings": {
            "full_rhs_authority_sha256": _authority_sha(config),
            "solution_packet_content_sha256": solve_packet["content_sha256"],
            "base_row_set_sha256s": {
                name: _canonical_sha(rows) for name, rows in base_rows.items()
            },
        },
        "common_tube": solve_packet["common_tube"],
        "base_74_row_references": base_refs,
        "dynamic_11_rows": dynamic_rows,
        "row_partition": {
            "base_rows": 74,
            "new_dynamic_rows": 11,
            "total_rows": 85,
            "lhs_state_indices": state_indices,
            "complete_0_through_84": True,
            "row_set_sha256": _canonical_sha(base_refs + dynamic_rows),
            "equation_origin_set_sha256": _canonical_sha(
                [row["equation_origin_sha256"] for row in base_refs]
                + [row["equation_origin"]["origin_sha256"] for row in dynamic_rows]
            ),
        },
        "claims": {
            "candidate_common_tube_exact_85_state_rhs_closed": True,
            "all_rows_have_equation_origin_seals": True,
            "all_new_rows_have_exact_zero_residual_replay": True,
            "linked_W_nodes_are_bound_to_explicit_predecessor_expressions": True,
            "fixed_r_positive_domain_full_rhs_closed": False,
            "global_domain_full_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
        },
    }
    packet = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(packet).encode("utf-8")) > config["caps"]["maximum_candidate_packet_bytes"]:
        raise System10CommonTubeFullRHSError("candidate packet cap exceeded")
    return packet


def _verify_candidate_packet(packet: dict[str, Any], index: int, config: dict[str, Any]) -> None:
    rows = packet.get("dynamic_11_rows", [])
    if (
        not _body_seal(packet)
        or packet.get("candidate_index") != index
        or packet.get("source_bindings", {}).get("full_rhs_authority_sha256")
        != _authority_sha(config)
        or len(packet.get("base_74_row_references", [])) != 74
        or len(rows) != 11
        or packet.get("row_partition", {}).get("total_rows") != 85
        or not packet.get("row_partition", {}).get("complete_0_through_84")
        or any(not row.get("exact_residual_replay", {}).get("zero") for row in rows)
    ):
        raise System10CommonTubeFullRHSError("candidate packet seal/replay mismatch")


def run_candidates(
    config_path: Path,
    output_dir: Path,
    candidate_indices: list[int],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, base_rows = _validate_config(config_path.resolve(), repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in candidate_indices:
        final = output_dir / f"candidate-{index:02d}.json"
        if final.exists():
            packet = _load_json(final)
            _verify_candidate_packet(packet, index, config)
            if packet != _build_candidate_packet(config, base_rows, index, repository):
                raise System10CommonTubeFullRHSError("existing candidate replay mismatch")
        else:
            packet = _build_candidate_packet(config, base_rows, index, repository)
            temporary = final.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, final)
        results.append(packet)
    return results


def build_receipt(
    config_path: Path, output_dir: Path, *, root: Path | None = None
) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, base_rows = _validate_config(config_path.resolve(), repository)
    packets = []
    for index in range(12):
        packet = _load_json(output_dir / f"candidate-{index:02d}.json")
        _verify_candidate_packet(packet, index, config)
        if packet != _build_candidate_packet(config, base_rows, index, repository):
            raise System10CommonTubeFullRHSError("candidate replay mismatch")
        packets.append(packet)
    packet_set_sha = hashlib.sha256(
        "".join(packet["content_sha256"] for packet in packets).encode("ascii")
    ).hexdigest()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact linked-DAG 85-state RHS for each of twelve candidates only on the "
            "preregistered common tube r=1, real |v_10|<=1/4. The 74 predecessor rows "
            "and eleven candidate A/W-solved rows retain individual equation-origin seals. "
            "No fixed-r>0 global RHS, propagation, hyperbolicity, or global theorem is claimed."
        ),
        "source_bindings": {
            "full_rhs_authority_sha256": _authority_sha(config),
            "ordered_candidate_packet_set_sha256": packet_set_sha,
        },
        "common_tube": {"r": "1", "real_v_10_interval": ["-1/4", "1/4"]},
        "counts": {
            "candidates": 12,
            "candidate_passes": 12,
            "candidate_blocks": 0,
            "predecessor_rows_per_candidate": 74,
            "new_dynamic_rows_per_candidate": 11,
            "new_dynamic_row_instances": 132,
            "new_exact_zero_residual_replays": 132,
            "equation_origin_seals": 1020,
            "total_rhs_rows_per_candidate": 85,
            "total_rhs_row_instances": 1020,
            "full_rhs_candidates_closed_on_common_tube": 12,
        },
        "candidate_results": [
            {
                "candidate_index": packet["candidate_index"],
                "candidate_id": packet["candidate_id"],
                "outcome": "PASS_EXACT_85_OF_85_LINKED_RHS_ON_COMMON_TUBE",
                "packet_content_sha256": packet["content_sha256"],
                "row_set_sha256": packet["row_partition"]["row_set_sha256"],
                "equation_origin_set_sha256": packet["row_partition"]["equation_origin_set_sha256"],
            }
            for packet in packets
        ],
        "claims": {
            "all_twelve_common_tube_exact_85_state_rhs_closed": True,
            "all_1020_rows_have_equation_origin_seals": True,
            "all_132_new_rows_have_exact_zero_residual_replay": True,
            "fixed_r_positive_domain_full_rhs_closed": False,
            "global_domain_full_rhs_closed": False,
            "constraint_propagation_closed": False,
            "hyperbolicity_closed": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10CommonTubeFullRHSError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, output_dir, root=root)
    final = output_dir / "receipt.json"
    temporary = final.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal twelve exact common-tube 85-row RHS packets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.config.resolve().parents[1]
    run_candidates(args.config, args.output, list(range(12)), root=root)
    write_receipt(args.config, args.output, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
