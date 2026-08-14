"""Unify and replay the exact 31,680 reachable A/B/C leaf derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-reachable-leaf-derivative-completion-config-1.0"
RESULT_SCHEMA = "sigma-quartic-reachable-leaf-derivative-completion-gate-1.0"
CAMPAIGN_ID = "quartic-reachable-leaf-derivative-completion-001"
CONFIG_PATH = "configs/backgrounds/quartic_reachable_leaf_derivative_completion_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_reachable_leaf_derivative_completion_gate.py"
TEST_PATH = "tests/test_quartic_reachable_leaf_derivative_completion_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-reachable-leaf-derivative-completion-gate/campaign.json"
)
CONFIG_FILE_SHA256 = "6acfe7ab5506a6cd55f86107d816815ccc244fce4f54f722c0c0e9ac89b907da"
BUNDLE_ROLES = (
    "differentiability",
    "p10_leaf",
    "pother_leaf",
    "p10_replay",
    "pother_replay",
    "coordinate_projection",
)
CONTRACT = {
    "candidate_count": 12,
    "reachable_leaf_derivative_obligations": 31680,
    "p10_leaf_roots": 7920,
    "pother_leaf_roots": 23760,
    "unique_registered_direction_atoms": 20,
    "bounded_ordered_d2_roots": 264,
    "registered_d2_entries_per_candidate": 5324,
    "full_d2_entries_per_candidate": 257499,
}
POLICIES = {
    "zero_admission": "only_exact_zero_arithmetic_roots_from_bound_dense_manifests",
    "leaf_completion": "candidate_atom_and_132_typed_leaf_labels_must_align_exactly",
    "d2_admission": "ordered_record_id_and_merkle_replay_root_must_validate",
    "d2_count_advance": "forbidden_for_already_registered_bounded_roots",
    "complete_d2f": "fail_closed",
    "global_h7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}
FIRST_BLOCKER = (
    "register_candidate_bound_A_B_C_leaf_derivatives_for_the_131_remaining_"
    "coordinate_columns_before_extending_beyond_the_5324_registered_D2_entries_"
    "per_candidate"
)


class ReachableLeafCompletionError(ValueError):
    """A bound leaf derivative, dense zero, or replay root changed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_text_sha(path: Path) -> str:
    """Hash the five legacy authorities in their registered LF byte form."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _production_text_sha(path: Path) -> str:
    """Hash this gate's text as emitted by the production CRLF materializer."""
    lf_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(lf_bytes.replace(b"\n", b"\r\n")).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise ReachableLeafCompletionError("leaf completion path must be portable")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ReachableLeafCompletionError("leaf completion path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any], config_path: Path) -> None:
    if _production_text_sha(config_path) != CONFIG_FILE_SHA256:
        raise ReachableLeafCompletionError("leaf completion config bytes changed")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("output_path") != OUTPUT_PATH
        or tuple(value.get("source_bundles", {})) != BUNDLE_ROLES
        or value.get("completion_contract") != CONTRACT
        or value.get("policies") != POLICIES
        or value.get("seals") != SEALS
    ):
        raise ReachableLeafCompletionError("leaf completion config contract changed")
    for bundle in value["source_bundles"].values():
        if set(bundle) != {
            "stem",
            "slug",
            "source_sha256",
            "config_sha256",
            "test_sha256",
            "artifact_sha256",
            "content_sha256",
        } or any(
            not re.fullmatch(r"[0-9a-f]{64}", bundle[key])
            for key in bundle
            if key.endswith("sha256")
        ):
            raise ReachableLeafCompletionError("leaf completion source bundle changed")


def _load_bundle(root: Path, role: str, bundle: Mapping[str, Any]) -> dict[str, Any]:
    stem, slug = str(bundle["stem"]), str(bundle["slug"])
    paths = {
        "source_sha256": f"src/sigma_theory_compiler/{stem}.py",
        "config_sha256": f"configs/backgrounds/{stem}.json",
        "test_sha256": f"tests/test_{stem}.py",
        "artifact_sha256": f"runs/physics-language/{slug}/campaign.json",
    }
    for key, relative in paths.items():
        path = _inside(root, relative)
        if not path.is_file():
            raise ReachableLeafCompletionError("leaf completion source authority changed")
        actual = (
            _file_sha(path)
            if key == "artifact_sha256" or role == "coordinate_projection"
            else _legacy_text_sha(path)
        )
        if actual != bundle[key]:
            raise ReachableLeafCompletionError("leaf completion source authority changed")
    artifact_path = _inside(root, paths["artifact_sha256"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("content_sha256") != bundle["content_sha256"] or artifact.get(
        "content_sha256"
    ) != _content_sha(artifact):
        raise ReachableLeafCompletionError("leaf completion source receipt changed")
    return artifact


def _load_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {
        role: _load_bundle(root, role, config["source_bundles"][role]) for role in BUNDLE_ROLES
    }
    diff = values["differentiability"]
    p10 = values["p10_leaf"]
    pother = values["pother_leaf"]
    p10_replay = values["p10_replay"]
    pother_replay = values["pother_replay"]
    projection = values["coordinate_projection"]
    if (
        diff.get("decision")
        != "pass_exact_D1_DAG_differentiability_boundary_31680_leaf_jets_missing_D2_blocked"
        or diff.get("gate_counts", {}).get("deduplicated_leaf_derivative_obligations") != 31680
        or p10.get("decision")
        != "pass_7920_P10_arbitrary_background_leaf_derivative_roots_D2_propagation_blocked"
        or pother.get("decision") != "pass_23760_Pother_leaf_roots_all_180_D2_records_replay_ready"
        or p10_replay.get("decision")
        != "pass_all_84_P10_ordered_D2_roots_exactly_replayed_Pother_blocked"
        or pother_replay.get("decision")
        != "pass_all_180_Pother_roots_all_264_bounded_targets_sealed"
        or projection.get("decision")
        != "pass_all_54_lower_covariant_projections_D2_count_preserved"
        or projection.get("gate_counts", {}).get("D2_entries_registered_per_candidate_after")
        != 5324
    ):
        raise ReachableLeafCompletionError("leaf completion predecessor boundary changed")
    return values


def _dag_value(dag: Mapping[str, Any], root: int) -> str:
    nodes = dag.get("nodes")
    if (
        not isinstance(nodes, list)
        or dag.get("node_count") != len(nodes)
        or dag.get("content_sha256") != _content_sha(dag)
        or not isinstance(root, int)
        or root < 0
        or root >= len(nodes)
    ):
        raise ReachableLeafCompletionError("leaf arithmetic DAG changed")
    node = nodes[root]
    if node.get("op") == "exact_constant":
        return str(node.get("value"))
    if node.get("op") == "exact_sympy_rational_expression":
        return str(node.get("expression"))
    raise ReachableLeafCompletionError("leaf arithmetic root operation changed")


def _dense_roots(packet: Mapping[str, Any], dag: Mapping[str, Any]) -> list[int]:
    zero = packet.get("zero_default_arithmetic_root")
    if _dag_value(dag, zero) != "0":
        raise ReachableLeafCompletionError("leaf dense default is not exact zero")
    roots = [zero] * 132
    positions: set[int] = set()
    for entry in packet.get("A_derivative_sparse_entries", []):
        position = 11 * int(entry["row"]) + int(entry["column"])
        if position in positions or _dag_value(dag, entry["arithmetic_root"]) != entry["value"]:
            raise ReachableLeafCompletionError("A leaf sparse root changed")
        positions.add(position)
        roots[position] = int(entry["arithmetic_root"])
    for entry in packet.get("source_chunk_column_derivative_sparse_entries", []):
        position = 121 + int(entry["row"])
        if position in positions or _dag_value(dag, entry["arithmetic_root"]) != entry["value"]:
            raise ReachableLeafCompletionError("B/C leaf sparse root changed")
        positions.add(position)
        roots[position] = int(entry["arithmetic_root"])
    if (
        packet.get("total_leaf_derivative_roots") != 132
        or packet.get("nonzero_leaf_derivative_roots") != len(positions)
        or any(_dag_value(dag, roots[position]) == "0" for position in positions)
        or packet.get("dense_root_manifest_sha256") != _sha(roots)
    ):
        raise ReachableLeafCompletionError("leaf dense root manifest changed")
    return roots


def _expected_leaf_labels(packet: Mapping[str, Any]) -> set[str]:
    labels = {f"A[{row},{column}]" for row in range(11) for column in range(11)}
    family = str(packet["source_chunk_family"])
    column = int(packet["source_chunk_input_column"])
    chunk = f"B_{family[2]}" if family.startswith("s0") else f"C_{family[1:]}"
    labels.update(f"{chunk}[{row},{column}]" for row in range(11))
    return labels


def _leaf_completion(
    diff: Mapping[str, Any],
    p10: Mapping[str, Any],
    pother: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    candidate_ids = [str(row["candidate_id"]) for row in diff["candidate_manifests"]]
    if len(candidate_ids) != 12 or len(set(candidate_ids)) != 12:
        raise ReachableLeafCompletionError("leaf candidate inventory changed")
    p10_by_id = {str(row["candidate_id"]): row for row in p10["candidate_manifests"]}
    pother_by_id = {str(row["candidate_id"]): row for row in pother["candidate_manifests"]}
    if set(p10_by_id) != set(candidate_ids) or set(pother_by_id) != set(candidate_ids):
        raise ReachableLeafCompletionError("leaf candidate alignment changed")
    all_obligation_ids: set[str] = set()
    certificates = []
    for diff_candidate in diff["candidate_manifests"]:
        candidate_id = str(diff_candidate["candidate_id"])
        family_packets = []
        for family, candidate, artifact in (
            ("P10", p10_by_id[candidate_id], p10),
            ("Pother", pother_by_id[candidate_id], pother),
        ):
            packets = candidate.get("direction_packets")
            if not isinstance(packets, list) or candidate.get("manifest_sha256") != _sha(packets):
                raise ReachableLeafCompletionError("leaf candidate packet manifest changed")
            dag = artifact["leaf_derivative_arithmetic_DAG"]
            for packet in packets:
                family_packets.append(
                    (family, packet, _dense_roots(packet, dag), dag["content_sha256"])
                )
        by_atom = {
            str(packet["coordinate_atom"]): (family, packet, roots, dag_sha)
            for family, packet, roots, dag_sha in family_packets
        }
        diff_packets = diff_candidate.get("derivative_packets")
        if (
            len(family_packets) != 20
            or len(by_atom) != 20
            or not isinstance(diff_packets, list)
            or len(diff_packets) != 20
            or {str(row["target_coordinate_atom"]) for row in diff_packets} != set(by_atom)
        ):
            raise ReachableLeafCompletionError("leaf direction partition changed")
        candidate_obligations = []
        packet_roots = []
        for diff_packet in diff_packets:
            atom = str(diff_packet["target_coordinate_atom"])
            family, packet, roots, dag_sha = by_atom[atom]
            obligations = diff_packet.get("leaf_derivative_obligations")
            labels = {str(row["component_input_label"]) for row in obligations}
            if len(obligations) != 132 or labels != _expected_leaf_labels(packet):
                raise ReachableLeafCompletionError("typed leaf label alignment changed")
            for obligation in obligations:
                obligation_id = str(obligation["leaf_derivative_obligation_id"])
                if (
                    obligation_id in all_obligation_ids
                    or obligation.get("candidate_id") != candidate_id
                    or obligation.get("target_coordinate_atom") != atom
                    or obligation.get("obligation_status") != "required_unregistered"
                    or obligation.get("component_input_coordinate_derivative_root_registered")
                    is not False
                    or obligation.get("component_input_coordinate_derivative_dag_sha256_registered")
                    is not False
                ):
                    raise ReachableLeafCompletionError("leaf obligation identity changed")
                all_obligation_ids.add(obligation_id)
                candidate_obligations.append(obligation_id)
            packet_roots.append(
                {
                    "family": family,
                    "coordinate_atom": atom,
                    "dense_root_manifest_sha256": _sha(roots),
                    "arithmetic_DAG_sha256": dag_sha,
                    "nonzero_roots": packet["nonzero_leaf_derivative_roots"],
                    "zero_roots": 132 - packet["nonzero_leaf_derivative_roots"],
                }
            )
        packet_roots.sort(key=lambda row: (row["family"], row["coordinate_atom"]))
        if (
            len(candidate_obligations) != 2640
            or sum(row["nonzero_roots"] for row in packet_roots) != 33
            or sum(row["zero_roots"] for row in packet_roots) != 2607
        ):
            raise ReachableLeafCompletionError("leaf candidate completion count changed")
        body = {
            "candidate_id": candidate_id,
            "obligation_count": 2640,
            "obligation_id_manifest_sha256": _sha(sorted(candidate_obligations)),
            "direction_packet_count": 20,
            "leaf_root_count": 2640,
            "nonzero_leaf_roots": 33,
            "exact_zero_leaf_roots": 2607,
            "direction_root_manifests": packet_roots,
            "all_obligations_exactly_matched": True,
        }
        certificates.append({**body, "content_sha256": _sha(body)})
    if len(all_obligation_ids) != 31680:
        raise ReachableLeafCompletionError("global leaf obligation completion changed")
    return certificates, all_obligation_ids


def _record_identity(record: Mapping[str, Any], *, p10: bool) -> dict[str, Any]:
    keys = (
        (
            "candidate_id",
            "coordinate_ordinal",
            "coordinate_atom",
            "coordinate_column",
            "D1_arithmetic_root",
            "D2_merkle_replay_root_sha256",
        )
        if p10
        else (
            "candidate_id",
            "coordinate_atom",
            "coordinate_ordinal",
            "D2_merkle_replay_root_sha256",
        )
    )
    return {key: record[key] for key in keys}


def _d2_completion(
    p10: Mapping[str, Any], pother: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    candidate_rows: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    for family, artifact, records_key, expected_per_candidate in (
        ("P10", p10, "ordered_P10_D2_records", 7),
        ("Pother", pother, "ordered_Pother_D2_records", 15),
    ):
        manifest_shas = []
        for candidate in artifact["candidate_manifests"]:
            candidate_id = str(candidate["candidate_id"])
            records = candidate.get(records_key)
            if not isinstance(records, list) or len(records) != expected_per_candidate:
                raise ReachableLeafCompletionError("bounded D2 record census changed")
            record_ids = []
            for record in records:
                record_id = str(record["ordered_D2_record_id"])
                if (
                    record_id != _sha(_record_identity(record, p10=family == "P10"))
                    or record_id in all_ids
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(record["D2_merkle_replay_root_sha256"])
                    )
                ):
                    raise ReachableLeafCompletionError("bounded D2 replay identity changed")
                all_ids.add(record_id)
                record_ids.append(record_id)
            if candidate.get("candidate_manifest_sha256") != _sha(record_ids):
                raise ReachableLeafCompletionError("bounded D2 candidate manifest changed")
            manifest_shas.append(candidate["candidate_manifest_sha256"])
            row = candidate_rows.setdefault(candidate_id, {"P10": [], "Pother": []})
            row[family] = records
        if artifact.get("manifest_sha256") != _sha(manifest_shas):
            raise ReachableLeafCompletionError("bounded D2 global manifest changed")
    if len(candidate_rows) != 12 or len(all_ids) != 264:
        raise ReachableLeafCompletionError("bounded D2 completion changed")
    summary = {}
    for candidate_id, rows in candidate_rows.items():
        pother_zero = sum(record["D2_root_exactly_zero"] for record in rows["Pother"])
        if pother_zero != 2:
            raise ReachableLeafCompletionError("bounded D2 exact-zero census changed")
        summary[candidate_id] = {
            "bounded_ordered_D2_roots": 22,
            "P10_roots": 7,
            "Pother_roots": 15,
            "Pother_exact_zero_roots": 2,
            "record_id_manifest_sha256": _sha(
                sorted(
                    record["ordered_D2_record_id"]
                    for records in rows.values()
                    for record in records
                )
            ),
        }
    return summary, all_ids


def _expected_body(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    certificates, obligation_ids = _leaf_completion(
        values["differentiability"], values["p10_leaf"], values["pother_leaf"]
    )
    d2_by_candidate, d2_ids = _d2_completion(values["p10_replay"], values["pother_replay"])
    for certificate in certificates:
        certificate["bounded_D2_completion"] = d2_by_candidate[certificate["candidate_id"]]
        certificate["content_sha256"] = _sha(
            {key: item for key, item in certificate.items() if key != "content_sha256"}
        )
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_31680_reachable_leaf_roots_and_264_bounded_D2_roots_exact",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "completion_theorem": {
            "name": "complete_reachable_A_B_C_leaf_partition_and_bounded_D2_replay",
            "exact_result": (
                "The 31,680 predecessor obligations split without overlap into 7,920 P10 and "
                "23,760 Pother exact arithmetic roots. Candidate, atom, and all 132 typed A/B/C "
                "labels align. Exact dense manifests certify 396 nonzero and 31,284 zero roots. "
                "Closed inverse/product DAG replays certify all 264 bounded ordered-D2 records."
            ),
            "zero_boundary": (
                "A zero is admitted only where the bound exact differentiation source emitted "
                "the arithmetic DAG zero root and the recomputed 132-slot dense manifest agrees."
            ),
            "scope_boundary": (
                "This consolidates already registered 20-direction leaf authority and 22 bounded "
                "row-10 D2 records per candidate. It does not infer derivatives for the remaining "
                "131 coordinate columns or increase the existing 5,324-entry D2 count."
            ),
        },
        "candidate_certificates": certificates,
        "completion_manifest_sha256": _sha(
            [certificate["content_sha256"] for certificate in certificates]
        ),
        "obligation_id_manifest_sha256": _sha(sorted(obligation_ids)),
        "bounded_D2_record_id_manifest_sha256": _sha(sorted(d2_ids)),
        "gate_counts": {
            "selected_candidates": 12,
            "reachable_leaf_derivative_obligations": 31680,
            "registered_exact_leaf_derivative_roots": 31680,
            "P10_leaf_derivative_roots": 7920,
            "Pother_leaf_derivative_roots": 23760,
            "nonzero_leaf_derivative_roots": 396,
            "exact_zero_leaf_derivative_roots": 31284,
            "unique_registered_derivative_atoms": 20,
            "bounded_ordered_D2_roots_registered": 264,
            "bounded_ordered_D2_roots_blocked": 0,
            "registered_D2_entries_per_candidate_before": 5324,
            "new_D2_entries_per_candidate": 0,
            "registered_D2_entries_per_candidate_after": 5324,
            "full_D2_entries_per_candidate": 257499,
            "remaining_D2_entries_per_candidate": 252175,
            "remaining_coordinate_columns_without_A_B_C_leaf_authority": 131,
            "complete_D2F_tensors": 0,
            "global_H7_closures": 0,
        },
        "claim_seals": {
            "all_31680_reachable_leaf_derivative_roots_registered": True,
            "no_zero_leaf_derivative_inferred": True,
            "all_264_bounded_ordered_D2_roots_registered": True,
            "D2_entry_count_advanced": False,
            "remaining_131_coordinate_leaf_families_registered": False,
            "complete_D2F": False,
            "global_H7": False,
            "candidate_theory_rejected": False,
        },
        "data_seals": dict(SEALS),
        "source_bindings": {
            "source": {
                "path": SOURCE_PATH,
                "production_file_sha256": _production_text_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {
                "path": CONFIG_PATH,
                "production_file_sha256": _production_text_sha(config_path),
            },
            "test": {
                "path": TEST_PATH,
                "production_file_sha256": _production_text_sha(_inside(root, TEST_PATH)),
            },
            "evidence": _copy(config["source_bundles"]),
        },
        "scope": (
            "exact union and dense-root replay of the 31,680 reachable candidate-bound A/B/C "
            "leaf derivatives plus the 264 already bounded ordered-D2 records; no inferred zero, "
            "D2 count increment, complete tensor, H7, rejection, or observation"
        ),
    }


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config, path)
    values = _load_inputs(project_root, config)
    body = _expected_body(project_root, path, config, values)
    return {**body, "content_sha256": _sha(body)}


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    project_root = Path(root or Path.cwd()).resolve()
    expected = build_campaign(root=project_root)
    if value.get("content_sha256") != _content_sha(value) or value != expected:
        raise ReachableLeafCompletionError("reachable leaf completion result changed")


def write_campaign(
    output_path: Path | str = OUTPUT_PATH,
    config_path: Path | str = CONFIG_PATH,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    result = build_campaign(config_path, root=project_root)
    path = _inside(project_root, str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        validate_campaign(json.loads(Path(args.output).read_text(encoding="utf-8")))
    else:
        write_campaign(args.output, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
