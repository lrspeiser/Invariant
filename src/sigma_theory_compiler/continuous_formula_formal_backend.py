from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from .action_health import analyze_action_health
from .formal_backend import load_field_contract
from .gates import algebraic_gates, sampled_static_convexity
from .grammar import Q, X, Z
from .high_throughput import build_basis, candidate_id, correction_expression, decode_ordinal
from .production_covariant_provenance import map_candidate_to_covariant_action
from .promotion_orchestrator import ELIGIBILITY
from .real_formula_execution import _assets, _batch_arrays

CONFIG_SCHEMA = "sigma-continuous-formula-formal-backend-config-1.0"
MANIFEST_SCHEMA = "sigma-continuous-formula-candidate-manifest-1.0"
EVIDENCE_SCHEMA = "sigma-continuous-formula-formal-evidence-1.0"
BACKEND_ID = "candidate_bound_covariant_action_health_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _validate_sealed(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != _sha(body):
        raise ValueError("formal backend content hash mismatch")


def load_backend_config(root: Path, path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "backend_id",
        "generator_config_path",
        "generator_config_file_sha256",
        "grammar_path",
        "grammar_file_sha256",
        "field_contract_path",
        "field_contract_file_sha256",
        "formal_controls_path",
        "formal_controls_file_sha256",
        "candidate_mapper_source_path",
        "candidate_mapper_source_file_sha256",
        "action_health_source_path",
        "action_health_source_file_sha256",
        "maximum_candidate_records",
        "coupling_magnitude",
        "convexity_tolerance",
        "maximum_universal_constants",
        "observational_data_opened",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
        "paid_llm_calls",
    }
    if set(config) != expected or config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("formal backend config contract mismatch")
    if config["backend_id"] != BACKEND_ID or config["maximum_candidate_records"] != 32:
        raise ValueError("formal backend identity or bound mismatch")
    bindings = (
        ("generator_config_path", "generator_config_file_sha256"),
        ("grammar_path", "grammar_file_sha256"),
        ("field_contract_path", "field_contract_file_sha256"),
        ("formal_controls_path", "formal_controls_file_sha256"),
        ("candidate_mapper_source_path", "candidate_mapper_source_file_sha256"),
        ("action_health_source_path", "action_health_source_file_sha256"),
    )
    for path_key, hash_key in bindings:
        bound = (root / config[path_key]).resolve()
        bound.relative_to(root.resolve())
        if hashlib.sha256(bound.read_bytes()).hexdigest() != config[hash_key]:
            raise ValueError(f"formal backend binding mismatch: {path_key}")
    if any(
        config[key] is not False
        for key in (
            "observational_data_opened",
            "dark_matter_or_halo_inputs",
            "redshift_distance_inputs",
            "paid_llm_calls",
        )
    ):
        raise ValueError("formal backend opened forbidden inputs")
    return config


def extract_candidate_manifest(
    payload: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay one CPU shard and emit a bounded, exact sampled-static survivor manifest."""
    generator, basis, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    ordinals, term_ids, term_counts, sign_masks = _batch_arrays(dict(payload))
    tolerance = float(generator["convexity_tolerance"])
    guard = float(payload["ambiguity_guard"])
    low, high = tolerance - guard, tolerance + guard
    statuses: list[int] = []
    survivors: list[dict[str, Any]] = []
    survivor_ordinals: list[int] = []
    for row, ordinal in enumerate(map(int, ordinals)):
        count = int(term_counts[row])
        signs = np.array(
            [1.0 if int(sign_masks[row]) & (1 << position) else -1.0 for position in range(count)]
        )
        candidate = np.array([1.0, 0.0, 0.0]) + float(generator["coupling_magnitude"]) * np.sum(
            hessians[term_ids[row, :count]] * signs[:, None, None], axis=0
        )
        hdd, hdp, hpp = candidate.T
        minimum = float(
            np.min(0.5 * (hdd + hpp - np.sqrt(np.maximum((hdd - hpp) ** 2 + 4 * hdp**2, 0.0))))
        )
        status = 0 if not np.isfinite(minimum) or minimum <= low else (2 if minimum <= high else 1)
        statuses.append(status)
        if status == 1:
            survivor_ordinals.append(ordinal)
            decoded = decode_ordinal(
                int(payload["basis_count"]), int(payload["max_action_terms"]), ordinal
            )
            if len(survivors) < 32:
                survivors.append(
                    {
                        "candidate_id": candidate_id(str(payload["protocol_version"]), decoded),
                        "ordinal": ordinal,
                        "term_ids": decoded["term_ids"],
                        "signs": decoded["signs"],
                        "correction_expression": correction_expression(decoded, basis),
                        "sampled_static_margin": minimum,
                    }
                )
    counts = {
        "reject": statuses.count(0),
        "pass": statuses.count(1),
        "ambiguous": statuses.count(2),
    }
    if counts != result["counts"]:
        raise ValueError("candidate extraction replay differs from evaluator counts")
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "batch": dict(result["batch"]),
        "candidate_root_sha256": result["status_root_sha256"],
        "screen_counts": counts,
        "all_survivor_ordinals_root_sha256": _sha(survivor_ordinals),
        "survivor_records": survivors,
        "survivor_record_count": len(survivor_ordinals),
        "sample_complete": len(survivor_ordinals) <= 32,
        "observations_opened": False,
        "forbidden_target_inputs_opened": False,
    }
    return _sealed(body)


def combine_candidate_manifests(manifests: list[Mapping[str, Any]], maximum: int) -> dict[str, Any]:
    for manifest in manifests:
        validate_candidate_manifest(manifest)
    ordered = sorted(manifests, key=lambda value: int(value["batch"]["start_ordinal"]))
    if any(
        left["batch"]["end_ordinal_exclusive"] != right["batch"]["start_ordinal"]
        for left, right in itertools.pairwise(ordered)
    ):
        raise ValueError("candidate manifest intervals are not contiguous and disjoint")
    survivors = [row for manifest in ordered for row in manifest["survivor_records"]]
    survivor_count = sum(int(manifest["survivor_record_count"]) for manifest in ordered)
    selected = survivors[:maximum]
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "batch": {
            "start_ordinal": ordered[0]["batch"]["start_ordinal"],
            "end_ordinal_exclusive": ordered[-1]["batch"]["end_ordinal_exclusive"],
            "candidate_count": sum(int(value["batch"]["candidate_count"]) for value in ordered),
        },
        "candidate_root_sha256": hashlib.sha256(
            "".join(value["candidate_root_sha256"] for value in ordered).encode()
        ).hexdigest(),
        "screen_counts": {
            key: sum(int(value["screen_counts"][key]) for value in ordered)
            for key in ("reject", "pass", "ambiguous")
        },
        "all_survivor_ordinals_root_sha256": _sha(
            [value["all_survivor_ordinals_root_sha256"] for value in ordered]
        ),
        "survivor_records": selected,
        "survivor_record_count": survivor_count,
        "sample_complete": all(value["sample_complete"] for value in ordered)
        and survivor_count <= maximum,
        "observations_opened": False,
        "forbidden_target_inputs_opened": False,
    }
    return _sealed(body)


def validate_candidate_manifest(value: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    expected = {
        "schema_version",
        "batch",
        "candidate_root_sha256",
        "screen_counts",
        "all_survivor_ordinals_root_sha256",
        "survivor_records",
        "survivor_record_count",
        "sample_complete",
        "observations_opened",
        "forbidden_target_inputs_opened",
        "content_sha256",
    }
    if set(value) != expected or value["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("candidate manifest contract mismatch")
    if (
        value["observations_opened"] is not False
        or value["forbidden_target_inputs_opened"] is not False
    ):
        raise ValueError("candidate manifest opened forbidden inputs")
    if not _SHA256.fullmatch(str(value["candidate_root_sha256"])) or not _SHA256.fullmatch(
        str(value["all_survivor_ordinals_root_sha256"])
    ):
        raise ValueError("candidate manifest root mismatch")
    batch = value["batch"]
    counts = value["screen_counts"]
    if (
        set(batch) != {"start_ordinal", "end_ordinal_exclusive", "candidate_count"}
        or set(counts) != {"reject", "pass", "ambiguous"}
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in counts.values()
        )
        or batch["end_ordinal_exclusive"] - batch["start_ordinal"] != batch["candidate_count"]
        or sum(counts.values()) != batch["candidate_count"]
        or value["survivor_record_count"] != counts["pass"]
    ):
        raise ValueError("candidate manifest count contract mismatch")
    records = value["survivor_records"]
    if len(records) > 32 or len({row["candidate_id"] for row in records}) != len(records):
        raise ValueError("candidate manifest record bound or identity mismatch")
    record_keys = {
        "candidate_id",
        "ordinal",
        "term_ids",
        "signs",
        "correction_expression",
        "sampled_static_margin",
    }
    if any(set(row) != record_keys for row in records):
        raise ValueError("candidate manifest record contract mismatch")
    if value["sample_complete"] is not (value["survivor_record_count"] == len(records)):
        raise ValueError("candidate manifest completeness mismatch")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"immutable candidate artifact differs: {path.name}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _semantic_health(report: Mapping[str, Any], candidate_dir: Path) -> dict[str, Any]:
    generated_names = (
        "generated_static_dictionary_ir",
        "generated_q_operator_ir",
        "generated_q_variation_ir",
        "generated_higher_jet_auxiliary_ir",
        "generated_x_operator_ir",
        "generated_adm_ir",
        "generated_legendre_ir",
        "generated_dirac_ir",
        "generated_stability_ir",
        "generated_principal_ir",
        "generated_hamiltonian_ir",
    )
    artifact_bindings = []
    for path in sorted(candidate_dir.rglob("*.json")):
        if path.name in {"action-health.json", "semantic-action-health.json"}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        artifact_bindings.append(
            {
                "path": path.relative_to(candidate_dir).as_posix(),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "declared_content_sha256": value.get("content_sha256"),
                "schema_version": value.get("schema_version"),
            }
        )
    return {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "promotion_allowed": report["promotion_allowed"],
        "family": report.get("family"),
        "input_action_sha256": report.get("input_action_sha256"),
        "physical_dof": report.get("physical_dof"),
        "gate_statuses": {
            name: gate.get("status") for name, gate in sorted(report.get("gates", {}).items())
        },
        "gate_scopes": {
            name: gate.get("scope") for name, gate in sorted(report.get("gates", {}).items())
        },
        "discovery_blockers": list(report.get("discovery_blockers", [])),
        "generated_artifacts": {
            name: {
                "status": report.get(name, {}).get("status"),
                "content_sha256": report.get(name, {}).get("content_sha256"),
            }
            for name in generated_names
        },
        "artifact_bindings": artifact_bindings,
        "observational_gates_unsealed": report.get("observational_gates_unsealed"),
        "interpretation": report.get("interpretation"),
    }


def _candidate_decision(
    mapping: Mapping[str, Any], health: Mapping[str, Any] | None
) -> tuple[str, str]:
    if mapping["decision"] == "reject":
        return "reject", str(mapping["reason"])
    if mapping["decision"] == "blocked":
        return "block", str(mapping["blockers"][0])
    if health is None:
        raise ValueError("mapped candidate missing action-health evidence")
    if health["status"] == "reject":
        return "reject", "candidate_action_health_hard_rejection"
    if health["promotion_allowed"] is True:
        return "block", "complete_comparable_candidate_evidence_not_registered"
    blockers = health.get("discovery_blockers", [])
    return "block", str(blockers[0] if blockers else "candidate_action_health_incomplete")


def build_formal_evidence(
    generated_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_candidate_manifest(manifest)
    if manifest["candidate_root_sha256"] != generated_receipt["candidate_root_sha256"]:
        raise ValueError("candidate manifest/generated receipt root mismatch")
    generator = json.loads((root / config["generator_config_path"]).read_text(encoding="utf-8"))
    grammar = json.loads((root / config["grammar_path"]).read_text(encoding="utf-8"))
    field_contract = load_field_contract(root / config["field_contract_path"])
    formal_report = json.loads((root / config["formal_controls_path"]).read_text(encoding="utf-8"))
    if formal_report.get("counts") != {"failed": 0, "passed": 118, "total": 118}:
        raise ValueError("formal backend requires the registered 118-of-118 control report")
    basis = build_basis(int(generator["basis_count"]))
    records = []
    symbolic_passes = 0
    mapped_count = 0
    health_count = 0
    hard_reject_count = 0
    for row in manifest["survivor_records"]:
        decoded = decode_ordinal(
            int(generator["basis_count"]), int(generator["max_action_terms"]), int(row["ordinal"])
        )
        expression_text = correction_expression(decoded, basis)
        if (
            candidate_id(str(generator["protocol_version"]), decoded) != row["candidate_id"]
            or decoded["term_ids"] != row["term_ids"]
            or decoded["signs"] != row["signs"]
            or expression_text != row["correction_expression"]
        ):
            raise ValueError("formal candidate lineage mismatch")
        expression = sp.sympify(expression_text, locals={"x": X, "q": Q, "z": Z, "sqrt": sp.sqrt})
        gate_rows = [
            gate.as_dict()
            for gate in algebraic_gates(
                expression,
                constants_count=4,
                maximum_constants=int(config["maximum_universal_constants"]),
            )
        ]
        convexity = sampled_static_convexity(
            expression,
            float(config["coupling_magnitude"]),
            samples={"d": [0.1, 1.0, 10.0], "p": [0.0, 0.5, 1.0], "state": [0.0, 0.5, 1.0]},
            tolerance=float(config["convexity_tolerance"]),
        ).as_dict()
        formal_local_pass = (
            all(gate["status"] == "pass" for gate in gate_rows) and convexity["status"] == "pass"
        )
        symbolic_passes += int(formal_local_pass)
        hessian = sp.hessian(expression, (X, Q, Z))
        candidate = {
            "candidate_id": row["candidate_id"],
            "ordinal": row["ordinal"],
            "term_ids": list(row["term_ids"]),
            "signs": list(row["signs"]),
            "correction_expression": expression_text,
            "source_manifest_sha256": manifest["content_sha256"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        mapping = map_candidate_to_covariant_action(
            candidate,
            generator,
            grammar,
            field_contract,
            source_sha256=manifest["content_sha256"],
        )
        mapped_count += int(mapping["decision"] == "mapped")
        semantic_health: dict[str, Any] | None = None
        action_spec_sha256: str | None = None
        provenance_sha256: str | None = None
        if mapping["decision"] == "mapped":
            candidate_dir = output_root / str(row["candidate_id"])
            spec_path = candidate_dir / "action-spec.json"
            action_spec_sha256 = _sha(mapping["action_spec"])
            provenance_sha256 = mapping["covariant_action_provenance"]["provenance_binding_sha256"]
            _write_immutable_json(spec_path, mapping["action_spec"])
            semantic_path = candidate_dir / "semantic-action-health.json"
            if semantic_path.exists():
                semantic_health = json.loads(semantic_path.read_text(encoding="utf-8"))
                _validate_sealed(semantic_health)
            else:
                health = analyze_action_health(
                    spec_path,
                    root / config["grammar_path"],
                    root / config["field_contract_path"],
                    candidate_dir / "action-health",
                    project_root=root,
                    formal_report=formal_report,
                )
                semantic_health = _sealed(_semantic_health(health, candidate_dir))
                _write_immutable_json(semantic_path, semantic_health)
            health_count += 1
        decision, blocker = _candidate_decision(mapping, semantic_health)
        hard_reject_count += int(decision == "reject")
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "ordinal": row["ordinal"],
                "correction_expression_sha256": _sha(expression_text),
                "exact_symbolic_hessian_sha256": _sha(sp.srepr(hessian)),
                "algebraic_gates": gate_rows,
                "sampled_static_convexity": convexity,
                "local_formula_preflight_pass": formal_local_pass,
                "covariant_mapping_decision": mapping["decision"],
                "covariant_mapping_payload_sha256": _sha(mapping),
                "covariant_action_compiled": mapping["decision"] == "mapped",
                "covariant_action_spec_sha256": action_spec_sha256,
                "covariant_action_provenance_sha256": provenance_sha256,
                "semantic_action_health": semantic_health,
                "semantic_action_health_sha256": (
                    semantic_health["content_sha256"] if semantic_health else None
                ),
                "decision": decision,
                "first_blocker": blocker,
            }
        )
    aggregate_decision = (
        "reject"
        if records and hard_reject_count == len(records) and manifest["sample_complete"]
        else "block"
    )
    evidence_body = {
        "schema_version": EVIDENCE_SCHEMA,
        "backend_id": BACKEND_ID,
        "candidate_root_sha256": generated_receipt["candidate_root_sha256"],
        "generated_receipt_sha256": generated_receipt["content_sha256"],
        "candidate_manifest_sha256": manifest["content_sha256"],
        "source_pass_candidate_count": manifest["survivor_record_count"],
        "formally_checked_candidate_count": len(records),
        "symbolic_local_preflight_pass_count": symbolic_passes,
        "covariant_action_mapped_count": mapped_count,
        "action_health_execution_count": health_count,
        "hard_reject_count": hard_reject_count,
        "candidate_records": records,
        "candidate_records_root_sha256": _sha(records),
        "decision": aggregate_decision,
        "first_blocker": (
            "candidate_manifest_is_bounded_not_complete"
            if not manifest["sample_complete"]
            else (
                "all_formal_candidates_hard_rejected"
                if aggregate_decision == "reject"
                else "complete_comparable_candidate_evidence_not_registered"
            )
        ),
        "complete_comparable_evidence": False,
        "observations_opened": False,
        "forbidden_target_inputs_opened": False,
        "candidate_hard_reject_count": hard_reject_count,
        "theory_rejected": False,
        "direct_rank_assignment": False,
    }
    evidence = _sealed(evidence_body)
    receipt = _sealed(
        {
            "candidate_root_sha256": generated_receipt["candidate_root_sha256"],
            "generated_receipt_sha256": generated_receipt["content_sha256"],
            "decision": aggregate_decision,
            "complete_comparable_evidence": False,
            "observations_opened": False,
            "forbidden_target_inputs_opened": False,
        }
    )
    return receipt, evidence


def validate_formal_evidence(value: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    expected = {
        "schema_version",
        "backend_id",
        "candidate_root_sha256",
        "generated_receipt_sha256",
        "candidate_manifest_sha256",
        "source_pass_candidate_count",
        "formally_checked_candidate_count",
        "symbolic_local_preflight_pass_count",
        "covariant_action_mapped_count",
        "action_health_execution_count",
        "hard_reject_count",
        "candidate_records",
        "candidate_records_root_sha256",
        "decision",
        "first_blocker",
        "complete_comparable_evidence",
        "observations_opened",
        "forbidden_target_inputs_opened",
        "candidate_hard_reject_count",
        "theory_rejected",
        "direct_rank_assignment",
        "content_sha256",
    }
    records = value.get("candidate_records", [])
    numeric_counts = (
        "source_pass_candidate_count",
        "formally_checked_candidate_count",
        "symbolic_local_preflight_pass_count",
        "covariant_action_mapped_count",
        "action_health_execution_count",
        "hard_reject_count",
        "candidate_hard_reject_count",
    )
    if (
        set(value) != expected
        or value.get("schema_version") != EVIDENCE_SCHEMA
        or value.get("backend_id") != BACKEND_ID
        or value.get("decision") not in {"block", "reject"}
        or value.get("complete_comparable_evidence") is not False
        or value.get("theory_rejected") is not False
        or value.get("direct_rank_assignment") is not False
        or value.get("observations_opened") is not False
        or value.get("forbidden_target_inputs_opened") is not False
        or any(
            not _SHA256.fullmatch(str(value.get(key, "")))
            for key in (
                "candidate_root_sha256",
                "generated_receipt_sha256",
                "candidate_manifest_sha256",
                "candidate_records_root_sha256",
            )
        )
        or any(
            not isinstance(value.get(key), int)
            or isinstance(value.get(key), bool)
            or value[key] < 0
            for key in numeric_counts
        )
        or value.get("candidate_records_root_sha256") != _sha(records)
        or value.get("formally_checked_candidate_count") != len(records)
        or value.get("candidate_hard_reject_count") != value.get("hard_reject_count")
        or value.get("hard_reject_count") > len(records)
        or value.get("symbolic_local_preflight_pass_count") > len(records)
        or value.get("covariant_action_mapped_count") > len(records)
        or value.get("action_health_execution_count") > len(records)
        or value.get("decision") == "reject"
        and value.get("hard_reject_count") != len(records)
        or value.get("first_blocker")
        not in {
            "candidate_manifest_is_bounded_not_complete",
            "all_formal_candidates_hard_rejected",
            "complete_comparable_candidate_evidence_not_registered",
        }
    ):
        raise ValueError("formal evidence contract mismatch")
    for record in value["candidate_records"]:
        health = record.get("semantic_action_health")
        if health is not None:
            _validate_sealed(health)
            if health["content_sha256"] != record.get("semantic_action_health_sha256"):
                raise ValueError("formal evidence health binding mismatch")
