"""Fail-closed registry and aggregate readiness for the math benchmark curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "invariant-math-benchmark-curriculum-config-1.0"
RESULT_SCHEMA = "invariant-math-benchmark-curriculum-readiness-1.0"
CONFIG_PATH = "configs/math_benchmark_curriculum_v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/math_benchmark_runner.py"
TEST_PATH = "tests/test_math_benchmark_runner.py"
OUTPUT_PATH = "runs/math/math-benchmark-curriculum-v1/readiness.json"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SLOT = re.compile(
    r"(historical|synthetic)\."
    r"(algebra|arithmetic|combinatorics|geometry|number_theory)\."
    r"(counterexample_construction|theorem_rediscovery)\.l([1-5])\.(001|002)\Z"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = _sha(body)
    return body


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("curriculum path is not a nonempty portable relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("curriculum path escapes project root") from error
    return path


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _load_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("curriculum config path changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    _exact_keys(
        config,
        {
            "schema_version",
            "curriculum_id",
            "slot_registry",
            "immutable_manifest_contract",
            "implemented_controls",
            "aggregate_metrics_schema",
            "output_path",
        },
        "curriculum config",
    )
    registry = config["slot_registry"]
    _exact_keys(
        registry,
        {
            "slot_kind",
            "cohorts",
            "domains",
            "artifact_types",
            "levels",
            "slots_per_partition",
            "slot_id_format",
        },
        "slot registry",
    )
    manifest = config["immutable_manifest_contract"]
    _exact_keys(
        manifest,
        {
            "registered_slot_count",
            "historical_slot_count",
            "synthetic_slot_count",
            "require_exact_file_sha256",
            "require_self_sealed_content_sha256",
            "unregistered_results_forbidden",
            "missing_results_are_not_ready",
            "curriculum_success_requires_all_slots_ready",
        },
        "immutable manifest contract",
    )
    metrics = config["aggregate_metrics_schema"]
    _exact_keys(
        metrics,
        {"count_fields", "partition_dimensions", "outcome_fields", "rate_fields"},
        "aggregate metrics schema",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["curriculum_id"] != "math-benchmark-curriculum-v1"
        or config["output_path"] != OUTPUT_PATH
        or registry
        != {
            "slot_kind": "blind_holdout",
            "cohorts": ["historical", "synthetic"],
            "domains": ["algebra", "arithmetic", "combinatorics", "geometry", "number_theory"],
            "artifact_types": ["counterexample_construction", "theorem_rediscovery"],
            "levels": [1, 2, 3, 4, 5],
            "slots_per_partition": 2,
            "slot_id_format": "{cohort}.{domain}.{artifact_type}.l{level}.{ordinal:03d}",
        }
        or manifest
        != {
            "registered_slot_count": 200,
            "historical_slot_count": 100,
            "synthetic_slot_count": 100,
            "require_exact_file_sha256": True,
            "require_self_sealed_content_sha256": True,
            "unregistered_results_forbidden": True,
            "missing_results_are_not_ready": True,
            "curriculum_success_requires_all_slots_ready": True,
        }
        or metrics
        != {
            "count_fields": [
                "registered_slots",
                "implemented_controls",
                "ready_slots",
                "missing_slots",
                "invalid_slots",
            ],
            "partition_dimensions": ["cohort", "domain", "artifact_type", "level"],
            "outcome_fields": ["pass", "reject", "blocked"],
            "rate_fields": ["coverage_numerator", "coverage_denominator"],
        }
    ):
        raise ValueError("curriculum closed contract changed")
    controls = config["implemented_controls"]
    if not isinstance(controls, list) or len(controls) != 2:
        raise ValueError("curriculum must bind exactly two implemented controls")
    expected_keys = {
        "slot_id",
        "artifact_path",
        "file_sha256",
        "content_sha256",
        "benchmark_id",
        "schema_version",
        "decision",
        "outcome_counts",
        "required_claims",
    }
    for control in controls:
        _exact_keys(control, expected_keys, "implemented control")
        if (
            _SLOT.fullmatch(control["slot_id"]) is None
            or _HASH.fullmatch(control["file_sha256"]) is None
            or _HASH.fullmatch(control["content_sha256"]) is None
            or set(control["outcome_counts"]) != {"pass", "reject", "blocked"}
            or any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in control["outcome_counts"].values()
            )
            or not control["required_claims"]
            or any(not isinstance(value, bool) for value in control["required_claims"].values())
        ):
            raise ValueError("implemented control contract changed")
        _resolve(root, control["artifact_path"])
    control_ids = [control["slot_id"] for control in controls]
    if len(set(control_ids)) != len(control_ids):
        raise ValueError("implemented control slot IDs contain duplicates")
    return config


def expand_slots(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = config["slot_registry"]
    slots = [
        {
            "slot_id": registry["slot_id_format"].format(
                cohort=cohort,
                domain=domain,
                artifact_type=artifact_type,
                level=level,
                ordinal=ordinal,
            ),
            "slot_kind": registry["slot_kind"],
            "cohort": cohort,
            "domain": domain,
            "artifact_type": artifact_type,
            "level": level,
            "partition_ordinal": ordinal,
        }
        for cohort in registry["cohorts"]
        for domain in registry["domains"]
        for artifact_type in registry["artifact_types"]
        for level in registry["levels"]
        for ordinal in range(1, registry["slots_per_partition"] + 1)
    ]
    slots.sort(key=lambda row: row["slot_id"])
    ids = [row["slot_id"] for row in slots]
    counts = Counter(row["cohort"] for row in slots)
    if (
        len(slots) != config["immutable_manifest_contract"]["registered_slot_count"]
        or len(set(ids)) != len(ids)
        or any(_SLOT.fullmatch(slot_id) is None for slot_id in ids)
        or counts
        != Counter(
            {
                "historical": config["immutable_manifest_contract"]["historical_slot_count"],
                "synthetic": config["immutable_manifest_contract"]["synthetic_slot_count"],
            }
        )
    ):
        raise ValueError("expanded curriculum registry is not closed and unique")
    return slots


def _inspect_implementation(
    root: Path, slot: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    path = _resolve(root, control["artifact_path"])
    expectation = {
        "artifact_path": control["artifact_path"],
        "file_sha256": control["file_sha256"],
        "content_sha256": control["content_sha256"],
        "benchmark_id": control["benchmark_id"],
        "schema_version": control["schema_version"],
        "decision": control["decision"],
        "outcome_counts": control["outcome_counts"],
        "required_claims": control["required_claims"],
    }
    base = {**slot, "implemented": True, "immutable_manifest_expectation": expectation}
    if not path.is_file():
        return {
            **base,
            "status": "missing",
            "reason": "registered_immutable_artifact_missing",
            "validation_errors": [],
            "outcome_counts": {"pass": 0, "reject": 0, "blocked": 0},
        }
    errors: list[str] = []
    actual_file_sha = _file_sha(path)
    if actual_file_sha != control["file_sha256"]:
        errors.append("file_sha256_mismatch")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        artifact = None
        errors.append("artifact_json_unreadable")
    if not isinstance(artifact, dict):
        if "artifact_json_unreadable" not in errors:
            errors.append("artifact_json_not_object")
    else:
        content_sha = artifact.get("content_sha256")
        if content_sha != control["content_sha256"]:
            errors.append("content_sha256_binding_mismatch")
        if content_sha != _sha(
            {key: value for key, value in artifact.items() if key != "content_sha256"}
        ):
            errors.append("content_self_seal_invalid")
        for field in ("benchmark_id", "schema_version", "decision"):
            if artifact.get(field) != control[field]:
                errors.append(f"{field}_mismatch")
        claims = artifact.get("claims")
        if not isinstance(claims, dict) or any(
            claims.get(key) is not expected for key, expected in control["required_claims"].items()
        ):
            errors.append("required_claims_mismatch")
        if (
            "decision_counts" in artifact
            and artifact["decision_counts"] != control["outcome_counts"]
        ):
            errors.append("outcome_counts_mismatch")
    if errors:
        return {
            **base,
            "status": "invalid",
            "reason": "immutable_artifact_failed_validation",
            "validation_errors": sorted(errors),
            "outcome_counts": {"pass": 0, "reject": 0, "blocked": 0},
        }
    return {
        **base,
        "status": "ready",
        "reason": "immutable_artifact_validated",
        "validation_errors": [],
        "outcome_counts": dict(control["outcome_counts"]),
    }


def _partition_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dimension in ("cohort", "domain", "artifact_type", "level"):
        groups: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[record[dimension]].append(record)
        result[dimension] = [
            {
                "value": value,
                "registered": len(rows),
                "ready": sum(row["status"] == "ready" for row in rows),
                "missing": sum(row["status"] == "missing" for row in rows),
                "invalid": sum(row["status"] == "invalid" for row in rows),
            }
            for value, rows in sorted(groups.items(), key=lambda item: str(item[0]))
        ]
    return result


def build_readiness(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = _load_config(root, config_path)
    slots = expand_slots(config)
    controls = {control["slot_id"]: control for control in config["implemented_controls"]}
    if not set(controls) <= {slot["slot_id"] for slot in slots}:
        raise ValueError("implemented control is outside the closed registry")
    records = []
    for slot in slots:
        control = controls.get(slot["slot_id"])
        if control is None:
            records.append(
                {
                    **slot,
                    "implemented": False,
                    "immutable_manifest_expectation": {
                        "required": True,
                        "registration_status": "missing",
                        "artifact_path": None,
                        "file_sha256": None,
                        "content_sha256": None,
                    },
                    "status": "missing",
                    "reason": "no_immutable_result_registered",
                    "validation_errors": [],
                    "outcome_counts": {"pass": 0, "reject": 0, "blocked": 0},
                }
            )
        else:
            records.append(_inspect_implementation(root, slot, control))
    status_counts = Counter(record["status"] for record in records)
    outcomes = {
        name: sum(record["outcome_counts"][name] for record in records)
        for name in ("pass", "reject", "blocked")
    }
    counts = {
        "registered_slots": len(records),
        "implemented_controls": len(controls),
        "ready_slots": status_counts["ready"],
        "missing_slots": status_counts["missing"],
        "invalid_slots": status_counts["invalid"],
    }
    complete = counts["ready_slots"] == counts["registered_slots"] and not (
        counts["missing_slots"] or counts["invalid_slots"]
    )
    return _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "curriculum_id": config["curriculum_id"],
            "decision": "ready" if complete else "not_ready_missing_or_invalid_benchmarks",
            "curriculum_success": complete,
            "registry_root_sha256": _sha(slots),
            "immutable_manifest_contract": config["immutable_manifest_contract"],
            "aggregate_metrics_schema": config["aggregate_metrics_schema"],
            "aggregate_metrics": {
                "counts": counts,
                "coverage": {
                    "coverage_numerator": counts["ready_slots"],
                    "coverage_denominator": counts["registered_slots"],
                },
                "outcomes_from_validated_controls_only": outcomes,
                "partitions": _partition_metrics(records),
            },
            "slots": records,
            "claims": {
                "closed_unique_registry_validated": True,
                "only_implemented_validated_benchmarks_marked_ready": True,
                "missing_benchmarks_count_as_success": False,
                "unregistered_results_admitted": False,
                "all_registered_benchmarks_implemented": complete,
                "curriculum_complete": complete,
            },
            "bindings": {
                label: {"path": relative, "file_sha256": _file_sha(_resolve(root, relative))}
                for label, relative in (
                    ("config", CONFIG_PATH),
                    ("source", SOURCE_PATH),
                    ("test", TEST_PATH),
                )
            },
        }
    )


def validate_readiness(
    value: Mapping[str, Any], root: Path, config_path: Path | None = None
) -> None:
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("content_sha256")
        != _sha({key: item for key, item in value.items() if key != "content_sha256"})
        or value.get("curriculum_success") is not False
        or value.get("decision") != "not_ready_missing_or_invalid_benchmarks"
        or value.get("aggregate_metrics", {}).get("counts")
        != {
            "registered_slots": 200,
            "implemented_controls": 2,
            "ready_slots": 2,
            "missing_slots": 198,
            "invalid_slots": 0,
        }
        or value.get("claims", {}).get("missing_benchmarks_count_as_success") is not False
        or value.get("claims", {}).get("curriculum_complete") is not False
    ):
        raise ValueError("math benchmark curriculum readiness contract changed")
    if dict(value) != build_readiness(root, config_path):
        raise ValueError("math benchmark curriculum immutable replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_readiness(root, config_path)
    validate_readiness(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable readiness artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = run(root, _resolve(root, arguments.config))
    _write_immutable(_resolve(root, arguments.output), result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
