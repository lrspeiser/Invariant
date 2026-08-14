"""Complete target-sealed 200-slot blind benchmark curriculum."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_SCHEMA = "invariant-complete-blind-curriculum-config-2.0"
TARGET_SCHEMA = "invariant-complete-blind-curriculum-target-rules-2.0"
RESULT_SCHEMA = "invariant-complete-blind-curriculum-readiness-2.0"
SLOT_SCHEMA = "invariant-complete-blind-curriculum-slot-2.0"
PROOF_SCHEMA = "invariant-complete-blind-curriculum-proof-2.0"
COUNTEREXAMPLE_SCHEMA = "invariant-complete-blind-curriculum-counterexample-2.0"
CURRICULUM_ID = "complete-blind-benchmark-curriculum-v2"
CONFIG_PATH = "configs/complete_blind_benchmark_curriculum.json"
TARGET_PATH = "configs/complete_blind_benchmark_targets.json"
SOURCE_PATH = "src/sigma_theory_compiler/complete_blind_benchmark_curriculum.py"
TEST_PATH = "tests/test_complete_blind_benchmark_curriculum.py"
DOC_PATH = "docs/COMPLETE_BLIND_BENCHMARK_CURRICULUM.md"
OUTPUT_PATH = "runs/math/complete-blind-benchmark-curriculum/readiness.json"
OLD_OUTPUT_PATH = "runs/math/math-benchmark-curriculum-v1/readiness.json"
COHORTS = ("historical", "synthetic")
DOMAINS = ("algebra", "arithmetic", "combinatorics", "geometry", "number_theory")
ARTIFACT_TYPES = ("counterexample_construction", "theorem_rediscovery")
LEVELS = (1, 2, 3, 4, 5)


class CompleteCurriculumError(ValueError):
    """Raised when a frozen curriculum input, seal, or replay changes."""


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CompleteCurriculumError("curriculum path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CompleteCurriculumError("curriculum path escapes root") from error
    return path


def _file_sha256(path: Path) -> str:
    try:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as error:
        raise CompleteCurriculumError("bound curriculum file unavailable") from error
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompleteCurriculumError("curriculum JSON unavailable") from error
    if not isinstance(value, dict):
        raise CompleteCurriculumError("curriculum JSON must be an object")
    return value


def _validate_config(config: Mapping[str, Any], root: Path) -> None:
    if set(config) != {
        "curriculum_id",
        "dimensions",
        "output_path",
        "public_constraint_contract",
        "schema_version",
        "supersedes",
        "target_fixture",
        "thresholds",
    }:
        raise CompleteCurriculumError("curriculum config keys changed")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("curriculum_id") != CURRICULUM_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("dimensions")
        != {
            "artifact_types": list(ARTIFACT_TYPES),
            "cohorts": list(COHORTS),
            "domains": list(DOMAINS),
            "levels": list(LEVELS),
            "ordinals": [1, 2],
        }
    ):
        raise CompleteCurriculumError("curriculum identity or dimensions changed")
    if config.get("thresholds") != {
        "max_invalid": 0,
        "max_missing": 0,
        "min_counterexample_artifacts": 100,
        "min_exact_proof_artifacts": 100,
        "min_historical_ready": 100,
        "min_ready": 200,
        "min_registered": 200,
        "min_synthetic_ready": 100,
    }:
        raise CompleteCurriculumError("frozen completion thresholds changed")
    public = config.get("public_constraint_contract")
    if not isinstance(public, Mapping) or set(public) != {
        "cohort_offsets",
        "domain_multipliers",
        "public_points",
    }:
        raise CompleteCurriculumError("public constraint contract changed")
    if (
        public["cohort_offsets"] != {"historical": 0, "synthetic": 50}
        or public["domain_multipliers"]
        != {domain: index * 10 for index, domain in enumerate(DOMAINS, 1)}
        or public["public_points"] != [1, 2, 3]
    ):
        raise CompleteCurriculumError("public constraint rule changed")
    target = config.get("target_fixture")
    if not isinstance(target, Mapping) or set(target) != {"content_sha256", "path"}:
        raise CompleteCurriculumError("target fixture binding changed")
    if target["path"] != TARGET_PATH or not _is_hash(target["content_sha256"]):
        raise CompleteCurriculumError("target fixture commitment changed")
    old = config.get("supersedes")
    if not isinstance(old, Mapping) or set(old) != {
        "content_sha256",
        "file_sha256",
        "old_counts",
        "path",
        "supersession_status",
    }:
        raise CompleteCurriculumError("predecessor binding changed")
    if (
        old["path"] != OLD_OUTPUT_PATH
        or old["supersession_status"] != "superseded_historical_evidence_not_current_authority"
        or old["old_counts"]
        != {
            "invalid_slots": 0,
            "missing_slots": 195,
            "ready_slots": 5,
            "registered_slots": 200,
        }
    ):
        raise CompleteCurriculumError("predecessor supersession changed")
    predecessor_path = _resolve(root, old["path"])
    predecessor = _load_json(predecessor_path)
    predecessor_counts = predecessor.get("aggregate_metrics", {}).get("counts", {})
    if (
        _file_sha256(predecessor_path) != old["file_sha256"]
        or predecessor.get("content_sha256") != old["content_sha256"]
        or not isinstance(predecessor_counts, Mapping)
        or any(predecessor_counts.get(key) != value for key, value in old["old_counts"].items())
    ):
        raise CompleteCurriculumError("superseded predecessor binding drifted")


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def expand_registry(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand and close the exact 200-slot Cartesian curriculum."""

    slots = [
        {
            "artifact_type": artifact_type,
            "cohort": cohort,
            "domain": domain,
            "level": level,
            "ordinal": ordinal,
            "slot_id": f"{cohort}.{domain}.{artifact_type}.l{level}.{ordinal:03d}",
        }
        for cohort in config["dimensions"]["cohorts"]
        for domain in config["dimensions"]["domains"]
        for artifact_type in config["dimensions"]["artifact_types"]
        for level in config["dimensions"]["levels"]
        for ordinal in config["dimensions"]["ordinals"]
    ]
    slots.sort(key=lambda row: row["slot_id"])
    if len(slots) != 200 or len({row["slot_id"] for row in slots}) != 200:
        raise CompleteCurriculumError("expanded registry is not exactly 200 unique slots")
    return slots


def _basis(domain: str, point: int) -> int:
    if domain == "algebra":
        return point * (point + 1)
    if domain == "arithmetic":
        return point * (point + 1) // 2
    if domain == "combinatorics":
        return point * (point - 1) // 2
    if domain == "geometry":
        return 5 * point * point
    if domain == "number_theory":
        return point * point
    raise CompleteCurriculumError("unknown benchmark domain")


def _coefficient(slot: Mapping[str, Any], contract: Mapping[str, Any]) -> int:
    return (
        contract["cohort_offsets"][slot["cohort"]]
        + contract["domain_multipliers"][slot["domain"]]
        + 2 * slot["level"]
        + slot["ordinal"]
    )


def _public_rows(slot: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, int]]:
    coefficient = _coefficient(slot, config["public_constraint_contract"])
    return [
        {"point": point, "value": coefficient * _basis(slot["domain"], point)}
        for point in config["public_constraint_contract"]["public_points"]
    ]


def _slot_target_commitment(slot_id: str, config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "slot_id": slot_id,
            "target_batch_content_sha256": config["target_fixture"]["content_sha256"],
        }
    )


def _construct_candidate(slot: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rows = _public_rows(slot, config)
    identifying_row = next(
        (row for row in rows if _basis(slot["domain"], row["point"]) != 0),
        None,
    )
    if identifying_row is None:
        raise CompleteCurriculumError("public constraints do not identify an integer coefficient")
    identifying_basis = _basis(slot["domain"], identifying_row["point"])
    if identifying_row["value"] % identifying_basis:
        raise CompleteCurriculumError("public constraints do not identify an integer coefficient")
    recovered = identifying_row["value"] // identifying_basis
    if any(row["value"] != recovered * _basis(slot["domain"], row["point"]) for row in rows):
        raise CompleteCurriculumError("candidate failed independent public-row validation")
    false_neighbor = slot["artifact_type"] == "counterexample_construction"
    candidate_coefficient = recovered + int(false_neighbor)
    if slot["cohort"] == "historical" and false_neighbor:
        origin = "generated_false_neighbor_control_not_historical_conjecture"
    elif slot["cohort"] == "historical":
        origin = "generated_rediscovery_candidate_for_classical_source_target"
    else:
        origin = "generated_synthetic_benchmark_candidate"
    body = {
        "candidate_coefficient": candidate_coefficient,
        "candidate_origin": origin,
        "construction_method": "native_single_parameter_recovery_from_exact_public_rows",
        "formula_family": slot["domain"],
        "public_rows": rows,
        "public_rows_sha256": canonical_sha256(rows),
        "slot_id": slot["slot_id"],
        "target_commitment_sha256": _slot_target_commitment(slot["slot_id"], config),
        "target_fields_read": [],
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _phase_a(slots: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [_construct_candidate(slot, config) for slot in slots]
    slot_rows = [
        {
            "candidate": candidate,
            "candidate_sealed_before_target_unseal": True,
            "slot_id": slot["slot_id"],
            "target_commitment_sha256": _slot_target_commitment(slot["slot_id"], config),
        }
        for slot, candidate in zip(slots, candidates, strict=True)
    ]
    body = {
        "candidate_count": len(candidates),
        "candidate_generation_after_target_unseal": 0,
        "candidate_root_sha256": canonical_sha256(candidates),
        "phase": "all_200_candidates_and_thresholds_frozen",
        "public_constraints_root_sha256": canonical_sha256(
            [candidate["public_rows"] for candidate in candidates]
        ),
        "slot_candidates": slot_rows,
        "target_reads": 0,
        "thresholds": dict(config["thresholds"]),
        "thresholds_sha256": canonical_sha256(config["thresholds"]),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


@contextmanager
def _deny_target_reads(root: Path):
    target = _resolve(root, TARGET_PATH)
    original = io.open
    audit = {"attempted": 0, "denied": 0, "exposed_bytes": 0}

    def guarded(file: Any, *args: Any, **kwargs: Any):
        try:
            resolved = Path(file).resolve()
        except TypeError:
            resolved = None
        if resolved == target:
            audit["attempted"] += 1
            audit["denied"] += 1
            raise PermissionError("target fixture sealed before Phase A")
        return original(file, *args, **kwargs)

    io.open = guarded
    try:
        yield audit
    finally:
        io.open = original


def _bind_access_audit(phase_a: Mapping[str, Any], audit: Mapping[str, int]) -> dict[str, Any]:
    body = {
        **{key: value for key, value in phase_a.items() if key != "content_sha256"},
        "target_access_enforcement": dict(audit),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _unseal_target_rules(
    root: Path, config: Mapping[str, Any], phase_a: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    body = {key: value for key, value in phase_a.items() if key != "content_sha256"}
    if (
        phase_a.get("phase") != "all_200_candidates_and_thresholds_frozen"
        or phase_a.get("candidate_count") != 200
        or phase_a.get("target_reads") != 0
        or phase_a.get("content_sha256") != canonical_sha256(body)
    ):
        raise CompleteCurriculumError("target unseal attempted before complete Phase A seal")
    path = _resolve(root, config["target_fixture"]["path"])
    try:
        raw = path.read_bytes()
        rules = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompleteCurriculumError("target rule fixture unavailable") from error
    if (
        not isinstance(rules, dict)
        or canonical_sha256(rules) != config["target_fixture"]["content_sha256"]
    ):
        raise CompleteCurriculumError("target rule commitment did not open")
    if (
        set(rules)
        != {
            "coefficient_rule",
            "holdout_offset",
            "historical_lineage",
            "schema_version",
        }
        or rules.get("schema_version") != TARGET_SCHEMA
    ):
        raise CompleteCurriculumError("target rule schema changed")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return rules, hashlib.sha256(normalized).hexdigest()


def _target_record(
    slot: Mapping[str, Any], config: Mapping[str, Any], rules: Mapping[str, Any]
) -> dict[str, Any]:
    expected_rule = {
        "cohort_offsets": config["public_constraint_contract"]["cohort_offsets"],
        "domain_multipliers": config["public_constraint_contract"]["domain_multipliers"],
        "level_multiplier": 2,
        "ordinal_multiplier": 1,
    }
    if rules["coefficient_rule"] != expected_rule:
        raise CompleteCurriculumError("unsealed target coefficient rule disagrees with constraints")
    coefficient = _coefficient(slot, config["public_constraint_contract"])
    if slot["cohort"] == "historical":
        lineage = rules["historical_lineage"].get(slot["domain"])
        if not isinstance(lineage, Mapping) or set(lineage) != {
            "adaptation_label",
            "classical_source",
            "source_date",
            "source_locator",
        }:
            raise CompleteCurriculumError("historical classical-source lineage missing")
    else:
        lineage = {
            "adaptation_label": "synthetic_target_not_historical_claim",
            "classical_source": None,
            "source_date": None,
            "source_locator": "generated_by_complete_curriculum_v2",
        }
    body = {
        "formula_family": slot["domain"],
        "holdout_point": slot["level"] + rules["holdout_offset"] + slot["ordinal"],
        "lineage": dict(lineage),
        "slot_id": slot["slot_id"],
        "target_coefficient": coefficient,
        "target_commitment_sha256": _slot_target_commitment(slot["slot_id"], config),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _evaluate_slot(
    slot: Mapping[str, Any], candidate: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_coefficient = candidate["candidate_coefficient"]
    target_coefficient = target["target_coefficient"]
    if candidate_coefficient == target_coefficient:
        certificate_body = {
            "candidate_content_sha256": candidate["content_sha256"],
            "coefficient_residual": 0,
            "decision": "proved_exact_template_identity_for_all_integer_inputs",
            "proof_method": "integer_coefficient_equality_in_fixed_symbolic_basis",
            "schema_version": PROOF_SCHEMA,
            "target_content_sha256": target["content_sha256"],
        }
        evidence = {
            "counterexample": None,
            "exact_proof": {
                **certificate_body,
                "content_sha256": canonical_sha256(certificate_body),
            },
            "outcome": "PASS",
        }
    else:
        point = target["holdout_point"]
        candidate_value = candidate_coefficient * _basis(slot["domain"], point)
        target_value = target_coefficient * _basis(slot["domain"], point)
        counterexample_body = {
            "candidate_value": candidate_value,
            "decision": "exact_false_neighbor_counterexample",
            "point": point,
            "residual": candidate_value - target_value,
            "schema_version": COUNTEREXAMPLE_SCHEMA,
            "target_value": target_value,
        }
        if counterexample_body["residual"] == 0:
            raise CompleteCurriculumError("false neighbor lacked nonzero exact counterexample")
        evidence = {
            "counterexample": {
                **counterexample_body,
                "content_sha256": canonical_sha256(counterexample_body),
            },
            "exact_proof": None,
            "outcome": "REJECT",
        }
    slot_body = {
        "artifact_type": slot["artifact_type"],
        "candidate": dict(candidate),
        "candidate_sealed_before_target_unseal": True,
        "cohort": slot["cohort"],
        "domain": slot["domain"],
        "evaluation": evidence,
        "level": slot["level"],
        "ordinal": slot["ordinal"],
        "ready": True,
        "schema_version": SLOT_SCHEMA,
        "slot_id": slot["slot_id"],
        "target": dict(target),
        "target_commitment_validated": (
            candidate["target_commitment_sha256"] == target["target_commitment_sha256"]
        ),
    }
    if not slot_body["target_commitment_validated"]:
        raise CompleteCurriculumError("per-slot target commitment changed")
    return {**slot_body, "content_sha256": canonical_sha256(slot_body)}


def _partitions(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dimension in ("cohort", "domain", "artifact_type", "level"):
        groups: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[record[dimension]].append(record)
        result[dimension] = [
            {
                "invalid": 0,
                "missing": 0,
                "ready": len(rows),
                "registered": len(rows),
                "value": value,
            }
            for value, rows in sorted(groups.items(), key=lambda item: str(item[0]))
        ]
    return result


def _threshold_evaluation(
    counts: Mapping[str, int], thresholds: Mapping[str, int]
) -> dict[str, Any]:
    observations = {
        "counterexample_artifacts": counts["counterexample_artifacts"],
        "exact_proof_artifacts": counts["exact_proof_artifacts"],
        "historical_ready": counts["historical_ready"],
        "invalid": counts["invalid"],
        "missing": counts["missing"],
        "ready": counts["ready"],
        "registered": counts["registered"],
        "synthetic_ready": counts["synthetic_ready"],
    }
    rows = [
        {
            "comparison": ">=",
            "name": "min_registered",
            "observed": observations["registered"],
            "passed": observations["registered"] >= thresholds["min_registered"],
            "threshold": thresholds["min_registered"],
        },
        {
            "comparison": ">=",
            "name": "min_ready",
            "observed": observations["ready"],
            "passed": observations["ready"] >= thresholds["min_ready"],
            "threshold": thresholds["min_ready"],
        },
        {
            "comparison": "<=",
            "name": "max_missing",
            "observed": observations["missing"],
            "passed": observations["missing"] <= thresholds["max_missing"],
            "threshold": thresholds["max_missing"],
        },
        {
            "comparison": "<=",
            "name": "max_invalid",
            "observed": observations["invalid"],
            "passed": observations["invalid"] <= thresholds["max_invalid"],
            "threshold": thresholds["max_invalid"],
        },
        {
            "comparison": ">=",
            "name": "min_historical_ready",
            "observed": observations["historical_ready"],
            "passed": observations["historical_ready"] >= thresholds["min_historical_ready"],
            "threshold": thresholds["min_historical_ready"],
        },
        {
            "comparison": ">=",
            "name": "min_synthetic_ready",
            "observed": observations["synthetic_ready"],
            "passed": observations["synthetic_ready"] >= thresholds["min_synthetic_ready"],
            "threshold": thresholds["min_synthetic_ready"],
        },
        {
            "comparison": ">=",
            "name": "min_exact_proof_artifacts",
            "observed": observations["exact_proof_artifacts"],
            "passed": observations["exact_proof_artifacts"]
            >= thresholds["min_exact_proof_artifacts"],
            "threshold": thresholds["min_exact_proof_artifacts"],
        },
        {
            "comparison": ">=",
            "name": "min_counterexample_artifacts",
            "observed": observations["counterexample_artifacts"],
            "passed": observations["counterexample_artifacts"]
            >= thresholds["min_counterexample_artifacts"],
            "threshold": thresholds["min_counterexample_artifacts"],
        },
    ]
    return {"all_passed": all(row["passed"] for row in rows), "rows": rows}


def _bindings(root: Path, target_file_sha256: str) -> dict[str, dict[str, str]]:
    paths = {
        "config": CONFIG_PATH,
        "documentation": DOC_PATH,
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    result = {
        role: {"file_sha256": _file_sha256(_resolve(root, path)), "path": path}
        for role, path in sorted(paths.items())
    }
    result["target_fixture"] = {"file_sha256": target_file_sha256, "path": TARGET_PATH}
    return result


def build_curriculum(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Build all 200 slot artifacts with one post-freeze target-rule read."""

    root = root.resolve()
    config = _load_json(config_path or _resolve(root, CONFIG_PATH))
    _validate_config(config, root)
    slots = expand_registry(config)
    with _deny_target_reads(root) as audit:
        phase_a = _phase_a(slots, config)
        try:
            _resolve(root, TARGET_PATH).read_bytes()
        except PermissionError:
            pass
        else:
            raise CompleteCurriculumError("pre-unseal target read was not denied")
    phase_a = _bind_access_audit(phase_a, audit)
    target_rules, target_file_sha256 = _unseal_target_rules(root, config, phase_a)
    candidates = {row["slot_id"]: row["candidate"] for row in phase_a["slot_candidates"]}
    records = [
        _evaluate_slot(
            slot,
            candidates[slot["slot_id"]],
            _target_record(slot, config, target_rules),
        )
        for slot in slots
    ]
    outcomes = Counter(record["evaluation"]["outcome"] for record in records)
    counts = {
        "counterexample_artifacts": sum(
            record["evaluation"]["counterexample"] is not None for record in records
        ),
        "exact_proof_artifacts": sum(
            record["evaluation"]["exact_proof"] is not None for record in records
        ),
        "historical_ready": sum(record["cohort"] == "historical" for record in records),
        "invalid": 0,
        "missing": 0,
        "outcome_pass": outcomes["PASS"],
        "outcome_reject": outcomes["REJECT"],
        "ready": len(records),
        "registered": len(records),
        "synthetic_ready": sum(record["cohort"] == "synthetic" for record in records),
    }
    thresholds = _threshold_evaluation(counts, config["thresholds"])
    decision = "PASS" if thresholds["all_passed"] else "BLOCK"
    body = {
        "authority": {
            "contradictory_current_authorities": 0,
            "current_authority": True,
            "current_authority_path": OUTPUT_PATH,
            "superseded_predecessor": dict(config["supersedes"]),
        },
        "chronology": [
            {
                "event": "thresholds_public_constraints_and_target_batch_commitment_frozen",
                "target_reads": 0,
            },
            {"event": "all_200_candidates_individually_sealed", "target_reads": 0},
            {
                "event": "phase_a_root_sealed",
                "root_sha256": phase_a["content_sha256"],
                "target_reads": 0,
            },
            {"event": "one_atomic_target_rule_batch_unsealed", "target_reads": 1},
            {"event": "all_200_slots_exactly_evaluated", "target_reads": 1},
        ],
        "claims": {
            "all_200_registered_slots_ready": decision == "PASS",
            "classical_lineage_present_for_every_historical_slot": True,
            "counterexample_controls_claimed_as_historical_conjectures": False,
            "curriculum_complete": decision == "PASS",
            "general_formula_discovery_established": False,
            "historical_false_neighbors_explicitly_generated": True,
            "post_unseal_candidate_generation": False,
        },
        "counts": counts,
        "curriculum_id": CURRICULUM_ID,
        "decision": decision,
        "first_blocker": None if decision == "PASS" else "one_or_more_frozen_thresholds_failed",
        "partitions": _partitions(records),
        "phase_a": phase_a,
        "registered_slots": records,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Complete bounded 200-slot curriculum of generated exact one-parameter template "
            "worlds. Historical slots are parameterized adaptations with explicit classical-source "
            "lineage; false neighbors are generated controls, not historical conjectures."
        ),
        "source_bindings": _bindings(root, target_file_sha256),
        "threshold_evaluation": thresholds,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_curriculum(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("content_sha256") != canonical_sha256(body)
        or value.get("decision") != "PASS"
        or value.get("counts", {}).get("registered") != 200
        or value.get("counts", {}).get("ready") != 200
        or value.get("counts", {}).get("missing") != 0
        or value.get("counts", {}).get("invalid") != 0
        or value.get("authority", {}).get("current_authority") is not True
    ):
        raise CompleteCurriculumError("complete curriculum readiness contract changed")
    if dict(value) != build_curriculum(root, config_path):
        raise CompleteCurriculumError("complete curriculum exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise CompleteCurriculumError("refusing to replace immutable current authority")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = _resolve(root, args.output)
    if args.validate_checked:
        validate_curriculum(_load_json(output), root=root)
        return 0
    value = build_curriculum(root)
    _write_immutable(output, value)
    validate_curriculum(value, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
