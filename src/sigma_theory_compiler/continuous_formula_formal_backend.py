from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from .action_health import analyze_action_health
from .action_ir import compile_action_file
from .formal_backend import load_field_contract
from .gates import algebraic_gates, sampled_static_convexity
from .grammar import Q, X, Z
from .high_throughput import build_basis, candidate_id, correction_expression, decode_ordinal
from .production_covariant_provenance import map_candidate_to_covariant_action
from .promotion_orchestrator import ELIGIBILITY
from .real_formula_execution import _assets, _batch_arrays

CONFIG_SCHEMA = "sigma-continuous-formula-formal-backend-config-1.0"
MANIFEST_SCHEMA = "sigma-continuous-formula-candidate-manifest-1.0"
EVIDENCE_SCHEMA = "sigma-continuous-formula-formal-evidence-1.1"
BACKEND_ID = "candidate_bound_covariant_action_health_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^STC2-[0-9a-f]{24}$")
_GENERATOR_EXPRESSION = re.compile(r"^[0-9xqzsqrt()+*/ .-]+$")
_GATE_STATUSES = {"pass", "reject", "unresolved", "blocked", "not_applicable"}
_GENERATED_HEALTH_NAMES = (
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
_GENERATED_ARTIFACT_PATHS = {
    "generated_static_dictionary_ir": "action-health/static-dictionary-ir.json",
    "generated_q_operator_ir": "action-health/q-operator-ir.json",
    "generated_q_variation_ir": "action-health/q-variation-ir.json",
    "generated_higher_jet_auxiliary_ir": "action-health/higher-jet-auxiliary-ir.json",
    "generated_x_operator_ir": "action-health/x-operator-ir.json",
    "generated_adm_ir": "action-health/adm-ir.json",
    "generated_legendre_ir": "action-health/legendre-ir.json",
    "generated_dirac_ir": "action-health/dirac-ir.json",
    "generated_stability_ir": "action-health/stability-ir.json",
    "generated_principal_ir": "action-health/principal-ir.json",
    "generated_hamiltonian_ir": "action-health/hamiltonian-ir.json",
}
_DEPENDENCY_BINDING_KEYS = (
    ("generator_config_path", "generator_config_file_sha256"),
    ("grammar_path", "grammar_file_sha256"),
    ("field_contract_path", "field_contract_file_sha256"),
    ("formal_controls_path", "formal_controls_file_sha256"),
    ("candidate_mapper_source_path", "candidate_mapper_source_file_sha256"),
    ("action_health_source_path", "action_health_source_file_sha256"),
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _dependency_file_sha256(path: Path) -> str:
    """Hash text dependencies with repository-stable LF line endings."""
    raw = path.read_bytes()
    try:
        normalized = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        normalized = raw
    return hashlib.sha256(normalized).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _validate_sealed(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != _sha(body):
        raise ValueError("formal backend content hash mismatch")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


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
    for path_key, hash_key in _DEPENDENCY_BINDING_KEYS:
        bound = (root / config[path_key]).resolve()
        bound.relative_to(root.resolve())
        if _dependency_file_sha256(bound) != config[hash_key]:
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
    interval_keys = ("start_ordinal", "end_ordinal_exclusive", "candidate_count")
    result_counts = result.get("counts")
    if (
        any(not _is_nonnegative_int(payload.get(key)) for key in interval_keys)
        or payload["candidate_count"] == 0
        or payload["end_ordinal_exclusive"] - payload["start_ordinal"] != payload["candidate_count"]
        or not isinstance(result_counts, Mapping)
        or set(result_counts) != {"reject", "pass", "ambiguous"}
        or any(not _is_nonnegative_int(count) for count in result_counts.values())
        or sum(result_counts.values()) != payload["candidate_count"]
        or not _is_sha256(result.get("status_root_sha256"))
    ):
        raise ValueError("candidate extraction input accounting is invalid")
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
    exact_status_root = hashlib.sha256(np.asarray(statuses, dtype=np.uint8).tobytes()).hexdigest()
    if exact_status_root != result.get("status_root_sha256"):
        raise ValueError("candidate extraction replay differs from evaluator status root")
    expected_batch = {
        "start_ordinal": int(payload["start_ordinal"]),
        "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
        "candidate_count": int(payload["candidate_count"]),
    }
    if result.get("batch") != expected_batch:
        raise ValueError("candidate extraction replay differs from evaluator batch")
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "batch": expected_batch,
        "candidate_root_sha256": exact_status_root,
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
    if not manifests:
        raise ValueError("candidate manifest combination requires at least one shard")
    if not _is_nonnegative_int(maximum) or not 1 <= maximum <= 32:
        raise ValueError("candidate manifest combination bound is invalid")
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
    sample_complete = (
        all(value["sample_complete"] for value in ordered) and survivor_count <= maximum
    )
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
        "all_survivor_ordinals_root_sha256": (
            _sha([row["ordinal"] for row in selected])
            if sample_complete
            else _sha([value["all_survivor_ordinals_root_sha256"] for value in ordered])
        ),
        "survivor_records": selected,
        "survivor_record_count": survivor_count,
        "sample_complete": sample_complete,
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
    if not _is_sha256(value["candidate_root_sha256"]) or not _is_sha256(
        value["all_survivor_ordinals_root_sha256"]
    ):
        raise ValueError("candidate manifest root mismatch")
    batch = value["batch"]
    counts = value["screen_counts"]
    if (
        not isinstance(batch, Mapping)
        or not isinstance(counts, Mapping)
        or set(batch) != {"start_ordinal", "end_ordinal_exclusive", "candidate_count"}
        or set(counts) != {"reject", "pass", "ambiguous"}
        or any(not _is_nonnegative_int(item) for item in (*batch.values(), *counts.values()))
        or batch["candidate_count"] == 0
        or batch["end_ordinal_exclusive"] - batch["start_ordinal"] != batch["candidate_count"]
        or sum(counts.values()) != batch["candidate_count"]
        or not _is_nonnegative_int(value["survivor_record_count"])
        or value["survivor_record_count"] != counts["pass"]
        or not isinstance(value["sample_complete"], bool)
    ):
        raise ValueError("candidate manifest count contract mismatch")
    records = value["survivor_records"]
    if not isinstance(records, list) or len(records) > 32:
        raise ValueError("candidate manifest record bound or identity mismatch")
    record_keys = {
        "candidate_id",
        "ordinal",
        "term_ids",
        "signs",
        "correction_expression",
        "sampled_static_margin",
    }
    ordinals: list[int] = []
    identities: list[str] = []
    for row in records:
        if not isinstance(row, Mapping) or set(row) != record_keys:
            raise ValueError("candidate manifest record contract mismatch")
        ordinal = row["ordinal"]
        term_ids = row["term_ids"]
        signs = row["signs"]
        margin = row["sampled_static_margin"]
        if (
            not isinstance(row["candidate_id"], str)
            or _CANDIDATE_ID.fullmatch(row["candidate_id"]) is None
            or not _is_nonnegative_int(ordinal)
            or not batch["start_ordinal"] <= ordinal < batch["end_ordinal_exclusive"]
            or not isinstance(term_ids, list)
            or not term_ids
            or len(term_ids) > 6
            or any(not _is_nonnegative_int(item) for item in term_ids)
            or term_ids != sorted(set(term_ids))
            or not isinstance(signs, list)
            or len(signs) != len(term_ids)
            or any(type(sign) is not int or sign not in {-1, 1} for sign in signs)
            or not isinstance(row["correction_expression"], str)
            or not row["correction_expression"]
            or not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
        ):
            raise ValueError("candidate manifest record value contract mismatch")
        ordinals.append(ordinal)
        identities.append(row["candidate_id"])
    if ordinals != sorted(set(ordinals)) or len(identities) != len(set(identities)):
        raise ValueError("candidate manifest record bound or identity mismatch")
    if bool(records) is not (value["survivor_record_count"] > 0):
        raise ValueError("candidate manifest survivor evidence mismatch")
    if value["sample_complete"] is not (value["survivor_record_count"] == len(records)):
        raise ValueError("candidate manifest completeness mismatch")
    if value["sample_complete"] and value["all_survivor_ordinals_root_sha256"] != _sha(ordinals):
        raise ValueError("candidate manifest survivor ordinal root mismatch")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"immutable candidate artifact differs: {path.name}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _dependency_bindings(root: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for path_key, hash_key in _DEPENDENCY_BINDING_KEYS:
        path = (root / str(config[path_key])).resolve()
        path.relative_to(root.resolve())
        file_sha256 = _dependency_file_sha256(path)
        if file_sha256 != config[hash_key]:
            raise ValueError(f"formal backend dependency changed: {path_key}")
        bindings.append(
            {
                "role": path_key.removesuffix("_path"),
                "path": path.relative_to(root.resolve()).as_posix(),
                "file_sha256": file_sha256,
            }
        )
    return bindings


def _artifact_bindings(candidate_dir: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for path in sorted(candidate_dir.rglob("*.json")):
        if path.name in {"action-health.json", "semantic-action-health.json"}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("semantic action-health artifact is not a JSON object")
        bindings.append(
            {
                "path": path.relative_to(candidate_dir).as_posix(),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "declared_content_sha256": value.get("content_sha256"),
                "schema_version": value.get("schema_version"),
                "status": value.get("status"),
            }
        )
    return bindings


def _current_cache_binding(
    *,
    root: Path,
    config: Mapping[str, Any],
    candidate_dir: Path,
    candidate_id_value: str,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    spec_path = candidate_dir / "action-spec.json"
    compiled = compile_action_file(
        spec_path,
        root / str(config["grammar_path"]),
        root / str(config["field_contract_path"]),
    )
    if compiled.get("valid") is not True or not _is_sha256(compiled.get("content_sha256")):
        raise ValueError("cached semantic action-health action no longer compiles")
    dependencies = _dependency_bindings(root, config)
    return {
        "schema_version": "sigma-continuous-formula-health-cache-binding-1.0",
        "backend_id": BACKEND_ID,
        "candidate_id": candidate_id_value,
        "action_spec_payload_sha256": _sha(mapping["action_spec"]),
        "action_spec_file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "input_action_sha256": compiled["content_sha256"],
        "covariant_mapping_payload_sha256": _sha(mapping),
        "dependency_bindings": dependencies,
        "dependency_bindings_root_sha256": _sha(dependencies),
    }


def _semantic_health(
    report: Mapping[str, Any],
    candidate_dir: Path,
    cache_binding: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_bindings = _artifact_bindings(candidate_dir)
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
            for name in _GENERATED_HEALTH_NAMES
        },
        "artifact_bindings": artifact_bindings,
        "artifact_bindings_root_sha256": _sha(artifact_bindings),
        "cache_binding": dict(cache_binding),
        "observational_gates_unsealed": report.get("observational_gates_unsealed"),
        "interpretation": report.get("interpretation"),
    }


def _validate_semantic_health(
    value: Mapping[str, Any],
    *,
    candidate_dir: Path | None = None,
    expected_cache_binding: Mapping[str, Any] | None = None,
) -> None:
    _validate_sealed(value)
    expected = {
        "schema_version",
        "status",
        "promotion_allowed",
        "family",
        "input_action_sha256",
        "physical_dof",
        "gate_statuses",
        "gate_scopes",
        "discovery_blockers",
        "generated_artifacts",
        "artifact_bindings",
        "artifact_bindings_root_sha256",
        "cache_binding",
        "observational_gates_unsealed",
        "interpretation",
        "content_sha256",
    }
    statuses = value.get("gate_statuses")
    scopes = value.get("gate_scopes")
    blockers = value.get("discovery_blockers")
    generated = value.get("generated_artifacts")
    artifacts = value.get("artifact_bindings")
    cache = value.get("cache_binding")
    if (
        set(value) != expected
        or value.get("schema_version") != "sigma-action-health-1.0"
        or value.get("status") not in {"pass", "control_pass", "reject", "unresolved"}
        or not isinstance(value.get("promotion_allowed"), bool)
        or not isinstance(value.get("family"), str)
        or not value["family"]
        or not _is_sha256(value.get("input_action_sha256"))
        or (
            value.get("physical_dof") is not None
            and not _is_nonnegative_int(value.get("physical_dof"))
        )
        or not isinstance(statuses, Mapping)
        or not statuses
        or not isinstance(scopes, Mapping)
        or set(scopes) != set(statuses)
        or any(not isinstance(name, str) or not name for name in statuses)
        or any(status not in _GATE_STATUSES for status in statuses.values())
        or any(not isinstance(scope, str) or not scope for scope in scopes.values())
        or not isinstance(blockers, list)
        or any(not isinstance(blocker, str) or not blocker for blocker in blockers)
        or len(blockers) != len(set(blockers))
        or value.get("observational_gates_unsealed") is not False
        or not isinstance(value.get("interpretation"), str)
        or not value["interpretation"]
    ):
        raise ValueError("semantic action-health contract mismatch")
    if (
        (value["promotion_allowed"] is True) is not (value["status"] == "pass" and not blockers)
        or value["status"] == "reject"
        and "reject" not in statuses.values()
        or value["status"] == "unresolved"
        and "unresolved" not in statuses.values()
        or value["status"] == "control_pass"
        and (not blockers or any(status != "pass" for status in statuses.values()))
    ):
        raise ValueError("semantic action-health status evidence mismatch")
    if not isinstance(generated, Mapping) or set(generated) != set(_GENERATED_HEALTH_NAMES):
        raise ValueError("semantic action-health generated-artifact contract mismatch")
    for item in generated.values():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"status", "content_sha256"}
            or item["status"] not in _GATE_STATUSES
            or item["content_sha256"] is not None
            and not _is_sha256(item["content_sha256"])
        ):
            raise ValueError("semantic action-health generated-artifact contract mismatch")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("semantic action-health file closure is empty")
    artifact_by_path: dict[str, Mapping[str, Any]] = {}
    artifact_keys = {
        "path",
        "file_sha256",
        "declared_content_sha256",
        "schema_version",
        "status",
    }
    for binding in artifacts:
        if not isinstance(binding, Mapping) or set(binding) != artifact_keys:
            raise ValueError("semantic action-health file binding contract mismatch")
        relative = Path(str(binding["path"]))
        if (
            not isinstance(binding["path"], str)
            or not binding["path"]
            or relative.is_absolute()
            or ".." in relative.parts
            or not _is_sha256(binding["file_sha256"])
            or binding["declared_content_sha256"] is not None
            and not _is_sha256(binding["declared_content_sha256"])
            or binding["schema_version"] is not None
            and not isinstance(binding["schema_version"], str)
            or binding["status"] is not None
            and not isinstance(binding["status"], str)
            or binding["path"] in artifact_by_path
        ):
            raise ValueError("semantic action-health file binding value mismatch")
        artifact_by_path[binding["path"]] = binding
    if list(artifact_by_path) != sorted(artifact_by_path):
        raise ValueError("semantic action-health file closure is not canonical")
    if value.get("artifact_bindings_root_sha256") != _sha(artifacts):
        raise ValueError("semantic action-health file closure root mismatch")
    action_spec = artifact_by_path.get("action-spec.json")
    action_ir = artifact_by_path.get("action-health/action-ir.json")
    if action_spec is None or action_ir is None:
        raise ValueError("semantic action-health action binding is missing")
    if action_ir["declared_content_sha256"] != value["input_action_sha256"]:
        raise ValueError("semantic action-health input action mismatch")
    for name, path in _GENERATED_ARTIFACT_PATHS.items():
        binding = artifact_by_path.get(path)
        if binding is None or generated[name] != {
            "status": binding["status"],
            "content_sha256": binding["declared_content_sha256"],
        }:
            raise ValueError("semantic action-health generated file binding mismatch")
    cache_expected = {
        "schema_version",
        "backend_id",
        "candidate_id",
        "action_spec_payload_sha256",
        "action_spec_file_sha256",
        "input_action_sha256",
        "covariant_mapping_payload_sha256",
        "dependency_bindings",
        "dependency_bindings_root_sha256",
    }
    if not isinstance(cache, Mapping) or set(cache) != cache_expected:
        raise ValueError("semantic action-health cache binding contract mismatch")
    dependencies = cache["dependency_bindings"]
    if (
        cache["schema_version"] != "sigma-continuous-formula-health-cache-binding-1.0"
        or cache["backend_id"] != BACKEND_ID
        or not isinstance(cache["candidate_id"], str)
        or _CANDIDATE_ID.fullmatch(cache["candidate_id"]) is None
        or any(
            not _is_sha256(cache[key])
            for key in (
                "action_spec_payload_sha256",
                "action_spec_file_sha256",
                "input_action_sha256",
                "covariant_mapping_payload_sha256",
                "dependency_bindings_root_sha256",
            )
        )
        or cache["input_action_sha256"] != value["input_action_sha256"]
        or cache["action_spec_file_sha256"] != action_spec["file_sha256"]
        or not isinstance(dependencies, list)
        or len(dependencies) != len(_DEPENDENCY_BINDING_KEYS)
        or cache["dependency_bindings_root_sha256"] != _sha(dependencies)
    ):
        raise ValueError("semantic action-health cache binding mismatch")
    dependency_keys = {"role", "path", "file_sha256"}
    roles: list[str] = []
    for binding in dependencies:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != dependency_keys
            or not isinstance(binding["role"], str)
            or not binding["role"]
            or not isinstance(binding["path"], str)
            or not binding["path"]
            or Path(binding["path"]).is_absolute()
            or ".." in Path(binding["path"]).parts
            or not _is_sha256(binding["file_sha256"])
        ):
            raise ValueError("semantic action-health dependency binding mismatch")
        roles.append(binding["role"])
    expected_roles = [path_key.removesuffix("_path") for path_key, _ in _DEPENDENCY_BINDING_KEYS]
    if roles != expected_roles:
        raise ValueError("semantic action-health dependency binding mismatch")
    if expected_cache_binding is not None and cache != expected_cache_binding:
        raise ValueError("semantic action-health cache is stale or transplanted")
    if candidate_dir is not None and artifacts != _artifact_bindings(candidate_dir):
        raise ValueError("semantic action-health bound file changed")


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
            cache_binding = _current_cache_binding(
                root=root,
                config=config,
                candidate_dir=candidate_dir,
                candidate_id_value=str(row["candidate_id"]),
                mapping=mapping,
            )
            semantic_path = candidate_dir / "semantic-action-health.json"
            if semantic_path.exists():
                semantic_health = json.loads(semantic_path.read_text(encoding="utf-8"))
                _validate_semantic_health(
                    semantic_health,
                    candidate_dir=candidate_dir,
                    expected_cache_binding=cache_binding,
                )
            else:
                health = analyze_action_health(
                    spec_path,
                    root / config["grammar_path"],
                    root / config["field_contract_path"],
                    candidate_dir / "action-health",
                    project_root=root,
                    formal_report=formal_report,
                )
                semantic_health = _sealed(_semantic_health(health, candidate_dir, cache_binding))
                _validate_semantic_health(
                    semantic_health,
                    candidate_dir=candidate_dir,
                    expected_cache_binding=cache_binding,
                )
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
                "covariant_mapping_payload": mapping,
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
        "candidate_manifest_sample_complete": manifest["sample_complete"],
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


def _validate_local_gate(value: Any, *, expected_name: str | None = None) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"name", "status", "reason", "evidence"}
        or not isinstance(value["name"], str)
        or not value["name"]
        or expected_name is not None
        and value["name"] != expected_name
        or value["status"] not in {"pass", "reject"}
        or not isinstance(value["reason"], str)
        or not value["reason"]
        or not isinstance(value["evidence"], Mapping)
        or not value["evidence"]
    ):
        raise ValueError("formal evidence local gate contract mismatch")
    return str(value["status"])


def _validate_mapping_payload(
    payload: Any,
    record: Mapping[str, Any],
    candidate_manifest_sha256: str,
) -> None:
    common = {
        "candidate_id",
        "ordinal",
        "correction_expression",
        "candidate_payload_sha256",
        "term_ids",
        "signs",
        "decision",
        "data_eligibility",
    }
    if (
        not isinstance(payload, Mapping)
        or payload.get("candidate_id") != record["candidate_id"]
        or payload.get("ordinal") != record["ordinal"]
        or not isinstance(payload.get("correction_expression"), str)
        or not payload["correction_expression"]
        or len(payload["correction_expression"]) > 4096
        or _GENERATOR_EXPRESSION.fullmatch(payload["correction_expression"]) is None
        or _sha(payload["correction_expression"]) != record["correction_expression_sha256"]
        or not _is_sha256(payload.get("candidate_payload_sha256"))
        or not isinstance(payload.get("term_ids"), list)
        or not payload["term_ids"]
        or any(not _is_nonnegative_int(item) for item in payload["term_ids"])
        or payload["term_ids"] != sorted(set(payload["term_ids"]))
        or not isinstance(payload.get("signs"), list)
        or len(payload["signs"]) != len(payload["term_ids"])
        or any(type(sign) is not int or sign not in {-1, 1} for sign in payload["signs"])
        or payload.get("decision") not in {"mapped", "blocked", "reject"}
        or payload.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("formal evidence covariant mapping contract mismatch")
    candidate_payload = {
        "candidate_id": payload["candidate_id"],
        "ordinal": payload["ordinal"],
        "term_ids": payload["term_ids"],
        "signs": payload["signs"],
        "correction_expression": payload["correction_expression"],
        "source_manifest_sha256": candidate_manifest_sha256,
        "data_eligibility": dict(ELIGIBILITY),
    }
    if payload["candidate_payload_sha256"] != _sha(candidate_payload):
        raise ValueError("formal evidence covariant mapping lineage mismatch")
    decision = payload["decision"]
    if decision == "reject":
        if (
            set(payload) != common | {"reason"}
            or not isinstance(payload["reason"], str)
            or not payload["reason"]
        ):
            raise ValueError("formal evidence mapping rejection lacks evidence")
        return
    if decision == "blocked":
        allowed_shapes = (
            common | {"blockers", "unsupported_term_ids", "nonlinear_q_term_ids"},
            common | {"blockers", "action_errors"},
        )
        blockers = payload.get("blockers")
        if (
            set(payload) not in allowed_shapes
            or not isinstance(blockers, list)
            or not blockers
            or any(not isinstance(blocker, str) or not blocker for blocker in blockers)
            or blockers != sorted(set(blockers))
        ):
            raise ValueError("formal evidence mapping block lacks evidence")
        if "unsupported_term_ids" in payload and (
            not isinstance(payload["unsupported_term_ids"], list)
            or any(not _is_nonnegative_int(item) for item in payload["unsupported_term_ids"])
            or not isinstance(payload["nonlinear_q_term_ids"], list)
            or any(not _is_nonnegative_int(item) for item in payload["nonlinear_q_term_ids"])
        ):
            raise ValueError("formal evidence mapping block detail mismatch")
        if "action_errors" in payload and (
            not isinstance(payload["action_errors"], list)
            or any(not isinstance(error, str) or not error for error in payload["action_errors"])
        ):
            raise ValueError("formal evidence mapping block detail mismatch")
        return
    expected_mapped = common | {
        "covariant_action_provenance",
        "action_spec",
        "formal_preflight",
        "scope",
    }
    provenance = payload.get("covariant_action_provenance")
    if (
        set(payload) != expected_mapped
        or not isinstance(payload.get("action_spec"), Mapping)
        or not isinstance(provenance, Mapping)
        or not _is_sha256(provenance.get("provenance_binding_sha256"))
        or provenance["provenance_binding_sha256"]
        != _sha(
            {key: item for key, item in provenance.items() if key != "provenance_binding_sha256"}
        )
        or provenance.get("candidate_id") != record["candidate_id"]
        or provenance.get("ordinal") != record["ordinal"]
        or provenance.get("action_spec_sha256") != _sha(payload["action_spec"])
        or not _is_sha256(provenance.get("input_action_sha256"))
        or provenance.get("data_eligibility") != ELIGIBILITY
        or not isinstance(payload.get("formal_preflight"), Mapping)
        or set(payload["formal_preflight"])
        != {"decision", "q_operator_status", "q_operator_conclusion"}
        or payload["formal_preflight"]["decision"]
        not in {"reject_higher_jet_regularity", "formal_backend_queue"}
        or not isinstance(payload.get("scope"), str)
        or not payload["scope"]
    ):
        raise ValueError("formal evidence mapped action contract mismatch")


def _validate_candidate_record(
    record: Any, candidate_manifest_sha256: str
) -> tuple[bool, bool, bool]:
    keys = {
        "candidate_id",
        "ordinal",
        "correction_expression_sha256",
        "exact_symbolic_hessian_sha256",
        "algebraic_gates",
        "sampled_static_convexity",
        "local_formula_preflight_pass",
        "covariant_mapping_decision",
        "covariant_mapping_payload",
        "covariant_mapping_payload_sha256",
        "covariant_action_compiled",
        "covariant_action_spec_sha256",
        "covariant_action_provenance_sha256",
        "semantic_action_health",
        "semantic_action_health_sha256",
        "decision",
        "first_blocker",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != keys
        or not isinstance(record["candidate_id"], str)
        or _CANDIDATE_ID.fullmatch(record["candidate_id"]) is None
        or not _is_nonnegative_int(record["ordinal"])
        or not _is_sha256(record["correction_expression_sha256"])
        or not _is_sha256(record["exact_symbolic_hessian_sha256"])
        or not isinstance(record["local_formula_preflight_pass"], bool)
        or record["covariant_mapping_decision"] not in {"mapped", "blocked", "reject"}
        or not _is_sha256(record["covariant_mapping_payload_sha256"])
        or not isinstance(record["covariant_action_compiled"], bool)
        or record["decision"] not in {"block", "reject"}
        or not isinstance(record["first_blocker"], str)
        or not record["first_blocker"]
    ):
        raise ValueError("formal evidence candidate record contract mismatch")
    payload = record["covariant_mapping_payload"]
    if record["covariant_mapping_payload_sha256"] != _sha(payload):
        raise ValueError("formal evidence covariant mapping hash mismatch")
    _validate_mapping_payload(payload, record, candidate_manifest_sha256)
    if record["covariant_mapping_decision"] != payload["decision"]:
        raise ValueError("formal evidence covariant mapping decision mismatch")
    expression = sp.sympify(
        payload["correction_expression"], locals={"x": X, "q": Q, "z": Z, "sqrt": sp.sqrt}
    )
    if record["exact_symbolic_hessian_sha256"] != _sha(sp.srepr(sp.hessian(expression, (X, Q, Z)))):
        raise ValueError("formal evidence symbolic Hessian mismatch")
    algebraic = record["algebraic_gates"]
    expected_gate_names = {
        "finite_origin",
        "vacuum_zero",
        "high_field_newtonian_limit",
        "new_spatial_state_information",
        "universal_constant_cap",
        "derivative_order",
        "one_metric_no_private_lensing_law",
    }
    if not isinstance(algebraic, list) or len(algebraic) != len(expected_gate_names):
        raise ValueError("formal evidence algebraic gate set mismatch")
    gate_names = [gate.get("name") if isinstance(gate, Mapping) else None for gate in algebraic]
    if len(set(gate_names)) != len(gate_names) or set(gate_names) != expected_gate_names:
        raise ValueError("formal evidence algebraic gate set mismatch")
    gate_statuses = [_validate_local_gate(gate) for gate in algebraic]
    sampled_status = _validate_local_gate(
        record["sampled_static_convexity"], expected_name="sampled_static_convexity"
    )
    local_pass = all(status == "pass" for status in (*gate_statuses, sampled_status))
    if record["local_formula_preflight_pass"] is not local_pass:
        raise ValueError("formal evidence local preflight decision mismatch")
    mapping_decision = payload["decision"]
    health = record["semantic_action_health"]
    if mapping_decision == "mapped":
        if (
            record["covariant_action_compiled"] is not True
            or not _is_sha256(record["covariant_action_spec_sha256"])
            or record["covariant_action_spec_sha256"] != _sha(payload["action_spec"])
            or not _is_sha256(record["covariant_action_provenance_sha256"])
            or record["covariant_action_provenance_sha256"]
            != payload["covariant_action_provenance"]["provenance_binding_sha256"]
            or not isinstance(health, Mapping)
        ):
            raise ValueError("formal evidence mapped candidate binding mismatch")
        _validate_semantic_health(health)
        cache = health["cache_binding"]
        expected_spec_file_sha256 = hashlib.sha256(
            (json.dumps(payload["action_spec"], indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest()
        if (
            health["content_sha256"] != record["semantic_action_health_sha256"]
            or cache["candidate_id"] != record["candidate_id"]
            or cache["action_spec_payload_sha256"] != record["covariant_action_spec_sha256"]
            or cache["action_spec_file_sha256"] != expected_spec_file_sha256
            or cache["covariant_mapping_payload_sha256"]
            != record["covariant_mapping_payload_sha256"]
            or cache["input_action_sha256"]
            != payload["covariant_action_provenance"]["input_action_sha256"]
        ):
            raise ValueError("formal evidence health binding mismatch")
    elif (
        record["covariant_action_compiled"] is not False
        or record["covariant_action_spec_sha256"] is not None
        or record["covariant_action_provenance_sha256"] is not None
        or health is not None
        or record["semantic_action_health_sha256"] is not None
    ):
        raise ValueError("formal evidence unmapped candidate binding mismatch")
    derived_decision, derived_blocker = _candidate_decision(payload, health)
    if record["decision"] != derived_decision or record["first_blocker"] != derived_blocker:
        raise ValueError("formal evidence candidate decision mismatch")
    if (
        derived_decision == "reject"
        and mapping_decision == "mapped"
        and (health["status"] != "reject" or "reject" not in health["gate_statuses"].values())
    ):
        raise ValueError("formal evidence candidate rejection lacks evidence")
    return local_pass, mapping_decision == "mapped", derived_decision == "reject"


def validate_formal_evidence(value: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    expected = {
        "schema_version",
        "backend_id",
        "candidate_root_sha256",
        "generated_receipt_sha256",
        "candidate_manifest_sha256",
        "source_pass_candidate_count",
        "candidate_manifest_sample_complete",
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
    records = value.get("candidate_records")
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
        or not isinstance(value.get("candidate_manifest_sample_complete"), bool)
        or value.get("decision") not in {"block", "reject"}
        or value.get("complete_comparable_evidence") is not False
        or value.get("theory_rejected") is not False
        or value.get("direct_rank_assignment") is not False
        or value.get("observations_opened") is not False
        or value.get("forbidden_target_inputs_opened") is not False
        or any(
            not _is_sha256(value.get(key))
            for key in (
                "candidate_root_sha256",
                "generated_receipt_sha256",
                "candidate_manifest_sha256",
                "candidate_records_root_sha256",
            )
        )
        or any(not _is_nonnegative_int(value.get(key)) for key in numeric_counts)
        or not isinstance(records, list)
        or len(records) > 32
        or value.get("candidate_records_root_sha256") != _sha(records)
        or value.get("formally_checked_candidate_count") != len(records)
        or value.get("source_pass_candidate_count") < len(records)
        or value["candidate_manifest_sample_complete"]
        and value.get("source_pass_candidate_count") != len(records)
    ):
        raise ValueError("formal evidence contract mismatch")
    derived = [
        _validate_candidate_record(record, value["candidate_manifest_sha256"]) for record in records
    ]
    identities = [record["candidate_id"] for record in records]
    ordinals = [record["ordinal"] for record in records]
    if len(identities) != len(set(identities)) or ordinals != sorted(set(ordinals)):
        raise ValueError("formal evidence candidate identity mismatch")
    local_count = sum(int(item[0]) for item in derived)
    mapped_count = sum(int(item[1]) for item in derived)
    reject_count = sum(int(item[2]) for item in derived)
    if (
        value["symbolic_local_preflight_pass_count"] != local_count
        or value["covariant_action_mapped_count"] != mapped_count
        or value["action_health_execution_count"] != mapped_count
        or value["hard_reject_count"] != reject_count
        or value["candidate_hard_reject_count"] != reject_count
    ):
        raise ValueError("formal evidence derived count mismatch")
    sample_complete = value["candidate_manifest_sample_complete"]
    derived_decision = (
        "reject" if records and reject_count == len(records) and sample_complete else "block"
    )
    derived_blocker = (
        "candidate_manifest_is_bounded_not_complete"
        if not sample_complete
        else (
            "all_formal_candidates_hard_rejected"
            if derived_decision == "reject"
            else "complete_comparable_candidate_evidence_not_registered"
        )
    )
    if value["decision"] != derived_decision or value.get("first_blocker") != derived_blocker:
        raise ValueError("formal evidence aggregate decision mismatch")
