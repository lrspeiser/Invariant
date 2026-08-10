from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .adm_dirac_promotion_evaluator import KNOWN_ANSWER_BUNDLE, bundle_binding
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-covariant-grammar-v3-seed-manifest-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def evaluate_pre_generation_constraints(
    seed: dict[str, Any], constraints: dict[str, Any]
) -> dict[str, Any]:
    invariants = set(seed.get("invariants", []))
    reasons: list[str] = []
    forbidden = invariants & set(constraints["forbidden_invariants"])
    if forbidden:
        reasons.append("forbidden_invariants:" + ",".join(sorted(forbidden)))
    if (
        seed.get("theory_contract") == "generic_unit_vector"
        and "Q_a_u" in invariants
    ):
        reasons.append("generic_unit_vector_q_operator_excluded_by_70_reject_taxonomy")
    if seed.get("family_id") in constraints["forbidden_family_ids"]:
        reasons.append("structurally_doomed_family_id")
    extra_fields = set(seed.get("field_ids", [])) - {"g_mu_nu"}
    if len(extra_fields) > int(constraints["maximum_extra_dynamical_fields"]):
        reasons.append("too_many_extra_dynamical_fields")
    if seed.get("enabled_for_generation"):
        adapters = seed.get("formal_adapters", [])
        if not adapters or not any(item.get("available") is True for item in adapters):
            reasons.append("enabled_seed_has_no_available_formal_adapter")
        if seed.get("hard_blockers"):
            reasons.append("enabled_seed_has_hard_blockers")
    body = {
        "decision": "reject" if reasons else "accept",
        "family_id": seed["family_id"],
        "reasons": sorted(reasons),
        "seed_lineage_sha256": _sha(seed),
        "data_eligibility": dict(ELIGIBILITY),
    }
    return _content(body)


def _load_hash_bound_json(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"artifact file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_content = descriptor.get("content_sha256")
    if expected_content is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        hashed = value.get("canonical") if "canonical" in value else body
        if value.get("content_sha256") != expected_content or _sha(hashed) != expected_content:
            raise ValueError(f"artifact content hash mismatch: {descriptor['path']}")
    return value


def _compile_control(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    action = _load_hash_bound_json(root, descriptor["action_artifact"])
    health = _load_hash_bound_json(root, descriptor["health_artifact"])
    action_hash = action["content_sha256"]
    if action_hash != descriptor["input_action_sha256"]:
        raise ValueError("known-answer action hash mismatch")
    if health.get("input_action_sha256") != action_hash:
        raise ValueError("known-answer health artifact is not action-hash-bound")
    if health.get("family") != descriptor["expected_health_family"]:
        raise ValueError("known-answer health family mismatch")
    gate_counts = Counter(gate["status"] for gate in health["gates"].values())
    if dict(sorted(gate_counts.items())) != descriptor["expected_gate_counts"]:
        raise ValueError("known-answer gate count mismatch")
    record = {
        "control_id": descriptor["control_id"],
        "classification": descriptor["classification"],
        "input_action_sha256": action_hash,
        "action_file_sha256": descriptor["action_artifact"]["file_sha256"],
        "health_file_sha256": descriptor["health_artifact"]["file_sha256"],
        "health_family": health["family"],
        "gate_counts": dict(sorted(gate_counts.items())),
        "unresolved_or_rejected_gates": {
            name: gate["status"]
            for name, gate in sorted(health["gates"].items())
            if gate["status"] != "pass"
        },
        "eligible_as_generated_candidate": False,
        "role": "hash_bound_known_answer_control",
    }
    if descriptor.get("promotion_bundle_binding_sha256") is not None:
        actual_binding = bundle_binding(KNOWN_ANSWER_BUNDLE)
        if actual_binding != descriptor["promotion_bundle_binding_sha256"]:
            raise ValueError("Einstein-Aether promotion bundle binding mismatch")
        if KNOWN_ANSWER_BUNDLE["input_action_sha256"] != action_hash:
            raise ValueError("promotion bundle is not bound to the control action")
        record["promotion_bundle_binding_sha256"] = actual_binding
    return {**record, "control_lineage_sha256": _sha(record)}


def _validate_predecessors(root: Path, descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for descriptor in descriptors:
        value = _load_hash_bound_json(root, descriptor)
        for field, expected in descriptor["expected_fields"].items():
            if value.get(field) != expected:
                raise ValueError(f"predecessor field mismatch: {descriptor['path']}:{field}")
        records.append(
            {
                "path": descriptor["path"],
                "file_sha256": descriptor["file_sha256"],
                "content_sha256": descriptor["content_sha256"],
                "expected_fields": descriptor["expected_fields"],
            }
        )
    return records


def _validate_negative_source(
    root: Path, control: dict[str, Any], predecessor_by_path: dict[str, dict[str, Any]]
) -> None:
    descriptor = predecessor_by_path[control["source_artifact_path"]]
    artifact = _load_hash_bound_json(root, descriptor)
    sample = next(
        (
            item
            for item in artifact.get("sample_rejections", [])
            if item.get("candidate_id") == control["candidate_id"]
        ),
        None,
    )
    if sample is None:
        raise ValueError("negative control is absent from its bound predecessor sample")
    for field in (
        "ordinal",
        "input_action_sha256",
        "provenance_binding_sha256",
    ):
        if sample.get(field) != control[field]:
            raise ValueError(f"negative control source mismatch: {field}")


def _validate_adapter_entrypoint(adapter: dict[str, Any]) -> None:
    if adapter.get("available") is not True:
        return
    module_name, separator, attribute = str(adapter.get("entrypoint", "")).partition(":")
    if not separator:
        raise ValueError("available formal adapter lacks module:function entrypoint")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"formal adapter entrypoint is not callable: {adapter['entrypoint']}")


def build_covariant_grammar_v3_seed_manifest(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 seed config violates fail-closed eligibility")
    predecessors = _validate_predecessors(root, config["predecessor_artifacts"])
    predecessor_by_path = {
        descriptor["path"]: descriptor for descriptor in config["predecessor_artifacts"]
    }
    controls = [_compile_control(root, item) for item in config["known_answer_controls"]]

    family_records = []
    concrete_seeds = []
    for family in config["typed_family_seeds"]:
        for adapter in family.get("formal_adapters", []):
            _validate_adapter_entrypoint(adapter)
        constraint_result = evaluate_pre_generation_constraints(
            family, config["pre_generation_constraints"]
        )
        if family["enabled_for_generation"] and constraint_result["decision"] != "accept":
            raise ValueError(f"enabled seed family rejected: {family['family_id']}")
        lineage = _sha(family)
        family_records.append(
            {
                **family,
                "pre_generation_decision": constraint_result["decision"],
                "pre_generation_reasons": constraint_result["reasons"],
                "family_lineage_sha256": lineage,
            }
        )
        if family["enabled_for_generation"]:
            for parameter_index, parameters in enumerate(family["parameter_points"]):
                seed_body = {
                    "schema_version": "sigma-covariant-grammar-v3-concrete-seed-1.0",
                    "family_id": family["family_id"],
                    "family_lineage_sha256": lineage,
                    "parameter_index": parameter_index,
                    "parameters": parameters,
                    "operator_atoms": family["operator_atoms"],
                    "theory_contract": family["theory_contract"],
                    "data_eligibility": dict(ELIGIBILITY),
                }
                concrete_seeds.append(
                    {
                        **seed_body,
                        "seed_id": "G3-" + _sha(seed_body)[:24],
                        "seed_lineage_sha256": _sha(seed_body),
                    }
                )

    negative_results = []
    for control in config["negative_controls"]:
        if control.get("source_artifact_path") is not None:
            _validate_negative_source(root, control, predecessor_by_path)
        result = evaluate_pre_generation_constraints(
            control["seed_spec"], config["pre_generation_constraints"]
        )
        if result["decision"] != "reject":
            raise ValueError("legacy negative control re-entered grammar-v3")
        negative_results.append(
            {
                "control_id": control["control_id"],
                "candidate_id": control.get("candidate_id"),
                "input_action_sha256": control.get("input_action_sha256"),
                "source_artifact_path": control.get("source_artifact_path"),
                "decision": result["decision"],
                "reasons": result["reasons"],
                "constraint_evidence_sha256": result["content_sha256"],
            }
        )

    concrete_seeds.sort(key=lambda item: item["seed_id"])
    control_counts = Counter(item["classification"] for item in controls)
    family_counts = Counter(
        "enabled" if item["enabled_for_generation"] else "disabled"
        for item in family_records
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "predecessor_artifacts": predecessors,
        "rejection_taxonomy": {
            "production_candidate_count": 70,
            "formal_reject_count": 70,
            "formal_pass_count": 0,
            "remaining_blocked_count": 0,
            "generation_rule": (
                "Q_a_u is excluded from the generic unit-vector generator; old F(X_a_u,Q_a_u) "
                "shapes cannot be emitted by grammar-v3"
            ),
        },
        "pre_generation_constraints": config["pre_generation_constraints"],
        "known_answer_controls": controls,
        "known_answer_control_counts": dict(sorted(control_counts.items())),
        "typed_family_seeds": family_records,
        "typed_family_counts": dict(sorted(family_counts.items())),
        "negative_controls": negative_results,
        "negative_control_counts": {"reject": len(negative_results)},
        "scalable_generator_hook": {
            "callable": (
                "sigma_theory_compiler.covariant_grammar_v3_seed_campaign:"
                "iter_scalable_seed_specs"
            ),
            "concrete_seed_count": len(concrete_seeds),
            "concrete_seeds": concrete_seeds,
            "next_step": (
                "stream these bounded typed seeds into candidate-specific action compilation, "
                "then require the declared family adapters before promotion"
            ),
        },
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Known-answer controls calibrate the pipeline and are never candidate discoveries. "
            "New typed families are seed queues with explicit adapters or blockers, not claims "
            "of viability."
        ),
    }
    return _content(body)


def iter_scalable_seed_specs(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported grammar-v3 seed manifest schema")
    if manifest.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("seed manifest eligibility is not fail-closed")
    yield from manifest["scalable_generator_hook"]["concrete_seeds"]
