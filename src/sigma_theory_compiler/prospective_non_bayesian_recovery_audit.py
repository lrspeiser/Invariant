"""Retrospective, target-blind repair audit for non-Bayesian tournament families."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    ProvenanceRecord,
    SchemaViolation,
    canonical_json_bytes,
    canonical_sha256,
)

CONFIG_SCHEMA = "sigma-prospective-non-bayesian-recovery-config-1.0"
RESULT_SCHEMA = "sigma-prospective-non-bayesian-recovery-result-1.0"
CAMPAIGN_ID = "prospective-non-bayesian-recovery-audit-001"
CONFIG_PATH = "configs/prospective_non_bayesian_recovery_audit.json"
SOURCE_PATH = "src/sigma_theory_compiler/prospective_non_bayesian_recovery_audit.py"
TEST_PATH = "tests/test_prospective_non_bayesian_recovery_audit.py"
OUTPUT_PATH = "runs/math/prospective-non-bayesian-recovery-audit/campaign.json"
FAMILIES = ("cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
WORLDS = ("prospective.graph_parity", "prospective.modular_affine")
REPAIR_ID = "sha256_counter_inventory_breadth_v1"
SCOPE = (
    "Retrospective audit of one preregistered, target-blind generic inventory-breadth repair "
    "against two already-unsealed tournament PASS worlds. Results do not establish prospective "
    "success, tuning-free discovery, truth, novelty, proof, or promotion eligibility."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ROLES = {
    "ablation_artifact",
    "ablation_config",
    "ablation_source",
    "ablation_test",
    "tournament_artifact",
    "tournament_config",
    "tournament_source",
    "tournament_test",
}
_CONFIG_KEYS = {
    "budgets",
    "campaign_id",
    "excluded_families",
    "frozen_worlds",
    "policies",
    "repair_contract",
    "repair_families",
    "schema_version",
    "snapshot_sha256",
    "source_bindings",
}
_BINDING_KEYS = {"content_sha256", "file_sha256", "path"}
_FROZEN_WORLD_KEYS = {
    "candidates",
    "public_seed",
    "public_world_sha256",
    "sealed_target_sha256",
    "world_id",
}
_FROZEN_CANDIDATE_KEYS = {"artifact_id", "content_sha256", "family", "hypothesis"}
_DIAGNOSES = {
    "cross_domain": "single_transferred_construction_was_hash_projected_to_one_inventory_member",
    "egraph": "single_extracted_eclass_was_hash_projected_to_one_inventory_member",
    "evolutionary": "single_selected_lineage_artifact_was_hash_projected_to_one_inventory_member",
    "grammar": "single_closed_grammar_expression_was_hash_projected_to_one_inventory_member",
    "llm": "single_quarantined_offline_proposal_was_hash_projected_to_one_inventory_member",
    "symbolic": "single_exact_symbolic_specialization_was_hash_projected_to_one_inventory_member",
}
_POLICIES = {
    "new_target_access": "forbidden_frozen_receipt_only",
    "post_design_tuning": "forbidden",
    "network_access": "forbidden",
    "live_sqlite_access": "forbidden",
    "runtime_process_control": "forbidden",
    "bayesian_repair": "excluded",
    "repair_establishes_prospective_success": False,
    "repair_establishes_truth": False,
    "repair_establishes_novelty": False,
    "repair_authorizes_promotion": False,
}


class NonBayesianRecoveryError(ValueError):
    """The recovery audit or its frozen provenance failed closed."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NonBayesianRecoveryError(f"{label} keys changed")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise NonBayesianRecoveryError("path must be portable and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise NonBayesianRecoveryError("path escapes project root") from error
    return path


def _load_json(path: Path, *, maximum_bytes: int = 1_000_000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        raise NonBayesianRecoveryError(f"JSON source missing or oversized: {path.name}")

    def reject_float(value: str) -> float:
        raise NonBayesianRecoveryError(f"floating JSON number is forbidden: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=reject_float)
        detached = json.loads(canonical_json_bytes(value))
    except NonBayesianRecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaViolation, TypeError) as error:
        raise NonBayesianRecoveryError(f"cannot load exact JSON: {path.name}") from error
    if not isinstance(detached, dict):
        raise NonBayesianRecoveryError("JSON root must be an object")
    return detached


def _sealed_content(value: Mapping[str, Any], label: str) -> str:
    content = value.get("content_sha256")
    if not isinstance(content, str) or _SHA256.fullmatch(content) is None:
        raise NonBayesianRecoveryError(f"{label} content hash changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if canonical_sha256(body) != content:
        raise NonBayesianRecoveryError(f"{label} content seal changed")
    return content


def _snapshot_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "source_bindings",
            "excluded_families",
            "repair_families",
            "repair_contract",
            "frozen_worlds",
        )
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(config, _CONFIG_KEYS, "config")
    if config["schema_version"] != CONFIG_SCHEMA or config["campaign_id"] != CAMPAIGN_ID:
        raise NonBayesianRecoveryError("config identity changed")
    _exact_keys(config["source_bindings"], _SOURCE_ROLES, "source bindings")
    for role, descriptor in config["source_bindings"].items():
        _exact_keys(descriptor, _BINDING_KEYS, f"binding {role}")
        if not isinstance(descriptor["path"], str) or not descriptor["path"]:
            raise NonBayesianRecoveryError(f"binding path changed: {role}")
        if _SHA256.fullmatch(str(descriptor["file_sha256"])) is None:
            raise NonBayesianRecoveryError(f"binding file hash changed: {role}")
        content = descriptor["content_sha256"]
        if content is not None and _SHA256.fullmatch(str(content)) is None:
            raise NonBayesianRecoveryError(f"binding content hash changed: {role}")
    if config["excluded_families"] != ["bayesian"]:
        raise NonBayesianRecoveryError("Bayesian exclusion changed")
    if config["repair_families"] != {family: REPAIR_ID for family in FAMILIES}:
        raise NonBayesianRecoveryError("repair family registry changed")
    if config["repair_contract"] != {
        "repair_id": REPAIR_ID,
        "input_fields": [
            "family",
            "frozen_candidate_ref",
            "ordinal",
            "public_world_sha256",
            "repair_id",
        ],
        "target_fields": [],
        "ordinals": [0, 1, 2, 3],
        "hypothesis_inventory_size": 11,
        "digest_prefix_hex_chars": 16,
    }:
        raise NonBayesianRecoveryError("repair contract changed")
    worlds = config["frozen_worlds"]
    if not isinstance(worlds, list) or [item.get("world_id") for item in worlds] != list(WORLDS):
        raise NonBayesianRecoveryError("frozen world inventory changed")
    for world in worlds:
        _exact_keys(world, _FROZEN_WORLD_KEYS, "frozen world")
        if (
            _SHA256.fullmatch(str(world["public_world_sha256"])) is None
            or _SHA256.fullmatch(str(world["sealed_target_sha256"])) is None
            or not isinstance(world["public_seed"], str)
            or not world["public_seed"]
        ):
            raise NonBayesianRecoveryError("frozen world binding changed")
        candidates = world["candidates"]
        if not isinstance(candidates, list) or [item.get("family") for item in candidates] != [
            "bayesian",
            *FAMILIES,
        ]:
            raise NonBayesianRecoveryError("frozen candidate family inventory changed")
        for candidate in candidates:
            _exact_keys(candidate, _FROZEN_CANDIDATE_KEYS, "frozen candidate")
            if (
                not isinstance(candidate["artifact_id"], str)
                or not candidate["artifact_id"].startswith("sig-")
                or _SHA256.fullmatch(str(candidate["content_sha256"])) is None
                or isinstance(candidate["hypothesis"], bool)
                or not isinstance(candidate["hypothesis"], int)
                or not 0 <= candidate["hypothesis"] < 11
            ):
                raise NonBayesianRecoveryError("frozen candidate identity changed")
    if config["budgets"] != {
        "maximum_repaired_candidates_per_family_world": 4,
        "maximum_repaired_candidates_total": 48,
        "maximum_gate_checks": 96,
        "required_pass_worlds": 2,
    }:
        raise NonBayesianRecoveryError("repair budgets changed")
    if config["policies"] != _POLICIES:
        raise NonBayesianRecoveryError("policy boundary changed")
    if config["snapshot_sha256"] != canonical_sha256(_snapshot_body(config)):
        raise NonBayesianRecoveryError("snapshot seal changed")


def _load_bound_sources(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    values: dict[str, dict[str, Any]] = {}
    for role, descriptor in sorted(config["source_bindings"].items()):
        path = _resolve(root, descriptor["path"])
        if not path.is_file() or _file_sha256(path) != descriptor["file_sha256"]:
            raise NonBayesianRecoveryError(f"bound source changed: {role}")
        paths[role] = path
        if path.suffix == ".json":
            value = _load_json(path)
            values[role] = value
            if (
                descriptor["content_sha256"] is not None
                and _sealed_content(value, role) != descriptor["content_sha256"]
            ):
                raise NonBayesianRecoveryError(f"bound content changed: {role}")
    return paths, values


def _validate_ablation(ablation: Mapping[str, Any], tournament_binding: Mapping[str, Any]) -> None:
    _sealed_content(ablation, "ablation")
    if (
        ablation.get("decision") != "pass_eight_seed_exact_replay_and_seven_family_ablation"
        or ablation.get("counts", {}).get("world_pass_to_reject_ablation_changes") != 2
        or ablation.get("claims", {}).get("stability_establishes_truth") is not False
        or ablation.get("tournament_binding")
        != {
            "path": tournament_binding["path"],
            "file_sha256": tournament_binding["file_sha256"],
            "content_sha256": tournament_binding["content_sha256"],
        }
    ):
        raise NonBayesianRecoveryError("ablation premise changed")
    records = ablation.get("ablations")
    if not isinstance(records, list):
        raise NonBayesianRecoveryError("ablation records changed")
    bayesian = [item for item in records if item.get("removed_family") == "bayesian"]
    if len(bayesian) != 1 or bayesian[0].get("world_pass_to_reject_count") != 2:
        raise NonBayesianRecoveryError("Bayesian dependence premise changed")


def _frozen_candidate_index(
    config_world: Mapping[str, Any], tournament_world: Mapping[str, Any]
) -> dict[str, CandidateArtifact]:
    raw = tournament_world.get("candidates")
    if not isinstance(raw, list) or len(raw) != 7:
        raise NonBayesianRecoveryError("tournament candidate set changed")
    try:
        candidates = {
            item.representation["family"]: item
            for item in (CandidateArtifact.from_dict(value) for value in raw)
        }
    except (SchemaViolation, TypeError, ValueError, KeyError) as error:
        raise NonBayesianRecoveryError("tournament candidate replay failed") from error
    expected = {
        item["family"]: (
            item["artifact_id"],
            item["content_sha256"],
            item["hypothesis"],
        )
        for item in config_world["candidates"]
    }
    actual = {
        family: (
            candidate.artifact_id,
            candidate.content_sha256,
            candidate.representation.get("hypothesis"),
        )
        for family, candidate in candidates.items()
    }
    if actual != expected:
        raise NonBayesianRecoveryError("frozen tournament candidate identities changed")
    return candidates


def _original_diagnoses(
    tournament_world: Mapping[str, Any], candidates: Mapping[str, CandidateArtifact]
) -> list[dict[str, Any]]:
    evaluations = tournament_world.get("evaluations")
    if not isinstance(evaluations, Mapping):
        raise NonBayesianRecoveryError("original evaluation map changed")
    diagnoses: list[dict[str, Any]] = []
    for family in FAMILIES:
        evaluation = evaluations.get(family)
        if not isinstance(evaluation, Mapping):
            raise NonBayesianRecoveryError(f"missing original evaluation: {family}")
        gates = evaluation.get("gate_outcomes")
        if not isinstance(gates, list):
            raise NonBayesianRecoveryError("original gate outcomes changed")
        statuses = {item.get("gate_id"): item.get("status") for item in gates}
        if statuses != {"hard_exact": "pass", "hard_holdout": "reject"}:
            raise NonBayesianRecoveryError("original non-Bayesian gate diagnosis changed")
        diagnoses.append(
            {
                "family": family,
                "candidate": candidates[family].ref.to_dict(),
                "candidate_hypothesis": candidates[family].representation["hypothesis"],
                "hard_exact": "pass",
                "hard_holdout": "reject",
                "diagnosis": _DIAGNOSES[family],
                "diagnosis_scope": (
                    "The frozen single candidate missed the exact holdout token. This diagnoses "
                    "bounded inventory coverage, not an incapacity of the generator family."
                ),
            }
        )
    return diagnoses


def _repair_candidate(
    original: CandidateArtifact,
    *,
    family: str,
    public_world_sha256: str,
    ordinal: int,
) -> CandidateArtifact:
    repair_input = {
        "family": family,
        "frozen_candidate_ref": original.ref.to_dict(),
        "ordinal": ordinal,
        "public_world_sha256": public_world_sha256,
        "repair_id": REPAIR_ID,
    }
    digest = canonical_sha256(repair_input)
    hypothesis = int(digest[:16], 16) % 11
    return CandidateArtifact.create(
        ArtifactKind.CONJECTURE,
        f"Target-blind repaired {family} inventory candidate {ordinal}.",
        {
            "family": family,
            "frozen_candidate": original.ref.to_dict(),
            "hypothesis": hypothesis,
            "hypothesis_inventory_size": 11,
            "ordinal": ordinal,
            "public_world_sha256": public_world_sha256,
            "repair_id": REPAIR_ID,
            "repair_input_sha256": digest,
            "target_fields_read": [],
        },
        ProvenanceRecord.create(
            original.provenance.domain_pack,
            repair_input,
            inputs=(original.ref,),
        ),
        assumptions=(
            "frozen tournament candidate and public-world digest only",
            "retrospective evaluation occurs only after target-blind repair generation",
        ),
        claims=("requires_frozen_holdout_replay",),
    )


def _build_phase_a(
    config: Mapping[str, Any], tournament_worlds: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[CandidateArtifact, ...]]]:
    repaired: dict[tuple[str, str], tuple[CandidateArtifact, ...]] = {}
    worlds: list[dict[str, Any]] = []
    for config_world in config["frozen_worlds"]:
        world_id = config_world["world_id"]
        tournament_world = tournament_worlds[world_id]
        candidates = _frozen_candidate_index(config_world, tournament_world)
        public = {
            "world_id": world_id,
            "public_seed": config_world["public_seed"],
            "hypothesis_inventory": list(range(11)),
            "target_disclosed": False,
        }
        if canonical_sha256(public) != config_world["public_world_sha256"]:
            raise NonBayesianRecoveryError("public-world replay changed")
        family_rows: list[dict[str, Any]] = []
        for family in FAMILIES:
            family_candidates = tuple(
                _repair_candidate(
                    candidates[family],
                    family=family,
                    public_world_sha256=config_world["public_world_sha256"],
                    ordinal=ordinal,
                )
                for ordinal in config["repair_contract"]["ordinals"]
            )
            repaired[(world_id, family)] = family_candidates
            family_rows.append(
                {
                    "family": family,
                    "repair_id": REPAIR_ID,
                    "frozen_candidate": candidates[family].ref.to_dict(),
                    "repair_candidates": [item.to_dict() for item in family_candidates],
                    "candidate_count": len(family_candidates),
                    "unique_hypothesis_count": len(
                        {item.representation["hypothesis"] for item in family_candidates}
                    ),
                    "target_fields_read": [],
                }
            )
        worlds.append(
            {
                "world_id": world_id,
                "public_world_sha256": config_world["public_world_sha256"],
                "families": family_rows,
            }
        )
    body = {
        "schema_version": "sigma-prospective-non-bayesian-repair-generation-1.0",
        "repair_id": REPAIR_ID,
        "generation_inputs": list(config["repair_contract"]["input_fields"]),
        "target_fields_read": [],
        "bayesian_excluded": True,
        "worlds": worlds,
        "counts": {
            "worlds": len(worlds),
            "families": len(FAMILIES),
            "repair_candidates": sum(len(items) for items in repaired.values()),
        },
    }
    return {**body, "content_sha256": canonical_sha256(body)}, repaired


def _target_for_world(config_world: Mapping[str, Any], tournament_world: Mapping[str, Any]) -> int:
    target = tournament_world.get("unsealed_target")
    if (
        not isinstance(target, Mapping)
        or set(target) != {"hypothesis", "world_id"}
        or target["world_id"] != config_world["world_id"]
        or isinstance(target["hypothesis"], bool)
        or not isinstance(target["hypothesis"], int)
        or not 0 <= target["hypothesis"] < 11
        or canonical_sha256(target) != config_world["sealed_target_sha256"]
        or tournament_world.get("sealed_target_sha256") != config_world["sealed_target_sha256"]
    ):
        raise NonBayesianRecoveryError("frozen target binding changed")
    return int(target["hypothesis"])


def _evaluate_repair_family(candidates: Sequence[CandidateArtifact], target: int) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate.validate()
        representation = candidate.representation
        exact_pass = (
            set(representation)
            == {
                "family",
                "frozen_candidate",
                "hypothesis",
                "hypothesis_inventory_size",
                "ordinal",
                "public_world_sha256",
                "repair_id",
                "repair_input_sha256",
                "target_fields_read",
            }
            and representation["repair_id"] == REPAIR_ID
            and representation["target_fields_read"] == []
            and representation["hypothesis_inventory_size"] == 11
            and isinstance(representation["hypothesis"], int)
            and not isinstance(representation["hypothesis"], bool)
            and 0 <= representation["hypothesis"] < 11
        )
        if not exact_pass:
            status = "block"
            reason = "repaired_candidate_contract_ambiguous"
        elif representation["hypothesis"] == target:
            status = "pass"
            reason = None
        else:
            status = "reject"
            reason = "exact_frozen_holdout_mismatch"
        evaluations.append(
            {
                "candidate": candidate.ref.to_dict(),
                "hypothesis": representation["hypothesis"],
                "hard_exact": "pass" if exact_pass else "block",
                "hard_holdout": status if exact_pass else "block",
                "status": status,
                "reason": reason,
            }
        )
    statuses = Counter(item["status"] for item in evaluations)
    if statuses["block"]:
        decision = "block"
        first_blocker = "ambiguous_repaired_candidate_contract"
    elif statuses["pass"]:
        decision = "pass"
        first_blocker = None
    else:
        decision = "reject"
        first_blocker = "fixed_target_blind_repair_inventory_exhausted_without_match"
    return {
        "decision": decision,
        "first_blocker": first_blocker,
        "evaluations": evaluations,
        "counts": {
            "candidates": len(evaluations),
            "pass": statuses["pass"],
            "reject": statuses["reject"],
            "block": statuses["block"],
        },
    }


def _lane_bindings(root: Path, config_path: Path) -> dict[str, dict[str, str]]:
    paths = {
        "config": config_path,
        "source": _resolve(root, SOURCE_PATH),
        "test": _resolve(root, TEST_PATH),
    }
    return {
        role: {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "file_sha256": _file_sha256(path),
        }
        for role, path in sorted(paths.items())
    }


def build_recovery_audit(
    config_path: str | Path = CONFIG_PATH, *, root: str | Path = "."
) -> dict[str, Any]:
    """Build the frozen retrospective audit without network, SQLite, or process control."""

    project_root = Path(root).resolve()
    resolved_config = _resolve(project_root, Path(config_path).as_posix())
    config = _load_json(resolved_config)
    _validate_config(config)
    _, values = _load_bound_sources(config, project_root)
    tournament = values["tournament_artifact"]
    ablation = values["ablation_artifact"]
    _sealed_content(tournament, "tournament")
    if (
        tournament.get("decision") != "completed_preregistered_three_world_single_unseal_tournament"
        or tournament.get("claims", {}).get("post_unseal_tuning_performed") is not False
        or tournament.get("claims", {}).get("exactly_one_atomic_target_unseal_batch") is not True
    ):
        raise NonBayesianRecoveryError("tournament freeze boundary changed")
    _validate_ablation(ablation, config["source_bindings"]["tournament_artifact"])
    raw_worlds = tournament.get("world_results")
    if not isinstance(raw_worlds, list):
        raise NonBayesianRecoveryError("tournament worlds changed")
    tournament_worlds = {item.get("world_id"): item for item in raw_worlds}
    if set(WORLDS) - set(tournament_worlds):
        raise NonBayesianRecoveryError("frozen PASS worlds missing")
    for world_id in WORLDS:
        if tournament_worlds[world_id].get("decision") != (
            "pass_at_least_one_target_blind_candidate_survived"
        ):
            raise NonBayesianRecoveryError("frozen PASS world decision changed")

    phase_a, repaired = _build_phase_a(config, tournament_worlds)
    phase_a_bytes = canonical_json_bytes(phase_a)
    for config_world in config["frozen_worlds"]:
        if config_world["sealed_target_sha256"].encode() in phase_a_bytes:
            raise NonBayesianRecoveryError("sealed target leaked into repair generation")

    diagnoses: list[dict[str, Any]] = []
    world_results: list[dict[str, Any]] = []
    recovered_worlds: set[str] = set()
    recovered_families: set[str] = set()
    decision_counts: Counter[str] = Counter()
    for config_world in config["frozen_worlds"]:
        world_id = config_world["world_id"]
        tournament_world = tournament_worlds[world_id]
        candidates = _frozen_candidate_index(config_world, tournament_world)
        diagnoses.extend(
            {"world_id": world_id, **item}
            for item in _original_diagnoses(tournament_world, candidates)
        )
        target = _target_for_world(config_world, tournament_world)
        family_results: list[dict[str, Any]] = []
        for family in FAMILIES:
            result = _evaluate_repair_family(repaired[(world_id, family)], target)
            family_results.append({"family": family, **result})
            decision_counts[result["decision"]] += 1
            if result["decision"] == "pass":
                recovered_worlds.add(world_id)
                recovered_families.add(family)
        world_results.append(
            {
                "world_id": world_id,
                "sealed_target_sha256": config_world["sealed_target_sha256"],
                "target_replay_source": "frozen_tournament_receipt_only",
                "family_results": family_results,
                "recovered_by_non_bayesian_family": any(
                    item["decision"] == "pass" for item in family_results
                ),
            }
        )
    if decision_counts["block"]:
        decision = "block_ambiguous_repair_contract"
        first_blocker = "at_least_one_family_world_repair_contract_ambiguous"
    elif recovered_worlds:
        decision = "pass_at_least_one_non_bayesian_family_recovered_a_frozen_pass_world"
        first_blocker = None
    else:
        decision = "reject_no_non_bayesian_recovery_within_fixed_repair_budget"
        first_blocker = "all_fixed_target_blind_repair_inventories_exhausted_without_match"
    total_candidates = sum(len(items) for items in repaired.values())
    total_gate_checks = total_candidates * 2
    if (
        total_candidates > config["budgets"]["maximum_repaired_candidates_total"]
        or total_gate_checks > config["budgets"]["maximum_gate_checks"]
    ):
        raise NonBayesianRecoveryError("repair accounting exceeded preregistered budget")
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "scope": SCOPE,
        "decision": decision,
        "first_blocker": first_blocker,
        "source_bindings": {
            "lane": _lane_bindings(project_root, resolved_config),
            "frozen_inputs": config["source_bindings"],
        },
        "snapshot_sha256": config["snapshot_sha256"],
        "preregistration": {
            "repair_contract": config["repair_contract"],
            "repair_families": config["repair_families"],
            "excluded_families": config["excluded_families"],
            "budgets": config["budgets"],
            "policies": config["policies"],
            "timing_scope": "registered_after_original_target_unseal_before_this_repair_replay",
        },
        "original_failure_diagnoses": diagnoses,
        "target_blind_generation": phase_a,
        "world_results": world_results,
        "counts": {
            "worlds": len(WORLDS),
            "non_bayesian_families": len(FAMILIES),
            "original_failed_family_worlds": len(WORLDS) * len(FAMILIES),
            "repair_candidates": total_candidates,
            "gate_checks": total_gate_checks,
            "family_world_passes": decision_counts["pass"],
            "family_world_rejects": decision_counts["reject"],
            "family_world_blocks": decision_counts["block"],
            "recovered_worlds": len(recovered_worlds),
            "recovering_families": len(recovered_families),
        },
        "recovered_world_ids": sorted(recovered_worlds),
        "recovering_families": sorted(recovered_families),
        "claims": {
            "bayesian_excluded": True,
            "all_repairs_generated_before_frozen_target_replay": True,
            "repair_generation_target_fields_read": [],
            "new_target_access_performed": False,
            "post_design_tuning_performed": False,
            "prospective_success_established": False,
            "truth_established": False,
            "novelty_established": False,
            "promotion_authorized": False,
        },
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_recovery_audit(
    result: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG_PATH,
    root: str | Path = ".",
) -> None:
    """Replay live frozen bindings and require exact receipt equality."""

    if not isinstance(result, Mapping):
        raise NonBayesianRecoveryError("audit result must be an object")
    expected = build_recovery_audit(config_path, root=root)
    try:
        actual_bytes = canonical_json_bytes(result)
        expected_bytes = canonical_json_bytes(expected)
    except (SchemaViolation, TypeError, ValueError) as error:
        raise NonBayesianRecoveryError("audit result is not exact JSON") from error
    if actual_bytes != expected_bytes:
        raise NonBayesianRecoveryError("audit result differs from exact live replay")


def write_recovery_audit(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = OUTPUT_PATH,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Build, validate, and write the canonical audit receipt."""

    project_root = Path(root).resolve()
    result = build_recovery_audit(config_path, root=project_root)
    validate_recovery_audit(result, config_path=config_path, root=project_root)
    output = _resolve(project_root, Path(output_path).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result) + b"\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    result = write_recovery_audit(args.config, args.output, root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_ID",
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "RESULT_SCHEMA",
    "SOURCE_PATH",
    "TEST_PATH",
    "NonBayesianRecoveryError",
    "build_recovery_audit",
    "validate_recovery_audit",
    "write_recovery_audit",
]
