"""Prospective test of the frozen non-Bayesian inventory-breadth repair."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .candidate_evaluation_ladder import evaluate_candidate, validate_evaluation_replay
from .candidate_pareto_explanations import (
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_replay,
)
from .prospective_blind_cross_generator_tournament import (
    _descriptor,
    _holdout,
    _ladder,
    _metrics,
    _native_cross_domain,
    _native_egraph,
    _native_evolutionary,
    _native_grammar,
    _native_llm,
    _native_symbolic,
)
from .prospective_non_bayesian_recovery_audit import validate_recovery_audit
from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    StageDefinition,
    StageOutcome,
    canonical_json_bytes,
    canonical_sha256,
)

CONFIG_SCHEMA = "sigma-prospective-repaired-non-bayesian-config-1.0"
RESULT_SCHEMA = "sigma-prospective-repaired-non-bayesian-result-1.0"
TARGET_SCHEMA = "sigma-prospective-repaired-target-fixture-1.0"
CAMPAIGN_ID = "prospective-repaired-non-bayesian-tournament-001"
CONFIG_PATH = "configs/prospective_repaired_non_bayesian_tournament.json"
TARGET_PATH = "configs/prospective_repaired_non_bayesian_targets.json"
SOURCE_PATH = "src/sigma_theory_compiler/prospective_repaired_non_bayesian_tournament.py"
TEST_PATH = "tests/test_prospective_repaired_non_bayesian_tournament.py"
OUTPUT_PATH = "runs/math/prospective-repaired-non-bayesian-tournament/campaign.json"
REPAIR_ID = "sha256_counter_inventory_breadth_v1"
FAMILIES = ("cross_domain", "egraph", "evolutionary", "grammar", "llm", "symbolic")
WORLD_ROWS = (
    (
        "prospective_repair.modular_orbit",
        "repair-orbit-public-20260813",
        "ca217aabd9fc6b3d8aa2435328b33eac294df6527fda54ceee574bb90376c14e",
    ),
    (
        "prospective_repair.discrete_curvature",
        "repair-curvature-public-20260813",
        "eab4c21bd4d849a0c5c1c9e6abffff62c8a8d038e4258b5460ffac9c14355338",
    ),
    (
        "prospective_repair.graph_walk",
        "repair-walk-public-20260813",
        "604aca90b1ee81f1d0db62a7a9dcdf083e82f9346006e88c8762a460e5f0775d",
    ),
)
CLAIMS = {
    "bayesian_generation_excluded": True,
    "all_six_non_bayesian_native_families_exercised_per_world": True,
    "all_repaired_candidates_generated_before_target_fixture_read": True,
    "exactly_one_atomic_target_unseal_batch": True,
    "post_unseal_generation_performed": False,
    "post_unseal_tuning_performed": False,
    "retrospective_repair_generalizes_universally": False,
    "generator_output_establishes_truth": False,
    "pareto_rank_establishes_truth": False,
    "corpus_absence_establishes_novelty": False,
    "promotion_authorized": False,
}


class ProspectiveRepairError(ValueError):
    """The prospective repair campaign failed a closed boundary."""


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ProspectiveRepairError("path must be portable and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ProspectiveRepairError("path escapes project root") from error
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _deny_target_fixture_reads(root: Path) -> Any:
    """Deny owned Python file reads of the hidden fixture during generation."""

    target = _resolve(root, TARGET_PATH)
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open
    audit = {"attempted": 0, "denied": 0, "denied_content_bytes_exposed": 0}

    def is_target(path: Any) -> bool:
        if isinstance(path, int):
            return False
        try:
            return Path(path).resolve() == target
        except (OSError, TypeError, ValueError):
            return False

    def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if is_target(file):
            audit["attempted"] += 1
            audit["denied"] += 1
            raise PermissionError("target fixture denied before phase-A seal")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if is_target(file):
            audit["attempted"] += 1
            audit["denied"] += 1
            raise PermissionError("target fixture denied before phase-A seal")
        return original_io_open(file, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.resolve() == target:
            audit["attempted"] += 1
            audit["denied"] += 1
            raise PermissionError("target fixture denied before phase-A seal")
        return original_path_open(path, *args, **kwargs)

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    Path.open = guarded_path_open
    try:
        yield audit
    finally:
        Path.open = original_path_open
        io.open = original_io_open
        builtins.open = original_builtin_open


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ProspectiveRepairError(f"JSON source missing or oversized: {path.name}")

    def reject_float(value: str) -> float:
        raise ProspectiveRepairError(f"floating JSON value forbidden: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=reject_float)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProspectiveRepairError(f"cannot load JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ProspectiveRepairError("JSON root must be an object")
    return value


def _sealed_content(value: Mapping[str, Any], label: str) -> str:
    content = value.get("content_sha256")
    if not isinstance(content, str) or len(content) != 64:
        raise ProspectiveRepairError(f"{label} content seal missing")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if canonical_sha256(body) != content:
        raise ProspectiveRepairError(f"{label} content seal changed")
    return content


def _expected_config() -> dict[str, Any]:
    return {
        "budgets": {
            "candidates_per_family_per_world": 4,
            "generator_work_items_per_world": 128,
            "hypothesis_inventory_size": 11,
            "maximum_gate_checks": 144,
            "maximum_pareto_work_units_per_world": 4096,
        },
        "campaign_id": CAMPAIGN_ID,
        "generator_families": list(FAMILIES),
        "policies": {
            "bayesian_generation": "excluded",
            "generator_target_access": "forbidden",
            "network_access": "forbidden",
            "post_unseal_generation": "forbidden",
            "post_unseal_tuning": "forbidden",
            "target_records_per_unseal": 3,
            "target_unseal_batches": 1,
        },
        "repair_contract": {
            "digest_prefix_hex_chars": 16,
            "hypothesis_inventory_size": 11,
            "input_fields": [
                "family",
                "native_candidate_ref",
                "ordinal",
                "public_world_sha256",
                "repair_id",
            ],
            "ordinals": [0, 1, 2, 3],
            "repair_id": REPAIR_ID,
            "target_fields": [],
        },
        "retrospective_predecessor": {
            "content_sha256": "18bde0a4ef3a5a41055283adc923bf9214f07f754593dca50cd6f260c8679c46",
            "file_sha256": "aabfa842037a4c5f42b743f795063304090d8d5452684203d86814efabf54b52",
            "path": "runs/math/prospective-non-bayesian-recovery-audit/campaign.json",
        },
        "schema_version": CONFIG_SCHEMA,
        "target_fixture": {
            "content_sha256": "24b52db04b89ca8aa7ca345133c6f892d5738cd37f3c1710ec6e6ea56024814c",
            "path": TARGET_PATH,
        },
        "worlds": [
            {"world_id": world_id, "public_seed": seed, "sealed_target_sha256": seal}
            for world_id, seed, seal in WORLD_ROWS
        ],
    }


def _load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ProspectiveRepairError("config path changed")
    value = _load_json(path)
    if value != _expected_config():
        raise ProspectiveRepairError("preregistered config changed")
    predecessor = value["retrospective_predecessor"]
    predecessor_path = _resolve(root, predecessor["path"])
    if _file_sha256(predecessor_path) != predecessor["file_sha256"]:
        raise ProspectiveRepairError("retrospective predecessor file changed")
    predecessor_value = _load_json(predecessor_path)
    if (
        _sealed_content(predecessor_value, "retrospective predecessor")
        != predecessor["content_sha256"]
    ):
        raise ProspectiveRepairError("retrospective predecessor content changed")
    validate_recovery_audit(predecessor_value, root=root)
    return value


def _public_world(world: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "world_id": world["world_id"],
        "public_seed": world["public_seed"],
        "hypothesis_inventory": list(range(11)),
        "target_disclosed": False,
    }


def _generate_native(
    root: Path, world: Mapping[str, Any], descriptor: DomainPackDescriptor
) -> tuple[dict[str, CandidateArtifact], dict[str, Any]]:
    domain = descriptor.ref
    world_id = world["world_id"]
    seed = world["public_seed"]
    generated = {
        "cross_domain": _native_cross_domain(domain, world_id),
        "egraph": _native_egraph(domain, world_id),
        "evolutionary": _native_evolutionary(domain, world_id, seed),
        "grammar": _native_grammar(root, domain),
        "llm": _native_llm(root, domain, world_id, seed),
        "symbolic": _native_symbolic(domain),
    }
    candidates = {family: generated[family][0] for family in FAMILIES}
    if len({candidate.artifact_id for candidate in candidates.values()}) != len(FAMILIES):
        raise ProspectiveRepairError("native candidate identities collided")
    if any(candidate.provenance.domain_pack != domain for candidate in candidates.values()):
        raise ProspectiveRepairError("native candidate escaped world domain")
    receipts = {
        family: {
            "native_candidate": candidates[family].ref.to_dict(),
            "receipt_sha256": canonical_sha256(generated[family][1]),
        }
        for family in FAMILIES
    }
    return candidates, receipts


def _repair_candidate(
    original: CandidateArtifact,
    *,
    family: str,
    public_world_sha256: str,
    ordinal: int,
) -> CandidateArtifact:
    inputs = {
        "family": family,
        "native_candidate_ref": original.ref.to_dict(),
        "ordinal": ordinal,
        "public_world_sha256": public_world_sha256,
        "repair_id": REPAIR_ID,
    }
    digest = canonical_sha256(inputs)
    hypothesis = int(digest[:16], 16) % 11
    return CandidateArtifact.create(
        ArtifactKind.CONJECTURE,
        f"Prospective repaired {family} candidate {ordinal}.",
        {
            "family": family,
            "hypothesis": hypothesis,
            "hypothesis_inventory_size": 11,
            "native_candidate": original.ref.to_dict(),
            "ordinal": ordinal,
            "public_world_sha256": public_world_sha256,
            "repair_id": REPAIR_ID,
            "repair_input_sha256": digest,
            "target_fields_read": [],
        },
        ProvenanceRecord.create(original.provenance.domain_pack, inputs, inputs=(original.ref,)),
        assumptions=("public world and native candidate only", "target fixture unavailable"),
        claims=("requires_single_atomic_holdout_unseal",),
    )


def _build_phase_a(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, tuple[CandidateArtifact, ...]]]:
    prepared: list[dict[str, Any]] = []
    candidates_by_world: dict[str, tuple[CandidateArtifact, ...]] = {}
    for world in config["worlds"]:
        descriptor = _descriptor(world)
        public = _public_world(world)
        public_sha = canonical_sha256(public)
        native, receipts = _generate_native(root, world, descriptor)
        repaired = tuple(
            _repair_candidate(
                native[family], family=family, public_world_sha256=public_sha, ordinal=ordinal
            )
            for family in FAMILIES
            for ordinal in config["repair_contract"]["ordinals"]
        )
        if len(repaired) != 24 or len({item.artifact_id for item in repaired}) != 24:
            raise ProspectiveRepairError("repaired candidate inventory changed")
        candidates_by_world[world["world_id"]] = repaired
        prepared.append(
            {
                "world": dict(world),
                "descriptor": descriptor,
                "public_world": public,
                "public_world_sha256": public_sha,
                "native_receipts": receipts,
                "candidate_refs": [candidate.ref.to_dict() for candidate in repaired],
            }
        )
    return prepared, candidates_by_world


def _phase_a_receipt(
    prepared: Sequence[Mapping[str, Any]], access_audit: Mapping[str, int]
) -> dict[str, Any]:
    body = {
        "schema_version": "sigma-prospective-repaired-generation-phase-1.0",
        "repair_id": REPAIR_ID,
        "bayesian_excluded": True,
        "target_fixture_reads": 0,
        "target_fields_read": [],
        "pre_unseal_io": {
            "enforcement_scope": (
                "owned_single_threaded_python_builtins_io_and_pathlib_read_surfaces_not_os_sandbox"
            ),
            "attempted_target_reads": access_audit["attempted"],
            "denied_target_reads": access_audit["denied"],
            "denied_content_bytes_exposed": access_audit["denied_content_bytes_exposed"],
            "successful_target_reads": 0,
        },
        "worlds": [
            {
                "world_id": row["world"]["world_id"],
                "public_world_sha256": row["public_world_sha256"],
                "native_receipts": row["native_receipts"],
                "candidate_refs": row["candidate_refs"],
            }
            for row in prepared
        ],
        "counts": {"worlds": 3, "families_per_world": 6, "candidates": 72},
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _unseal_targets(
    root: Path, config: Mapping[str, Any], *, phase_a_sealed: bool
) -> dict[str, dict[str, Any]]:
    if not phase_a_sealed:
        raise ProspectiveRepairError("target fixture read attempted before phase-A seal")
    fixture_path = _resolve(root, config["target_fixture"]["path"])
    fixture = _load_json(fixture_path)
    if canonical_sha256(fixture) != config["target_fixture"]["content_sha256"]:
        raise ProspectiveRepairError("target fixture content changed")
    if set(fixture) != {"schema_version", "targets"} or fixture["schema_version"] != TARGET_SCHEMA:
        raise ProspectiveRepairError("target fixture schema changed")
    targets = fixture["targets"]
    if not isinstance(targets, list) or len(targets) != 3:
        raise ProspectiveRepairError("target fixture inventory changed")
    by_world: dict[str, dict[str, Any]] = {}
    configured = {world["world_id"]: world for world in config["worlds"]}
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"hypothesis", "target_seed", "world_id"}:
            raise ProspectiveRepairError("target record schema changed")
        world_id = target["world_id"]
        if world_id not in configured or world_id in by_world:
            raise ProspectiveRepairError("target world identity changed")
        expected = int(hashlib.sha256(target["target_seed"].encode()).hexdigest()[:16], 16) % 11
        if target["hypothesis"] != expected:
            raise ProspectiveRepairError("target seed derivation changed")
        if canonical_sha256(target) != configured[world_id]["sealed_target_sha256"]:
            raise ProspectiveRepairError("target commitment changed")
        by_world[world_id] = target
    if set(by_world) != set(configured):
        raise ProspectiveRepairError("atomic target batch incomplete")
    return by_world


class _RepairPack:
    def __init__(self, descriptor: DomainPackDescriptor, target: int) -> None:
        self._descriptor = descriptor
        self._target = target

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return self._descriptor

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        representation = artifact.representation
        valid = (
            set(representation)
            == {
                "family",
                "hypothesis",
                "hypothesis_inventory_size",
                "native_candidate",
                "ordinal",
                "public_world_sha256",
                "repair_id",
                "repair_input_sha256",
                "target_fields_read",
            }
            and representation["family"] in FAMILIES
            and representation["repair_id"] == REPAIR_ID
            and representation["target_fields_read"] == []
            and isinstance(representation["hypothesis"], int)
            and not isinstance(representation["hypothesis"], bool)
            and 0 <= representation["hypothesis"] < 11
        )
        check = CheckResult.create(
            f"{stage.stage_id}.closed_repair_contract",
            valid,
            {"artifact": artifact.artifact_id, "prior": sorted(prior_outcomes)},
        )
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            OutcomeStatus.PASS if valid else OutcomeStatus.REJECT,
            (check,),
            reason_codes=() if valid else ("invalid_repair_candidate",),
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        passed = (
            gate.gate_id == "hard_exact" or artifact.representation["hypothesis"] == self._target
        )
        check = CheckResult.create(
            f"{gate.gate_id}.prospective_repair_contract",
            passed,
            {
                "artifact": artifact.artifact_id,
                "candidate_hypothesis": artifact.representation["hypothesis"],
                "target_compared": gate.gate_id == "hard_holdout",
            },
        )
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS if passed else OutcomeStatus.REJECT,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (check,),
            reason_codes=() if passed else ("sealed_holdout_counterexample",),
        )


def _evaluate_world(
    prepared: Mapping[str, Any], candidates: Sequence[CandidateArtifact], target: Mapping[str, Any]
) -> dict[str, Any]:
    descriptor = prepared["descriptor"]
    pack = _RepairPack(descriptor, target["hypothesis"])
    ladder = _ladder(descriptor)
    evaluations: list[dict[str, Any]] = []
    eligible: list[CandidateArtifact] = []
    eligible_gates: list[GateOutcome] = []
    for candidate in candidates:
        result = evaluate_candidate(pack, candidate, ladder)
        validate_evaluation_replay(result, pack, candidate)
        evaluations.append(result)
        if result["all_required_gates_passed"]:
            eligible.append(candidate)
            eligible_gates.extend(GateOutcome.from_dict(row) for row in result["gate_outcomes"])
    metrics = _metrics(eligible)
    pareto = None
    if eligible:
        directions = {"lineage_inputs": "maximize", "representation_bytes": "minimize"}
        limits = ParetoLimits(24, 2, 2, 4096)
        pareto = build_pareto_explanations(
            eligible,
            eligible_gates,
            metrics,
            required_gate_ids=("hard_exact", "hard_holdout"),
            metric_directions=directions,
            limits=limits,
        )
        validate_pareto_replay(
            pareto,
            eligible,
            eligible_gates,
            metrics,
            required_gate_ids=("hard_exact", "hard_holdout"),
            metric_directions=directions,
            limits=limits,
        )
    status_counts = Counter(
        "pass" if result["all_required_gates_passed"] else "reject" for result in evaluations
    )
    family_pass = Counter(candidate.representation["family"] for candidate in eligible)
    return {
        "world_id": prepared["world"]["world_id"],
        "public_world_sha256": prepared["public_world_sha256"],
        "sealed_target_sha256": prepared["world"]["sealed_target_sha256"],
        "unsealed_target": dict(target),
        "holdout": _holdout(
            prepared["world"],
            {"hypothesis": target["hypothesis"], "world_id": target["world_id"]},
        ),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "evaluations": evaluations,
        "metric_receipts": [metric.to_dict() for metric in metrics],
        "pareto": pareto,
        "eligible_candidate_refs": [candidate.ref.to_dict() for candidate in eligible],
        "eligible_families": dict(sorted(family_pass.items())),
        "counts": {
            "candidates": len(candidates),
            "pass": status_counts["pass"],
            "reject": status_counts["reject"],
            "block": 0,
        },
        "decision": (
            "pass_fixed_repair_found_holdout_match"
            if eligible
            else "reject_fixed_repair_budget_exhausted"
        ),
    }


def _lane_bindings(root: Path) -> dict[str, dict[str, str]]:
    paths = {
        "config": _resolve(root, CONFIG_PATH),
        "source": _resolve(root, SOURCE_PATH),
        "target_fixture": _resolve(root, TARGET_PATH),
        "test": _resolve(root, TEST_PATH),
    }
    return {
        role: {
            "path": path.relative_to(root).as_posix(),
            "file_sha256": _file_sha256(path),
        }
        for role, path in sorted(paths.items())
    }


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Build the preregistered campaign with one post-generation atomic target read."""

    root = root.resolve()
    config = _load_config(root, config_path)
    with _deny_target_fixture_reads(root) as access_audit:
        prepared, candidates_by_world = _build_phase_a(root, config)
        try:
            with builtins.open(_resolve(root, TARGET_PATH), encoding="utf-8"):
                pass
        except PermissionError:
            pass
        else:
            raise ProspectiveRepairError("pre-unseal target read was not denied")
    if access_audit != {"attempted": 1, "denied": 1, "denied_content_bytes_exposed": 0}:
        raise ProspectiveRepairError("pre-unseal target-read enforcement changed")
    phase_a = _phase_a_receipt(prepared, access_audit)
    targets = _unseal_targets(root, config, phase_a_sealed=True)
    world_results = [
        _evaluate_world(
            row,
            candidates_by_world[row["world"]["world_id"]],
            targets[row["world"]["world_id"]],
        )
        for row in prepared
    ]
    decisions = Counter(row["decision"].split("_", 1)[0] for row in world_results)
    eligible = sum(row["counts"]["pass"] for row in world_results)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "completed_prospective_repaired_non_bayesian_single_unseal_tournament",
        "config_sha256": canonical_sha256(config),
        "source_bindings": _lane_bindings(root),
        "retrospective_predecessor": dict(config["retrospective_predecessor"]),
        "chronology": [
            {"phase": "preregistered_config_loaded", "target_fixture_reads": 0},
            {"phase": "six_native_families_generated", "target_fixture_reads": 0},
            {
                "phase": "seventy_two_repaired_candidates_sealed",
                "root_sha256": phase_a["content_sha256"],
                "target_fixture_reads": 0,
            },
            {"phase": "atomic_three_target_unseal", "target_fixture_reads": 1},
            {"phase": "common_ladder_and_pareto_replayed", "target_fixture_reads": 1},
        ],
        "phase_a": phase_a,
        "world_results": world_results,
        "counts": {
            "worlds": 3,
            "native_families": 6,
            "native_generation_events": 18,
            "repaired_candidates": 72,
            "gate_checks": 144,
            "world_pass": decisions["pass"],
            "world_reject": decisions["reject"],
            "world_block": 0,
            "pareto_eligible_candidates": eligible,
            "target_fixture_reads_pre_unseal": 0,
            "target_fixture_reads_total": 1,
            "post_unseal_generation_events": 0,
            "post_unseal_tuning_events": 0,
        },
        "claims": CLAIMS,
        "scope": (
            "Prospective bounded test of a previously frozen four-candidate SHA-counter breadth "
            "repair across three new hidden finite worlds and six non-Bayesian native families. "
            "A PASS is evidence about this fixed benchmark only, not truth, novelty, or general "
            "formula-discovery completeness."
        ),
        "first_remaining_blocker": (
            "replicate_with_independently_generated_worlds_and_semantic_not_token_holdouts"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_campaign(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    """Fail closed unless a receipt exactly replays from live bound inputs."""

    expected = build_campaign(root, config_path)
    if dict(value) != expected:
        raise ProspectiveRepairError("prospective repaired tournament replay mismatch")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProspectiveRepairError("refusing to replace immutable campaign")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = build_campaign(root)
    _write_immutable(_resolve(root, args.output), result)
    validate_campaign(result, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
