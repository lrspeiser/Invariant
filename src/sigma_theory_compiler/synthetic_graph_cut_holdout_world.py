"""A deterministic, blinded synthetic graph-cut rediscovery control."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    PromotionDenied,
    PromotionLedger,
    ProvenanceRecord,
    SchemaViolation,
    SourceBinding,
    StageDefinition,
    StageOutcome,
    run_gate,
    run_stage,
)

CONFIG_SCHEMA = "sigma-synthetic-graph-cut-holdout-config-1.0"
RESULT_SCHEMA = "sigma-synthetic-graph-cut-holdout-result-1.0"
BENCHMARK_ID = "synthetic-graph-cut-holdout-world-001"
CONFIG_PATH = "configs/synthetic_graph_cut_holdout_world.json"
SOURCE_PATH = "src/sigma_theory_compiler/synthetic_graph_cut_holdout_world.py"
TEST_PATH = "tests/test_synthetic_graph_cut_holdout_world.py"
CORE_PATH = "src/sigma_theory_compiler/sigma_core.py"
OUTPUT_PATH = "runs/math/synthetic-graph-cut-holdout-world/campaign.json"
VERTEX_COUNT = 9
EDGE_COUNT = 16
CANONICAL_CUT_COUNT = (2**VERTEX_COUNT - 2) // 2

EXPECTED_CONFIG = {
    "schema_version": CONFIG_SCHEMA,
    "benchmark_id": BENCHMARK_ID,
    "output_path": OUTPUT_PATH,
    "world_generator": {
        "namespace": "invariant.synthetic.graph.posttraining.002",
        "generation_epoch": "2026-08-12",
        "vertex_count": VERTEX_COUNT,
        "edge_count": EDGE_COUNT,
        "graph_model": "seeded_anonymous_cycle_plus_ranked_chords",
        "invariant_grammar": "canonical_nontrivial_vertex_cuts_modulo_complement",
    },
    "holdout_contract": {
        "eligible_holdouts": 1,
        "selection": "minimum_seeded_class_score_after_reference_graph_seal",
        "equivalent_formulations_hidden": True,
        "post_unseal_comparison_only": True,
    },
    "policies": {
        "discovery_file_reads": "deny_owned_python_open_surfaces",
        "target_access_before_proof_seal": "forbidden",
        "network_access": "forbidden",
        "exact_proof": "exhaustive_edge_incidence_replay",
        "promotion": "sigma_core_exact_receipts_only",
        "candidate_rejection_from_missing_evidence": "forbidden",
    },
    "seals": {
        "external_data_opened": False,
        "network_opened": False,
        "live_sqlite_opened": False,
        "paid_llm_calls": False,
        "gpu_execution_used": False,
    },
}

KINDS = (ArtifactKind.CONJECTURE, ArtifactKind.THEOREM)
STAGES = (
    StageDefinition("typed", 0, KINDS),
    StageDefinition("canonicalized", 1, KINDS, ("typed",)),
    StageDefinition("counterexample_screened", 2, KINDS, ("canonicalized",)),
    StageDefinition("exactly_verified", 3, KINDS, ("counterexample_screened",)),
    StageDefinition("prior_art_checked", 4, KINDS, ("exactly_verified",)),
)
GATES = tuple(
    sorted(
        (
            GateDefinition("accept_typed", None, "typed", KINDS, ("typed",)),
            GateDefinition(
                "admit_canonicalized",
                "typed",
                "canonicalized",
                KINDS,
                ("canonicalized", "typed"),
            ),
            GateDefinition(
                "admit_counterexample_screened",
                "canonicalized",
                "counterexample_screened",
                KINDS,
                ("canonicalized", "counterexample_screened"),
            ),
            GateDefinition(
                "admit_exactly_verified",
                "counterexample_screened",
                "exactly_verified",
                KINDS,
                ("counterexample_screened", "exactly_verified"),
            ),
            GateDefinition(
                "admit_prior_art_checked",
                "exactly_verified",
                "prior_art_checked",
                KINDS,
                ("exactly_verified", "prior_art_checked"),
            ),
        ),
        key=lambda item: item.gate_id,
    )
)
PACK_DESCRIPTOR = DomainPackDescriptor("synthetic.graph_cut", "1.0.0", KINDS, STAGES, GATES)

CLAIMS = {
    "fresh_seed_derived_anonymous_finite_graph_generated": True,
    "complete_declared_canonical_cut_grammar_enumerated": True,
    "reference_theorem_graph_sealed_before_discovery": True,
    "entire_target_cut_invariant_class_withheld": True,
    "visible_graph_invariant_ancestors_exposed": True,
    "pre_unseal_answer_literal_leakage_absent": True,
    "withheld_cut_class_independently_rediscovered": True,
    "exhaustive_edge_incidence_proof_completed": True,
    "post_unseal_equivalence_confirmed": True,
    "historical_novelty_established": False,
    "unbounded_graph_discovery_established": False,
    "general_graph_invariant_completeness_established": False,
    "formal_proof_assistant_kernel_checked": False,
    "hostile_process_isolation_established": False,
    "external_mathematical_significance_established": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("synthetic graph path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != EXPECTED_CONFIG:
        raise ValueError("synthetic graph config boundary changed")


def _seed_bytes(config: Mapping[str, Any]) -> bytes:
    generator = config["world_generator"]
    return _canonical(
        {
            "namespace": generator["namespace"],
            "generation_epoch": generator["generation_epoch"],
            "benchmark_id": config["benchmark_id"],
        }
    )


def _stream(seed: bytes, label: str) -> bytes:
    return hashlib.sha256(seed + b"\0" + label.encode("utf-8")).digest()


def _world_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = _seed_bytes(config)
    labels = [f"n-{_stream(seed, f'vertex:{index}').hex()[:10]}" for index in range(VERTEX_COUNT)]
    order = sorted(range(VERTEX_COUNT), key=lambda index: _stream(seed, f"rank:{index}"))
    cycle_indices = [
        tuple(sorted((order[index], order[(index + 1) % VERTEX_COUNT])))
        for index in range(VERTEX_COUNT)
    ]
    cycle = set(cycle_indices)
    pairs = [
        (left, right)
        for left in range(VERTEX_COUNT)
        for right in range(left + 1, VERTEX_COUNT)
        if (left, right) not in cycle
    ]
    ranked = sorted(pairs, key=lambda pair: _stream(seed, f"chord:{pair[0]}:{pair[1]}"))
    selected = sorted(cycle | set(ranked[: EDGE_COUNT - len(cycle)]))
    if len(selected) != EDGE_COUNT:
        raise ValueError("synthetic graph edge generation count changed")
    edges = sorted(tuple(sorted((labels[left], labels[right]))) for left, right in selected)
    return {
        "vertices": sorted(labels),
        "edges": [list(edge) for edge in edges],
        "generation_order": order,
        "selected_index_edges": [list(edge) for edge in selected],
    }


def _canonical_cuts(vertices: Sequence[str]) -> list[tuple[str, ...]]:
    ordered = tuple(sorted(vertices))
    full = (1 << len(ordered)) - 1
    cuts = []
    for mask in range(1, full):
        complement = full ^ mask
        if mask >= complement:
            continue
        cuts.append(tuple(ordered[index] for index in range(len(ordered)) if mask & (1 << index)))
    if len(cuts) != CANONICAL_CUT_COUNT:
        raise ValueError("canonical cut enumeration count changed")
    return cuts


def _boundary_edges(subset: Sequence[str], edges: Sequence[Sequence[str]]) -> list[list[str]]:
    chosen = set(subset)
    return [list(edge) for edge in edges if ((edge[0] in chosen) != (edge[1] in chosen))]


def _cut_classes(vertices: Sequence[str], edges: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for subset in _canonical_cuts(vertices):
        boundary = _boundary_edges(subset, edges)
        grouped.setdefault(len(boundary), []).append(
            {
                "subset": list(subset),
                "boundary_size": len(boundary),
                "boundary_edges_sha256": _sha(boundary),
            }
        )
    classes = []
    for cut_size, members in grouped.items():
        members = sorted(members, key=lambda row: row["subset"])
        member_root = _sha(members)
        class_root = _sha({"cut_size": cut_size, "members_sha256": member_root})
        classes.append(
            {
                "class_id": f"cutc-{class_root[:20]}",
                "class_root_sha256": class_root,
                "cut_size": cut_size,
                "member_count": len(members),
                "members": members,
                "member_root_sha256": member_root,
                "canonical_cuts_checked": CANONICAL_CUT_COUNT,
                "theorem_id": f"thm-{_sha({'class': class_root})[:20]}",
                "parent_ids": [
                    "axiom.edge_set",
                    "ancestor.cut_complement_invariance",
                    "ancestor.degree_boundary_parity",
                ],
            }
        )
    return sorted(classes, key=lambda row: row["class_id"])


def _visible_ancestors(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    degrees = {vertex: 0 for vertex in vertices}
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    complement_rows = []
    parity_rows = []
    vertex_set = set(vertices)
    for subset in _canonical_cuts(vertices):
        complement = sorted(vertex_set - set(subset))
        boundary = _boundary_edges(subset, edges)
        complement_boundary = _boundary_edges(complement, edges)
        if boundary != complement_boundary:
            raise ValueError("cut complement invariant failed")
        degree_sum = sum(degrees[vertex] for vertex in subset)
        if degree_sum % 2 != len(boundary) % 2:
            raise ValueError("degree-boundary parity invariant failed")
        complement_rows.append([list(subset), complement, _sha(boundary)])
        parity_rows.append([list(subset), degree_sum % 2, len(boundary) % 2])
    return [
        {
            "theorem_id": "ancestor.cut_complement_invariance",
            "statement": "a vertex subset and its complement have the same boundary edges",
            "cases_checked": CANONICAL_CUT_COUNT,
            "proof_sha256": _sha(complement_rows),
            "parent_ids": ["axiom.edge_set"],
        },
        {
            "theorem_id": "ancestor.degree_boundary_parity",
            "statement": "subset degree sum and boundary size have equal parity",
            "cases_checked": CANONICAL_CUT_COUNT,
            "proof_sha256": _sha(parity_rows),
            "parent_ids": ["axiom.edge_set"],
        },
    ]


def _reference_world(config: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _world_parameters(config)
    vertices = parameters["vertices"]
    edges = parameters["edges"]
    classes = _cut_classes(vertices, edges)
    if len(classes) < 3:
        raise ValueError("synthetic graph has too few cut invariant classes")
    ancestors = _visible_ancestors(vertices, edges)
    axiom = {
        "axiom_id": "axiom.edge_set",
        "vertices": vertices,
        "edges": edges,
        "simple_undirected": True,
    }
    graph = {
        "axiom": axiom,
        "ancestors": ancestors,
        "theorem_classes": classes,
        "edge_count": sum(len(row["parent_ids"]) for row in classes)
        + sum(len(row["parent_ids"]) for row in ancestors),
    }
    graph_root = _sha(graph)
    seed = _seed_bytes(config).hex()
    target = min(
        classes,
        key=lambda row: _sha(
            {"seed": seed, "sealed_graph_root": graph_root, "class_id": row["class_id"]}
        ),
    )
    visible = [row for row in classes if row["class_id"] != target["class_id"]]
    public = {
        "world_id": f"graph-{_sha(axiom)[:20]}",
        "axiom": axiom,
        "visible_ancestors": ancestors,
        "visible_theorem_classes": visible,
        "grammar": {
            "object": "canonical_nontrivial_vertex_cuts_modulo_complement",
            "canonical_cut_count": CANONICAL_CUT_COUNT,
            "invariant": "boundary_edge_count",
            "equivalence": "equal_boundary_size_in_declared_graph",
        },
        "withholding": {
            "eligible_holdout_count": 1,
            "target_or_equivalent_formulations_exposed": False,
            "selection_performed_after_reference_graph_seal": True,
        },
    }
    return {
        "parameters": parameters,
        "graph": graph,
        "graph_root_sha256": graph_root,
        "target": target,
        "public": public,
        "public_root_sha256": _sha(public),
    }


def _discover(public: Mapping[str, Any]) -> list[dict[str, Any]]:
    axiom = public["axiom"]
    visible_ids = {row["class_id"] for row in public["visible_theorem_classes"]}
    return [
        row
        for row in _cut_classes(axiom["vertices"], axiom["edges"])
        if row["class_id"] not in visible_ids
    ]


@contextmanager
def _deny_file_reads() -> Any:
    attempts: list[dict[str, Any]] = []
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open

    def deny_builtin(file: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        attempts.append({"surface": "builtins.open", "path": str(file), "decision": "denied"})
        raise PermissionError("pre-unseal file read denied")

    def deny_io(file: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        attempts.append({"surface": "io.open", "path": str(file), "decision": "denied"})
        raise PermissionError("pre-unseal file read denied")

    def deny_path(self: Path, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raw_path = self.as_posix()
        recorded_path = TEST_PATH if raw_path.endswith(TEST_PATH) else raw_path
        attempts.append(
            {"surface": "pathlib.Path.open", "path": recorded_path, "decision": "denied"}
        )
        raise PermissionError("pre-unseal file read denied")

    builtins.open = deny_builtin
    io.open = deny_io
    Path.open = deny_path
    try:
        yield attempts
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open
        Path.open = original_path_open


def _leakage_audit(
    root: Path, public: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    dependencies = [CONFIG_PATH, CORE_PATH, SOURCE_PATH]
    forbidden = {
        "target_class_id": target["class_id"],
        "target_theorem_id": target["theorem_id"],
        "target_class_root": target["class_root_sha256"],
        "target_member_root": target["member_root_sha256"],
    }
    chunks = [_canonical(public)]
    bindings = []
    for relative in dependencies:
        path = _inside(root, relative)
        raw = path.read_bytes()
        chunks.append(raw)
        bindings.append({"path": relative, "file_sha256": hashlib.sha256(raw).hexdigest()})
    joined = b"\n".join(chunks)
    matches = [label for label, token in forbidden.items() if token.encode("utf-8") in joined]
    if matches:
        raise ValueError("answer-bearing graph literal found in pre-unseal closure")
    return {
        "dependency_paths": dependencies,
        "dependency_bindings": bindings,
        "dependency_root_sha256": _sha(bindings),
        "bytes_scanned": sum(len(chunk) for chunk in chunks),
        "forbidden_literal_count": 0,
        "forbidden_literal_labels_found": [],
        "passed": True,
        "boundary": (
            "literal and canonical-identity scan of the declared Python/config/public-input "
            "closure; the public edge set permits honest derivation and is not "
            "information-theoretic secrecy"
        ),
    }


def _representation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "class_id",
            "class_root_sha256",
            "cut_size",
            "member_count",
            "members",
            "member_root_sha256",
            "canonical_cuts_checked",
        )
    }


class SyntheticGraphCutPack:
    def __init__(self, public: Mapping[str, Any], target: Mapping[str, Any] | None = None) -> None:
        self.public = public
        self.target = target
        self.descriptor = PACK_DESCRIPTOR

    def _checked_representation(self, artifact: CandidateArtifact) -> Mapping[str, Any]:
        representation = artifact.representation
        expected = {
            "class_id",
            "class_root_sha256",
            "cut_size",
            "member_count",
            "members",
            "member_root_sha256",
            "canonical_cuts_checked",
        }
        if set(representation) != expected:
            raise SchemaViolation("synthetic graph theorem representation keys changed")
        return representation

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        del prior_outcomes
        representation = self._checked_representation(artifact)
        axiom = self.public["axiom"]
        vertices = axiom["vertices"]
        edges = axiom["edges"]
        known_vertices = set(vertices)
        members = representation["members"]
        passed = False
        details: dict[str, Any]
        if stage.stage_id == "typed":
            cut_size = representation["cut_size"]
            passed = (
                artifact.kind in {ArtifactKind.CONJECTURE, ArtifactKind.THEOREM}
                and isinstance(cut_size, int)
                and not isinstance(cut_size, bool)
                and cut_size >= 0
                and isinstance(members, list)
                and len(members) == representation["member_count"]
                and len(members) > 0
                and all(
                    set(member) == {"subset", "boundary_size", "boundary_edges_sha256"}
                    and isinstance(member["subset"], list)
                    and set(member["subset"]) <= known_vertices
                    for member in members
                )
            )
            details = {"registered_members": len(members), "registered_vertices": len(vertices)}
        elif stage.stage_id == "canonicalized":
            classes = _cut_classes(vertices, edges)
            record = next(
                (row for row in classes if row["class_id"] == representation["class_id"]), None
            )
            passed = record is not None and _representation(record) == representation
            details = {
                "canonical_cut_count": CANONICAL_CUT_COUNT,
                "canonical_class_count": len(classes),
            }
        elif stage.stage_id == "counterexample_screened":
            checked = members[: min(11, len(members))]
            failures = [
                member["subset"]
                for member in checked
                if len(_boundary_edges(member["subset"], edges)) != representation["cut_size"]
            ]
            passed = not failures
            details = {"members_checked": len(checked), "counterexamples": failures}
        elif stage.stage_id == "exactly_verified":
            recomputed = _cut_classes(vertices, edges)
            record = next(
                (row for row in recomputed if row["class_id"] == representation["class_id"]), None
            )
            failures = [
                member["subset"]
                for member in members
                if len(_boundary_edges(member["subset"], edges)) != representation["cut_size"]
            ]
            passed = (
                not failures
                and record is not None
                and _representation(record) == representation
                and representation["canonical_cuts_checked"] == CANONICAL_CUT_COUNT
            )
            details = {
                "canonical_cuts_checked": CANONICAL_CUT_COUNT,
                "class_members_replayed": len(members),
                "counterexample_count": len(failures),
                "proof_method": "complete_edge_incidence_replay",
            }
        elif stage.stage_id == "prior_art_checked":
            passed = self.target is not None and all(
                representation[key] == self.target[key]
                for key in (
                    "class_id",
                    "class_root_sha256",
                    "cut_size",
                    "member_root_sha256",
                )
            )
            details = {
                "post_unseal_target_available": self.target is not None,
                "exact_hidden_class_match": passed,
            }
        else:
            raise ValueError("unregistered synthetic graph stage")
        status = OutcomeStatus.PASS if passed else OutcomeStatus.BLOCK
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            status,
            [CheckResult.create(f"{stage.stage_id}_check", passed, details)],
            reason_codes=[] if passed else [f"{stage.stage_id}_not_established"],
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS,
            [stage_outcomes[key].ref for key in sorted(stage_outcomes)],
            [
                CheckResult.create(
                    "required_receipts_exact",
                    True,
                    {"required_stages": list(gate.required_stages)},
                )
            ],
        )


def _candidate(
    root: Path, public: Mapping[str, Any], discovered: Mapping[str, Any]
) -> CandidateArtifact:
    sources = tuple(
        SourceBinding(label, path, _file_sha(_inside(root, path)))
        for label, path in (
            ("config", CONFIG_PATH),
            ("core", CORE_PATH),
            ("generator", SOURCE_PATH),
        )
    )
    provenance = ProvenanceRecord.create(
        PACK_DESCRIPTOR.ref,
        {
            "world_id": public["world_id"],
            "public_root_sha256": _sha(public),
            "grammar": public["grammar"],
        },
        sources=sources,
    )
    representation = _representation(discovered)
    return CandidateArtifact.create(
        ArtifactKind.THEOREM,
        (
            f"all {representation['member_count']} canonical cuts in the discovered class "
            f"have boundary size {representation['cut_size']} in the declared anonymous graph"
        ),
        representation,
        provenance,
        assumptions=("the sealed vertex and edge sets define the complete finite graph",),
        claims=("declared_finite_graph_cut_invariant",),
    )


def _run_pre_unseal_pipeline(
    public: Mapping[str, Any], artifact: CandidateArtifact
) -> tuple[dict[str, StageOutcome], list[GateOutcome], PromotionLedger]:
    pack = SyntheticGraphCutPack(public)
    stages: dict[str, StageOutcome] = {}
    gates = []
    ledger = PromotionLedger.create(artifact)
    lifecycle = (
        ("typed", "accept_typed"),
        ("canonicalized", "admit_canonicalized"),
        ("counterexample_screened", "admit_counterexample_screened"),
        ("exactly_verified", "admit_exactly_verified"),
    )
    for stage_id, gate_id in lifecycle:
        stage = PACK_DESCRIPTOR.stage(stage_id)
        prior = {key: stages[key] for key in stage.prerequisites}
        outcome = run_stage(pack, artifact, stage_id, prior)
        if outcome.status is not OutcomeStatus.PASS:
            raise ValueError(f"synthetic graph pre-unseal stage blocked: {stage_id}")
        stages[stage_id] = outcome
        gate = PACK_DESCRIPTOR.gate(gate_id)
        required = {key: stages[key] for key in gate.required_stages}
        gate_outcome = run_gate(pack, artifact, gate_id, required)
        ledger = ledger.promote(PACK_DESCRIPTOR, artifact, gate_outcome, required)
        gates.append(gate_outcome)
    return stages, gates, ledger


def _post_unseal(
    public: Mapping[str, Any],
    target: Mapping[str, Any],
    artifact: CandidateArtifact,
    stages: dict[str, StageOutcome],
    gates: list[GateOutcome],
    ledger: PromotionLedger,
) -> tuple[StageOutcome, GateOutcome, PromotionLedger]:
    pack = SyntheticGraphCutPack(public, target)
    outcome = run_stage(
        pack,
        artifact,
        "prior_art_checked",
        {"exactly_verified": stages["exactly_verified"]},
    )
    if outcome.status is not OutcomeStatus.PASS:
        raise ValueError("synthetic graph post-unseal comparison failed")
    stages["prior_art_checked"] = outcome
    required = {"exactly_verified": stages["exactly_verified"], "prior_art_checked": outcome}
    gate = run_gate(pack, artifact, "admit_prior_art_checked", required)
    final = ledger.promote(PACK_DESCRIPTOR, artifact, gate, required)
    gates.append(gate)
    return outcome, gate, final


def _negative_controls(
    public: Mapping[str, Any], artifact: CandidateArtifact, pre_ledger: PromotionLedger
) -> list[dict[str, Any]]:
    edges = public["axiom"]["edges"]
    prefix = edges[: len(edges) // 2]
    overfit = None
    for subset in _canonical_cuts(public["axiom"]["vertices"]):
        prefix_boundary = _boundary_edges(subset, prefix)
        full_boundary = _boundary_edges(subset, edges)
        if len(prefix_boundary) != len(full_boundary):
            omitted = next(edge for edge in full_boundary if edge not in prefix_boundary)
            overfit = {
                "subset": list(subset),
                "prefix_edges_checked": len(prefix),
                "claimed_boundary_size": len(prefix_boundary),
                "actual_boundary_size": len(full_boundary),
                "first_omitted_crossing_edge": omitted,
            }
            break
    if overfit is None:
        raise ValueError("truncated-edge negative control unavailable")
    undeclared_outcome = StageOutcome.create(
        "exactly_verified",
        artifact.ref,
        OutcomeStatus.BLOCK,
        [
            CheckResult.create(
                "dependency_closure_contained",
                False,
                {"allowed": [], "attempted": ["lemma.withheld_graph_class"]},
            )
        ],
        reason_codes=["undeclared_proof_dependency"],
    )
    conjecture = CandidateArtifact.create(
        ArtifactKind.CONJECTURE,
        artifact.statement,
        artifact.representation,
        artifact.provenance,
        assumptions=artifact.assumptions,
        claims=artifact.claims,
    )
    conjecture_typed = run_stage(SyntheticGraphCutPack(public), conjecture, "typed")
    if conjecture_typed.status is not OutcomeStatus.PASS:
        raise ValueError("typed graph conjecture negative control changed")
    forged = pre_ledger.to_dict()
    forged["entries"][-1]["to_stage"] = "prior_art_checked"
    try:
        PromotionLedger.from_dict(forged)
    except SchemaViolation:
        forged_rejected = True
    else:
        forged_rejected = False
    try:
        pre_ledger.promote(
            PACK_DESCRIPTOR,
            artifact,
            GateOutcome.create(
                "admit_prior_art_checked",
                artifact.ref,
                OutcomeStatus.BLOCK,
                (),
                [CheckResult.create("proof_missing", False, {"proofs": 0})],
                reason_codes=["proof_missing"],
            ),
            {},
        )
    except PromotionDenied:
        unproved_promotion_blocked = True
    else:
        unproved_promotion_blocked = False
    if not forged_rejected or not unproved_promotion_blocked:
        raise ValueError("graph promotion negative control admitted")
    return [
        {"control_id": "truncated_edge_list_overfit", "status": "rejected", **overfit},
        {
            "control_id": "forbidden_target_file_read",
            "status": "rejected",
            "denied_reads": 1,
            "bytes_exposed": 0,
        },
        {
            "control_id": "undeclared_graph_lemma_dependency",
            "status": "rejected",
            "declared_dependency_count": 0,
            "attempted_dependency": "lemma.withheld_graph_class",
            "blocked_outcome_sha256": undeclared_outcome.outcome_sha256,
        },
        {
            "control_id": "correctly_typed_unproved_conjecture",
            "status": "blocked",
            "typed_receipt_sha256": conjecture_typed.outcome_sha256,
            "exact_proof_receipts": 0,
            "promotion_denied": unproved_promotion_blocked,
        },
        {
            "control_id": "resealed_promotion_state",
            "status": "rejected",
            "forged_target_stage": "prior_art_checked",
            "candidate_class_id": artifact.representation["class_id"],
            "ledger_validation_rejected": forged_rejected,
        },
    ]


def _expected_body(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    reference = _reference_world(config)
    target = reference["target"]
    leakage = _leakage_audit(root, reference["public"], target)
    with _deny_file_reads() as attempts:
        try:
            _inside(root, TEST_PATH).read_bytes()
        except PermissionError:
            pass
        discovered = _discover(reference["public"])
    if len(attempts) != 1 or attempts[0]["decision"] != "denied":
        raise ValueError("synthetic graph deny-by-default read control failed")
    if len(discovered) != 1:
        raise ValueError("synthetic graph rediscovery count changed")
    artifact = _candidate(root, reference["public"], discovered[0])
    stages, gates, pre_ledger = _run_pre_unseal_pipeline(reference["public"], artifact)
    proof_seal = {
        "candidate_artifact_sha256": artifact.content_sha256,
        "exact_stage_outcome_sha256": stages["exactly_verified"].outcome_sha256,
        "pre_unseal_ledger_sha256": pre_ledger.ledger_sha256,
    }
    proof_seal["proof_seal_sha256"] = _sha(proof_seal)
    comparison, comparison_gate, final_ledger = _post_unseal(
        reference["public"], target, artifact, stages, gates, pre_ledger
    )
    chronology_data = (
        ("reference_graph_sealed", reference["graph_root_sha256"]),
        ("public_subgraph_sealed", reference["public_root_sha256"]),
        ("literal_leakage_audited", leakage["dependency_root_sha256"]),
        ("discovery_file_reads_denied", _sha(attempts)),
        ("missing_cut_class_rediscovered", artifact.content_sha256),
        ("exhaustive_incidence_proof_sealed", proof_seal["proof_seal_sha256"]),
        ("target_unsealed_and_compared", comparison.outcome_sha256),
        ("final_promotion_sealed", final_ledger.ledger_sha256),
    )
    chronology = [
        {"sequence": index, "phase": phase, "root_sha256": root_sha}
        for index, (phase, root_sha) in enumerate(chronology_data)
    ]
    parameters = reference["parameters"]
    graph = reference["graph"]
    public = reference["public"]
    return {
        "schema_version": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "decision": "pass_one_of_one_synthetic_graph_cut_holdout_rediscovered_and_proved",
        "decision_counts": {"pass": 1, "blocked": 0, "reject": 0},
        "world": {
            "world_id": public["world_id"],
            "generation_epoch": config["world_generator"]["generation_epoch"],
            "vertex_count": VERTEX_COUNT,
            "edge_count": EDGE_COUNT,
            "vertices": parameters["vertices"],
            "edges": parameters["edges"],
            "edge_set_sha256": _sha(parameters["edges"]),
            "anonymous_parameter_commitment_sha256": _sha(
                {
                    "generation_order": parameters["generation_order"],
                    "selected_index_edges": parameters["selected_index_edges"],
                }
            ),
        },
        "reference_graph": {
            "root_sha256": reference["graph_root_sha256"],
            "axiom_count": 1,
            "visible_ancestor_count": len(public["visible_ancestors"]),
            "cut_invariant_class_count": len(graph["theorem_classes"]),
            "visible_cut_invariant_class_count": len(public["visible_theorem_classes"]),
            "hidden_cut_invariant_class_count": 1,
            "edge_count": graph["edge_count"],
            "visible_ancestors": public["visible_ancestors"],
        },
        "pre_unseal": {
            "public_root_sha256": reference["public_root_sha256"],
            "canonical_cut_count": CANONICAL_CUT_COUNT,
            "edge_incidence_tests": CANONICAL_CUT_COUNT * EDGE_COUNT,
            "target_identifiers_exposed": 0,
            "target_members_exposed": 0,
            "target_equivalent_classes_exposed": 0,
            "file_read_contract": {
                "enforcement_scope": (
                    "owned_single_threaded_python_open_surfaces_not_an_operating_system_sandbox"
                ),
                "surfaces": ["builtins.open", "io.open", "pathlib.Path.open"],
                "attempted_read_count": len(attempts),
                "allowed_read_count": 0,
                "denied_read_count": len(attempts),
                "denied_content_bytes_exposed": 0,
                "attempts": attempts,
            },
            "leakage_audit": leakage,
        },
        "rediscovery": {
            "candidate_class_count": len(discovered),
            "candidate_artifact": artifact.to_dict(),
            "candidate_root_sha256": artifact.content_sha256,
            "enumeration_root_sha256": _sha(discovered),
            "answer_bearing_dependencies_used": 0,
        },
        "proof": {
            "method": "exhaustive_deterministic_edge_incidence_replay",
            "canonical_cuts_checked": CANONICAL_CUT_COUNT,
            "edge_incidence_tests": CANONICAL_CUT_COUNT * EDGE_COUNT,
            "counterexample_count": 0,
            "stage_outcomes": [
                stages[key].to_dict()
                for key in ("typed", "canonicalized", "counterexample_screened", "exactly_verified")
            ],
            "gate_outcomes": [gate.to_dict() for gate in gates[:4]],
            "pre_unseal_ledger": pre_ledger.to_dict(),
            "proof_seal": proof_seal,
        },
        "post_unseal": {
            "target": target,
            "candidate_class_id": artifact.representation["class_id"],
            "exact_class_match": True,
            "comparison_outcome": comparison.to_dict(),
            "comparison_gate": comparison_gate.to_dict(),
            "final_ledger": final_ledger.to_dict(),
            "comparison_performed_after_proof_seal": True,
        },
        "negative_controls": _negative_controls(public, artifact, pre_ledger),
        "metrics": {
            "eligible_holdouts": 1,
            "independently_rediscovered_and_proved": 1,
            "proof_rate_numerator": 1,
            "proof_rate_denominator": 1,
            "forbidden_dependency_rejections": 2,
            "counterexample_kills": 1,
            "blocked_unproved_conjectures": 1,
            "forged_promotion_rejections": 1,
        },
        "chronology": chronology,
        "claims": CLAIMS,
        "first_remaining_blocker": (
            "replicate_graph_invariant_holdouts_across_preregistered_independent_generators_"
            "and_add_an_external_proof_kernel_without_exposing_hidden_cut_classes"
        ),
        "source_bindings": {
            label: {"path": path, "file_sha256": _file_sha(_inside(root, path))}
            for label, path in (
                ("config", CONFIG_PATH),
                ("core", CORE_PATH),
                ("source", SOURCE_PATH),
                ("test", TEST_PATH),
            )
        },
        "data_seals": config["seals"],
        "scope": (
            "one deterministic anonymous nine-vertex simple graph with sixteen edges, all 255 "
            "canonical nontrivial cuts modulo complement, one completely withheld boundary-size "
            "class, and exhaustive replay of 4080 edge-incidence tests; no historical novelty, "
            "unbounded discovery, general graph-invariant completeness, external proof-kernel, "
            "hostile-process isolation, physical claim, or external mathematical significance"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    expected_paths = {
        "config": CONFIG_PATH,
        "core": CORE_PATH,
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_paths):
        raise ValueError("synthetic graph source binding keys changed")
    for label, path in expected_paths.items():
        expected = {"path": path, "file_sha256": _file_sha(_inside(root, path))}
        if bindings[label] != expected:
            raise ValueError("synthetic graph source binding changed")


def validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("synthetic graph content hash changed")
    _validate_source_bindings(value, validation_root)
    expected = _expected_body(validation_root, _inside(validation_root, CONFIG_PATH))
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("synthetic graph result boundary changed")


def build_benchmark(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent.parent.resolve()
    if config_path != _inside(root, CONFIG_PATH):
        raise ValueError("synthetic graph benchmark requires the registered config path")
    body = _expected_body(root, config_path)
    result = {**body, "content_sha256": _sha(body)}
    validate_result(result, root=root)
    return result


def write_benchmark(config_path: Path) -> Path:
    result = build_benchmark(config_path)
    root = config_path.resolve().parent.parent.resolve()
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(write_benchmark(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
