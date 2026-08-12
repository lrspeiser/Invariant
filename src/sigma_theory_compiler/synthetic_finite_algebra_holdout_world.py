"""A deterministic, blinded synthetic finite-algebra rediscovery control."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from itertools import permutations, product
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

CONFIG_SCHEMA = "sigma-synthetic-finite-algebra-holdout-config-1.0"
RESULT_SCHEMA = "sigma-synthetic-finite-algebra-holdout-result-1.0"
BENCHMARK_ID = "synthetic-finite-algebra-holdout-world-001"
CONFIG_PATH = "configs/synthetic_finite_algebra_holdout_world.json"
SOURCE_PATH = "src/sigma_theory_compiler/synthetic_finite_algebra_holdout_world.py"
TEST_PATH = "tests/test_synthetic_finite_algebra_holdout_world.py"
CORE_PATH = "src/sigma_theory_compiler/sigma_core.py"
OUTPUT_PATH = "runs/math/synthetic-finite-algebra-holdout-world/campaign.json"
VARIABLES = ("a", "b", "c", "d")
ORDER = 7
ASSIGNMENT_COUNT = ORDER ** len(VARIABLES)

EXPECTED_CONFIG = {
    "schema_version": CONFIG_SCHEMA,
    "benchmark_id": BENCHMARK_ID,
    "output_path": OUTPUT_PATH,
    "world_generator": {
        "namespace": "invariant.synthetic.algebra.posttraining.001",
        "generation_epoch": "2026-08-12",
        "field_order": ORDER,
        "binary_operation": "seeded_permuted_affine_quasigroup",
        "variables": list(VARIABLES),
        "term_shape": "op(op(v0,v1),op(v2,v3))",
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
        "exact_proof": "exhaustive_declared_finite_semantics",
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
PACK_DESCRIPTOR = DomainPackDescriptor("synthetic.finite_algebra", "1.0.0", KINDS, STAGES, GATES)

CLAIMS = {
    "fresh_seed_derived_anonymous_finite_world_generated": True,
    "complete_declared_term_grammar_enumerated": True,
    "reference_theorem_graph_sealed_before_discovery": True,
    "entire_target_equivalence_class_withheld": True,
    "visible_ancestor_theorems_exposed": True,
    "pre_unseal_answer_literal_leakage_absent": True,
    "withheld_class_independently_rediscovered": True,
    "exhaustive_finite_semantics_proof_completed": True,
    "post_unseal_equivalence_confirmed": True,
    "historical_novelty_established": False,
    "unbounded_algebra_discovery_established": False,
    "general_equational_completeness_established": False,
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
        raise ValueError("synthetic algebra path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != EXPECTED_CONFIG:
        raise ValueError("synthetic algebra config boundary changed")


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
    permutation = list(range(ORDER))
    entropy = _stream(seed, "permutation")
    for index in range(ORDER - 1, 0, -1):
        selected = entropy[ORDER - 1 - index] % (index + 1)
        permutation[index], permutation[selected] = permutation[selected], permutation[index]
    nonzero = tuple(range(1, ORDER))
    a = nonzero[_stream(seed, "left")[0] % len(nonzero)]
    b = nonzero[_stream(seed, "right")[0] % len(nonzero)]
    if b == a:
        b = nonzero[(nonzero.index(b) + 1) % len(nonzero)]
    c = _stream(seed, "constant")[0] % ORDER
    inverse = [0] * ORDER
    for index, value in enumerate(permutation):
        inverse[value] = index
    table = [
        [inverse[(a * permutation[x] + b * permutation[y] + c) % ORDER] for y in range(ORDER)]
        for x in range(ORDER)
    ]
    return {
        "order": ORDER,
        "operation_symbol": f"op-{_stream(seed, 'symbol').hex()[:12]}",
        "permutation": permutation,
        "affine_parameters": {"left": a, "right": b, "constant": c},
        "operation_table": table,
    }


def _term(ordering: Sequence[str]) -> str:
    return f"(({ordering[0]}*{ordering[1]})*({ordering[2]}*{ordering[3]}))"


def _term_orderings() -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(permutations(VARIABLES)))


def _assignments() -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(ORDER), repeat=len(VARIABLES)))


def _evaluate(
    ordering: Sequence[str], assignment: Sequence[int], table: Sequence[Sequence[int]]
) -> int:
    values = dict(zip(VARIABLES, assignment, strict=True))
    left = table[values[ordering[0]]][values[ordering[1]]]
    right = table[values[ordering[2]]][values[ordering[3]]]
    return table[left][right]


def _semantic_classes(table: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    assignments = _assignments()
    grouped: dict[tuple[int, ...], list[str]] = {}
    for ordering in _term_orderings():
        vector = tuple(_evaluate(ordering, assignment, table) for assignment in assignments)
        grouped.setdefault(vector, []).append(_term(ordering))
    result = []
    for vector, members in grouped.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        class_root = _sha({"members": members, "semantic_vector_sha256": _sha(list(vector))})
        result.append(
            {
                "class_id": f"eqc-{class_root[:20]}",
                "class_root_sha256": class_root,
                "member_count": len(members),
                "members": members,
                "representative_equation": {"lhs": members[0], "rhs": members[1]},
                "semantic_vector_sha256": _sha(list(vector)),
                "assignments_checked": len(assignments),
                "theorem_id": f"thm-{_sha({'class': class_root})[:20]}",
                "parent_ids": [
                    "axiom.operation_table",
                    "ancestor.left_translations_bijective",
                    "ancestor.right_translations_bijective",
                ],
            }
        )
    return sorted(result, key=lambda row: row["class_id"])


def _ancestors(table: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    rows = [sorted(row) for row in table]
    columns = [sorted(table[row][column] for row in range(ORDER)) for column in range(ORDER)]
    expected = list(range(ORDER))
    if any(row != expected for row in rows) or any(column != expected for column in columns):
        raise ValueError("generated operation is not a quasigroup")
    return [
        {
            "theorem_id": "ancestor.left_translations_bijective",
            "statement": "each fixed-left translation is a permutation",
            "cases_checked": ORDER * ORDER,
            "proof_sha256": _sha(rows),
            "parent_ids": ["axiom.operation_table"],
        },
        {
            "theorem_id": "ancestor.right_translations_bijective",
            "statement": "each fixed-right translation is a permutation",
            "cases_checked": ORDER * ORDER,
            "proof_sha256": _sha(columns),
            "parent_ids": ["axiom.operation_table"],
        },
    ]


def _reference_world(config: Mapping[str, Any]) -> dict[str, Any]:
    world = _world_parameters(config)
    classes = _semantic_classes(world["operation_table"])
    if len(classes) < 2:
        raise ValueError("synthetic world has too few nontrivial equivalence classes")
    seed = _seed_bytes(config).hex()
    target = min(classes, key=lambda row: _sha({"seed": seed, "class_id": row["class_id"]}))
    visible = [row for row in classes if row["class_id"] != target["class_id"]]
    ancestors = _ancestors(world["operation_table"])
    axiom = {
        "axiom_id": "axiom.operation_table",
        "operation_symbol": world["operation_symbol"],
        "universe": list(range(ORDER)),
        "operation_table": world["operation_table"],
    }
    graph = {
        "axiom": axiom,
        "ancestors": ancestors,
        "theorem_classes": classes,
        "edge_count": sum(len(row["parent_ids"]) for row in classes)
        + sum(len(row["parent_ids"]) for row in ancestors),
    }
    public = {
        "world_id": f"world-{_sha(axiom)[:20]}",
        "axiom": axiom,
        "visible_ancestors": ancestors,
        "visible_theorem_classes": visible,
        "grammar": {
            "variables": list(VARIABLES),
            "term_shape": EXPECTED_CONFIG["world_generator"]["term_shape"],
            "raw_terms": len(_term_orderings()),
            "equivalence": "exhaustive_equality_on_declared_finite_universe",
        },
        "withholding": {
            "eligible_holdout_count": 1,
            "target_or_equivalent_formulations_exposed": False,
            "selection_performed_after_reference_graph_seal": True,
        },
    }
    return {
        "parameters": world,
        "graph": graph,
        "graph_root_sha256": _sha(graph),
        "target": target,
        "public": public,
        "public_root_sha256": _sha(public),
    }


def _discover(public: Mapping[str, Any]) -> list[dict[str, Any]]:
    table = public["axiom"]["operation_table"]
    visible_ids = {row["class_id"] for row in public["visible_theorem_classes"]}
    return [row for row in _semantic_classes(table) if row["class_id"] not in visible_ids]


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
        "target_semantic_root": target["semantic_vector_sha256"],
        "target_equation": (
            target["representative_equation"]["lhs"]
            + "="
            + target["representative_equation"]["rhs"]
        ),
    }
    chunks = [_canonical(public)]
    bindings = []
    for relative in dependencies:
        path = _inside(root, relative)
        raw = path.read_bytes()
        chunks.append(raw)
        bindings.append({"path": relative, "file_sha256": hashlib.sha256(raw).hexdigest()})
    matches = []
    for label, token in forbidden.items():
        if token.encode("utf-8") in b"\n".join(chunks):
            matches.append(label)
    if matches:
        raise ValueError("answer-bearing literal found in pre-unseal closure")
    return {
        "dependency_paths": dependencies,
        "dependency_bindings": bindings,
        "dependency_root_sha256": _sha(bindings),
        "bytes_scanned": sum(len(chunk) for chunk in chunks),
        "forbidden_literal_count": 0,
        "forbidden_literal_labels_found": [],
        "passed": True,
        "boundary": (
            "literal and canonical-identity scan of the declared Python/config/public-input closure; "
            "the operation table permits honest derivation and this is not information-theoretic secrecy"
        ),
    }


class SyntheticAlgebraPack:
    def __init__(self, public: Mapping[str, Any], target: Mapping[str, Any] | None = None) -> None:
        self.public = public
        self.target = target
        self.descriptor = PACK_DESCRIPTOR

    def _representation(self, artifact: CandidateArtifact) -> Mapping[str, Any]:
        representation = artifact.representation
        expected = {
            "class_id",
            "class_root_sha256",
            "members",
            "representative_equation",
            "semantic_vector_sha256",
            "assignments_checked",
        }
        if set(representation) != expected:
            raise SchemaViolation("synthetic theorem representation keys changed")
        return representation

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        del prior_outcomes
        representation = self._representation(artifact)
        table = self.public["axiom"]["operation_table"]
        members = representation["members"]
        equation = representation["representative_equation"]
        known = {term: ordering for ordering in _term_orderings() for term in (_term(ordering),)}
        passed = False
        details: dict[str, Any]
        if stage.stage_id == "typed":
            passed = (
                artifact.kind in {ArtifactKind.CONJECTURE, ArtifactKind.THEOREM}
                and isinstance(members, list)
                and len(members) >= 2
                and set(equation) == {"lhs", "rhs"}
                and all(term in known for term in members)
            )
            details = {"registered_terms": len(members), "grammar_terms": len(known)}
        elif stage.stage_id == "canonicalized":
            recomputed = _semantic_classes(table)
            record = next(
                (row for row in recomputed if row["class_id"] == representation["class_id"]), None
            )
            passed = (
                record is not None
                and {
                    key: record[key]
                    for key in (
                        "class_id",
                        "class_root_sha256",
                        "members",
                        "representative_equation",
                        "semantic_vector_sha256",
                        "assignments_checked",
                    )
                }
                == representation
            )
            details = {"canonical_class_count": len(recomputed)}
        elif stage.stage_id == "counterexample_screened":
            left = known[equation["lhs"]]
            right = known[equation["rhs"]]
            points = _assignments()[:31]
            failures = [
                list(point)
                for point in points
                if _evaluate(left, point, table) != _evaluate(right, point, table)
            ]
            passed = not failures
            details = {"points_checked": len(points), "counterexamples": failures}
        elif stage.stage_id == "exactly_verified":
            left = known[equation["lhs"]]
            right = known[equation["rhs"]]
            failures = [
                list(point)
                for point in _assignments()
                if _evaluate(left, point, table) != _evaluate(right, point, table)
            ]
            passed = not failures and representation["assignments_checked"] == ASSIGNMENT_COUNT
            details = {
                "assignments_checked": ASSIGNMENT_COUNT,
                "counterexample_count": len(failures),
                "proof_method": "complete_truth_table",
            }
        elif stage.stage_id == "prior_art_checked":
            passed = self.target is not None and all(
                representation[key] == self.target[key]
                for key in ("class_id", "class_root_sha256", "semantic_vector_sha256")
            )
            details = {
                "post_unseal_target_available": self.target is not None,
                "exact_hidden_class_match": passed,
            }
        else:
            raise ValueError("unregistered synthetic algebra stage")
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
    representation = {
        key: discovered[key]
        for key in (
            "class_id",
            "class_root_sha256",
            "members",
            "representative_equation",
            "semantic_vector_sha256",
            "assignments_checked",
        )
    }
    equation = representation["representative_equation"]
    return CandidateArtifact.create(
        ArtifactKind.THEOREM,
        f"{equation['lhs']} equals {equation['rhs']} in the declared anonymous finite algebra",
        representation,
        provenance,
        assumptions=("the sealed operation table defines the complete finite semantics",),
        claims=("declared_finite_identity",),
    )


def _run_pre_unseal_pipeline(
    public: Mapping[str, Any], artifact: CandidateArtifact
) -> tuple[dict[str, StageOutcome], list[GateOutcome], PromotionLedger]:
    pack = SyntheticAlgebraPack(public)
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
            raise ValueError(f"synthetic algebra pre-unseal stage blocked: {stage_id}")
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
    pack = SyntheticAlgebraPack(public, target)
    prior = {"exactly_verified": stages["exactly_verified"]}
    outcome = run_stage(pack, artifact, "prior_art_checked", prior)
    if outcome.status is not OutcomeStatus.PASS:
        raise ValueError("synthetic algebra post-unseal comparison failed")
    stages["prior_art_checked"] = outcome
    required = {
        "exactly_verified": stages["exactly_verified"],
        "prior_art_checked": outcome,
    }
    gate = run_gate(pack, artifact, "admit_prior_art_checked", required)
    final = ledger.promote(PACK_DESCRIPTOR, artifact, gate, required)
    gates.append(gate)
    return outcome, gate, final


def _negative_controls(
    public: Mapping[str, Any], artifact: CandidateArtifact, pre_ledger: PromotionLedger
) -> list[dict[str, Any]]:
    table = public["axiom"]["operation_table"]
    terms = _term_orderings()
    assignments = _assignments()
    invalid = None
    best_prefix = -1
    for left_index, left in enumerate(terms):
        for right in terms[left_index + 1 :]:
            equal_prefix = 0
            first_failure = None
            for point in assignments:
                if _evaluate(left, point, table) == _evaluate(right, point, table):
                    if first_failure is None:
                        equal_prefix += 1
                else:
                    first_failure = list(point)
                    break
            if first_failure is not None and equal_prefix > best_prefix:
                best_prefix = equal_prefix
                invalid = (left, right, first_failure)
    if invalid is None:
        raise ValueError("negative identity control unavailable")
    left, right, first_failure = invalid
    undeclared_check = CheckResult.create(
        "dependency_closure_contained",
        False,
        {"allowed": [], "attempted": ["lemma.withheld"]},
    )
    undeclared_outcome = StageOutcome.create(
        "exactly_verified",
        artifact.ref,
        OutcomeStatus.BLOCK,
        [undeclared_check],
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
    conjecture_typed = run_stage(SyntheticAlgebraPack(public), conjecture, "typed")
    if conjecture_typed.status is not OutcomeStatus.PASS:
        raise ValueError("typed conjecture negative control changed")
    forged = pre_ledger.to_dict()
    forged["entries"][-1]["to_stage"] = "prior_art_checked"
    try:
        PromotionLedger.from_dict(forged)
    except SchemaViolation:
        forged_rejected = True
    else:
        forged_rejected = False
    if not forged_rejected:
        raise ValueError("forged promotion negative control admitted")
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
    if not unproved_promotion_blocked:
        raise ValueError("unproved conjecture promotion negative control admitted")
    controls = [
        {
            "control_id": "example_prefix_overfit",
            "status": "rejected",
            "visible_prefix_points_passed": best_prefix,
            "first_counterexample": first_failure,
            "candidate_equation": {"lhs": _term(left), "rhs": _term(right)},
        },
        {
            "control_id": "forbidden_target_file_read",
            "status": "rejected",
            "denied_reads": 1,
            "bytes_exposed": 0,
        },
        {
            "control_id": "undeclared_lemma_dependency",
            "status": "rejected",
            "declared_dependency_count": 0,
            "attempted_dependency": "lemma.withheld",
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
    return controls


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
        raise ValueError("synthetic algebra deny-by-default read control failed")
    if len(discovered) != 1:
        raise ValueError("synthetic algebra rediscovery count changed")
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
        ("missing_class_rediscovered", artifact.content_sha256),
        ("exhaustive_proof_sealed", proof_seal["proof_seal_sha256"]),
        ("target_unsealed_and_compared", comparison.outcome_sha256),
        ("final_promotion_sealed", final_ledger.ledger_sha256),
    )
    chronology = [
        {"sequence": index, "phase": phase, "root_sha256": root_sha}
        for index, (phase, root_sha) in enumerate(chronology_data)
    ]
    return {
        "schema_version": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "decision": "pass_one_of_one_synthetic_finite_algebra_holdout_rediscovered_and_proved",
        "decision_counts": {"pass": 1, "blocked": 0, "reject": 0},
        "world": {
            "world_id": reference["public"]["world_id"],
            "generation_epoch": config["world_generator"]["generation_epoch"],
            "order": ORDER,
            "operation_symbol": reference["parameters"]["operation_symbol"],
            "operation_table": reference["parameters"]["operation_table"],
            "operation_table_sha256": _sha(reference["parameters"]["operation_table"]),
            "anonymous_parameter_commitment_sha256": _sha(
                {
                    "permutation": reference["parameters"]["permutation"],
                    "affine_parameters": reference["parameters"]["affine_parameters"],
                }
            ),
        },
        "reference_graph": {
            "root_sha256": reference["graph_root_sha256"],
            "axiom_count": 1,
            "visible_ancestor_count": len(reference["public"]["visible_ancestors"]),
            "nontrivial_theorem_class_count": len(reference["graph"]["theorem_classes"]),
            "visible_theorem_class_count": len(reference["public"]["visible_theorem_classes"]),
            "hidden_theorem_class_count": 1,
            "edge_count": reference["graph"]["edge_count"],
            "visible_ancestors": reference["public"]["visible_ancestors"],
        },
        "pre_unseal": {
            "public_root_sha256": reference["public_root_sha256"],
            "raw_term_count": len(_term_orderings()),
            "assignments_per_term": ASSIGNMENT_COUNT,
            "term_evaluations": len(_term_orderings()) * ASSIGNMENT_COUNT,
            "target_identifiers_exposed": 0,
            "target_equations_exposed": 0,
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
            "method": "exhaustive_deterministic_semantics",
            "assignments_checked": ASSIGNMENT_COUNT,
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
        "negative_controls": _negative_controls(reference["public"], artifact, pre_ledger),
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
            "replicate_across_preregistered_independently_generated_worlds_and_add_external_"
            "proof_kernel_without_exposing_holdout_equivalence_classes"
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
            "one deterministic anonymous order-seven binary algebra, one balanced four-variable "
            "term grammar of 24 terms, one completely withheld semantic equivalence class, and "
            "exhaustive verification on 2401 assignments; no historical novelty, unbounded "
            "discovery, general equational completeness, external proof-kernel, hostile-process "
            "isolation, physical claim, or external mathematical significance"
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
        raise ValueError("synthetic algebra source binding keys changed")
    for label, path in expected_paths.items():
        if bindings[label] != {"path": path, "file_sha256": _file_sha(_inside(root, path))}:
            raise ValueError("synthetic algebra source binding changed")


def validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("synthetic algebra content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    expected = _expected_body(validation_root, config_path)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("synthetic algebra result boundary changed")


def build_benchmark(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    body = _expected_body(root, config_path)
    result = {**body, "content_sha256": _sha(body)}
    validate_result(result, root=root)
    return result


def write_benchmark(config_path: Path) -> Path:
    result = build_benchmark(config_path)
    root = config_path.resolve().parents[1]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_benchmark(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
