"""Fail-closed orchestration from Formula Discovery Job v1 to evaluation and Pareto."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .candidate_evaluation_ladder import (
    EvaluationLadder,
    EvaluationPhase,
    EvaluationStep,
    evaluate_candidate,
    validate_evaluation_replay,
)
from .candidate_pareto_explanations import (
    MetricReceipt,
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_replay,
)
from .formula_discovery_job import (
    RESULT_SCHEMA as DISCOVERY_RESULT_SCHEMA,
)
from .formula_discovery_job import (
    run_formula_discovery_job,
    validate_formula_discovery_result,
)
from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    ProvenanceRecord,
    SchemaViolation,
    StageDefinition,
    StageOutcome,
    canonical_sha256,
)

BATCH_SCHEMA = "sigma-formula-discovery-orchestration-1.0"
WRAPPER_SCHEMA = "sigma-formula-discovery-orchestrated-candidate-1.0"
MAXIMUM_JOBS = 32
REQUIRED_GATES = ("hard_structure", "hard_validation")
METRIC_DIRECTIONS = {"coefficient_count": "minimize", "expression_bytes": "minimize"}
CLAIMS = {
    "discovery_pass_is_scientific_truth": False,
    "failed_hard_gate_compensated_by_metric": False,
    "novelty_established": False,
    "promotion_authorized": False,
}

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_BATCH_KEYS = {
    "batch_id",
    "candidates",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "domain_pack",
    "evaluations",
    "jobs",
    "ladder",
    "metric_receipts",
    "pareto",
    "problem_set_sha256",
    "schema_version",
    "scope",
}
_WRAPPER_KEYS = {
    "discovery_result",
    "job_id",
    "problem",
    "source_candidate",
    "wrapper_schema",
}


class FormulaDiscoveryOrchestrationError(ValueError):
    """An orchestration input or sealed report crossed a closed boundary."""


def _descriptor() -> DomainPackDescriptor:
    kinds = (ArtifactKind.FORMULA,)
    return DomainPackDescriptor(
        "formula.discovery.orchestration",
        "1.0.0",
        kinds,
        (
            StageDefinition("generated", 0, kinds),
            StageDefinition("independently_validated", 1, kinds, ("generated",)),
        ),
        (
            GateDefinition("hard_structure", None, "generated", kinds, ("generated",)),
            GateDefinition(
                "hard_validation",
                "generated",
                "independently_validated",
                kinds,
                ("generated", "independently_validated"),
            ),
        ),
    )


def _ladder() -> EvaluationLadder:
    return EvaluationLadder.create(
        _descriptor(),
        (
            EvaluationStep("generated", "hard_structure", EvaluationPhase.CHEAP),
            EvaluationStep("independently_validated", "hard_validation", EvaluationPhase.FORMAL),
        ),
    )


def _embedded(artifact: CandidateArtifact) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    representation = artifact.representation
    if (
        set(representation) != _WRAPPER_KEYS
        or representation.get("wrapper_schema") != WRAPPER_SCHEMA
    ):
        raise FormulaDiscoveryOrchestrationError("wrapper candidate schema changed")
    problem, result = representation["problem"], representation["discovery_result"]
    if not isinstance(problem, Mapping) or not isinstance(result, Mapping):
        raise FormulaDiscoveryOrchestrationError("embedded job values must be objects")
    validate_formula_discovery_result(result, problem)
    source = CandidateArtifact.from_dict(result["candidate"])
    if (
        representation["source_candidate"] != source.ref.to_dict()
        or artifact.provenance.inputs != (source.ref,)
        or representation["job_id"] != result["job_id"]
    ):
        raise FormulaDiscoveryOrchestrationError("wrapper source binding changed")
    return problem, result


class _FormulaJobPack:
    @property
    def descriptor(self) -> DomainPackDescriptor:
        return _descriptor()

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        _, result = _embedded(artifact)
        valid = (
            result.get("schema_version") == DISCOVERY_RESULT_SCHEMA
            and result.get("candidate") is not None
            and result.get("decision") in {"PASS", "REJECT"}
        )
        check = CheckResult.create(
            f"{stage.stage_id}.formula_job_binding",
            valid,
            {
                "artifact": artifact.ref.to_dict(),
                "discovery_result_sha256": result.get("content_sha256"),
                "prior_stage_ids": sorted(prior_outcomes),
            },
        )
        return StageOutcome.create(
            stage.stage_id,
            artifact.ref,
            OutcomeStatus.PASS if valid else OutcomeStatus.BLOCK,
            (check,),
            reason_codes=() if valid else ("invalid_discovery_job_binding",),
        )

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        _, result = _embedded(artifact)
        if gate.gate_id == "hard_structure":
            status = OutcomeStatus.PASS
            passed = True
            reasons: tuple[str, ...] = ()
        else:
            status = OutcomeStatus.PASS if result["decision"] == "PASS" else OutcomeStatus.REJECT
            passed = status is OutcomeStatus.PASS
            reasons = () if passed else ("formula_job_validation_rejected",)
        check = CheckResult.create(
            f"{gate.gate_id}.formula_job_decision",
            passed,
            {
                "decision": result["decision"],
                "discovery_result_sha256": result["content_sha256"],
            },
        )
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            status,
            tuple(stage_outcomes[key].ref for key in sorted(stage_outcomes)),
            (check,),
            reason_codes=reasons,
        )


def _wrapper(problem: Mapping[str, Any], result: Mapping[str, Any]) -> CandidateArtifact:
    source = CandidateArtifact.from_dict(result["candidate"])
    representation = {
        "wrapper_schema": WRAPPER_SCHEMA,
        "job_id": result["job_id"],
        "problem": dict(problem),
        "discovery_result": dict(result),
        "source_candidate": source.ref.to_dict(),
    }
    provenance = ProvenanceRecord.create(
        _descriptor().ref,
        {
            "adapter": BATCH_SCHEMA,
            "discovery_result_sha256": result["content_sha256"],
            "problem_sha256": result["problem_sha256"],
        },
        inputs=(source.ref,),
    )
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"orchestrated formula-discovery candidate for {result['job_id']}",
        representation,
        provenance,
        assumptions=("caller-supplied exact formula discovery job",),
        claims=("orchestrated_candidate",),
    )


def _metrics(candidate: CandidateArtifact) -> tuple[MetricReceipt, MetricReceipt]:
    _, result = _embedded(candidate)
    coefficients = result["synthesis"]["coefficients"]
    expression = result["candidate"]["representation"]["expression"]
    values = {
        "coefficient_count": len(coefficients),
        "expression_bytes": len(expression.encode("utf-8")),
    }
    return tuple(
        MetricReceipt.create(
            candidate.ref,
            metric_id,
            direction,
            values[metric_id],
            canonical_sha256(
                {
                    "candidate": candidate.ref.to_dict(),
                    "discovery_result_sha256": result["content_sha256"],
                    "metric_id": metric_id,
                    "value": values[metric_id],
                }
            ),
        )
        for metric_id, direction in sorted(METRIC_DIRECTIONS.items())
    )  # type: ignore[return-value]


def build_formula_discovery_orchestration(
    problems: Sequence[Mapping[str, Any]], *, batch_id: str = "formula.discovery.batch"
) -> dict[str, Any]:
    """Run, evaluate, and hard-gate-rank a bounded set of caller jobs."""

    if not isinstance(batch_id, str) or _IDENTIFIER.fullmatch(batch_id) is None:
        raise FormulaDiscoveryOrchestrationError("batch_id is not canonical")
    problem_rows = tuple(problems)
    if not 1 <= len(problem_rows) <= MAXIMUM_JOBS or any(
        not isinstance(problem, Mapping) for problem in problem_rows
    ):
        raise FormulaDiscoveryOrchestrationError("job batch size or type is invalid")
    results = [(problem, run_formula_discovery_job(problem)) for problem in problem_rows]
    job_ids = [result["job_id"] for _, result in results]
    if len(set(job_ids)) != len(job_ids) or "unbound" in job_ids:
        raise FormulaDiscoveryOrchestrationError("job IDs must be unique and bound")
    results.sort(key=lambda row: row[1]["job_id"])
    jobs = [
        {
            "job_id": result["job_id"],
            "problem_sha256": result["problem_sha256"],
            "result": result,
        }
        for _, result in results
    ]
    candidates = tuple(
        _wrapper(problem, result) for problem, result in results if result["candidate"] is not None
    )
    if not candidates:
        raise FormulaDiscoveryOrchestrationError("no caller job emitted a candidate")
    candidates = tuple(sorted(candidates, key=lambda item: item.artifact_id))
    pack, ladder = _FormulaJobPack(), _ladder()
    evaluation_pairs = []
    outcomes = []
    receipts = []
    for candidate in candidates:
        evaluation = evaluate_candidate(pack, candidate, ladder)
        validate_evaluation_replay(evaluation, pack, candidate)
        evaluation_pairs.append((candidate.artifact_id, evaluation))
        outcomes.extend(GateOutcome.from_dict(row) for row in evaluation["gate_outcomes"])
        receipts.extend(_metrics(candidate))
    limits = ParetoLimits(MAXIMUM_JOBS, len(REQUIRED_GATES), len(METRIC_DIRECTIONS), 4096)
    pareto = build_pareto_explanations(
        candidates,
        outcomes,
        receipts,
        required_gate_ids=REQUIRED_GATES,
        metric_directions=METRIC_DIRECTIONS,
        limits=limits,
    )
    validate_pareto_replay(
        pareto,
        candidates,
        outcomes,
        receipts,
        required_gate_ids=REQUIRED_GATES,
        metric_directions=METRIC_DIRECTIONS,
        limits=limits,
    )
    decisions = Counter(result["decision"] for _, result in results)
    statuses = Counter(evaluation["status"] for _, evaluation in evaluation_pairs)
    body = {
        "schema_version": BATCH_SCHEMA,
        "batch_id": batch_id,
        "problem_set_sha256": canonical_sha256(
            [problem for problem, _ in sorted(results, key=lambda row: row[1]["job_id"])]
        ),
        "jobs": jobs,
        "domain_pack": _descriptor().to_dict(),
        "ladder": ladder.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "evaluations": [evaluation for _, evaluation in sorted(evaluation_pairs)],
        "metric_receipts": [receipt.to_dict() for receipt in receipts],
        "pareto": pareto,
        "counts": {
            "caller_jobs": len(results),
            "job_passes": decisions["PASS"],
            "job_rejects": decisions["REJECT"],
            "job_blocks": decisions["BLOCK"],
            "candidates": len(candidates),
            "evaluation_passes": statuses["pass"],
            "evaluation_rejects": statuses["reject"],
            "evaluation_blocks": statuses["block"],
            "hard_gate_eligible": pareto["counts"]["hard_gate_eligible"],
            "pareto_fronts": pareto["counts"]["pareto_fronts"],
        },
        "decision": "completed_formula_jobs_then_hard_gates_then_exact_pareto",
        "claims": dict(CLAIMS),
        "scope": (
            "domain-neutral orchestration of caller-supplied Formula Discovery Job v1 records; "
            "only candidates passing every registered hard gate enter exact Pareto fronts"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_formula_discovery_orchestration(
    report: Mapping[str, Any], problems: Sequence[Mapping[str, Any]]
) -> None:
    """Validate the closed report and require deterministic end-to-end replay."""

    if set(report) != _BATCH_KEYS or report.get("schema_version") != BATCH_SCHEMA:
        raise FormulaDiscoveryOrchestrationError("orchestration report schema changed")
    body = {key: value for key, value in report.items() if key != "content_sha256"}
    if report.get("content_sha256") != canonical_sha256(body):
        raise FormulaDiscoveryOrchestrationError("orchestration report seal changed")
    try:
        for candidate in report["candidates"]:
            CandidateArtifact.from_dict(candidate).validate()
    except (SchemaViolation, TypeError, ValueError) as error:
        raise FormulaDiscoveryOrchestrationError("embedded Sigma candidate changed") from error
    if dict(report) != build_formula_discovery_orchestration(problems, batch_id=report["batch_id"]):
        raise FormulaDiscoveryOrchestrationError("orchestration exact replay changed")


__all__ = [
    "BATCH_SCHEMA",
    "CLAIMS",
    "MAXIMUM_JOBS",
    "METRIC_DIRECTIONS",
    "REQUIRED_GATES",
    "FormulaDiscoveryOrchestrationError",
    "build_formula_discovery_orchestration",
    "validate_formula_discovery_orchestration",
]
