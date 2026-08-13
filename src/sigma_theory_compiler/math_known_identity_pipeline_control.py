"""Sealed end-to-end control for Math Pack evaluation and Pareto admission.

This is a known-answer mechanics control, not a blind rediscovery benchmark. The correct identity
and an intentionally wrong formula traverse the real MathDomainPack and batch pipeline; only the
formally admitted identity receives a metric receipt and Pareto front.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .candidate_discovery_pipeline import (
    DiscoveryPipelineLimits,
    run_candidate_discovery_pipeline,
    validate_candidate_discovery_replay,
    validate_candidate_discovery_result,
)
from .candidate_evaluation_ladder import EvaluationLadder, EvaluationPhase, EvaluationStep
from .candidate_pareto_explanations import MetricReceipt, ParetoLimits
from .math_expression_ir import Equation, symbol
from .math_pack import MathDomainPack, math_candidate_representation, math_pack_descriptor
from .math_types import IntegerType
from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    ProvenanceRecord,
    SourceBinding,
    canonical_json_bytes,
    canonical_sha256,
)

SCHEMA_VERSION = "sigma-math-known-identity-pipeline-control-1.0"
CONFIG_PATH = "configs/math_known_identity_pipeline_control.json"
SOURCE_PATH = "src/sigma_theory_compiler/math_known_identity_pipeline_control.py"
TEST_PATH = "tests/test_math_known_identity_pipeline_control.py"
PRIOR_ART_PATH = "runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json"
OUTPUT_PATH = "runs/math-language/math-known-identity-pipeline-control/campaign.json"

_CLAIMS = {
    "known_answer_control_passed": True,
    "wrong_formula_rejected_by_counterexample": True,
    "soft_metric_admitted_only_after_all_hard_gates": True,
    "blind_rediscovery_proved": False,
    "general_formula_discovery_proved": False,
    "novelty_established": False,
    "promotion_authorized": False,
}


class KnownIdentityControlError(ValueError):
    """The control or one of its immutable bindings changed."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KnownIdentityControlError(f"{path} must contain a JSON object")
    return value


def _ladder() -> EvaluationLadder:
    phases = {
        "typed": EvaluationPhase.CHEAP,
        "canonicalized": EvaluationPhase.CHEAP,
        "counterexample_screened": EvaluationPhase.SYMBOLIC,
        "exactly_verified": EvaluationPhase.FORMAL,
        "prior_art_checked": EvaluationPhase.FORMAL,
    }
    descriptor = math_pack_descriptor()
    return EvaluationLadder.create(
        descriptor,
        tuple(
            EvaluationStep(stage.stage_id, f"admit_{stage.stage_id}", phases[stage.stage_id])
            for stage in descriptor.stages
        ),
    )


def _candidate(
    formula: Equation,
    label: str,
    prior_art_sha256: str,
) -> CandidateArtifact:
    descriptor = math_pack_descriptor()
    return CandidateArtifact.create(
        ArtifactKind.IDENTITY,
        f"Known-answer pipeline control: {label}.",
        math_candidate_representation(
            formula,
            {"n": IntegerType(0, 100)},
            exact_assignments=({"n": 0}, {"n": 1}, {"n": 2}, {"n": 10}),
            random_trials=32,
            adversarial_limit=16,
            seed=20260812,
            prior_art_receipt_sha256=prior_art_sha256,
        ),
        ProvenanceRecord.create(
            descriptor.ref,
            {"control": "known_identity_pipeline", "label": label},
            sources=(SourceBinding("prior_art_receipt", PRIOR_ART_PATH, prior_art_sha256),),
        ),
        assumptions=("n is an integer in the closed interval [0,100]",),
        claims=("known_answer_pipeline_control",),
    )


def _inputs(prior_art_sha256: str) -> tuple[CandidateArtifact, CandidateArtifact]:
    n = symbol("n")
    correct = _candidate(
        Equation(n * (n + 1) / 2, (n**2 + n) / 2),
        "algebraically identical natural-sum forms",
        prior_art_sha256,
    )
    wrong = _candidate(
        Equation(n * (n + 1) / 2, n**2),
        "deliberately wrong square formula",
        prior_art_sha256,
    )
    return correct, wrong


def _metric_receipt(candidate: CandidateArtifact, metric: dict[str, Any]) -> MetricReceipt:
    value = len(canonical_json_bytes(candidate.representation["formula"]))
    return MetricReceipt.create(
        candidate.ref,
        metric["metric_id"],
        metric["direction"],
        Fraction(value),
        canonical_sha256(
            {
                "method": "canonical_math_ir_formula_byte_length",
                "candidate": candidate.artifact_id,
                "value": value,
            }
        ),
    )


def _validate_pipeline_boundary(
    pipeline: dict[str, Any], correct: CandidateArtifact, wrong: CandidateArtifact
) -> None:
    rows = {row["candidate"]["artifact_id"]: row for row in pipeline["candidate_rows"]}
    evaluations = {row["artifact"]["artifact_id"]: row for row in pipeline["evaluations"]}
    correct_row = rows[correct.artifact_id]
    wrong_row = rows[wrong.artifact_id]
    correct_evaluation = evaluations[correct.artifact_id]
    wrong_evaluation = evaluations[wrong.artifact_id]
    if not (
        correct_row["status"] == "pass"
        and correct_row["all_required_gates_passed"] is True
        and correct_row["pareto_front"] == 1
        and wrong_row["status"] == "reject"
        and wrong_row["all_required_gates_passed"] is False
        and wrong_row["pareto_front"] is None
    ):
        raise KnownIdentityControlError("control candidate outcomes changed")
    if not (
        correct_evaluation["terminal"]["kind"] == "complete"
        and correct_evaluation["counts"]["attempted_by_phase"]
        == {"cheap": 2, "symbolic": 1, "formal": 2, "observational": 0}
        and wrong_evaluation["terminal"]
        == {"kind": "stage", "outcome_id": "counterexample_screened"}
        and wrong_evaluation["counts"]["attempted_by_phase"]
        == {"cheap": 2, "symbolic": 1, "formal": 0, "observational": 0}
        and wrong_evaluation["skipped_steps"] == ["exactly_verified", "prior_art_checked"]
    ):
        raise KnownIdentityControlError("control phase boundary changed")
    if pipeline["counts"] != {
        "candidates": 2,
        "evaluation_statuses": {"pass": 1, "block": 0, "reject": 1, "error": 0},
        "all_required_gates_passed": 1,
        "ranked_candidates": 1,
        "unranked_candidates": 1,
        "observational_phase_opened": 0,
        "metric_receipts": 1,
        "pareto_fronts": 1,
    }:
        raise KnownIdentityControlError("control aggregate counts changed")


def _build(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_config_keys = {
        "schema_version",
        "control_id",
        "source_bindings",
        "limits",
        "metric",
        "claims",
    }
    if set(config) != expected_config_keys or config["schema_version"] != SCHEMA_VERSION:
        raise KnownIdentityControlError("control config schema changed")
    bindings = config["source_bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {"source", "test", "prior_art"}:
        raise KnownIdentityControlError("control source bindings changed")
    for role, path in (("source", SOURCE_PATH), ("test", TEST_PATH), ("prior_art", PRIOR_ART_PATH)):
        row = bindings[role]
        if set(row) != {"path", "file_sha256"} or row["path"] != path:
            raise KnownIdentityControlError(f"{role} binding schema changed")
        if _sha_file(root / path) != row["file_sha256"]:
            raise KnownIdentityControlError(f"{role} file hash changed")
    limits = config["limits"]
    if limits != {"maximum_candidates": 2, "maximum_work_units": 4}:
        raise KnownIdentityControlError("control limits changed")
    metric = config["metric"]
    if metric != {"metric_id": "canonical_formula_bytes", "direction": "minimize"}:
        raise KnownIdentityControlError("control metric changed")
    if config["claims"] != _CLAIMS:
        raise KnownIdentityControlError("control claims changed")
    prior_art_sha256 = bindings["prior_art"]["file_sha256"]
    correct, wrong = _inputs(prior_art_sha256)
    receipt = _metric_receipt(correct, metric)
    pipeline = run_candidate_discovery_pipeline(
        MathDomainPack(),
        (correct, wrong),
        _ladder(),
        (receipt,),
        metric_directions={metric["metric_id"]: metric["direction"]},
        limits=DiscoveryPipelineLimits(maximum_candidates=limits["maximum_candidates"]),
        pareto_limits=ParetoLimits(
            maximum_candidates=2,
            maximum_hard_gates=5,
            maximum_metrics=1,
            maximum_work_units=limits["maximum_work_units"],
        ),
    )
    _validate_pipeline_boundary(pipeline, correct, wrong)
    rows = {row["candidate"]["artifact_id"]: row for row in pipeline["candidate_rows"]}
    body = {
        "schema_version": SCHEMA_VERSION,
        "control_id": config["control_id"],
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _sha_file(root / CONFIG_PATH)},
            **bindings,
        },
        "candidate_roles": {
            "correct": correct.ref.to_dict(),
            "wrong": wrong.ref.to_dict(),
        },
        "pipeline_result": pipeline,
        "exact_counts": {
            "candidates": 2,
            "hard_gate_pass": 1,
            "counterexample_reject": 1,
            "formal_proofs": 1,
            "prior_art_checks_after_proof": 1,
            "metric_receipts": 1,
            "ranked_candidates": 1,
            "pareto_fronts": 1,
            "promotion_authorized": 0,
        },
        "correct_candidate": {
            "status": rows[correct.artifact_id]["status"],
            "pareto_front": rows[correct.artifact_id]["pareto_front"],
            "all_required_gates_passed": rows[correct.artifact_id]["all_required_gates_passed"],
        },
        "wrong_candidate": {
            "status": rows[wrong.artifact_id]["status"],
            "pareto_front": rows[wrong.artifact_id]["pareto_front"],
            "exclusion_reason": rows[wrong.artifact_id]["exclusion_reason"],
            "formal_phase_attempted": False,
        },
        "decision": "pass_known_identity_and_reject_wrong_formula_end_to_end_control",
        "claims": dict(_CLAIMS),
        "scope": (
            "known-answer mechanics control over one exact rational identity and one deliberately "
            "wrong formula; not blind rediscovery, general proof capability, novelty, or promotion"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def build_result(root: Path) -> dict[str, Any]:
    return _build(root, _read_json(root / CONFIG_PATH))


def validate_result(value: dict[str, Any], root: Path) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "control_id",
        "source_bindings",
        "candidate_roles",
        "pipeline_result",
        "exact_counts",
        "correct_candidate",
        "wrong_candidate",
        "decision",
        "claims",
        "scope",
        "content_sha256",
    }:
        raise KnownIdentityControlError("control result schema changed")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != canonical_sha256(unsigned):
        raise KnownIdentityControlError("control result content hash changed")
    validate_candidate_discovery_result(value["pipeline_result"])
    config = _read_json(root / CONFIG_PATH)
    prior_art_sha256 = config["source_bindings"]["prior_art"]["file_sha256"]
    correct, wrong = _inputs(prior_art_sha256)
    metric = config["metric"]
    receipt = _metric_receipt(correct, metric)
    validate_candidate_discovery_replay(
        value["pipeline_result"], MathDomainPack(), (correct, wrong), (receipt,)
    )
    if value != _build(root, config):
        raise KnownIdentityControlError("control result immutable replay changed")


def write_result(root: Path) -> Path:
    path = root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_result(root)
    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return path


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "KnownIdentityControlError",
    "build_result",
    "validate_result",
    "write_result",
]
