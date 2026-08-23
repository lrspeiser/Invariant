"""Build a candidate-bound exact -> CAS -> SMT -> interval -> Lean release ladder.

The stored receipt calibrates the ladder on externally authored known controls and proves that
bounded-unknown candidates remain blocked.  It never releases a mathematical or novelty claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp
import z3

from .external_creativity_multi_host import validate_receipt as validate_multi_host_receipt
from .external_creativity_validation import (
    PUBLIC_CONFIG_PATH,
    Benchmark,
    Candidate,
    SealedTarget,
    independently_predict,
    load_public_benchmarks,
    unseal_targets,
    verify_known_formula,
)
from .external_creativity_validation import (
    RECEIPT_SCHEMA as CAMPAIGN_SCHEMA,
)
from .external_creativity_validation import (
    _interval_value as interval_value,
)
from .external_creativity_validation import (
    _safe_expression as safe_expression,
)
from .external_creativity_validation import (
    _sympy_to_z3 as sympy_to_z3,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/serious_claim_verification_ladder.json"
OUTPUT_PATH = "runs/math/serious-claim-verification-ladder/receipt.json"
CONFIG_SCHEMA = "invariant-serious-claim-verification-ladder-config-1.1"
SCHEMA_VERSION = "invariant-serious-claim-verification-ladder-1.1"
CHAIN_SCHEMA = "invariant-candidate-verification-chain-1.1"
STAGE_SCHEMA = "invariant-candidate-verification-stage-1.1"
REQUIRED_STAGES = ("exact_arithmetic", "cas", "smt", "interval", "lean")
MATHEMATICAL_MUTATION_BACKENDS = REQUIRED_STAGES[:-1]
MATHEMATICAL_MUTATION_OPERATOR = "add_exact_unit"
LEAN_MUTATION_STATUS = "PENDING_SEPARATE_CI_ARTIFACT"
MUTATIONS = (
    "missing_stage",
    "reordered_stages",
    "candidate_scope_substitution",
    "broken_previous_stage_link",
    "backend_unavailable",
)
_HEX = frozenset("0123456789abcdef")


class SeriousClaimVerificationError(ValueError):
    """The candidate verification ladder failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SeriousClaimVerificationError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SeriousClaimVerificationError(f"{label} is not a SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SeriousClaimVerificationError(f"could not read {label}") from error
    if not isinstance(value, dict):
        raise SeriousClaimVerificationError(f"{label} is not an object")
    return value


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise SeriousClaimVerificationError(f"{label} escapes the project root") from error
    if not path.is_file():
        raise SeriousClaimVerificationError(f"{label} is missing")
    return path


def _validate_seal(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SeriousClaimVerificationError(f"{label} content seal changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    value = _read_json(_under(root, CONFIG_PATH, "ladder config"), "ladder config")
    _strict(
        value,
        {
            "known_control_suite",
            "ladder_id",
            "mathematical_wrong_formula_control",
            "mutation_controls",
            "negative_control_benchmark_ids",
            "release_policy",
            "required_stage_order",
            "schema_version",
            "sources",
        },
        "ladder config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise SeriousClaimVerificationError("ladder config schema changed")
    if tuple(value["required_stage_order"]) != REQUIRED_STAGES:
        raise SeriousClaimVerificationError("serious-claim backend order changed")
    if tuple(value["mutation_controls"]) != MUTATIONS:
        raise SeriousClaimVerificationError("ladder mutation controls changed")
    _strict(
        value["sources"],
        {
            "campaign_receipt",
            "lean_source",
            "multi_host_receipt",
            "public_benchmarks",
            "sealed_targets",
        },
        "ladder sources",
    )
    if value["sources"]["public_benchmarks"] != PUBLIC_CONFIG_PATH:
        raise SeriousClaimVerificationError("public benchmark source changed")
    _strict(
        value["known_control_suite"],
        {"benchmark_ids", "lean_target", "lean_theorems"},
        "known-control suite",
    )
    if len(value["known_control_suite"]["benchmark_ids"]) < 2:
        raise SeriousClaimVerificationError("too few known controls")
    mathematical_control = value["mathematical_wrong_formula_control"]
    _strict(
        mathematical_control,
        {"lean_status", "operator", "required_backends"},
        "mathematical wrong-formula control",
    )
    if (
        mathematical_control["operator"] != MATHEMATICAL_MUTATION_OPERATOR
        or tuple(mathematical_control["required_backends"]) != MATHEMATICAL_MUTATION_BACKENDS
        or mathematical_control["lean_status"] != LEAN_MUTATION_STATUS
    ):
        raise SeriousClaimVerificationError("mathematical wrong-formula policy changed")
    policy = value["release_policy"]
    required_policy = {
        "backend_unavailable_is_block",
        "backend_wrong_formula_mutation_required",
        "candidate_identity_bound_at_every_stage",
        "independent_reproduction_required",
        "known_control_calibration_can_release_serious_claim",
        "lean_wrong_formula_artifact_required",
        "named_human_prior_art_review_required",
        "positive_result_required_at_every_stage",
        "previous_stage_hash_bound",
    }
    _strict(policy, required_policy, "ladder release policy")
    if (
        any(
            policy[key] is not True
            for key in required_policy - {"known_control_calibration_can_release_serious_claim"}
        )
        or policy["known_control_calibration_can_release_serious_claim"] is not False
    ):
        raise SeriousClaimVerificationError("ladder release policy weakened")
    return value


def _campaign(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(
        _under(root, config["sources"]["campaign_receipt"], "campaign receipt"),
        "campaign receipt",
    )
    _validate_seal(value, "campaign receipt")
    if value.get("schema_version") != CAMPAIGN_SCHEMA:
        raise SeriousClaimVerificationError("campaign receipt schema changed")
    policy = value.get("serious_claim_policy", {})
    if tuple(policy.get("required_backends", ())) != REQUIRED_STAGES:
        raise SeriousClaimVerificationError("campaign backend policy changed")
    if policy.get("released_claims") != 0:
        raise SeriousClaimVerificationError("campaign unexpectedly released a serious claim")
    return value


def _multi_host(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(
        _under(root, config["sources"]["multi_host_receipt"], "multi-host receipt"),
        "multi-host receipt",
    )
    validate_multi_host_receipt(value)
    return value


def _candidate_bindings(
    benchmarks: Mapping[str, Mapping[str, Any]], benchmark_ids: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for benchmark_id in benchmark_ids:
        benchmark = benchmarks.get(benchmark_id)
        if benchmark is None or benchmark.get("target_kind") != "known_formula":
            raise SeriousClaimVerificationError("known-control benchmark is missing or changed")
        candidates = benchmark.get("ranked_candidates", [])
        if not candidates:
            raise SeriousClaimVerificationError("known-control candidate is missing")
        candidate = candidates[0]
        if candidate.get("train_loss") != "0" or candidate.get("holdout_loss") != "0":
            raise SeriousClaimVerificationError("known-control best candidate is not exact")
        result.append(
            {
                "behavior_sha256": candidate["behavior_sha256"],
                "benchmark_id": benchmark_id,
                "candidate_id": candidate["candidate_id"],
                "expression_sha256": canonical_sha256({"expression": candidate["expression"]}),
                "target_commitment_sha256": benchmark["target_commitment_opened"],
            }
        )
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _candidate_from_record(record: Mapping[str, Any]) -> Candidate:
    try:
        candidate = Candidate(
            candidate_id=str(record["candidate_id"]),
            family=str(record["family"]),
            representation=str(record["representation"]),
            expression=str(record["expression"]),
            recurrence_coefficients=tuple(
                Fraction(str(value)) for value in record["recurrence_coefficients"]
            ),
            recurrence_seed=tuple(Fraction(str(value)) for value in record["recurrence_seed"]),
            invariants=tuple(str(value) for value in record["invariants"]),
            proof_plan=tuple(str(value) for value in record["proof_plan"]),
            proposer=str(record["proposer"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise SeriousClaimVerificationError(
            "known-control candidate could not be replayed"
        ) from error
    if candidate.representation != "sympy_expression":
        raise SeriousClaimVerificationError("known-control candidate left the expression backend")
    return candidate


def _runtime_known_controls(
    root: Path,
    config: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> dict[str, tuple[Benchmark, SealedTarget, Candidate, Mapping[str, Any]]]:
    public, public_benchmarks = load_public_benchmarks(root)
    if public["sealed_targets_path"] != config["sources"]["sealed_targets"]:
        raise SeriousClaimVerificationError("sealed-target source changed")
    public_by_id = {item.benchmark_id: item for item in public_benchmarks}
    target_by_id = {
        item.benchmark_id: item for item in unseal_targets(root, public, public_benchmarks)
    }
    campaign_by_id = {item["benchmark_id"]: item for item in campaign["benchmarks"]}
    result = {}
    for benchmark_id in config["known_control_suite"]["benchmark_ids"]:
        try:
            campaign_benchmark = campaign_by_id[benchmark_id]
            benchmark = public_by_id[benchmark_id]
            target = target_by_id[benchmark_id]
            record = campaign_benchmark["ranked_candidates"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise SeriousClaimVerificationError("known-control runtime inputs changed") from error
        candidate = _candidate_from_record(record)
        if (
            target.target_kind != "known_formula"
            or target.reference_formula is None
            or target.commitment != campaign_benchmark["target_commitment_opened"]
        ):
            raise SeriousClaimVerificationError("known-control sealed target changed")
        positive = verify_known_formula(candidate, benchmark, target)
        if any(
            positive["backends"].get(backend) is not True
            for backend in MATHEMATICAL_MUTATION_BACKENDS
        ):
            raise SeriousClaimVerificationError("known-control backend reexecution failed")
        result[benchmark_id] = (benchmark, target, candidate, positive)
    return result


def _wrong_formula_mutation(
    backend: str,
    benchmark: Benchmark,
    target: SealedTarget,
    candidate: Candidate,
) -> dict[str, Any]:
    if backend not in MATHEMATICAL_MUTATION_BACKENDS or target.reference_formula is None:
        raise SeriousClaimVerificationError("wrong-formula mutation backend changed")
    mutated_expression = f"({candidate.expression}) + 1"
    mutation_body = {
        "benchmark_id": benchmark.benchmark_id,
        "candidate_id": candidate.candidate_id,
        "mutated_expression": mutated_expression,
        "operator": MATHEMATICAL_MUTATION_OPERATOR,
    }
    mutation_id = "mutation." + canonical_sha256(mutation_body)[:24]
    mutated = Candidate(
        candidate_id=mutation_id,
        family=candidate.family,
        representation=candidate.representation,
        expression=mutated_expression,
        recurrence_coefficients=candidate.recurrence_coefficients,
        recurrence_seed=candidate.recurrence_seed,
        invariants=candidate.invariants,
        proof_plan=candidate.proof_plan,
        proposer="verification_mutation",
    )
    witness_row = target.holdout_records[0]
    original_output = independently_predict(candidate, benchmark, (witness_row,))[0]
    mutated_output = independently_predict(mutated, benchmark, (witness_row,))[0]
    if original_output != witness_row.output or mutated_output is None:
        raise SeriousClaimVerificationError("wrong-formula exact witness could not be replayed")
    residual = mutated_output - original_output
    if residual != 1:
        raise SeriousClaimVerificationError("wrong-formula mutation is not an exact unit offset")

    found = safe_expression(mutated_expression, benchmark.aliases)
    reference = safe_expression(target.reference_formula, benchmark.aliases)
    symbolic_residual = sp.cancel(found - reference)
    symbols = [sp.Symbol(alias, real=True) for alias in benchmark.aliases]
    substitutions = dict(zip(symbols, witness_row.inputs, strict=True))
    witness = {
        "inputs": {
            alias: _fraction_text(value)
            for alias, value in zip(benchmark.aliases, witness_row.inputs, strict=True)
        },
        "mutated_output": _fraction_text(mutated_output),
        "original_output": _fraction_text(original_output),
        "residual": _fraction_text(residual),
    }

    if backend == "exact_arithmetic":
        rejected = mutated_output != witness_row.output
        backend_result = "MISMATCH_WITNESS"
        backend_evidence = {
            "comparison": "mutated_output != sealed_holdout_output",
            "implementation": "python_stdlib_fraction_ast_v1",
        }
    elif backend == "cas":
        rejected = symbolic_residual != 0
        backend_result = "NONZERO_NORMAL_FORM"
        backend_evidence = {"normalized_symbolic_residual": str(symbolic_residual)}
    elif backend == "smt":
        z3_variables = {
            symbol: z3.Real(alias) for symbol, alias in zip(symbols, benchmark.aliases, strict=True)
        }
        solver = z3.Solver()
        for symbol, value in substitutions.items():
            solver.add(z3_variables[symbol] == z3.RealVal(f"{value.numerator}/{value.denominator}"))
        solver.add(sympy_to_z3(found, z3_variables) != sympy_to_z3(reference, z3_variables))
        solver_result = solver.check()
        rejected = solver_result == z3.sat
        backend_result = "SAT_COUNTERMODEL"
        backend_evidence = {
            "solver_result": str(solver_result),
            "witness_constraints_bound": True,
        }
    else:
        enclosure = interval_value(found, substitutions) - interval_value(reference, substitutions)
        rejected = not (enclosure.a <= 0 <= enclosure.b)
        backend_result = "ZERO_EXCLUDED"
        backend_evidence = {
            "enclosure_relation": "strictly_positive",
            "zero_excluded": rejected,
        }
    if rejected is not True:
        raise SeriousClaimVerificationError(f"{backend} accepted a wrong known-control formula")
    return {
        "backend": backend,
        "backend_evidence": backend_evidence,
        "backend_result": backend_result,
        "benchmark_id": benchmark.benchmark_id,
        "candidate_id": candidate.candidate_id,
        "mutated_expression": mutated_expression,
        "mutated_expression_sha256": canonical_sha256({"expression": mutated_expression}),
        "mutation_id": mutation_id,
        "mutation_operator": MATHEMATICAL_MUTATION_OPERATOR,
        "reference_formula_sha256": canonical_sha256(
            {"reference_formula": target.reference_formula}
        ),
        "target_commitment_sha256": target.commitment,
        "witness": witness,
        "wrong_formula_rejected": True,
    }


def _stage(
    backend: str,
    order: int,
    candidate_scope_sha256: str,
    previous_stage_sha256: str | None,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": STAGE_SCHEMA,
        "backend": backend,
        "order": order,
        "candidate_scope_sha256": candidate_scope_sha256,
        "previous_stage_sha256": previous_stage_sha256,
        "backend_available": True,
        "positive_control_passed": True,
        "evidence": dict(evidence),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def _validate_stage_evidence(
    stage: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> None:
    backend = stage["backend"]
    evidence = stage["evidence"]
    bindings_by_id = {item["benchmark_id"]: item for item in bindings}
    if backend == "lean":
        _strict(
            evidence,
            {
                "artifact_content_sha256",
                "artifact_id",
                "kernel_checked",
                "source_sha256",
                "target",
                "theorems",
                "wrong_formula_kernel_control",
            },
            "Lean stage evidence",
        )
        kernel_control = evidence["wrong_formula_kernel_control"]
        _strict(
            kernel_control,
            {"artifact_bound", "required_for_serious_claim", "status"},
            "Lean wrong-formula control",
        )
        if evidence["kernel_checked"] is not True or kernel_control != {
            "artifact_bound": False,
            "required_for_serious_claim": True,
            "status": LEAN_MUTATION_STATUS,
        }:
            raise SeriousClaimVerificationError("Lean wrong-formula evidence changed")
        _sha(evidence["artifact_content_sha256"], "Lean artifact content")
        _sha(evidence["source_sha256"], "Lean source")
        return

    _strict(
        evidence,
        {"campaign_content_sha256", "controls", "mathematical_mutation_controls"},
        f"{backend} stage evidence",
    )
    _sha(evidence["campaign_content_sha256"], "campaign content")
    controls = evidence["controls"]
    mutations = evidence["mathematical_mutation_controls"]
    if (
        not isinstance(controls, list)
        or not isinstance(mutations, list)
        or len(controls) != len(bindings)
        or len(mutations) != len(bindings)
    ):
        raise SeriousClaimVerificationError(f"{backend} control coverage changed")
    for control in controls:
        expected = {
            "backend_passed",
            "benchmark_id",
            "campaign_reported_backend_passed",
            "candidate_id",
            "independent_positive_reexecution",
        }
        if backend == "exact_arithmetic":
            expected.add("independent_exact_match")
        _strict(control, expected, f"{backend} positive control")
        binding = bindings_by_id.get(control["benchmark_id"])
        if (
            binding is None
            or control["candidate_id"] != binding["candidate_id"]
            or control["backend_passed"] is not True
            or control["campaign_reported_backend_passed"] is not True
            or control["independent_positive_reexecution"] is not True
            or (backend == "exact_arithmetic" and control["independent_exact_match"] is not True)
        ):
            raise SeriousClaimVerificationError(f"{backend} positive control changed")

    expected_results = {
        "exact_arithmetic": "MISMATCH_WITNESS",
        "cas": "NONZERO_NORMAL_FORM",
        "smt": "SAT_COUNTERMODEL",
        "interval": "ZERO_EXCLUDED",
    }
    for mutation in mutations:
        _strict(
            mutation,
            {
                "backend",
                "backend_evidence",
                "backend_result",
                "benchmark_id",
                "candidate_id",
                "mutated_expression",
                "mutated_expression_sha256",
                "mutation_id",
                "mutation_operator",
                "reference_formula_sha256",
                "target_commitment_sha256",
                "witness",
                "wrong_formula_rejected",
            },
            f"{backend} mathematical mutation",
        )
        binding = bindings_by_id.get(mutation["benchmark_id"])
        expected_mutation_id = (
            "mutation."
            + canonical_sha256(
                {
                    "benchmark_id": mutation["benchmark_id"],
                    "candidate_id": mutation["candidate_id"],
                    "mutated_expression": mutation["mutated_expression"],
                    "operator": mutation["mutation_operator"],
                }
            )[:24]
        )
        if (
            binding is None
            or mutation["backend"] != backend
            or mutation["candidate_id"] != binding["candidate_id"]
            or mutation["target_commitment_sha256"] != binding["target_commitment_sha256"]
            or mutation["mutation_operator"] != MATHEMATICAL_MUTATION_OPERATOR
            or mutation["mutation_id"] != expected_mutation_id
            or mutation["mutated_expression_sha256"]
            != canonical_sha256({"expression": mutation["mutated_expression"]})
            or mutation["backend_result"] != expected_results[backend]
            or mutation["wrong_formula_rejected"] is not True
        ):
            raise SeriousClaimVerificationError(f"{backend} wrong-formula rejection changed")
        _sha(mutation["reference_formula_sha256"], "reference formula")
        witness = mutation["witness"]
        _strict(
            witness,
            {"inputs", "mutated_output", "original_output", "residual"},
            f"{backend} mutation witness",
        )
        try:
            original = Fraction(witness["original_output"])
            mutated = Fraction(witness["mutated_output"])
            residual = Fraction(witness["residual"])
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise SeriousClaimVerificationError(
                f"{backend} mutation witness is not exact"
            ) from error
        if residual != 1 or mutated - original != residual:
            raise SeriousClaimVerificationError(f"{backend} mutation witness changed")
        backend_evidence = mutation["backend_evidence"]
        if backend == "exact_arithmetic":
            expected_backend_evidence = {
                "comparison": "mutated_output != sealed_holdout_output",
                "implementation": "python_stdlib_fraction_ast_v1",
            }
        elif backend == "cas":
            expected_backend_evidence = {"normalized_symbolic_residual": "1"}
        elif backend == "smt":
            expected_backend_evidence = {
                "solver_result": "sat",
                "witness_constraints_bound": True,
            }
        else:
            expected_backend_evidence = {
                "enclosure_relation": "strictly_positive",
                "zero_excluded": True,
            }
        if backend_evidence != expected_backend_evidence:
            raise SeriousClaimVerificationError(f"{backend} backend witness changed")


def validate_candidate_chain(chain: Mapping[str, Any]) -> None:
    _strict(
        chain,
        {
            "candidate_bindings",
            "candidate_scope_sha256",
            "chain_id",
            "claims",
            "content_sha256",
            "purpose",
            "schema_version",
            "stages",
            "status",
        },
        "candidate verification chain",
    )
    _validate_seal(chain, "candidate verification chain")
    if chain["schema_version"] != CHAIN_SCHEMA:
        raise SeriousClaimVerificationError("candidate verification chain schema changed")
    bindings = chain["candidate_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise SeriousClaimVerificationError("candidate verification chain has no candidates")
    for binding in bindings:
        _strict(
            binding,
            {
                "behavior_sha256",
                "benchmark_id",
                "candidate_id",
                "expression_sha256",
                "target_commitment_sha256",
            },
            "candidate verification binding",
        )
        for key in (
            "behavior_sha256",
            "expression_sha256",
            "target_commitment_sha256",
        ):
            _sha(binding[key], f"candidate verification {key}")
    expected_scope = canonical_sha256({"candidate_bindings": bindings})
    if chain["candidate_scope_sha256"] != expected_scope:
        raise SeriousClaimVerificationError("candidate verification scope changed")
    stages = chain["stages"]
    if not isinstance(stages, list) or len(stages) != len(REQUIRED_STAGES):
        raise SeriousClaimVerificationError("candidate verification stage coverage changed")
    previous = None
    for order, (expected_backend, stage) in enumerate(zip(REQUIRED_STAGES, stages, strict=True)):
        _strict(
            stage,
            {
                "backend",
                "backend_available",
                "candidate_scope_sha256",
                "content_sha256",
                "evidence",
                "order",
                "positive_control_passed",
                "previous_stage_sha256",
                "schema_version",
            },
            "candidate verification stage",
        )
        _validate_seal(stage, "candidate verification stage")
        if (
            stage["schema_version"] != STAGE_SCHEMA
            or stage["backend"] != expected_backend
            or stage["order"] != order
            or stage["candidate_scope_sha256"] != expected_scope
            or stage["previous_stage_sha256"] != previous
            or stage["backend_available"] is not True
            or stage["positive_control_passed"] is not True
        ):
            raise SeriousClaimVerificationError("candidate verification stage policy failed")
        _validate_stage_evidence(stage, bindings)
        previous = stage["content_sha256"]
    if chain["status"] != "PASS_KNOWN_CONTROL_BACKEND_LADDER":
        raise SeriousClaimVerificationError("candidate verification chain status changed")
    claims = chain["claims"]
    _strict(
        claims,
        {
            "all_five_backend_wrong_formula_controls_complete",
            "known_control_calibration_passed",
            "literature_novelty_established",
            "new_candidate_verified",
            "serious_claim_released",
        },
        "candidate verification claims",
    )
    if claims != {
        "all_five_backend_wrong_formula_controls_complete": False,
        "known_control_calibration_passed": True,
        "literature_novelty_established": False,
        "new_candidate_verified": False,
        "serious_claim_released": False,
    }:
        raise SeriousClaimVerificationError("candidate verification claim boundary changed")


def _known_control_chain(
    root: Path,
    config: Mapping[str, Any],
    campaign: Mapping[str, Any],
    multi_host: Mapping[str, Any],
) -> dict[str, Any]:
    benchmarks = {item["benchmark_id"]: item for item in campaign["benchmarks"]}
    benchmark_ids = config["known_control_suite"]["benchmark_ids"]
    bindings = _candidate_bindings(benchmarks, benchmark_ids)
    runtime_controls = _runtime_known_controls(root, config, campaign)
    scope = canonical_sha256({"candidate_bindings": bindings})
    stages = []
    previous = None
    for order, backend in enumerate(REQUIRED_STAGES):
        if backend == "lean":
            lean_source = _under(root, config["sources"]["lean_source"], "Lean source")
            source_text = lean_source.read_text(encoding="utf-8")
            theorem_names = config["known_control_suite"]["lean_theorems"]
            if any(name.rsplit(".", 1)[-1] not in source_text for name in theorem_names):
                raise SeriousClaimVerificationError("known-control Lean theorem is missing")
            if multi_host.get("lean", {}).get("kernel_checked") is not True:
                raise SeriousClaimVerificationError("known-control Lean artifact did not pass")
            evidence = {
                "artifact_content_sha256": multi_host["lean"]["content_sha256"],
                "artifact_id": multi_host["lean"]["artifact_id"],
                "kernel_checked": True,
                "source_sha256": _normalized_file_sha256(lean_source),
                "target": config["known_control_suite"]["lean_target"],
                "theorems": list(theorem_names),
                "wrong_formula_kernel_control": {
                    "artifact_bound": False,
                    "required_for_serious_claim": True,
                    "status": LEAN_MUTATION_STATUS,
                },
            }
        else:
            rows = []
            mathematical_mutations = []
            for benchmark_id in benchmark_ids:
                benchmark = benchmarks[benchmark_id]
                runtime_benchmark, target, candidate, positive = runtime_controls[benchmark_id]
                formal = benchmark["formal_verification"]["backends"]
                if formal.get(backend) is not True:
                    raise SeriousClaimVerificationError(
                        f"known-control {backend} evidence did not pass"
                    )
                row = {
                    "backend_passed": True,
                    "benchmark_id": benchmark_id,
                    "campaign_reported_backend_passed": True,
                    "candidate_id": benchmark["ranked_candidates"][0]["candidate_id"],
                    "independent_positive_reexecution": positive["backends"][backend],
                }
                if backend == "exact_arithmetic":
                    row["independent_exact_match"] = benchmark["independent_exact_reproduction"][
                        "match"
                    ]
                    if row["independent_exact_match"] is not True:
                        raise SeriousClaimVerificationError(
                            "known-control independent exact reproduction failed"
                        )
                rows.append(row)
                mathematical_mutations.append(
                    _wrong_formula_mutation(backend, runtime_benchmark, target, candidate)
                )
            evidence = {
                "campaign_content_sha256": campaign["content_sha256"],
                "controls": rows,
                "mathematical_mutation_controls": mathematical_mutations,
            }
        current = _stage(backend, order, scope, previous, evidence)
        stages.append(current)
        previous = current["content_sha256"]
    body = {
        "schema_version": CHAIN_SCHEMA,
        "chain_id": config["ladder_id"] + ".known-controls",
        "purpose": "known_control_calibration_only",
        "candidate_bindings": bindings,
        "candidate_scope_sha256": scope,
        "stages": stages,
        "status": "PASS_KNOWN_CONTROL_BACKEND_LADDER",
        "claims": {
            "all_five_backend_wrong_formula_controls_complete": False,
            "known_control_calibration_passed": True,
            "literature_novelty_established": False,
            "new_candidate_verified": False,
            "serious_claim_released": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_candidate_chain(body)
    return body


def _negative_controls(
    config: Mapping[str, Any], campaign: Mapping[str, Any]
) -> list[dict[str, Any]]:
    benchmarks = {item["benchmark_id"]: item for item in campaign["benchmarks"]}
    result = []
    for benchmark_id in config["negative_control_benchmark_ids"]:
        benchmark = benchmarks.get(benchmark_id)
        if benchmark is None or benchmark.get("target_kind") != "bounded_unknown":
            raise SeriousClaimVerificationError("bounded-unknown negative control changed")
        backends = benchmark["formal_verification"]["backends"]
        missing = [backend for backend in REQUIRED_STAGES if backends.get(backend) is not True]
        if not missing or benchmark["claims"]["serious_claim_released"] is not False:
            raise SeriousClaimVerificationError("bounded-unknown control did not fail closed")
        result.append(
            {
                "benchmark_id": benchmark_id,
                "missing_or_failed_backends": missing,
                "serious_claim_released": False,
                "status": "BLOCKED_INCOMPLETE_BACKEND_LADDER",
            }
        )
    return result


def _reseal_chain(chain: dict[str, Any]) -> None:
    previous = None
    for stage in chain["stages"]:
        stage["previous_stage_sha256"] = previous
        body = {key: item for key, item in stage.items() if key != "content_sha256"}
        stage["content_sha256"] = canonical_sha256(body)
        previous = stage["content_sha256"]
    body = {key: item for key, item in chain.items() if key != "content_sha256"}
    chain["content_sha256"] = canonical_sha256(body)


def _mutation_controls(chain: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = []
    for mutation in MUTATIONS:
        changed = deepcopy(chain)
        if mutation == "missing_stage":
            del changed["stages"][2]
        elif mutation == "reordered_stages":
            changed["stages"][1], changed["stages"][2] = (
                changed["stages"][2],
                changed["stages"][1],
            )
        elif mutation == "candidate_scope_substitution":
            changed["stages"][3]["candidate_scope_sha256"] = "0" * 64
        elif mutation == "broken_previous_stage_link":
            changed["stages"][4]["previous_stage_sha256"] = "0" * 64
        elif mutation == "backend_unavailable":
            changed["stages"][4]["backend_available"] = False
        if mutation in {"reordered_stages", "candidate_scope_substitution", "backend_unavailable"}:
            _reseal_chain(changed)
        else:
            body = {key: item for key, item in changed.items() if key != "content_sha256"}
            changed["content_sha256"] = canonical_sha256(body)
        try:
            validate_candidate_chain(changed)
        except SeriousClaimVerificationError as error:
            controls.append(
                {
                    "mutation_id": mutation,
                    "rejected": True,
                    "reason_class": type(error).__name__,
                }
            )
        else:
            raise SeriousClaimVerificationError(f"ladder mutation survived: {mutation}")
    return controls


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    campaign = _campaign(root, config)
    multi_host = _multi_host(root, config)
    chain = _known_control_chain(root, config, campaign, multi_host)
    negative = _negative_controls(config, campaign)
    mutations = _mutation_controls(chain)
    body = {
        "schema_version": SCHEMA_VERSION,
        "ladder_id": config["ladder_id"],
        "source_bindings": {
            "campaign_receipt": {
                "content_sha256": campaign["content_sha256"],
                "path": config["sources"]["campaign_receipt"],
            },
            "config": {
                "normalized_file_sha256": _normalized_file_sha256(root / CONFIG_PATH),
                "path": CONFIG_PATH,
            },
            "multi_host_receipt": {
                "content_sha256": multi_host["content_sha256"],
                "path": config["sources"]["multi_host_receipt"],
            },
            "public_benchmarks": {
                "normalized_file_sha256": _normalized_file_sha256(
                    root / config["sources"]["public_benchmarks"]
                ),
                "path": config["sources"]["public_benchmarks"],
            },
            "sealed_targets": {
                "normalized_file_sha256": _normalized_file_sha256(
                    root / config["sources"]["sealed_targets"]
                ),
                "path": config["sources"]["sealed_targets"],
            },
        },
        "known_control_chain": chain,
        "negative_controls": negative,
        "mutation_controls": mutations,
        "summary": {
            "backend_mathematical_mutations_rejected": len(chain["candidate_bindings"])
            * len(MATHEMATICAL_MUTATION_BACKENDS),
            "known_control_candidates": len(chain["candidate_bindings"]),
            "lean_kernel_mutation_artifact_bound": False,
            "negative_controls_blocked": len(negative),
            "required_stage_order": list(REQUIRED_STAGES),
            "structural_mutations_rejected": len(mutations),
            "status": "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION",
        },
        "release_gate": {
            "backend_wrong_formula_mutation_required": True,
            "candidate_specific_chain_required": True,
            "independent_reproduction_required": True,
            "lean_wrong_formula_artifact_required": True,
            "named_human_prior_art_review_required": True,
            "serious_claims_released": 0,
            "status": "BLOCKED_NO_NEW_CANDIDATE_COMPLETE_LADDER",
        },
        "claims": {
            "all_five_backend_mutations_complete": False,
            "backend_availability_is_proof": False,
            "known_control_calibration_is_novelty": False,
            "novel_formula_established": False,
            "serious_claim_released": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "known_control_chain",
            "ladder_id",
            "mutation_controls",
            "negative_controls",
            "release_gate",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "serious-claim ladder receipt",
    )
    _validate_seal(value, "serious-claim ladder receipt")
    if value["schema_version"] != SCHEMA_VERSION:
        raise SeriousClaimVerificationError("serious-claim ladder schema changed")
    validate_candidate_chain(value["known_control_chain"])
    summary = value["summary"]
    _strict(
        summary,
        {
            "backend_mathematical_mutations_rejected",
            "known_control_candidates",
            "lean_kernel_mutation_artifact_bound",
            "negative_controls_blocked",
            "required_stage_order",
            "status",
            "structural_mutations_rejected",
        },
        "serious-claim ladder summary",
    )
    if (
        summary.get("status") != "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION"
        or tuple(summary.get("required_stage_order", ())) != REQUIRED_STAGES
        or summary.get("known_control_candidates", 0) < 2
        or summary.get("backend_mathematical_mutations_rejected")
        != summary.get("known_control_candidates", 0) * len(MATHEMATICAL_MUTATION_BACKENDS)
        or summary.get("lean_kernel_mutation_artifact_bound") is not False
        or summary.get("negative_controls_blocked", 0) < 2
        or summary.get("structural_mutations_rejected") != len(MUTATIONS)
    ):
        raise SeriousClaimVerificationError("serious-claim ladder summary changed")
    negative = value["negative_controls"]
    for item in negative:
        _strict(
            item,
            {
                "benchmark_id",
                "missing_or_failed_backends",
                "serious_claim_released",
                "status",
            },
            "negative candidate control",
        )
    if any(
        item.get("status") != "BLOCKED_INCOMPLETE_BACKEND_LADDER"
        or item.get("serious_claim_released") is not False
        or not item.get("missing_or_failed_backends")
        for item in negative
    ):
        raise SeriousClaimVerificationError("negative candidate control changed")
    mutations = value["mutation_controls"]
    for item in mutations:
        _strict(
            item,
            {"mutation_id", "reason_class", "rejected"},
            "ladder mutation control",
        )
    if [item.get("mutation_id") for item in mutations] != list(MUTATIONS) or any(
        item.get("rejected") is not True for item in mutations
    ):
        raise SeriousClaimVerificationError("ladder mutation evidence changed")
    release = value["release_gate"]
    _strict(
        release,
        {
            "backend_wrong_formula_mutation_required",
            "candidate_specific_chain_required",
            "independent_reproduction_required",
            "lean_wrong_formula_artifact_required",
            "named_human_prior_art_review_required",
            "serious_claims_released",
            "status",
        },
        "serious-claim release gate",
    )
    if (
        release.get("status") != "BLOCKED_NO_NEW_CANDIDATE_COMPLETE_LADDER"
        or release.get("serious_claims_released") != 0
        or release.get("candidate_specific_chain_required") is not True
        or release.get("backend_wrong_formula_mutation_required") is not True
        or release.get("independent_reproduction_required") is not True
        or release.get("lean_wrong_formula_artifact_required") is not True
        or release.get("named_human_prior_art_review_required") is not True
    ):
        raise SeriousClaimVerificationError("serious-claim release boundary changed")
    _strict(
        value["claims"],
        {
            "all_five_backend_mutations_complete",
            "backend_availability_is_proof",
            "known_control_calibration_is_novelty",
            "novel_formula_established",
            "serious_claim_released",
        },
        "serious-claim ladder claims",
    )
    if any(value["claims"].values()):
        raise SeriousClaimVerificationError("serious-claim ladder claim boundary changed")
    if root is not None:
        root = root.resolve()
        config = load_config(root)
        campaign = _campaign(root, config)
        multi_host = _multi_host(root, config)
        bindings = value["source_bindings"]
        _strict(
            bindings,
            {
                "campaign_receipt",
                "config",
                "multi_host_receipt",
                "public_benchmarks",
                "sealed_targets",
            },
            "serious-claim source bindings",
        )
        _strict(
            bindings["campaign_receipt"],
            {"content_sha256", "path"},
            "campaign source binding",
        )
        _strict(
            bindings["config"],
            {"normalized_file_sha256", "path"},
            "ladder config source binding",
        )
        _strict(
            bindings["multi_host_receipt"],
            {"content_sha256", "path"},
            "multi-host source binding",
        )
        for key in ("public_benchmarks", "sealed_targets"):
            _strict(
                bindings[key],
                {"normalized_file_sha256", "path"},
                f"{key} source binding",
            )
        if (
            bindings["config"]["path"] != CONFIG_PATH
            or bindings["campaign_receipt"]["path"] != config["sources"]["campaign_receipt"]
            or bindings["multi_host_receipt"]["path"] != config["sources"]["multi_host_receipt"]
            or bindings["public_benchmarks"]["path"] != config["sources"]["public_benchmarks"]
            or bindings["sealed_targets"]["path"] != config["sources"]["sealed_targets"]
            or bindings.get("config", {}).get("normalized_file_sha256")
            != _normalized_file_sha256(root / CONFIG_PATH)
            or bindings.get("campaign_receipt", {}).get("content_sha256")
            != campaign["content_sha256"]
            or bindings.get("multi_host_receipt", {}).get("content_sha256")
            != multi_host["content_sha256"]
            or bindings["public_benchmarks"]["normalized_file_sha256"]
            != _normalized_file_sha256(root / config["sources"]["public_benchmarks"])
            or bindings["sealed_targets"]["normalized_file_sha256"]
            != _normalized_file_sha256(root / config["sources"]["sealed_targets"])
        ):
            raise SeriousClaimVerificationError("serious-claim ladder source binding changed")
        expected_chain = _known_control_chain(root, config, campaign, multi_host)
        if value["known_control_chain"] != expected_chain:
            raise SeriousClaimVerificationError("known-control ladder evidence changed")
        if value["negative_controls"] != _negative_controls(config, campaign):
            raise SeriousClaimVerificationError("negative-control ladder evidence changed")
        if value["mutation_controls"] != _mutation_controls(expected_chain):
            raise SeriousClaimVerificationError("ladder mutation audit changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    if args.command == "build":
        receipt = build_receipt(args.root)
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = _read_json(args.receipt.resolve(), "serious-claim ladder receipt")
        validate_receipt(receipt, args.root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
