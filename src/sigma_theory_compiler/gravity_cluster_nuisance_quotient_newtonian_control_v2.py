from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as sbc_v1
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v3 as sbc_v3
from sigma_theory_compiler import (
    gravity_cluster_nuisance_quotient_sbc_v3_adjudicator as v3_adjudicator,
)
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_newtonian_control_v2.json")
ARTIFACT_DIR = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-newtonian-control-v2"
)
PREDICTOR_PATH = ARTIFACT_DIR / "target-blind-predictors-v2.json"
CONTROLS_PATH = ARTIFACT_DIR / "bounded-controls-v2.json"
SMOKE_PATH = ARTIFACT_DIR / "bounded-smoke-v2.json"
UNAUTHORIZED_PATH = ARTIFACT_DIR / "authorization-current-unauthorized-v2-final.json"
AUTHORIZATION_CONTROLS_PATH = ARTIFACT_DIR / "authorization-controls-v2-final.json"
PRODUCTION_RESULT_PATH = ARTIFACT_DIR / "matched-newtonian-control-v2-production.npz"
IMPLEMENTATION_RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/"
    "nuisance-quotient-newtonian-control-implementation-v2r1-final.json"
)
TEST_PATH = Path(
    "tests/test_gravity_cluster_nuisance_quotient_newtonian_control_v2.py"
)

ADJUDICATOR_SOURCE_PATH = Path(
    "src/sigma_theory_compiler/"
    "gravity_cluster_nuisance_quotient_sbc_v3_adjudicator.py"
)
ADJUDICATOR_CONFIG_PATH = Path(
    "configs/gravity_cluster_nuisance_quotient_sbc_v3_adjudicator_v1.json"
)
ADJUDICATOR_RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/"
    "nuisance-quotient-sbc-v3-adjudicator-v1.json"
)

# Bound only to the independently frozen strict adjudicator, never to raw V3 check.
EXPECTED_ADJUDICATOR_SOURCE_SHA256 = (
    "cad4a844a061ebac0fcb5b8a86e754b8358f2440ac31e6d1249904d7c6e01844"
)
EXPECTED_ADJUDICATOR_CONFIG_SHA256 = (
    "f6e12b086f4129a97711a8a382aa81042ea30c6d99bbc76dd6f95446c52acb3b"
)
EXPECTED_ADJUDICATOR_RECEIPT_SHA256 = (
    "40ebef17e95ad1f4e1b14fb88e010c076b2b36a5fc40152f23e771e6dcea301c"
)

CONFIG_SCHEMA = (
    "invariant-gravity-cluster-nuisance-quotient-newtonian-control-config-2.0"
)
PREDICTOR_SCHEMA = "invariant-gravity-target-blind-newtonian-predictors-2.0"
CONTROLS_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-controls-2.0"
SMOKE_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-smoke-2.0"
RESULT_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-control-result-2.0"
UNAUTHORIZED_SCHEMA = (
    "invariant-gravity-nuisance-quotient-newtonian-authorization-2.0-unauthorized"
)
AUTHORIZED_SCHEMA = (
    "invariant-gravity-nuisance-quotient-newtonian-authorization-2.0-authorized"
)
APPROVAL_SCHEMA = (
    "invariant-gravity-nuisance-quotient-newtonian-external-approval-2.0"
)
IMPLEMENTATION_SCHEMA = (
    "invariant-gravity-nuisance-quotient-newtonian-implementation-2.0"
)
STRICT_ADJUDICATOR_SCHEMA = (
    "invariant-gravity-nuisance-quotient-sbc-v3-adjudicator-receipt-1.0"
)
STRICT_ADJUDICATOR_STATUS = (
    "strictly_verified_v3_synthetic_pass_newtonian_eligible_production_locked"
)
STRICT_ADJUDICATOR_DECISION = (
    "V3_SYNTHETIC_SBC_PASSED_NEWTONIAN_CONTROL_MAY_UNLOCK_"
    "CANDIDATE_PRODUCTION_REMAINS_LOCKED"
)

TRUTH_UNIT = np.full(17, 0.5)
TARGET_SEED = 749_301
PREDICTOR_ROWS = 80
RUN_ID = "matched-newtonian-control-v2-production-1"
MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS = 233_504
PAIRED_MAXIMUM_LIKELIHOOD_EVALUATIONS = 467_008
MAXIMUM_PAID_EXTERNAL_COST_USD = 0.0
FORBIDDEN_PREDICTOR_FIELDS = {
    "target",
    "observation",
    "residual",
    "cluster",
    "galaxy",
    "object",
    "survey",
    "holdout",
    "confirmation",
    "independent",
    "development",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    if not target.is_file():
        raise RuntimeError(f"bound artifact is absent: {path}")
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_binding(binding: dict[str, Any], label: str) -> Path:
    strict_keys(binding, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError(f"{label} missing or tampered")
    return target


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_PREDICTOR_FIELDS):
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def predictor_body() -> dict[str, Any]:
    rng = np.random.default_rng(731_209)
    source = np.exp(np.linspace(math.log(0.03), math.log(6.0), PREDICTOR_ROWS))
    sigma = 0.055 + 0.025 * (1.0 + np.sin(np.linspace(0.0, 4.0 * math.pi, PREDICTOR_ROWS)))
    raw = rng.normal(size=(PREDICTOR_ROWS, 10))
    q, _ = np.linalg.qr(raw)
    basis = q[:, :10] * np.linspace(2.0, 0.8, 10)[None, :]
    rows = [
        {
            "synthetic_row_index": index,
            "dimensionless_newtonian_baryonic_acceleration": float(source[index]),
            "dimensionless_log_uncertainty": float(sigma[index]),
            "dimensionless_nuisance_basis": basis[index].tolist(),
        }
        for index in range(PREDICTOR_ROWS)
    ]
    body = {
        "schema_version": PREDICTOR_SCHEMA,
        "status": "synthetic_target_blind_predictors_frozen",
        "generator_seed": 731_209,
        "row_count": PREDICTOR_ROWS,
        "predictor_semantics": {
            "dimensionless_only": True,
            "synthetic_only": True,
            "target_fields_present": False,
            "real_object_or_survey_labels_present": False,
            "rank_ten_quotient_design": True,
        },
        "rows": rows,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def write_predictors(output: Path) -> dict[str, Any]:
    if confined(output) != ROOT / PREDICTOR_PATH:
        raise RuntimeError("predictor output path changed")
    body = predictor_body()
    sampler.write_json(output, body)
    return body


def load_predictors(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("synthetic target-blind predictor packet missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    if body != predictor_body() or _contains_forbidden_key(body["rows"]):
        raise RuntimeError("synthetic target-blind predictor packet changed")
    basis = np.asarray(
        [row["dimensionless_nuisance_basis"] for row in body["rows"]], dtype=float
    )
    if basis.shape != (PREDICTOR_ROWS, 10) or np.linalg.matrix_rank(basis) != 10:
        raise RuntimeError("synthetic predictor design lost quotient rank ten")
    return body


def predictor_arrays(packet: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = packet["rows"]
    source = np.asarray(
        [row["dimensionless_newtonian_baryonic_acceleration"] for row in rows],
        dtype=float,
    )
    sigma = np.asarray(
        [row["dimensionless_log_uncertainty"] for row in rows], dtype=float
    )
    basis = np.asarray(
        [row["dimensionless_nuisance_basis"] for row in rows], dtype=float
    )
    return source, sigma, basis


def sufficient_observation(
    packet: dict[str, Any], prior_config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, sigma, basis = predictor_arrays(packet)
    truth = sampler.composite_values(TRUTH_UNIT, prior_config) / sbc_v1.COMPOSITE_SCALES
    noise = np.random.default_rng(TARGET_SEED).normal(size=PREDICTOR_ROWS)
    generated_log_targets = np.log(source) + basis @ truth + sigma * noise
    residual = generated_log_targets - np.log(source)
    weights = 1.0 / sigma**2
    precision = basis.T @ (weights[:, None] * basis)
    observation = np.linalg.solve(precision, basis.T @ (weights * residual))
    return observation, precision, generated_log_targets


def per_fit_call_accounting() -> dict[str, int]:
    chains = int(sbc_v3.CANDIDATE_INFERENCE["replicates"]) * int(
        sbc_v3.CANDIDATE_INFERENCE["particles_per_replicate"]
    )
    sweeps = sum(
        int(sbc_v3.CANDIDATE_INFERENCE[name])
        for name in (
            "adaptation_sweeps",
            "fixed_kernel_settling_sweeps",
            "retained_sweeps",
        )
    )
    maximum = chains * (1 + sweeps * len(sbc_v3.TRANSPORT_BLOCKS))
    if maximum != MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS:
        raise RuntimeError("V3 per-fit call ceiling changed")
    return {
        "initial_likelihood_evaluations": chains,
        "transport_likelihood_evaluations": chains
        * sweeps
        * len(sbc_v3.TRANSPORT_BLOCKS),
        "maximum_newtonian_control_likelihood_evaluations": maximum,
        "matched_v3_candidate_per_fit_likelihood_evaluations": maximum,
        "maximum_paired_likelihood_evaluations": 2 * maximum,
        "real_forward_model_evaluations": 0,
    }


def run_request() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "matched_newtonian_control_runs": 1,
        "maximum_newtonian_control_likelihood_evaluations": (
            MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS
        ),
        "matched_v3_candidate_per_fit_likelihood_evaluations": (
            MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS
        ),
        "maximum_paired_likelihood_evaluations": (
            PAIRED_MAXIMUM_LIKELIHOOD_EVALUATIONS
        ),
        "maximum_paid_external_cost_usd": MAXIMUM_PAID_EXTERNAL_COST_USD,
        "network_calls": 0,
        "paid_or_model_calls": 0,
        "real_rows": 0,
        "output_path": PRODUCTION_RESULT_PATH.as_posix(),
    }


def strict_adjudicator_bindings() -> dict[str, dict[str, str]]:
    return {
        "source": {
            "path": ADJUDICATOR_SOURCE_PATH.as_posix(),
            "file_sha256": EXPECTED_ADJUDICATOR_SOURCE_SHA256,
        },
        "config": {
            "path": ADJUDICATOR_CONFIG_PATH.as_posix(),
            "file_sha256": EXPECTED_ADJUDICATOR_CONFIG_SHA256,
        },
        "receipt": {
            "path": ADJUDICATOR_RECEIPT_PATH.as_posix(),
            "file_sha256": EXPECTED_ADJUDICATOR_RECEIPT_SHA256,
        },
    }


def validate_strict_v3_adjudicator() -> dict[str, Any]:
    bindings = strict_adjudicator_bindings()
    for name, binding in bindings.items():
        validate_binding(binding, f"strict_v3_adjudicator.{name}")
    checked = v3_adjudicator.check_receipt(
        ROOT / ADJUDICATOR_CONFIG_PATH,
        EXPECTED_ADJUDICATOR_CONFIG_SHA256,
        ROOT / ADJUDICATOR_RECEIPT_PATH,
    )
    receipt = json.loads(
        (ROOT / ADJUDICATOR_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    if (
        receipt.get("schema_version") != STRICT_ADJUDICATOR_SCHEMA
        or receipt.get("status") != STRICT_ADJUDICATOR_STATUS
        or receipt.get("decision") != STRICT_ADJUDICATOR_DECISION
        or checked.get("valid") is not True
        or checked.get("passed") is not True
        or checked.get("v3_synthetic_sbc_passed") is not True
        or checked.get("newtonian_control_unlock") is not True
        or checked.get("candidate_production_unlock") is not False
        or checked.get("scientific_claim_allowed") is not False
    ):
        raise RuntimeError("strict V3 adjudicator did not unlock Newtonian control")
    return checked


def frozen_authorization_bindings(expected_config_sha256: str) -> dict[str, Any]:
    return {
        "newtonian_v2_config": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": expected_config_sha256,
        },
        "newtonian_v2_source": artifact_binding(Path(__file__)),
        "strict_v3_adjudicator": strict_adjudicator_bindings(),
    }


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if target != ROOT / CONFIG_PATH or not target.is_file():
        raise RuntimeError("Newtonian-control V2 config path changed")
    if file_sha256(target) != expected_sha256:
        raise RuntimeError("Newtonian-control V2 config hash changed")
    body = json.loads(target.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "production_result_path",
            "strict_v3_adjudicator",
            "frozen_v3_bindings",
            "predictor_packet",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "structural_kernel",
            "diagnostic_protocol",
            "call_accounting",
            "target_generator",
            "authorization_policy",
            "run_request",
            "data_boundary",
            "claim_boundary",
        },
        "Newtonian-control V2 config",
    )
    if (
        body["schema_version"] != CONFIG_SCHEMA
        or body["status"] != "prepared_external_approval_required_production_false"
        or body["production_result_path"] != PRODUCTION_RESULT_PATH.as_posix()
    ):
        raise RuntimeError("Newtonian-control V2 identity changed")
    source = confined(ROOT / body["implementation_source"])
    if source != Path(__file__).resolve() or normalized_sha256(source) != body[
        "implementation_source_normalized_sha256"
    ]:
        raise RuntimeError("Newtonian-control V2 source changed after freeze")
    if body["strict_v3_adjudicator"] != {
        "gate_kind": "strict_v3_adjudicator_not_raw_v3_check",
        "bindings": strict_adjudicator_bindings(),
        "required_schema": STRICT_ADJUDICATOR_SCHEMA,
        "required_status": STRICT_ADJUDICATOR_STATUS,
        "required_decision": STRICT_ADJUDICATOR_DECISION,
        "refusal_before_contract_or_predictor_load": True,
    }:
        raise RuntimeError("strict V3 adjudicator gate changed")
    for name, binding in body["frozen_v3_bindings"].items():
        validate_binding(binding, f"frozen_v3_bindings.{name}")
    if (
        body["exact_primitive_priors"] != sampler.PRIMITIVE_PRIORS
        or body["exact_primitive_priors"]
        != uncertainty.load_config(ROOT)["continuous_priors"]
        or body["primitive_prior_semantics"]
        != (
            "17_independent_uniform_primitives_with_clipped_six_factor_"
            "stellar_pushforward_clip_0.4_2.5"
        )
        or body["structural_kernel"] != sbc_v3.CANDIDATE_INFERENCE
        or body["diagnostic_protocol"]
        != {
            "implementation": "canonical_rank_normalized_split_rhat_bulk_tail_ess",
            "rank_protocol": sbc_v3.RANK_PROTOCOL,
            "scientific_gates": sbc_v3.GATES,
        }
        or body["call_accounting"] != per_fit_call_accounting()
        or body["run_request"] != run_request()
    ):
        raise RuntimeError("V3-matched priors, kernel, diagnostics, or calls changed")
    expected_policy = {
        "target_generator": {
            "law": (
                "newtonian_baryonic_acceleration_times_exponential_"
                "linear_quotient_pushforward"
            ),
            "truth_unit_value_for_all_17_primitives": 0.5,
            "noise_seed": TARGET_SEED,
            "gaussian_sufficient_statistic_dimension": 10,
            "targets_generated_only_after_both_production_gates": True,
            "generated_targets_persisted_as_rows": False,
        },
        "authorization_policy": {
            "strict_v3_adjudicator_required_first": True,
            "exact_external_approval_required_second": True,
            "contract_and_predictors_loaded_only_after_both_gates": True,
            "production_authorized_by_default": False,
            "explicit_cli_sentinel_required": True,
            "all_generated_artifacts_atomic_no_clobber": True,
        },
        "data_boundary": {
            "synthetic_target_blind_predictor_rows": PREDICTOR_ROWS,
            "real_development_rows": 0,
            "real_holdout_rows": 0,
            "real_confirmation_rows": 0,
            "real_independent_rows": 0,
            "network_calls": 0,
            "paid_or_model_calls": 0,
            "production_runs": 0,
        },
        "claim_boundary": {
            "package_prepared": True,
            "bounded_controls_only": True,
            "full_matched_newtonian_run_completed": False,
            "false_selection_rate_measured": False,
            "candidate_physics_supported": False,
            "newtonian_null_rejected": False,
            "publication_readiness_changed": False,
            "scientific_claim_allowed": False,
        },
    }
    for name, expected in expected_policy.items():
        if body[name] != expected:
            raise RuntimeError(f"Newtonian-control V2 frozen policy changed: {name}")
    predictor = load_predictors(
        ROOT / body["predictor_packet"]["path"],
        body["predictor_packet"]["file_sha256"],
    )
    if predictor["content_sha256"] != body["predictor_packet"]["content_sha256"]:
        raise RuntimeError("predictor content seal changed")
    body["_config_sha256"] = expected_sha256
    return body


def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    prior_config = uncertainty.load_config(ROOT)
    packet = load_predictors(
        ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    _source, _sigma, basis = predictor_arrays(packet)
    observation, precision, _targets = sufficient_observation(packet, prior_config)
    kernel = sbc_v3.kernel_controls(prior_config)
    passed = bool(
        kernel["passed"]
        and basis.shape == (PREDICTOR_ROWS, 10)
        and np.linalg.matrix_rank(basis) == 10
        and np.all(np.linalg.eigvalsh(precision) > 0.0)
        and observation.shape == (10,)
    )
    return {
        "schema_version": CONTROLS_SCHEMA,
        "status": "bounded_synthetic_controls_only",
        "passed": passed,
        "exact_v3_structural_kernel_controls": kernel,
        "predictor_rows": PREDICTOR_ROWS,
        "predictor_basis_rank": int(np.linalg.matrix_rank(basis)),
        "sufficient_statistic_dimension": len(observation),
        "precision_positive_definite": bool(np.all(np.linalg.eigvalsh(precision) > 0.0)),
        "full_matched_newtonian_run_launched": False,
        "production_likelihood_evaluations": 0,
        "real_rows": 0,
        "network_calls": 0,
        "paid_or_model_calls": 0,
    }


def bounded_smoke(contract: dict[str, Any]) -> dict[str, Any]:
    prior_config = uncertainty.load_config(ROOT)
    packet = load_predictors(
        ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    observation, precision, _targets = sufficient_observation(packet, prior_config)
    particles_by_replicate = sbc_v3.candidate_sobol_starts(0)
    evaluator = sbc_v3.CandidateLikelihood(observation, precision, prior_config)
    accepted = 0
    attempted = 0
    orbit_attempted = 0
    orbit_accepted = 0
    for replicate, original in enumerate(particles_by_replicate):
        particles = original.copy()
        likelihood = evaluator.batch(particles)
        rng = np.random.default_rng(
            int(sbc_v3.SEED_LINEAGE["candidate_transition_base"]) + replicate
        )
        orbit = sampler.orbit_sweep(
            particles, rng, prior_config, sbc_v3.CANDIDATE_INFERENCE
        )
        orbit_attempted += sum(
            int(orbit[f"{move}_attempted"]) for move in sampler.ORBIT_NAMES
        )
        orbit_accepted += sum(
            int(orbit[f"{move}_accepted"]) for move in sampler.ORBIT_NAMES
        )
        for block in sbc_v3.TRANSPORT_BLOCKS:
            row = sbc_v3.pcn_block_transition(
                particles,
                likelihood,
                evaluator,
                block,
                float(block["initial_beta"]),
                rng,
            )
            attempted += int(row["attempted"])
            accepted += int(row["accepted"])
    expected_calls = int(np.prod(particles_by_replicate.shape[:2])) * (
        1 + len(sbc_v3.TRANSPORT_BLOCKS)
    )
    passed = bool(
        evaluator.calls == expected_calls
        and 0 < accepted < attempted
        and orbit_attempted > 0
        and orbit_accepted > 0
    )
    return {
        "schema_version": SMOKE_SCHEMA,
        "status": "bounded_one_sweep_smoke_not_scientific_adjudication",
        "passed": passed,
        "transport_blocks": [row["block_id"] for row in sbc_v3.TRANSPORT_BLOCKS],
        "likelihood_evaluations": evaluator.calls,
        "expected_likelihood_evaluations": expected_calls,
        "transport_attempted": attempted,
        "transport_accepted": accepted,
        "orbit_attempted": orbit_attempted,
        "orbit_accepted": orbit_accepted,
        "full_matched_newtonian_run_launched": False,
        "production_likelihood_evaluations": 0,
        "scientific_adjudication": False,
        "real_rows": 0,
    }


def write_unauthorized(
    expected_config_sha256: str, output: Path
) -> dict[str, Any]:
    if confined(output) != ROOT / UNAUTHORIZED_PATH:
        raise RuntimeError("current unauthorized manifest path changed")
    body = {
        "schema_version": UNAUTHORIZED_SCHEMA,
        "status": "external_approval_required",
        "production_authorized": False,
        "run_request": run_request(),
        "frozen_artifacts": frozen_authorization_bindings(expected_config_sha256),
        "external_approval": None,
        "production_runs": 0,
    }
    body["content_sha256"] = content_sha256(body)
    sampler.write_json(output, body)
    return body


def validate_unauthorized(
    path: Path, expected_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("Newtonian V2 unauthorized manifest missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256", None)
    if (
        observed != content_sha256(unhashed)
        or unhashed
        != {
            "schema_version": UNAUTHORIZED_SCHEMA,
            "status": "external_approval_required",
            "production_authorized": False,
            "run_request": run_request(),
            "frozen_artifacts": frozen_authorization_bindings(
                expected_config_sha256
            ),
            "external_approval": None,
            "production_runs": 0,
        }
    ):
        raise RuntimeError("Newtonian V2 unauthorized manifest changed")
    return body


def validate_external_approval(
    path: Path, expected_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("exact external approval record missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "approved_by",
            "approval_id",
            "production_authorized",
            "run_request",
            "frozen_artifacts",
        },
        "external approval",
    )
    if (
        body["schema_version"] != APPROVAL_SCHEMA
        or body["approved_by"] != "Henry"
        or not isinstance(body["approval_id"], str)
        or not body["approval_id"].strip()
        or body["production_authorized"] is not True
        or body["run_request"] != run_request()
        or body["frozen_artifacts"]
        != frozen_authorization_bindings(expected_config_sha256)
    ):
        raise RuntimeError("external approval does not exactly authorize frozen V2 run")
    return body


def promote_authorization(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    external_approval_path: Path,
    expected_external_approval_sha256: str,
    expected_config_sha256: str,
    output: Path,
) -> dict[str, Any]:
    unauthorized = validate_unauthorized(
        unauthorized_path, expected_unauthorized_sha256, expected_config_sha256
    )
    approval = validate_external_approval(
        external_approval_path,
        expected_external_approval_sha256,
        expected_config_sha256,
    )
    body = {
        "schema_version": AUTHORIZED_SCHEMA,
        "status": "exact_external_approval_verified_not_consumed",
        "production_authorized": True,
        "run_request": run_request(),
        "frozen_artifacts": unauthorized["frozen_artifacts"],
        "external_approval": {
            "path": confined(external_approval_path).relative_to(ROOT).as_posix(),
            "file_sha256": expected_external_approval_sha256,
            "approval_id": approval["approval_id"],
        },
        "production_runs": 0,
    }
    body["content_sha256"] = content_sha256(body)
    sampler.write_json(output, body)
    return body


def validate_authorization(
    path: Path, expected_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("exact Newtonian V2 authorization missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256", None)
    if observed != content_sha256(unhashed):
        raise RuntimeError("Newtonian V2 authorization content hash changed")
    strict_keys(
        unhashed,
        {
            "schema_version",
            "status",
            "production_authorized",
            "run_request",
            "frozen_artifacts",
            "external_approval",
            "production_runs",
        },
        "authorized manifest",
    )
    if (
        unhashed["schema_version"] != AUTHORIZED_SCHEMA
        or unhashed["status"] != "exact_external_approval_verified_not_consumed"
        or unhashed["production_authorized"] is not True
        or unhashed["run_request"] != run_request()
        or unhashed["frozen_artifacts"]
        != frozen_authorization_bindings(expected_config_sha256)
        or unhashed["production_runs"] != 0
    ):
        raise RuntimeError("Newtonian V2 production remains externally unauthorized")
    approval_binding = unhashed["external_approval"]
    strict_keys(
        approval_binding, {"path", "file_sha256", "approval_id"}, "approval binding"
    )
    approval = validate_external_approval(
        ROOT / approval_binding["path"],
        approval_binding["file_sha256"],
        expected_config_sha256,
    )
    if approval["approval_id"] != approval_binding["approval_id"]:
        raise RuntimeError("external approval id changed")
    return body


def authorization_controls(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    validate_unauthorized(
        unauthorized_path, expected_unauthorized_sha256, expected_config_sha256
    )
    negative = {}
    for label, override in (
        ("wrong_approver", {"approved_by": "not-Henry"}),
        ("empty_approval_id", {"approval_id": ""}),
        (
            "extra_call",
            {
                "run_request": {
                    **run_request(),
                    "maximum_newtonian_control_likelihood_evaluations": (
                        MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS + 1
                    ),
                }
            },
        ),
        ("nonzero_cost", {"run_request": {**run_request(), "maximum_paid_external_cost_usd": 0.01}}),
    ):
        approval = {
            "schema_version": APPROVAL_SCHEMA,
            "approved_by": "Henry",
            "approval_id": "DISPOSABLE-CONTROL-ONLY",
            "production_authorized": True,
            "run_request": run_request(),
            "frozen_artifacts": frozen_authorization_bindings(
                expected_config_sha256
            ),
        }
        approval.update(override)
        with tempfile.TemporaryDirectory(prefix="newtonian-v2-auth-", dir=ARTIFACT_DIR) as directory:
            path = Path(directory) / "approval.json"
            sampler.write_json(path, approval)
            try:
                validate_external_approval(
                    path, file_sha256(path), expected_config_sha256
                )
            except RuntimeError:
                negative[label] = True
            else:
                negative[label] = False
    return {
        "schema_version": "invariant-gravity-nuisance-quotient-newtonian-authorization-controls-2.0",
        "passed": all(negative.values()),
        "negative_controls": negative,
        "current_manifest_schema": UNAUTHORIZED_SCHEMA,
        "current_production_authorized": False,
        "authorized_manifest_persisted": False,
        "production_runs": 0,
    }


def _production_pass(summary: dict[str, Any], draws: np.ndarray, truth: np.ndarray) -> tuple[bool, dict[str, Any]]:
    lower, median, upper = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
    standard = np.std(draws, axis=0, ddof=1)
    median_z = np.abs(median - truth) / np.maximum(standard, np.finfo(float).tiny)
    covered = (lower <= truth) & (truth <= upper)
    gates = sbc_v3.GATES
    diagnostics_passed = bool(
        summary["all_coordinates_diagnostic_valid"]
        and summary["maximum_rhat"] is not None
        and summary["maximum_rhat"] <= gates["maximum_rank_normalized_split_rhat"]
        and summary["minimum_bulk_ess"]
        >= gates["minimum_bulk_effective_samples_per_valid_coordinate"]
        and summary["minimum_tail_ess"]
        >= gates["minimum_tail_effective_samples_per_valid_coordinate"]
    )
    recovery = {
        "coordinates_with_truth_in_marginal_95_interval": int(np.count_nonzero(covered)),
        "required_coordinates_with_truth_in_marginal_95_interval": 8,
        "maximum_absolute_posterior_median_z": float(np.max(median_z)),
        "maximum_allowed_absolute_posterior_median_z": 3.0,
        "single_dataset_false_selection_rate_measured": False,
    }
    return bool(
        diagnostics_passed
        and recovery["coordinates_with_truth_in_marginal_95_interval"] >= 8
        and recovery["maximum_absolute_posterior_median_z"] <= 3.0
    ), recovery


def execute_production(
    authorization_path: Path,
    expected_authorization_sha256: str,
    config_path: Path,
    expected_config_sha256: str,
    output: Path,
) -> dict[str, Any]:
    # Chronology is security-relevant: neither config nor predictor is read first.
    adjudication = validate_strict_v3_adjudicator()
    authorization = validate_authorization(
        authorization_path, expected_authorization_sha256, expected_config_sha256
    )
    if confined(output) != ROOT / PRODUCTION_RESULT_PATH:
        raise RuntimeError("Newtonian V2 production output path changed")
    contract = load_contract(config_path, expected_config_sha256)
    prior_config = uncertainty.load_config(ROOT)
    packet = load_predictors(
        ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    observation, precision, _targets = sufficient_observation(packet, prior_config)
    draws, fit_summary, calls = sbc_v3.run_candidate_fit(
        observation, precision, prior_config, global_index=0
    )
    if calls != MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS:
        raise RuntimeError("Newtonian V2 exact call accounting changed")
    truth = sampler.composite_values(TRUTH_UNIT, prior_config)
    passed, recovery = _production_pass(fit_summary, draws, truth)
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "matched_newtonian_control_completed_not_candidate_physics",
        "decision": (
            "MATCHED_NEWTONIAN_CONTROL_PASSED"
            if passed
            else "MATCHED_NEWTONIAN_CONTROL_FAILED_RESULT_RETAINED"
        ),
        "passed": passed,
        "config_sha256": expected_config_sha256,
        "strict_adjudicator": adjudication,
        "authorization": {
            "path": confined(authorization_path).relative_to(ROOT).as_posix(),
            "file_sha256": expected_authorization_sha256,
            "approval_id": authorization["external_approval"]["approval_id"],
        },
        "fit_summary": fit_summary,
        "truth_recovery": recovery,
        "call_accounting": {
            "actual_likelihood_evaluations": calls,
            "frozen_maximum_likelihood_evaluations": (
                MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS
            ),
            "real_forward_model_evaluations": 0,
        },
        "data_boundary": {
            **contract["data_boundary"],
            "production_runs": 1,
        },
        "claim_boundary": {
            **contract["claim_boundary"],
            "bounded_controls_only": False,
            "full_matched_newtonian_run_completed": True,
        },
    }

    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            posterior_composites=draws,
            injected_truth=truth,
            sufficient_observation=observation,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    sampler._write_then_publish_no_clobber(output, writer, suffix=".npz.tmp")
    return summary


def implementation_receipt(
    config_path: Path,
    expected_config_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    unauthorized_path: Path,
    authorization_controls_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(config_path, expected_config_sha256)
    controls = json.loads(confined(controls_path).read_text(encoding="utf-8"))
    smoke = json.loads(confined(smoke_path).read_text(encoding="utf-8"))
    unauthorized = validate_unauthorized(
        unauthorized_path, file_sha256(confined(unauthorized_path)), expected_config_sha256
    )
    authorization_control = json.loads(
        confined(authorization_controls_path).read_text(encoding="utf-8")
    )
    if controls != controls_receipt(contract) or smoke.get("passed") is not True:
        raise RuntimeError("bounded Newtonian V2 evidence failed")
    if unauthorized["production_authorized"] is not False:
        raise RuntimeError("current Newtonian V2 manifest unexpectedly authorized")
    if (
        authorization_control.get("passed") is not True
        or authorization_control.get("current_production_authorized") is not False
        or authorization_control.get("production_runs") != 0
    ):
        raise RuntimeError("Newtonian V2 authorization controls failed")
    result = {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "status": "package_prepared_strict_v3_pass_external_approval_required",
        "decision": "MATCHED_NEWTONIAN_CONTROL_V2_PREPARED_NOT_EXECUTED",
        "evidence": {
            "source": artifact_binding(Path(__file__)),
            "config": artifact_binding(config_path),
            "predictors": artifact_binding(ROOT / PREDICTOR_PATH),
            "controls": artifact_binding(controls_path),
            "smoke": artifact_binding(smoke_path),
            "unauthorized": artifact_binding(unauthorized_path),
            "authorization_controls": artifact_binding(authorization_controls_path),
            "strict_v3_adjudicator": strict_adjudicator_bindings(),
            "tests": artifact_binding(ROOT / TEST_PATH),
        },
        "run_request": run_request(),
        "gates": {
            "strict_v3_adjudicator_passed": True,
            "external_approval_present": False,
            "production_authorized": False,
            "full_matched_newtonian_run_completed": False,
        },
        "data_boundary": contract["data_boundary"],
        "claim_boundary": contract["claim_boundary"],
    }
    result["content_sha256"] = content_sha256(result)
    sampler.write_json(output, result)
    return result


def check(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    load_contract(config_path, expected_config_sha256)
    receipt = json.loads(confined(receipt_path).read_text(encoding="utf-8"))
    unhashed = dict(receipt)
    observed = unhashed.pop("content_sha256", None)
    if (
        observed != content_sha256(unhashed)
        or receipt.get("schema_version") != IMPLEMENTATION_SCHEMA
        or receipt.get("status")
        != "package_prepared_strict_v3_pass_external_approval_required"
        or receipt.get("decision")
        != "MATCHED_NEWTONIAN_CONTROL_V2_PREPARED_NOT_EXECUTED"
        or receipt.get("gates", {}).get("strict_v3_adjudicator_passed") is not True
        or receipt.get("gates", {}).get("production_authorized") is not False
        or receipt.get("gates", {}).get("full_matched_newtonian_run_completed")
        is not False
    ):
        raise RuntimeError("Newtonian V2 implementation receipt changed")
    for name, binding in receipt["evidence"].items():
        if name == "strict_v3_adjudicator":
            for nested, row in binding.items():
                validate_binding(row, f"receipt.evidence.{name}.{nested}")
        else:
            validate_binding(binding, f"receipt.evidence.{name}")
    validate_strict_v3_adjudicator()
    return {
        "valid": True,
        "status": receipt["status"],
        "strict_v3_adjudicator_passed": True,
        "external_approval_present": False,
        "production_authorized": False,
        "full_matched_newtonian_run_completed": False,
        "maximum_requested_likelihood_evaluations": (
            MAXIMUM_PRODUCTION_LIKELIHOOD_EVALUATIONS
        ),
        "maximum_paid_external_cost_usd": MAXIMUM_PAID_EXTERNAL_COST_USD,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    predictors = commands.add_parser("build-predictors")
    predictors.add_argument("--output", type=Path, required=True)
    for name in ("controls", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--expected-config-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    unauthorized = commands.add_parser("write-unauthorized")
    unauthorized.add_argument("--expected-config-sha256", required=True)
    unauthorized.add_argument("--output", type=Path, required=True)
    auth_controls = commands.add_parser("authorization-controls")
    auth_controls.add_argument("--unauthorized", type=Path, required=True)
    auth_controls.add_argument("--expected-unauthorized-sha256", required=True)
    auth_controls.add_argument("--expected-config-sha256", required=True)
    auth_controls.add_argument("--output", type=Path, required=True)
    promote = commands.add_parser("promote-authorization")
    promote.add_argument("--unauthorized", type=Path, required=True)
    promote.add_argument("--expected-unauthorized-sha256", required=True)
    promote.add_argument("--external-approval", type=Path, required=True)
    promote.add_argument("--expected-external-approval-sha256", required=True)
    promote.add_argument("--expected-config-sha256", required=True)
    promote.add_argument("--output", type=Path, required=True)
    receipt = commands.add_parser("write-implementation-receipt")
    receipt.add_argument("--config", type=Path, required=True)
    receipt.add_argument("--expected-config-sha256", required=True)
    receipt.add_argument("--controls", type=Path, required=True)
    receipt.add_argument("--smoke", type=Path, required=True)
    receipt.add_argument("--unauthorized", type=Path, required=True)
    receipt.add_argument("--authorization-controls", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    check_command = commands.add_parser("check")
    check_command.add_argument("--config", type=Path, required=True)
    check_command.add_argument("--expected-config-sha256", required=True)
    check_command.add_argument("--implementation-receipt", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--expected-authorization-sha256", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--expected-config-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute-frozen-newtonian-control-v2", action="store_true")
    args = parser.parse_args()

    if args.command == "build-predictors":
        result = write_predictors(args.output)
    elif args.command in {"controls", "smoke"}:
        contract = load_contract(args.config, args.expected_config_sha256)
        result = controls_receipt(contract) if args.command == "controls" else bounded_smoke(contract)
        sampler.write_json(args.output, result)
    elif args.command == "write-unauthorized":
        result = write_unauthorized(args.expected_config_sha256, args.output)
    elif args.command == "authorization-controls":
        result = authorization_controls(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.expected_config_sha256,
        )
        sampler.write_json(args.output, result)
    elif args.command == "promote-authorization":
        result = promote_authorization(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.external_approval,
            args.expected_external_approval_sha256,
            args.expected_config_sha256,
            args.output,
        )
    elif args.command == "write-implementation-receipt":
        result = implementation_receipt(
            args.config,
            args.expected_config_sha256,
            args.controls,
            args.smoke,
            args.unauthorized,
            args.authorization_controls,
            args.output,
        )
    elif args.command == "check":
        result = check(
            args.config, args.expected_config_sha256, args.implementation_receipt
        )
    else:
        if not args.execute_frozen_newtonian_control_v2:
            raise RuntimeError("explicit Newtonian-control V2 production sentinel required")
        result = execute_production(
            args.authorization,
            args.expected_authorization_sha256,
            args.config,
            args.expected_config_sha256,
            args.output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
