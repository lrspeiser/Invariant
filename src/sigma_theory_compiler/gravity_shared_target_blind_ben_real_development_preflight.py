"""Fail-closed preflight for the first development-only real B+E+N evaluation.

The requested evaluation was stopped before metric or score construction because
the only local row-level SPARC packet mixes exploration and reserved confirmation
objects.  This append-only module records that access boundary failure.  It never
opens that packet, any X-COP FITS target, or any group/lensing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_real_development_preflight_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_real_development_preflight.py")
RECEIPT_PATH = Path("runs/gravity/shared-target-blind-ben-real-development-preflight-v1.json")
LEDGER_PATH = Path(
    "runs/gravity/shared-target-blind-ben-real-development-preflight-v1/access-ledger.json"
)

CONFIG_SCHEMA = "invariant-gravity-ben-real-development-preflight-config-1.0"
LEDGER_SCHEMA = "invariant-gravity-ben-real-development-access-ledger-1.0"
RECEIPT_SCHEMA = "invariant-gravity-ben-real-development-preflight-receipt-1.0"
DECISION = "BLOCKED_MIXED_SPARC_CONFIRMATION_ACCESS_NO_REAL_BEN_SCORE"

SYNTHETIC_COMMIT = "e89c86d040b46a2dd0154c25adbeb8a24e3b0bc2"
SYNTHETIC_COMMIT_TIME = "2026-08-29T13:53:07-07:00"
REGISTRY_SHA256 = "45966eae73d7641ea982a7eea47aad883a9ff344baf121b91b901c32ef819f19"
MIXED_PATH = "configs/sparc_rotation_curves_full_v1.json"
MIXED_GIT_BLOB_SHA1 = "1dc2a31fabf5025065e37e5c8676c4fc91a525df"
MIXED_GIT_BLOB_BYTES = 247_315

EXACT_TEST_BINDING = {
    "path": "tests/test_gravity_shared_target_blind_ben_real_development_preflight.py",
    "file_sha256": "b5399423172f1810438fdc8ea0097016a1f9567fedade9c97b928df3e1a6556e",
}

EXACT_SOURCE_BINDINGS = {
    "synthetic_config": {
        "path": "configs/gravity_shared_target_blind_ben_synthetic_execution_v1.json",
        "file_sha256": "e5cf64b215873e286b2b16364b40b99a26fcb00bca7605063f4ddcad2066220d",
    },
    "synthetic_source": {
        "path": "src/sigma_theory_compiler/gravity_shared_target_blind_ben_synthetic_execution.py",
        "file_sha256": "6f1a7091c6a3d14ec6869c7ee30f8a38ff871343246d57b048883eda8a26fcc5",
    },
    "synthetic_test": {
        "path": "tests/test_gravity_shared_target_blind_ben_synthetic_execution.py",
        "file_sha256": "563da043471e41ef512a0db4fefc6067b054964cb3bb86b187f7503230d7072b",
    },
    "synthetic_receipt": {
        "path": "runs/gravity/shared-target-blind-ben-synthetic-execution-v1.json",
        "file_sha256": "05493ee23a86bd6f16e9eb460e3c175628f30d5946c3b6b6b7441fc31099db64",
        "content_sha256": "c5ecdfd36d0b940c3c11c9a92054c0c575ed3551d66034bdcb9d592fa2f702c3",
    },
    "sparc_exposure_receipt": {
        "path": "runs/gpu-baryonic-screen/per-object-decomposition-full-sparc-v1.json",
        "file_sha256": "8d6057b220d0312ee36c2cdcdc7c2fadff97c253ee7a9a415e0e741ed308f584",
        "content_sha256": "e37c6ab0c2548639a0540851a2355955e09d228e609aa327952ab8ae2605c497",
    },
    "item61_receipt": {
        "path": "runs/gravity/roadmap/item-61-cross-scale-gate-v1.json",
        "file_sha256": "bd8ce28f5f6ca5f39e1cb98ad753850155bfc32b92f18c5afbaaf213e1474fe8",
        "content_sha256": "b6969690ad11afd71a3c8034c5cc56005f7c6e657e7ec8b261abb1bcf6eb8e1d",
    },
}

EXACT_SCOPE = {
    "requested_real_evaluation_runs": 1,
    "completed_real_evaluation_runs": 0,
    "requested_sparc_exploration_objects": 139,
    "requested_sparc_exploration_rows": 2720,
    "requested_xcop_development_objects": 8,
    "sparc_confirmation_allowed": False,
    "little_things_independent_allowed": False,
    "xcop_confirmation_or_holdout_allowed": False,
    "group_allowed": False,
    "lensing_allowed": False,
    "inferred_total_mass_allowed": False,
}

EXACT_INCIDENT = {
    "incident_id": "BEN-REAL-V1-MIXED-SPARC-RG-ACCESS",
    "mixed_file_path": MIXED_PATH,
    "mixed_file_git_commit": SYNTHETIC_COMMIT,
    "mixed_file_git_blob_sha1": MIXED_GIT_BLOB_SHA1,
    "mixed_file_git_blob_bytes": MIXED_GIT_BLOB_BYTES,
    "command_class": "read_only_regex_search_rg_against_one_line_consolidated_json",
    "command_scope": [
        "runs/gpu-baryonic-screen/per-object-decomposition-full-sparc-v1.json",
        MIXED_PATH,
        "src/sigma_theory_compiler/sparc_full_sample.py",
    ],
    "regex_intent": "locate exploration/confirmation row-count and rotation-data interfaces",
    "one_line_json_match_emitted_entire_mixed_file_before_transport_truncation": True,
    "process_read_entire_mixed_file": True,
    "tool_output_was_truncated": True,
    "exact_raw_rows_visible_to_agent_reconstructable": False,
    "known_confirmation_objects_with_raw_rows_visible": [
        "D512-2",
        "DDO154",
        "ESO444-G084",
    ],
    "known_visible_confirmation_raw_row_lower_bound": 23,
    "exact_confirmation_raw_rows_surfaced": None,
    "confirmation_access_is_definitely_nonzero": True,
    "local_sparc_confirmation_remains_sealed_for_this_descendant": False,
}

EXACT_CHRONOLOGY = {
    "synthetic_commit_predates_mixed_file_access": True,
    "candidate_registry_frozen_before_mixed_file_access": True,
    "candidate_registry_raw_count_before_access": 240,
    "candidate_registry_equivalence_count_before_access": 60,
    "real_metric_definitions_frozen_before_access": 0,
    "real_thresholds_frozen_before_access": 0,
    "real_candidates_selected_before_access": 0,
    "real_candidate_scores_before_access": 0,
    "real_metric_definitions_after_access": 0,
    "real_thresholds_after_access": 0,
    "real_candidates_selected_after_access": 0,
    "real_candidate_scores_after_access": 0,
    "formula_repairs_after_access": 0,
}

EXACT_BOUNDARY = {
    "mixed_sparc_files_opened": 1,
    "mixed_sparc_file_bytes_read_by_process": MIXED_GIT_BLOB_BYTES,
    "sparc_real_metric_calls": 0,
    "sparc_real_candidate_score_calls": 0,
    "xcop_predictor_rows_read": 0,
    "xcop_target_rows_read": 0,
    "xcop_target_files_opened": 0,
    "xcop_real_metric_calls": 0,
    "xcop_real_candidate_score_calls": 0,
    "group_rows_read": 0,
    "group_score_calls": 0,
    "lensing_rows_read": 0,
    "lensing_score_calls": 0,
    "inferred_total_mass_rows_read": 0,
    "network_calls": 0,
    "model_calls": 0,
    "paid_calls": 0,
    "gpu_calls": 0,
    "cpu_formula_evaluation_calls": 0,
    "cpu_gpu_parity_calls": 0,
}

EXACT_RECOVERY = [
    {
        "option_id": "RECLASSIFY_LOCAL_SPARC_DEVELOPMENT_ONLY",
        "action": (
            "In a new version, explicitly reclassify every locally accessible SPARC row as "
            "development-only and reuse only already-frozen Item61 aggregate metrics; do not "
            "claim any local SPARC confirmation remains sealed."
        ),
        "authorized_here": False,
        "executed_here": False,
    },
    {
        "option_id": "OBTAIN_GENUINELY_EXTERNAL_CONFIRMATION",
        "action": (
            "Obtain and preregister a genuinely external confirmation source before any new "
            "candidate scoring."
        ),
        "authorized_here": False,
        "executed_here": False,
    },
]

EXACT_CLAIMS = {
    "blocked_preflight_only": True,
    "real_ben_evaluation_executed": False,
    "real_ben_candidate_supported_or_refuted": False,
    "candidate_ranking_created": False,
    "cross_domain_metric_created": False,
    "local_sparc_confirmation_sealed_for_descendant": False,
    "xcop_target_access_occurred": False,
    "group_or_lensing_access_occurred": False,
    "synthetic_registry_changed": False,
    "publication_readiness_changed": False,
    "scientific_claim_allowed": False,
}


class BENRealDevelopmentPreflightError(RuntimeError):
    """Raised when a frozen access-boundary fact or artifact changes."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENRealDevelopmentPreflightError(f"path escaped repository: {path}") from error
    return target


def strict_keys(value: Any, expected: set[str], label: str) -> None:
    actual = set(value) if isinstance(value, dict) else set()
    if actual != expected:
        raise BENRealDevelopmentPreflightError(
            f"{label} keys changed; missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def read_json(path: Path) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise BENRealDevelopmentPreflightError(f"required artifact absent: {path}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BENRealDevelopmentPreflightError(f"JSON artifact is not an object: {path}")
    return value


def validate_binding(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    expected = {"path", "file_sha256"}
    if "content_sha256" in binding:
        expected.add("content_sha256")
    strict_keys(binding, expected, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise BENRealDevelopmentPreflightError(f"{label} missing or changed")
    payload = read_json(target) if "content_sha256" in binding else {}
    if "content_sha256" in binding and payload.get("content_sha256") != binding["content_sha256"]:
        raise BENRealDevelopmentPreflightError(f"{label} content seal changed")
    return payload


def artifact_binding(path: Path, *, content: bool = False) -> dict[str, str]:
    target = confined(path)
    binding = {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }
    if content:
        payload = read_json(target)
        binding["content_sha256"] = str(payload["content_sha256"])
    return binding


def validate_contract(config: Mapping[str, Any]) -> None:
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "verifier_test",
            "source_bindings",
            "frozen_candidate_registry",
            "requested_scope",
            "access_incident",
            "chronology",
            "data_boundary",
            "recovery_options",
            "claim_boundary",
            "output_paths",
        },
        "real B+E+N blocked-preflight config",
    )
    exact = {
        "schema_version": CONFIG_SCHEMA,
        "status": "blocked_before_real_scoring_due_mixed_sparc_access_boundary_breach",
        "purpose": (
            "Retain the failed first real B+E+N launch attempt without scoring any target, "
            "because a mixed local SPARC packet exposed reserved confirmation rows."
        ),
        "frozen_candidate_registry": {
            "freeze_commit": SYNTHETIC_COMMIT,
            "freeze_commit_time": SYNTHETIC_COMMIT_TIME,
            "architecture_id": "BEN-additive-cross-scale-v1",
            "formula_template": "A_nuisance*E_local_base+B_continuous_gate*N_additive_channel",
            "A_role": "bounded_source_calibration_nuisance_only",
            "M_temporal_phase_operator_included": False,
            "raw_candidate_count": 240,
            "equivalence_class_count": 60,
            "candidate_registry_content_sha256": REGISTRY_SHA256,
            "candidate_proposals_may_only_come_from_committed_registry": True,
        },
        "requested_scope": EXACT_SCOPE,
        "access_incident": EXACT_INCIDENT,
        "chronology": EXACT_CHRONOLOGY,
        "data_boundary": EXACT_BOUNDARY,
        "recovery_options": EXACT_RECOVERY,
        "claim_boundary": EXACT_CLAIMS,
        "output_paths": {
            "access_ledger": LEDGER_PATH.as_posix(),
            "receipt": RECEIPT_PATH.as_posix(),
        },
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise BENRealDevelopmentPreflightError(f"frozen {key} changed")
    source = confined(ROOT / str(config["implementation_source"]))
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise BENRealDevelopmentPreflightError("implementation source changed")
    if config["verifier_test"] != EXACT_TEST_BINDING:
        raise BENRealDevelopmentPreflightError("frozen verifier_test changed")
    validate_binding(config["verifier_test"], "verifier_test")
    bindings = config["source_bindings"]
    strict_keys(
        bindings,
        {
            "synthetic_config",
            "synthetic_source",
            "synthetic_test",
            "synthetic_receipt",
            "sparc_exposure_receipt",
            "item61_receipt",
        },
        "source_bindings",
    )
    if bindings != EXACT_SOURCE_BINDINGS:
        raise BENRealDevelopmentPreflightError("frozen source_bindings changed")
    for label, binding in bindings.items():
        payload = validate_binding(binding, f"source_bindings.{label}")
        if label == "synthetic_receipt":
            registry = payload.get("candidate_registry", {})
            if (
                registry.get("content_sha256") != REGISTRY_SHA256
                or registry.get("raw_candidate_count") != 240
                or registry.get("equivalence_class_count") != 60
            ):
                raise BENRealDevelopmentPreflightError("synthetic candidate registry changed")
        elif label == "sparc_exposure_receipt":
            if (
                payload.get("counts")
                != {
                    "admitted_galaxies": 174,
                    "confirmation_galaxies_declared": 35,
                    "exploration_galaxies_fitted": 139,
                    "exploration_points_fitted": 2720,
                    "law_spaces": 14,
                    "one_parameter_fits": 1946,
                    "published_galaxies": 175,
                    "published_points": 3391,
                    "widened_from_galaxies": 6,
                    "widened_from_points": 214,
                }
                or payload.get("claims", {}).get("confirmation_set_fitted") is not False
            ):
                raise BENRealDevelopmentPreflightError("SPARC exposure boundary changed")
        elif label == "item61_receipt":
            if (
                payload.get("decision")
                != "ITEM61_CROSS_SCALE_GATE_NOT_PASSED_EXACT_PARAMETERIZATION_RETAINED"
                or payload.get("counts", {}).get("sparc_rows") != 2720
                or payload.get("claims", {}).get("universal_cross_scale_gate_passed") is not False
            ):
                raise BENRealDevelopmentPreflightError("Item61 retained result changed")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if target != ROOT / CONFIG_PATH:
        raise BENRealDevelopmentPreflightError("config path changed")
    if file_sha256(target) != expected_sha256:
        raise BENRealDevelopmentPreflightError("config hash changed")
    config = read_json(target)
    validate_contract(config)
    return config


def expected_access_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": LEDGER_SCHEMA,
        "status": "mixed_sparc_confirmation_access_breach_retained_fail_closed",
        "decision": DECISION,
        "incident": config["access_incident"],
        "chronology": config["chronology"],
        "data_boundary": config["data_boundary"],
        "consequence": {
            "real_evaluation_aborted_before_metric_or_score": True,
            "local_sparc_reserved_confirmation_cannot_be_claimed_sealed_for_this_descendant": True,
            "candidate_registry_remains_the_committed_240_raw_60_equivalence_classes": True,
            "no_candidate_failure_or_success_exists_to_retain": True,
        },
        "recovery_options": config["recovery_options"],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def expected_receipt(config: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "blocked_preflight_retained_no_real_evaluation",
        "decision": DECISION,
        "evidence": {
            "source": artifact_binding(Path(__file__)),
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "tests": artifact_binding(ROOT / TEST_PATH),
            "access_ledger": artifact_binding(ROOT / LEDGER_PATH, content=True),
            **{label: dict(binding) for label, binding in config["source_bindings"].items()},
        },
        "access_ledger_content_sha256": ledger["content_sha256"],
        "frozen_candidate_registry": config["frozen_candidate_registry"],
        "requested_scope": config["requested_scope"],
        "chronology": config["chronology"],
        "data_boundary": config["data_boundary"],
        "outcome": {
            "real_metric_definitions": 0,
            "real_thresholds": 0,
            "real_candidates_selected": 0,
            "real_candidates_scored": 0,
            "real_domain_scores": {},
            "joint_scores": {},
            "ablations_scored": 0,
            "matched_controls_scored": 0,
            "whole_object_folds_run": 0,
            "formula_repairs": 0,
            "failures_or_ties_retained": 0,
        },
        "recovery_options": config["recovery_options"],
        "claim_boundary": config["claim_boundary"],
        "limitations": [
            "The committed 240/60 synthetic grammar predates the access incident, but no real metric or candidate selection was frozen.",
            "The mixed one-line SPARC JSON was read by a regex command and emitted before transport truncation, so the exact raw confirmation-row exposure cannot be reconstructed.",
            "No SPARC, X-COP, group, or lensing candidate score exists in this receipt.",
            "Neither recovery option is authorized or executed here.",
        ],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as error:
        raise BENRealDevelopmentPreflightError(
            f"atomic no-clobber publication refused existing artifact: {target}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write(config_path: Path, expected_config_sha256: str) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    ledger = expected_access_ledger(config)
    write_json_no_clobber(ROOT / LEDGER_PATH, ledger)
    receipt = expected_receipt(config, ledger)
    write_json_no_clobber(ROOT / RECEIPT_PATH, receipt)
    return receipt


def check(config_path: Path, expected_config_sha256: str, receipt_path: Path) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    ledger = expected_access_ledger(config)
    if read_json(ROOT / LEDGER_PATH) != ledger:
        raise BENRealDevelopmentPreflightError("access ledger does not reconstruct exactly")
    target = confined(receipt_path)
    if target != ROOT / RECEIPT_PATH:
        raise BENRealDevelopmentPreflightError("receipt path changed")
    receipt = read_json(target)
    if receipt != expected_receipt(config, ledger):
        raise BENRealDevelopmentPreflightError("receipt does not reconstruct exactly")
    for label, binding in receipt["evidence"].items():
        validate_binding(binding, f"receipt.evidence.{label}")
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "real_evaluation_executed": False,
        "real_candidates_scored": 0,
        "local_sparc_confirmation_sealed_for_descendant": False,
        "xcop_target_rows_read": 0,
        "scientific_claim_allowed": False,
        "receipt_sha256": file_sha256(target),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    writer = commands.add_parser("write")
    writer.add_argument("--config", type=Path, required=True)
    writer.add_argument("--expected-config-sha256", required=True)
    checker = commands.add_parser("check")
    checker.add_argument("--config", type=Path, required=True)
    checker.add_argument("--expected-config-sha256", required=True)
    checker.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write":
        result = write(args.config, args.expected_config_sha256)
    else:
        result = check(args.config, args.expected_config_sha256, args.receipt)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
