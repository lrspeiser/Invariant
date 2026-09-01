from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_bound_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load hash-bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_bound_module(
    "invariant_v6r3_bound_sampler",
    Path(__file__).with_name("quotient_aware_sampler_v6r3_prototype.py"),
)

SCHEMA = "invariant-gravity-cluster-quotient-sampler-contract-6.4"
UNAUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-6.4-unauthorized"
AUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-6.4-authorized"
APPROVAL_SCHEMA = "invariant-gravity-quotient-sampler-external-approval-6.4"
AUTHORIZATION_CONTROLS_SCHEMA = (
    "invariant-gravity-quotient-sampler-authorization-transition-controls-6.4"
)
WRITE_RACE_CONTROLS_SCHEMA = "invariant-gravity-atomic-no-clobber-controls-6.4"
COMPOSITES = base.COMPOSITES
ACTIVE_INDICES = base.ACTIVE_INDICES
ORBIT_NAMES = base.ORBIT_NAMES
PRIMITIVE_PRIORS = base.PRIMITIVE_PRIORS
MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS = 1_575_104

SOURCE_PATHS = {
    "v6r3_sampler": "work/gravity/quotient_aware_sampler_v6r3_prototype.py",
    "v6r2_support": "work/gravity/quotient_aware_sampler_v6r2_prototype.py",
    "v6r1_kernel": "work/gravity/quotient_aware_sampler_v6_prototype.py",
    "uncertainty_config": "configs/gravity_cluster_uncertainty_program_v1.json",
    "uncertainty_module": "src/sigma_theory_compiler/gravity_cluster_uncertainty_program.py",
    "quotient_config": "configs/gravity_cluster_nuisance_quotient_audit_v1.json",
    "quotient_module": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_audit.py",
    "quotient_receipt": "runs/gravity/publication-readiness/nuisance-quotient-audit-v1.json",
    "comparator_module": "src/sigma_theory_compiler/gravity_cluster_comparator_suite.py",
    "comparator_receipt": "runs/gravity/publication-readiness/comparator-suite-v1.json",
    "item59_config": "configs/gravity_item59_xcop_forward_observable_gate_v1.json",
    "item59_module": "src/sigma_theory_compiler/gravity_item59_xcop_forward_observable_gate.py",
    "item59_result": "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json",
}

PRODUCTION_SETTINGS = base.PRODUCTION_SETTINGS
SMOKE_SETTINGS = base.SMOKE_SETTINGS
START_GENERATION = base.START_GENERATION
DATA_SEAL = base.DATA_SEAL
ORBIT_VALIDATION = base.ORBIT_VALIDATION
DIAGNOSTIC_VALIDATION = base.DIAGNOSTIC_VALIDATION
UNIFORM_TARGET_CONTROL = base.UNIFORM_TARGET_CONTROL
COMPLETION_THRESHOLDS = base.COMPLETION_THRESHOLDS
MECHANICS_THRESHOLDS = base.MECHANICS_THRESHOLDS
ADJUDICATION = base.ADJUDICATION
AUTHORIZATION_POLICY = {
    "unauthorized_schema": UNAUTHORIZED_SCHEMA,
    "authorized_schema": AUTHORIZED_SCHEMA,
    "external_approval_schema": APPROVAL_SCHEMA,
    "contract_status": "external_approval_required_before_production",
    "separate_status_and_boundary_validation": True,
    "external_approval_must_bind_all_frozen_artifacts": True,
    "production_authorized_by_default": False,
    "explicit_cli_sentinel_required": True,
    "unauthorized_attempt_fails_before_contract_or_runtime_packet_load": True,
    "all_generated_artifacts_use_atomic_same_filesystem_no_clobber": True,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def confined(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return resolved


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_artifact_binding(row: dict[str, Any], label: str) -> Path:
    strict_keys(row, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(row["path"]))
    if not target.is_file() or file_sha256(target) != row["file_sha256"]:
        raise RuntimeError(f"artifact binding missing or tampered: {label}")
    return target


def _publish_complete_temp_no_clobber(
    temporary_path: Path,
    destination: Path,
    *,
    before_link: Callable[[], None] | None = None,
) -> None:
    temporary = confined(temporary_path)
    target = confined(destination)
    if temporary.parent != target.parent:
        raise RuntimeError("atomic publication requires a same-directory temporary file")
    if before_link is not None:
        before_link()
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise RuntimeError("atomic no-clobber publication refused an existing target") from error


def _write_then_publish_no_clobber(
    destination: Path,
    writer: Callable[[Any], None],
    *,
    suffix: str,
    before_link: Callable[[], None] | None = None,
) -> None:
    target = confined(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".v6r4-complete-",
            suffix=suffix,
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        _publish_complete_temp_no_clobber(Path(temporary_name), target, before_link=before_link)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_json(
    path: Path,
    value: dict[str, Any],
    *,
    before_link: Callable[[], None] | None = None,
) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def writer(handle: Any) -> None:
        handle.write(payload)

    _write_then_publish_no_clobber(path, writer, suffix=".json.tmp", before_link=before_link)


def atomic_save_result(
    output: Path,
    *,
    traces: np.ndarray,
    ending_particles: np.ndarray,
    ending_log_likelihood: np.ndarray,
    summary: dict[str, Any],
    before_link: Callable[[], None] | None = None,
) -> None:
    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            composite_traces=traces,
            ending_particles=ending_particles,
            ending_log_likelihood=ending_log_likelihood,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    _write_then_publish_no_clobber(output, writer, suffix=".npz.tmp", before_link=before_link)


def validate_source_bindings(bindings: dict[str, Any]) -> None:
    strict_keys(bindings, set(SOURCE_PATHS), "source_bindings")
    for name, expected_path in SOURCE_PATHS.items():
        row = bindings[name]
        strict_keys(row, {"path", "file_sha256"}, f"source_bindings.{name}")
        if row["path"] != expected_path:
            raise RuntimeError(f"source binding path changed for {name}")
        validate_artifact_binding(row, f"source_bindings.{name}")


def maximum_forward_calls(settings: dict[str, Any]) -> int:
    return base.maximum_forward_calls(settings)


def expected_call_accounting() -> dict[str, Any]:
    return base.expected_call_accounting()


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    contract_path = confined(path)
    observed_hash = file_sha256(contract_path)
    if observed_hash != expected_sha256:
        raise RuntimeError("contract hash differs from the expected hash")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    strict_keys(
        contract,
        {
            "schema_version",
            "status",
            "purpose",
            "prototype_source",
            "prototype_source_normalized_sha256",
            "source_bindings",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "train_packet",
            "sobol_start_population",
            "start_generation",
            "family",
            "likelihood_split",
            "data_seal",
            "production_settings",
            "smoke_settings",
            "orbit_validation",
            "diagnostic_validation",
            "uniform_target_invariance_control",
            "completion_thresholds",
            "mechanics_thresholds",
            "call_accounting",
            "adjudication",
            "authorization_policy",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA:
        raise RuntimeError("contract schema changed")
    if contract["status"] != "external_approval_required_before_production":
        raise RuntimeError("contract status is not external_approval_required")
    if (
        contract["family"] != "cross_scale_boundary"
        or contract["likelihood_split"] != "development_train"
    ):
        raise RuntimeError("family or likelihood split changed")
    prototype = confined(ROOT / str(contract["prototype_source"]))
    if prototype != Path(__file__).resolve():
        raise RuntimeError("contract points to another executable")
    if normalized_sha256(prototype) != contract["prototype_source_normalized_sha256"]:
        raise RuntimeError("V6R4 executable changed after freeze")
    validate_source_bindings(contract["source_bindings"])
    if contract["exact_primitive_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("exact 17 primitive priors changed")
    if contract["primitive_prior_semantics"] != (
        "17_independent_uniform_primitives_with_clipped_six_factor_stellar_pushforward_clip_0.4_2.5"
    ):
        raise RuntimeError("primitive prior semantics changed")
    uncertainty_config = json.loads(
        (ROOT / SOURCE_PATHS["uncertainty_config"]).read_text(encoding="utf-8")
    )
    if uncertainty_config["continuous_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("hash-bound uncertainty prior definitions changed")
    for name in ("train_packet", "sobol_start_population"):
        validate_artifact_binding(contract[name], name)
    base.base.load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    base.base.validate_sobol_starts(
        ROOT / contract["sobol_start_population"]["path"],
        contract["sobol_start_population"]["file_sha256"],
    )
    expected_nested = {
        "start_generation": START_GENERATION,
        "data_seal": DATA_SEAL,
        "production_settings": PRODUCTION_SETTINGS,
        "smoke_settings": SMOKE_SETTINGS,
        "orbit_validation": ORBIT_VALIDATION,
        "diagnostic_validation": DIAGNOSTIC_VALIDATION,
        "uniform_target_invariance_control": UNIFORM_TARGET_CONTROL,
        "completion_thresholds": COMPLETION_THRESHOLDS,
        "mechanics_thresholds": MECHANICS_THRESHOLDS,
        "call_accounting": expected_call_accounting(),
        "adjudication": ADJUDICATION,
        "authorization_policy": AUTHORIZATION_POLICY,
    }
    for name, expected in expected_nested.items():
        if contract[name] != expected:
            raise RuntimeError(f"frozen nested contract object changed: {name}")
    base.validate_sampler_settings(contract["production_settings"], "production settings", 4, 512)
    base.validate_sampler_settings(contract["smoke_settings"], "smoke settings", 2, 32)
    contract["_execution_contract_sha256"] = observed_hash
    return contract


def run_sampler(
    contract: dict[str, Any], settings: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    original_saver = base.atomic_save_result

    def v6r4_saver(
        target: Path,
        *,
        traces: np.ndarray,
        ending_particles: np.ndarray,
        ending_log_likelihood: np.ndarray,
        summary: dict[str, Any],
    ) -> None:
        summary["schema_version"] = "invariant-gravity-cluster-quotient-sampler-result-6.4"
        summary["publication_semantics"] = {
            "same_filesystem": True,
            "atomic_commit_primitive": "hard_link_complete_temp_to_absent_destination",
            "destination_replacement_allowed": False,
            "temporary_cleanup_required": True,
        }
        atomic_save_result(
            target,
            traces=traces,
            ending_particles=ending_particles,
            ending_log_likelihood=ending_log_likelihood,
            summary=summary,
        )

    base.atomic_save_result = v6r4_saver
    try:
        result = base.run_sampler(contract, settings, output, smoke=smoke)
    finally:
        base.atomic_save_result = original_saver
    return result


def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    result = base.controls_receipt(contract)
    result["schema_version"] = "invariant-gravity-cluster-quotient-sampler-controls-6.4"
    result["artifact_publication"] = {
        "atomic_same_filesystem_no_clobber": True,
        "destination_replacement_allowed": False,
    }
    return result


def _concurrent_creator_race(kind: str) -> dict[str, Any]:
    creator_bytes = f"V6R4-{kind}-CONCURRENT-CREATOR-WON".encode("ascii")
    ready_for_creator = threading.Event()
    creator_finished = threading.Event()
    creator_error: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix=f"v6r4-{kind}-race-", dir=ROOT / "work" / "gravity"
    ) as temporary_directory:
        directory = Path(temporary_directory)
        destination = directory / ("race.npz" if kind == "npz" else "race.json")

        def creator() -> None:
            try:
                if not ready_for_creator.wait(timeout=10.0):
                    raise RuntimeError("publisher never reached the finalization barrier")
                with destination.open("xb") as handle:
                    handle.write(creator_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, RuntimeError) as error:
                creator_error.append(type(error).__name__)
            finally:
                creator_finished.set()

        creator_thread = threading.Thread(target=creator, daemon=True)
        creator_thread.start()

        def before_link() -> None:
            ready_for_creator.set()
            if not creator_finished.wait(timeout=10.0):
                raise RuntimeError("concurrent creator did not finish")

        publication_rejected = False
        try:
            if kind == "npz":
                atomic_save_result(
                    destination,
                    traces=np.zeros((1, 1, 4, 1)),
                    ending_particles=np.zeros((1, 1, 17)),
                    ending_log_likelihood=np.zeros((1, 1)),
                    summary={"schema_version": "v6r4-race-control"},
                    before_link=before_link,
                )
            elif kind == "json":
                write_json(
                    destination,
                    {"schema_version": "v6r4-race-control"},
                    before_link=before_link,
                )
            else:
                raise RuntimeError(f"unknown race-control kind: {kind}")
        except RuntimeError as error:
            publication_rejected = "no-clobber" in str(error)
        creator_thread.join(timeout=10.0)
        target_preserved = destination.read_bytes() == creator_bytes
        temporary_files = list(directory.glob(".v6r4-complete-*"))
        result = {
            "kind": kind,
            "passed": bool(
                publication_rejected
                and not creator_error
                and target_preserved
                and not temporary_files
            ),
            "creator_ran_after_complete_temp_before_final_link": True,
            "publication_rejected_existing_destination": publication_rejected,
            "concurrent_creator_target_bytes_preserved": target_preserved,
            "temporary_files_remaining": len(temporary_files),
            "creator_errors": creator_error,
        }
    result["temporary_directory_removed"] = not directory.exists()
    result["passed"] = bool(result["passed"] and result["temporary_directory_removed"])
    return result


def write_race_controls() -> dict[str, Any]:
    npz_control = _concurrent_creator_race("npz")
    json_control = _concurrent_creator_race("json")
    return {
        "schema_version": WRITE_RACE_CONTROLS_SCHEMA,
        "passed": bool(npz_control["passed"] and json_control["passed"]),
        "platform": sys.platform,
        "commit_primitive": "os.link_complete_same_directory_temp_to_absent_destination",
        "replacement_fallback": None,
        "npz_concurrent_creator": npz_control,
        "json_concurrent_creator": json_control,
    }


def validate_bound_controls_and_smoke(
    contract_hash: str, controls_path: Path, smoke_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls, summary = base.validate_bound_controls_and_smoke(
        contract_hash, controls_path, smoke_path
    )
    if controls.get("schema_version") != (
        "invariant-gravity-cluster-quotient-sampler-controls-6.4"
    ):
        raise RuntimeError("controls receipt is not V6R4")
    if summary.get("schema_version") != ("invariant-gravity-cluster-quotient-sampler-result-6.4"):
        raise RuntimeError("smoke receipt is not V6R4")
    publication = summary.get("publication_semantics", {})
    if publication != {
        "same_filesystem": True,
        "atomic_commit_primitive": "hard_link_complete_temp_to_absent_destination",
        "destination_replacement_allowed": False,
        "temporary_cleanup_required": True,
    }:
        raise RuntimeError("smoke publication semantics changed")
    return controls, summary


def frozen_artifact_bindings(
    contract: dict[str, Any], contract_path: Path, controls_path: Path, smoke_path: Path
) -> dict[str, dict[str, str]]:
    return {
        "contract": artifact_binding(contract_path),
        "prototype_source": artifact_binding(ROOT / contract["prototype_source"]),
        "train_packet": artifact_binding(ROOT / contract["train_packet"]["path"]),
        "sobol_start_population": artifact_binding(
            ROOT / contract["sobol_start_population"]["path"]
        ),
        "controls": artifact_binding(controls_path),
        "smoke": artifact_binding(smoke_path),
    }


def write_unauthorized_manifest(
    contract_path: Path,
    expected_contract_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path, expected_contract_sha256)
    validate_bound_controls_and_smoke(expected_contract_sha256, controls_path, smoke_path)
    body = {
        "schema_version": UNAUTHORIZED_SCHEMA,
        "status": "external_approval_required_controls_and_smoke_bound",
        "artifact_bindings": frozen_artifact_bindings(
            contract, contract_path, controls_path, smoke_path
        ),
        "production_authorization": {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        },
        "claim_boundary": {
            "controls_passed": True,
            "bounded_smoke_executed": True,
            "candidate_production_executed": False,
            "candidate_claim_allowed": False,
            "production_execution_allowed": False,
            "external_approval_required": True,
            "newtonian_control_unlocked": False,
            "simulation_based_calibration_unlocked": False,
        },
    }
    write_json(output, body)
    return body


def validate_frozen_artifact_bindings(bindings: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        bindings,
        {
            "contract",
            "prototype_source",
            "train_packet",
            "sobol_start_population",
            "controls",
            "smoke",
        },
        "authorization.artifact_bindings",
    )
    paths = {
        name: validate_artifact_binding(row, f"authorization.artifact_bindings.{name}")
        for name, row in bindings.items()
    }
    contract = load_contract(paths["contract"], bindings["contract"]["file_sha256"])
    if paths["prototype_source"] != Path(__file__).resolve():
        raise RuntimeError("authorization does not bind this V6R4 executable")
    validate_bound_controls_and_smoke(
        contract["_execution_contract_sha256"], paths["controls"], paths["smoke"]
    )
    if bindings["train_packet"] != contract["train_packet"]:
        raise RuntimeError("authorization train packet differs from contract")
    if bindings["sobol_start_population"] != contract["sobol_start_population"]:
        raise RuntimeError("authorization Sobol population differs from contract")
    return contract


def validate_unauthorized_body(body: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "production_authorization",
            "claim_boundary",
        },
        "unauthorized manifest",
    )
    if body["schema_version"] != UNAUTHORIZED_SCHEMA:
        raise RuntimeError("unauthorized manifest schema changed")
    if body["status"] != "external_approval_required_controls_and_smoke_bound":
        raise RuntimeError("unauthorized manifest status changed")
    if body["production_authorization"] != {
        "authorized": False,
        "approved_by": None,
        "approval_id": None,
        "maximum_forward_evaluations": 0,
    }:
        raise RuntimeError("unauthorized production fields changed")
    if body["claim_boundary"] != {
        "controls_passed": True,
        "bounded_smoke_executed": True,
        "candidate_production_executed": False,
        "candidate_claim_allowed": False,
        "production_execution_allowed": False,
        "external_approval_required": True,
        "newtonian_control_unlocked": False,
        "simulation_based_calibration_unlocked": False,
    }:
        raise RuntimeError("unauthorized claim boundary changed")
    return validate_frozen_artifact_bindings(body["artifact_bindings"])


def validate_external_approval(
    path: Path, expected_sha256: str, exact_bindings: dict[str, Any]
) -> dict[str, Any]:
    approval_path = confined(path)
    if not approval_path.is_file() or file_sha256(approval_path) != expected_sha256:
        raise RuntimeError("external approval record missing or tampered")
    body = json.loads(approval_path.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
            "artifact_bindings",
        },
        "external approval record",
    )
    if body["schema_version"] != APPROVAL_SCHEMA:
        raise RuntimeError("external approval schema changed")
    if body["status"] != "explicit_external_production_approval":
        raise RuntimeError("external approval status changed")
    if body["approved_by"] != "Henry":
        raise RuntimeError("external approval must be approved_by Henry")
    if not isinstance(body["approval_id"], str) or not body["approval_id"].strip():
        raise RuntimeError("external approval_id must be nonempty")
    if int(body["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS:
        raise RuntimeError("external approval maximum call count changed")
    if body["artifact_bindings"] != exact_bindings:
        raise RuntimeError("external approval does not bind exact frozen artifacts")
    return body


def promote_authorization(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    approval_path: Path,
    expected_approval_sha256: str,
    output: Path,
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("unauthorized manifest hash differs from expected hash")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    approval = validate_external_approval(
        approval_path, expected_approval_sha256, unauthorized["artifact_bindings"]
    )
    body = {
        "schema_version": AUTHORIZED_SCHEMA,
        "status": "production_explicitly_authorized_by_external_approval",
        "artifact_bindings": unauthorized["artifact_bindings"],
        "external_approval_binding": {
            **artifact_binding(approval_path),
            "approval_id": approval["approval_id"],
        },
        "production_authorization": {
            "authorized": True,
            "approved_by": "Henry",
            "approval_id": approval["approval_id"],
            "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        },
        "authorized_execution_boundary": {
            "production_execution_allowed": True,
            "external_approval_satisfied": True,
            "candidate_result_exists": False,
            "candidate_claim_allowed_before_result": False,
            "production_result_must_retain_failed_gates": True,
            "simulation_based_calibration_follows_candidate_pass": True,
            "matched_newtonian_control_follows_sbc": True,
            "source_covariance_follows_newtonian": True,
        },
    }
    write_json(output, body)
    return body


def validate_authorized_body(body: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "external_approval_binding",
            "production_authorization",
            "authorized_execution_boundary",
        },
        "authorized manifest",
    )
    if body["schema_version"] != AUTHORIZED_SCHEMA:
        raise RuntimeError("authorized manifest schema changed")
    if body["status"] != "production_explicitly_authorized_by_external_approval":
        raise RuntimeError("authorized manifest status changed")
    production = body["production_authorization"]
    strict_keys(
        production,
        {
            "authorized",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
        },
        "authorized manifest.production_authorization",
    )
    if (
        production["authorized"] is not True
        or production["approved_by"] != "Henry"
        or not isinstance(production["approval_id"], str)
        or not production["approval_id"].strip()
        or int(production["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
    ):
        raise RuntimeError("authorized production fields are incomplete")
    expected_boundary = {
        "production_execution_allowed": True,
        "external_approval_satisfied": True,
        "candidate_result_exists": False,
        "candidate_claim_allowed_before_result": False,
        "production_result_must_retain_failed_gates": True,
        "simulation_based_calibration_follows_candidate_pass": True,
        "matched_newtonian_control_follows_sbc": True,
        "source_covariance_follows_newtonian": True,
    }
    if body["authorized_execution_boundary"] != expected_boundary:
        raise RuntimeError("authorized execution boundary changed")
    contract = validate_frozen_artifact_bindings(body["artifact_bindings"])
    approval_binding = body["external_approval_binding"]
    strict_keys(
        approval_binding,
        {"path", "file_sha256", "approval_id"},
        "authorized manifest.external_approval_binding",
    )
    approval = validate_external_approval(
        ROOT / approval_binding["path"],
        approval_binding["file_sha256"],
        body["artifact_bindings"],
    )
    if (
        approval["approval_id"] != approval_binding["approval_id"]
        or approval["approval_id"] != production["approval_id"]
    ):
        raise RuntimeError("authorized approval identifiers disagree")
    return contract


def validate_authorization(
    path: Path, expected_sha256: str, *, require_production: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authorization_path = confined(path)
    if not authorization_path.is_file() or file_sha256(authorization_path) != expected_sha256:
        raise RuntimeError("authorization manifest hash differs from expected hash")
    body = json.loads(authorization_path.read_text(encoding="utf-8"))
    schema = body.get("schema_version")
    if schema == UNAUTHORIZED_SCHEMA:
        if require_production:
            raise RuntimeError(
                "external approval is required; refusal occurs before contract or packet load"
            )
        return body, validate_unauthorized_body(body)
    if schema == AUTHORIZED_SCHEMA:
        return body, validate_authorized_body(body)
    raise RuntimeError("authorization manifest has no recognized V6R4 schema")


def injected_approval_body(bindings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body = {
        "schema_version": APPROVAL_SCHEMA,
        "status": "explicit_external_production_approval",
        "approved_by": "Henry",
        "approval_id": "V6R4-AUTHORIZATION-TRANSITION-CONTROL-ONLY",
        "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        "artifact_bindings": bindings,
    }
    body.update(overrides)
    return body


def _authorization_transition_once(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("authorization-control input hash changed")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    negative_results: dict[str, bool] = {}
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="v6r4-authorization-control-", dir=ROOT / "work" / "gravity"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)

        def expect_rejection(name: str, approval: dict[str, Any]) -> None:
            approval_path = temporary_path / f"{name}-approval.json"
            output_path = temporary_path / f"{name}-authorized.json"
            write_json(approval_path, approval)
            try:
                promote_authorization(
                    unauthorized_target,
                    expected_unauthorized_sha256,
                    approval_path,
                    file_sha256(approval_path),
                    output_path,
                )
            except RuntimeError:
                negative_results[name] = not output_path.exists()
                return
            negative_results[name] = False

        bindings = unauthorized["artifact_bindings"]
        expect_rejection(
            "wrong_approver", injected_approval_body(bindings, approved_by="Not Henry")
        )
        expect_rejection("empty_approval_id", injected_approval_body(bindings, approval_id=""))
        expect_rejection(
            "wrong_maximum_calls",
            injected_approval_body(
                bindings,
                maximum_forward_evaluations=MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS - 1,
            ),
        )
        tampered_bindings = json.loads(json.dumps(bindings))
        tampered_bindings["smoke"]["file_sha256"] = "0" * 64
        expect_rejection("wrong_artifact_binding", injected_approval_body(tampered_bindings))

        approval_path = temporary_path / "positive-control-external-approval.json"
        authorized_path = temporary_path / "positive-control-authorized.json"
        write_json(approval_path, injected_approval_body(bindings))
        promote_authorization(
            unauthorized_target,
            expected_unauthorized_sha256,
            approval_path,
            file_sha256(approval_path),
            authorized_path,
        )
        validated, contract = validate_authorization(
            authorized_path,
            file_sha256(authorized_path),
            require_production=True,
        )
        positive_passed = bool(
            validated["schema_version"] == AUTHORIZED_SCHEMA
            and validated["status"] == "production_explicitly_authorized_by_external_approval"
            and validated["production_authorization"]["authorized"] is True
            and contract is not None
        )
    temporary_removed = bool(temporary_path is not None and not temporary_path.exists())
    return {
        "negative_controls": negative_results,
        "positive_disposable_authorized_control": {
            "passed": positive_passed,
            "authorized_schema": AUTHORIZED_SCHEMA,
            "authorized_status": ("production_explicitly_authorized_by_external_approval"),
            "approval_logical_id": ("V6R4-AUTHORIZATION-TRANSITION-CONTROL-ONLY"),
            "approved_by": "Henry",
            "maximum_forward_evaluations": (MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS),
            "frozen_artifact_binding_count": 6,
            "manifest_disposable": True,
            "production_launched": False,
        },
        "temporary_artifacts_removed": temporary_removed,
        "production_runs": 0,
    }


def authorization_transition_controls(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    first = _authorization_transition_once(unauthorized_path, expected_unauthorized_sha256)
    second = _authorization_transition_once(unauthorized_path, expected_unauthorized_sha256)
    replay_equal = first == second
    passed = bool(
        all(first["negative_controls"].values())
        and first["positive_disposable_authorized_control"]["passed"]
        and first["temporary_artifacts_removed"]
        and first["production_runs"] == 0
        and replay_equal
    )
    return {
        "schema_version": AUTHORIZATION_CONTROLS_SCHEMA,
        "passed": passed,
        **first,
        "exact_replay_equality": replay_equal,
        "volatile_physical_temp_paths_or_hashes_in_receipt": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("controls", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)

    race_command = subparsers.add_parser("write-race-controls")
    race_command.add_argument("--output", type=Path, required=True)

    unauthorized_command = subparsers.add_parser("write-unauthorized")
    unauthorized_command.add_argument("--contract", type=Path, required=True)
    unauthorized_command.add_argument("--expected-contract-sha256", required=True)
    unauthorized_command.add_argument("--controls", type=Path, required=True)
    unauthorized_command.add_argument("--smoke", type=Path, required=True)
    unauthorized_command.add_argument("--output", type=Path, required=True)

    promote_command = subparsers.add_parser("promote-authorization")
    promote_command.add_argument("--unauthorized", type=Path, required=True)
    promote_command.add_argument("--expected-unauthorized-sha256", required=True)
    promote_command.add_argument("--external-approval", type=Path, required=True)
    promote_command.add_argument("--expected-external-approval-sha256", required=True)
    promote_command.add_argument("--output", type=Path, required=True)

    validate_command = subparsers.add_parser("validate-authorization")
    validate_command.add_argument("--authorization", type=Path, required=True)
    validate_command.add_argument("--expected-authorization-sha256", required=True)
    validate_command.add_argument("--require-production", action="store_true")

    transition_command = subparsers.add_parser("authorization-controls")
    transition_command.add_argument("--unauthorized", type=Path, required=True)
    transition_command.add_argument("--expected-unauthorized-sha256", required=True)
    transition_command.add_argument("--output", type=Path, required=True)

    run_command = subparsers.add_parser("run")
    run_command.add_argument("--authorization", type=Path, required=True)
    run_command.add_argument("--expected-authorization-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--execute-frozen-production-v6r4", action="store_true")

    args = parser.parse_args()
    if args.command in {"controls", "smoke"}:
        contract = load_contract(args.contract, args.expected_contract_sha256)
        if args.command == "controls":
            result = controls_receipt(contract)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
            if not result["passed"]:
                raise SystemExit(2)
            return
        result = run_sampler(contract, contract["smoke_settings"], args.output, smoke=True)
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "write-race-controls":
        result = write_race_controls()
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.command == "write-unauthorized":
        result = write_unauthorized_manifest(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "promote-authorization":
        result = promote_authorization(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.external_approval,
            args.expected_external_approval_sha256,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "validate-authorization":
        body, contract = validate_authorization(
            args.authorization,
            args.expected_authorization_sha256,
            require_production=args.require_production,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": body["schema_version"],
                    "status": body["status"],
                    "production_authorized": body["production_authorization"]["authorized"],
                    "execution_contract_sha256": (
                        contract["_execution_contract_sha256"] if contract is not None else None
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "authorization-controls":
        result = authorization_transition_controls(
            args.unauthorized, args.expected_unauthorized_sha256
        )
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if not args.execute_frozen_production_v6r4:
        raise RuntimeError(
            "production requires the explicit --execute-frozen-production-v6r4 sentinel"
        )
    _authorization, contract = validate_authorization(
        args.authorization,
        args.expected_authorization_sha256,
        require_production=True,
    )
    if contract is None:
        raise RuntimeError("authorized contract was not loaded")
    result = run_sampler(contract, contract["production_settings"], args.output, smoke=False)
    print(json.dumps(result, sort_keys=True))
    if not result["production_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
