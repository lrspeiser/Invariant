"""Final evidence-bound adjudication of the 33-task full-3D theory-first roadmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_full3d_roadmap_adjudication_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_full3d_roadmap_adjudication_v1.py")
TEST_PATH = Path("tests/test_open_gravity_full3d_roadmap_adjudication_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-full3d-roadmap-adjudication-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_full3d_roadmap_adjudication_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_full3d_roadmap_adjudication_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_full3d_roadmap_adjudication_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-full3d-roadmap-adjudication-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "42597757d8a9d1b53d65127e44df9037334d5be82e3dba7145533ad382ed0fdd"
_CONFIG_CONTENT_SHA256 = "32b25e31fcbb2b0be492e968d4c34cf6eb8e4cdf200cf4682b28dcf77eb2be48"
_SCHEMA = "invariant-open-gravity-full3d-roadmap-adjudication-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-full3d-roadmap-adjudication-receipt-1.0"


class RoadmapAdjudicationError(RuntimeError):
    """Raised whenever a roadmap claim or binding fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoadmapAdjudicationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_ROOT / expected).resolve()
    _require(path.is_relative_to(_ROOT), f"{label} escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoadmapAdjudicationError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RoadmapAdjudicationError("committed evidence unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "evidence",
        "tasks",
        "required_checks",
        "next_decision",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-full3d-roadmap-adjudication-v1", "ID changed")
    _require(
        config["status"] == "FROZEN_FULL3D_THEORY_FIRST_ROADMAP_ADJUDICATION", "status changed"
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require([row["task"] for row in config["tasks"]] == list(range(1, 34)), "task ledger changed")
    _require(len(config["evidence"]) == 11, "evidence count changed")
    _require(len(config["required_checks"]) == 9, "checks changed")
    _require(config["next_decision"]["campaign_ready"] is False, "campaign decision changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "roadmap config")
    validate_config(config)
    for evidence in config["evidence"]:
        committed = _git_show(evidence["commit"], evidence["path"])
        _require(
            hashlib.sha256(committed).hexdigest() == evidence["sha256"],
            f"committed {evidence['id']} changed",
        )
        _require(
            file_sha256(_ROOT / evidence["path"]) == evidence["sha256"],
            f"working {evidence['id']} changed",
        )
    return config


def load_evidence(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for row in config["evidence"]:
        path = (_ROOT / row["path"]).resolve()
        _require(path.is_relative_to(_ROOT), "evidence escaped repository")
        _require(file_sha256(path) == row["sha256"], "evidence changed before read")
        receipts[row["id"]] = _read_json(path, row["id"])
    return receipts


def _gate(passed: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "metrics": dict(metrics)}


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(config)
    tasks = {row["task"]: row for row in config["tasks"]}
    gates: dict[str, dict[str, Any]] = {}
    radial = evidence["RADIAL_NEGATIVE_POSTRUN"]
    radial_counts = radial["counts"]
    radial_classes = radial["classifications"]
    gates["PRIOR_NEGATIVE_RESULT_PRESERVED"] = _gate(
        radial["status"] == "PASS_STABLE_REAGGREGATION_ZERO_SURVIVORS"
        and radial_counts["galaxies_scored"] == 139
        and radial_counts["clusters_scored"] == 8
        and radial_counts["dashboards"] == 147
        and radial_classes["stable_cross_domain_survivors"] == 0,
        {
            "galaxies": 139,
            "clusters": 8,
            "dashboards": 147,
            "galaxy_invalid_cells": radial_counts["galaxy_cells_source_gate_invalid"],
            "cluster_invalid_cells": radial_counts["cluster_cells_source_gate_invalid"],
            "cross_domain_survivors": 0,
        },
    )

    status_counts = dict(sorted(Counter(row["status"] for row in tasks.values()).items()))
    gates["EXACT_33_TASK_LEDGER"] = _gate(
        set(tasks) == set(range(1, 34)) and sum(status_counts.values()) == 33,
        {"tasks": 33, "status_counts": status_counts},
    )

    matrix = evidence["THEORY_GATE_MATRIX"]["matrix"]
    gates["THEORY_MATRIX_420_BY_25"] = _gate(
        matrix["mechanisms"] == 420 and matrix["gates"] == 25 and matrix["rows"] == 10_500,
        {
            "mechanisms": matrix["mechanisms"],
            "gates": matrix["gates"],
            "rows": matrix["rows"],
            "pass_target_free": matrix["status_counts"]["PASS_TARGET_FREE"],
            "required_unrun": matrix["status_counts"]["REQUIRED_UNRUN"],
        },
    )

    recovery = evidence["RECOVERY_FALSIFICATION"]["suite"]
    gates["SIX_EXECUTABLE_FIELDS_SYNTHETICALLY_FALSIFIABLE"] = _gate(
        recovery["mechanisms"] == 6 and recovery["passed"] == 11 and recovery["failed"] == 0,
        {
            "mechanisms": recovery["mechanisms"],
            "recovery_null_falsification_gates": recovery["passed"],
            "real_response_scoring_eligible": recovery["real_response_scoring_eligible"],
        },
    )

    closure = evidence["MULTISECTOR_CLOSURE"]["suite"]
    gates["MULTISECTOR_PARTIAL_BLOCKED_BOUNDARY"] = _gate(
        closure["sectors"] == 11
        and closure["status_counts"] == {"BLOCKED": 5, "PARTIAL": 6}
        and closure["observational_authority"] is False,
        {"sectors": 11, **closure["status_counts"], "observational_authority": False},
    )

    source = evidence["SOURCE_ACQUISITION_ENVIRONMENT"]["suite"]
    gates["ZERO_REAL_FULL3D_SOURCE_READY"] = _gate(
        source["objects"] == 147
        and source["full_3d_source_ready_objects"] == 0
        and source["campaign_ready"] is False,
        {"objects": 147, "full_3d_ready": 0, "campaign_ready": False},
    )

    gates["NO_REAL_CAMPAIGN_MANIFEST_OR_RESPONSE_RUN"] = _gate(
        tasks[27]["status"] == "COMPLETE_ZERO_ELIGIBLE"
        and all(tasks[index]["status"] == "NOT_RUN_CONDITIONAL_GATE" for index in range(28, 33))
        and tasks[33]["status"] == "COMPLETE_STOP"
        and config["next_decision"]["campaign_ready"] is False,
        {"eligible_entries": 0, "tasks_28_through_32_run": 0, "stopped_at_gate": True},
    )

    gates["ANTI_LOOP_RULE_ENFORCED"] = _gate(
        tasks[1]["status"] == "COMPLETE"
        and tasks[7]["status"] == "COMPLETE"
        and tasks[9]["status"] == "BLOCKED"
        and tasks[12]["status"] == "BLOCKED"
        and radial_classes["stable_cross_domain_survivors"] == 0,
        {
            "radial_campaign_reopened": False,
            "local_multiplier_promoted": False,
            "blocked_transport_repaired_post_hoc": False,
            "action_failure_erased": False,
        },
    )

    gates["ZERO_NEW_RESPONSE_ACCESS"] = _gate(
        all(value == 0 for value in config["access_contract"].values()),
        config["access_contract"],
    )
    _require(list(gates) == config["required_checks"], "check order changed")
    _require(all(row["passed"] is True for row in gates.values()), "roadmap check failed")
    return {
        "tasks": config["tasks"],
        "task_status_counts": status_counts,
        "checks": gates,
        "passed": len(gates),
        "failed": 0,
        "next_decision": config["next_decision"],
        "campaign_ready": False,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_THEORY_FIRST_FULL3D_ROADMAP_SOURCE_BLOCKED_STOP",
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / _CANONICAL_CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {"path": _CANONICAL_TEST_PATH.as_posix(), "sha256": file_sha256(test_path)},
            "evidence": config["evidence"],
        },
        "adjudication": run_suite(config),
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    _require(payload == build_receipt(), "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "roadmap receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "task_status_counts": receipt["adjudication"]["task_status_counts"],
                    "checks_passed": receipt["adjudication"]["passed"],
                    "campaign_ready": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
