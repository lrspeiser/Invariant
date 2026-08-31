"""Metadata-only 3-D source acquisition, environment, and history gap ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler import open_gravity_source_availability_contract_v2 as source_v2

CONFIG_PATH = Path("configs/open_gravity_3d_source_acquisition_environment_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_3d_source_acquisition_environment_v1.py")
TEST_PATH = Path("tests/test_open_gravity_3d_source_acquisition_environment_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-3d-source-acquisition-environment-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_3d_source_acquisition_environment_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_source_acquisition_environment_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_3d_source_acquisition_environment_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-3d-source-acquisition-environment-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "4f9e2f1435cf02107c42176625513c0e0e17015a8e6dd6118823de829509a829"
_CONFIG_CONTENT_SHA256 = "04c6e91e681ce2fcdc43e6902f9c9edfb4116789353deecc701701d76acec004"
_SCHEMA = "invariant-open-gravity-3d-source-acquisition-environment-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-3d-source-acquisition-environment-receipt-1.0"


class AcquisitionLedgerError(RuntimeError):
    """Raised when a source or authority boundary fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionLedgerError(message)


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
        raise AcquisitionLedgerError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AcquisitionLedgerError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "galaxy_products",
        "cluster_products",
        "forbidden_source_substitutes",
        "environment_history_contract",
        "upgrade_rules",
        "required_gates",
        "authority",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-3d-source-acquisition-environment-v1",
        "ID changed",
    )
    _require(
        config["status"] == "FROZEN_METADATA_ONLY_SOURCE_AND_ENVIRONMENT_GAPS", "status changed"
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(len(config["galaxy_products"]) == 6, "galaxy products changed")
    _require(len(config["cluster_products"]) == 5, "cluster products changed")
    _require(len(config["required_gates"]) == 12, "gates changed")
    _require(not any(config["authority"].values()), "authority changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "acquisition config")
    validate_config(config)
    for binding in config["bindings"]:
        for artifact in binding["artifacts"]:
            expected = artifact["sha256"]
            _require(
                hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
                == expected,
                f"committed {binding['role']} changed",
            )
            _require(
                file_sha256(_ROOT / artifact["path"]) == expected,
                f"working {binding['role']} changed",
            )
    return config


def build_object_ledger(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    _source_config, predecessor = source_v2.load_inputs()
    objects = predecessor["objects"]
    galaxies = objects["SPARC"]
    clusters = objects["XCOP"]
    _require(len(galaxies) == 139 and len(set(galaxies)) == 139, "SPARC object ledger changed")
    _require(len(clusters) == 8 and len(set(clusters)) == 8, "X-COP object ledger changed")
    stellar_available = set(objects["XCOP_stellar_profile_available"])
    stellar_missing = set(objects["XCOP_stellar_profile_missing"])
    _require(stellar_available | stellar_missing == set(clusters), "stellar split changed")
    galaxy_products = [row["id"] for row in config["galaxy_products"]]
    cluster_products = [row["id"] for row in config["cluster_products"]]
    rows: list[dict[str, Any]] = []
    for object_id in galaxies:
        row = {
            "domain": "SPARC",
            "object_id": object_id,
            "current_geometry": "RADIAL_SOURCE_CURVE_ONLY",
            "full_3d_status": "SOURCE_BLOCKED_MISSING_DEPTH",
            "environment_status": "SOURCE_BLOCKED_MISSING_ENVIRONMENT",
            "history_status": "SOURCE_BLOCKED_MISSING_HISTORY",
            "required_products": galaxy_products,
            "stellar_profile_1d_available": False,
            "response_access_authorized": False,
        }
        row["row_sha256"] = content_sha256(row)
        rows.append(row)
    for object_id in clusters:
        row = {
            "domain": "XCOP",
            "object_id": object_id,
            "current_geometry": "SPHERICAL_1D",
            "full_3d_status": "SPHERICAL_ONLY",
            "environment_status": "SOURCE_BLOCKED_MISSING_ENVIRONMENT",
            "history_status": "SOURCE_BLOCKED_MISSING_HISTORY",
            "required_products": cluster_products,
            "stellar_profile_1d_available": object_id in stellar_available,
            "response_access_authorized": False,
        }
        row["row_sha256"] = content_sha256(row)
        rows.append(row)
    return rows


def _stream_root(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _gate(passed: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "metrics": dict(metrics)}


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    ledger = build_object_ledger(config)
    galaxies = [row for row in ledger if row["domain"] == "SPARC"]
    clusters = [row for row in ledger if row["domain"] == "XCOP"]
    gates: dict[str, dict[str, Any]] = {}
    gates["EXACT_147_OBJECT_LEDGER"] = _gate(
        len(ledger) == 147 and len(galaxies) == 139 and len(clusters) == 8,
        {"objects": len(ledger), "SPARC": len(galaxies), "XCOP": len(clusters)},
    )
    full3d = sum(row["full_3d_status"] == "FULL_3D_SOURCE_READY" for row in ledger)
    gates["ZERO_CURRENT_FULL3D_READY_OBJECTS"] = _gate(
        full3d == 0,
        {
            "full_3d_source_ready": full3d,
            "galaxy_blocked_depth": len(galaxies),
            "cluster_spherical_only": len(clusters),
        },
    )
    galaxy_roles = {row["role"] for row in config["galaxy_products"]}
    gates["GALAXY_PRODUCT_ROLES_COMPLETE"] = _gate(
        galaxy_roles == {"SOURCE", "SOURCE_MODEL", "SOURCE_GEOMETRY", "SOURCE_ENVIRONMENT"},
        {"products": len(config["galaxy_products"]), "roles": sorted(galaxy_roles)},
    )
    cluster_roles = {row["role"] for row in config["cluster_products"]}
    gates["CLUSTER_PRODUCT_ROLES_COMPLETE"] = _gate(
        cluster_roles == {"SOURCE", "SOURCE_MODEL", "SOURCE_ENVIRONMENT"},
        {
            "products": len(config["cluster_products"]),
            "roles": sorted(cluster_roles),
            "stellar_profile_1d_available": sum(
                row["stellar_profile_1d_available"] for row in clusters
            ),
            "stellar_profile_1d_missing": sum(
                not row["stellar_profile_1d_available"] for row in clusters
            ),
        },
    )
    forbidden = config["forbidden_source_substitutes"]
    gates["SOURCE_RESPONSE_SEPARATION"] = _gate(
        len(forbidden) == 6
        and any("rotation residual" in value for value in forbidden)
        and any("lensing mass" in value for value in forbidden)
        and any("pressure or temperature" in value for value in forbidden),
        {"forbidden_substitutes": len(forbidden), "response_derived_source_allowed": False},
    )
    environment = config["environment_history_contract"]
    gates["ENVIRONMENT_LABELS_PRE_RESPONSE"] = _gate(
        len(environment["labels_required_before_response"]) == 6
        and environment["response_derived_environment_forbidden"] is True,
        {
            "labels": environment["labels_required_before_response"],
            "all_current_labels_available": False,
        },
    )
    gates["MATCHED_COVARIATES_AND_NEGATIVE_CONTROLS"] = _gate(
        len(environment["matched_covariates"]) == 7 and len(environment["negative_controls"]) == 7,
        {
            "matched_covariates": len(environment["matched_covariates"]),
            "negative_controls": len(environment["negative_controls"]),
        },
    )
    gates["HISTORY_REQUIRES_REAL_HISTORY_OR_DECLARED_SIMULATION"] = _gate(
        "source_time_series_or_simulation_id" in environment["history_fields"]
        and "model_based_flag" in environment["history_fields"]
        and all(row["history_status"] == "SOURCE_BLOCKED_MISSING_HISTORY" for row in ledger),
        {"history_fields": len(environment["history_fields"]), "currently_history_ready": 0},
    )
    root = _stream_root(ledger)
    gates["OBJECT_STATUS_HASH_ROOT"] = _gate(
        len(root) == 64 and len({row["row_sha256"] for row in ledger}) == 147,
        {"rows": 147, "stream_sha256": root, "unique_row_hashes": 147},
    )
    gates["ACQUISITION_AUTHORITY_WITHHELD"] = _gate(
        config["authority"]["network_acquisition_authorized"] is False
        and config["authority"]["response_access_authorized"] is False,
        config["authority"],
    )
    gates["CAMPAIGN_FREEZE_WITHHELD"] = _gate(
        config["authority"]["campaign_manifest_authorized"] is False and full3d == 0,
        {"campaign_manifest_authorized": False, "reason": "zero current full-3D-ready objects"},
    )
    gates["ZERO_RESPONSE_ACCESS"] = _gate(
        all(value == 0 for value in config["access_contract"].values()), config["access_contract"]
    )
    _require(list(gates) == config["required_gates"], "gate order changed")
    _require(all(row["passed"] is True for row in gates.values()), "acquisition ledger gate failed")
    return {
        "objects": 147,
        "object_ledger": ledger,
        "object_ledger_stream_sha256": root,
        "gates": gates,
        "passed": len(gates),
        "failed": 0,
        "full_3d_source_ready_objects": 0,
        "campaign_ready": False,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_METADATA_ONLY_147_OBJECT_SOURCE_GAP_LEDGER_CAMPAIGN_WITHHELD",
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
            "predecessors": config["bindings"],
        },
        "suite": run_suite(config),
        "galaxy_products": config["galaxy_products"],
        "cluster_products": config["cluster_products"],
        "environment_history_contract": config["environment_history_contract"],
        "authority": config["authority"],
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
    validate_receipt_payload(_read_json(_output_path(), "acquisition ledger receipt"))


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
                    "objects": receipt["suite"]["objects"],
                    "full_3d_ready": receipt["suite"]["full_3d_source_ready_objects"],
                    "campaign_ready": receipt["suite"]["campaign_ready"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
