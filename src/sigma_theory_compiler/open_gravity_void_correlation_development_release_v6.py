"""Source-free v6 executor contract integrating audited VAST1 and VAST2 parsers."""

from __future__ import annotations

import argparse
import ast
import ctypes
import gzip
import hashlib
import inspect
import json
import marshal
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from . import open_gravity_void_correlation_development_release_v1 as v1
from . import open_gravity_void_correlation_development_release_v2 as v2
from . import open_gravity_void_correlation_development_release_v3 as v3
from . import open_gravity_void_correlation_development_release_v4 as v4
from . import open_gravity_void_correlation_development_release_v5 as v5
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1
from . import open_gravity_void_vast1_source_parser_contract_v1 as vast1_contract
from . import open_gravity_void_vast2_source_parser_contract_v2 as vast2_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v6.json"
MODULE_PATH = (
    REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v6.py"
)
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v6.py"
OUTPUT_PATH = (
    REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v6/receipt.json"
)
FINAL_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v6"
STAGING_ROOT = REPO_ROOT / "work/open-gravity-void-correlation-development-score-v6-staging"
FAILURE_DIRECTORY = (
    REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v6-failures"
)
CONSUMPTION_DIRECTORY = (
    REPO_ROOT
    / "runs/gravity/open-gravity-void-correlation-development-release-v6-authorization-consumed"
)

_CONFIG_RAW_SHA256 = "48e02b1648791b0ada32398fe4b12c9dc6dc3a72b355cac66fb4609872b9d9e8"
_CONFIG_CONTENT_SHA256 = "cb599dbe1b9459dad3144ac1dd645d26319b65e0227276b7a1c3d1facc4e171f"
_MODULE_SEMANTIC_SHA256 = "2ce399c0e58b5e9809518261750bb62b6b5991fce51cac021b32def05c9a78e3"
_TEST_RAW_SHA256 = "16c20e9155acb6504047cff61aca9f39e5fa0a596bfdba11e561ced1aa98f54f"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_PERMUTATIONS = 10000
_ARTIFACT_NAMES = frozenset(v2._ARTIFACT_NAMES)


class DevelopmentReleaseV6Error(RuntimeError):
    """Fail-closed v6 freeze, gate, source, or owned-run error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV6Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_sha256(value: Any) -> str:
    return bytes_sha256(_canonical(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f'{name} = "') for name in _SELF_CONSTANTS):
            continue
        lines.append(line)
    return bytes_sha256("\n".join(lines).encode("utf-8"))


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def canonical_file(relative: str) -> Path:
    _require(
        isinstance(relative, str) and relative and "\\" not in relative, "invalid relative path"
    )
    _require(
        all(part not in {"", ".", ".."} for part in relative.split("/")), "unsafe relative path"
    )
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute(), "absolute path rejected")
    candidate = REPO_ROOT.joinpath(*pure.parts)
    _require(
        candidate.resolve(strict=True) == candidate and not candidate.is_symlink(),
        "path redirected",
    )
    _require(REPO_ROOT in candidate.parents, "path escapes repository")
    return candidate


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v6 config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "v6 config content drift")
    _require(
        value["status"]
        == "DRAFT_SOURCE_FREE_VAST1_VAST2_PARSER_INTEGRATED_EXECUTOR_AWAIT_INDEPENDENT_REAUDIT",
        "v6 config status drift",
    )
    _require(value["authority"]["scientific_runs_allowed"] == 0, "v6 run authority introduced")
    _require(
        value["authority"]["authorizations_may_be_consumed"] is False,
        "v6 consumption authority introduced",
    )
    return value


def _load_bound_json(section: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"v6 bound raw drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(
        value["content_sha256"] == section["content_sha256"],
        f"v6 bound content drift: {section['path']}",
    )
    _require(
        value["content_sha256"] == _self_hash(value), f"v6 bound self-hash drift: {section['path']}"
    )
    if "status" in section:
        _require(value["status"] == section["status"], f"v6 bound status drift: {section['path']}")
    return value


def validate_release_chain(config: Mapping[str, Any]) -> dict[str, str]:
    modules = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}
    for name, section in config["development_packet_chain"].items():
        module = modules[name]
        _require(
            file_sha256(canonical_file(section["config"]["path"]))
            == section["config"]["raw_sha256"],
            f"{name} config drift",
        )
        module_path = canonical_file(section["module"]["path"])
        _require(
            file_sha256(module_path) == section["module"]["raw_sha256"], f"{name} module raw drift"
        )
        _require(
            module.module_semantic_sha256(module_path) == section["module"]["semantic_sha256"],
            f"{name} semantic drift",
        )
        _require(
            file_sha256(canonical_file(section["test"]["path"])) == section["test"]["raw_sha256"],
            f"{name} test drift",
        )
        _load_bound_json(section["receipt"])

    v5_audit = _load_bound_json(config["development_v5_independent_audit"])
    _require(
        v5_audit["decision"] == config["development_v5_independent_audit"]["decision"],
        "v5 audit decision drift",
    )
    _require(v5_audit["scientific_run_authority"] is False, "v5 audit authority escalation")
    _require(
        v5_audit["subject_binding"]["receipt"]["self_sha256"]
        == config["development_packet_chain"]["v5"]["receipt"]["content_sha256"],
        "v5 audit subject drift",
    )

    parser = config["vast1_parser_contract"]
    parser_config_path = canonical_file(parser["config"]["path"])
    _require(
        file_sha256(parser_config_path) == parser["config"]["raw_sha256"],
        "VAST1 parser config raw drift",
    )
    _require(
        content_sha256(json.loads(parser_config_path.read_text()))
        == parser["config"]["content_sha256"],
        "VAST1 parser config content drift",
    )
    parser_module = canonical_file(parser["module"]["path"])
    _require(
        file_sha256(parser_module) == parser["module"]["raw_sha256"],
        "VAST1 parser module raw drift",
    )
    _require(
        vast1_contract.module_semantic_sha256(parser_module) == parser["module"]["semantic_sha256"],
        "VAST1 parser semantic drift",
    )
    _require(
        file_sha256(canonical_file(parser["test"]["path"])) == parser["test"]["raw_sha256"],
        "VAST1 parser test drift",
    )
    parser_receipt = _load_bound_json(parser["receipt"])
    audit = _load_bound_json(parser["independent_audit"])
    _require(
        audit["decision"] == parser["independent_audit"]["decision"], "VAST1 audit decision drift"
    )
    _require(
        audit["authority"]["scientific_development_runs_allowed"] == 0,
        "VAST1 audit scope escalation",
    )
    _require(
        audit["authority"]["executor_successor_packages_allowed"] == 1,
        "VAST1 audit successor count drift",
    )
    ledger_path = canonical_file(parser["ledger"]["path"])
    _require(file_sha256(ledger_path) == parser["ledger"]["raw_sha256"], "VAST1 ledger raw drift")
    _require(
        parser_receipt["row_disposition_root_sha256"] == parser["ledger"]["root_sha256"],
        "VAST1 ledger root drift",
    )

    parser2 = config["vast2_parser_contract"]
    parser2_config_path = canonical_file(parser2["config"]["path"])
    _require(
        file_sha256(parser2_config_path) == parser2["config"]["raw_sha256"],
        "VAST2 parser config raw drift",
    )
    _require(
        content_sha256(json.loads(parser2_config_path.read_text()))
        == parser2["config"]["content_sha256"],
        "VAST2 parser config content drift",
    )
    parser2_module = canonical_file(parser2["module"]["path"])
    _require(
        file_sha256(parser2_module) == parser2["module"]["raw_sha256"],
        "VAST2 parser module raw drift",
    )
    _require(
        vast2_contract.module_semantic_sha256(parser2_module)
        == parser2["module"]["semantic_sha256"],
        "VAST2 parser semantic drift",
    )
    _require(
        file_sha256(canonical_file(parser2["test"]["path"])) == parser2["test"]["raw_sha256"],
        "VAST2 parser test drift",
    )
    parser2_receipt = _load_bound_json(parser2["receipt"])
    parser2_summary = _load_bound_json(parser2["summary"])
    parser2_audit = _load_bound_json(parser2["independent_audit"])
    _require(
        parser2_audit["decision"] == parser2["independent_audit"]["decision"],
        "VAST2 audit decision drift",
    )
    _require(
        parser2_audit["authority"]["scientific_run_authority"] is False,
        "VAST2 audit authority escalation",
    )
    parser2_ledger_path = canonical_file(parser2["ledger"]["path"])
    _require(
        file_sha256(parser2_ledger_path) == parser2["ledger"]["raw_sha256"],
        "VAST2 ledger raw drift",
    )
    _require(
        parser2_receipt["artifacts"]["artifacts/vast2-row-dispositions.jsonl"]["content_sha256"]
        == parser2["ledger"]["content_sha256"],
        "VAST2 ledger content drift",
    )
    _require(
        parser2_receipt["row_disposition_root_sha256"]
        == parser2_summary["row_disposition_root_sha256"]
        == parser2["ledger"]["root_sha256"],
        "VAST2 ledger root drift",
    )
    _require(
        parser2_audit["subject"]["receipt"]["self_sha256"] == parser2["receipt"]["content_sha256"],
        "VAST2 audit subject drift",
    )

    failure = _load_bound_json(config["retained_v4_failure"])
    _require(
        failure["stage"] == "VAST1_OWNED_STREAM"
        and failure["reason_code"] == "DEVELOPMENTRELEASEV1ERROR",
        "v4 failure drift",
    )
    _require(failure["access_counts"]["development_scores"] == 0, "v4 failure scored")
    return {
        "v5_receipt_content_sha256": config["development_packet_chain"]["v5"]["receipt"][
            "content_sha256"
        ],
        "v5_audit_content_sha256": config["development_v5_independent_audit"]["content_sha256"],
        "v4_failure_content_sha256": config["retained_v4_failure"]["content_sha256"],
        "vast1_parser_receipt_content_sha256": parser["receipt"]["content_sha256"],
        "vast1_parser_audit_content_sha256": parser["independent_audit"]["content_sha256"],
        "vast2_parser_receipt_content_sha256": parser2["receipt"]["content_sha256"],
        "vast2_parser_audit_content_sha256": parser2["independent_audit"]["content_sha256"],
    }


def parse_vast1_record_v6(framed: bytes, *, source_index: int, framed_start: int) -> dict[str, Any]:
    return vast1_contract.parse_vast1_record(
        framed, source_index=source_index, framed_start=framed_start
    )


def parse_vast2_record_v6(framed: bytes, *, source_index: int, framed_start: int) -> dict[str, Any]:
    return vast2_contract.v1.parse_vast2_record(
        framed, source_index=source_index, framed_start=framed_start
    )


def validate_vast_duplicate_keys_v6(
    table1_rows: Sequence[tuple[str, int, int]],
    table2_rows: Sequence[tuple[str, int, float, float, float, float]],
) -> dict[str, int]:
    maximal = [(str(cosmo), int(void), int(edge)) for cosmo, void, edge in table1_rows]
    _require(
        all(
            cosmo in ("Planck2018", "WMAP5") and 0 <= void <= 1183 and edge in (0, 1, 2)
            for cosmo, void, edge in maximal
        ),
        "invalid audited VAST1 key",
    )
    table1_keys = [(cosmo, void) for cosmo, void, _ in maximal]
    _require(len(table1_keys) == len(set(table1_keys)), "duplicate audited VAST1 group key")
    edge_by_key = {(cosmo, void): edge for cosmo, void, edge in maximal}
    sphere_keys: list[tuple[str, int, float, float, float, float]] = []
    for cosmo, void, x, y, z, radius in table2_rows:
        key = (str(cosmo), int(void), float(x), float(y), float(z), float(radius))
        _require(
            key[0] in ("Planck2018", "WMAP5") and 0 <= key[1] <= 1183,
            "invalid audited VAST2 group key",
        )
        _require(
            all(math.isfinite(value) for value in key[2:]) and key[5] > 0.0,
            "invalid audited VAST2 sphere",
        )
        _require((key[0], key[1]) in edge_by_key, "unmatched audited VAST2 sphere")
        sphere_keys.append(key)
    _require(
        len(sphere_keys) == len(set(sphere_keys)), "duplicate audited VAST2 semantic sphere key"
    )
    observed_groups = {(key[0], key[1]) for key in sphere_keys}
    _require(observed_groups == set(table1_keys), "VAST1/VAST2 group union mismatch")
    retained = sum(
        key[0] == "Planck2018" and edge_by_key[(key[0], key[1])] == 0 for key in sphere_keys
    )
    excluded_planck_edge = sum(
        key[0] == "Planck2018" and edge_by_key[(key[0], key[1])] != 0 for key in sphere_keys
    )
    return {
        "groups": len(observed_groups),
        "sphere_rows": len(sphere_keys),
        "retained": retained,
        "excluded_edge": excluded_planck_edge,
        "excluded_nonplanck": len(sphere_keys) - retained - excluded_planck_edge,
    }


def prepare_vast_geometry_v6(
    table1_rows: Sequence[Mapping[str, Any]],
    table2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = validate_vast_duplicate_keys_v6(
        [(str(row["Cosmo"]), int(row["void"]), int(row["edge"])) for row in table1_rows],
        [
            (
                str(row["Cosmo"]),
                int(row["void"]),
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["Rad"]),
            )
            for row in table2_rows
        ],
    )
    edge_by_key: dict[tuple[str, int], int] = {}
    for row in table1_rows:
        if str(row["Cosmo"]) != "Planck2018":
            continue
        key = (str(row["Cosmo"]), int(row["void"]))
        edge_by_key[key] = int(row["edge"])
        expected = v1.geometry_v3.radec_to_xyz(
            float(row["RAdeg"]), float(row["DEdeg"]), float(row["s"])
        )
        observed = (float(row["x"]), float(row["y"]), float(row["z"]))
        _require(
            all(
                math.isclose(observed[index], float(expected[index]), rel_tol=1e-9, abs_tol=1e-8)
                for index in range(3)
            ),
            "repaired VAST1 coordinate probe failed",
        )
    spheres: list[tuple[tuple[float, float, float], float]] = []
    for row in table2_rows:
        key = (str(row["Cosmo"]), int(row["void"]))
        if key[0] != "Planck2018" or edge_by_key[key] != 0:
            continue
        spheres.append(
            (
                (float(row["x"]) / 0.674, float(row["y"]) / 0.674, float(row["z"]) / 0.674),
                float(row["Rad"]) / 0.674,
            )
        )
    _require(len(spheres) == summary["retained"], "repaired retained-sphere count drift")
    return {"spheres_Mpc": spheres, **summary}


def _fixture_conformance(config: Mapping[str, Any]) -> dict[str, Any]:
    fixture1 = (config["vast1_integration"]["fixture_raw_ascii"] + "\n").encode("ascii")
    row1 = parse_vast1_record_v6(fixture1, source_index=0, framed_start=0)
    fixture2 = (config["vast2_integration"]["fixture_raw_ascii"] + "\n").encode("ascii")
    row2 = parse_vast2_record_v6(fixture2, source_index=0, framed_start=0)
    fixture3 = (config["vast2_integration"]["wmap_zero_fixture_raw_ascii"] + "\n").encode("ascii")
    row3 = parse_vast2_record_v6(fixture3, source_index=39735, framed_start=4211910)
    return {
        "vast1_void_zero_accepted": row1["void"] == 0,
        "vast1_edge_one_accepted": row1["edge"] == 1,
        "vast1_payload_length_181_accepted": row1["payload_bytes"] == 181,
        "vast1_framed_sha256": row1["framed_raw_sha256"],
        "vast1_payload_sha256": row1["payload_raw_sha256"],
        "vast2_void_zero_accepted": row2["void"] == 0,
        "vast2_payload_length_105_accepted": row2["payload_bytes"] == 105,
        "vast2_framed_sha256": row2["framed_raw_sha256"],
        "vast2_payload_sha256": row2["payload_raw_sha256"],
        "vast2_wmap_zero_accepted": row3["Cosmo"] == "WMAP5" and row3["void"] == 0,
        "vast2_wmap_framed_sha256": row3["framed_raw_sha256"],
        "vast2_wmap_payload_sha256": row3["payload_raw_sha256"],
    }


def _validate_fixed_directory(path: Path, expected: Path) -> None:
    _require(path == expected and path.is_absolute(), "v6 fixed directory drift")
    root = REPO_ROOT.resolve(strict=True)
    _require(root == REPO_ROOT and not REPO_ROOT.is_symlink(), "v6 repository root redirected")
    _require(path.resolve(strict=False) == path, "v6 fixed directory redirected")
    cursor = REPO_ROOT
    for part in path.relative_to(REPO_ROOT).parts:
        cursor /= part
        _require(not cursor.is_symlink(), "v6 fixed directory symlink")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _authorization_binding(
    contract_receipt: Mapping[str, Any], reaudit: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "v6_receipt_raw_sha256": file_sha256(OUTPUT_PATH),
        "v6_receipt_content_sha256": contract_receipt["content_sha256"],
        "v6_config_raw_sha256": file_sha256(CONFIG_PATH),
        "v6_module_raw_sha256": file_sha256(MODULE_PATH),
        "v6_test_raw_sha256": file_sha256(TEST_PATH),
        "v6_reaudit_raw_sha256": file_sha256(
            canonical_file(load_config()["future_gates"]["independent_reaudit_path"])
        ),
        "v6_reaudit_content_sha256": reaudit["content_sha256"],
    }


def _load_future_gates(
    contract_receipt: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    config = load_config()
    reaudit_path = canonical_file(config["future_gates"]["independent_reaudit_path"])
    reaudit = json.loads(reaudit_path.read_text(encoding="utf-8"))
    _require(reaudit["content_sha256"] == _self_hash(reaudit), "v6 re-audit self-hash")
    _require(
        reaudit["status"] == config["future_gates"]["independent_reaudit_required_status"],
        "v6 re-audit status",
    )
    _require(
        reaudit.get("scientific_run_authority") is False, "v6 re-audit improperly authorizes run"
    )
    authorization_path = canonical_file(config["future_gates"]["one_run_authorization_path"])
    payload = authorization_path.read_bytes()
    authorization = json.loads(payload)
    _require(
        set(authorization)
        == {
            "schema",
            "status",
            "decision",
            "authorization_id",
            "uses_allowed",
            "hard_seals",
            "binding",
            "content_sha256",
        },
        "v6 authorization exact keys",
    )
    _require(
        authorization["schema"]
        == "invariant-open-gravity-void-correlation-v6-one-run-authorization-1.0",
        "v6 authorization schema",
    )
    _require(authorization["status"] == "PASS_ONE_DEVELOPMENT_RUN_ONLY", "v6 authorization status")
    _require(
        authorization["decision"] == "AUTHORIZE_EXACTLY_ONE_V6_DEVELOPMENT_RUN",
        "v6 authorization decision",
    )
    _require(
        _HEX64.fullmatch(str(authorization["authorization_id"])) is not None, "v6 authorization ID"
    )
    _require(
        authorization["uses_allowed"] == 1 and not isinstance(authorization["uses_allowed"], bool),
        "v6 authorization uses",
    )
    _require(
        authorization["hard_seals"] == config["future_executor"]["hard_seals"],
        "v6 authorization seals",
    )
    _require(
        authorization["binding"] == _authorization_binding(contract_receipt, reaudit),
        "v6 authorization binding",
    )
    _require(
        authorization["content_sha256"] == _self_hash(authorization), "v6 authorization self-hash"
    )
    return payload, authorization, reaudit


def _consumption_marker(
    authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-v6-authorization-consumption-1.0",
        "authorization_id": authorization["authorization_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "contract_receipt_content_sha256": contract_receipt["content_sha256"],
        "uses_consumed": 1,
        "content_sha256": "",
    }
    marker["content_sha256"] = _self_hash(marker)
    return marker


def _consume_authorization(
    authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT
        / "runs/gravity/open-gravity-void-correlation-development-release-v6-authorization-consumed",
    )
    CONSUMPTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    marker = _consumption_marker(authorization, contract_receipt)
    target = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
    _require(not target.exists() and not target.is_symlink(), "v6 authorization replay")
    with target.open("xb") as handle:
        handle.write(_pretty(marker))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(CONSUMPTION_DIRECTORY)
    return marker, target


def _expected_success_counts() -> dict[str, int]:
    return {
        "authorization_consumptions": 1,
        "cf4_source_opens": 1,
        "cf4_gzip_passes": 1,
        "cf4_identifier_rows_reverified": 38053,
        "cf4_development_scientific_rows_decoded": 22897,
        "cf4_validation_scientific_rows_decoded": 0,
        "cf4_confirmation_scientific_rows_decoded": 0,
        "vast_table1_source_opens": 1,
        "vast_table1_rows_decoded": 2347,
        "vast_table2_source_opens": 1,
        "vast_table2_gzip_passes": 1,
        "vast_table2_rows_decoded": 80080,
        "mask_source_opens": 1,
        "pantheon_source_opens": 0,
        "development_scores": 1,
    }


def build_final_receipt_v6(
    artifacts: Mapping[str, bytes],
    authorization: Mapping[str, Any],
    contract_receipt: Mapping[str, Any],
    reaudit: Mapping[str, Any],
    order_hashes: Sequence[str],
) -> dict[str, Any]:
    _require(
        set(artifacts) == _ARTIFACT_NAMES and len(order_hashes) == _PERMUTATIONS,
        "v6 final artifact/order set",
    )
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-score-receipt-6.0",
        "package_id": "open-gravity-void-correlation-development-score-v6",
        "status": "PASS_V6_DEVELOPMENT_SCORE_VALIDATION_CONFIRMATION_PANTHEON_SEALED",
        "decision": summary["classification"]["empirical_label"],
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "content_sha256": authorization["content_sha256"],
        },
        "release_chain": {"v6_contract": _authorization_binding(contract_receipt, reaudit)},
        "artifacts": {
            name: {
                "bytes": len(payload),
                "raw_sha256": bytes_sha256(payload),
                "content_sha256": v2._artifact_content(name, payload),
            }
            for name, payload in sorted(artifacts.items())
        },
        "roots": summary["roots"],
        "permutation_order_hashes": list(order_hashes),
        "permutation_order_root_sha256": v3._order_root(order_hashes),
        "counts": {
            "development_rows": summary["development_rows"],
            "permutations": summary["permutation"]["rows"],
        },
        "countermodels": summary["countermodels"],
        "access_counts": summary["access_counts"],
        "hard_seals": load_config()["future_executor"]["hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_final_payloads_v6(
    payloads: Mapping[str, bytes],
    authorization: Mapping[str, Any],
    contract_receipt: Mapping[str, Any],
    reaudit: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == _ARTIFACT_NAMES | {"receipt.json"}, "v6 final package set")
    artifacts = {name: payloads[name] for name in _ARTIFACT_NAMES}
    v3._v2_validate_artifacts(artifacts)
    receipt = json.loads(payloads["receipt.json"])
    _require(receipt["content_sha256"] == _self_hash(receipt), "v6 final receipt self-hash")
    expected = build_final_receipt_v6(
        artifacts, authorization, contract_receipt, reaudit, receipt["permutation_order_hashes"]
    )
    _require(receipt == expected, "v6 final receipt mismatch")
    rows = v2._reconstruct_scored_rows(
        v2._parse_jsonl(artifacts["artifacts/development-rows.jsonl"])
    )
    permutation_rows = v2._parse_jsonl(artifacts["artifacts/permutation-statistics.jsonl"])
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    claimed = {
        "observed": float.fromhex(summary["permutation"]["observed_hex"]),
        "permutation_statistics": [float.fromhex(row["statistic_hex"]) for row in permutation_rows],
        "tail_count": summary["permutation"]["tail_count"],
        "p_value": float.fromhex(summary["permutation"]["p_value_hex"]),
    }
    v3._exact_validate_regenerated(
        rows, claimed, receipt["permutation_order_hashes"], permutations=_PERMUTATIONS
    )
    _require(receipt["access_counts"] == _expected_success_counts(), "v6 final access counts")
    return receipt


def run_development_once() -> str:
    """Future sole entry; presently sealed behind re-audit plus separate one-run authorization."""
    contract_receipt = check_receipt()
    authorization_payload, authorization, reaudit = _load_future_gates(contract_receipt)
    marker, marker_path = _consume_authorization(authorization, contract_receipt)
    config = load_config()
    counts = {key: 0 for key in _expected_success_counts()}
    counts["authorization_consumptions"] = 1
    decoded_ids: list[int] = []
    stage = "AUTHORIZED_AND_CONSUMED"
    generated_payloads: dict[str, bytes] | None = None
    payload_ids: dict[str, int] | None = None
    payload_hashes: dict[str, str] | None = None
    final_capability = object()
    unspent_capability: object | None = final_capability

    class HashingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle
            self.digest = hashlib.sha256()
            self.bytes_read = 0

        def read(self, size: int = -1) -> bytes:
            data = self.handle.read(size)
            self.digest.update(data)
            self.bytes_read += len(data)
            return data

    def source_path(name: str) -> Path:
        _require(name in config["sources"], "v6 source name")
        return canonical_file(config["sources"][name]["path"])

    def check_raw(name: str, size: int, digest: str) -> None:
        source = config["sources"][name]
        _require(size == source["bytes"] and digest == source["raw_sha256"], f"{name} source drift")

    def gzip_lines(name: str, open_key: str, pass_key: str) -> Iterator[bytes]:
        with source_path(name).open("rb") as raw:
            counts[open_key] += 1
            reader = HashingReader(raw)
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                yield from stream
            check_raw(name, reader.bytes_read, reader.digest.hexdigest())
            counts[pass_key] += 1

    def plain_lines(name: str, open_key: str) -> Iterator[bytes]:
        digest = hashlib.sha256()
        size = 0
        with source_path(name).open("rb") as handle:
            counts[open_key] += 1
            for line in handle:
                digest.update(line)
                size += len(line)
                yield line
        check_raw(name, size, digest.hexdigest())

    def read_cf4(ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        nonlocal stage
        stage = "CF4_OWNED_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        observed = 0
        for index, line in enumerate(
            gzip_lines("CF4_TABLE4", "cf4_source_opens", "cf4_gzip_passes")
        ):
            _require(index < len(ledger), "v6 CF4 overflow")
            entry = ledger[index]
            v2.validate_ledger_entry(entry)
            payload = v1._frame_payload(line, 157)
            _require(
                index == entry["source_index"] and offset == entry["framed_start"], "v6 CF4 order"
            )
            _require(bytes_sha256(line) == entry["framed_raw_sha256"], "v6 CF4 frame hash")
            identifier, canonical = ids_v1.parse_i7_identifier(payload[:7])
            _require(
                identifier == entry["identifier"] and canonical == entry["canonical_identifier"],
                "v6 CF4 ID",
            )
            counts["cf4_identifier_rows_reverified"] += 1
            if entry["role"] == "development":
                rows.append(
                    v2.parse_cf4_development_record_v2(
                        line, entry, source_index=index, framed_start=offset
                    )
                )
                decoded_ids.append(identifier)
                counts["cf4_development_scientific_rows_decoded"] += 1
            offset += len(line)
            observed += 1
        _require(observed == len(ledger) == 38053 and len(rows) == 22897, "v6 CF4 coverage")
        return rows

    def read_vast1() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST1_AUDITED_PARSER_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        for index, line in enumerate(plain_lines("VAST_TABLE1", "vast_table1_source_opens")):
            row = parse_vast1_record_v6(line, source_index=index, framed_start=offset)
            rows.append(row)
            counts["vast_table1_rows_decoded"] += 1
            offset += len(line)
        _require(len(rows) == 2347, "v6 VAST1 row count")
        _require(
            [row["source_index"] for row in rows if row["void"] == 0] == [0, 1163],
            "v6 VAST1 zero-ID positions",
        )
        edge_counts = {str(edge): sum(row["edge"] == edge for row in rows) for edge in (0, 1, 2)}
        _require(
            edge_counts == config["vast1_integration"]["expected_edge_counts"],
            "v6 VAST1 edge counts",
        )
        return rows

    def read_vast2() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST2_AUDITED_PARSER_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        uncompressed_digest = hashlib.sha256()
        for index, line in enumerate(
            gzip_lines("VAST_TABLE2", "vast_table2_source_opens", "vast_table2_gzip_passes")
        ):
            row = parse_vast2_record_v6(line, source_index=index, framed_start=offset)
            rows.append(row)
            counts["vast_table2_rows_decoded"] += 1
            uncompressed_digest.update(line)
            offset += len(line)
        integration = config["vast2_integration"]
        _require(len(rows) == integration["expected_rows"], "v6 VAST2 row count")
        _require(
            offset == integration["expected_uncompressed_bytes"], "v6 VAST2 uncompressed byte count"
        )
        _require(
            uncompressed_digest.hexdigest() == integration["expected_uncompressed_raw_sha256"],
            "v6 VAST2 uncompressed hash",
        )
        vast2_contract.validate_no_semantic_sphere_duplicates(rows)
        cosmo_counts = {
            cosmo: sum(row["Cosmo"] == cosmo for row in rows) for cosmo in ("Planck2018", "WMAP5")
        }
        _require(
            cosmo_counts == integration["expected_cosmology_counts"], "v6 VAST2 cosmology counts"
        )
        _require(
            len({(row["Cosmo"], row["void"]) for row in rows}) == integration["expected_groups"],
            "v6 VAST2 group count",
        )
        _require(
            sum(row["void"] == 0 for row in rows)
            == integration["expected_zero_identifier_sphere_rows"],
            "v6 VAST2 zero-ID count",
        )
        return rows

    def read_mask() -> bytes:
        nonlocal stage
        stage = "MASK_OWNED_READ"
        with source_path("MASK_U8").open("rb") as handle:
            counts["mask_source_opens"] += 1
            payload = handle.read()
        check_raw("MASK_U8", len(payload), bytes_sha256(payload))
        v1.validate_mask(payload)
        return payload

    def write_failure(error: Exception) -> None:
        _validate_fixed_directory(
            FAILURE_DIRECTORY,
            REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v6-failures",
        )
        FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        failure: dict[str, Any] = {
            "schema": "invariant-open-gravity-void-correlation-development-failure-6.0",
            "status": "RETAINED_V6_RUN_FAILURE_NO_PARTIAL_SUCCESS",
            "authorization_id": authorization["authorization_id"],
            "stage": stage,
            "reason_code": type(error).__name__.upper(),
            "access_counts": dict(counts),
            "authorized_development_ids": sorted(set(decoded_ids)),
            "hard_seals_preserved": config["future_executor"]["hard_seals"],
            "content_sha256": "",
        }
        failure["content_sha256"] = _self_hash(failure)
        target = (
            FAILURE_DIRECTORY
            / f"{authorization['authorization_id']}-{failure['content_sha256']}.json"
        )
        _require(not target.exists(), "v6 failure collision")
        with target.open("xb") as handle:
            handle.write(_pretty(failure))
            handle.flush()
            os.fsync(handle.fileno())

    def final_write() -> str:
        nonlocal unspent_capability
        _require(unspent_capability is final_capability, "v6 final capability absent or replayed")
        unspent_capability = None
        _require(
            generated_payloads is not None
            and payload_ids is not None
            and payload_hashes is not None,
            "v6 payloads absent",
        )
        _require(
            payload_ids == {name: id(payload) for name, payload in generated_payloads.items()},
            "v6 payload identity drift",
        )
        _require(
            payload_hashes
            == {name: bytes_sha256(payload) for name, payload in generated_payloads.items()},
            "v6 payload hash drift",
        )
        _require(check_receipt() == contract_receipt, "v6 contract drift at write")
        payload2, authorization2, reaudit2 = _load_future_gates(contract_receipt)
        _require(
            payload2 == authorization_payload
            and authorization2 == authorization
            and reaudit2 == reaudit,
            "v6 gates drift at write",
        )
        _require(
            marker_path.read_bytes() == _pretty(marker)
            and marker == _consumption_marker(authorization, contract_receipt),
            "v6 marker drift",
        )
        validate_code_pins()
        validate_release_chain(load_config())
        validate_final_payloads_v6(generated_payloads, authorization, contract_receipt, reaudit)
        _validate_fixed_directory(
            FINAL_DIRECTORY,
            REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v6",
        )
        _validate_fixed_directory(
            STAGING_ROOT,
            REPO_ROOT / "work/open-gravity-void-correlation-development-score-v6-staging",
        )
        _require(not FINAL_DIRECTORY.exists(), "v6 final output already exists")
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT)).resolve()
        try:
            for name, payload in sorted(generated_payloads.items()):
                target = staging.joinpath(*name.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            validate_final_payloads_v6(
                v2._read_fixed_package(staging), authorization, contract_receipt, reaudit
            )
            FINAL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
                move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
                move.restype = ctypes.c_int
                _require(
                    bool(move(str(staging), str(FINAL_DIRECTORY), 0x00000008)),
                    "v6 MoveFileExW failed",
                )
            else:
                os.rename(staging, FINAL_DIRECTORY)
            return "PROMOTED_COMPLETE"
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    try:
        ledger = v2.load_identifier_ledger()
        cf4 = read_cf4(ledger)
        vast1 = read_vast1()
        vast2 = read_vast2()
        mask = read_mask()
        stage = "DERIVE_SCORE_AND_GENERATE"
        geometry = prepare_vast_geometry_v6(vast1, vast2)
        rows = [v1.derive_development_row(row, mask, geometry["spheres_Mpc"]) for row in cf4]
        profile = v1.profile_grid_details(rows)
        countermodels = v1.score_countermodels(rows)
        permutation = v3.regenerate_permutations_from_rows(rows, _PERMUTATIONS)
        counts["development_scores"] += 1
        _require(counts == _expected_success_counts(), "v6 owned counts incomplete")
        artifacts = v2.assemble_development_artifacts_v2(
            rows, ledger, profile, permutation, countermodels, counts, []
        )
        final_receipt = build_final_receipt_v6(
            artifacts, authorization, contract_receipt, reaudit, permutation["order_hashes"]
        )
        generated_payloads = {**artifacts, "receipt.json": _pretty(final_receipt)}
        payload_ids = {name: id(payload) for name, payload in generated_payloads.items()}
        payload_hashes = {
            name: bytes_sha256(payload) for name, payload in generated_payloads.items()
        }
        stage = "PRIVATE_FINAL_WRITE"
        return final_write()
    except Exception as error:
        write_failure(error)
        raise


_RUN_ORIGINAL = run_development_once
_RUN_CODE_SHA256 = bytes_sha256(marshal.dumps(run_development_once.__code__))


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v6 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v6 test raw drift")
    _require(run_development_once is _RUN_ORIGINAL, "v6 runner identity drift")
    _require(
        bytes_sha256(marshal.dumps(run_development_once.__code__)) == _RUN_CODE_SHA256,
        "v6 runner code drift",
    )


def _runner_structure() -> dict[str, bool]:
    source = inspect.getsource(run_development_once)
    tree = ast.parse(source)
    outer = tree.body[0]
    calls = [
        node.func.id
        for node in ast.walk(outer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    nested = [node.name for node in outer.body if isinstance(node, ast.FunctionDef)]
    return {
        "no_arguments": not inspect.signature(run_development_once).parameters,
        "future_gates_first": calls.index("check_receipt")
        < calls.index("_load_future_gates")
        < calls.index("_consume_authorization"),
        "audited_vast1_parser_called": "parse_vast1_record_v6" in calls,
        "audited_vast2_parser_called": "parse_vast2_record_v6" in calls,
        "repaired_geometry_join_called": "prepare_vast_geometry_v6" in calls,
        "private_final_write": nested.count("final_write") == 1 and calls.count("final_write") == 1,
        "obsolete_vast_parsers_absent": "v1.parse_vast_table1_record" not in source
        and "v1.parse_vast_table2_record" not in source
        and "v1.prepare_vast_geometry" not in source,
        "vast2_duplicate_gate_retained": "vast2_contract.validate_no_semantic_sphere_duplicates"
        in source,
        "pcg64_regeneration_retained": "v3.regenerate_permutations_from_rows" in source
        and "_PERMUTATIONS" in source,
    }


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixture = _fixture_conformance(config)
    structure = _runner_structure()
    return [
        {"check_id": "V1_V5_PACKETS_AND_V5_AUDIT_BYTE_EXACT", "passed": True},
        {"check_id": "VAST1_CONTRACT_AND_AUDIT_BOUND", "passed": True},
        {"check_id": "VAST2_V2_CONTRACT_AND_AUDIT_BOUND", "passed": True},
        {"check_id": "RETAINED_V4_FAILURE_BOUND", "passed": True},
        {
            "check_id": "OFFICIAL_VAST1_VOID_ZERO_FIXTURE_ACCEPTED",
            "passed": fixture["vast1_void_zero_accepted"],
        },
        {
            "check_id": "VAST1_FIXTURE_HASH_EXACT",
            "passed": fixture["vast1_framed_sha256"]
            == "e1aaeccae3e857121fd4b1b31895d21cf590e145f0421341e3c6ec7e6418a0a7",
        },
        {
            "check_id": "OFFICIAL_VAST2_VOID_ZERO_FIXTURE_ACCEPTED",
            "passed": fixture["vast2_void_zero_accepted"]
            and fixture["vast2_payload_length_105_accepted"],
        },
        {
            "check_id": "VAST2_FIXTURE_HASHES_EXACT",
            "passed": fixture["vast2_framed_sha256"]
            == config["vast2_integration"]["fixture_framed_sha256"]
            and fixture["vast2_payload_sha256"]
            == config["vast2_integration"]["fixture_payload_sha256"],
        },
        {
            "check_id": "VAST2_BOTH_COSMOLOGIES_ZERO_FIXTURES_EXACT",
            "passed": fixture["vast2_wmap_zero_accepted"]
            and fixture["vast2_wmap_framed_sha256"]
            == config["vast2_integration"]["wmap_zero_fixture_framed_sha256"]
            and fixture["vast2_wmap_payload_sha256"]
            == config["vast2_integration"]["wmap_zero_fixture_payload_sha256"],
        },
        {"check_id": "FULL_OWNED_RUNNER_STRUCTURE", "passed": all(structure.values())},
        {
            "check_id": "TWO_GATE_SEPARATION",
            "passed": "does not itself authorize" in config["future_gates"]["rule"],
        },
        {
            "check_id": "ZERO_CURRENT_AUTHORITY_AND_ACCESS",
            "passed": config["current_access_accounting"]["allowed_vast1_fixture_rows_decoded"] == 1
            and config["current_access_accounting"]["allowed_vast2_fixture_rows_decoded"] == 2
            and all(
                value == 0
                for key, value in config["current_access_accounting"].items()
                if key
                not in {"allowed_vast1_fixture_rows_decoded", "allowed_vast2_fixture_rows_decoded"}
            ),
        },
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    bindings = validate_release_chain(config)
    gates = conformance_gates(config)
    _require(all(gate["passed"] for gate in gates), "v6 conformance failure")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-6.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "bindings": bindings,
        "development_packet_chain": config["development_packet_chain"],
        "development_v5_independent_audit": config["development_v5_independent_audit"],
        "retained_v4_failure": config["retained_v4_failure"],
        "vast1_parser_contract": config["vast1_parser_contract"],
        "vast1_integration": config["vast1_integration"],
        "vast2_parser_contract": config["vast2_parser_contract"],
        "vast2_integration": config["vast2_integration"],
        "future_executor": config["future_executor"],
        "future_gates": config["future_gates"],
        "authority": config["authority"],
        "conformance_gates": gates,
        "access_accounting": config["current_access_accounting"],
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "next_gate": config["next_gate"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _write_receipt() -> str:
    payload = _pretty(build_receipt())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(
            OUTPUT_PATH.is_file()
            and not OUTPUT_PATH.is_symlink()
            and OUTPUT_PATH.read_bytes() == payload,
            "existing v6 receipt differs",
        )
        return "EXISTING_IDENTICAL"
    descriptor, name = tempfile.mkstemp(prefix="receipt.", suffix=".tmp", dir=OUTPUT_PATH.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(
        observed == expected and observed["content_sha256"] == _self_hash(observed),
        "v6 receipt drift",
    )
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(_write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_V6_DUAL_PARSER_SOURCE_FREE_EXECUTOR_NO_RUN_AUTHORITY")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
