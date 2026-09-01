"""Append-only lint-only successor for the frozen Solar synthetic v1 package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from sigma_theory_compiler import (
    open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1 as v1,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

CONFIG_PATH = Path(
    "configs/open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v2.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v2.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v2.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-v2-r1"
)
PROJECTION_PATH = OUTPUT_DIR / "corrected-module-projection.py"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = (
    "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-"
    "lint-successor-2.0"
)
_RECEIPT_SCHEMA = (
    "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-"
    "lint-successor-receipt-2.0"
)
_PACKAGE_ID = "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-v2"
_PRE_STATUS = "FROZEN_LINT_ONLY_SUCCESSOR_PRE_RESPONSE"
_STATUS = "FROZEN_LINT_ONLY_SUCCESSOR_COMPLETE_AWAITING_DISTINCT_AUDIT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_path(value: str | Path) -> Path:
    parsed = PurePosixPath(str(value).replace("\\", "/"))
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("Solar v2 path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("Solar v2 path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_predecessor: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "purpose",
        "predecessor_package_id",
        "predecessor_receipt_content_sha256",
        "predecessor_bindings",
        "lint_corrections",
        "preserved_v1_results",
        "output_directory",
        "outputs",
        "access_contract",
    }
    _require(set(config) == expected, "Solar v2 config keys changed")
    _require(config["schema"] == _SCHEMA, "Solar v2 schema changed")
    _require(
        config["package_id"] == _PACKAGE_ID
        and config["version"] == "v2.0.0"
        and config["status"] == _PRE_STATUS,
        "Solar v2 identity changed",
    )
    _require(
        config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "Solar v2 claim class changed",
    )
    _require(
        config["predecessor_package_id"]
        == "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-v1",
        "Solar v2 predecessor changed",
    )
    bindings = config["predecessor_bindings"]
    _require(len(bindings) == 9, "Solar v2 predecessor inventory changed")
    ids = [row["id"] for row in bindings]
    _require(ids == sorted(set(ids)), "Solar v2 predecessor bindings are not sorted")
    for row in bindings:
        _require(
            set(row) == {"id", "path", "bytes", "sha256"},
            "Solar v2 predecessor binding schema changed",
        )
        if verify_predecessor:
            path = _repo_path(row["path"])
            _require(path.is_file(), f"Solar v1 predecessor missing: {row['id']}")
            _require(path.stat().st_size == row["bytes"], f"Solar v1 bytes changed: {row['id']}")
            _require(_file_sha256(path) == row["sha256"], f"Solar v1 hash changed: {row['id']}")
    corrections = config["lint_corrections"]
    _require(
        [row["id"] for row in corrections]
        == [
            "REMOVE_UNUSED_DISCOVERY_STATUS_IMPORT",
            "ANNOTATE_INTENTIONAL_RETAINED_FAILURE_CATCH",
        ],
        "Solar v2 lint correction inventory changed",
    )
    _require(
        corrections[0]
        == {
            "id": "REMOVE_UNUSED_DISCOVERY_STATUS_IMPORT",
            "before": "    DiscoveryStatus,\n",
            "after": "",
        },
        "Solar v2 unused-import correction changed",
    )
    _require(
        corrections[1]["before"] == "            except Exception as error:\n"
        and corrections[1]["after"]
        == "            except Exception as error:  # noqa: BLE001\n"
        and "BaseException" in corrections[1]["policy"],
        "Solar v2 retained-failure annotation changed",
    )
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "Solar v2 output directory changed",
    )
    _require(
        config["outputs"]
        == {
            "corrected_module_projection": PROJECTION_PATH.as_posix(),
            "receipt": RECEIPT_PATH.as_posix(),
        },
        "Solar v2 outputs changed",
    )
    _require(
        all(value == 0 for value in config["access_contract"].values()),
        "Solar v2 access boundary changed",
    )


def _binding(config: Mapping[str, Any], binding_id: str) -> Mapping[str, Any]:
    return next(row for row in config["predecessor_bindings"] if row["id"] == binding_id)


def corrected_module_projection(config: Mapping[str, Any]) -> bytes:
    module = _binding(config, "V1_MODULE")
    source_path = _repo_path(module["path"])
    source_bytes = source_path.read_bytes()
    _require(_file_sha256(source_path) == module["sha256"], "Solar v1 module changed")
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SchemaViolation("Solar v1 module is not UTF-8") from error
    for correction in config["lint_corrections"]:
        before = correction["before"]
        after = correction["after"]
        _require(text.count(before) == 1, f"Solar v2 correction target changed: {correction['id']}")
        text = text.replace(before, after, 1)
    _require("    DiscoveryStatus,\n" not in text, "unused import survived projection")
    _require(
        text.count("except Exception as error:  # noqa: BLE001") == 1,
        "retained-failure catch annotation changed",
    )
    compile(text, PROJECTION_PATH.as_posix(), "exec")
    return text.encode("utf-8")


def _preserved_result_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": receipt["status"],
        "target_count": receipt["target_count"],
        "epoch_count": receipt["epoch_count"],
        "source_domain_count": receipt["source_domain_count"],
        "mechanism_count": receipt["mechanism_count"],
        "scenario_count": receipt["scenario_count"],
        "common_abi_execution_count": receipt["common_abi_execution_count"],
        "successful_common_abi_execution_count": receipt[
            "successful_common_abi_execution_count"
        ],
        "candidate_comparison_count": receipt["candidate_comparison_count"],
        "replay_entry_count": receipt["replay_entry_count"],
        "blocked_ledger_entry_count": receipt["blocked_ledger_entry_count"],
        "truth_recovered_count": receipt["truth_recovered_count"],
        "distinct_truth_recovered_count": receipt["distinct_truth_recovered_count"],
        "numerical_failure_count": len(receipt["numerical_failures"]),
    }


def build_receipt() -> tuple[dict[str, Any], bytes]:
    config = load_config()
    validate_config(config)
    replayed = v1.check()
    receipt_binding = _binding(config, "V1_RECEIPT")
    predecessor_receipt = json.loads(
        _repo_path(receipt_binding["path"]).read_text(encoding="utf-8")
    )
    _require(replayed == predecessor_receipt, "Solar v1 replay receipt changed")
    _require(
        predecessor_receipt["content_sha256"]
        == config["predecessor_receipt_content_sha256"],
        "Solar v1 receipt content hash changed",
    )
    preserved = _preserved_result_projection(predecessor_receipt)
    _require(preserved == config["preserved_v1_results"], "Solar v1 results changed")
    projection = corrected_module_projection(config)
    predecessor_hashes = {row["id"]: row["sha256"] for row in config["predecessor_bindings"]}
    body = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": _PACKAGE_ID,
        "version": "v2.0.0",
        "status": _STATUS,
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_LINT_ONLY_SUCCESSOR",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "predecessor_package_id": config["predecessor_package_id"],
        "predecessor_receipt_raw_sha256": receipt_binding["sha256"],
        "predecessor_receipt_content_sha256": predecessor_receipt["content_sha256"],
        "predecessor_hashes": predecessor_hashes,
        "predecessor_replay_passed": True,
        "v1_subject_bytes_modified": False,
        "scientific_payload_reused_byte_exact": True,
        "new_scientific_scenarios_generated": 0,
        "new_candidate_comparisons_computed": 0,
        "lint_corrections": config["lint_corrections"],
        "lint_correction_count": 2,
        "corrected_projection_sha256": hashlib.sha256(projection).hexdigest(),
        "corrected_projection_bytes": len(projection),
        "preserved_v1_results": preserved,
        "preserved_v1_artifact_sha256": predecessor_receipt["artifact_sha256"],
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": _file_sha256(_ROOT / MODULE_PATH),
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "access_accounting": config["access_contract"],
        "limitations": [
            "This successor changes lint presentation only and generates no new scientific scenario, prediction, comparison, score, or claim.",
            "The corrected module projection is an exact two-edit projection of the immutable v1 module; v1 remains the executable scientific evidence package.",
            "All v1 synthetic limitations, degeneracies, misses, adapter blocks, and claim boundaries remain unchanged.",
            "A distinct auditor must verify the exact two-edit projection and every predecessor hash before admitting v2.",
        ],
    }
    return {**body, "content_sha256": _json_sha256(body)}, projection


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite changed artifact: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def freeze() -> str:
    receipt, projection = build_receipt()
    statuses = (
        _write_once(_ROOT / PROJECTION_PATH, projection),
        _write_once(_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ",".join(statuses)


def check() -> dict[str, Any]:
    receipt, projection = build_receipt()
    expected = (
        (PROJECTION_PATH, projection),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for relative, payload in expected:
        path = _ROOT / relative
        _require(path.is_file() and path.read_bytes() == payload, f"Solar v2 artifact changed: {relative}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        receipt = check()
        print(receipt["status"], receipt["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
