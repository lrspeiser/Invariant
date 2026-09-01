"""Append-only mechanical release repair for TWELL-400 rebind/replay v1.

V2 validates the frozen v1 scientific payload byte-for-byte and semantically. It
does not import the v1 subject module, regenerate predictions, mirror scientific
artifacts, load a response-valued member, or compute a scientific score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_twell_400_source_shaped_rebind_replay_v2.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_twell_400_source_shaped_rebind_replay_v2.py"
)
TEST_PATH = Path("tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v2.py")
OUTPUT_DIR = Path("runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v2")
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
CONFIG_SCHEMA = "open-gravity-twell-400-source-shaped-rebind-replay-config-2.0"
RECEIPT_SCHEMA = "open-gravity-twell-400-source-shaped-rebind-replay-receipt-2.0"
DECISION = "FORMAT_ONLY_SUCCESSOR_FROZEN_AWAITING_DISTINCT_INDEPENDENT_AUDIT"

_SCIENCE_IDS = {
    "V1_COMPATIBILITY_LEDGER",
    "V1_PARAMETER_CELL_DISPOSITION_LEDGER",
    "V1_EXECUTION_BINDINGS",
    "V1_SOURCE_PROJECTIONS_LEDGER",
    "V1_SOURCE_PROJECTIONS_VALUES",
    "V1_UNIQUE_EXECUTIONS",
    "V1_PREDICTIONS_VALUES",
    "V1_REPLAY_LEDGER",
    "V1_EQUIVALENCE_TIES",
    "V1_INVARIANCE_GATES",
}


class TwellSourceRebindV2Error(RuntimeError):
    """Raised when the append-only mechanical successor fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwellSourceRebindV2Error(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TwellSourceRebindV2Error(f"noncanonical value: {error}") from error


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    _require(array.dtype.name in {"float64", "int64"}, "unsupported array dtype")
    _require(bool(np.all(np.isfinite(array))), "nonfinite frozen array")
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _repo_path(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value.replace("\\", "/"))
    _require(
        not parsed.is_absolute() and all(part not in {"", ".", ".."} for part in parsed.parts),
        f"path escaped repository: {value}",
    )
    repo = root.resolve(strict=True)
    result = (repo / parsed.as_posix()).resolve(strict=False)
    _require(result == repo or repo in result.parents, f"path escaped repository: {value}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TwellSourceRebindV2Error(f"could not load JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TwellSourceRebindV2Error(
                    f"invalid JSONL row: {path}:{line_number}"
                ) from error
            _require(isinstance(value, dict), f"JSONL object required: {path}:{line_number}")
            yield value


def _jsonl_semantic_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for row in _jsonl_rows(path):
        if count:
            digest.update(b",")
        digest.update(canonical_bytes(content_sha256(row)))
        count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def _npz_semantic_sha256(path: Path) -> tuple[str, int]:
    hashes: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            _require(
                key.startswith(("prediction__", "source_projection__")),
                f"unexpected v1 NPZ member role: {key}",
            )
            hashes[key] = array_sha256(archive[key])
    return content_sha256(hashes), len(hashes)


def _binding_semantics(path: Path, kind: str) -> tuple[str, int]:
    if kind == "OPAQUE":
        return file_sha256(path), 1
    if kind == "JSONL":
        return _jsonl_semantic_sha256(path)
    if kind == "NPZ":
        return _npz_semantic_sha256(path)
    payload = _read_json(path)
    if kind == "RECEIPT":
        observed = payload.get("content_sha256")
        _require(observed == _receipt_content_sha256(payload), "v1 receipt self-seal failed")
        return str(observed), 1
    _require(kind == "JSON", f"unknown binding kind: {kind}")
    return content_sha256(payload), 1


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    repo = root.resolve(strict=True)
    config = _read_json(_repo_path(repo, CONFIG_PATH.as_posix()))
    validate_config(repo, config, verify_files=True)
    return config


def validate_config(root: Path, config: Mapping[str, Any], *, verify_files: bool = True) -> None:
    _require(
        set(config)
        == {
            "schema",
            "package_id",
            "version",
            "status",
            "claim_class",
            "output_directory",
            "v1_bindings",
            "blocked_audit",
            "preservation_contract",
            "format_repair_contract",
            "access_contract",
            "claim_boundary",
        },
        "config keys changed",
    )
    _require(config["schema"] == CONFIG_SCHEMA, "config schema changed")
    _require(config["package_id"] == OUTPUT_DIR.name, "package ID changed")
    _require(config["version"] == "v1.0.1", "package version changed")
    _require(
        config["status"] == "APPEND_ONLY_MECHANICAL_FORMAT_REPAIR_PRE_RESPONSE",
        "package status changed",
    )
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim changed")
    _require(config["output_directory"] == OUTPUT_DIR.as_posix(), "output path changed")
    bindings = list(config["v1_bindings"])
    _require(len(bindings) == 14, "v1 binding count changed")
    _require(len({row["id"] for row in bindings}) == 14, "duplicate v1 binding")
    _require(
        {row["id"] for row in bindings if row["id"] in _SCIENCE_IDS} == _SCIENCE_IDS,
        "science binding inventory changed",
    )
    _require(
        sum(row["id"] in _SCIENCE_IDS for row in bindings) == 10, "science artifact count changed"
    )
    if verify_files:
        for row in bindings:
            path = _repo_path(root, str(row["path"]))
            _require(path.is_file(), f"missing v1 binding: {row['id']}")
            _require(file_sha256(path) == row["raw_sha256"], f"v1 raw hash drift: {row['id']}")
            semantic, count = _binding_semantics(path, str(row["kind"]))
            _require(semantic == row["content_sha256"], f"v1 content hash drift: {row['id']}")
            _require(count == row["row_count"], f"v1 row/member count drift: {row['id']}")
    audit = config["blocked_audit"]
    _require(audit["status"] == "BLOCK_MECHANICAL_FORMAT_GATE_ONLY", "blocked audit status changed")
    if verify_files:
        audit_path = _repo_path(root, str(audit["path"]))
        program_path = _repo_path(root, str(audit["audit_program_path"]))
        _require(file_sha256(audit_path) == audit["raw_sha256"], "blocked audit raw drift")
        _require(
            file_sha256(program_path) == audit["audit_program_raw_sha256"], "audit program drift"
        )
        audit_payload = _read_json(audit_path)
        _require(
            audit_payload.get("content_sha256") == audit["content_sha256"]
            and _receipt_content_sha256(audit_payload) == audit["content_sha256"],
            "blocked audit content drift",
        )
    preservation = config["preservation_contract"]
    _require(
        preservation["v1_science_artifact_count"] == 10
        and preservation["v2_science_artifact_mirror_emitted"] is False
        and preservation["v1_files_modified"] == 0
        and preservation["scientific_payload_recomputed"] is False
        and preservation["scientific_payload_reused_byte_exact"] is True,
        "preservation contract changed",
    )
    _require(
        preservation["card_count"] == 400
        and preservation["parameter_cell_count"] == 1184
        and sum(preservation["compatibility_counts"].values()) == 2000
        and sum(preservation["unique_execution_counts"].values()) == 2592
        and sum(preservation["replay_counts"].values()) == 62208,
        "preserved counts changed",
    )
    repair = config["format_repair_contract"]
    _require(
        repair["repair_scope"] == ["V2_MODULE", "V2_TEST"]
        and repair["v1_files_formatted_or_modified"] == 0
        and repair["alternate_working_directory_check_required"] is True
        and repair["mechanical_release_claim_before_distinct_audit"] is False,
        "format repair scope changed",
    )
    _require(
        all(value == 0 for value in config["access_contract"].values()), "access boundary changed"
    )
    claims = config["claim_boundary"]
    _require(claims["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim ceiling changed")
    _require(
        all(value is False for key, value in claims.items() if key != "claim_class"),
        "claim boundary weakened",
    )


def _binding_by_id(config: Mapping[str, Any], binding_id: str) -> Mapping[str, Any]:
    return next(row for row in config["v1_bindings"] if row["id"] == binding_id)


def _status_counts(path: Path) -> Counter[str]:
    return Counter(str(row["status"]) for row in _jsonl_rows(path))


def validate_v1_scientific_payload(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    def path(binding_id: str) -> Path:
        return _repo_path(root, str(_binding_by_id(config, binding_id)["path"]))

    compatibility = _status_counts(path("V1_COMPATIBILITY_LEDGER"))
    unique = list(_jsonl_rows(path("V1_UNIQUE_EXECUTIONS")))
    replay = _status_counts(path("V1_REPLAY_LEDGER"))
    unique_counts = Counter(str(row["status"]) for row in unique)
    invalid_formula_counts = Counter(
        str(row["formula_id"]) for row in unique if row["status"] == "NUMERICAL_INVALID"
    )
    equivalence = _read_json(path("V1_EQUIVALENCE_TIES"))
    v1_receipt = _read_json(path("V1_RECEIPT"))
    blocked_audit = _read_json(_repo_path(root, str(config["blocked_audit"]["path"])))
    expected = config["preservation_contract"]
    _require(dict(compatibility) == expected["compatibility_counts"], "compatibility counts drift")
    _require(dict(unique_counts) == expected["unique_execution_counts"], "unique counts drift")
    _require(dict(replay) == expected["replay_counts"], "replay counts drift")
    _require(
        dict(invalid_formula_counts) == expected["invalid_formula_counts"],
        "invalid formula counts drift",
    )
    _require(
        equivalence["finite_source_prediction_tie_group_count"]
        == expected["finite_source_prediction_tie_group_count"]
        and equivalence["finite_fixture_similarity_promoted_to_formula_identity"] is False,
        "equivalence accounting drift",
    )
    _require(
        v1_receipt["access_accounting"]["response_values_opened"] == 0
        and v1_receipt["access_accounting"]["scientific_scores_computed"] == 0,
        "v1 access accounting drift",
    )
    _require(
        blocked_audit["scientific_replay"] == "PASS_SOURCE_ONLY_SYNTHETIC"
        and blocked_audit["mismatches"] == 0
        and blocked_audit["independent_array_comparisons"] == 5184
        and blocked_audit["response_values_opened"] == 0
        and blocked_audit["scientific_scores_computed"] == 0,
        "independent numerical evidence drift",
    )
    return {
        "compatibility_counts": dict(compatibility),
        "unique_execution_counts": dict(unique_counts),
        "replay_counts": dict(replay),
        "invalid_formula_counts": dict(invalid_formula_counts),
        "finite_source_prediction_tie_group_count": equivalence[
            "finite_source_prediction_tie_group_count"
        ],
        "independent_array_comparisons": blocked_audit["independent_array_comparisons"],
        "independent_recomputation_mismatches": blocked_audit["mismatches"],
    }


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    repo = root.resolve(strict=True)
    config = load_config(repo)
    preserved = validate_v1_scientific_payload(repo, config)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_FORMAT_ONLY_SUCCESSOR_AWAITING_DISTINCT_AUDIT",
        "decision": DECISION,
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "scientific_claim": "NONE_SOURCE_ONLY_SYNTHETIC_PAYLOAD_PRESERVED",
        "distinct_independent_audit_required": True,
        "independent_audit_completed": False,
        "package_hashes": {
            "config_raw_sha256": file_sha256(repo / CONFIG_PATH),
            "module_raw_sha256": file_sha256(repo / MODULE_PATH),
            "test_raw_sha256": file_sha256(repo / TEST_PATH),
        },
        "v1_bindings": {
            row["id"]: {
                "path": row["path"],
                "kind": row["kind"],
                "raw_sha256": row["raw_sha256"],
                "content_sha256": row["content_sha256"],
                "row_count": row["row_count"],
            }
            for row in config["v1_bindings"]
        },
        "blocked_audit": dict(config["blocked_audit"]),
        "repair": {
            "kind": "MECHANICAL_FORMAT_GATE_AND_RELEASE_CLAIM_CORRECTION_ONLY",
            "v1_mechanical_disposition": "BLOCKED_RUFF_FORMAT_CHECK_FAILURE",
            "v1_numerical_disposition": "INDEPENDENTLY_RECOMPUTED_SOURCE_ONLY_SYNTHETIC_MATCH",
            "v1_science_artifacts_preserved_byte_exact": 10,
            "v1_files_modified": 0,
            "v2_science_artifact_mirror_emitted": False,
            "scientific_payload_recomputed": False,
            "ruff_check_required_and_externally_run": True,
            "ruff_format_check_required_and_externally_run": True,
            "mechanical_pass_claimed_before_distinct_audit": False,
        },
        "preserved_scientific_results": preserved,
        "access_accounting": {
            **config["access_contract"],
            "v1_source_only_prediction_npz_members_verified": 5184,
            "v1_source_projection_npz_members_verified": 160,
            "v1_science_artifacts_hash_verified": 10,
        },
        "claim_boundary": dict(config["claim_boundary"]),
        "limitations": [
            "V1 remains mechanically blocked because its frozen module and test fail ruff format --check; the previous Ruff-passed handoff overclaimed the format gate.",
            "The blocked independent audit establishes exact source-only numerical replay while rejecting the v1 mechanical release claim.",
            "V2 creates no scientific mirror and does not regenerate any source projection, prediction, execution, replay, tie, or compatibility result.",
            "The 38 unique and 912 replay-level numerical invalids remain unchanged and explicit.",
            "The other 274 cards remain temporal- or driver-blocked; this format repair makes none of them source-executable.",
            "No response-valued NPZ member, candidate value, variance value, truth value, response row, or scientific score is opened or computed.",
            "Ruff check and ruff format --check are external mechanical gates over the v2 module and test; a distinct auditor must rerun them before admission.",
            "No empirical support, empirical rejection, novelty, or publication-readiness claim follows from this synthetic repair.",
        ],
    }
    receipt["content_sha256"] = _receipt_content_sha256(receipt)
    return receipt


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical v2 receipt")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent v2 receipt differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> str:
    repo = root.resolve(strict=True)
    return _atomic_no_clobber(repo / RECEIPT_PATH, _pretty_json_bytes(build_receipt(repo)))


def validate_receipt(root: Path = Path(".")) -> dict[str, Any]:
    repo = root.resolve(strict=True)
    stored = _read_json(repo / RECEIPT_PATH)
    _require(
        stored.get("content_sha256") == _receipt_content_sha256(stored),
        "v2 receipt self-seal failed",
    )
    _require(stored == build_receipt(repo), "v2 receipt differs from deterministic rebuild")
    return stored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "write", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    publication: str | None = None
    if args.command == "write":
        publication = write_receipt(root)
        receipt = validate_receipt(root)
    elif args.command == "check":
        receipt = validate_receipt(root)
    else:
        receipt = build_receipt(root)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "v1_science_artifacts_preserved": receipt["repair"][
                    "v1_science_artifacts_preserved_byte_exact"
                ],
                "unique_execution_counts": receipt["preserved_scientific_results"][
                    "unique_execution_counts"
                ],
                "replay_counts": receipt["preserved_scientific_results"]["replay_counts"],
                "response_values_opened": receipt["access_accounting"]["response_values_opened"],
                "scientific_scores_computed": receipt["access_accounting"][
                    "scientific_scores_computed"
                ],
                "publication": publication,
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
