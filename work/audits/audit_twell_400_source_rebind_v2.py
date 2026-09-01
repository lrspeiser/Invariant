"""Distinct audit of the frozen TWELL-400 v2 format-only successor.

The evidence reconstruction in this program does not import the subject module.
The subject is loaded only after that reconstruction, for deterministic rebuild
and fail-closed mutation checks.  Only the two source-only synthetic NPZ files
bound by v1 are opened; no response, candidate, variance, or truth payload is
located or read.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT_PROGRAM = Path("work/audits/audit_twell_400_source_rebind_v2.py")
AUDIT_RECEIPT = Path(
    "work/audits/"
    "open-gravity-twell-400-source-shaped-rebind-replay-v2-"
    "independent-audit-pass-76f23426.json"
)
V2_CONFIG = Path("configs/open_gravity_twell_400_source_shaped_rebind_replay_v2.json")
V2_MODULE = Path(
    "src/sigma_theory_compiler/open_gravity_twell_400_source_shaped_rebind_replay_v2.py"
)
V2_TEST = Path("tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v2.py")
V2_RECEIPT = Path("runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v2/receipt.json")
V2_OUTPUT = V2_RECEIPT.parent

EXPECTED_V2_RAW = {
    V2_CONFIG.as_posix(): "88912f922369003b659a832b9c0eca07f87fb57daa2e3932364df1d5ae87d99c",
    V2_MODULE.as_posix(): "76f2342605c0e818abf2cc50d3d77d16f9a24ec56aa56b98f61fc57d7c61dd78",
    V2_TEST.as_posix(): "122fae5b0b2fc18108c58494357ada18fa460d31d3f60e3df27a866ed6f35224",
    V2_RECEIPT.as_posix(): "7082fcf24ff54e63cfbaee2b1c702876b3d8fc86d01905883d792bb5fc3457de",
}
EXPECTED_V2_RECEIPT_CONTENT = "aa10578b22698a298b9f2e871f164419a48c8547c9583dae979ea87b44e788e6"
EXPECTED_V1_MAIN_RAW = {
    "V1_CONFIG": "12df390acedd7afb650b101b714758b5bd8e1ccf2f5f36da7a1779040533688a",
    "V1_MODULE": "babbeb727660623bca3816a3ad58d87c319774657ed09f18f00e89c76b2e9e18",
    "V1_TEST": "fdf5ea8ff098ccc3dec5731396501f9374149ae54cb4ba83d484d863a42646cb",
    "V1_RECEIPT": "bcaf6df58e39a9dc52329e8afd78546c905cba20271858bb68e0e3b4e19bdab4",
}
EXPECTED_BLOCKED_AUDIT_RAW = "41f5ae2d01aea055ac10373b7789fe7cba0da543f9d73e253dde0273dac502e1"
EXPECTED_BLOCKED_AUDIT_CONTENT = "786ac589b8e2221d7679bb58d4c2b1e2338dddda74893c82d1d6bd8fb912da24"
EXPECTED_BLOCKED_AUDIT_PROGRAM_RAW = (
    "8bf6552989c9d4edfbf28936bc16360b19eea9262672516ac3330dbce108c505"
)
SCIENCE_IDS = {
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


class AuditFailure(RuntimeError):
    """A required frozen or semantic property did not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    require(array.dtype.name in {"float64", "int64"}, "unsupported frozen array dtype")
    require(bool(np.all(np.isfinite(array))), "nonfinite frozen array")
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value.replace("\\", "/"))
    require(
        not parsed.is_absolute() and all(part not in {"", ".", ".."} for part in parsed.parts),
        f"unsafe repository path: {value}",
    )
    repo = root.resolve(strict=True)
    result = (repo / parsed.as_posix()).resolve(strict=False)
    require(result == repo or repo in result.parents, f"repository escape: {value}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL object required: {path}:{line_number}")
            rows.append(value)
    return rows


def jsonl_semantic_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for row in read_jsonl(path):
        if count:
            digest.update(b",")
        digest.update(canonical_bytes(content_sha256(row)))
        count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def npz_semantic_sha256(path: Path) -> tuple[str, int]:
    hashes: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as archive:
        members = sorted(archive.files)
        require(
            all(name.startswith(("prediction__", "source_projection__")) for name in members),
            f"forbidden NPZ member role: {path}",
        )
        for name in members:
            hashes[name] = array_sha256(archive[name])
    return content_sha256(hashes), len(hashes)


def binding_semantics(path: Path, kind: str) -> tuple[str, int]:
    if kind == "OPAQUE":
        return raw_sha256(path), 1
    if kind == "JSONL":
        return jsonl_semantic_sha256(path)
    if kind == "NPZ":
        return npz_semantic_sha256(path)
    payload = read_json(path)
    if kind == "RECEIPT":
        observed = payload.get("content_sha256")
        require(observed == self_content_sha256(payload), f"receipt self-seal failed: {path}")
        return str(observed), 1
    require(kind == "JSON", f"unknown binding kind: {kind}")
    return content_sha256(payload), 1


def binding_path(bindings: Mapping[str, Mapping[str, Any]], binding_id: str) -> Path:
    return safe_path(ROOT, str(bindings[binding_id]["path"]))


def verify_frozen_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed_v2 = {path: raw_sha256(ROOT / path) for path in EXPECTED_V2_RAW}
    require(observed_v2 == EXPECTED_V2_RAW, "v2 builder-handoff raw hash mismatch")

    config = read_json(ROOT / V2_CONFIG)
    receipt = read_json(ROOT / V2_RECEIPT)
    require(
        receipt.get("content_sha256") == EXPECTED_V2_RECEIPT_CONTENT
        and self_content_sha256(receipt) == EXPECTED_V2_RECEIPT_CONTENT,
        "v2 receipt content seal mismatch",
    )
    require(
        receipt["package_hashes"]
        == {
            "config_raw_sha256": EXPECTED_V2_RAW[V2_CONFIG.as_posix()],
            "module_raw_sha256": EXPECTED_V2_RAW[V2_MODULE.as_posix()],
            "test_raw_sha256": EXPECTED_V2_RAW[V2_TEST.as_posix()],
        },
        "v2 receipt package hash pins mismatch",
    )
    require(config["package_id"] == V2_OUTPUT.name, "v2 config package mismatch")
    require(config["output_directory"] == V2_OUTPUT.as_posix(), "v2 output pin mismatch")
    require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim ceiling drift")
    require(
        config["status"] == "APPEND_ONLY_MECHANICAL_FORMAT_REPAIR_PRE_RESPONSE",
        "v2 config status drift",
    )
    require(
        receipt["status"] == "FROZEN_FORMAT_ONLY_SUCCESSOR_AWAITING_DISTINCT_AUDIT"
        and receipt["independent_audit_completed"] is False
        and receipt["distinct_independent_audit_required"] is True,
        "v2 pre-audit disposition drift",
    )

    rows = list(config["v1_bindings"])
    require(len(rows) == 14, "v1 binding count mismatch")
    require(len({row["id"] for row in rows}) == 14, "duplicate v1 binding")
    bindings = {str(row["id"]): row for row in rows}
    require(
        set(bindings) == SCIENCE_IDS | set(EXPECTED_V1_MAIN_RAW),
        "v1 binding inventory mismatch",
    )
    require(
        {key: bindings[key]["raw_sha256"] for key in EXPECTED_V1_MAIN_RAW} == EXPECTED_V1_MAIN_RAW,
        "v1 main-file handoff pins mismatch",
    )

    verified: dict[str, Any] = {}
    for binding_id, row in bindings.items():
        path = binding_path(bindings, binding_id)
        require(
            path.is_file() and not path.is_symlink(), f"missing or linked v1 file: {binding_id}"
        )
        observed_raw = raw_sha256(path)
        require(observed_raw == row["raw_sha256"], f"v1 raw drift: {binding_id}")
        semantic, count = binding_semantics(path, str(row["kind"]))
        require(semantic == row["content_sha256"], f"v1 semantic drift: {binding_id}")
        require(count == row["row_count"], f"v1 row/member count drift: {binding_id}")
        verified[binding_id] = {
            "raw_sha256": observed_raw,
            "content_sha256": semantic,
            "row_or_member_count": count,
        }

    receipt_bindings = receipt["v1_bindings"]
    require(set(receipt_bindings) == set(bindings), "v2 receipt v1 inventory mismatch")
    for binding_id, row in bindings.items():
        require(
            receipt_bindings[binding_id]
            == {
                key: row[key]
                for key in ("path", "kind", "raw_sha256", "content_sha256", "row_count")
            },
            f"v2 receipt v1 pin mismatch: {binding_id}",
        )

    blocked_binding = config["blocked_audit"]
    require(blocked_binding == receipt["blocked_audit"], "blocked-audit receipt/config mismatch")
    require(blocked_binding["raw_sha256"] == EXPECTED_BLOCKED_AUDIT_RAW, "blocked raw pin drift")
    require(
        blocked_binding["content_sha256"] == EXPECTED_BLOCKED_AUDIT_CONTENT,
        "blocked content pin drift",
    )
    require(
        blocked_binding["audit_program_raw_sha256"] == EXPECTED_BLOCKED_AUDIT_PROGRAM_RAW,
        "blocked audit-program pin drift",
    )
    blocked_path = safe_path(ROOT, str(blocked_binding["path"]))
    blocked_program = safe_path(ROOT, str(blocked_binding["audit_program_path"]))
    require(raw_sha256(blocked_path) == EXPECTED_BLOCKED_AUDIT_RAW, "blocked audit raw drift")
    require(
        raw_sha256(blocked_program) == EXPECTED_BLOCKED_AUDIT_PROGRAM_RAW,
        "blocked audit program raw drift",
    )
    blocked = read_json(blocked_path)
    require(
        blocked.get("content_sha256") == EXPECTED_BLOCKED_AUDIT_CONTENT
        and self_content_sha256(blocked) == EXPECTED_BLOCKED_AUDIT_CONTENT,
        "blocked audit self-seal drift",
    )
    require(
        blocked["status"] == "BLOCK_MECHANICAL_FORMAT_GATE_ONLY"
        and blocked["scientific_replay"] == "PASS_SOURCE_ONLY_SYNTHETIC"
        and blocked["mismatches"] == 0,
        "blocked audit disposition drift",
    )

    output_entries = sorted(path.name for path in (ROOT / V2_OUTPUT).iterdir())
    require(output_entries == ["receipt.json"], "v2 emitted a science mirror or extra artifact")
    require(not (ROOT / V2_RECEIPT).is_symlink(), "v2 receipt is a symlink")
    return config, receipt, {"bindings": verified, "blocked_audit": blocked}


def verify_scientific_semantics(
    config: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = {str(row["id"]): row for row in config["v1_bindings"]}
    compatibility = read_jsonl(binding_path(bindings, "V1_COMPATIBILITY_LEDGER"))
    cells = read_jsonl(binding_path(bindings, "V1_PARAMETER_CELL_DISPOSITION_LEDGER"))
    execution_bindings = read_jsonl(binding_path(bindings, "V1_EXECUTION_BINDINGS"))
    source_projections = read_jsonl(binding_path(bindings, "V1_SOURCE_PROJECTIONS_LEDGER"))
    unique = read_jsonl(binding_path(bindings, "V1_UNIQUE_EXECUTIONS"))
    replay = read_jsonl(binding_path(bindings, "V1_REPLAY_LEDGER"))
    ties = read_json(binding_path(bindings, "V1_EQUIVALENCE_TIES"))
    v1_receipt = read_json(binding_path(bindings, "V1_RECEIPT"))

    compatibility_counts = Counter(str(row["status"]) for row in compatibility)
    require(
        compatibility_counts
        == Counter({"EXECUTABLE": 110, "SOURCE_BLOCKED": 290, "INCOMPATIBLE_FEATURE_SET": 1600}),
        "compatibility counts mismatch",
    )
    formula_source_counts = Counter(str(row["formula_id"]) for row in compatibility)
    require(
        len(formula_source_counts) == 400 and set(formula_source_counts.values()) == {5},
        "400-by-5 compatibility structure mismatch",
    )
    require(len(cells) == 1184, "parameter-cell count mismatch")
    require(len(execution_bindings) == 126, "execution-binding count mismatch")
    require(len(source_projections) == 8, "source-projection object count mismatch")

    unique_counts = Counter(str(row["status"]) for row in unique)
    invalid_formula_counts = Counter(
        str(row["formula_id"]) for row in unique if row["status"] == "NUMERICAL_INVALID"
    )
    require(
        unique_counts == Counter({"COMPLETED": 2554, "NUMERICAL_INVALID": 38}),
        "unique execution counts mismatch",
    )
    require(
        invalid_formula_counts
        == Counter(
            {
                "TW2-A11-D03": 16,
                "TW2-A11-D04": 5,
                "TW2-A11-D06": 16,
                "TW2-A11-D07": 1,
            }
        ),
        "invalid formula distribution mismatch",
    )
    require(len({row["object_id"] for row in unique}) == 8, "unique object count mismatch")
    require(len({row["cell_id"] for row in unique}) == 324, "unique cell count mismatch")
    require(
        len({(row["object_id"], row["cell_id"]) for row in unique}) == 2592,
        "object-cell uniqueness mismatch",
    )
    require(
        all(
            row["scientific_score"] is False
            and row["deterministic_byte_identical"] is True
            and row["deterministic_repetitions"] == 2
            and row["numerical_valid"] == (row["status"] == "COMPLETED")
            for row in unique
        ),
        "unique execution status/access semantics mismatch",
    )

    predictions_path = binding_path(bindings, "V1_PREDICTIONS_VALUES")
    prediction_comparisons = 0
    with np.load(predictions_path, allow_pickle=False) as archive:
        members = set(archive.files)
        expected_members = {
            str(row[key]) for row in unique for key in ("factor_value_key", "g_eff_value_key")
        }
        require(
            members == expected_members and len(members) == 5184, "prediction inventory mismatch"
        )
        require(
            all(name.startswith("prediction__") for name in members), "forbidden prediction role"
        )
        for row in unique:
            require(
                array_sha256(archive[str(row["factor_value_key"])]) == row["factor_sha256"],
                "factor array hash mismatch",
            )
            require(
                array_sha256(archive[str(row["g_eff_value_key"])]) == row["g_eff_sha256"],
                "effective-acceleration array hash mismatch",
            )
            prediction_comparisons += 2
    require(prediction_comparisons == 5184, "prediction comparison count mismatch")

    tie_groups = Counter((str(row["object_id"]), str(row["prediction_sha256"])) for row in unique)
    independent_tie_count = sum(count > 1 for count in tie_groups.values())
    require(independent_tie_count == 122, "independent finite-source tie count mismatch")
    require(
        ties["finite_source_prediction_tie_group_count"] == 122
        and len(ties["finite_source_prediction_ties"]) == 122
        and ties["finite_fixture_similarity_promoted_to_formula_identity"] is False
        and all(
            row["promoted_to_formula_identity"] is False
            for row in ties["finite_source_prediction_ties"]
        ),
        "tie ledger semantics mismatch",
    )

    replay_counts = Counter(str(row["status"]) for row in replay)
    require(
        replay_counts == Counter({"COMPLETED": 61296, "NUMERICAL_INVALID": 912}),
        "replay counts mismatch",
    )
    unique_by_key = {(str(row["object_id"]), str(row["cell_id"])): row for row in unique}
    replay_multiplicity: Counter[tuple[str, str]] = Counter()
    for row in replay:
        key = (str(row["object_id"]), str(row["cell_id"]))
        require(key in unique_by_key, "replay references unknown unique execution")
        original = unique_by_key[key]
        require(
            row["status"] == original["status"]
            and row["numerical_valid"] == original["numerical_valid"]
            and row["prediction_sha256"] == original["prediction_sha256"]
            and row["result_sha256"] == original["result_sha256"]
            and row["metric_sha256"] == original["metric_sha256"],
            "replay-to-unique semantic mismatch",
        )
        require(
            row["response_value_accessed"] is False and row["scientific_score"] is False,
            "replay response/score boundary violated",
        )
        replay_multiplicity[key] += 1
    require(
        len(replay_multiplicity) == 2592 and set(replay_multiplicity.values()) == {24},
        "24-scenario replay fan-out mismatch",
    )
    require(len({row["scenario_id"] for row in replay}) == 192, "scenario count mismatch")

    source_members = 0
    with np.load(
        binding_path(bindings, "V1_SOURCE_PROJECTIONS_VALUES"), allow_pickle=False
    ) as archive:
        require(
            len(archive.files) == 160
            and all(name.startswith("source_projection__") for name in archive.files),
            "source-projection NPZ inventory mismatch",
        )
        for name in archive.files:
            array_sha256(archive[name])
            source_members += 1

    access = receipt["access_accounting"]
    require(
        access["response_npz_members_opened"] == 0
        and access["response_values_opened"] == 0
        and access["candidate_npz_members_opened"] == 0
        and access["variance_npz_members_opened"] == 0
        and access["truth_npz_members_opened"] == 0
        and access["scientific_scores_computed"] == 0,
        "v2 access accounting mismatch",
    )
    require(
        v1_receipt["access_accounting"]["response_values_opened"] == 0
        and v1_receipt["access_accounting"]["scientific_scores_computed"] == 0,
        "v1 access accounting mismatch",
    )
    require(
        receipt["claim_boundary"]
        == {
            "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
            "empirical_rejection_claimed": False,
            "empirical_support_claimed": False,
            "independent_pass_claimed": False,
            "other_274_cards_made_source_executable": False,
            "publication_readiness_claimed": False,
        },
        "claim boundary mismatch",
    )
    return {
        "card_count": 400,
        "parameter_cell_count": 1184,
        "compatibility_counts": dict(sorted(compatibility_counts.items())),
        "unique_execution_counts": dict(sorted(unique_counts.items())),
        "invalid_formula_counts": dict(sorted(invalid_formula_counts.items())),
        "replay_counts": dict(sorted(replay_counts.items())),
        "replay_fanout_per_unique_execution": 24,
        "source_scenario_count": 192,
        "finite_source_prediction_tie_group_count": independent_tie_count,
        "prediction_array_hash_comparisons": prediction_comparisons,
        "source_projection_array_members_verified": source_members,
        "response_values_opened": 0,
        "scientific_scores_computed": 0,
    }


def run_command(arguments: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    require(completed.returncode == 0, f"command failed: {' '.join(arguments)}")
    return {
        "argv": arguments,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def verify_toolchain() -> list[dict[str, Any]]:
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v1.py",
            "tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v2.py",
        ],
        [sys.executable, "-m", "ruff", "check", V2_MODULE.as_posix(), V2_TEST.as_posix()],
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            V2_MODULE.as_posix(),
            V2_TEST.as_posix(),
        ],
    ]
    return [run_command(command, ROOT) for command in commands]


def load_subject() -> ModuleType:
    path = ROOT / V2_MODULE
    spec = importlib.util.spec_from_file_location("frozen_twell_v2_subject", path)
    require(spec is not None and spec.loader is not None, "could not construct subject import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_subject_failure(
    label: str, operation: Callable[[], Any], error_type: type[BaseException]
) -> str:
    try:
        operation()
    except error_type as error:
        return f"{label}: {type(error).__name__}: {error}"
    raise AuditFailure(f"adversarial mutation was accepted: {label}")


def verify_adversarial_mutations(
    config: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[str]:
    subject = load_subject()
    error = subject.TwellSourceRebindV2Error
    outcomes: list[str] = []

    mutated = copy.deepcopy(config)
    mutated["status"] = "MUTATED"
    outcomes.append(
        expect_subject_failure(
            "config status mutation",
            lambda: subject.validate_config(ROOT, mutated, verify_files=False),
            error,
        )
    )

    mutated = copy.deepcopy(config)
    mutated["access_contract"]["response_values_opened"] = 1
    outcomes.append(
        expect_subject_failure(
            "response-access mutation",
            lambda: subject.validate_config(ROOT, mutated, verify_files=False),
            error,
        )
    )

    mutated = copy.deepcopy(config)
    mutated["claim_boundary"]["empirical_support_claimed"] = True
    outcomes.append(
        expect_subject_failure(
            "empirical-claim mutation",
            lambda: subject.validate_config(ROOT, mutated, verify_files=False),
            error,
        )
    )

    mutated = copy.deepcopy(config)
    mutated["v1_bindings"][0]["raw_sha256"] = "0" * 64
    outcomes.append(
        expect_subject_failure(
            "v1 raw-hash mutation",
            lambda: subject.validate_config(ROOT, mutated, verify_files=True),
            error,
        )
    )

    mutated = copy.deepcopy(config)
    mutated["preservation_contract"]["unique_execution_counts"] = {
        "COMPLETED": 2553,
        "NUMERICAL_INVALID": 39,
    }
    subject.validate_config(ROOT, mutated, verify_files=False)
    outcomes.append(
        expect_subject_failure(
            "sum-preserving semantic-count mutation",
            lambda: subject.validate_v1_scientific_payload(ROOT, mutated),
            error,
        )
    )

    outcomes.append(
        expect_subject_failure(
            "repository path traversal",
            lambda: subject._repo_path(ROOT, "../escape"),
            error,
        )
    )

    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["status"] = "MUTATED"
    require(
        tampered_receipt["content_sha256"] != self_content_sha256(tampered_receipt),
        "receipt mutation did not break independent self-seal",
    )
    outcomes.append("receipt status mutation: independent self-seal mismatch detected")

    with tempfile.TemporaryDirectory(prefix="twell-v2-adversarial-") as directory:
        temporary = Path(directory)
        forbidden_npz = temporary / "forbidden.npz"
        np.savez(forbidden_npz, response__forbidden=np.ones(2, dtype=np.float64))
        outcomes.append(
            expect_subject_failure(
                "forbidden response-role NPZ member",
                lambda: subject._npz_semantic_sha256(forbidden_npz),
                error,
            )
        )
        occupied = temporary / "occupied.json"
        occupied.write_bytes(b"frozen")
        outcomes.append(
            expect_subject_failure(
                "nonidentical no-clobber target",
                lambda: subject._atomic_no_clobber(occupied, b"mutated"),
                error,
            )
        )
    return outcomes


def verify_determinism_and_alternate_cwd() -> dict[str, Any]:
    subject = load_subject()
    stored = read_json(ROOT / V2_RECEIPT)
    first = subject.build_receipt(ROOT)
    second = subject.build_receipt(ROOT)
    require(first == second == stored, "subject deterministic rebuild mismatch")

    before = raw_sha256(ROOT / V2_RECEIPT)
    with tempfile.TemporaryDirectory(prefix="twell-v2-alternate-cwd-") as directory:
        alternate = Path(directory)
        check = run_command(
            [sys.executable, str(ROOT / V2_MODULE), "check", "--root", str(ROOT)],
            alternate,
        )
        write = run_command(
            [sys.executable, str(ROOT / V2_MODULE), "write", "--root", str(ROOT)],
            alternate,
        )
        require('"publication": "EXISTING_IDENTICAL"' in write["stdout"], "no-clobber not reported")
        require(not any(alternate.iterdir()), "alternate working directory was polluted")
        alternate_path = str(alternate)
    after = raw_sha256(ROOT / V2_RECEIPT)
    require(before == after == EXPECTED_V2_RAW[V2_RECEIPT.as_posix()], "receipt was clobbered")
    return {
        "two_in_process_rebuilds_byte_semantically_identical": True,
        "alternate_cwd": alternate_path,
        "alternate_cwd_check": check,
        "alternate_cwd_write": write,
        "publication_result": "EXISTING_IDENTICAL",
        "receipt_raw_before": before,
        "receipt_raw_after": after,
        "alternate_cwd_polluted": False,
    }


def unique_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise AuditFailure(f"refusing to replace existing audit receipt: {path}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def make_receipt(
    frozen: Mapping[str, Any],
    scientific: Mapping[str, Any],
    toolchain: Iterable[Mapping[str, Any]],
    adversarial: list[str],
    deterministic: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "open-gravity-independent-audit-receipt-1.0",
        "audit_id": "open-gravity-twell-400-source-shaped-rebind-replay-v2-independent-audit-pass-76f23426",
        "auditor_task": "/root/twell_v2_distinct_audit",
        "subject_package": "open-gravity-twell-400-source-shaped-rebind-replay-v2",
        "subject_module_sha256_prefix": "76f23426",
        "status": "PASS",
        "verdict": "PASS_APPEND_ONLY_FORMAT_ONLY_SUCCESSOR",
        "scope": "MECHANICAL_SUCCESSOR_AND_PRESERVED_SOURCE_ONLY_SYNTHETIC_SEMANTICS",
        "scientific_claim": "NONE_SOURCE_ONLY_SYNTHETIC_PAYLOAD_PRESERVED",
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "independence": {
            "builder_and_auditor_are_distinct_tasks": True,
            "evidence_reconstruction_imported_subject": False,
            "subject_loaded_after_reconstruction_for_mutation_and_determinism_checks_only": True,
            "v1_scientific_payload_recomputed": False,
            "v1_source_only_payload_hash_and_relationship_checks": True,
        },
        "audit_program": {
            "path": AUDIT_PROGRAM.as_posix(),
            "raw_sha256": raw_sha256(ROOT / AUDIT_PROGRAM),
        },
        "subject_v2_raw_sha256": dict(EXPECTED_V2_RAW),
        "subject_v2_receipt_content_sha256": EXPECTED_V2_RECEIPT_CONTENT,
        "v1_main_raw_sha256": dict(EXPECTED_V1_MAIN_RAW),
        "v1_binding_verification": frozen["bindings"],
        "blocked_v1_audit_binding": {
            "raw_sha256": EXPECTED_BLOCKED_AUDIT_RAW,
            "content_sha256": EXPECTED_BLOCKED_AUDIT_CONTENT,
            "audit_program_raw_sha256": EXPECTED_BLOCKED_AUDIT_PROGRAM_RAW,
            "status": frozen["blocked_audit"]["status"],
            "scientific_replay": frozen["blocked_audit"]["scientific_replay"],
            "mismatches": frozen["blocked_audit"]["mismatches"],
        },
        "preserved_scientific_semantics": dict(scientific),
        "v2_output_inventory": ["receipt.json"],
        "v2_science_mirror_emitted": False,
        "toolchain": list(toolchain),
        "determinism_and_no_clobber": dict(deterministic),
        "adversarial_mutations_rejected": adversarial,
        "access_accounting": {
            "response_npz_members_opened": 0,
            "response_values_opened": 0,
            "candidate_npz_members_opened": 0,
            "variance_npz_members_opened": 0,
            "truth_npz_members_opened": 0,
            "scientific_scores_computed": 0,
            "source_only_prediction_npz_members_verified": 5184,
            "source_projection_npz_members_verified": 160,
        },
        "limitations": [
            "This PASS admits only the append-only format/mechanical successor.",
            "V1 remains mechanically blocked; its independently matched source-only numerical payload is preserved rather than rehabilitated in place.",
            "Thirty-eight unique and 912 replay-level numerical invalids remain retained.",
            "One hundred twenty-two finite-source prediction ties are not formula identities.",
            "The other 274 cards remain source-blocked or incompatible; this successor does not make them executable.",
            "No response-valued member or scientific score was opened, computed, supported, or rejected.",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    receipt["content_sha256"] = self_content_sha256(receipt)
    return receipt


def main() -> int:
    config, subject_receipt, frozen = verify_frozen_chain()
    scientific = verify_scientific_semantics(config, subject_receipt)
    toolchain = verify_toolchain()
    adversarial = verify_adversarial_mutations(config, subject_receipt)
    deterministic = verify_determinism_and_alternate_cwd()
    receipt = make_receipt(frozen, scientific, toolchain, adversarial, deterministic)
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    unique_write(ROOT / AUDIT_RECEIPT, payload)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "verdict": receipt["verdict"],
                "audit_receipt": AUDIT_RECEIPT.as_posix(),
                "audit_receipt_raw_sha256": raw_sha256(ROOT / AUDIT_RECEIPT),
                "audit_receipt_content_sha256": receipt["content_sha256"],
                "response_values_opened": 0,
                "scientific_scores_computed": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
