from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path(
    "configs/gravity_matter_lensing_split_gate_range_source_publication_candidate_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "gravity_matter_lensing_split_gate_range_source_publication_candidate.py"
)
TEST_PATH = Path(
    "tests/test_gravity_matter_lensing_split_gate_range_source_publication_candidate.py"
)
DRAFT_PATH = Path("docs/GRAVITY_SPLIT_GATE_RANGE_SOURCE_TRADEOFF_THEORY_NOTE_V1.md")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-split-gate-range-source-publication-candidate-v1.json"
)
EXPECTED_CONFIG_RAW_SHA256 = "730fdf89ca22827b310cb3481e435ced222ac71dc246d7d0f6254534c7bdb4a4"
EXPECTED_MODULE_SEMANTIC_SHA256 = "1caa214422ab32d8721077956f521d8af01b4ae0f7d75dff354b16a017e4d3b6"
EXPECTED_TEST_RAW_SHA256 = "4746f9542738d451899281f9787a33380dca75aaa9d0dff7717edb9cfb1ffeb2"

SCHEMA = "invariant-gravity-matter-lensing-split-gate-range-source-publication-candidate-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-matter-lensing-split-gate-range-source-publication-candidate-receipt-1.0"
)
ARTIFACT_ID = "gravity-matter-lensing-split-gate-range-source-publication-candidate-v1"
DECISION = "NARROW_RANGE_SOURCE_TRADEOFF_NOTE_CANDIDATE_PENDING_EXPERT_PRIORITY_AND_EFT_REVIEW"


class SplitGateRangeSourcePublicationError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitGateRangeSourcePublicationError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise SplitGateRangeSourcePublicationError("JSON artifact must be an object")
    return value


def _content_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _content_sha256(body)


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:"[0-9a-f]{64}"|"__MODULE_SEMANTIC_SHA256__")',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SplitGateRangeSourcePublicationError("committed predecessor unavailable")
    return completed.stdout


def _expected_claims() -> dict[str, bool]:
    return {
        "exact_restricted_theorem": True,
        "primary_source_benchmark_passed": True,
        "independent_numeric_benchmark_passed": True,
        "scientifically_interesting": True,
        "narrow_note_candidate": True,
        "historical_novelty_established": False,
        "independent_expert_review_passed": False,
        "radiative_stability_established": False,
        "full_coupled_solution": False,
        "successful_gravity_model": False,
        "observational_support": False,
        "publication_ready": False,
    }


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    if _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise SplitGateRangeSourcePublicationError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("artifact_id") != ARTIFACT_ID:
        raise SplitGateRangeSourcePublicationError("config identity changed")
    if config.get("package") != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "draft_path": DRAFT_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise SplitGateRangeSourcePublicationError("package paths changed")
    if config.get("claim_boundary") != _expected_claims():
        raise SplitGateRangeSourcePublicationError("claim boundary changed")
    adjudication = config.get("publication_adjudication")
    if not isinstance(adjudication, dict) or adjudication.get("decision") != DECISION:
        raise SplitGateRangeSourcePublicationError("publication decision changed")
    if adjudication.get("worth_claiming_as_successful_gravity_model") is not False:
        raise SplitGateRangeSourcePublicationError("model-success claim changed")
    if config.get("access_ledger") != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise SplitGateRangeSourcePublicationError("access boundary changed")
    return config


def _local_integrity(base: Path) -> dict[str, str]:
    module = base / MODULE_PATH
    semantic = _module_semantic_sha256(module)
    if semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise SplitGateRangeSourcePublicationError("module semantics changed")
    test_raw = _sha256_file(base / TEST_PATH)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise SplitGateRangeSourcePublicationError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_binding_set(
    base: Path, binding: Mapping[str, Any], *, committed: bool
) -> dict[str, str]:
    output: dict[str, str] = {}
    commit = binding.get("commit")
    if committed and not isinstance(commit, str):
        raise SplitGateRangeSourcePublicationError("missing committed predecessor")
    for role in ("config", "module", "test", "receipt"):
        path = binding[f"{role}_path"]
        expected = binding[f"{role}_sha256"]
        current = _sha256_file(base / path)
        if current != expected:
            raise SplitGateRangeSourcePublicationError("bound predecessor bytes changed")
        if committed and _sha256_bytes(_git_show(base, commit, path)) != expected:
            raise SplitGateRangeSourcePublicationError("committed predecessor bytes changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / binding["receipt_path"])
    expected_content = binding["receipt_content_sha256"]
    if receipt.get("content_sha256") != expected_content or _self_hash(receipt) != expected_content:
        raise SplitGateRangeSourcePublicationError("bound receipt content changed")
    output["receipt_content_sha256"] = expected_content
    if committed:
        output["commit"] = str(commit)
    return output


def _validate_policy_and_draft(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    binding = config["policy_and_draft"]
    policy = _sha256_file(base / binding["policy_path"])
    draft = _sha256_file(base / binding["draft_path"])
    if policy != binding["policy_sha256"] or draft != binding["draft_sha256"]:
        raise SplitGateRangeSourcePublicationError("policy or draft bytes changed")
    text = (base / binding["draft_path"]).read_text(encoding="utf-8")
    markers = (
        "architecture-class leading law",
        "not a universal law of gravity",
        "What would falsify or demote the result",
        "Claims not made",
        "Historical novelty",
    )
    if not all(marker in text for marker in markers):
        raise SplitGateRangeSourcePublicationError("draft scope markers changed")
    return {
        "policy_sha256": policy,
        "draft_sha256": draft,
    }


def _symbolic_checks() -> dict[str, bool]:
    x, s, z, mass, y0, a_inf, b_inf = sp.symbols("x s z mass y0 a_inf b_inf", positive=True)
    gate = z * x**s
    gate_x = sp.diff(gate, x)
    gate_h = gate_x + 2 * x * sp.diff(gate, x, 2)
    chi_c = sp.sqrt(2 * a_inf / (mass**2 * gate_x))
    chi_k = sp.sqrt(2 * b_inf / (mass**2 * gate_h))
    ell = sp.sqrt(y0) / (mass * sp.sqrt(gate))
    q_c = mass**2 * gate * chi_c
    q_k = mass**2 * gate * chi_k
    product_c = sp.sqrt(y0) * sp.sqrt(2 * a_inf / s) * sp.sqrt(x)
    product_k = sp.sqrt(y0) * sp.sqrt(2 * b_inf / (s * (2 * s - 1))) * sp.sqrt(x)
    r = sp.symbols("r", real=True)
    return {
        "P04_AMPLITUDE_SCALING": sp.simplify(
            chi_c / (sp.sqrt(2 * a_inf / (mass**2 * s * z)) * x ** (-(s - 1) / 2))
        )
        == 1
        and sp.simplify(
            chi_k / (sp.sqrt(2 * b_inf / (mass**2 * s * (2 * s - 1) * z)) * x ** (-(s - 1) / 2))
        )
        == 1,
        "P05_RANGE_SCALING": sp.simplify(ell / (sp.sqrt(y0) / (mass * sp.sqrt(z)) * x ** (-s / 2)))
        == 1,
        "P06_SOURCE_SCALING": sp.simplify(
            q_c / (mass * sp.sqrt(2 * a_inf * z / s) * x ** ((s + 1) / 2))
        )
        == 1
        and sp.simplify(
            q_k / (mass * sp.sqrt(2 * b_inf * z / (s * (2 * s - 1))) * x ** ((s + 1) / 2))
        )
        == 1,
        "P07_PRODUCT_CANCELLATION": sp.simplify(q_c * ell / product_c) == 1
        and sp.simplify(q_k * ell / product_k) == 1,
        "P08_CRITICAL_EXPONENT": sp.simplify(r - (s + 1) / 2) == r - s / 2 - sp.Rational(1, 2),
        "P09_NO_FREE_RANGE": sp.simplify((2 * r - 1) / 2 - (r - sp.Rational(1, 2))) == 0,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _local_integrity(base)
    theorem = _validate_binding_set(base, config["theorem_predecessor"], committed=True)
    novelty = _validate_binding_set(base, config["novelty_benchmark"], committed=False)
    policy_draft = _validate_policy_and_draft(base, config)
    symbolic = _symbolic_checks()
    checks = {
        "P01_CONFIG_POLICY_AND_DRAFT_SEALS": local["config_raw_sha256"]
        == EXPECTED_CONFIG_RAW_SHA256,
        "P02_COMMITTED_THEOREM_BINDING": theorem["commit"]
        == config["theorem_predecessor"]["commit"],
        "P03_NOVELTY_BENCHMARK_BINDING": novelty["receipt_content_sha256"]
        == config["novelty_benchmark"]["receipt_content_sha256"],
        **symbolic,
        "P10_DRAFT_SCOPE_MARKERS": policy_draft["draft_sha256"]
        == config["policy_and_draft"]["draft_sha256"],
        "P11_PUBLICATION_ADJUDICATION": config["publication_adjudication"][
            "scientifically_interesting"
        ]
        is True
        and config["publication_adjudication"]["worth_preparing_as_narrow_theory_note"] is True
        and config["publication_adjudication"]["worth_claiming_as_successful_gravity_model"]
        is False,
        "P12_CLAIM_CEILING_AND_ZERO_ACCESS": config["claim_boundary"] == _expected_claims()
        and not any(config["access_ledger"].values()),
    }
    if set(checks) != set(config["required_checks"]) or not all(checks.values()):
        raise SplitGateRangeSourcePublicationError("publication checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": ARTIFACT_ID,
        "status": config["status"],
        "decision": DECISION,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "theorem_binding": theorem,
        "novelty_binding": novelty,
        "policy_and_draft_binding": policy_draft,
        "implementation_binding": local,
        "maximal_claim": config["maximal_claim"],
        "paper_support": config["paper_support"],
        "publication_adjudication": config["publication_adjudication"],
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    stored = _read_json(base / OUTPUT_PATH)
    expected = build_receipt(base)
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise SplitGateRangeSourcePublicationError("stored receipt differs from rebuild")
    return stored


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    destination = (base / OUTPUT_PATH).resolve()
    data = (json.dumps(build_receipt(base), indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise SplitGateRangeSourcePublicationError("refusing to replace nonidentical receipt")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise SplitGateRangeSourcePublicationError("receipt publication race") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        receipt = validate_receipt()
        print(f"VALID {receipt['content_sha256']}")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "checks_passed": receipt["checks_passed"],
                    "scientifically_interesting": receipt["claim_boundary"][
                        "scientifically_interesting"
                    ],
                    "successful_gravity_model": receipt["claim_boundary"][
                        "successful_gravity_model"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
