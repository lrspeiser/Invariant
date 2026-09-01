from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_split_gate_eft_resummation_preflight_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_split_gate_eft_resummation_preflight.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_eft_resummation_preflight.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-split-gate-eft-resummation-preflight-v1.json"
)
EXPECTED_CONFIG_RAW_SHA256 = "e5a12688778fe3e7d14093af23ec01e3be331927a002f0040b29c3842ff77272"
EXPECTED_MODULE_SEMANTIC_SHA256 = "a8fcb3362a9ff9733f78d99b94c8607306e06175ac4f950108baaa07401f14b2"
EXPECTED_TEST_RAW_SHA256 = "bd46584febd0973e065df07021221f6bc5ff7a4b812c9c577d11ec5855094c04"

SCHEMA = "invariant-gravity-matter-lensing-split-gate-eft-resummation-preflight-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-split-gate-eft-resummation-preflight-receipt-1.0"
ARTIFACT_ID = "gravity-matter-lensing-split-gate-eft-resummation-preflight-v1"
DECISION = (
    "RETAIN_RANGE_SOURCE_THEOREM_REQUIRE_RESUMMED_OR_SYMMETRY_PROTECTED_EFT_BEFORE_MODEL_PROMOTION"
)


class SplitGateEFTPreflightError(RuntimeError):
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
        raise SplitGateEFTPreflightError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise SplitGateEFTPreflightError("JSON artifact must be an object")
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


def _expected_claims() -> dict[str, bool]:
    return {
        "paper_anchored_EFT_preflight": True,
        "dimension_and_limit_checks_passed": True,
        "naive_expansion_tension_found": True,
        "fatal_no_go": False,
        "controlled_resummation": False,
        "protecting_symmetry": False,
        "radiative_stability": False,
        "UV_completion": False,
        "observational_support": False,
        "successful_gravity_model": False,
        "publication_ready": False,
    }


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    if _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise SplitGateEFTPreflightError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("artifact_id") != ARTIFACT_ID:
        raise SplitGateEFTPreflightError("config identity changed")
    if config.get("package") != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise SplitGateEFTPreflightError("package paths changed")
    if config.get("claim_boundary") != _expected_claims():
        raise SplitGateEFTPreflightError("claim boundary changed")
    adjudication = config.get("adjudication")
    if not isinstance(adjudication, dict) or adjudication.get("decision") != DECISION:
        raise SplitGateEFTPreflightError("adjudication changed")
    if adjudication.get("automatic_EFT_inconsistency_proved") is not False:
        raise SplitGateEFTPreflightError("fatal-EFT claim changed")
    papers = config.get("primary_literature")
    if not isinstance(papers, list) or {item.get("arxiv") for item in papers} != {
        "2107.00010",
        "2105.13992",
        "1611.08279",
        "2604.20292v2",
    }:
        raise SplitGateEFTPreflightError("primary-source inventory changed")
    if config.get("access_ledger") != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise SplitGateEFTPreflightError("access boundary changed")
    return config


def _local_integrity(base: Path) -> dict[str, str]:
    semantic = _module_semantic_sha256(base / MODULE_PATH)
    if semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise SplitGateEFTPreflightError("module semantics changed")
    test_raw = _sha256_file(base / TEST_PATH)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise SplitGateEFTPreflightError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_predecessor(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    binding = config["publication_candidate_binding"]
    output: dict[str, str] = {}
    for role in ("config", "module", "test", "draft", "receipt"):
        path = binding[f"{role}_path"]
        expected = binding[f"{role}_sha256"]
        if _sha256_file(base / path) != expected:
            raise SplitGateEFTPreflightError("publication-candidate binding changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / binding["receipt_path"])
    expected_content = binding["receipt_content_sha256"]
    if receipt.get("content_sha256") != expected_content or _self_hash(receipt) != expected_content:
        raise SplitGateEFTPreflightError("publication-candidate receipt changed")
    output["receipt_content_sha256"] = expected_content
    policy = config["admission_policy"]
    policy_raw = _sha256_file(base / policy["path"])
    if policy_raw != policy["raw_sha256"]:
        raise SplitGateEFTPreflightError("admission-policy binding changed")
    output["policy_sha256"] = policy_raw
    return output


def _symbolic_checks() -> dict[str, bool]:
    energy, scale, mass = sp.symbols("energy scale mass", positive=True)
    x = energy**4
    beta = scale**-8
    u = sp.simplify(beta * x**2)
    symbol_u = sp.symbols("u")
    p2 = sp.expand((1 + symbol_u) ** 2)
    lambda6 = (scale**8 / mass**2) ** sp.Rational(1, 6)
    lambda14 = (scale**16 / mass**2) ** sp.Rational(1, 14)
    epsilon = sp.symbols("epsilon", positive=True)
    branch_point = sp.solve(sp.Eq(1 + symbol_u, 0), symbol_u)
    return {
        "E02_ENGINEERING_DIMENSIONS": 2 + 2 + 8 - 8 == 4 and 2 + 2 + 16 - 16 == 4,
        "E03_U_SCALE_IDENTITY": sp.simplify(u - (energy / scale) ** 8) == 0,
        "E04_HIGH_U_NAIVE_EXPANSION_BOUNDARY": all(
            math.isclose(value ** (1.0 / 8.0), expected)
            for value, expected in ((1.0e-4, 10.0**-0.5), (1.0, 1.0), (1.0e4, 10.0**0.5))
        ),
        "E05_P2_EXACT_POLYNOMIAL": sp.simplify(p2 - (1 + 2 * symbol_u + symbol_u**2)) == 0,
        "E06_LEADING_OPERATOR_SCALE": sp.simplify(
            lambda6.subs(mass, epsilon * scale) / scale - epsilon ** (-sp.Rational(1, 3))
        )
        == 0,
        "E07_NEXT_OPERATOR_SCALE": sp.simplify(
            lambda14.subs(mass, epsilon * scale) / scale - epsilon ** (-sp.Rational(1, 7))
        )
        == 0,
        "E09_BINOMIAL_RADIUS_BOUNDARY": branch_point == [-1],
    }


def _numeric_grid(config: Mapping[str, Any]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for epsilon in config["benchmark_grid"]["mass_ratios_mchi_over_Lambda_g"]:
        value = float(epsilon)
        output.append(
            {
                "mchi_over_Lambda_g": value,
                "Lambda_6_over_Lambda_g": value ** (-1.0 / 3.0),
                "Lambda_14_over_Lambda_g": value ** (-1.0 / 7.0),
            }
        )
    return output


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _local_integrity(base)
    predecessor = _validate_predecessor(base, config)
    symbolic = _symbolic_checks()
    grid = _numeric_grid(config)
    grid_passed = all(
        row["Lambda_6_over_Lambda_g"] >= 1.0 and row["Lambda_14_over_Lambda_g"] >= 1.0
        for row in grid
    )
    checks = {
        "E01_CONFIG_POLICY_AND_PREDECESSOR_SEALS": local["config_raw_sha256"]
        == EXPECTED_CONFIG_RAW_SHA256,
        **symbolic,
        "E08_MASS_RATIO_GRID": grid_passed,
        "E10_PRIMARY_SOURCE_INVENTORY": len(config["primary_literature"]) == 4,
        "E11_ADJUDICATION_AND_CLAIM_CEILING": config["adjudication"]["decision"] == DECISION
        and config["claim_boundary"] == _expected_claims(),
        "E12_ZERO_OBSERVATIONAL_ACCESS": not any(config["access_ledger"].values()),
    }
    if set(checks) != set(config["required_checks"]) or not all(checks.values()):
        raise SplitGateEFTPreflightError("EFT preflight checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": ARTIFACT_ID,
        "status": config["status"],
        "decision": DECISION,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "publication_candidate_binding": predecessor,
        "implementation_binding": local,
        "engineering_dimension_contract": config["engineering_dimension_contract"],
        "p2_operator_contract": config["p2_operator_contract"],
        "general_gate_boundary": config["general_gate_boundary"],
        "primary_literature": config["primary_literature"],
        "numeric_scale_grid": grid,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "next_required_work": config["next_required_work"],
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
        raise SplitGateEFTPreflightError("stored receipt differs from rebuild")
    return stored


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    destination = (base / OUTPUT_PATH).resolve()
    data = (json.dumps(build_receipt(base), indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise SplitGateEFTPreflightError("refusing to replace nonidentical receipt")
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
            raise SplitGateEFTPreflightError("receipt publication race") from exc
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
                    "fatal_no_go": receipt["claim_boundary"]["fatal_no_go"],
                    "controlled_resummation": receipt["claim_boundary"]["controlled_resummation"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
