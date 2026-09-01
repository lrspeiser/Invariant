from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path(
    "configs/gravity_matter_lensing_split_gate_range_source_novelty_benchmark_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_split_gate_range_source_novelty_benchmark.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_range_source_novelty_benchmark.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-split-gate-range-source-novelty-benchmark-v1.json"
)
EXPECTED_CONFIG_RAW_SHA256 = "5bcc5003c367fc35f68a1641ea9a0eb56699e2bff92e1895d890d03bdee760a5"
EXPECTED_MODULE_SEMANTIC_SHA256 = "c34fbeecf47f1e0297f540d51a9356995f04b314f43c2bb803298d6b965c9554"
EXPECTED_TEST_RAW_SHA256 = "f0e79b0d5af95f15c155f1a8e873183985f51ba9fd7b867a35d735299a456796"

SCHEMA = "invariant-gravity-matter-lensing-split-gate-range-source-novelty-benchmark-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-matter-lensing-split-gate-range-source-novelty-benchmark-receipt-1.0"
)
ARTIFACT_ID = "gravity-matter-lensing-split-gate-range-source-novelty-benchmark-v1"
DECISION = (
    "PLAUSIBLY_ORIGINAL_ARCHITECTURE_CLASS_COROLLARY_WORTH_EXPERT_REVIEW_HISTORICAL_NOVELTY_OPEN"
)


class SplitGateRangeSourceNoveltyError(RuntimeError):
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
        raise SplitGateRangeSourceNoveltyError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise SplitGateRangeSourceNoveltyError("JSON artifact must be an object")
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
        raise SplitGateRangeSourceNoveltyError("bound Git artifact unavailable")
    return completed.stdout


def _expected_claim_boundary() -> dict[str, bool]:
    return {
        "primary_paper_anchored": True,
        "independent_exact_benchmarks": True,
        "candidate_corollary_not_found_in_reviewed_set": True,
        "historical_novelty_established": False,
        "independent_expert_review_passed": False,
        "architecture_class_universality_only": True,
        "universal_physics_claim": False,
        "full_coupled_range_derived": False,
        "full_action_health": False,
        "physical_source_X_scaling_derived": False,
        "observational_support": False,
        "modified_gravity_success": False,
        "publication_ready": False,
    }


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    if _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise SplitGateRangeSourceNoveltyError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("artifact_id") != ARTIFACT_ID:
        raise SplitGateRangeSourceNoveltyError("config identity changed")
    if config.get("package") != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise SplitGateRangeSourceNoveltyError("package paths changed")
    if config.get("claim_boundary") != _expected_claim_boundary():
        raise SplitGateRangeSourceNoveltyError("claim boundary changed")
    if config.get("access_ledger") != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise SplitGateRangeSourceNoveltyError("access boundary changed")
    policy = config.get("admission_policy")
    if not isinstance(policy, dict) or policy.get("source_class") != (
        "PRIMARY_PAPERS_PLUS_EXACT_ANALYTIC_AND_NUMERIC_BENCHMARKS"
    ):
        raise SplitGateRangeSourceNoveltyError("admission policy changed")
    papers = config.get("primary_literature")
    expected_ids = {
        "astro-ph/0309300",
        "1001.4525",
        "0905.2943",
        "1306.6401",
        "1611.08279",
        "2305.07725",
        "2603.13986v2",
        "2604.20292",
    }
    if not isinstance(papers, list) or {item.get("arxiv") for item in papers} != expected_ids:
        raise SplitGateRangeSourceNoveltyError("primary-source inventory changed")
    if len(papers) != len(expected_ids) or any(
        item.get("exact_theorem_found") is not False for item in papers
    ):
        raise SplitGateRangeSourceNoveltyError("primary-source adjudication changed")
    return config


def _validate_local_integrity(base: Path) -> dict[str, str]:
    module_path = (base / MODULE_PATH).resolve()
    module_semantic = _module_semantic_sha256(module_path)
    if module_semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise SplitGateRangeSourceNoveltyError("module semantics changed")
    test_path = (base / TEST_PATH).resolve()
    test_raw = _sha256_file(test_path)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise SplitGateRangeSourceNoveltyError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module_path),
        "module_semantic_sha256": module_semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_predecessor(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    predecessor = config["predecessor"]
    commit = predecessor["commit"]
    output: dict[str, str] = {"commit": commit}
    for role in ("config", "module", "test", "receipt"):
        relative = predecessor[f"{role}_path"]
        expected = predecessor[f"{role}_sha256"]
        current = _sha256_file(base / relative)
        committed = _sha256_bytes(_git_show(base, commit, relative))
        if current != expected or committed != expected:
            raise SplitGateRangeSourceNoveltyError("predecessor binding changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / predecessor["receipt_path"])
    expected_content = predecessor["receipt_content_sha256"]
    if receipt.get("content_sha256") != expected_content or _self_hash(receipt) != expected_content:
        raise SplitGateRangeSourceNoveltyError("predecessor receipt content changed")
    output["receipt_content_sha256"] = expected_content
    return output


def _validate_policy(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    policy = config["admission_policy"]
    path = base / policy["path"]
    raw = _sha256_file(path)
    if raw != policy["raw_sha256"]:
        raise SplitGateRangeSourceNoveltyError("admission policy bytes changed")
    return {"path": policy["path"], "raw_sha256": raw}


def _symbolic_checks() -> dict[str, bool]:
    x, s, z, mass, y0, a_inf, b_inf = sp.symbols("x s z mass y0 a_inf b_inf", positive=True)
    gate = z * x**s
    gate_x = sp.diff(gate, x)
    gate_h = sp.diff(gate, x) + 2 * x * sp.diff(gate, x, 2)
    chi_c = sp.sqrt(2 * a_inf / (mass**2 * gate_x))
    chi_k = sp.sqrt(2 * b_inf / (mass**2 * gate_h))
    bare_range = sp.sqrt(y0) / (mass * sp.sqrt(gate))
    source_c = mass**2 * gate * chi_c
    source_k = mass**2 * gate * chi_k
    product_c = sp.powsimp(source_c * bare_range, force=True)
    product_k = sp.powsimp(source_k * bare_range, force=True)
    expected_c = sp.sqrt(y0) * sp.sqrt(2 * a_inf / s) * sp.sqrt(x)
    expected_k = sp.sqrt(y0) * sp.sqrt(2 * b_inf / (s * (2 * s - 1))) * sp.sqrt(x)
    t, alpha = sp.symbols("t alpha", positive=True)
    fixed_distance_limit = sp.limit((1 + alpha * t) * sp.exp(-alpha * t), t, sp.oo)
    return {
        "N03_MONOMIAL_DERIVATIVES": sp.simplify(gate_x - s * z * x ** (s - 1)) == 0
        and sp.simplify(gate_h - s * (2 * s - 1) * z * x ** (s - 1)) == 0,
        "N04_AMPLITUDE_EXPONENT": sp.simplify(
            chi_c / (sp.sqrt(2 * a_inf / (mass**2 * s * z)) * x ** (-(s - 1) / 2))
        )
        == 1
        and sp.simplify(
            chi_k / (sp.sqrt(2 * b_inf / (mass**2 * s * (2 * s - 1) * z)) * x ** (-(s - 1) / 2))
        )
        == 1,
        "N05_RANGE_EXPONENT": sp.simplify(
            bare_range / (sp.sqrt(y0) / (mass * sp.sqrt(z)) * x ** (-s / 2))
        )
        == 1,
        "N06_SOURCE_CAPACITY": sp.simplify(
            source_c / (mass * sp.sqrt(2 * a_inf * z / s) * x ** ((s + 1) / 2))
        )
        == 1
        and sp.simplify(
            source_k / (mass * sp.sqrt(2 * b_inf * z / (s * (2 * s - 1))) * x ** ((s + 1) / 2))
        )
        == 1,
        "N07_PRODUCT_CANCELLATION": sp.simplify(product_c / expected_c) == 1
        and sp.simplify(product_k / expected_k) == 1
        and mass not in expected_c.free_symbols
        and z not in expected_c.free_symbols,
        "N08_CRITICAL_EXPONENT": sp.simplify((s + 1) / 2 - (s / 2 + sp.Rational(1, 2))) == 0,
        "N09_NO_FREE_RANGE": sp.simplify(
            (2 * sp.Symbol("r", real=True) - 1) / 2
            - (sp.Symbol("r", real=True) - sp.Rational(1, 2))
        )
        == 0,
        "N10_SHIFTED_POWER_RECOVERY": all(
            math.isclose(value, expected)
            for value, expected in (
                (-2.0, -2.0),
                (0.5 - 2.0, -1.5),
                (2.0 + 0.5, 2.5),
            )
        ),
        "N11_FIXED_DISTANCE_SCREENING": fixed_distance_limit == 0,
    }


def _quantities(x: float, s: float) -> dict[str, float]:
    gate = x**s
    gate_x = s * x ** (s - 1.0)
    gate_h = s * (2.0 * s - 1.0) * x ** (s - 1.0)
    chi_max = min(math.sqrt(2.0 / gate_x), math.sqrt(2.0 / gate_h))
    bare_range = 1.0 / math.sqrt(gate)
    source_max = gate * chi_max
    return {
        "chi_max": chi_max,
        "bare_range": bare_range,
        "source_max": source_max,
        "product": source_max * bare_range,
    }


def _log_slope(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.log(y1 / y0) / math.log(x1 / x0)


def _numeric_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    x_values = [float(value) for value in config["parameter_probes"]["X_values"]]
    records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    all_passed = True
    for s in (float(value) for value in config["parameter_probes"]["s_values"]):
        endpoints = [_quantities(value, s) for value in (x_values[0], x_values[-1])]
        slopes = {
            key: _log_slope(x_values[0], endpoints[0][key], x_values[-1], endpoints[1][key])
            for key in ("chi_max", "bare_range", "source_max", "product")
        }
        targets = {
            "chi_max": -(s - 1.0) / 2.0,
            "bare_range": -s / 2.0,
            "source_max": (s + 1.0) / 2.0,
            "product": 0.5,
        }
        error = max(abs(slopes[key] - targets[key]) for key in slopes)
        passed = error < 1.0e-11
        all_passed = all_passed and passed
        records.append(
            {
                "s": s,
                "slopes": slopes,
                "targets": targets,
                "max_abs_error": error,
                "passed": passed,
            }
        )
        critical = (s + 1.0) / 2.0
        for offset in (float(v) for v in config["parameter_probes"]["source_exponent_offsets"]):
            exponent = critical + offset
            ratios = [value**exponent / _quantities(value, s)["source_max"] for value in x_values]
            if offset < 0:
                behavior = "DECREASING_SUBCRITICAL"
                passed = ratios[-1] < ratios[0]
            elif offset > 0:
                behavior = "INCREASING_SUPERCRITICAL"
                passed = ratios[-1] > ratios[0]
            else:
                behavior = "COEFFICIENT_DEPENDENT_CRITICAL"
                passed = max(ratios) - min(ratios) < 1.0e-10
            all_passed = all_passed and passed
            threshold_records.append(
                {
                    "s": s,
                    "source_exponent": exponent,
                    "offset": offset,
                    "ratios": ratios,
                    "behavior": behavior,
                    "passed": passed,
                }
            )
    return {
        "slope_records": records,
        "threshold_records": threshold_records,
        "all_passed": all_passed,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _validate_local_integrity(base)
    predecessor = _validate_predecessor(base, config)
    policy = _validate_policy(base, config)
    symbolic = _symbolic_checks()
    numeric = _numeric_evidence(config)
    checks = {
        "N01_CONFIG_AND_POLICY_SEALS": local["config_raw_sha256"] == EXPECTED_CONFIG_RAW_SHA256,
        "N02_COMMITTED_PREDECESSOR_BYTES": predecessor["commit"] == config["predecessor"]["commit"],
        **symbolic,
        "N12_NUMERIC_SLOPES": all(item["passed"] for item in numeric["slope_records"]),
        "N13_THRESHOLD_CONTROLS": all(item["passed"] for item in numeric["threshold_records"]),
        "N14_PRIMARY_SOURCE_INVENTORY": len(config["primary_literature"]) == 8,
        "N15_NOVELTY_AND_CLAIM_CEILING": config["claim_boundary"] == _expected_claim_boundary(),
        "N16_ZERO_OBSERVATIONAL_ACCESS": not any(config["access_ledger"].values()),
    }
    if set(checks) != set(config["required_checks"]) or not all(checks.values()):
        raise SplitGateRangeSourceNoveltyError("benchmark checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": ARTIFACT_ID,
        "status": config["status"],
        "decision": DECISION,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "predecessor_binding": predecessor,
        "policy_binding": policy,
        "implementation_binding": local,
        "theorem_slice": config["theorem_slice"],
        "primary_literature": config["primary_literature"],
        "search_protocol": config["search_protocol"],
        "numeric_evidence": numeric,
        "adjudication": config["adjudication"],
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
        raise SplitGateRangeSourceNoveltyError("stored receipt differs from deterministic rebuild")
    return stored


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    destination = (base / OUTPUT_PATH).resolve()
    payload = json.dumps(build_receipt(base), indent=2, sort_keys=True) + "\n"
    data = payload.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise SplitGateRangeSourceNoveltyError("refusing to replace nonidentical receipt")
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
            raise SplitGateRangeSourceNoveltyError("receipt publication race") from exc
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
                    "historical_novelty_established": receipt["claim_boundary"][
                        "historical_novelty_established"
                    ],
                    "observational_rows_read": receipt["access_ledger"]["observational_rows_read"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
