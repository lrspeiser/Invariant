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

CONFIG_PATH = Path("configs/gravity_matter_lensing_split_gate_periodic_spurion_completion_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_split_gate_periodic_spurion_completion.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_periodic_spurion_completion.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-split-gate-periodic-spurion-completion-v1.json"
)
EXPECTED_CONFIG_RAW_SHA256 = "ce25279ef33bccf238aef6eb40d614740f7a10fa66792b9b1b785a7e54ca653b"
EXPECTED_MODULE_SEMANTIC_SHA256 = "7415254520d53b2afb367d58c208e8eafd78dc1483a73e4c448675ca3370fef3"
EXPECTED_TEST_RAW_SHA256 = "78713929f56778f68339826e9a03d1890c756799dc80e8a0758613f4b0456da1"

SCHEMA = "invariant-gravity-matter-lensing-split-gate-periodic-spurion-completion-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-matter-lensing-split-gate-periodic-spurion-completion-receipt-1.0"
)
ARTIFACT_ID = "gravity-matter-lensing-split-gate-periodic-spurion-completion-v1"
DECISION = (
    "PARTIAL_PERIODIC_COMPLETION_RETAINS_P2_TRADEOFF_BUT_DOES_NOT_SOLVE_LARGE_U_GATE_RESUMMATION"
)


class SplitGatePeriodicCompletionError(RuntimeError):
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
        raise SplitGatePeriodicCompletionError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise SplitGatePeriodicCompletionError("JSON artifact must be an object")
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
        "paper_anchored_completion": True,
        "exact_periodic_symmetry": True,
        "small_field_recovery": True,
        "strong_gate_tradeoff_recovery": True,
        "fatal_no_go": False,
        "radiative_stability": False,
        "large_u_resummation": False,
        "full_coupled_health": False,
        "observational_support": False,
        "successful_gravity_model": False,
        "publication_ready": False,
    }


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    if _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise SplitGatePeriodicCompletionError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("artifact_id") != ARTIFACT_ID:
        raise SplitGatePeriodicCompletionError("config identity changed")
    if config.get("package") != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise SplitGatePeriodicCompletionError("package paths changed")
    if config.get("claim_boundary") != _expected_claims():
        raise SplitGatePeriodicCompletionError("claim boundary changed")
    if config.get("adjudication", {}).get("decision") != DECISION:
        raise SplitGatePeriodicCompletionError("adjudication changed")
    papers = config.get("primary_literature")
    if not isinstance(papers, list) or {item.get("arxiv") for item in papers} != {
        "hep-ph/9503331",
        "1611.08279",
        "2107.00010",
        "2604.20292v2",
    }:
        raise SplitGatePeriodicCompletionError("primary-source inventory changed")
    if config.get("access_ledger") != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise SplitGatePeriodicCompletionError("access boundary changed")
    return config


def _local_integrity(base: Path) -> dict[str, str]:
    semantic = _module_semantic_sha256(base / MODULE_PATH)
    if semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise SplitGatePeriodicCompletionError("module semantics changed")
    test_raw = _sha256_file(base / TEST_PATH)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise SplitGatePeriodicCompletionError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_predecessor(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    binding = config["eft_preflight_binding"]
    output: dict[str, str] = {}
    for role in ("config", "module", "test", "receipt"):
        expected = binding[f"{role}_sha256"]
        if _sha256_file(base / binding[f"{role}_path"]) != expected:
            raise SplitGatePeriodicCompletionError("EFT-preflight binding changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / binding["receipt_path"])
    expected_content = binding["receipt_content_sha256"]
    if receipt.get("content_sha256") != expected_content or _self_hash(receipt) != expected_content:
        raise SplitGatePeriodicCompletionError("EFT-preflight receipt changed")
    output["receipt_content_sha256"] = expected_content
    policy = config["admission_policy"]
    policy_raw = _sha256_file(base / policy["path"])
    if policy_raw != policy["raw_sha256"]:
        raise SplitGatePeriodicCompletionError("admission-policy binding changed")
    output["policy_sha256"] = policy_raw
    return output


def _symbolic_checks() -> dict[str, bool]:
    chi, f, mu, z_x, theta, h, y0, mass, z = sp.symbols(
        "chi f mu z_x theta h y0 mass z", positive=True
    )
    potential = mu**4 * (1 - sp.cos(chi / f))
    mixing = mu**4 * sp.sin(chi / f) * z_x / f
    small_potential = sp.series(potential, chi, 0, 6).removeO()
    small_mixing = sp.series(mixing, chi, 0, 4).removeO()
    mass_substitution = {mu**4: mass**2 * f**2}
    expected_potential = mass**2 * chi**2 / 2 - mass**2 * chi**4 / (24 * f**2)
    expected_mixing = mass**2 * chi * z_x - mass**2 * chi**3 * z_x / (6 * f**2)
    theta_bound = sp.acos(1 - h)
    source_factor = sp.simplify(sp.sin(theta_bound))
    s = sp.symbols("s", positive=True)
    strong_source_exponent = sp.simplify(s - (s - 1) / 2)
    strong_product_exponent = sp.simplify(strong_source_exponent - s / 2)
    low_product_exponent = sp.simplify(s - s / 2)
    return {
        "S02_PERIODIC_AND_SHIFT_IDENTITIES": sp.simplify(sp.cos(theta + 2 * sp.pi) - sp.cos(theta))
        == 0
        and sp.simplify(sp.sin(theta + 2 * sp.pi) - sp.sin(theta)) == 0,
        "S03_SMALL_FIELD_POTENTIAL_RECOVERY": sp.simplify(
            small_potential.subs(mass_substitution) - expected_potential
        )
        == 0,
        "S04_SMALL_FIELD_MIXING_RECOVERY": sp.simplify(
            small_mixing.subs(mass_substitution) - expected_mixing
        )
        == 0,
        "S05_MASS_CURVATURE_AND_RANGE": sp.simplify(
            sp.diff(potential, chi, 2).subs(mass_substitution) - mass**2 * sp.cos(chi / f)
        )
        == 0
        and sp.simplify(
            (sp.sqrt(y0) / (mass * sp.sqrt(z * sp.cos(theta)))) ** 2
            - y0 / (mass**2 * z * sp.cos(theta))
        )
        == 0,
        "S06_EXACT_HEALTH_PHASE_CEILING": sp.simplify(1 - sp.cos(theta_bound) - h) == 0,
        "S07_EXACT_SOURCE_CAPACITY": sp.simplify(source_factor**2 - (2 * h - h**2)) == 0,
        "S08_STRONG_GATE_ASYMPTOTICS": strong_source_exponent == (s + 1) / 2
        and strong_product_exponent == sp.Rational(1, 2),
        "S09_LOW_POWER_BRANCH_CHANGE": low_product_exponent == s / 2,
    }


def _source_capacity(x: float, s: float) -> dict[str, float]:
    gate = x**s
    gate_x = s * x ** (s - 1.0)
    gate_h = s * (2.0 * s - 1.0) * x ** (s - 1.0)
    margin = min(1.0 / gate_x, 1.0 / gate_h)
    h = margin
    if h < 1.0:
        phase = math.acos(1.0 - h)
        source = gate * math.sqrt(2.0 * h - h * h)
        regime = "HEALTH_LIMITED"
    else:
        phase = math.pi / 2.0
        source = gate
        regime = "MASS_CURVATURE_LIMITED"
    range_at_minimum = gate**-0.5
    return {
        "h": h,
        "phase_ceiling": phase,
        "source_capacity": source,
        "range_at_minimum": range_at_minimum,
        "product": source * range_at_minimum,
        "regime": regime,
    }


def _log_slope(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.log(y1 / y0) / math.log(x1 / x0)


def _numeric_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    x_values = [float(value) for value in config["numeric_contract"]["X_values"]]
    slope_records: list[dict[str, Any]] = []
    all_passed = True
    for s in (float(value) for value in config["numeric_contract"]["s_values"]):
        low = _source_capacity(x_values[0], s)
        high = _source_capacity(x_values[-1], s)
        source_slope = _log_slope(
            x_values[0], low["source_capacity"], x_values[-1], high["source_capacity"]
        )
        product_slope = _log_slope(x_values[0], low["product"], x_values[-1], high["product"])
        if s < 1.0:
            target_source, target_product = s, s / 2.0
        else:
            target_source, target_product = (s + 1.0) / 2.0, 0.5
        tolerance = 2.0e-4
        passed = (
            abs(source_slope - target_source) < tolerance
            and abs(product_slope - target_product) < tolerance
        )
        all_passed = all_passed and passed
        slope_records.append(
            {
                "s": s,
                "source_slope": source_slope,
                "source_target": target_source,
                "product_slope": product_slope,
                "product_target": target_product,
                "low_regime": low["regime"],
                "high_regime": high["regime"],
                "passed": passed,
            }
        )
    phase_records: list[dict[str, Any]] = []
    for case in config["numeric_contract"]["designed_phase_cases"]:
        s = float(case["s"])
        x = float(case["X"])
        theta = float(case["theta"])
        gate_x = s * x ** (s - 1.0)
        gate_h = s * (2.0 * s - 1.0) * x ** (s - 1.0)
        potential = 1.0 - math.cos(theta)
        c_value = 1.0 - potential * gate_x
        k_value = 1.0 - potential * gate_h
        mass_curvature = x**s * math.cos(theta)
        expected = case["expected"]
        if expected == "HEALTHY_POSITIVE_MASS":
            passed = c_value > 0.0 and k_value > 0.0 and mass_curvature > 0.0
        elif expected == "K_NEGATIVE":
            passed = k_value < 0.0
        elif expected == "ZERO_MASS_CURVATURE":
            passed = abs(mass_curvature) < 1.0e-14
        else:
            passed = mass_curvature < 0.0
        all_passed = all_passed and passed
        phase_records.append(
            {
                "id": case["id"],
                "C": c_value,
                "K": k_value,
                "mass_curvature": mass_curvature,
                "expected": expected,
                "passed": passed,
            }
        )
    return {
        "slope_records": slope_records,
        "phase_records": phase_records,
        "all_passed": all_passed,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _local_integrity(base)
    predecessor = _validate_predecessor(base, config)
    symbolic = _symbolic_checks()
    numeric = _numeric_evidence(config)
    checks = {
        "S01_CONFIG_POLICY_AND_PREDECESSOR_SEALS": local["config_raw_sha256"]
        == EXPECTED_CONFIG_RAW_SHA256,
        **symbolic,
        "S10_NUMERIC_SLOPES_AND_PHASE_FAILURES": numeric["all_passed"],
        "S11_PRIMARY_SOURCE_AND_CLAIM_BOUNDARY": len(config["primary_literature"]) == 4
        and config["claim_boundary"] == _expected_claims(),
        "S12_ZERO_OBSERVATIONAL_ACCESS": not any(config["access_ledger"].values()),
    }
    if set(checks) != set(config["required_checks"]) or not all(checks.values()):
        raise SplitGatePeriodicCompletionError("periodic-completion checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": ARTIFACT_ID,
        "status": config["status"],
        "decision": DECISION,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "eft_preflight_binding": predecessor,
        "implementation_binding": local,
        "action_contract": config["action_contract"],
        "principal_and_health_contract": config["principal_and_health_contract"],
        "exact_source_capacity": config["exact_source_capacity"],
        "primary_literature": config["primary_literature"],
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
        raise SplitGatePeriodicCompletionError("stored receipt differs from rebuild")
    return stored


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    destination = (base / OUTPUT_PATH).resolve()
    data = (json.dumps(build_receipt(base), indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise SplitGatePeriodicCompletionError("refusing to replace nonidentical receipt")
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
            raise SplitGatePeriodicCompletionError("receipt publication race") from exc
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
                    "p2_tradeoff_recovered": receipt["adjudication"][
                        "p2_range_source_theorem_recovered"
                    ],
                    "large_u_resummation_solved": receipt["adjudication"][
                        "large_u_Z_resummation_solved"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
