"""Target-free asymptotic range/source theorem for the split mass gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_split_gate_range_source_tradeoff_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_split_gate_range_source_tradeoff.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_range_source_tradeoff.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-split-gate-range-source-tradeoff-v1.json")
CONFIG_CANONICAL_SHA256 = "e6390329ae7df07c3ef6b9871040f816db693cb908b5f45afa094d8d92cbabd5"

CONFIG_SCHEMA = "invariant-gravity-matter-lensing-split-gate-range-source-tradeoff-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-split-gate-range-source-tradeoff-receipt-1.0"
DECISION = "PROMISING_SECOND_ANALYTIC_RESULT_REQUIRES_EXPERT_AND_EXHAUSTIVE_NOVELTY_REVIEW"


class SplitGateTradeoffError(RuntimeError):
    """Raised when a theorem, binding, or publication guard fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixed_root(root: Path | None = None) -> Path:
    expected = _repo_root().resolve()
    candidate = expected if root is None else root.resolve()
    _require(candidate == expected, "noncanonical repository root refused")
    return expected


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SplitGateTradeoffError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitGateTradeoffError(message)


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SplitGateTradeoffError(f"missing committed binding: {commit}:{relative}")
    return result.stdout


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'^CONFIG_CANONICAL_SHA256 = "[^"]*"$',
        'CONFIG_CANONICAL_SHA256 = "<SEALED_CONFIG>"',
        text,
        flags=re.MULTILINE,
    )
    return _sha256_bytes(normalized.encode())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _content_sha256(body)


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _fixed_root(root)
    config = _read_json(base / CONFIG_PATH)
    _require(_content_sha256(config) == CONFIG_CANONICAL_SHA256, "tradeoff config changed")
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("status") == "CANDIDATE_ASYMPTOTIC_TRADEOFF_THEOREM_NOT_PREPRINT_READY",
        "config status changed",
    )
    _require(config.get("output_path") == OUTPUT_PATH.as_posix(), "output path changed")
    _require(
        [item.get("id") for item in config.get("bindings", [])]
        == [
            "SPLIT_GATE_ACTION",
            "STATIC_SOURCE_BOUND",
            "UNIVERSAL_CONFORMAL_SOURCE",
            "SOLAR_GW_NECESSARY_CONDITIONS",
        ],
        "binding inventory changed",
    )
    expected_claims = {
        "candidate_original_asymptotic_theorem": True,
        "historical_novelty_established": False,
        "independent_expert_review_passed": False,
        "general_variable_coefficient_bound": False,
        "full_coupled_range_derived": False,
        "full_action_health": False,
        "physical_source_X_scaling_derived": False,
        "on_shell_material_solution": False,
        "observational_support": False,
        "modified_gravity_success": False,
        "publication_ready": False,
    }
    _require(config.get("claim_boundary") == expected_claims, "claim boundary changed")
    _require(all(value == 0 for value in config.get("zero_access", {}).values()), "access changed")
    implementation = config.get("implementation", {})
    _require(implementation.get("module_path") == SOURCE_PATH.as_posix(), "module path changed")
    _require(implementation.get("test_path") == TEST_PATH.as_posix(), "test path changed")
    _require(
        _module_semantic_sha256(base / SOURCE_PATH) == implementation.get("module_semantic_sha256"),
        "module semantics changed",
    )
    _require(
        _sha256_file(base / TEST_PATH) == implementation.get("test_sha256"),
        "test binding changed",
    )
    return config


def validate_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            current = root / relative
            _require(current.is_file(), f"missing binding: {binding['id']} {role}")
            _require(
                _sha256_file(current) == expected,
                f"working-tree binding changed: {binding['id']} {role}",
            )
            _require(
                _sha256_bytes(_git_show(root, binding["commit"], relative)) == expected,
                f"commit binding changed: {binding['id']} {role}",
            )
        receipt = _read_json(root / binding["receipt_path"])
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            f"receipt content changed: {binding['id']}",
        )
        receipts[binding["id"]] = receipt
    return receipts


def symbolic_checks() -> dict[str, bool]:
    x, z, s, mass, y0, a_inf, b_inf = sp.symbols(
        "X z s m_chi Y0 A_infinity B_infinity", positive=True
    )
    gate = z * x**s
    gate_x = sp.diff(gate, x)
    gate_xx = sp.diff(gate_x, x)
    h_gate = sp.simplify(gate_x + 2 * x * gate_xx)
    chi_c = sp.sqrt(2 * a_inf / (mass**2 * gate_x))
    chi_k = sp.sqrt(2 * b_inf / (mass**2 * h_gate))
    ell = sp.sqrt(y0) / (mass * sp.sqrt(gate))
    q_c = sp.simplify(mass**2 * gate * chi_c)
    q_k = sp.simplify(mass**2 * gate * chi_k)

    chi_c_constant = sp.sqrt(2 * a_inf / (mass**2 * s * z))
    chi_k_constant = sp.sqrt(2 * b_inf / (mass**2 * s * (2 * s - 1) * z))
    ell_constant = sp.sqrt(y0) / (mass * sp.sqrt(z))
    q_c_constant = mass * sp.sqrt(2 * a_inf * z / s)
    q_k_constant = mass * sp.sqrt(2 * b_inf * z / (s * (2 * s - 1)))
    product_c_constant = sp.sqrt(2 * y0 * a_inf / s)
    product_k_constant = sp.sqrt(2 * y0 * b_inf / (s * (2 * s - 1)))

    p, beta = sp.symbols("p beta", positive=True)
    shifted = (1 + beta * x**2) ** p
    shifted_slope = sp.simplify(x * sp.diff(shifted, x) / shifted)
    shifted_curvature = sp.simplify(x**2 * sp.diff(shifted, x, 2) / shifted)

    u = beta * x**2
    p2 = (1 + u) ** 2
    p2_x = sp.diff(p2, x)
    p2_h = sp.simplify(p2_x + 2 * x * sp.diff(p2_x, x))
    p2_chi_k = sp.sqrt(2 * b_inf / (mass**2 * p2_h))
    p2_q_k = sp.simplify(mass**2 * p2 * p2_chi_k)

    t, radius = sp.symbols("t radius", positive=True)
    screened = (1 + radius * t) * sp.exp(-radius * t)

    checks = {
        "T02_C_BRANCH_AMPLITUDE_LIMIT": sp.simplify(chi_c * x ** ((s - 1) / 2) - chi_c_constant)
        == 0,
        "T03_K_BRANCH_AMPLITUDE_LIMIT": sp.simplify(chi_k * x ** ((s - 1) / 2) - chi_k_constant)
        == 0,
        "T04_RANGE_LIMIT": sp.simplify(ell * x ** (s / 2) - ell_constant) == 0,
        "T05_C_SOURCE_LIMIT": sp.simplify(q_c / x ** ((s + 1) / 2) - q_c_constant) == 0,
        "T06_K_SOURCE_LIMIT": sp.simplify(q_k / x ** ((s + 1) / 2) - q_k_constant) == 0,
        "T07_C_PRODUCT_LIMIT": sp.simplify(q_c * ell / sp.sqrt(x) - product_c_constant) == 0,
        "T08_K_PRODUCT_LIMIT": sp.simplify(q_k * ell / sp.sqrt(x) - product_k_constant) == 0,
        "T09_CRITICAL_EXPONENT": sp.simplify((s + 1) / 2 - (s / 2 + sp.Rational(1, 2))) == 0,
        "T10_SHIFTED_POWER_MAPPING": sp.simplify(sp.limit(shifted_slope, x, sp.oo) - 2 * p) == 0
        and sp.simplify(sp.limit(shifted_curvature, x, sp.oo) - 2 * p * (2 * p - 1)) == 0,
        "T11_COMMITTED_P2_RECOVERY": sp.limit(p2_q_k / x ** sp.Rational(5, 2), x, sp.oo)
        == mass * beta * sp.sqrt(b_inf / 14)
        and sp.limit(x**3 * p2_chi_k**2, x, sp.oo) == b_inf / (14 * mass**2 * beta**2),
        "T12_FIXED_DISTANCE_SCREENING": sp.limit(screened, t, sp.oo) == 0,
    }
    _require(all(checks.values()), "symbolic theorem check failed")
    return checks


def _exact_shifted_values(
    power: float, x: float, constants: Mapping[str, float]
) -> dict[str, float]:
    a_inf = float(constants["A_infinity"])
    b_inf = float(constants["B_infinity"])
    y0 = float(constants["Y0"])
    mass = float(constants["m_chi"])
    beta = float(constants["z"])
    u = beta * x * x
    gate = (1.0 + u) ** power
    gate_x = 2.0 * power * beta * x * (1.0 + u) ** (power - 1.0)
    gate_xx = 2.0 * power * beta * (1.0 + u) ** (power - 1.0) + 4.0 * power * (
        power - 1.0
    ) * beta**2 * x**2 * (1.0 + u) ** (power - 2.0)
    h_gate = gate_x + 2.0 * x * gate_xx
    _require(gate_x > 0.0 and h_gate > 0.0, "numeric branch not positive")
    chi_c = math.sqrt(2.0 * a_inf / (mass**2 * gate_x))
    chi_k = math.sqrt(2.0 * b_inf / (mass**2 * h_gate))
    chi_max = min(chi_c, chi_k)
    ell = math.sqrt(y0) / (mass * math.sqrt(gate))
    q_max = mass**2 * gate * chi_max
    return {"ell": ell, "chi_max": chi_max, "q_max": q_max, "product": ell * q_max}


def _log_slope(left: float, right: float, x_left: float, x_right: float) -> float:
    return math.log(right / left) / math.log(x_right / x_left)


def numeric_probes(config: Mapping[str, Any]) -> dict[str, Any]:
    probes = config["parameter_probes"]
    x_values = [float(value) for value in probes["X_values"]]
    constants = {key: float(value) for key, value in probes["constants"].items()}
    records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    for power_value in probes["monomial_powers"]:
        power = float(power_value)
        values = [_exact_shifted_values(power, x, constants) for x in x_values]
        slopes = {
            key: _log_slope(values[-2][key], values[-1][key], x_values[-2], x_values[-1])
            for key in ("ell", "chi_max", "q_max", "product")
        }
        targets = {
            "ell": -power,
            "chi_max": 0.5 - power,
            "q_max": power + 0.5,
            "product": 0.5,
        }
        errors = {key: abs(slopes[key] - targets[key]) for key in targets}
        passed = max(errors.values()) < 2.0e-6
        records.append(
            {
                "power": power,
                "effective_s": 2.0 * power,
                "slopes": slopes,
                "targets": targets,
                "max_abs_error": max(errors.values()),
                "passed": passed,
            }
        )

        critical = power + 0.5
        q_ceiling = [item["q_max"] for item in values]
        for offset_value in probes["source_exponent_offsets"]:
            offset = float(offset_value)
            exponent = critical + offset
            ratios = [x**exponent / q for x, q in zip(x_values, q_ceiling, strict=True)]
            if offset < 0.0:
                behavior = "DECREASING_SUBCRITICAL"
                passed_threshold = ratios[-1] < ratios[0]
            elif offset > 0.0:
                behavior = "INCREASING_SUPERCRITICAL"
                passed_threshold = ratios[-1] > ratios[0]
            else:
                behavior = "COEFFICIENT_DEPENDENT_CRITICAL"
                passed_threshold = abs(ratios[-1] / ratios[-2] - 1.0) < 2.0e-6
            threshold_records.append(
                {
                    "power": power,
                    "critical_exponent": critical,
                    "source_exponent": exponent,
                    "offset": offset,
                    "ratios": ratios,
                    "behavior": behavior,
                    "passed": passed_threshold,
                }
            )
    _require(all(item["passed"] for item in records), "numeric exponent convergence failed")
    _require(
        all(item["passed"] for item in threshold_records),
        "numeric threshold classification failed",
    )
    return {
        "exponent_records": records,
        "threshold_records": threshold_records,
        "all_passed": True,
        "designed_supercritical_growth_cases": sum(
            item["offset"] > 0.0 for item in threshold_records
        ),
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _fixed_root(root)
    config = load_config(base)
    bound_receipts = validate_bindings(base, config)
    symbolic = symbolic_checks()
    numeric = numeric_probes(config)
    checks = {
        "T01_EXACT_COMMITTED_BINDINGS": len(bound_receipts) == 4,
        **symbolic,
        "T13_NUMERIC_EXPONENT_CONVERGENCE": all(
            row["passed"] for row in numeric["exponent_records"]
        ),
        "T14_DESIGNED_THRESHOLD_FAILURES": numeric["designed_supercritical_growth_cases"] == 4
        and all(row["passed"] for row in numeric["threshold_records"]),
        "T15_CLAIM_CEILING": config["claim_boundary"]["candidate_original_asymptotic_theorem"]
        is True
        and all(
            config["claim_boundary"][key] is False
            for key in (
                "historical_novelty_established",
                "independent_expert_review_passed",
                "general_variable_coefficient_bound",
                "full_coupled_range_derived",
                "full_action_health",
                "physical_source_X_scaling_derived",
                "on_shell_material_solution",
                "observational_support",
                "modified_gravity_success",
                "publication_ready",
            )
        ),
    }
    _require(set(checks) == set(config["required_checks"]), "required check inventory changed")
    _require(all(checks.values()), "tradeoff adjudication failed")

    power_table = _power_table(config)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "PROMISING_SECOND_ASYMPTOTIC_THEOREM_CANDIDATE_NOT_PREPRINT_READY",
        "decision": DECISION,
        "content_sha256": "",
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "raw_sha256": _sha256_file(base / CONFIG_PATH),
            "canonical_sha256": CONFIG_CANONICAL_SHA256,
        },
        "implementation_binding": {
            "module_path": SOURCE_PATH.as_posix(),
            "module_raw_sha256": _sha256_file(base / SOURCE_PATH),
            "module_semantic_sha256": _module_semantic_sha256(base / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_sha256": _sha256_file(base / TEST_PATH),
        },
        "predecessor_receipt_contents": {
            key: value["content_sha256"] for key, value in sorted(bound_receipts.items())
        },
        "maximal_theorem": config["asymptotic_theorem"],
        "shifted_power_family": config["shifted_power_family"],
        "fifth_force_link": config["fifth_force_link"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "power_table": power_table,
        "numeric_evidence": numeric,
        "literature_positioning": config["primary_literature_positioning"],
        "novelty_search_scope": config["novelty_search_scope"],
        "publication_adjudication": config["publication_adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access": config["zero_access"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "analysis_id",
        "status",
        "decision",
        "content_sha256",
        "config_binding",
        "implementation_binding",
        "predecessor_receipt_contents",
        "maximal_theorem",
        "shifted_power_family",
        "fifth_force_link",
        "checks",
        "checks_passed",
        "power_table",
        "numeric_evidence",
        "literature_positioning",
        "novelty_search_scope",
        "publication_adjudication",
        "claim_boundary",
        "zero_access",
    }
    _require(set(receipt) == expected_keys, "receipt keys changed")
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("analysis_id") == config["analysis_id"], "receipt identity changed")
    _require(
        receipt.get("status") == "PROMISING_SECOND_ASYMPTOTIC_THEOREM_CANDIDATE_NOT_PREPRINT_READY",
        "receipt status changed",
    )
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(receipt.get("content_sha256") == _self_hash(receipt), "receipt self-hash changed")
    _require(receipt.get("checks_passed") == 15, "receipt check count changed")
    _require(set(receipt.get("checks", {})) == set(config["required_checks"]), "check keys changed")
    _require(all(receipt.get("checks", {}).values()), "receipt contains failed checks")
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "receipt claims changed")
    _require(receipt.get("zero_access") == config["zero_access"], "receipt access changed")
    _require(
        receipt.get("maximal_theorem") == config["asymptotic_theorem"],
        "receipt theorem changed",
    )
    _require(
        receipt.get("shifted_power_family") == config["shifted_power_family"],
        "shifted family changed",
    )
    _require(receipt.get("fifth_force_link") == config["fifth_force_link"], "force link changed")
    _require(receipt.get("power_table") == _power_table(config), "power table changed")
    _require(receipt.get("numeric_evidence") == numeric_probes(config), "numeric evidence changed")
    _require(
        receipt.get("literature_positioning") == config["primary_literature_positioning"],
        "literature scope changed",
    )
    _require(
        receipt.get("novelty_search_scope") == config["novelty_search_scope"],
        "novelty scope changed",
    )
    _require(
        receipt.get("publication_adjudication") == config["publication_adjudication"],
        "publication adjudication changed",
    )
    _require(
        receipt.get("predecessor_receipt_contents")
        == {binding["id"]: binding["receipt_content_sha256"] for binding in config["bindings"]},
        "predecessor content ledger changed",
    )
    _require(
        receipt.get("config_binding", {}).get("canonical_sha256") == CONFIG_CANONICAL_SHA256,
        "receipt config binding changed",
    )
    _require(
        receipt.get("implementation_binding", {}).get("module_semantic_sha256")
        == config["implementation"]["module_semantic_sha256"]
        and receipt.get("implementation_binding", {}).get("test_sha256")
        == config["implementation"]["test_sha256"],
        "receipt implementation binding changed",
    )


def _power_table(config: Mapping[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "power_p": float(power),
            "regular_variation_s": 2.0 * float(power),
            "range_exponent": -float(power),
            "amplitude_exponent": 0.5 - float(power),
            "source_ceiling_exponent": float(power) + 0.5,
            "critical_source_exponent": float(power) + 0.5,
        }
        for power in config["parameter_probes"]["monomial_powers"]
    ]


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise SplitGateTradeoffError(f"refusing to overwrite nonidentical receipt: {path}")
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
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise SplitGateTradeoffError(f"receipt publication race: {path}") from None
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path | None = None) -> str:
    base = _fixed_root(root)
    receipt = build_receipt(base)
    payload = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    return _atomic_no_replace(base / OUTPUT_PATH, payload)


def check_receipt() -> str:
    base = _repo_root()
    expected = build_receipt(base)
    path = (base / OUTPUT_PATH).resolve()
    _require(path == (base / OUTPUT_PATH).resolve(), "receipt path changed")
    _require(path.is_file(), "receipt missing")
    stored = _read_json(path)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    validate_receipt(stored, load_config(base))
    return "VALID"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "checks_passed": receipt["checks_passed"],
                    "publication_ready": receipt["claim_boundary"]["publication_ready"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
