"""No-data conditional no-go analysis for a growing two-scalar kinetic gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_conditional_no_go_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_kinetic_gate_conditional_no_go.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_kinetic_gate_conditional_no_go.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-conditional-no-go-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-conditional-no-go-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-conditional-no-go-receipt-1.0"
DECISION = (
    "CONDITIONAL_NO_GO_FOR_GLOBALLY_NONNEGATIVE_TIMELIKE_MIXING_IN_SMOOTH_"
    "GROWING_KINETIC_GATES_REMEDIES_PREREGISTERED_NOT_VALIDATED"
)
EXPECTED_CONFIG_FILE_SHA256 = "333f2ef850138c7e4fb6d509d0fcf16a769620c79cf7a8eb8cef002337812608"
EXPECTED_CONFIG_CONTENT_SHA256 = "652f7bf37a23720d26d23ca1e3dab86b37ef3912b1694e0064f92590133b6e19"

SYMBOLIC_CHECK_IDS = (
    "S01_DIRECT_CHAIN_RULE_ZX",
    "S02_DIRECT_CHAIN_RULE_ZXX",
    "S03_MIXING_W_IDENTITY",
    "S04_LOG_SLOPE_IDENTITY",
    "S05_COMPARISON_SOLUTION_ODE",
    "S06_COMPARISON_INITIAL_VALUE",
    "S07_FINITE_BLOWUP_DENOMINATOR",
    "S08_EQUALITY_GATE_LOG_SLOPE",
    "S09_EQUALITY_GATE_ZERO_MIXING",
    "S10_SHIFTED_POWER_MIXING",
    "S11_SHIFTED_POWER_THRESHOLD_FACTOR",
    "S12_P2_PREDECESSOR_REGRESSION",
    "S13_CONSTANT_GATE_REGRESSION",
    "S14_FULL_DETERMINANT_DECOMPOSITION",
    "S15_XCHI_STABILIZATION_BOUND",
)
REMEDY_IDS = (
    "SOURCE_ONLY_GATE",
    "SPLIT_KINETIC_AND_MASS_GATES",
    "DYNAMIC_XCHI_ENVELOPE",
    "ORTHOGONAL_FIELD_SPACE_METRIC",
    "AUXILIARY_NONPROPAGATING_CHANNEL",
)


class KineticGateConditionalNoGoError(RuntimeError):
    """Raised when a binding, derivation, or publication gate fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KineticGateConditionalNoGoError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KineticGateConditionalNoGoError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    return {
        "check_id": check_id,
        "passed": simplified == 0,
        "residual": "0" if simplified == 0 else sp.sstr(simplified),
        "statement": statement,
    }


def run_symbolic_suite() -> dict[str, Any]:
    x, beta, u = sp.symbols("X beta u", positive=True)
    z_function = sp.Function("Z")
    direct_z = z_function(beta * x**2)
    direct_zx = sp.diff(direct_z, x)
    direct_zxx = sp.diff(direct_zx, x)
    expected_zx = 2 * beta * x * sp.Subs(sp.diff(z_function(u), u), u, beta * x**2)
    expected_zxx = 2 * beta * sp.Subs(
        sp.diff(z_function(u), u), u, beta * x**2
    ) + 4 * beta**2 * x**2 * sp.Subs(sp.diff(z_function(u), u, 2), u, beta * x**2)

    z, w_u, w_uu, z_u, z_uu = sp.symbols("Z w_u w_uu Z_u Z_uu", real=True)
    zx = 2 * u * z_u / x
    zxx = 2 * u * (z_u + 2 * u * z_uu) / x**2
    mixing = z * (zx + 2 * x * zxx) - 4 * x * zx**2
    mixing_w = (2 * u * z**2 / x) * (3 * w_u + 4 * u * w_uu - 4 * u * w_u**2)
    mixing_substitution = sp.expand(mixing.subs({z_u: z * w_u, z_uu: z * (w_uu + w_u**2)}))

    w = sp.Function("w")
    q = u * sp.diff(w(u), u)
    q_dot = u * sp.diff(q, u)
    log_slope_residual = sp.expand(
        u * (3 * sp.diff(w(u), u) + 4 * u * sp.diff(w(u), u, 2) - 4 * u * sp.diff(w(u), u) ** 2)
        - (4 * q_dot - q - 4 * q**2)
    )

    t, t0, q0 = sp.symbols("t t0 q0", positive=True)
    growth = sp.exp((t - t0) / 4)
    denominator = 1 + 4 * q0 * (1 - growth)
    comparison = q0 * growth / denominator
    blowup_time = t0 + 4 * sp.log(1 + 1 / (4 * q0))

    u0, z0 = sp.symbols("u0 Z0", positive=True)
    radius = (u / u0) ** sp.Rational(1, 4)
    equality_denominator = 1 + 4 * q0 * (1 - radius)
    equality_gate = z0 / equality_denominator
    equality_q = sp.simplify(u * sp.diff(sp.log(equality_gate), u))
    equality_expected_q = q0 * radius / equality_denominator
    equality_criterion = sp.simplify(
        4 * u * sp.diff(equality_q, u) - equality_q - 4 * equality_q**2
    )

    p = sp.symbols("p", positive=True)
    shifted_gate = (1 + u) ** p
    shifted_w = sp.log(shifted_gate)
    shifted_bracket = sp.simplify(
        3 * sp.diff(shifted_w, u)
        + 4 * u * sp.diff(shifted_w, u, 2)
        - 4 * u * sp.diff(shifted_w, u) ** 2
    )
    shifted_expected = p * (3 - (1 + 4 * p) * u) / (1 + u) ** 2
    shifted_mixing = sp.simplify(2 * u * shifted_gate**2 * shifted_bracket / x)
    p2_expected = 12 * u * (1 + u) ** 2 * (1 - 3 * u) / x

    a_phi, x_chi, m = sp.symbols("A_phi X_chi M", real=True)
    determinant = z * a_phi + x_chi * m
    determinant_boundary = sp.simplify(determinant.subs(x_chi, z * a_phi / (-m)))

    checks = [
        _check(
            "S01_DIRECT_CHAIN_RULE_ZX",
            direct_zx - expected_zx,
            "Direct differentiation gives Z_X=2*beta*X*Z_u.",
        ),
        _check(
            "S02_DIRECT_CHAIN_RULE_ZXX",
            direct_zxx - expected_zxx,
            "A second direct derivative gives the exact Z_XX chain rule.",
        ),
        _check(
            "S03_MIXING_W_IDENTITY",
            mixing_substitution - mixing_w,
            "The mixing term equals the frozen logarithmic-gate identity.",
        ),
        _check(
            "S04_LOG_SLOPE_IDENTITY",
            log_slope_residual,
            "q=u*w_u converts the mixing bracket into 4*dq/dt-q-4*q^2.",
        ),
        _check(
            "S05_COMPARISON_SOLUTION_ODE",
            sp.diff(comparison, t) - comparison / 4 - comparison**2,
            "The comparison curve solves y'=y/4+y^2.",
        ),
        _check(
            "S06_COMPARISON_INITIAL_VALUE",
            comparison.subs(t, t0) - q0,
            "The comparison curve has y(t0)=q0.",
        ),
        _check(
            "S07_FINITE_BLOWUP_DENOMINATOR",
            denominator.subs(t, blowup_time),
            "Its denominator vanishes after the frozen finite log-u interval.",
        ),
        _check(
            "S08_EQUALITY_GATE_LOG_SLOPE",
            equality_q - equality_expected_q,
            "The integrated equality gate has the comparison log slope.",
        ),
        _check(
            "S09_EQUALITY_GATE_ZERO_MIXING",
            equality_criterion,
            "The equality gate has M=0 before its finite boundary.",
        ),
        _check(
            "S10_SHIFTED_POWER_MIXING",
            shifted_bracket - shifted_expected,
            "The shifted-power mixing bracket is exact.",
        ),
        _check(
            "S11_SHIFTED_POWER_THRESHOLD_FACTOR",
            sp.factor((1 + u) ** 2 * shifted_bracket / p) - (3 - (1 + 4 * p) * u),
            "The sign threshold is u=3/(1+4p).",
        ),
        _check(
            "S12_P2_PREDECESSOR_REGRESSION",
            shifted_mixing.subs(p, 2) - p2_expected,
            "p=2 recovers the predecessor u=1/3 factorization.",
        ),
        _check(
            "S13_CONSTANT_GATE_REGRESSION", shifted_mixing.subs(p, 0), "A constant gate has q=M=0."
        ),
        _check(
            "S14_FULL_DETERMINANT_DECOMPOSITION",
            determinant - (z * a_phi + x_chi * m),
            "The full slice is the P sector plus X_chi*M.",
        ),
        _check(
            "S15_XCHI_STABILIZATION_BOUND",
            determinant_boundary,
            "The frozen X_chi bound is the determinant-zero boundary when M<0.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "derivation_routes": [
            "direct X differentiation",
            "independent logarithmic-slope substitution",
            "Riccati comparison integration",
        ],
        "checks": checks,
        "all_passed": True,
        "expressions": {
            "mixing_w": "(2*u*Z^2/X)*(3*w_u+4*u*w_uu-4*u*w_u^2)",
            "mixing_q": "(2*Z^2/X)*(4*dq/dt-q-4*q^2)",
            "comparison_solution": "q0*exp((t-t0)/4)/[1+4*q0*(1-exp((t-t0)/4))]",
            "comparison_blowup_delta_t": "4*ln(1+1/(4*q0))",
            "shifted_power_bracket": "p*[3-(1+4*p)*u]/(1+u)^2",
        },
    }


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    records: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["numeric_cases"]:
        u = float(case["u"])
        if case["family"] == "shifted_power":
            p = float(case["p"])
            x = math.sqrt(u)
            z = (1.0 + u) ** p
            value = (2.0 * u * z**2 / x) * p * (3.0 - (1.0 + 4.0 * p) * u) / (1.0 + u) ** 2
        else:
            q0 = float(case["q0"])
            u0 = float(case["u0"])
            radius = (u / u0) ** 0.25
            q = q0 * radius / (1.0 + 4.0 * q0 * (1.0 - radius))
            dq_dt = q / 4.0 + q**2
            value = 4.0 * dq_dt - q - 4.0 * q**2
        sign = "zero" if abs(value) <= tolerance else "positive" if value > 0 else "negative"
        records.append(
            {
                "case_id": case["case_id"],
                "value": format(value, ".17g"),
                "sign": sign,
                "passed": sign == case["expected_sign"],
            }
        )
    _require(all(item["passed"] for item in records), "numeric suite failed")
    return {"tolerance": format(tolerance, ".17g"), "cases": records, "all_passed": True}


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "scope_and_definitions",
            "analytic_contract",
            "family_and_counterexample_contract",
            "full_determinant_caveat",
            "remedy_preregistration",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-kinetic-gate-conditional-no-go-v1",
        "analysis identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    bindings = config["predecessor_bindings"]
    _require(
        tuple(item["binding_id"] for item in bindings)
        == ("theory_preflight", "bounded_symbolic_derivation", "external_metric_principal_symbol"),
        "predecessor inventory changed",
    )
    binding_keys = {
        "binding_id",
        "git_commit",
        "config_path",
        "config_file_sha256",
        "module_path",
        "module_file_sha256",
        "test_path",
        "test_file_sha256",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "receipt_schema_version",
        "receipt_decision",
    }
    for item in bindings:
        _strict(item, binding_keys, f"predecessor {item.get('binding_id')}")
    scope = config["scope_and_definitions"]
    _require(
        scope["physical_branch"]
        == "aligned timelike local jets with X=X_phi>0, X_chi>=0, and u=beta*X^2>0",
        "timelike sign scope changed",
    )
    _require(
        scope["active_mixing_scope"].endswith("X_chi=0 removes this contribution."),
        "active-gradient caveat changed",
    )
    analytic = config["analytic_contract"]
    _require(
        analytic["timelike_sign_criterion"] == "For X>0 and Z>0, M>=0 iff 4*dq/dt>=q+4*q^2.",
        "sign criterion changed",
    )
    _require(analytic["comparison_blowup_time"] == "T-t0=4*ln(1+1/(4*q0))", "blowup time changed")
    family = config["family_and_counterexample_contract"]
    _require(
        family["shifted_power_threshold"] == "M>=0 iff 0<u<=3/(1+4*p) on X>0",
        "family threshold changed",
    )
    _require(len(family["bounded_domain_counterexamples"]) == 3, "bounded examples changed")
    remedies = config["remedy_preregistration"]
    _require(
        tuple(item["remedy_id"] for item in remedies) == REMEDY_IDS, "remedy inventory changed"
    )
    _require(
        all(item["healthy_or_working_claim"] is False for item in remedies),
        "remedy health overstated",
    )
    _require(
        tuple(config["machine_check_contract"]["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic contract changed",
    )
    _require(
        len(config["machine_check_contract"]["numeric_cases"]) == 5, "numeric inventory changed"
    )
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["full_determinant_no_go"] is False
        and adjudication["full_H3"] is False
        and adjudication["full_H4"] is False,
        "full no-go or health gate unlocked",
    )
    claims = config["claim_boundary"]
    _require(
        claims["conditional_external_metric_timelike_mixing_theorem_established"] is True,
        "conditional theorem disabled",
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key != "conditional_external_metric_timelike_mixing_theorem_established"
        ),
        "claim boundary overstated",
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "access state changed",
    )


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    _require(path.is_file(), "config missing")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_predecessors(root: Path, config: Mapping[str, Any]) -> None:
    for binding in config["predecessor_bindings"]:
        for path_key, hash_key in (
            ("config_path", "config_file_sha256"),
            ("module_path", "module_file_sha256"),
            ("test_path", "test_file_sha256"),
            ("receipt_path", "receipt_file_sha256"),
        ):
            path = root / binding[path_key]
            _require(path.is_file(), f"predecessor missing: {binding['binding_id']} {path_key}")
            _require(
                _file_sha(path) == binding[hash_key],
                f"predecessor changed: {binding['binding_id']} {path_key}",
            )
        receipt = _read_json(root / binding["receipt_path"])
        _require(
            receipt.get("schema_version") == binding["receipt_schema_version"],
            f"predecessor schema changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            f"predecessor content changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("decision") == binding["receipt_decision"],
            f"predecessor decision changed: {binding['binding_id']}",
        )


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _validate_predecessors(root, config)
    _require((root / SOURCE_PATH).is_file(), "implementation missing")
    _require((root / TEST_PATH).is_file(), "test missing")
    symbolic = run_symbolic_suite()
    numeric = run_numeric_suite(config)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "conditional_timelike_kinetic_gate_no_go_machine_verified_scope_restricted",
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "source_path": SOURCE_PATH.as_posix(),
            "source_file_sha256": _file_sha(root / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(root / TEST_PATH),
        },
        "predecessor_bindings": config["predecessor_bindings"],
        "scope_and_definitions": config["scope_and_definitions"],
        "analytic_contract": config["analytic_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "family_and_counterexample_contract": config["family_and_counterexample_contract"],
        "full_determinant_caveat": config["full_determinant_caveat"],
        "remedy_preregistration": config["remedy_preregistration"],
        "adjudication": config["adjudication"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "bounded_domain_counterexamples": len(
                config["family_and_counterexample_contract"]["bounded_domain_counterexamples"]
            ),
            "remedies_preregistered": len(config["remedy_preregistration"]),
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The theorem concerns the sign of the mixing contribution on aligned timelike external-metric local jets with an active chi gradient; it is not an unconditional action no-go.",
            "A positive P-sector contribution or a sufficiently suppressed on-shell X_chi can keep the frozen local determinant nonnegative on a restricted domain.",
            "The sign equivalence is for X>0; on X<0 the prefactor reverses and this theorem is not claimed.",
            "Bounded-domain gates can keep M nonnegative before a threshold or finite comparison boundary.",
            "The preregistered alternative architectures have not been derived, screened, simulated, or shown healthy.",
            "Metric constraints, on-shell backgrounds, global hyperbolicity, EFT cutoff, lensing, and observational support remain unresolved.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(
        receipt,
        {
            "schema_version",
            "analysis_id",
            "status",
            "decision",
            "config_binding",
            "implementation_binding",
            "predecessor_bindings",
            "scope_and_definitions",
            "analytic_contract",
            "symbolic_suite",
            "numeric_suite",
            "family_and_counterexample_contract",
            "full_determinant_caveat",
            "remedy_preregistration",
            "adjudication",
            "counts",
            "claim_boundary",
            "zero_access_and_compute",
            "limitations",
            "content_sha256",
        },
        "receipt",
    )
    body = dict(receipt)
    digest = body.pop("content_sha256")
    _require(digest == _sha(body), "receipt content hash changed")
    _require(
        receipt["schema_version"] == RECEIPT_SCHEMA and receipt["decision"] == DECISION,
        "receipt identity changed",
    )
    _require(
        receipt["config_binding"]["content_sha256"] == _sha(config),
        "receipt config binding changed",
    )
    _require(
        receipt["predecessor_bindings"] == config["predecessor_bindings"],
        "receipt predecessors changed",
    )
    _require(
        receipt["analytic_contract"] == config["analytic_contract"], "analytic contract changed"
    )
    _require(
        receipt["remedy_preregistration"] == config["remedy_preregistration"], "remedies changed"
    )
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claim boundary changed",
    )
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 15,
        "symbolic count changed",
    )
    _require(
        counts["numeric_cases"] == counts["numeric_cases_passed"] == 5, "numeric count changed"
    )
    _require(
        counts["bounded_domain_counterexamples"] == 3 and counts["remedies_preregistered"] == 5,
        "scope count changed",
    )
    _require(
        all(
            counts[key] == 0
            for key in (
                "observational_files_opened",
                "network_calls",
                "model_or_paid_calls",
                "gpu_calls",
            )
        ),
        "receipt access changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise KineticGateConditionalNoGoError(f"refusing to overwrite different receipt: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise KineticGateConditionalNoGoError(
                f"concurrent creator won; output preserved: {path}"
            ) from exc
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> tuple[dict[str, Any], str]:
    root = root.resolve()
    receipt = build_receipt(root)
    return receipt, _atomic_no_replace(root / OUTPUT_PATH, _canonical_bytes(receipt))


def check_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    expected = build_receipt(root)
    stored = _read_json(root / OUTPUT_PATH)
    validate_receipt(stored, config)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    return stored


def _summary(receipt: Mapping[str, Any], publication: str | None = None) -> dict[str, Any]:
    result = {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "content_sha256": receipt["content_sha256"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
        "conditional_theorem": receipt["claim_boundary"][
            "conditional_external_metric_timelike_mixing_theorem_established"
        ],
        "unconditional_action_no_go": receipt["claim_boundary"][
            "unconditional_action_no_go_established"
        ],
    }
    if publication is not None:
        result["publication"] = publication
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        if args.action == "write":
            receipt, publication = write_receipt(args.root)
            result = _summary(receipt, publication)
        else:
            result = _summary(check_receipt(args.root))
        print(json.dumps(result, sort_keys=True))
        return 0
    except KineticGateConditionalNoGoError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
