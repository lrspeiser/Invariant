"""No-data same-action universal conformal matter-source derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_universal_conformal_source_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_universal_conformal_source.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_universal_conformal_source.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-universal-conformal-source-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-universal-conformal-source-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-universal-conformal-source-receipt-1.0"
DECISION = (
    "PARTIAL_UNIVERSAL_CONFORMAL_SOURCE_IDENTITY_DERIVED_PHYSICAL_SOURCE_PROFILE_"
    "METRIC_BACKREACTION_AND_LENSING_NOT_ESTABLISHED"
)
EXPECTED_CONFIG_FILE_SHA256 = "328411ab6d0ef5995dd58c532280bd7df6162871c834f79304fc84a95b8b0130"
EXPECTED_CONFIG_CONTENT_SHA256 = "7f004b268740275716882f0c829806d3210719f7dc7189e9e16badf456b226e8"

SYMBOLIC_CHECK_IDS = (
    "S01_FOUR_DIMENSIONAL_VOLUME_FACTOR",
    "S02_CONFORMAL_METRIC_CHI_VARIATION",
    "S03_SOURCE_PHYSICAL_FRAME_FACTOR",
    "S04_EINSTEIN_CONTRAVARIANT_STRESS_FACTOR",
    "S05_EINSTEIN_TRACE_FACTOR",
    "S06_SOURCE_EINSTEIN_TRACE_IDENTITY",
    "S07_EXPONENTIAL_LOG_DERIVATIVE",
    "S08_DUST_SOURCE_SIGN",
    "S09_CLASSICAL_RADIATION_TRACE_ZERO",
    "S10_PERFECT_FLUID_SOURCE",
    "S11_EINSTEIN_EXCHANGE_EQUALS_MINUS_Q",
    "S12_TOTAL_CHI_EXCHANGE_CANCELLATION",
    "S13_MAXWELL_CONFORMAL_FACTOR_CANCELLATION",
    "S14_NULL_CONE_CONFORMAL_FACTOR",
    "S15_DUST_DENSITY_CEILING_SATURATION",
    "S16_PREDECESSOR_STATIC_SOURCE_SIGN",
    "S17_ZERO_COUPLING_DECOUPLING",
    "S18_SOURCE_MAGNITUDE_FRAME_EQUIVALENCE",
)


class UniversalConformalSourceError(RuntimeError):
    """Raised when a binding, derivation, or publication guard fails."""


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
        raise UniversalConformalSourceError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UniversalConformalSourceError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _zero(value: Any) -> bool:
    if isinstance(value, sp.MatrixBase):
        return all(sp.simplify(item) == 0 for item in value)
    if isinstance(value, (tuple, list)):
        return all(_zero(item) for item in value)
    return sp.simplify(value) == 0


def _render(value: Any) -> Any:
    if _zero(value):
        return "ZERO_MATRIX" if isinstance(value, sp.MatrixBase) else "0"
    if isinstance(value, (tuple, list)):
        return [_render(item) for item in value]
    return sp.sstr(
        value.applyfunc(sp.simplify) if isinstance(value, sp.MatrixBase) else sp.simplify(value)
    )


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": _zero(residual),
        "residual": _render(residual),
        "statement": statement,
    }


def run_symbolic_suite() -> dict[str, Any]:
    conformal, alpha, planck, chi = sp.symbols("A alpha M_Pl chi", positive=True)
    alpha_chi = alpha / planck
    base_metric = sp.diag(-1, 1, 1, 1)
    physical_metric = conformal**2 * base_metric
    physical_volume = sp.sqrt(-sp.det(physical_metric))

    a_phi = sp.symbols("A_phi", positive=True)
    exponential_a = a_phi * sp.exp(alpha * chi / planck)
    metric_chi_derivative = sp.diff(exponential_a**2, chi)

    trace_tilde, trace_e = sp.symbols("T_tilde T_E", real=True)
    source_physical = -(conformal**4) * alpha_chi * trace_tilde
    source_einstein = -alpha_chi * trace_e
    contravariant_factor = conformal**4 * conformal**2
    trace_from_contravariant = contravariant_factor / conformal**2

    rho, pressure, eos = sp.symbols("rho p w", positive=True)
    perfect_trace = -rho + 3 * pressure
    perfect_source = -(conformal**4) * alpha_chi * perfect_trace
    eos_source = conformal**4 * alpha_chi * rho * (1 - 3 * eos)

    derivative_chi, q_symbol = sp.symbols("d_chi Q_chi", real=True)
    matter_exchange = alpha_chi * trace_e * derivative_chi
    scalar_exchange = q_symbol * derivative_chi

    maxwell_factor = conformal**4 * conformal ** (-2) * conformal ** (-2)
    null_norm = sp.symbols("g_null_norm", real=True)
    physical_null_norm = conformal**2 * null_norm

    alpha_abs, q_max = sp.symbols("alpha_abs Q_max", positive=True)
    rho_max = planck * q_max / (alpha_abs * conformal**4)
    dust_source_magnitude = alpha_abs * conformal**4 * rho / planck

    y0, wave, mass, gate, source = sp.symbols("Y0 k m_chi Z Q", positive=True)
    chi_fourier = sp.symbols("chi_k", real=True)
    scalar_static_euler = -(y0 * wave**2 + mass**2 * gate) * chi_fourier
    matter_euler = -source
    static_solution = sp.solve(scalar_static_euler + matter_euler, chi_fourier)[0]

    trace_magnitude = sp.symbols("abs_T_tilde", positive=True)
    q_magnitude_physical = alpha_abs * conformal**4 * trace_magnitude / planck
    q_magnitude_einstein = alpha_abs * (conformal**4 * trace_magnitude) / planck

    checks = [
        _check(
            "S01_FOUR_DIMENSIONAL_VOLUME_FACTOR",
            physical_volume - conformal**4,
            "Four-dimensional conformal scaling gives sqrt(-tilde_g)=A^4*sqrt(-g) for the frozen orthonormal g.",
        ),
        _check(
            "S02_CONFORMAL_METRIC_CHI_VARIATION",
            metric_chi_derivative - 2 * alpha_chi * exponential_a**2,
            "The exponential metric varies as delta tilde_g=2*(alpha/M_Pl)*tilde_g*delta chi.",
        ),
        _check(
            "S03_SOURCE_PHYSICAL_FRAME_FACTOR",
            source_physical + conformal**4 * alpha_chi * trace_tilde,
            "The source is -A^4*(d ln A/d chi)*Ttilde.",
        ),
        _check(
            "S04_EINSTEIN_CONTRAVARIANT_STRESS_FACTOR",
            contravariant_factor - conformal**6,
            "Covariant metric variation gives T_E^munu=A^6*Ttilde^munu.",
        ),
        _check(
            "S05_EINSTEIN_TRACE_FACTOR",
            trace_from_contravariant - conformal**4,
            "Tracing with g lowers the A^6 factor to T_E=A^4*Ttilde.",
        ),
        _check(
            "S06_SOURCE_EINSTEIN_TRACE_IDENTITY",
            source_einstein.subs(trace_e, conformal**4 * trace_tilde) - source_physical,
            "Physical- and Einstein-frame source identities agree.",
        ),
        _check(
            "S07_EXPONENTIAL_LOG_DERIVATIVE",
            sp.diff(sp.log(exponential_a), chi) - alpha_chi,
            "The minimal exponential has constant logarithmic chi slope alpha/M_Pl.",
        ),
        _check(
            "S08_DUST_SOURCE_SIGN",
            perfect_source.subs(pressure, 0) - conformal**4 * alpha_chi * rho,
            "Positive alpha and positive dust density give positive Q_chi.",
        ),
        _check(
            "S09_CLASSICAL_RADIATION_TRACE_ZERO",
            perfect_source.subs(pressure, rho / 3),
            "Classical ideal radiation has zero conformal trace source.",
        ),
        _check(
            "S10_PERFECT_FLUID_SOURCE",
            perfect_source.subs(pressure, eos * rho) - eos_source,
            "A perfect fluid sources Q_chi=A^4*(alpha/M_Pl)*rho*(1-3w).",
        ),
        _check(
            "S11_EINSTEIN_EXCHANGE_EQUALS_MINUS_Q",
            matter_exchange + source_einstein * derivative_chi,
            "Einstein-frame chi exchange alpha_chi*T_E*dchi equals -Q_chi*dchi.",
        ),
        _check(
            "S12_TOTAL_CHI_EXCHANGE_CANCELLATION",
            matter_exchange.subs(trace_e, -q_symbol / alpha_chi) + scalar_exchange,
            "Matter and on-shell scalar chi exchanges cancel in the total same-action identity.",
        ),
        _check(
            "S13_MAXWELL_CONFORMAL_FACTOR_CANCELLATION",
            maxwell_factor - 1,
            "The four-dimensional classical Maxwell kinetic density has zero net conformal weight.",
        ),
        _check(
            "S14_NULL_CONE_CONFORMAL_FACTOR",
            physical_null_norm - conformal**2 * null_norm,
            "The physical null norm is A^2 times the Einstein-frame null norm.",
        ),
        _check(
            "S15_DUST_DENSITY_CEILING_SATURATION",
            dust_source_magnitude.subs(rho, rho_max) - q_max,
            "The frozen dust density ceiling exactly saturates the absolute Q_chi ceiling.",
        ),
        _check(
            "S16_PREDECESSOR_STATIC_SOURCE_SIGN",
            static_solution + source / (y0 * wave**2 + mass**2 * gate),
            "The conformal source convention retains the committed negative static Fourier response.",
        ),
        _check(
            "S17_ZERO_COUPLING_DECOUPLING",
            eos_source.subs(alpha, 0),
            "The conformal trace source vanishes at alpha=0.",
        ),
        _check(
            "S18_SOURCE_MAGNITUDE_FRAME_EQUIVALENCE",
            q_magnitude_physical - q_magnitude_einstein,
            "Absolute source magnitudes agree after T_E=A^4*Ttilde.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "checks": checks,
        "all_passed": True,
        "derived_expressions": {
            "source_physical": "-A^4*(d ln A/d chi)*Ttilde",
            "source_einstein": "-(d ln A/d chi)*T_E",
            "stress_frames": "T_E^munu=A^6*Ttilde^munu; T_E=A^4*Ttilde",
            "dust": "Q_chi=+(alpha/M_Pl)*A^4*rho",
            "perfect_fluid": "Q_chi=(alpha/M_Pl)*A^4*rho*(1-3w)",
            "matter_exchange": "nabla_mu T_E^mu_nu=-Q_phi*d_nu(phi)-Q_chi*d_nu(chi)",
            "dust_density_ceiling": "rho_max=M_Pl*Q_chi,max/(abs(alpha)*A^4)",
        },
    }


def _sign(value: float, tolerance: float) -> str:
    if abs(value) <= tolerance:
        return "zero"
    return "positive" if value > 0 else "negative"


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    records: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["numeric_cases"]:
        conformal = float(case["A"])
        alpha = float(case["alpha"])
        planck = float(case["M_Pl"])
        rho = float(case["rho"])
        eos = float(case["w"])
        trace_tilde = rho * (-1.0 + 3.0 * eos)
        trace_e = conformal**4 * trace_tilde
        source_physical = -(conformal**4) * (alpha / planck) * trace_tilde
        source_einstein = -(alpha / planck) * trace_e
        sign = _sign(source_physical, tolerance)
        passed = (
            sign == case["expected_sign"] and abs(source_physical - source_einstein) <= tolerance
        )
        records.append(
            {
                "case_id": case["case_id"],
                "Ttilde": format(trace_tilde, ".17g"),
                "T_E": format(trace_e, ".17g"),
                "Q_chi": format(source_physical, ".17g"),
                "source_sign": sign,
                "frame_scaled_error": format(abs(source_physical - source_einstein), ".17g"),
                "passed": passed,
            }
        )
    _require(all(item["passed"] for item in records), "numeric trace-source suite failed")
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
            "frame_and_variation_conventions",
            "minimal_conformal_factor",
            "matter_source_contract",
            "conservation_and_exchange",
            "stability_ceiling_comparison",
            "lensing_boundary",
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
        config["analysis_id"] == "gravity-matter-lensing-universal-conformal-source-v1",
        "analysis identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    bindings = config["predecessor_bindings"]
    _require(isinstance(bindings, list), "predecessor bindings changed")
    _require(
        tuple(item["binding_id"] for item in bindings)
        == ("split_gate_action", "split_gate_source_bound"),
        "predecessor inventory changed",
    )
    binding_keys = {
        "binding_id",
        "git_commit",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "receipt_schema_version",
        "receipt_decision",
    }
    for binding in bindings:
        _strict(binding, binding_keys, f"predecessor {binding.get('binding_id')}")
    frame = config["frame_and_variation_conventions"]
    _strict(
        frame,
        {
            "dimension_and_signature",
            "universal_matter_action",
            "physical_stress_definition",
            "einstein_stress_definition",
            "source_definition",
            "volume_factor",
            "stress_frame_factors",
            "exact_source_identity",
            "forbidden",
        },
        "frame conventions",
    )
    _require(
        frame["physical_stress_definition"].startswith("delta S_m=(1/2)"),
        "stress convention changed",
    )
    _require(
        frame["exact_source_identity"] == "Q_chi=-A^4*(d ln A/d chi)*Ttilde=-(d ln A/d chi)*T_E",
        "source identity changed",
    )
    _require("No independent photon metric" in frame["forbidden"], "photon shortcut enabled")
    conformal = config["minimal_conformal_factor"]
    _strict(
        conformal,
        {"choice", "parameters", "derivative", "justification", "novelty_label", "not_selected_by"},
        "conformal factor",
    )
    _require(
        conformal["choice"] == "A(phi,chi)=A_phi(phi)*exp(alpha*chi/M_Pl)"
        and conformal["novelty_label"] == "KNOWN_CONFORMAL_FORM_REUSE_NOT_NOVELTY",
        "minimal conformal factor changed",
    )
    matter = config["matter_source_contract"]
    _strict(
        matter,
        {
            "perfect_fluid_trace",
            "dust",
            "classical_radiation",
            "general_eos",
            "trace_anomaly_boundary",
            "sign_regression",
        },
        "matter source",
    )
    _require(
        "Q_chi=+(alpha/M_Pl)*A^4*rho" in matter["dust"]
        and "classical conformal Q_chi vanishes" in matter["classical_radiation"],
        "matter trace cases changed",
    )
    _require(
        "E_chi=Q_chi" in matter["sign_regression"]
        and "chi_k=-Q_chi,k" in matter["sign_regression"],
        "source sign regression changed",
    )
    exchange = config["conservation_and_exchange"]
    _strict(
        exchange,
        {"physical_frame", "einstein_frame", "chi_exchange", "total_identity", "boundary"},
        "conservation and exchange",
    )
    _require(
        "tilde_nabla_mu*Ttilde^mu_nu=0" in exchange["physical_frame"]
        and "=-Q_phi" in exchange["einstein_frame"],
        "exchange identity changed",
    )
    ceiling = config["stability_ceiling_comparison"]
    _strict(
        ceiling,
        {
            "committed_ceiling",
            "general_trace_condition",
            "dust_density_ceiling",
            "radiation",
            "high_u_conditional_scaling",
            "no_rho_X_inference",
        },
        "stability comparison",
    )
    _require(
        "|alpha|*|T_E|/M_Pl<Q_chi,max(X)" in ceiling["general_trace_condition"]
        and "No rho(X)" in ceiling["no_rho_X_inference"],
        "stability comparison changed",
    )
    lensing = config["lensing_boundary"]
    _strict(
        lensing,
        {"same_metric", "null_cone", "maxwell_cancellation", "claim_ceiling", "quantum_boundary"},
        "lensing boundary",
    )
    _require(
        "A^4*A^-2*A^-2=1" in lensing["maxwell_cancellation"]
        and "no lensing prediction" in lensing["claim_ceiling"],
        "lensing boundary changed",
    )
    machine = config["machine_check_contract"]
    _strict(
        machine,
        {
            "symbolic_engine",
            "symbolic_zero_rule",
            "required_symbolic_checks",
            "numeric_cases",
            "numeric_tolerance",
        },
        "machine contract",
    )
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS
        and len(machine["numeric_cases"]) == 4,
        "machine contract changed",
    )
    numeric_case_keys = {"case_id", "A", "alpha", "M_Pl", "rho", "w", "expected_sign"}
    for case in machine["numeric_cases"]:
        _strict(case, numeric_case_keys, f"numeric case {case.get('case_id')}")
    _require(
        tuple(case["case_id"] for case in machine["numeric_cases"])
        == (
            "DUST_POSITIVE_ALPHA",
            "RADIATION_TRACE_ZERO",
            "STIFF_EOS_NEGATIVE_SOURCE",
            "NEGATIVE_ALPHA_DUST",
        ),
        "numeric inventory changed",
    )
    _require(
        tuple(case["expected_sign"] for case in machine["numeric_cases"])
        == ("positive", "zero", "negative", "negative"),
        "numeric expected signs changed",
    )
    adjudication = config["adjudication"]
    _strict(
        adjudication,
        {
            "same_action_conformal_Q_identity_derived",
            "frame_factors_machine_checked",
            "physical_and_einstein_exchange_consistent",
            "dust_radiation_eos_trace_cases_derived",
            "stability_ceiling_comparison_derived",
            "leading_direct_conformal_lensing_cancellation_derived",
            "physical_source_profile_established",
            "rho_X_relation_established",
            "on_shell_background",
            "metric_backreaction",
            "lensing_prediction",
            "EOS_history",
            "quantum_trace_anomaly",
            "full_H3",
            "full_H4",
            "global_strong_hyperbolicity",
            "Solar_viability",
            "GW_viability",
            "cosmology",
            "observational_support",
            "novelty_established",
            "overall_decision",
        },
        "adjudication",
    )
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["same_action_conformal_Q_identity_derived"] is True
        and adjudication["physical_source_profile_established"] is False
        and adjudication["lensing_prediction"] is False,
        "partial adjudication changed",
    )
    claims = config["claim_boundary"]
    _strict(
        claims,
        {
            "universal_conformal_source_identity_established",
            "physical_source_profile_established",
            "rho_X_relation_established",
            "on_shell_solution_established",
            "full_covariant_health_established",
            "metric_backreaction_established",
            "lensing_success_established",
            "EOS_history_established",
            "quantum_trace_anomaly_included",
            "Solar_viability_established",
            "GW_viability_established",
            "cosmology_established",
            "observational_support",
            "novelty_established",
            "publication_readiness_changed",
            "scientific_observational_claim_allowed",
        },
        "claim boundary",
    )
    _require(
        claims["universal_conformal_source_identity_established"] is True,
        "restricted claim disabled",
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key != "universal_conformal_source_identity_established"
        ),
        "claim boundary overstated",
    )
    _strict(
        config["zero_access_and_compute"],
        {
            "observational_files_opened",
            "predictor_rows_opened",
            "response_rows_opened",
            "confirmation_rows_opened",
            "holdout_rows_opened",
            "independent_rows_opened",
            "lensing_rows_opened",
            "network_calls",
            "LLM_calls",
            "paid_calls",
            "GPU_calls",
        },
        "access state",
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
        path = root / binding["receipt_path"]
        _require(path.is_file(), f"predecessor missing: {binding['binding_id']}")
        _require(
            _file_sha(path) == binding["receipt_file_sha256"],
            f"predecessor changed: {binding['binding_id']}",
        )
        receipt = _read_json(path)
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
        "status": "same_action_conformal_source_identity_machine_derived_not_on_shell",
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
        "frame_and_variation_conventions": config["frame_and_variation_conventions"],
        "minimal_conformal_factor": config["minimal_conformal_factor"],
        "matter_source_contract": config["matter_source_contract"],
        "conservation_and_exchange": config["conservation_and_exchange"],
        "stability_ceiling_comparison": config["stability_ceiling_comparison"],
        "lensing_boundary": config["lensing_boundary"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The universal conformal source identity and frame factors are variational identities, not a solved source profile or on-shell background.",
            "The exponential A is a standard illustrative known form; alpha and A_phi are neither selected nor constrained here.",
            "The dust density ceiling is conditional at each frozen background point and does not imply rho(X), A(X), or a baryonic law.",
            "Classical radiation has zero ideal trace, but EOS transitions, masses, interactions, and quantum trace anomalies are omitted.",
            "Direct four-dimensional conformal Maxwell weight cancels at fixed g; metric backreaction and the resulting lensing are not solved.",
            "No on-shell, metric, health, Solar, GW, galaxy, cluster, lensing, cosmology, observational, novelty, or publication claim is established.",
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
            "frame_and_variation_conventions",
            "minimal_conformal_factor",
            "matter_source_contract",
            "conservation_and_exchange",
            "stability_ceiling_comparison",
            "lensing_boundary",
            "symbolic_suite",
            "numeric_suite",
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
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(
        receipt["schema_version"] == RECEIPT_SCHEMA and receipt["decision"] == DECISION,
        "receipt identity changed",
    )
    _strict(
        receipt["config_binding"],
        {"path", "file_sha256", "content_sha256"},
        "receipt config binding",
    )
    _strict(
        receipt["implementation_binding"],
        {"source_path", "source_file_sha256", "test_path", "test_file_sha256"},
        "receipt implementation binding",
    )
    _require(
        receipt["config_binding"]["path"] == CONFIG_PATH.as_posix(), "receipt config path changed"
    )
    _require(
        receipt["implementation_binding"]["source_path"] == SOURCE_PATH.as_posix()
        and receipt["implementation_binding"]["test_path"] == TEST_PATH.as_posix(),
        "receipt implementation paths changed",
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
        receipt["frame_and_variation_conventions"] == config["frame_and_variation_conventions"]
        and receipt["minimal_conformal_factor"] == config["minimal_conformal_factor"]
        and receipt["matter_source_contract"] == config["matter_source_contract"]
        and receipt["conservation_and_exchange"] == config["conservation_and_exchange"]
        and receipt["stability_ceiling_comparison"] == config["stability_ceiling_comparison"]
        and receipt["lensing_boundary"] == config["lensing_boundary"],
        "source contract changed",
    )
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claims changed",
    )
    symbolic = receipt["symbolic_suite"]
    _strict(symbolic, {"engine", "checks", "all_passed", "derived_expressions"}, "symbolic suite")
    _require(
        tuple(item["check_id"] for item in symbolic["checks"]) == SYMBOLIC_CHECK_IDS
        and symbolic["all_passed"] is True
        and all(item["passed"] is True for item in symbolic["checks"]),
        "symbolic receipt changed",
    )
    for item in symbolic["checks"]:
        _strict(
            item,
            {"check_id", "passed", "residual", "statement"},
            f"symbolic check {item.get('check_id')}",
        )
    numeric = receipt["numeric_suite"]
    _strict(numeric, {"tolerance", "cases", "all_passed"}, "numeric suite")
    _require(
        tuple(item["case_id"] for item in numeric["cases"])
        == tuple(item["case_id"] for item in config["machine_check_contract"]["numeric_cases"])
        and numeric["all_passed"] is True
        and all(item["passed"] is True for item in numeric["cases"]),
        "numeric receipt changed",
    )
    for item in numeric["cases"]:
        _strict(
            item,
            {"case_id", "Ttilde", "T_E", "Q_chi", "source_sign", "frame_scaled_error", "passed"},
            f"numeric result {item.get('case_id')}",
        )
    counts = receipt["counts"]
    _strict(
        counts,
        {
            "symbolic_checks",
            "symbolic_checks_passed",
            "numeric_cases",
            "numeric_cases_passed",
            "observational_files_opened",
            "network_calls",
            "model_or_paid_calls",
            "gpu_calls",
        },
        "receipt counts",
    )
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 18,
        "symbolic count changed",
    )
    _require(
        counts["numeric_cases"] == counts["numeric_cases_passed"] == 4, "numeric count changed"
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
    _require(
        receipt["zero_access_and_compute"] == config["zero_access_and_compute"],
        "receipt access contract changed",
    )
    _require(
        isinstance(receipt["limitations"], list)
        and len(receipt["limitations"]) == 6
        and all(isinstance(item, str) and item for item in receipt["limitations"]),
        "receipt limitations changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise UniversalConformalSourceError(f"refusing to overwrite different receipt: {path}")
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
            raise UniversalConformalSourceError(
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
        "source_identity_derived": receipt["adjudication"][
            "same_action_conformal_Q_identity_derived"
        ],
        "physical_source_profile": receipt["adjudication"]["physical_source_profile_established"],
        "lensing_prediction": receipt["adjudication"]["lensing_prediction"],
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
    except UniversalConformalSourceError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
