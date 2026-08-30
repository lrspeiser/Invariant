"""No-data Solar-System and gravitational-wave necessary conditions."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_solar_gw_necessary_conditions_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_solar_gw_necessary_conditions.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_solar_gw_necessary_conditions.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-solar-gw-necessary-conditions-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-solar-gw-necessary-conditions-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-solar-gw-necessary-conditions-receipt-1.0"
DECISION = "PARTIAL_SOLAR_AND_GW_NECESSARY_CONDITIONS_DERIVED_BOTH_PHYSICAL_GATES_BLOCKED"
EXPECTED_CONFIG_FILE_SHA256 = "e4c1c99e2fde3a9305667fb1d749fe035c378232300859bc9e480791fd738b06"
EXPECTED_CONFIG_CONTENT_SHA256 = "932779636b2298e0fc199097488df8c56da62f674c9b59f1aa8d79108f1741e7"

SYMBOLIC_CHECK_IDS = (
    "S01_CONFORMAL_PHI_POTENTIAL",
    "S02_CONFORMAL_PSI_POTENTIAL",
    "S03_PPN_GAMMA_RATIO",
    "S04_PPN_GAMMA_DEVIATION",
    "S05_SOLAR_SIGNED_LOWER_ENDPOINT",
    "S06_SOLAR_SIGNED_UPPER_ENDPOINT",
    "S07_CANONICAL_COUPLING_BUDGET",
    "S08_YUKAWA_HOMOGENEOUS_EQUATION",
    "S09_YUKAWA_FORCE_RATIO",
    "S10_HIGH_U_YUKAWA_SUPPRESSION",
    "S11_CONFORMAL_NULL_CONE",
    "S12_HOMOGENEOUS_DISFORMAL_PHOTON_SPEED",
    "S13_HOMOGENEOUS_DISFORMAL_SPEED_RATIO",
    "S14_DISFORMAL_Q_INTERVAL_ENDPOINTS",
    "S15_STATIC_RADIAL_DISFORMAL_SPEED_RATIO",
    "S16_CONFORMAL_TRACE_SOURCE",
)

CONFIG_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "purpose",
    "predecessor_bindings",
    "threshold_provenance",
    "frozen_background_scope",
    "conformal_solar_contract",
    "cone_contract",
    "gate_adjudication",
    "claim_boundary",
    "machine_check_contract",
    "zero_access_and_compute",
    "output_path",
}
PREDECESSOR_KEYS = {
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
THRESHOLD_KEYS = {
    "scope",
    "solar_gamma_absolute_ceiling",
    "solar_source",
    "gw_fractional_speed_ceiling",
    "gw_source",
    "measurement_claim",
}
BACKGROUND_KEYS = {"solar", "ppn_subcase", "source_subcase", "gw", "forbidden_inference"}
SOLAR_KEYS = {
    "universal_metric",
    "linear_physical_potentials",
    "fifth_force_definition",
    "matter_variation_derivation",
    "ppn_derivation",
    "signed_necessary_interval",
    "attractive_necessary_ceiling",
    "canonical_coupling_subcase",
    "split_chi_yukawa_subcase",
    "high_acceleration_implication",
    "lensing_implication",
    "not_derived",
}
CONE_KEYS = {
    "einstein_tensor_cone",
    "conformal_photon_cone",
    "conformal_gate_status",
    "homogeneous_disformal_metric",
    "homogeneous_disformal_cones",
    "disformal_necessary_interval",
    "nonnegative_disformal_ceiling",
    "static_radial_disformal_metric",
    "static_radial_disformal_cones",
    "disformal_gate_status",
    "not_claimed",
}
ADJUDICATION_KEYS = {
    "conformal_fifth_force_ppn_relation_derived",
    "conformal_trace_source_functional_derived",
    "solar_necessary_inequality_derived",
    "split_chi_restricted_yukawa_ratio_derived",
    "conformal_tensor_photon_cones_aligned",
    "disformal_cone_ratio_derived",
    "disformal_necessary_inequality_derived",
    "physical_matter_source_derived",
    "solar_on_shell_background_solved",
    "metric_solution_solved",
    "full_ppn_parameters_derived",
    "late_time_on_shell_background_solved",
    "full_constrained_characteristics_derived",
    "solar_gate_passed",
    "gw_gate_passed",
    "overall_decision",
}
CLAIM_KEYS = {
    "restricted_necessary_conditions_established",
    "solar_viability_established",
    "cassini_prediction_computed",
    "gw_speed_observational_pass_established",
    "disformal_branch_viable",
    "physical_source_law_established",
    "on_shell_background_established",
    "full_covariant_health_established",
    "lensing_success_established",
    "observational_support",
    "novelty_established",
    "publication_readiness_changed",
    "scientific_observational_claim_allowed",
}
MACHINE_KEYS = {
    "symbolic_engine",
    "symbolic_zero_rule",
    "required_symbolic_checks",
    "numeric_yukawa_mr_probes",
    "designed_blocked_gates",
}
ZERO_KEYS = {
    "observational_files_opened",
    "observational_rows_opened",
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
}


class SolarGwNecessaryConditionsError(RuntimeError):
    """Raised when a frozen necessary-condition contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SolarGwNecessaryConditionsError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    _require(actual == keys, f"{label} schema changed: {sorted(actual ^ keys)}")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    payload = _canonical_bytes(value)
    return hashlib.sha256(payload[:-1]).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SolarGwNecessaryConditionsError(f"cannot read JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _check(check_id: str, actual: sp.Expr, expected: sp.Expr, statement: str) -> dict[str, Any]:
    residual = sp.simplify(actual - expected)
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(residual),
        "passed": residual == 0,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(config, CONFIG_KEYS, "config")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-solar-gw-necessary-conditions-v1",
        "analysis identity changed",
    )
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content hash changed")

    bindings = config["predecessor_bindings"]
    _require(isinstance(bindings, list) and len(bindings) == 4, "predecessor inventory changed")
    for index, binding in enumerate(bindings):
        _strict(binding, PREDECESSOR_KEYS, f"predecessor[{index}]")
    _require(
        [item["binding_id"] for item in bindings]
        == [
            "theory_preflight",
            "external_metric_principal_symbol",
            "split_gate_action",
            "split_gate_source_bound",
        ],
        "predecessor order changed",
    )
    _require(
        bindings[-1]["git_commit"] == "7216ff7319e0f38c7639926b31e7f4e881f9a64a",
        "source-bound predecessor changed",
    )

    _strict(config["threshold_provenance"], THRESHOLD_KEYS, "threshold provenance")
    thresholds = config["threshold_provenance"]
    _require(
        thresholds["solar_gamma_absolute_ceiling"] == "23/1000000"
        and thresholds["gw_fractional_speed_ceiling"] == "1/1000000000000000"
        and thresholds["measurement_claim"] is False,
        "threshold or provenance scope changed",
    )
    _require("H7" in thresholds["solar_source"], "Solar threshold provenance changed")
    _require("H6" in thresholds["gw_source"], "GW threshold provenance changed")

    _strict(config["frozen_background_scope"], BACKGROUND_KEYS, "background scope")
    _strict(config["conformal_solar_contract"], SOLAR_KEYS, "Solar contract")
    _strict(config["cone_contract"], CONE_KEYS, "cone contract")
    _strict(config["gate_adjudication"], ADJUDICATION_KEYS, "gate adjudication")
    _strict(config["claim_boundary"], CLAIM_KEYS, "claim boundary")
    _strict(config["machine_check_contract"], MACHINE_KEYS, "machine contract")
    _strict(config["zero_access_and_compute"], ZERO_KEYS, "zero access")

    solar = config["conformal_solar_contract"]
    _require(
        "gamma_PPN-1=-2*epsilon/(1+epsilon)" in solar["ppn_derivation"], "PPN relation changed"
    )
    _require(
        "Q_a=-(1/sqrt(-g))*delta S_m/delta(varphi_a)" in solar["matter_variation_derivation"]
        and "=-(partial_a ln A)*T_E" in solar["matter_variation_derivation"],
        "conformal trace source derivation changed",
    )
    _require("23/1999977" in solar["attractive_necessary_ceiling"], "Solar ceiling changed")
    _require("23/3999954" in solar["canonical_coupling_subcase"], "coupling budget changed")
    _require(
        "epsilon_chi(r)=2*alpha_chi^2" in solar["split_chi_yukawa_subcase"], "Yukawa ratio changed"
    )
    _require(len(solar["not_derived"]) == 5, "Solar blockers changed")

    cones = config["cone_contract"]
    _require("c_GW/c_gamma=1" in cones["conformal_photon_cone"], "conformal cone changed")
    _require("Delta^(-1/2)" in cones["homogeneous_disformal_cones"], "disformal cone changed")
    _require(
        cones["conformal_gate_status"].startswith("BLOCKED_")
        and cones["disformal_gate_status"].startswith("BLOCKED_"),
        "cone gate unblocked",
    )

    adjudication = config["gate_adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    for key in (
        "conformal_fifth_force_ppn_relation_derived",
        "conformal_trace_source_functional_derived",
        "solar_necessary_inequality_derived",
        "split_chi_restricted_yukawa_ratio_derived",
        "conformal_tensor_photon_cones_aligned",
        "disformal_cone_ratio_derived",
        "disformal_necessary_inequality_derived",
    ):
        _require(adjudication[key] is True, f"derived result disabled: {key}")
    for key in (
        "physical_matter_source_derived",
        "solar_on_shell_background_solved",
        "metric_solution_solved",
        "full_ppn_parameters_derived",
        "late_time_on_shell_background_solved",
        "full_constrained_characteristics_derived",
        "solar_gate_passed",
        "gw_gate_passed",
    ):
        _require(adjudication[key] is False, f"blocked gate overclaimed: {key}")

    claims = config["claim_boundary"]
    _require(
        claims["restricted_necessary_conditions_established"] is True, "restricted claim disabled"
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key != "restricted_necessary_conditions_established"
        ),
        "claim boundary overstated",
    )
    machine = config["machine_check_contract"]
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(
        machine["designed_blocked_gates"] == ["SOLAR_SYSTEM", "GW_AND_PHOTON_CONES"],
        "blocked gate inventory changed",
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "access state changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


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
            _require(path.is_file(), f"predecessor missing: {binding['binding_id']}:{path_key}")
            _require(
                _file_sha(path) == binding[hash_key],
                f"predecessor changed: {binding['binding_id']}:{path_key}",
            )
        receipt = _read_json(root / binding["receipt_path"])
        _require(
            receipt.get("schema_version") == binding["receipt_schema_version"],
            "predecessor schema changed",
        )
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            "predecessor content changed",
        )
        _require(
            receipt.get("decision") == binding["receipt_decision"], "predecessor decision changed"
        )


def run_symbolic_suite() -> dict[str, Any]:
    epsilon, u_potential = sp.symbols("epsilon U", real=True)
    delta_s, delta_g = sp.symbols("delta_S delta_G", positive=True)
    alpha, mass, m_eff, m_pl, radius, y0 = sp.symbols("alpha M m_eff M_Pl r Y0", positive=True)
    conformal, k2 = sp.symbols("A k2", positive=True)
    q, radial_s = sp.symbols("q s", real=True)
    alpha_a, trace = sp.symbols("alpha_a T_E", real=True)

    phi_e = -u_potential
    psi_e = -u_potential
    a_scalar = -epsilon * u_potential
    phi = phi_e + a_scalar
    psi = psi_e - a_scalar
    gamma = sp.factor(psi / phi)

    solar_lower = -delta_s / (2 + delta_s)
    solar_upper = delta_s / (2 - delta_s)
    canonical_budget = solar_upper / 2

    yukawa_shape = sp.exp(-m_eff * radius) / radius
    yukawa_laplacian = sp.diff(yukawa_shape, radius, 2) + 2 * sp.diff(yukawa_shape, radius) / radius
    chi = -alpha * mass * yukawa_shape / (4 * sp.pi * y0 * m_pl)
    einstein_phi = -mass / (8 * sp.pi * m_pl**2 * radius)
    scalar_a = alpha * chi / m_pl
    yukawa_ratio = sp.factor(sp.diff(scalar_a, radius) / sp.diff(einstein_phi, radius))
    high_u_ratio = (
        2
        * alpha**2
        * (1 + mass * (1 + u_potential) * radius / sp.sqrt(y0))
        * sp.exp(-mass * (1 + u_potential) * radius / sp.sqrt(y0))
        / y0
    )

    photon_speed = sp.sqrt(1 - q)
    gw_photon_ratio = 1 / photon_speed
    q_lower = 1 - 1 / (1 - delta_g) ** 2
    q_upper = 1 - 1 / (1 + delta_g) ** 2

    checks = [
        _check(
            "S01_CONFORMAL_PHI_POTENTIAL",
            phi,
            -(1 + epsilon) * u_potential,
            "The universal conformal factor adds a to the temporal potential.",
        ),
        _check(
            "S02_CONFORMAL_PSI_POTENTIAL",
            psi,
            -(1 - epsilon) * u_potential,
            "The same factor subtracts a from the spatial potential.",
        ),
        _check(
            "S03_PPN_GAMMA_RATIO",
            gamma,
            (1 - epsilon) / (1 + epsilon),
            "The common 1/r subcase fixes gamma_PPN rather than assuming it.",
        ),
        _check(
            "S04_PPN_GAMMA_DEVIATION",
            gamma - 1,
            -2 * epsilon / (1 + epsilon),
            "The PPN deviation is the exact conformal fifth-force relation.",
        ),
        _check(
            "S05_SOLAR_SIGNED_LOWER_ENDPOINT",
            -2 * solar_lower / (1 + solar_lower),
            delta_s,
            "The signed repulsive endpoint saturates the inherited absolute gamma ceiling.",
        ),
        _check(
            "S06_SOLAR_SIGNED_UPPER_ENDPOINT",
            2 * solar_upper / (1 + solar_upper),
            delta_s,
            "The attractive endpoint saturates the inherited absolute gamma ceiling.",
        ),
        _check(
            "S07_CANONICAL_COUPLING_BUDGET",
            2 * canonical_budget,
            solar_upper,
            "A canonical scalar contributes epsilon=2 alpha^2/kappa.",
        ),
        _check(
            "S08_YUKAWA_HOMOGENEOUS_EQUATION",
            yukawa_laplacian - m_eff**2 * yukawa_shape,
            sp.Integer(0),
            "The decaying Yukawa profile solves the exterior homogeneous equation for r>0.",
        ),
        _check(
            "S09_YUKAWA_FORCE_RATIO",
            yukawa_ratio,
            2 * alpha**2 * (1 + m_eff * radius) * sp.exp(-m_eff * radius) / y0,
            "The restricted point-source force ratio includes the Yukawa derivative factor.",
        ),
        _check(
            "S10_HIGH_U_YUKAWA_SUPPRESSION",
            sp.limit(high_u_ratio, u_potential, sp.oo),
            sp.Integer(0),
            "The split mass gate suppresses the restricted chi force at fixed positive r.",
        ),
        _check(
            "S11_CONFORMAL_NULL_CONE",
            (k2 / conformal**2).subs(k2, 0),
            sp.Integer(0),
            "On a g-null covector, the conformal inverse-metric null form also vanishes.",
        ),
        _check(
            "S12_HOMOGENEOUS_DISFORMAL_PHOTON_SPEED",
            photon_speed**2,
            1 - q,
            "The homogeneous disformal photon speed squared is Delta=1-q.",
        ),
        _check(
            "S13_HOMOGENEOUS_DISFORMAL_SPEED_RATIO",
            gw_photon_ratio,
            (1 - q) ** sp.Rational(-1, 2),
            "Einstein tensor waves and disformal photons have ratio Delta^-1/2.",
        ),
        _check(
            "S14_DISFORMAL_Q_INTERVAL_ENDPOINTS",
            1 / (1 - q_lower) + 1 / (1 - q_upper),
            2 * (1 + delta_g**2),
            "The squared speed ratios at the exact q endpoints are (1-delta_G)^2 and (1+delta_G)^2.",
        ),
        _check(
            "S15_STATIC_RADIAL_DISFORMAL_SPEED_RATIO",
            1 / ((1 + radial_s) ** sp.Rational(-1, 2)),
            sp.sqrt(1 + radial_s),
            "A radial disformal gradient separates the radial photon and tensor cones.",
        ),
        _check(
            "S16_CONFORMAL_TRACE_SOURCE",
            -sp.Rational(1, 2) * 2 * alpha_a * trace,
            -alpha_a * trace,
            "Varying one universal conformal metric gives Q_a=-(partial_a ln A)*T_E.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic suite order changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic suite failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "all_passed": True,
        "checks": checks,
        "exact_results": {
            "solar_signed_epsilon_interval": ["-23/2000023", "23/1999977"],
            "solar_attractive_epsilon_ceiling": "23/1999977",
            "canonical_alpha_squared_over_kappa_ceiling": "23/3999954",
            "conformal_speed_ratio": "1",
            "disformal_q_interval": [
                "1-(1-1/1000000000000000)^(-2)",
                "1-(1+1/1000000000000000)^(-2)",
            ],
        },
    }


def _float_string(value: float) -> str:
    _require(math.isfinite(value), "non-finite numeric result")
    return format(value, ".17g")


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    delta_s = sp.Rational(23, 1_000_000)
    delta_g = sp.Rational(1, 1_000_000_000_000_000)
    lower_s = -delta_s / (2 + delta_s)
    upper_s = delta_s / (2 - delta_s)
    q_lower = 1 - 1 / (1 - delta_g) ** 2
    q_upper = 1 - 1 / (1 + delta_g) ** 2
    yukawa = []
    for raw in config["machine_check_contract"]["numeric_yukawa_mr_probes"]:
        mr = float(raw)
        factor = (1 + mr) * math.exp(-mr)
        yukawa.append(
            {
                "m_eff_r": raw,
                "force_suppression_relative_to_massless": _float_string(factor),
                "bounded_between_zero_and_one": 0.0 <= factor <= 1.0,
            }
        )
    _require(all(item["bounded_between_zero_and_one"] for item in yukawa), "Yukawa probe failed")
    _require(
        [float(item["force_suppression_relative_to_massless"]) for item in yukawa]
        == sorted(
            [float(item["force_suppression_relative_to_massless"]) for item in yukawa],
            reverse=True,
        ),
        "Yukawa suppression is not monotone on frozen probes",
    )
    return {
        "all_passed": True,
        "solar_threshold_bookkeeping": {
            "delta_gamma": str(delta_s),
            "signed_epsilon_lower": str(lower_s),
            "signed_epsilon_upper": str(upper_s),
            "attractive_epsilon_ceiling_decimal": _float_string(float(upper_s)),
            "canonical_alpha_squared_over_kappa_ceiling": str(upper_s / 2),
        },
        "gw_threshold_bookkeeping": {
            "delta_speed": str(delta_g),
            "q_lower": str(q_lower),
            "q_upper": str(q_upper),
            "q_lower_decimal": _float_string(float(q_lower)),
            "q_upper_decimal": _float_string(float(q_upper)),
        },
        "yukawa_force_probes": yukawa,
        "designed_blocked_gates": ["SOLAR_SYSTEM", "GW_AND_PHOTON_CONES"],
    }


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
        "status": "restricted_necessary_conditions_machine_derived_physical_gates_blocked",
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
        "threshold_provenance": config["threshold_provenance"],
        "frozen_background_scope": config["frozen_background_scope"],
        "conformal_solar_contract": config["conformal_solar_contract"],
        "cone_contract": config["cone_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "gate_adjudication": config["gate_adjudication"],
        "claim_boundary": config["claim_boundary"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_yukawa_probes": len(numeric["yukawa_force_probes"]),
            "numeric_yukawa_probes_passed": sum(
                item["bounded_between_zero_and_one"] for item in numeric["yukawa_force_probes"]
            ),
            "designed_blocked_gates": len(numeric["designed_blocked_gates"]),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "zero_access_and_compute": config["zero_access_and_compute"],
        "remaining_blockers": [
            "Freeze A(phi,chi), P(X), V(phi), parameters, branch, cutoff, and boundary conditions without Solar-only coefficients.",
            "Derive Q_phi and Q_chi by varying the same universal matter action for a finite material source; do not insert rho by hand.",
            "Solve and match a regular static spherical Sun/interior/exterior scalar-metric background, retaining nonconstant coefficients and active phi-chi mixing.",
            "Derive the full weak-field metric and direct light-time, orbital, beta_PPN, preferred-frame, and disformal observables from that one solution.",
            "Solve an on-shell late-time cosmological background and the full constrained tensor-scalar-matter characteristic system.",
            "For a nonzero disformal branch, satisfy the exact frozen q interval everywhere relevant and establish invertibility, hyperbolicity, and EFT control.",
            "Only after all theory inputs and adjudicators are frozen may separately authorized observational rows be opened; this package opens none.",
        ],
        "limitations": [
            "The PPN relation is exact only within the frozen weak-field common-1/r conformal subcase; Yukawa and nonlinear profiles require direct predictions.",
            "The point-source Yukawa force ratio is a restricted constant-coefficient illustration, not the missing physical source/background solution.",
            "Conformal cone alignment is structural and local; it is not an empirical GW-speed pass or proof of the full constrained system.",
            "The disformal inequalities are necessary cone conditions on prescribed backgrounds, not proof that either background exists dynamically.",
            "No observational data, likelihood, fit, Solar/GW pass, lensing result, novelty result, or publication claim is produced.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
    validate_receipt(receipt, config)
    return receipt


RECEIPT_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "decision",
    "config_binding",
    "implementation_binding",
    "predecessor_bindings",
    "threshold_provenance",
    "frozen_background_scope",
    "conformal_solar_contract",
    "cone_contract",
    "symbolic_suite",
    "numeric_suite",
    "gate_adjudication",
    "claim_boundary",
    "counts",
    "zero_access_and_compute",
    "remaining_blockers",
    "limitations",
    "content_sha256",
}
COUNT_KEYS = {
    "symbolic_checks",
    "symbolic_checks_passed",
    "numeric_yukawa_probes",
    "numeric_yukawa_probes_passed",
    "designed_blocked_gates",
    "observational_files_opened",
    "observational_rows_opened",
    "network_calls",
    "model_or_paid_calls",
    "gpu_calls",
}


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(receipt, RECEIPT_KEYS, "receipt")
    body = dict(receipt)
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["analysis_id"] == config["analysis_id"], "receipt identity changed")
    _require(receipt["decision"] == DECISION, "receipt decision changed")
    _strict(receipt["config_binding"], {"path", "file_sha256", "content_sha256"}, "config binding")
    _strict(
        receipt["implementation_binding"],
        {"source_path", "source_file_sha256", "test_path", "test_file_sha256"},
        "implementation binding",
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
        receipt["threshold_provenance"] == config["threshold_provenance"],
        "threshold provenance changed",
    )
    _require(
        receipt["conformal_solar_contract"] == config["conformal_solar_contract"],
        "Solar contract changed",
    )
    _require(receipt["cone_contract"] == config["cone_contract"], "cone contract changed")
    _require(
        receipt["gate_adjudication"] == config["gate_adjudication"], "gate adjudication changed"
    )
    _require(receipt["claim_boundary"] == config["claim_boundary"], "claims changed")
    _strict(receipt["counts"], COUNT_KEYS, "counts")
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 16,
        "symbolic count changed",
    )
    _require(
        counts["numeric_yukawa_probes"] == counts["numeric_yukawa_probes_passed"] == 3,
        "numeric count changed",
    )
    _require(counts["designed_blocked_gates"] == 2, "blocked gate count changed")
    _require(
        all(
            counts[key] == 0
            for key in (
                "observational_files_opened",
                "observational_rows_opened",
                "network_calls",
                "model_or_paid_calls",
                "gpu_calls",
            )
        ),
        "receipt access changed",
    )
    _require(
        receipt["zero_access_and_compute"] == config["zero_access_and_compute"],
        "zero-access ledger changed",
    )
    _require(len(receipt["remaining_blockers"]) == 7, "blocker inventory changed")
    _require(len(receipt["limitations"]) == 5, "limitation inventory changed")


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise SolarGwNecessaryConditionsError(f"refusing to overwrite different receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SolarGwNecessaryConditionsError(
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
        "solar_gate_passed": receipt["gate_adjudication"]["solar_gate_passed"],
        "gw_gate_passed": receipt["gate_adjudication"]["gw_gate_passed"],
        "observational_rows_opened": receipt["counts"]["observational_rows_opened"],
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
    except SolarGwNecessaryConditionsError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
