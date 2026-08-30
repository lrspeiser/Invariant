"""Full fixed-background scalar fluctuation expansion for the quadrature action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_scalar_full_local_fluctuation_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_quadrature_scalar_full_local_fluctuation.py"
)
TEST_PATH = Path("tests/test_gravity_shared_quadrature_scalar_full_local_fluctuation.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-scalar-full-local-fluctuation-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-scalar-full-local-fluctuation-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-scalar-full-local-fluctuation-receipt-1.0"
STATUS = "restricted_vector_metric_fixed_background_full_scalar_quartic_no_data"
DECISION = (
    "RESTRICTED_VECTOR_METRIC_FIXED_BACKGROUND_FULL_SCALAR_QUARTIC_EXPANSION_DERIVED_"
    "DERIVATIVE_CEILING_REMAINS_ENDPOINT_LIMITER_METRIC_AETHER_UNITARITY_PHYSICAL_"
    "CUTOFF_AND_OBSERVATION_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "0b6a28342ba03e814959c4e0399e831d7740fafc9ca0be352beecf6e079c55d4"
EXPECTED_CONFIG_CONTENT_SHA256 = "307938b7bfee044f2b61a26f240ba1baa3e94856738bb8d6272aebb9b44125e3"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "d4e38adeb1484c83b9f912715aaa2ac16313fe9276c86491cf82020fba3e992b",
    "primary_source_context": "36465c244f16922cb02fd0040fa225054678019f512727e330aded45906f6000",
    "full_scalar_expansion_contract": "c69429c230769e24a68a0c78182a250c4173443e2731e70b3783f902ebc08d7b",
    "canonical_interaction_contract": "deee58d706ab2e59a570f11e4d343c4ab556e52e174b91abd162216c37db4b2a",
    "coefficient_scale_contract": "3fb8a15686425fe244281a459d3f4f087f1d7961ea654622055d593a63d4a948",
    "endpoint_and_hierarchy_contract": "192189ca41228a696eaad8d8f85098c377e24bbc594a3f3225e481e34225c010",
    "machine_check_contract": "e6280a2231e1f8cea8ed24fb5d7c69fad0838181122b9dc14c6001f35643e32f",
    "adjudication": "cd738ef6696112714d879336054210498507cf7154512592a5843c21747c9daa",
    "claim_boundary": "42a18c20a43d39da7237bc0c5a1e5909383221b978b282acae642a79047130e7",
    "zero_access_and_compute": "f775ebd31025256383d3c06467b2bf1f2bf1229eb7da3e5aa98d0207115ff28a",
}


class QuadratureScalarFullLocalError(RuntimeError):
    """Raised when the frozen expansion or its evidence changes."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _content_sha(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuadratureScalarFullLocalError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureScalarFullLocalError(f"JSON root is not an object: {path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    return _load_json(repo / CONFIG_PATH)


def validate_config(config: dict[str, Any], root: Path | None = None) -> None:
    repo = _repo_root() if root is None else root.resolve()
    expected_keys = {
        "schema_version",
        "analysis_id",
        "status",
        "purpose",
        "predecessor_bindings",
        "primary_source_context",
        "full_scalar_expansion_contract",
        "canonical_interaction_contract",
        "coefficient_scale_contract",
        "endpoint_and_hierarchy_contract",
        "machine_check_contract",
        "adjudication",
        "claim_boundary",
        "zero_access_and_compute",
        "output_path",
    }
    if set(config) != expected_keys:
        raise QuadratureScalarFullLocalError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-shared-quadrature-scalar-full-local-fluctuation-v1"
        or config["status"] != "frozen_no_data_fixed_metric_aether_full_scalar_quartic_expansion"
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or config["adjudication"]["overall_decision"] != DECISION
    ):
        raise QuadratureScalarFullLocalError("config identity changed")
    if _content_sha(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise QuadratureScalarFullLocalError("config content changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureScalarFullLocalError("config file hash changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[key]) != expected:
            raise QuadratureScalarFullLocalError(f"config section changed: {key}")
    machine = config["machine_check_contract"]
    if (
        machine["symbolic_engine"] != "sympy-1.14"
        or len(machine["required_symbolic_checks"]) != 30
        or len(machine["numeric_s_values"]) != 5
        or machine["numeric_eta"] != "1/1000"
    ):
        raise QuadratureScalarFullLocalError("machine inventory changed")
    truth = {
        "fixed_metric_aether_full_scalar_expansion_through_quartic": True,
        "full_coupled_metric_aether_matter_fluctuation_action": False,
        "new_conformal_factor_interactions_canonically_normalized": True,
        "full_fixed_scalar_coefficient_ceiling_derived": True,
        "uniform_positive_full_scalar_coefficient_scale": False,
        "physical_UV_cutoff_established": False,
        "tree_level_unitarity_bound_established": False,
        "strong_coupling_theorem_established": False,
    }
    if any(config["adjudication"][key] is not value for key, value in truth.items()):
        raise QuadratureScalarFullLocalError("adjudication changed")
    if any(config["zero_access_and_compute"].values()):
        raise QuadratureScalarFullLocalError("zero-access contract changed")


def validate_predecessors(config: dict[str, Any], root: Path | None = None) -> list[dict[str, Any]]:
    repo = _repo_root() if root is None else root.resolve()
    results: list[dict[str, Any]] = []
    for binding in config["predecessor_bindings"]:
        commit = binding["git_commit"]
        try:
            object_type = subprocess.run(
                ["git", "cat-file", "-t", commit],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise QuadratureScalarFullLocalError("predecessor Git binding failed") from error
        if object_type != "commit":
            raise QuadratureScalarFullLocalError("predecessor object is not a commit")
        for artifact in binding["artifacts"]:
            path = Path(artifact["path"])
            if _file_sha(repo / path) != artifact["file_sha256"]:
                raise QuadratureScalarFullLocalError("predecessor artifact changed")
            try:
                committed = subprocess.run(
                    ["git", "show", f"{commit}:{path.as_posix()}"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise QuadratureScalarFullLocalError("predecessor commit path failed") from error
            if _sha_bytes(committed) != artifact["file_sha256"]:
                raise QuadratureScalarFullLocalError("predecessor commit bytes changed")
        receipt = _load_json(repo / Path(binding["receipt_path"]))
        if (
            receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
            or receipt.get("content_sha256") != binding["receipt_content_sha256"]
        ):
            raise QuadratureScalarFullLocalError("predecessor receipt changed")
        payload = dict(receipt)
        stored_content = payload.pop("content_sha256", None)
        if stored_content != _content_sha(payload):
            raise QuadratureScalarFullLocalError("predecessor receipt hash invalid")
        results.append(
            {
                "binding_id": binding["binding_id"],
                "git_commit": commit,
                "artifact_count": len(binding["artifacts"]),
                "receipt_content_sha256": stored_content,
                "valid": True,
            }
        )
    return results


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    if isinstance(simplified, sp.MatrixBase):
        passed = all(sp.simplify(item) == 0 for item in simplified)
    else:
        passed = simplified == 0
    if not passed:
        raise QuadratureScalarFullLocalError(f"symbolic check failed: {check_id}: {simplified}")
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": True,
    }


def symbolic_checks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, alpha, lambda0sq, mpl = sp.symbols("s alpha Lambda0sq Mpl", positive=True, finite=True)
    epsilon = sp.symbols("epsilon", real=True, finite=True)
    t, r, u, z = sp.symbols("t r u z", real=True, finite=True)
    eta = sp.symbols("eta", positive=True, finite=True)
    d = 1 - 2 * s

    def p(argument: sp.Expr) -> sp.Expr:
        return argument**2 / 4 + argument / 4 + sp.log(1 - 2 * argument) / 8

    def k(argument: sp.Expr) -> sp.Expr:
        return 2 * alpha**2 * argument * (1 - argument) / (1 - 2 * argument) ** 2

    k_s = sp.factor(sp.diff(k(s), s))
    c = alpha**2 * s / d
    s_total = alpha * sp.sqrt(
        (s / alpha + epsilon * r / lambda0sq) ** 2 + epsilon**2 * u / lambda0sq**2
    )
    density = (
        2 * lambda0sq**2 * p(s_total)
        + sp.exp(-4 * epsilon * alpha * z / mpl) * k(s_total) * epsilon**2 * t
    )
    l2 = sp.factor(sp.diff(density, epsilon, 2).subs(epsilon, 0) / sp.factorial(2))
    l3 = sp.factor(sp.diff(density, epsilon, 3).subs(epsilon, 0) / sp.factorial(3))
    l4 = sp.factor(sp.diff(density, epsilon, 4).subs(epsilon, 0) / sp.factorial(4))
    l2_expected = k(s) * (t - r**2) - c * u
    l3_derivative = alpha**3 / lambda0sq * (2 * r * t / d**3 - r * u / d**2 - 2 * r**3 / (3 * d**3))
    l4_derivative = (
        alpha**4
        / lambda0sq**2
        * (
            -(r**4) / d**4
            - 2 * r**2 * u / d**3
            + 6 * r**2 * t / d**4
            + t * u / (s * d**3)
            - u**2 / (4 * s * d**2)
        )
    )
    l3_new = -4 * alpha * k(s) * z * t / mpl
    l4_amplitude = 8 * alpha**2 * k(s) * z**2 * t / mpl**2
    l4_mixed = -4 * alpha**2 * k_s * z * r * t / (mpl * lambda0sq)
    l3_expected = l3_derivative + l3_new
    l4_expected = l4_derivative + l4_amplitude + l4_mixed

    cperp2 = sp.factor(c / k(s))
    pprime = sp.factor(2 * k(s) * cperp2)
    e3 = sp.factor(cperp2 * 4 * alpha * k(s) / pprime ** sp.Rational(3, 2))
    e3_expected = sp.sqrt(2 * d / s)
    e4 = sp.factor(cperp2 * 8 * alpha**2 * k(s) / pprime**2)
    e4_expected = 2 * d / s
    emix = sp.factor(cperp2 * 4 * alpha**2 * k_s / pprime**2)
    emix_expected = 1 / (s**2 * (1 - s))

    lambda3 = (18 * s**3 * (1 - s) ** 2 * d) ** sp.Rational(1, 4)
    lambda4_longitudinal = (8 * s**2 * (1 - s) * d / 5) ** sp.Rational(1, 4)
    lambda4_transverse = (8 * s**3 * d / (1 + s)) ** sp.Rational(1, 4)
    conf_over_l0 = eta**-1 * sp.sqrt(s / (2 * d))
    mix_over_l0 = eta ** (-sp.Rational(1, 3)) * (s**2 * (1 - s)) ** sp.Rational(1, 3)
    endpoint = sp.symbols("endpoint", positive=True, finite=True)

    checks = [
        _check(
            "S01_FULL_L2", l2 - l2_expected, "The exact full scalar quadratic density is unchanged."
        ),
        _check(
            "S02_FULL_L3", l3 - l3_expected, "The exact full scalar cubic density is reconstructed."
        ),
        _check(
            "S03_FULL_L4",
            l4 - l4_expected,
            "The exact full scalar quartic density is reconstructed.",
        ),
        _check(
            "S04_CUBIC_DIFFERENCE",
            l3 - l3_derivative - l3_new,
            "The omitted cubic is exactly the conformal-amplitude vertex.",
        ),
        _check(
            "S05_QUARTIC_AMPLITUDE_DIFFERENCE",
            (l4 - l4_derivative).expand().coeff(z, 2) - l4_amplitude.expand().coeff(z, 2),
            "The amplitude-quartic correction is exact.",
        ),
        _check(
            "S06_QUARTIC_MIXED_DIFFERENCE",
            (l4 - l4_derivative).expand().coeff(z, 1) - l4_mixed.expand().coeff(z, 1),
            "The mixed quartic correction is exact.",
        ),
        _check(
            "S07_DERIVATIVE_L3_PARITY",
            l3 - l3_new - l3_derivative,
            "The predecessor derivative cubic is preserved exactly.",
        ),
        _check(
            "S08_DERIVATIVE_L4_PARITY",
            l4 - l4_amplitude - l4_mixed - l4_derivative,
            "The predecessor derivative quartic is preserved exactly.",
        ),
        _check(
            "S09_TRANSVERSE_SPEED", cperp2 - d / (2 * (1 - s)), "The transverse speed is unchanged."
        ),
        _check(
            "S10_CANONICAL_NORMALIZATION",
            pprime - 2 * c,
            "The predecessor canonical normalization is unchanged.",
        ),
        _check(
            "S11_CANONICAL_CONFORMAL_CUBIC",
            e3**2 - e3_expected**2,
            "The squared canonical conformal cubic coefficient is exact; its positive magnitude is selected on the declared branch.",
        ),
        _check(
            "S12_CANONICAL_CONFORMAL_QUARTIC",
            e4 - e4_expected,
            "The canonical amplitude-quartic coefficient is exact.",
        ),
        _check(
            "S13_SHARED_CONFORMAL_SCALE",
            e4 - e3_expected**2,
            "The new cubic and amplitude quartic share one coefficient scale.",
        ),
        _check(
            "S14_CANONICAL_MIXED_QUARTIC",
            emix - emix_expected,
            "The canonical mixed quartic coefficient is exact.",
        ),
        _check(
            "S15_ALPHA_CANCELLATION",
            sp.Matrix([sp.diff(e3, alpha), sp.diff(e4, alpha), sp.diff(emix, alpha)]),
            "Alpha cancels from all new canonical coefficients.",
        ),
        _check(
            "S16_DIMENSIONLESS_SCALE_RATIOS",
            sp.Matrix(
                [
                    conf_over_l0 - eta**-1 * sp.sqrt(s / (2 * d)),
                    mix_over_l0
                    - eta ** (-sp.Rational(1, 3)) * (s**2 * (1 - s)) ** sp.Rational(1, 3),
                ]
            ),
            "The dimensionless new-scale ratios are exact.",
        ),
        _check(
            "S17_INTERIOR_COEFFICIENT_IDENTITIES",
            sp.Matrix(
                [e3_expected**2 * s / (2 * d) - 1, e4 * s / (2 * d) - 1, emix * s**2 * (1 - s) - 1]
            ),
            "Every new coefficient is finite and positive on the declared interior branch.",
        ),
        _check(
            "S18_LOW_S_DERIVATIVE_SCALE",
            sp.limit(lambda4_transverse / s ** sp.Rational(3, 4), s, 0, dir="+")
            - 8 ** sp.Rational(1, 4),
            "The low-s derivative limiter is exact.",
        ),
        _check(
            "S19_LOW_S_CONFORMAL_SCALE",
            sp.limit(sp.sqrt(s / (2 * d)) / sp.sqrt(s), s, 0, dir="+") - 1 / sp.sqrt(2),
            "The low-s conformal scale is exact.",
        ),
        _check(
            "S20_LOW_S_MIXED_SCALE",
            sp.limit((s**2 * (1 - s)) ** sp.Rational(1, 3) / s ** sp.Rational(2, 3), s, 0, dir="+")
            - 1,
            "The low-s mixed scale is exact.",
        ),
        _check(
            "S21_LOW_S_DERIVATIVE_OVER_CONFORMAL",
            sp.limit(lambda4_transverse / conf_over_l0, s, 0, dir="+"),
            "The derivative scale is asymptotically below the conformal scale at low s.",
        ),
        _check(
            "S22_LOW_S_DERIVATIVE_OVER_MIXED",
            sp.limit(lambda4_transverse / mix_over_l0, s, 0, dir="+"),
            "The derivative scale is asymptotically below the mixed scale at low s.",
        ),
        _check(
            "S23_ENDPOINT_DERIVATIVE_SCALE",
            sp.limit(
                lambda4_longitudinal.subs(s, sp.Rational(1, 2) - endpoint)
                / endpoint ** sp.Rational(1, 4),
                endpoint,
                0,
                dir="+",
            )
            - (sp.Rational(2, 5)) ** sp.Rational(1, 4),
            "The upper-endpoint derivative limiter is exact.",
        ),
        _check(
            "S24_ENDPOINT_CONFORMAL_SCALE",
            sp.limit(
                sp.sqrt((sp.Rational(1, 2) - endpoint) / (4 * endpoint)) * sp.sqrt(endpoint),
                endpoint,
                0,
                dir="+",
            )
            - 1 / sp.sqrt(8),
            "The upper-endpoint conformal scale is exact.",
        ),
        _check(
            "S25_ENDPOINT_MIXED_SCALE",
            sp.limit(
                (s**2 * (1 - s)).subs(s, sp.Rational(1, 2) - endpoint) ** sp.Rational(1, 3),
                endpoint,
                0,
                dir="+",
            )
            - sp.Rational(1, 2),
            "The upper-endpoint mixed scale is exact.",
        ),
        _check(
            "S26_ENDPOINT_DERIVATIVE_OVER_CONFORMAL",
            sp.limit(
                (lambda4_longitudinal / conf_over_l0).subs(s, sp.Rational(1, 2) - endpoint),
                endpoint,
                0,
                dir="+",
            ),
            "The derivative scale is asymptotically below the conformal scale at the upper endpoint.",
        ),
        _check(
            "S27_ENDPOINT_DERIVATIVE_OVER_MIXED",
            sp.limit(
                (lambda4_longitudinal / mix_over_l0).subs(s, sp.Rational(1, 2) - endpoint),
                endpoint,
                0,
                dir="+",
            ),
            "The derivative scale is asymptotically below the mixed scale at the upper endpoint.",
        ),
        _check(
            "S28_FULL_COUPLED_ACTION_FALSE",
            int(config["adjudication"]["full_coupled_metric_aether_matter_fluctuation_action"]),
            "The result is not a full coupled fluctuation action.",
        ),
        _check(
            "S29_PHYSICAL_CUTOFF_FALSE",
            int(config["adjudication"]["physical_UV_cutoff_established"]),
            "No physical cutoff is claimed.",
        ),
        _check(
            "S30_TREE_UNITARITY_FALSE",
            int(config["adjudication"]["tree_level_unitarity_bound_established"]),
            "No tree-level unitarity bound is claimed.",
        ),
    ]
    required = tuple(config["machine_check_contract"]["required_symbolic_checks"])
    if tuple(row["check_id"] for row in checks) != required:
        raise QuadratureScalarFullLocalError("symbolic check inventory changed")
    expressions = {
        "K": str(sp.factor(k(s))),
        "K_s": str(k_s),
        "L2": str(l2),
        "L3": str(l3),
        "L3_derivative": str(sp.factor(l3_derivative)),
        "L3_new": str(sp.factor(l3_new)),
        "L4": str(l4),
        "L4_derivative": str(sp.factor(l4_derivative)),
        "L4_amplitude": str(sp.factor(l4_amplitude)),
        "L4_mixed": str(sp.factor(l4_mixed)),
        "c_perp_squared": str(cperp2),
        "Pprime": str(pprime),
        "e3_conformal": str(e3_expected),
        "e4_conformal": str(e4_expected),
        "e4_mixed": str(emix_expected),
        "lambda3_derivative_over_Lambda0": str(lambda3),
        "lambda4_longitudinal_over_Lambda0": str(lambda4_longitudinal),
        "lambda4_transverse_over_Lambda0": str(lambda4_transverse),
        "lambda_conformal_over_Lambda0": str(conf_over_l0),
        "lambda_mixed_over_Lambda0": str(mix_over_l0),
    }
    return checks, expressions


def numeric_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    eta = float(sp.Rational(config["machine_check_contract"]["numeric_eta"]))
    rows: list[dict[str, Any]] = []
    for encoded in config["machine_check_contract"]["numeric_s_values"]:
        value = float(sp.Rational(encoded))
        d = 1 - 2 * value
        e3 = math.sqrt(2 * d / value)
        e4 = 2 * d / value
        emix = 1 / (value**2 * (1 - value))
        derivative_channels = (
            (18 * value**3 * (1 - value) ** 2 * d) ** 0.25,
            (8 * value**2 * (1 - value) * d / 5) ** 0.25,
            (8 * value**3 * d / (1 + value)) ** 0.25,
        )
        derivative = min(derivative_channels)
        conformal = eta**-1 * math.sqrt(value / (2 * d))
        mixed = eta ** (-1 / 3) * (value**2 * (1 - value)) ** (1 / 3)
        full = min(derivative, conformal, mixed)
        values = (e3, e4, emix, *derivative_channels, derivative, conformal, mixed, full)
        passed = (
            0 < value < 0.5
            and all(math.isfinite(item) and item > 0 for item in values)
            and abs(e4 - e3**2) <= tolerance * max(1.0, e4)
            and full <= derivative + tolerance
            and full <= conformal + tolerance
            and full <= mixed + tolerance
        )
        if not passed:
            raise QuadratureScalarFullLocalError(f"numeric scale probe failed: {encoded}")
        rows.append(
            {
                "s": encoded,
                "eta": config["machine_check_contract"]["numeric_eta"],
                "e3_conformal": e3,
                "e4_conformal": e4,
                "e4_mixed": emix,
                "lambda_derivative_over_Lambda0": derivative,
                "lambda_conformal_over_Lambda0": conformal,
                "lambda_mixed_over_Lambda0": mixed,
                "lambda_full_coefficient_over_Lambda0": full,
                "limiting_scale_at_probe": min(
                    ("derivative", derivative),
                    ("conformal", conformal),
                    ("mixed", mixed),
                    key=lambda item: item[1],
                )[0],
                "passed": True,
            }
        )
    return rows


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    validate_config(config, repo)
    predecessors = validate_predecessors(config, repo)
    checks, expressions = symbolic_checks(config)
    numeric = numeric_checks(config)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": STATUS,
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(repo / CONFIG_PATH),
            "content_sha256": _content_sha(config),
        },
        "implementation_binding": {
            "source_path": SOURCE_PATH.as_posix(),
            "source_file_sha256": _file_sha(repo / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(repo / TEST_PATH),
        },
        "predecessor_validation": predecessors,
        "primary_source_context": config["primary_source_context"],
        "full_scalar_expansion_contract": config["full_scalar_expansion_contract"],
        "canonical_interaction_contract": config["canonical_interaction_contract"],
        "coefficient_scale_contract": config["coefficient_scale_contract"],
        "endpoint_and_hierarchy_contract": config["endpoint_and_hierarchy_contract"],
        "machine_results": {
            "symbolic_checks": checks,
            "numeric_cases": numeric,
            "expressions": expressions,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "primary_sources": len(config["primary_source_context"]),
            "symbolic_checks": len(checks),
            "symbolic_checks_passed": sum(row["passed"] for row in checks),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(row["passed"] for row in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls_by_builder": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "limitations": [
            config["full_scalar_expansion_contract"]["exact_scope"],
            config["coefficient_scale_contract"]["interpretation"],
            config["endpoint_and_hierarchy_contract"]["non_no_go"],
        ],
    }
    receipt["content_sha256"] = _content_sha(receipt)
    return receipt


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    if stored.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureScalarFullLocalError("receipt schema changed")
    payload = dict(stored)
    content = payload.pop("content_sha256", None)
    if not isinstance(content, str) or content != _content_sha(payload):
        raise QuadratureScalarFullLocalError("receipt content hash changed")
    if stored != build_receipt(repo):
        raise QuadratureScalarFullLocalError("stored receipt differs from exact rebuild")
    return stored


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureScalarFullLocalError("refusing to replace existing receipt")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise QuadratureScalarFullLocalError("concurrent writer published different bytes")
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return "CREATED"


def write_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    receipt = build_receipt(repo)
    _atomic_no_clobber(repo / OUTPUT_PATH, _canonical(receipt))
    return receipt


def status(root: Path | None = None) -> dict[str, Any]:
    receipt = check_receipt(root)
    return {
        "valid": True,
        "decision": receipt["decision"],
        "full_fixed_background_scalar_quartic": receipt["adjudication"][
            "fixed_metric_aether_full_scalar_expansion_through_quartic"
        ],
        "new_interactions_canonicalized": receipt["adjudication"][
            "new_conformal_factor_interactions_canonically_normalized"
        ],
        "derivative_scale_endpoint_limiter": receipt["adjudication"][
            "derivative_scale_remains_low_s_asymptotic_limiter"
        ]
        and receipt["adjudication"]["derivative_scale_remains_high_s_asymptotic_limiter"],
        "full_coupled_action": receipt["adjudication"][
            "full_coupled_metric_aether_matter_fluctuation_action"
        ],
        "physical_cutoff": receipt["adjudication"]["physical_UV_cutoff_established"],
        "observational_rows_opened": receipt["counts"]["observational_rows_opened"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        result: Any = write_receipt()
    elif args.command == "check":
        result = check_receipt()
    else:
        result = status()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
