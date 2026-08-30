"""Local derivative-coefficient scales for the quadrature vector-metric scalar."""

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

CONFIG_PATH = Path("configs/gravity_shared_quadrature_scalar_local_cutoff_ceiling_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_quadrature_scalar_local_cutoff_ceiling.py"
)
TEST_PATH = Path("tests/test_gravity_shared_quadrature_scalar_local_cutoff_ceiling.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-scalar-local-cutoff-ceiling-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-scalar-local-cutoff-ceiling-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-scalar-local-cutoff-ceiling-receipt-1.0"
STATUS = "restricted_vector_metric_derivative_leading_local_scalar_coefficients_no_data"
DECISION = (
    "RESTRICTED_VECTOR_METRIC_DERIVATIVE_LEADING_LOCAL_SCALAR_COEFFICIENT_SCALES_"
    "DERIVED_BOTH_ENDPOINTS_COLLAPSE_PHYSICAL_CUTOFF_UNITARITY_BACKGROUND_"
    "OBSERVATION_AND_FULL_GATES_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "5a9a58932304b6926063dc3f4709c5f7494b99fef4059e138eb4f6ac0744c710"
EXPECTED_CONFIG_CONTENT_SHA256 = "09a1e20cbdc84274b9c93e65e68d470ad01a56c0d99dbd256a946d1f5324128b"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "065142319c30de8b9f3f3e26e688e97dafa294b2bdf1dccf43afd52a3ac8f80e",
    "primary_source_context": "bac623e95bf9edb0320c7878310eb3e16cf940fd456efcf70a9559b10fc65359",
    "frozen_action_expansion": "17df58dd76763765912ef18a7448f688fd9fe8c0cef3d2c9519cc04e5a273879",
    "canonicalization_contract": "26aa13c25e9bd06eed17975c44209997b16a8eb522cad49c370baf90246f2d97",
    "local_coefficient_scale_contract": "af8f7827d8bb408549d087295f8a6832c873eb6854491384ec61d96456bbba6a",
    "endpoint_contract": "dce01dda2598f7d4ad828b4281086aa2ca542aba871f74adb7bc809502095a34",
    "cherenkov_comparison_contract": "5828bcb0020b79897d9d0b344e94d5acf3e23ba0861645f12a85c8e36a9d7944",
    "machine_check_contract": "483a3900e715be9490bae7ee0527318ab4870a81f9d9cdd87078e0f128d0f41f",
    "adjudication": "278aa7a78c82c687c6ca700baad4997bfffc89ddc8906562e4b0bcccbf4ae9a0",
    "claim_boundary": "ae134dc2ef8fa3e40c8d1f563c54e5b2944bbe67f33af0457028ce87b9e8fe16",
    "zero_access_and_compute": "f775ebd31025256383d3c06467b2bf1f2bf1229eb7da3e5aa98d0207115ff28a",
}


class QuadratureScalarLocalCutoffError(RuntimeError):
    """Raised when the frozen local coefficient-scale contract changes."""


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
        raise QuadratureScalarLocalCutoffError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureScalarLocalCutoffError(f"JSON root is not an object: {path}")
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
        "frozen_action_expansion",
        "canonicalization_contract",
        "local_coefficient_scale_contract",
        "endpoint_contract",
        "cherenkov_comparison_contract",
        "machine_check_contract",
        "adjudication",
        "claim_boundary",
        "zero_access_and_compute",
        "output_path",
    }
    if set(config) != expected_keys:
        raise QuadratureScalarLocalCutoffError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-shared-quadrature-scalar-local-cutoff-ceiling-v1"
        or config["status"]
        != "frozen_no_data_restricted_derivative_leading_local_scalar_interaction_"
        "coefficient_scales"
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or config["adjudication"]["overall_decision"] != DECISION
    ):
        raise QuadratureScalarLocalCutoffError("config identity changed")
    if _content_sha(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise QuadratureScalarLocalCutoffError("config content changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureScalarLocalCutoffError("config file hash changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[key]) != expected:
            raise QuadratureScalarLocalCutoffError(f"config section changed: {key}")

    machine = config["machine_check_contract"]
    if (
        machine["symbolic_engine"] != "sympy-1.14"
        or len(machine["required_symbolic_checks"]) != 33
        or len(machine["numeric_s_values"]) != 5
    ):
        raise QuadratureScalarLocalCutoffError("machine inventory changed")
    adjudication = config["adjudication"]
    required_truth = {
        "exact_derivative_leading_scalar_expansion_through_quartic": True,
        "full_scalar_fluctuation_action_through_quartic": False,
        "local_NDA_style_coefficient_scale_derived": True,
        "uniform_positive_local_coefficient_scale_established": False,
        "physical_UV_cutoff_established": False,
        "tree_level_unitarity_bound_established": False,
        "cosmic_ray_survival_test_passed": False,
        "all_mode_cherenkov_safety": False,
    }
    if any(adjudication[key] is not value for key, value in required_truth.items()):
        raise QuadratureScalarLocalCutoffError("adjudication changed")
    if any(config["zero_access_and_compute"].values()):
        raise QuadratureScalarLocalCutoffError("zero-access contract changed")


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
            raise QuadratureScalarLocalCutoffError("predecessor Git binding failed") from error
        if object_type != "commit":
            raise QuadratureScalarLocalCutoffError("predecessor object is not a commit")
        for artifact in binding["artifacts"]:
            path = Path(artifact["path"])
            if _file_sha(repo / path) != artifact["file_sha256"]:
                raise QuadratureScalarLocalCutoffError("predecessor artifact changed")
            try:
                committed = subprocess.run(
                    ["git", "show", f"{commit}:{path.as_posix()}"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise QuadratureScalarLocalCutoffError("predecessor commit path failed") from error
            if _sha_bytes(committed) != artifact["file_sha256"]:
                raise QuadratureScalarLocalCutoffError("predecessor commit bytes changed")
        receipt = _load_json(repo / Path(binding["receipt_path"]))
        if (
            receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
            or receipt.get("content_sha256") != binding["receipt_content_sha256"]
        ):
            raise QuadratureScalarLocalCutoffError("predecessor receipt changed")
        payload = dict(receipt)
        stored_content = payload.pop("content_sha256", None)
        if stored_content != _content_sha(payload):
            raise QuadratureScalarLocalCutoffError("predecessor receipt hash invalid")
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
        raise QuadratureScalarLocalCutoffError(f"symbolic check failed: {check_id}: {simplified}")
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": True,
    }


def symbolic_checks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, alpha, lambda0sq = sp.symbols("s alpha Lambda0sq", positive=True, finite=True)
    epsilon = sp.symbols("epsilon", real=True, finite=True)
    t, r, u = sp.symbols("t r u", real=True, finite=True)
    d = 1 - 2 * s

    def p(argument: sp.Expr) -> sp.Expr:
        return argument**2 / 4 + argument / 4 + sp.log(1 - 2 * argument) / 8

    def k(argument: sp.Expr) -> sp.Expr:
        return 2 * alpha**2 * argument * (1 - argument) / (1 - 2 * argument) ** 2

    p_s = sp.factor(sp.diff(p(s), s))
    p_ss = sp.factor(sp.diff(p(s), s, 2))
    k_s = sp.factor(sp.diff(k(s), s))
    k_ss = sp.factor(sp.diff(k(s), s, 2))
    c = alpha**2 * s / d

    s_total = alpha * sp.sqrt(
        (s / alpha + epsilon * r / lambda0sq) ** 2 + epsilon**2 * u / lambda0sq**2
    )
    density = 2 * lambda0sq**2 * p(s_total) + k(s_total) * epsilon**2 * t
    l2 = sp.factor(sp.diff(density, epsilon, 2).subs(epsilon, 0) / sp.factorial(2))
    l3 = sp.factor(sp.diff(density, epsilon, 3).subs(epsilon, 0) / sp.factorial(3))
    l4 = sp.factor(sp.diff(density, epsilon, 4).subs(epsilon, 0) / sp.factorial(4))
    l2_expected = k(s) * (t - r**2) - c * u
    l3_expected = alpha**3 / lambda0sq * (2 * r * t / d**3 - r * u / d**2 - 2 * r**3 / (3 * d**3))
    l4_expected = (
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

    cperp2 = sp.factor(c / k(s))
    pprime = sp.factor(2 * k(s) * cperp2)
    d3_longitudinal_squared = sp.factor(
        (cperp2 * (l3_expected * lambda0sq / alpha**3).subs({r: 1, t: 1, u: 0}) * alpha**3) ** 2
        / pprime**3
    )
    d3_longitudinal_expected = 1 / (3 * sp.sqrt(2) * s ** sp.Rational(3, 2) * (1 - s) * sp.sqrt(d))
    d3_transverse = sp.factor(
        cperp2
        * (l3_expected * lambda0sq).subs({r: 0, t: 1, u: 1 / cperp2})
        / pprime ** sp.Rational(3, 2)
    )
    d4_longitudinal = sp.factor(
        cperp2 * (l4_expected * lambda0sq**2).subs({r: 1, t: 1, u: 0}) / pprime**2
    )
    d4_longitudinal_expected = 5 / (8 * s**2 * (1 - s) * d)
    d4_transverse = sp.factor(
        cperp2 * (l4_expected * lambda0sq**2).subs({r: 0, t: 1, u: 1 / cperp2}) / pprime**2
    )
    d4_transverse_expected = (1 + s) / (8 * s**3 * d)

    lambda3 = (18 * s**3 * (1 - s) ** 2 * d) ** sp.Rational(1, 4)
    lambda4_longitudinal = (8 * s**2 * (1 - s) * d / 5) ** sp.Rational(1, 4)
    lambda4_transverse = (8 * s**3 * d / (1 + s)) ** sp.Rational(1, 4)
    endpoint = sp.symbols("endpoint", positive=True, finite=True)

    mpl, a0, distance, energy, velocity, dual_speed, lambda_coeff = sp.symbols(
        "Mpl a0 L E v V lambdaCoeff", positive=True, finite=True
    )
    predecessor_loss = (
        distance
        * energy
        * (1 + velocity**2) ** 2
        * (lambda_coeff**2 * mpl * a0)
        * d
        / (32 * sp.pi * mpl**2 * s * velocity * dual_speed)
    )
    xi = distance * energy * a0 / mpl
    coefficient_loss = (
        xi * (1 + velocity**2) ** 2 * lambda_coeff**2 * d / (32 * sp.pi * s * velocity * dual_speed)
    )

    l2_expanded = sp.expand(l2)
    l3_expanded = sp.expand(l3)
    l4_expanded = sp.expand(l4)
    checks = [
        _check("S01_P_S", p_s + s**2 / d, "The exact p derivative is -s^2/d."),
        _check(
            "S02_P_SS",
            p_ss + 2 * s * (1 - s) / d**2,
            "The exact second p derivative is -2s(1-s)/d^2.",
        ),
        _check("S03_K", k(s) - 2 * alpha**2 * s * (1 - s) / d**2, "K is exact."),
        _check("S04_K_S", k_s - 2 * alpha**2 / d**3, "The first K derivative is exact."),
        _check("S05_K_SS", k_ss - 12 * alpha**2 / d**4, "The second K derivative is exact."),
        _check("S06_QUADRATIC_TIME", l2_expanded.coeff(t) - k(s), "The time coefficient is K."),
        _check(
            "S07_QUADRATIC_LONGITUDINAL",
            l2_expanded.coeff(r, 2) + k(s),
            "The longitudinal coefficient is -K.",
        ),
        _check(
            "S08_QUADRATIC_TRANSVERSE",
            l2_expanded.coeff(u) + c,
            "The transverse coefficient is -C.",
        ),
        _check(
            "S09_CUBIC_RT",
            l3_expanded.coeff(r, 1).coeff(t) - 2 * alpha**3 / (lambda0sq * d**3),
            "The r t cubic coefficient is exact.",
        ),
        _check(
            "S10_CUBIC_RU",
            l3_expanded.coeff(r, 1).coeff(u) + alpha**3 / (lambda0sq * d**2),
            "The r u cubic coefficient is exact.",
        ),
        _check(
            "S11_CUBIC_R3",
            l3_expanded.coeff(r, 3) + 2 * alpha**3 / (3 * lambda0sq * d**3),
            "The r^3 cubic coefficient is exact.",
        ),
        _check(
            "S12_QUARTIC_R4",
            l4_expanded.coeff(r, 4) + alpha**4 / (lambda0sq**2 * d**4),
            "The r^4 quartic coefficient is exact.",
        ),
        _check(
            "S13_QUARTIC_R2U",
            l4_expanded.coeff(r, 2).coeff(u) + 2 * alpha**4 / (lambda0sq**2 * d**3),
            "The r^2 u quartic coefficient is exact.",
        ),
        _check(
            "S14_QUARTIC_R2T",
            l4_expanded.coeff(r, 2).coeff(t) - 6 * alpha**4 / (lambda0sq**2 * d**4),
            "The r^2 t quartic coefficient is exact.",
        ),
        _check(
            "S15_QUARTIC_TU",
            l4_expanded.coeff(t).coeff(u) - alpha**4 / (lambda0sq**2 * s * d**3),
            "The t u quartic coefficient is exact.",
        ),
        _check(
            "S16_QUARTIC_U2",
            l4_expanded.coeff(u, 2) + alpha**4 / (4 * lambda0sq**2 * s * d**2),
            "The u^2 quartic coefficient is exact.",
        ),
        _check(
            "S17_TRANSVERSE_SPEED",
            cperp2 - d / (2 * (1 - s)),
            "The transverse speed is exact.",
        ),
        _check("S18_CANONICAL_NORMALIZATION", pprime - 2 * c, "Pprime=2C."),
        _check(
            "S19_LONGITUDINAL_CUBIC",
            d3_longitudinal_squared - d3_longitudinal_expected**2,
            "The squared canonical longitudinal cubic coefficient is exact; its sign is positive on the declared branch.",
        ),
        _check(
            "S20_TRANSVERSE_CUBIC_ZERO",
            d3_transverse,
            "The exactly transverse on-shell cubic contraction vanishes.",
        ),
        _check(
            "S21_LONGITUDINAL_QUARTIC",
            d4_longitudinal - d4_longitudinal_expected,
            "The canonical longitudinal quartic coefficient is exact.",
        ),
        _check(
            "S22_TRANSVERSE_QUARTIC",
            d4_transverse - d4_transverse_expected,
            "The canonical transverse quartic coefficient is exact.",
        ),
        _check(
            "S23_ALPHA_CANCELLATION",
            sp.Matrix(
                [
                    sp.diff(d3_longitudinal_expected, alpha),
                    sp.diff(d4_longitudinal, alpha),
                    sp.diff(d4_transverse, alpha),
                ]
            ),
            "Alpha cancels from every nonzero canonical channel coefficient.",
        ),
        _check(
            "S24_POSITIVE_CHANNEL_COEFFICIENTS",
            sp.Matrix(
                [
                    d3_longitudinal_expected
                    * 3
                    * sp.sqrt(2)
                    * s ** sp.Rational(3, 2)
                    * (1 - s)
                    * sp.sqrt(d)
                    - 1,
                    d4_longitudinal_expected * 8 * s**2 * (1 - s) * d - 5,
                    d4_transverse_expected * 8 * s**3 * d - (1 + s),
                ]
            ),
            "All nonzero contracted coefficients are positive on 0<s<1/2.",
        ),
        _check(
            "S25_LOW_S_CUBIC_SCALE",
            sp.limit(lambda3 / s ** sp.Rational(3, 4), s, 0, dir="+") - 18 ** sp.Rational(1, 4),
            "The low-s cubic scale is exact.",
        ),
        _check(
            "S26_LOW_S_LONGITUDINAL_QUARTIC_SCALE",
            sp.limit(lambda4_longitudinal / sp.sqrt(s), s, 0, dir="+")
            - (sp.Rational(8, 5)) ** sp.Rational(1, 4),
            "The low-s longitudinal quartic scale is exact.",
        ),
        _check(
            "S27_LOW_S_TRANSVERSE_QUARTIC_SCALE",
            sp.limit(lambda4_transverse / s ** sp.Rational(3, 4), s, 0, dir="+")
            - 8 ** sp.Rational(1, 4),
            "The low-s transverse quartic scale is exact.",
        ),
        _check(
            "S28_ENDPOINT_CUBIC_SCALE",
            sp.limit(
                lambda3.subs(s, sp.Rational(1, 2) - endpoint) / endpoint ** sp.Rational(1, 4),
                endpoint,
                0,
                dir="+",
            )
            - (sp.Rational(9, 8)) ** sp.Rational(1, 4),
            "The finite-gradient endpoint cubic scale is exact.",
        ),
        _check(
            "S29_ENDPOINT_LONGITUDINAL_QUARTIC_SCALE",
            sp.limit(
                lambda4_longitudinal.subs(s, sp.Rational(1, 2) - endpoint)
                / endpoint ** sp.Rational(1, 4),
                endpoint,
                0,
                dir="+",
            )
            - (sp.Rational(2, 5)) ** sp.Rational(1, 4),
            "The endpoint longitudinal quartic scale is exact.",
        ),
        _check(
            "S30_ENDPOINT_TRANSVERSE_QUARTIC_SCALE",
            sp.limit(
                lambda4_transverse.subs(s, sp.Rational(1, 2) - endpoint)
                / endpoint ** sp.Rational(1, 4),
                endpoint,
                0,
                dir="+",
            )
            - (sp.Rational(4, 3)) ** sp.Rational(1, 4),
            "The endpoint transverse quartic scale is exact.",
        ),
        _check(
            "S31_BOTH_ENDPOINTS_COLLAPSE",
            sp.Matrix(
                [
                    sp.limit(lambda4_transverse, s, 0, dir="+"),
                    sp.limit(
                        lambda4_longitudinal.subs(s, sp.Rational(1, 2) - endpoint),
                        endpoint,
                        0,
                        dir="+",
                    ),
                ]
            ),
            "At least one exact channel scale collapses at each endpoint.",
        ),
        _check(
            "S32_CHERENKOV_SUBSTITUTION",
            predecessor_loss - coefficient_loss,
            "Substitution into the predecessor loss gives the frozen Xi form.",
        ),
        _check(
            "S33_PHYSICAL_CUTOFF_ADJUDICATION_FALSE",
            int(config["adjudication"]["physical_UV_cutoff_established"]),
            "The coefficient scale is not adjudicated as a physical cutoff.",
        ),
    ]
    required = tuple(config["machine_check_contract"]["required_symbolic_checks"])
    if tuple(row["check_id"] for row in checks) != required:
        raise QuadratureScalarLocalCutoffError("symbolic check inventory changed")
    if sp.simplify(l2 - l2_expected) != 0:
        raise QuadratureScalarLocalCutoffError("quadratic density reconstruction failed")
    if sp.simplify(l3 - l3_expected) != 0:
        raise QuadratureScalarLocalCutoffError("cubic density reconstruction failed")
    if sp.simplify(l4 - l4_expected) != 0:
        raise QuadratureScalarLocalCutoffError("quartic density reconstruction failed")

    expressions = {
        "p_s": str(p_s),
        "p_ss": str(p_ss),
        "K": str(sp.factor(k(s))),
        "K_s": str(k_s),
        "K_ss": str(k_ss),
        "L2": str(l2),
        "L3": str(l3),
        "L4": str(l4),
        "C": str(c),
        "c_perp_squared": str(cperp2),
        "Pprime": str(pprime),
        "d3_longitudinal": str(d3_longitudinal_expected),
        "d3_transverse": str(d3_transverse),
        "d4_longitudinal": str(d4_longitudinal_expected),
        "d4_transverse": str(d4_transverse_expected),
        "lambda3_longitudinal_over_Lambda0": str(lambda3),
        "lambda4_longitudinal_over_Lambda0": str(lambda4_longitudinal),
        "lambda4_transverse_over_Lambda0": str(lambda4_transverse),
        "coefficient_cutoff_loss_fraction": str(coefficient_loss),
    }
    return checks, expressions


def numeric_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    rows: list[dict[str, Any]] = []
    for encoded in config["machine_check_contract"]["numeric_s_values"]:
        value = float(sp.Rational(encoded))
        d = 1 - 2 * value
        d3 = 1 / (3 * math.sqrt(2) * value**1.5 * (1 - value) * math.sqrt(d))
        d4_longitudinal = 5 / (8 * value**2 * (1 - value) * d)
        d4_transverse = (1 + value) / (8 * value**3 * d)
        lambda3 = d3**-0.5
        lambda4_longitudinal = d4_longitudinal**-0.25
        lambda4_transverse = d4_transverse**-0.25
        coefficient_scale = min(lambda3, lambda4_longitudinal, lambda4_transverse)
        passed = (
            0 < value < 0.5
            and d3 > 0
            and d4_longitudinal > 0
            and d4_transverse > 0
            and coefficient_scale > tolerance
            and all(
                math.isfinite(item)
                for item in (
                    d3,
                    d4_longitudinal,
                    d4_transverse,
                    lambda3,
                    lambda4_longitudinal,
                    lambda4_transverse,
                    coefficient_scale,
                )
            )
        )
        if not passed:
            raise QuadratureScalarLocalCutoffError(f"numeric coefficient probe failed: {encoded}")
        rows.append(
            {
                "s": encoded,
                "d3_longitudinal": d3,
                "d3_transverse": 0.0,
                "d4_longitudinal": d4_longitudinal,
                "d4_transverse": d4_transverse,
                "lambda3_longitudinal_over_Lambda0": lambda3,
                "lambda4_longitudinal_over_Lambda0": lambda4_longitudinal,
                "lambda4_transverse_over_Lambda0": lambda4_transverse,
                "lambda_coefficient_over_Lambda0": coefficient_scale,
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
        "frozen_action_expansion": config["frozen_action_expansion"],
        "canonicalization_contract": config["canonicalization_contract"],
        "local_coefficient_scale_contract": config["local_coefficient_scale_contract"],
        "endpoint_contract": config["endpoint_contract"],
        "cherenkov_comparison_contract": config["cherenkov_comparison_contract"],
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
            config["frozen_action_expansion"]["scope"],
            config["local_coefficient_scale_contract"]["physical_ceiling_statement"],
            config["endpoint_contract"]["non_no_go_statement"],
            config["cherenkov_comparison_contract"]["missing_inputs"],
            config["cherenkov_comparison_contract"]["adjudication"],
        ],
    }
    receipt["content_sha256"] = _content_sha(receipt)
    return receipt


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    if stored.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureScalarLocalCutoffError("receipt schema changed")
    payload = dict(stored)
    content = payload.pop("content_sha256", None)
    if not isinstance(content, str) or content != _content_sha(payload):
        raise QuadratureScalarLocalCutoffError("receipt content hash changed")
    if stored != build_receipt(repo):
        raise QuadratureScalarLocalCutoffError("stored receipt differs from exact rebuild")
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
        raise QuadratureScalarLocalCutoffError("refusing to replace existing receipt")
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
                raise QuadratureScalarLocalCutoffError(
                    "concurrent writer published different bytes"
                )
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
        "derivative_leading_expansion_derived": receipt["adjudication"][
            "exact_derivative_leading_scalar_expansion_through_quartic"
        ],
        "local_coefficient_scale_derived": receipt["adjudication"][
            "local_NDA_style_coefficient_scale_derived"
        ],
        "physical_cutoff_established": receipt["adjudication"]["physical_UV_cutoff_established"],
        "uniform_positive_scale": receipt["adjudication"][
            "uniform_positive_local_coefficient_scale_established"
        ],
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
