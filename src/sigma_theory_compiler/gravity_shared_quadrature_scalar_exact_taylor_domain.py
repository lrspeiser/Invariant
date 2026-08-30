"""Exact local scalar-jet Taylor domain for the quadrature vector-metric action."""

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

CONFIG_PATH = Path("configs/gravity_shared_quadrature_scalar_exact_taylor_domain_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_quadrature_scalar_exact_taylor_domain.py"
)
TEST_PATH = Path("tests/test_gravity_shared_quadrature_scalar_exact_taylor_domain.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-scalar-exact-taylor-domain-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-scalar-exact-taylor-domain-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-scalar-exact-taylor-domain-receipt-1.0"
STATUS = "restricted_fixed_background_exact_scalar_jet_taylor_domain_no_data"
DECISION = (
    "RESTRICTED_FIXED_BACKGROUND_EXACT_SCALAR_JET_TAYLOR_DOMAIN_DERIVED_BOTH_"
    "CANONICAL_ENDPOINT_RADII_COLLAPSE_NOT_A_PHYSICAL_CUTOFF_UNITARITY_OR_FULL_"
    "COUPLED_HEALTH_RESULT"
)
EXPECTED_CONFIG_FILE_SHA256 = "a34a8faae6a72410fedadbe7ea7c195138fe43dc9fd104d82b950b3bbe78be9d"
EXPECTED_CONFIG_CONTENT_SHA256 = "9757c02246953b60eca7493a6ac40be9ff060b19a8a416eda4f7f7237b06ab67"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "da5e3e9d8a2136885eb6e185fe7cd5e1efcd8d167a630fa63b4c473dc4970621",
    "exact_jet_domain_contract": "6efd265b8a5d43999eadd93ee6692b7fcce8fdcb803528d67e91346eed10cd4e",
    "canonical_radius_contract": "11970264d1c90298d2c4044a9aa5a7fbb4a08d26dfcbadf47cbfba8cc7a9bb5f",
    "endpoint_and_quartic_consistency_contract": "54f4d111b4cbdd05aa5a4490c32629a02472ada4ca2a9d09c992f06e4df0feba",
    "machine_check_contract": "55a88f71539735130ca3dbf2d85a8662abf8eafb56b5a800d24cd4076069a2ef",
    "adjudication": "c7bcc92c2617486af9088a936c36ef01eeaa2c946061ffedfc62df37893a08ed",
    "claim_boundary": "1091ad532b720cd3da534797820c5d50c6deb53f20c8da22a615cc27dc4db151",
    "zero_access_and_compute": "f775ebd31025256383d3c06467b2bf1f2bf1229eb7da3e5aa98d0207115ff28a",
}


class QuadratureScalarTaylorDomainError(RuntimeError):
    """Raised when the exact Taylor-domain contract or evidence changes."""


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
        raise QuadratureScalarTaylorDomainError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureScalarTaylorDomainError(f"JSON root is not an object: {path}")
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
        "exact_jet_domain_contract",
        "canonical_radius_contract",
        "endpoint_and_quartic_consistency_contract",
        "machine_check_contract",
        "adjudication",
        "claim_boundary",
        "zero_access_and_compute",
        "output_path",
    }
    if set(config) != expected_keys:
        raise QuadratureScalarTaylorDomainError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-shared-quadrature-scalar-exact-taylor-domain-v1"
        or config["status"] != "frozen_no_data_fixed_background_exact_scalar_jet_taylor_domain"
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or config["adjudication"]["overall_decision"] != DECISION
    ):
        raise QuadratureScalarTaylorDomainError("config identity changed")
    if _content_sha(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise QuadratureScalarTaylorDomainError("config content changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureScalarTaylorDomainError("config file hash changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[key]) != expected:
            raise QuadratureScalarTaylorDomainError(f"config section changed: {key}")
    machine = config["machine_check_contract"]
    if (
        machine["symbolic_engine"] != "sympy-1.14"
        or len(machine["required_symbolic_checks"]) != 25
        or len(machine["numeric_s_values"]) != 5
    ):
        raise QuadratureScalarTaylorDomainError("machine inventory changed")
    truth = {
        "exact_fixed_background_real_scalar_jet_domain_derived": True,
        "exact_longitudinal_local_Taylor_radius_derived": True,
        "exact_transverse_local_Taylor_radius_derived": True,
        "endpoint_collapse_is_quartic_truncation_artifact": False,
        "full_coupled_metric_aether_matter_analyticity_domain": False,
        "physical_UV_cutoff_established": False,
        "tree_level_unitarity_bound_established": False,
        "strong_coupling_theorem_established": False,
    }
    if any(config["adjudication"][key] is not value for key, value in truth.items()):
        raise QuadratureScalarTaylorDomainError("adjudication changed")
    if any(config["zero_access_and_compute"].values()):
        raise QuadratureScalarTaylorDomainError("zero-access contract changed")


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
            raise QuadratureScalarTaylorDomainError("predecessor Git binding failed") from error
        if object_type != "commit":
            raise QuadratureScalarTaylorDomainError("predecessor object is not a commit")
        for artifact in binding["artifacts"]:
            path = Path(artifact["path"])
            if _file_sha(repo / path) != artifact["file_sha256"]:
                raise QuadratureScalarTaylorDomainError("predecessor artifact changed")
            try:
                committed = subprocess.run(
                    ["git", "show", f"{commit}:{path.as_posix()}"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise QuadratureScalarTaylorDomainError("predecessor commit path failed") from error
            if _sha_bytes(committed) != artifact["file_sha256"]:
                raise QuadratureScalarTaylorDomainError("predecessor commit bytes changed")
        receipt = _load_json(repo / Path(binding["receipt_path"]))
        if (
            receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
            or receipt.get("content_sha256") != binding["receipt_content_sha256"]
        ):
            raise QuadratureScalarTaylorDomainError("predecessor receipt changed")
        payload = dict(receipt)
        stored_content = payload.pop("content_sha256", None)
        if stored_content != _content_sha(payload):
            raise QuadratureScalarTaylorDomainError("predecessor receipt hash invalid")
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
        raise QuadratureScalarTaylorDomainError(f"symbolic check failed: {check_id}: {simplified}")
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": True,
    }


def symbolic_checks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, alpha, mpl = sp.symbols("s alpha Mpl", positive=True, finite=True)
    rho, upsilon, ql, qt2, z = sp.symbols("rho upsilon qL qT2 z", real=True, finite=True)
    d = 1 - 2 * s
    argument_squared = (s + rho) ** 2 + upsilon
    pprime = 2 * alpha**2 * s / d
    cperp2 = d / (2 * (1 - s))
    ql_factor_squared = sp.factor(pprime / alpha**2)
    qt2_factor = sp.factor(cperp2 * pprime / alpha**2)
    rho_radius_low = s
    rho_radius_high = sp.Rational(1, 2) - s
    crossover = 1 / (2 * sp.sqrt(2))
    rl_low = sp.sqrt(2 * s / d) * rho_radius_low
    rl_high = sp.sqrt(2 * s / d) * rho_radius_high
    rt_low = s ** sp.Rational(3, 2) / sp.sqrt(1 - s)
    rt_high = sp.sqrt(s * (sp.Rational(1, 4) - s**2) / (1 - s))
    rt_real = rt_high
    endpoint = sp.symbols("endpoint", positive=True, finite=True)
    lambda4_transverse = (8 * s**3 * d / (1 + s)) ** sp.Rational(1, 4)
    lambda4_longitudinal = (8 * s**2 * (1 - s) * d / 5) ** sp.Rational(1, 4)
    amplitude_factor = sp.exp(-4 * alpha * z / mpl)
    canonical_ellipse = (s + ql * sp.sqrt(d / (2 * s))) ** 2 + (1 - s) * qt2 / s

    checks = [
        _check(
            "S01_DIMENSIONLESS_JET_ARGUMENT",
            argument_squared - ((s + rho) ** 2 + upsilon),
            "The exact squared jet argument is reconstructed.",
        ),
        _check(
            "S02_REAL_BRANCH_ELLIPSE",
            sp.powdenest(
                canonical_ellipse.subs(
                    {
                        ql: sp.sqrt(2 * s / d) * rho,
                        qt2: s * upsilon / (1 - s),
                    }
                )
                - argument_squared,
                force=True,
            ),
            "The exact real branch is the shifted canonical ellipse.",
        ),
        _check(
            "S03_LONGITUDINAL_ZERO_GRADIENT_BOUNDARY",
            argument_squared.subs({rho: -s, upsilon: 0}),
            "The negative longitudinal boundary reaches the non-differentiable zero norm.",
        ),
        _check(
            "S04_LONGITUDINAL_POLE_BOUNDARY",
            argument_squared.subs({rho: sp.Rational(1, 2) - s, upsilon: 0}) - sp.Rational(1, 4),
            "The positive longitudinal boundary reaches s_total=1/2.",
        ),
        _check(
            "S05_LONGITUDINAL_TAYLOR_RADIUS",
            sp.Matrix(
                [
                    s - rho_radius_low,
                    (sp.Rational(1, 2) - s) - rho_radius_high,
                ]
            ),
            "The two longitudinal distances defining min(s,1/2-s) are exact.",
        ),
        _check(
            "S06_TRANSVERSE_SQRT_BRANCH",
            argument_squared.subs({rho: 0, upsilon: -(s**2)}),
            "The negative-upsilon square-root branch point is exact.",
        ),
        _check(
            "S07_TRANSVERSE_POLE",
            argument_squared.subs({rho: 0, upsilon: sp.Rational(1, 4) - s**2}) - sp.Rational(1, 4),
            "The positive-upsilon p/K pole is exact.",
        ),
        _check(
            "S08_TRANSVERSE_CROSSOVER",
            crossover**2 - (sp.Rational(1, 4) - crossover**2),
            "The transverse nearest-singularity crossover is exact.",
        ),
        _check(
            "S09_CANONICAL_LONGITUDINAL_MAP",
            ql_factor_squared - 2 * s / d,
            "The squared canonical longitudinal jet map is exact.",
        ),
        _check(
            "S10_CANONICAL_TRANSVERSE_MAP",
            qt2_factor - s / (1 - s),
            "The canonical transverse-squared jet map is exact.",
        ),
        _check(
            "S11_CANONICAL_LONGITUDINAL_RADIUS",
            sp.Matrix(
                [rl_low**2 - 2 * s**3 / d, rl_high**2 - 2 * s * (sp.Rational(1, 2) - s) ** 2 / d]
            ),
            "Both canonical longitudinal radius branches are exact.",
        ),
        _check(
            "S12_CANONICAL_TRANSVERSE_LOW_BRANCH",
            rt_low**2 - s**3 / (1 - s),
            "The low-s canonical transverse Taylor radius is exact.",
        ),
        _check(
            "S13_CANONICAL_TRANSVERSE_HIGH_BRANCH",
            rt_high**2 - s * (sp.Rational(1, 4) - s**2) / (1 - s),
            "The high-s canonical transverse Taylor radius is exact.",
        ),
        _check(
            "S14_REAL_TRANSVERSE_POLE_RADIUS",
            rt_real**2 - s * (sp.Rational(1, 4) - s**2) / (1 - s),
            "The real transverse p/K pole radius is exact.",
        ),
        _check(
            "S15_AMPLITUDE_FACTOR_ENTIRE",
            sp.diff(amplitude_factor, z, 5) - (-4 * alpha / mpl) ** 5 * amplitude_factor,
            "The amplitude factor has finite derivatives of every displayed order and no finite singularity.",
        ),
        _check(
            "S16_LOW_S_LONGITUDINAL_RADIUS",
            sp.limit(rl_low / s ** sp.Rational(3, 2), s, 0, dir="+") - sp.sqrt(2),
            "The low-s longitudinal radius asymptote is exact.",
        ),
        _check(
            "S17_LOW_S_TRANSVERSE_RADIUS",
            sp.limit(rt_low / s ** sp.Rational(3, 2), s, 0, dir="+") - 1,
            "The low-s transverse radius asymptote is exact.",
        ),
        _check(
            "S18_HIGH_S_LONGITUDINAL_RADIUS",
            sp.limit(
                rl_high.subs(s, sp.Rational(1, 2) - endpoint) / sp.sqrt(endpoint),
                endpoint,
                0,
                dir="+",
            )
            - 1 / sp.sqrt(2),
            "The upper-endpoint longitudinal radius asymptote is exact.",
        ),
        _check(
            "S19_HIGH_S_TRANSVERSE_RADIUS",
            sp.limit(
                rt_high.subs(s, sp.Rational(1, 2) - endpoint) / sp.sqrt(endpoint),
                endpoint,
                0,
                dir="+",
            )
            - 1,
            "The upper-endpoint transverse radius asymptote is exact.",
        ),
        _check(
            "S20_LOW_S_QUARTIC_RADIUS_CONSISTENCY",
            sp.limit(lambda4_transverse**2 / rt_low, s, 0, dir="+") - 2 * sp.sqrt(2),
            "The low-s quartic scale squared matches the exact-radius power law.",
        ),
        _check(
            "S21_HIGH_S_QUARTIC_RADIUS_CONSISTENCY",
            sp.limit(
                (lambda4_longitudinal**2 / rl_high).subs(s, sp.Rational(1, 2) - endpoint),
                endpoint,
                0,
                dir="+",
            )
            - 2 / sp.sqrt(5),
            "The upper-endpoint quartic scale squared matches the exact-radius power law.",
        ),
        _check(
            "S22_BOTH_ENDPOINTS_COLLAPSE",
            sp.Matrix(
                [
                    sp.limit(rl_low, s, 0, dir="+"),
                    sp.limit(rt_low, s, 0, dir="+"),
                    sp.limit(rl_high.subs(s, sp.Rational(1, 2) - endpoint), endpoint, 0, dir="+"),
                    sp.limit(rt_high.subs(s, sp.Rational(1, 2) - endpoint), endpoint, 0, dir="+"),
                ]
            ),
            "Both canonical radii collapse at both branch endpoints.",
        ),
        _check(
            "S23_FULL_COUPLED_DOMAIN_FALSE",
            int(config["adjudication"]["full_coupled_metric_aether_matter_analyticity_domain"]),
            "No full coupled analyticity domain is claimed.",
        ),
        _check(
            "S24_PHYSICAL_CUTOFF_FALSE",
            int(config["adjudication"]["physical_UV_cutoff_established"]),
            "No physical cutoff is claimed.",
        ),
        _check(
            "S25_TREE_UNITARITY_FALSE",
            int(config["adjudication"]["tree_level_unitarity_bound_established"]),
            "No tree-level unitarity bound is claimed.",
        ),
    ]
    required = tuple(config["machine_check_contract"]["required_symbolic_checks"])
    if tuple(row["check_id"] for row in checks) != required:
        raise QuadratureScalarTaylorDomainError("symbolic check inventory changed")
    expressions = {
        "s_total_squared": str(argument_squared),
        "canonical_real_branch_ellipse": str(canonical_ellipse),
        "rho_zero_gradient_singularity": str(-s),
        "rho_outer_pole_singularity": str(sp.Rational(1, 2) - s),
        "rho_taylor_radius": "Min(s, 1/2 - s)",
        "upsilon_sqrt_branch": str(-(s**2)),
        "upsilon_outer_pole": str(sp.Rational(1, 4) - s**2),
        "upsilon_taylor_radius": "Min(s**2, 1/4 - s**2)",
        "transverse_crossover": str(crossover),
        "qL_map_squared": str(ql_factor_squared),
        "qT2_map": str(qt2_factor),
        "RL_low": str(rl_low),
        "RL_high": str(rl_high),
        "RT_low": str(rt_low),
        "RT_high": str(rt_high),
        "RT_real_pole": str(rt_real),
        "amplitude_factor": str(amplitude_factor),
    }
    return checks, expressions


def numeric_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    crossover = 1 / (2 * math.sqrt(2))
    rows: list[dict[str, Any]] = []
    for encoded in config["machine_check_contract"]["numeric_s_values"]:
        value = float(sp.Rational(encoded))
        d = 1 - 2 * value
        rho_radius = min(value, 0.5 - value)
        upsilon_sqrt = value**2
        upsilon_pole = 0.25 - value**2
        upsilon_radius = min(upsilon_sqrt, upsilon_pole)
        ql_radius = math.sqrt(2 * value / d) * rho_radius
        qt2_radius = value / (1 - value) * upsilon_radius
        qt_radius = math.sqrt(qt2_radius)
        qt_real_pole = math.sqrt(value / (1 - value) * upsilon_pole)
        passed = (
            0 < value < 0.5
            and all(
                math.isfinite(item) and item > tolerance
                for item in (
                    rho_radius,
                    upsilon_radius,
                    ql_radius,
                    qt2_radius,
                    qt_radius,
                    qt_real_pole,
                )
            )
            and qt_radius <= qt_real_pole + tolerance
        )
        if not passed:
            raise QuadratureScalarTaylorDomainError(
                f"numeric Taylor-domain probe failed: {encoded}"
            )
        rows.append(
            {
                "s": encoded,
                "rho_taylor_radius": rho_radius,
                "rho_limiting_singularity": "zero_gradient_kink"
                if value <= 0.25
                else "outer_pK_pole",
                "upsilon_taylor_radius": upsilon_radius,
                "upsilon_limiting_singularity": "complex_sqrt_branch"
                if value <= crossover
                else "outer_pK_pole",
                "canonical_longitudinal_radius": ql_radius,
                "canonical_transverse_squared_radius": qt2_radius,
                "canonical_transverse_radius": qt_radius,
                "canonical_real_transverse_pole_radius": qt_real_pole,
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
        "exact_jet_domain_contract": config["exact_jet_domain_contract"],
        "canonical_radius_contract": config["canonical_radius_contract"],
        "endpoint_and_quartic_consistency_contract": config[
            "endpoint_and_quartic_consistency_contract"
        ],
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
            config["exact_jet_domain_contract"]["scope"],
            config["canonical_radius_contract"]["interpretation"],
            config["endpoint_and_quartic_consistency_contract"]["non_no_go"],
        ],
    }
    receipt["content_sha256"] = _content_sha(receipt)
    return receipt


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    if stored.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureScalarTaylorDomainError("receipt schema changed")
    payload = dict(stored)
    content = payload.pop("content_sha256", None)
    if not isinstance(content, str) or content != _content_sha(payload):
        raise QuadratureScalarTaylorDomainError("receipt content hash changed")
    if stored != build_receipt(repo):
        raise QuadratureScalarTaylorDomainError("stored receipt differs from exact rebuild")
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
        raise QuadratureScalarTaylorDomainError("refusing to replace existing receipt")
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
                raise QuadratureScalarTaylorDomainError(
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
        "exact_scalar_jet_domain": receipt["adjudication"][
            "exact_fixed_background_real_scalar_jet_domain_derived"
        ],
        "canonical_radii": receipt["adjudication"]["canonical_jet_radii_derived"],
        "quartic_truncation_artifact": receipt["adjudication"][
            "endpoint_collapse_is_quartic_truncation_artifact"
        ],
        "full_coupled_domain": receipt["adjudication"][
            "full_coupled_metric_aether_matter_analyticity_domain"
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
