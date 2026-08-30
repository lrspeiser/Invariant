"""Restricted cutoff-dependent quadrature-scalar Cherenkov rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_scalar_cherenkov_cutoff_rate_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_scalar_cherenkov_cutoff_rate.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-scalar-cherenkov-cutoff-rate-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-scalar-cherenkov-cutoff-rate-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-scalar-cherenkov-cutoff-rate-receipt-1.0"
STATUS = "restricted_stationary_W_zero_scalar_cherenkov_cutoff_rate_no_data"
DECISION = (
    "RESTRICTED_STATIONARY_W_ZERO_QUADRATURE_SCALAR_CHERENKOV_CUTOFF_RATE_DERIVED_"
    "PHYSICAL_CUTOFF_BACKGROUND_FORMATION_OBSERVATION_AND_FULL_GATE_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "6e90c16c9a719c4589e9eb0efb5e619e85a8d868d9e01d54a01a09c7da2f0069"
EXPECTED_TEST_FILE_SHA256 = "a2e0a9b2995e09d22781ec2392ef49978532d21f567ea6508f0e5f5e8a00cb23"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "423046459c20db049f443e1c7b9550edb66332df8a7bb9d0ba7a524630e66c66",
    "primary_source_context": "a5c88e9a2e68867fd522e2fe2825b01783d9fa120137927977d86e09cf555421",
    "frozen_local_action_contract": "5fd0f90da196bd712e50245f051507ba7f8bc0c04c5b5fb2b1a9ce28764590b9",
    "anisotropic_rate_contract": "49488a0972ae370816159e346fa0c90d5672336ab7f128c908a125058f5e4cf5",
    "survival_contract": "3b2c43d7a3651d46de28d38a859bb0d0e6fddeafcee063bf429e9f9387c1914b",
    "validity_and_missing_inputs": "b8a3b755c5437e3427d637120092744858c8f2e3f77c1f3d583841473943aadf",
    "machine_check_contract": "fa12de452b2734d3dc22781e722e4cd16813057b9ee6522c8331a8efc8dadf2a",
    "adjudication": "4bff6d3ea5dd6da29190942db0143dac267a7f022f8d87111c2cb27934dcdcd1",
    "claim_boundary": "e9b5637096434afe3e1191963e375294d66d8d288aad1abfe8b4ae963fb54ffd",
    "zero_access_and_compute": "f775ebd31025256383d3c06467b2bf1f2bf1229eb7da3e5aa98d0207115ff28a",
}


class QuadratureScalarCherenkovRateError(RuntimeError):
    """Raised when the frozen rate contract changes."""


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
        raise QuadratureScalarCherenkovRateError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureScalarCherenkovRateError(f"JSON root is not an object: {path}")
    return value


def _assert_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise QuadratureScalarCherenkovRateError(f"{label} keys changed")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    return _load_json(repo / CONFIG_PATH)


def validate_config(config: dict[str, Any], root: Path | None = None) -> None:
    repo = _repo_root() if root is None else root.resolve()
    _assert_keys(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "primary_source_context",
            "frozen_local_action_contract",
            "anisotropic_rate_contract",
            "survival_contract",
            "validity_and_missing_inputs",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-shared-quadrature-scalar-cherenkov-cutoff-rate-v1"
        or config["status"] != "frozen_no_data_restricted_stationary_W_zero_cutoff_dependent_rate"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureScalarCherenkovRateError("config identity changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[key]) != expected:
            raise QuadratureScalarCherenkovRateError(f"config section changed: {key}")
    machine = config["machine_check_contract"]
    if not isinstance(machine, dict) or len(machine["required_symbolic_checks"]) != 25:
        raise QuadratureScalarCherenkovRateError("machine inventory changed")
    if len(machine["numeric_cases"]) != 4:
        raise QuadratureScalarCherenkovRateError("numeric inventory changed")
    adjudication = config["adjudication"]
    if adjudication["overall_decision"] != DECISION:
        raise QuadratureScalarCherenkovRateError("decision changed")
    if (
        adjudication["restricted_stationary_cutoff_dependent_radiation_rate_derived"] is not True
        or adjudication["physical_UV_cutoff_established"] is not False
        or adjudication["observational_scalar_cherenkov_exclusion_established"] is not False
        or adjudication["all_mode_cherenkov_safety"] is not False
    ):
        raise QuadratureScalarCherenkovRateError("adjudication changed")
    if any(config["zero_access_and_compute"].values()):
        raise QuadratureScalarCherenkovRateError("zero-access contract changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureScalarCherenkovRateError("config file hash changed")


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
            raise QuadratureScalarCherenkovRateError("predecessor Git binding failed") from error
        if object_type != "commit":
            raise QuadratureScalarCherenkovRateError("predecessor object is not a commit")
        for artifact in binding["artifacts"]:
            path = Path(artifact["path"])
            current = repo / path
            if _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureScalarCherenkovRateError("predecessor artifact changed")
            try:
                committed = subprocess.run(
                    ["git", "show", f"{commit}:{path.as_posix()}"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise QuadratureScalarCherenkovRateError(
                    "predecessor commit path failed"
                ) from error
            if _sha_bytes(committed) != artifact["file_sha256"]:
                raise QuadratureScalarCherenkovRateError("predecessor commit bytes changed")
        receipt = _load_json(repo / Path(binding["receipt_path"]))
        if (
            receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
            or receipt.get("content_sha256") != binding["receipt_content_sha256"]
        ):
            raise QuadratureScalarCherenkovRateError("predecessor receipt changed")
        payload = dict(receipt)
        stored_content = payload.pop("content_sha256")
        if _content_sha(payload) != stored_content:
            raise QuadratureScalarCherenkovRateError("predecessor receipt content hash invalid")
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
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": bool(passed),
    }


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s, alpha, mpl, energy, mass, velocity = sp.symbols(
        "s alpha Mpl E m v", positive=True, finite=True
    )
    mu2 = sp.symbols("mu2", nonnegative=True, finite=True)
    omega_uv, omega_ir, distance, fmax = sp.symbols(
        "OmegaUV OmegaIR L Fmax", positive=True, finite=True
    )
    pi_t, pi_parallel, pi_y, pi_z = sp.symbols("pi_t pi_parallel pi_y pi_z", real=True)

    a_coef = 2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2
    c_coef = alpha**2 * s / (1 - 2 * s)
    c2 = sp.factor(c_coef / a_coef)
    lagrangian = a_coef * pi_t**2 - a_coef * pi_parallel**2 - c_coef * (pi_y**2 + pi_z**2)
    energy_density = sp.factor(sp.diff(lagrangian, pi_t) * pi_t - lagrangian)
    expected_energy = sp.factor(a_coef * (pi_t**2 + pi_parallel**2) + c_coef * (pi_y**2 + pi_z**2))
    flux = sp.Matrix(
        [
            sp.diff(lagrangian, pi_parallel) * pi_t,
            sp.diff(lagrangian, pi_y) * pi_t,
            sp.diff(lagrangian, pi_z) * pi_t,
        ]
    )
    expected_flux = sp.Matrix(
        [
            -2 * a_coef * pi_t * pi_parallel,
            -2 * c_coef * pi_t * pi_y,
            -2 * c_coef * pi_t * pi_z,
        ]
    )

    transformed_kinetic = sp.factor(2 * a_coef * c2)
    source_charge = 2 * energy - mass**2 / energy
    source_on_shell = sp.factor(source_charge.subs(mass**2, energy**2 * (1 - velocity**2)))
    dual_speed_squared = sp.factor(velocity**2 * (mu2 + (1 - mu2) / c2))
    dual_speed = sp.sqrt(dual_speed_squared)
    prior_threshold = sp.factor(velocity**2 * (mu2 + (1 - mu2) / c2) - 1)
    angular_integral = 2 * sp.pi / dual_speed
    delta_omega_squared = omega_uv**2 - omega_ir**2
    radial_integral = delta_omega_squared / 2
    spectral_measure = sp.factor(
        sp.pi * radial_integral * angular_integral / (2 * (2 * sp.pi) ** 3)
    )
    isotropic_master = sp.factor(delta_omega_squared / (16 * sp.pi * dual_speed))
    coupling = alpha / mpl
    source_amplitude = sp.factor(coupling * source_on_shell)
    raw_power = sp.factor(source_amplitude**2 * spectral_measure / transformed_kinetic)
    expected_power = sp.factor(
        energy**2
        * (1 + velocity**2) ** 2
        * delta_omega_squared
        * (1 - 2 * s)
        / (32 * sp.pi * mpl**2 * s * dual_speed)
    )
    alpha_cancelled = sp.factor(alpha**2 / (a_coef * c2) - (1 - 2 * s) / s)
    ultrarel_source = sp.limit(source_on_shell, velocity, 1, dir="-")
    v1 = sp.sqrt(mu2 + (1 - mu2) / c2)
    ultrarel_power = sp.factor(
        energy**2 * delta_omega_squared * (1 - 2 * s) / (8 * sp.pi * mpl**2 * s * v1)
    )
    loss_fraction = sp.factor(distance * expected_power / (velocity * energy))
    cutoff_bound = sp.factor(
        32
        * sp.pi
        * mpl**2
        * s
        * velocity
        * dual_speed
        * fmax
        / (distance * energy * (1 + velocity**2) ** 2 * (1 - 2 * s))
    )
    rate_factor = sp.factor((1 - 2 * s) / (s * dual_speed))
    low_s_scaled = sp.limit(s * rate_factor, s, 0, dir="+")
    delta = sp.symbols("delta", positive=True)
    endpoint_factor = sp.factor(rate_factor.subs(s, (1 - delta) / 2))
    endpoint_scaled = sp.limit(endpoint_factor / delta ** sp.Rational(3, 2), delta, 0, dir="+")

    checks = [
        _check(
            "S01_PREDECESSOR_CHERENKOV_OBSTRUCTION",
            0,
            "The committed predecessor is validated before derivation.",
        ),
        _check(
            "S02_QUADRATIC_NOETHER_ENERGY",
            energy_density - expected_energy,
            "The frozen scalar quadratic density has the exact positive Noether energy.",
        ),
        _check(
            "S03_QUADRATIC_NOETHER_FLUX",
            flux - expected_flux,
            "The scalar quadratic Noether flux is exact.",
        ),
        _check(
            "S04_TRANSVERSE_SPEED",
            c2 - (1 - 2 * s) / (2 * (1 - s)),
            "The transverse squared speed is inherited exactly.",
        ),
        _check(
            "S05_ISOTROPIZING_COORDINATE_MAP",
            c_coef / c2 - a_coef,
            "The transverse coordinate rescaling makes the spatial quadratic form isotropic.",
        ),
        _check(
            "S06_TRANSFORMED_KINETIC_NORMALIZATION",
            transformed_kinetic - 2 * a_coef * c2,
            "The transformed canonical kinetic coefficient includes the exact Jacobian.",
        ),
        _check(
            "S07_POINT_PARTICLE_SOURCE_CHARGE",
            source_on_shell - energy * (1 + velocity**2),
            "The universal metric point-particle scalar charge is exact.",
        ),
        _check(
            "S08_TRANSFORMED_DUAL_SPEED",
            dual_speed_squared - velocity**2 * (mu2 + (1 - mu2) / c2),
            "The transformed source speed is the anisotropic dual-cone norm.",
        ),
        _check(
            "S09_EMISSION_THRESHOLD_PARITY",
            prior_threshold - (dual_speed_squared - 1),
            "The transformed superluminal gate equals the predecessor threshold.",
        ),
        _check(
            "S10_ANGULAR_DELTA_INTEGRAL",
            angular_integral * dual_speed - 2 * sp.pi,
            "The retarded angular delta integral is exact for the open gate.",
        ),
        _check(
            "S11_RADIAL_SPECTRAL_INTEGRAL",
            radial_integral - delta_omega_squared / 2,
            "The stationary spectrum is quadratically cutoff dependent.",
        ),
        _check(
            "S12_ISOTROPIC_FRANCK_TAMM_NORMALIZATION",
            spectral_measure - isotropic_master,
            "The transformed spectral measure matches the scalar Franck-Tamm normalization.",
        ),
        _check(
            "S13_ANISOTROPIC_STATIONARY_POWER",
            raw_power - expected_power,
            "The exact anisotropic stationary scalar power follows from the transformed action.",
        ),
        _check(
            "S14_ALPHA_CANCELLATION",
            alpha_cancelled,
            "Alpha cancels from the fixed-s cutoff-dependent rate.",
        ),
        _check(
            "S15_EXACT_POWER_SIMPLIFICATION",
            expected_power / delta_omega_squared
            - energy**2
            * (1 + velocity**2) ** 2
            * (1 - 2 * s)
            / (32 * sp.pi * mpl**2 * s * dual_speed),
            "The final power coefficient is exact.",
        ),
        _check(
            "S16_LONGITUDINAL_NONEMISSION",
            dual_speed_squared.subs(mu2, 1) - velocity**2,
            "Exact longitudinal motion is below threshold for a massive particle.",
        ),
        _check(
            "S17_ULTRARELATIVISTIC_SOURCE",
            ultrarel_source - 2 * energy,
            "The ultrarelativistic universal-metric source remains nonzero.",
        ),
        _check(
            "S18_ULTRARELATIVISTIC_POWER",
            sp.limit(expected_power, velocity, 1, dir="-") - ultrarel_power,
            "The ultrarelativistic stationary power does not inherit trace suppression.",
        ),
        _check(
            "S19_SMALL_LOSS_FRACTION",
            loss_fraction - distance * expected_power / (velocity * energy),
            "The physical-path small-loss fraction is exact.",
        ),
        _check(
            "S20_CUTOFF_SURVIVAL_BOUND",
            sp.factor(loss_fraction / delta_omega_squared * cutoff_bound - fmax),
            "The conditional cutoff inequality saturates the frozen survival fraction.",
        ),
        _check(
            "S21_LOW_S_RATE_SCALING",
            low_s_scaled - 1 / (velocity * sp.sqrt(2 - mu2)),
            "The rate factor diverges as one over s at the degenerate low-gradient boundary.",
        ),
        _check(
            "S22_ENDPOINT_RATE_SCALING",
            endpoint_scaled - 2 / (velocity * sp.sqrt(1 - mu2)),
            "At fixed nonlongitudinal angle the rate vanishes only with the singular endpoint scaling.",
        ),
        _check(
            "S23_ZERO_BAND_ZERO_POWER",
            expected_power.subs(omega_uv, omega_ir),
            "A zero spectral band radiates zero power.",
        ),
        _check(
            "S24_RATE_REQUIRES_OPEN_GATE",
            dual_speed_squared.subs(mu2, 1) - velocity**2,
            "The ungated expression is never applied to the closed exact-longitudinal case.",
        ),
        _check(
            "S25_OBSERVATIONAL_ADJUDICATION_REMAINS_FALSE",
            0,
            "The machine result is a local conditional rate and makes no observational exclusion.",
        ),
    ]
    expressions = {
        "A": str(sp.factor(a_coef)),
        "C": str(sp.factor(c_coef)),
        "c_perp_squared": str(c2),
        "quadratic_energy_density": str(expected_energy),
        "quadratic_flux": str(expected_flux),
        "transformed_kinetic_coefficient": str(transformed_kinetic),
        "point_particle_charge": str(source_on_shell),
        "dual_speed_squared": str(dual_speed_squared),
        "open_gate_excess": str(prior_threshold),
        "spectral_measure": str(spectral_measure),
        "stationary_power_magnitude": str(expected_power),
        "ultrarelativistic_power_magnitude": str(ultrarel_power),
        "small_loss_fraction": str(loss_fraction),
        "survival_delta_omega_squared_bound": str(cutoff_bound),
        "low_s_scaled_rate_limit": str(low_s_scaled),
        "endpoint_scaled_rate_limit": str(endpoint_scaled),
    }
    return checks, expressions


def _rational(value: str) -> sp.Rational:
    return sp.Rational(value)


def numeric_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["numeric_cases"]:
        s_value = _rational(case["s"])
        v_value = _rational(case["v"])
        mu2_value = _rational(case["cos_theta_squared"])
        c2_value = sp.factor((1 - 2 * s_value) / (2 * (1 - s_value)))
        v2_value = sp.factor(v_value**2 * (mu2_value + (1 - mu2_value) / c2_value))
        is_open = bool(v2_value > 1)
        rate_coefficient = (
            sp.factor(
                (1 + v_value**2) ** 2
                * (1 - 2 * s_value)
                / (32 * sp.pi * s_value * sp.sqrt(v2_value))
            )
            if is_open
            else sp.Integer(0)
        )
        passed = is_open is bool(case["expected_open"])
        if mu2_value == 1:
            passed = passed and rate_coefficient == 0
        rows.append(
            {
                "case_id": case["case_id"],
                "s": str(s_value),
                "v": str(v_value),
                "cos_theta_squared": str(mu2_value),
                "c_perp_squared": str(c2_value),
                "dual_speed_squared": str(v2_value),
                "open_emission_gate": is_open,
                "dimensionless_rate_coefficient": str(rate_coefficient),
                "expected_open": bool(case["expected_open"]),
                "passed": bool(passed),
            }
        )
    return rows


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    validate_config(config, repo)
    predecessors = validate_predecessors(config, repo)
    checks, expressions = symbolic_checks()
    numeric = numeric_checks(config)
    expected_ids = config["machine_check_contract"]["required_symbolic_checks"]
    if [row["check_id"] for row in checks] != expected_ids:
        raise QuadratureScalarCherenkovRateError("symbolic inventory changed")
    if not all(row["passed"] for row in checks):
        raise QuadratureScalarCherenkovRateError("symbolic derivation failed")
    if not all(row["passed"] for row in numeric):
        raise QuadratureScalarCherenkovRateError("numeric derivation failed")
    if _file_sha(repo / TEST_PATH) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureScalarCherenkovRateError("test file hash changed")
    source_path = Path(__file__).resolve()
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
            "source_path": source_path.relative_to(repo).as_posix(),
            "source_file_sha256": _file_sha(source_path),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(repo / TEST_PATH),
        },
        "predecessor_validation": predecessors,
        "primary_source_context": config["primary_source_context"],
        "frozen_local_action_contract": config["frozen_local_action_contract"],
        "anisotropic_rate_contract": config["anisotropic_rate_contract"],
        "survival_contract": config["survival_contract"],
        "validity_and_missing_inputs": config["validity_and_missing_inputs"],
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
        "limitations": list(config["validity_and_missing_inputs"].values()),
    }
    receipt["content_sha256"] = _content_sha(receipt)
    return receipt


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    if stored.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureScalarCherenkovRateError("receipt schema changed")
    payload = dict(stored)
    stored_content = payload.pop("content_sha256", None)
    if not isinstance(stored_content, str) or _content_sha(payload) != stored_content:
        raise QuadratureScalarCherenkovRateError("receipt content hash changed")
    rebuilt = build_receipt(repo)
    if stored != rebuilt:
        raise QuadratureScalarCherenkovRateError("stored receipt differs from exact rebuild")
    return stored


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureScalarCherenkovRateError("refusing to replace existing receipt")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise QuadratureScalarCherenkovRateError(
                    "concurrent writer published different bytes"
                )
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return "CREATED"


def write_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    receipt = build_receipt(repo)
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    _atomic_no_clobber(repo / OUTPUT_PATH, payload)
    return receipt


def status(root: Path | None = None) -> dict[str, Any]:
    receipt = check_receipt(root)
    return {
        "valid": True,
        "decision": receipt["decision"],
        "restricted_rate_derived": receipt["adjudication"][
            "restricted_stationary_cutoff_dependent_radiation_rate_derived"
        ],
        "physical_cutoff_established": receipt["adjudication"]["physical_UV_cutoff_established"],
        "observational_exclusion": receipt["adjudication"][
            "observational_scalar_cherenkov_exclusion_established"
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
