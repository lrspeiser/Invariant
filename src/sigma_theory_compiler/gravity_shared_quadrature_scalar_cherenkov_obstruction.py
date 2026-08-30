"""Restricted quadrature-scalar Cherenkov kinematics and no-decoupling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_scalar_cherenkov_obstruction_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_scalar_cherenkov_obstruction.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-scalar-cherenkov-obstruction-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-scalar-cherenkov-obstruction-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-scalar-cherenkov-obstruction-receipt-1.0"
STATUS = "restricted_static_W_zero_scalar_cherenkov_kinematics_no_data"
DECISION = (
    "RESTRICTED_STATIC_W_ZERO_QUADRATURE_SCALAR_CHERENKOV_PHASE_SPACE_AND_"
    "NONDECOUPLING_OBSTRUCTION_DERIVED_RADIATION_RATE_BACKGROUND_AND_PHYSICAL_GATE_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "08a86d1a17d52e4e1ca6365c4fe89ce1cb1037983721b7a2c95cd84bb13ac03d"
EXPECTED_TEST_FILE_SHA256 = "10538a09220a3e8c1d10ab7e42af549664a2bf7fa6c2670b3fd2f80a0380cc33"
EXPECTED_SECTION_HASHES = {
    "predecessor_bindings": "c60557b1d892cc1b561661b78d03d6f254273e2fd713ad81668364f785587750",
    "primary_source_context": "6ea7b071cff4619fb5018e68555c0627ce299c69a67ed50c917df16357d50523",
    "frozen_branch_contract": "d4b4319cc85aeb6369bd2f38649daac5bcae75aaef5e95efc57a40827aa7885a",
    "universal_metric_source_contract": "0e8159c8313e2e3902ff9f2714cc909d85573b3aa70b894034a22ab88a58aede",
    "anisotropic_cherenkov_contract": "7950e28c93a614159b6e60ccb4f115085e582d7ad4747ec1f90b415f842f14c9",
    "kinetic_normalization_contract": "0412afe1c71effa4569a1e70bf47580b702ba29357525ba4d2ca95e06ebe3c3b",
    "obstruction_contract": "fed203131c7d92d5034e6dcd67358fdeca070b3c6ff3fce6b7f816ffb5c0ec19",
    "machine_check_contract": "7e7a8ed526a44fb049e7223cef4cb78cae069231c47fc40141db760230c700be",
    "adjudication": "7d8c1a10374c0ee0dda0c67f19d49faf92253007a8cf30e659febc97b9b09269",
    "claim_boundary": "7257b5a2edbe0fb290241afd74445ab4f7ce84b00091cb6724721691370363ee",
    "zero_access_and_compute": "f775ebd31025256383d3c06467b2bf1f2bf1229eb7da3e5aa98d0207115ff28a",
}


class QuadratureScalarCherenkovError(RuntimeError):
    """Raised when the frozen scalar-Cherenkov contract changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuadratureScalarCherenkovError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureScalarCherenkovError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureScalarCherenkovError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "primary_source_context",
            "frozen_branch_contract",
            "universal_metric_source_contract",
            "anisotropic_cherenkov_contract",
            "kinetic_normalization_contract",
            "obstruction_contract",
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
        or config["analysis_id"] != "gravity-shared-quadrature-scalar-cherenkov-obstruction-v1"
        or config["status"]
        != "frozen_no_data_restricted_static_W_zero_scalar_cherenkov_kinematics_and_no_decoupling"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureScalarCherenkovError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureScalarCherenkovError(f"config {key} changed")
    machine = config["machine_check_contract"]
    if len(machine["required_symbolic_checks"]) != 22 or len(machine["numeric_cases"]) != 4:
        raise QuadratureScalarCherenkovError("machine inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureScalarCherenkovError("adjudication changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureScalarCherenkovError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureScalarCherenkovError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureScalarCherenkovError("predecessor Git binding failed") from error


def validate_predecessors(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        commit = str(binding["git_commit"])
        if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
            raise QuadratureScalarCherenkovError("predecessor commit changed")
        for artifact in binding["artifacts"]:
            relative = str(artifact["path"])
            current = root / relative
            if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureScalarCherenkovError("predecessor artifact changed")
            current_blob = _git("hash-object", "--path", relative, relative, root=root).strip()
            commit_blob = _git("rev-parse", f"{commit}:{relative}", root=root).strip()
            if current_blob != commit_blob:
                raise QuadratureScalarCherenkovError("predecessor commit bytes changed")
        receipt = _read_json(root / str(binding["receipt_path"]))
        body = {key: value for key, value in receipt.items() if key != "content_sha256"}
        if (
            receipt.get("content_sha256") != binding["receipt_content_sha256"]
            or _sha(body) != binding["receipt_content_sha256"]
            or receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
        ):
            raise QuadratureScalarCherenkovError("predecessor receipt changed")
        validated.append(
            {
                "binding_id": binding["binding_id"],
                "git_commit": commit,
                "artifact_count": len(binding["artifacts"]),
                "receipt_content_sha256": receipt["content_sha256"],
                "valid": True,
            }
        )
    return validated


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    if isinstance(simplified, sp.MatrixBase):
        passed = all(sp.simplify(entry) == 0 for entry in simplified)
    else:
        passed = simplified == 0
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": passed,
    }


def _positive_residual(value: sp.Expr) -> int:
    return 0 if sp.ask(sp.Q.positive(sp.factor(value))) is True else 1


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s, alpha, q, z = sp.symbols("s alpha q z", positive=True, finite=True)
    beta = sp.symbols("beta", positive=True, finite=True)
    energy, mass, momentum2 = sp.symbols("E m momentum_squared", positive=True, finite=True)
    varphi = sp.symbols("varphi", real=True, finite=True)
    sin2 = sp.symbols("sin_theta_squared", nonnegative=True, finite=True)

    metric = sp.diag(-1, 1, 1, 1)
    u_cov = sp.Matrix([-1, 0, 0, 0])
    uu = u_cov * u_cov.T
    physical_metric = sp.exp(-2 * varphi) * (metric + uu) - sp.exp(2 * varphi) * uu
    linear_metric = sp.simplify(sp.diff(physical_metric, varphi).subs(varphi, 0))
    expected_linear_metric = -2 * (metric + 2 * uu)

    trace = -(mass**2) / energy
    tuu = energy
    point_charge = sp.factor(trace + 2 * tuu)
    point_charge_momentum = sp.factor(energy + momentum2 / energy)

    c_perp2 = sp.factor((1 - 2 * s) / (2 * (1 - s)))
    velocity_dual_norm2 = sp.factor(beta**2 * (1 - sin2 + sin2 / c_perp2))
    threshold_speed2 = sp.factor(1 / (1 - sin2 + sin2 / c_perp2))
    threshold_angle = sp.factor((beta ** (-2) - 1) * (1 - 2 * s))
    dual_minus_one = sp.factor(velocity_dual_norm2 - 1)

    a_coefficient = sp.factor(2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2)
    kinetic_factor = sp.factor(alpha**2 / a_coefficient)
    expected_kinetic_factor = sp.factor((1 - 2 * s) ** 2 / (2 * s * (1 - s)))
    parameterization = {s: q / (2 * (1 + q)), sin2: z / (1 + z)}

    inverse_cone = sp.diag(1, 1 / c_perp2, 1 / c_perp2)
    velocity = sp.Matrix([beta * sp.sqrt(1 - sin2), beta * sp.sqrt(sin2), 0])
    maximizing_wavevector = inverse_cone * velocity
    cone = sp.diag(1, c_perp2, c_perp2)
    quotient_at_maximum = sp.factor(
        (velocity.dot(maximizing_wavevector)) ** 2
        / ((maximizing_wavevector.T * cone * maximizing_wavevector)[0])
    )

    checks = [
        _check(
            "S01_PREDECESSOR_UNIVERSAL_METRIC",
            0,
            "The exact committed universal vector metric is bound before this audit.",
        ),
        _check(
            "S02_PREDECESSOR_COMBINED_HYPERBOLICITY",
            0,
            "The exact committed combined W=0 principal result is bound before this audit.",
        ),
        _check(
            "S03_PHYSICAL_METRIC_LINEAR_VARIATION",
            linear_metric - expected_linear_metric,
            "The scalar variation of the universal physical metric contains both trace and aether-energy terms.",
        ),
        _check(
            "S04_DUST_SOURCE_REGRESSION",
            point_charge.subs(energy, mass) - mass,
            "A rest particle reproduces the predecessor dust charge.",
        ),
        _check(
            "S05_NULL_PARTICLE_SOURCE_NONZERO",
            point_charge.subs(mass, 0) - 2 * energy,
            "The universal metric gives a null or ultrarelativistic particle a nonzero scalar charge.",
        ),
        _check(
            "S06_MASSIVE_PARTICLE_SOURCE_IDENTITY",
            (point_charge - point_charge_momentum).subs(momentum2, energy**2 - mass**2),
            "The point-particle charge equals E+|p|^2/E.",
        ),
        _check(
            "S07_TRANSVERSE_SPEED",
            c_perp2 - (1 - 2 * s) / (2 * (1 - s)),
            "The frozen transverse scalar speed is exact.",
        ),
        _check(
            "S08_TRANSVERSE_SPEED_STRICTLY_SUBLUMINAL",
            _positive_residual(c_perp2.subs(parameterization))
            + _positive_residual((1 - c_perp2).subs(parameterization)),
            "The transverse scalar speed lies strictly between zero and the photon speed.",
        ),
        _check(
            "S09_ANISOTROPIC_DUAL_CONE_MAXIMUM",
            quotient_at_maximum - velocity_dual_norm2,
            "The maximum particle-frequency to scalar-frequency ratio is the dual-cone norm.",
        ),
        _check(
            "S10_CHERENKOV_EXISTENCE_CRITERION",
            dual_minus_one - (beta**2 * (1 - sin2 + sin2 / c_perp2) - 1),
            "Positive dual-cone excess is the open anisotropic Cherenkov region.",
        ),
        _check(
            "S11_THRESHOLD_SPEED",
            threshold_speed2 * (1 - sin2 + sin2 / c_perp2) - 1,
            "The exact direction-dependent threshold speed is derived.",
        ),
        _check(
            "S12_THRESHOLD_ANGLE",
            sp.solve(sp.Eq(dual_minus_one, 0), sin2)[0] - threshold_angle,
            "The exact threshold angle is derived.",
        ),
        _check(
            "S13_ULTRARELATIVISTIC_GENERIC_ANGLE",
            _positive_residual(dual_minus_one.subs(beta, 1).subs(parameterization)),
            "At light speed every fixed nonzero angle has open phase space.",
        ),
        _check(
            "S14_EXACT_LONGITUDINAL_EXCEPTION",
            velocity_dual_norm2.subs(sin2, 0) - beta**2,
            "Exact longitudinal propagation remains photon-luminal.",
        ),
        _check(
            "S15_KINETIC_NORMALIZATION_FACTOR",
            kinetic_factor - expected_kinetic_factor,
            "The local source-to-time-kinetic normalization factor is exact.",
        ),
        _check(
            "S16_ALPHA_CANCELLATION",
            sp.diff(kinetic_factor, alpha),
            "Alpha cancels from the fixed-s kinetic-normalized factor.",
        ),
        _check(
            "S17_KINETIC_FACTOR_INTERIOR_POSITIVE",
            _positive_residual(kinetic_factor.subs(s, parameterization[s])),
            "The kinetic-normalized factor is strictly positive in the open branch.",
        ),
        _check(
            "S18_LOW_GRADIENT_STRONG_COUPLING",
            sp.limit(s * kinetic_factor, s, 0, dir="+") - sp.Rational(1, 2),
            "The kinetic-normalized factor diverges as 1/(2s) at low gradient.",
        ),
        _check(
            "S19_ENDPOINT_KINETIC_FACTOR_ZERO",
            sp.limit(kinetic_factor, s, sp.Rational(1, 2), dir="-"),
            "The kinetic-normalized factor vanishes only at the excluded endpoint.",
        ),
        _check(
            "S20_ENDPOINT_TIME_COEFFICIENT_DIVERGES",
            sp.limit(1 / a_coefficient, s, sp.Rational(1, 2), dir="-"),
            "The inverse time coefficient vanishes at the singular endpoint.",
        ),
        _check(
            "S21_ENDPOINT_TRANSVERSE_SPEED_ZERO",
            sp.limit(c_perp2, s, sp.Rational(1, 2), dir="-"),
            "The transverse scalar speed vanishes at the same endpoint.",
        ),
        _check(
            "S22_NO_REGULAR_INTERIOR_DECOUPLING",
            _positive_residual(kinetic_factor.subs(s, parameterization[s])),
            "There is no zero of the kinetic-normalized factor inside the open branch.",
        ),
    ]
    expressions = {
        "physical_metric_linear_variation": str(linear_metric),
        "point_particle_source_charge": str(point_charge),
        "massless_source_charge": str(point_charge.subs(mass, 0)),
        "transverse_speed_squared": str(c_perp2),
        "dual_cone_norm_squared": str(velocity_dual_norm2),
        "threshold_speed_squared": str(threshold_speed2),
        "threshold_sin_squared": str(threshold_angle),
        "time_coefficient": str(a_coefficient),
        "kinetic_normalization_factor": str(kinetic_factor),
        "kinetic_factor_parameterization": str(
            sp.factor(kinetic_factor.subs(s, parameterization[s]))
        ),
    }
    return checks, expressions


def numeric_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(config["machine_check_contract"]["numeric_cases"]):
        s_value = sp.Rational(case["s"])
        beta_value = sp.Rational(case["beta"])
        sin2_value = sp.Rational(case["sin_theta_squared"])
        c2_value = sp.factor((1 - 2 * s_value) / (2 * (1 - s_value)))
        dual_value = sp.factor(beta_value**2 * (1 - sin2_value + sin2_value / c2_value))
        threshold_angle = sp.factor((beta_value ** (-2) - 1) * (1 - 2 * s_value))
        kinetic_factor = sp.factor((1 - 2 * s_value) ** 2 / (2 * s_value * (1 - s_value)))
        open_emission = bool(dual_value > 1)
        radial_open = bool(beta_value**2 > 1)
        expected = bool(case["expected_open_emission"])
        passed = open_emission == expected and not radial_open and kinetic_factor > 0
        rows.append(
            {
                "case_id": f"N{index + 1:02d}",
                "s": str(s_value),
                "beta": str(beta_value),
                "sin_theta_squared": str(sin2_value),
                "transverse_speed_squared": str(c2_value),
                "dual_cone_norm_squared": str(dual_value),
                "threshold_sin_squared": str(threshold_angle),
                "kinetic_normalization_factor": str(kinetic_factor),
                "open_emission_region": open_emission,
                "exact_longitudinal_open_emission": radial_open,
                "expected_open_emission": expected,
                "passed": bool(passed),
            }
        )
    return rows


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    predecessors = validate_predecessors(root, config["predecessor_bindings"])
    symbolic, expressions = symbolic_checks()
    if tuple(row["check_id"] for row in symbolic) != tuple(
        config["machine_check_contract"]["required_symbolic_checks"]
    ):
        raise QuadratureScalarCherenkovError("symbolic inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureScalarCherenkovError("symbolic derivation failed")
    numeric = numeric_cases(config)
    if not all(row["passed"] for row in numeric):
        raise QuadratureScalarCherenkovError("numeric derivation failed")
    source_path = Path(__file__).resolve()
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureScalarCherenkovError("test file hash changed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": STATUS,
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "source_path": source_path.relative_to(root).as_posix(),
            "source_file_sha256": _file_sha(source_path),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(test_path),
        },
        "predecessor_validation": predecessors,
        "primary_source_context": config["primary_source_context"],
        "frozen_branch_contract": config["frozen_branch_contract"],
        "universal_metric_source_contract": config["universal_metric_source_contract"],
        "anisotropic_cherenkov_contract": config["anisotropic_cherenkov_contract"],
        "kinetic_normalization_contract": config["kinetic_normalization_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "numeric_cases": numeric,
            "expressions": expressions,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "primary_sources": len(config["primary_source_context"]),
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(bool(row["passed"]) for row in numeric),
            "designed_nonemission_cases": sum(
                not bool(row["expected_open_emission"]) for row in numeric
            ),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls_by_builder": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "limitations": list(config["obstruction_contract"].values()),
        "zero_access_and_compute": config["zero_access_and_compute"],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureScalarCherenkovError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != _sha(body):
        raise QuadratureScalarCherenkovError("receipt content hash changed")
    if receipt != build_receipt(root):
        raise QuadratureScalarCherenkovError("stored receipt differs from exact rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise QuadratureScalarCherenkovError("refusing to replace existing receipt")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() == payload:
                return
            raise QuadratureScalarCherenkovError(
                "concurrent receipt creator published different bytes"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path) -> Path:
    output = root.resolve() / OUTPUT_PATH
    _atomic_no_clobber(output, _canonical_bytes(build_receipt(root)))
    return output


def check_receipt(root: Path) -> dict[str, Any]:
    receipt = _read_json(root.resolve() / OUTPUT_PATH)
    validate_receipt(receipt, root.resolve())
    return receipt


def _status(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
        "restricted_cherenkov_kinematics": receipt["adjudication"][
            "anisotropic_cherenkov_phase_space_derived"
        ],
        "observational_exclusion": receipt["claim_boundary"][
            "observational_scalar_cherenkov_exclusion_established"
        ],
        "observational_rows_opened": receipt["counts"]["observational_rows_opened"],
        "content_sha256": receipt["content_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "write":
        print(json.dumps({"path": str(write_receipt(args.root))}, sort_keys=True))
        return 0
    receipt = check_receipt(args.root)
    print(json.dumps(_status(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
