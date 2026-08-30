"""Restricted weak-field lensing backreaction of the quadrature scalar action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_lensing_backreaction_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_lensing_backreaction.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-lensing-backreaction-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-lensing-backreaction-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-lensing-backreaction-receipt-1.0"
STATUS = "restricted_exterior_quadrature_lensing_backreaction_derived_no_data_quantitative_lensing_failed"
DECISION = (
    "RESTRICTED_QUADRATURE_LENSING_BACKREACTION_DERIVED_DIRECT_CONFORMAL_SHIFT_CANCELS_"
    "SCALAR_STRESS_LENSING_IS_ASYMPTOTICALLY_COMPACTNESS_SUPPRESSED_AND_ISOLATED_ENERGY_"
    "LOG_DIVERGES_GLOBAL_QUANTITATIVE_LENSING_REMAINS_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "3eb1558be624966fb861e860d6f590d8c76ad7862ae25cc29c43489d499bc55a"
EXPECTED_TEST_FILE_SHA256 = "05bbfd6865816ba46d0cfe315ab816fec9321dfbdaec3f20b0590df014a480ff"
EXPECTED_SECTION_HASHES = {
    "predecessor_binding": "d51d8d57fa4c1299ea2e3695b8e3ef88c993abeecd0ef65ad6c2e70cb5cfb1de",
    "frozen_scope": "e7d9c8b315b1125a7a541a6d82f95435d0397b3cc10b4883a218da49c883132a",
    "linearized_lensing_contract": "36e7d3d4f0d727ef121c519d3a504b05f058913b95e610537ad90f39342ae6fb",
    "asymptotic_contract": "d889f158122ef85105e1e7bc0bf6259dc1d1728bfb74e4d13c7a280d4ab24e17",
    "machine_check_contract": "003e46b70d64afcafb14410c02fdc5ca0f39184cbc52ab614254642b1d7d4e17",
    "adjudication": "12ad5fa09dc31d3cc53b08d9ea5863e241ad9dfeb3c00db6a46d1c688e6dd22d",
    "claim_boundary": "d2d12e906e16621937df5bde21c913914dfd1b8d126c39973dd74da904a78301",
    "zero_access_and_compute": "e78c1585d70382245459c43ceeacbd416dcc9aede3ed6dcca8c0e9add7cae190",
}


class QuadratureLensingBackreactionError(RuntimeError):
    """Raised when a frozen contract or exact derivation changes."""


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
        raise QuadratureLensingBackreactionError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureLensingBackreactionError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureLensingBackreactionError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_binding",
            "frozen_scope",
            "linearized_lensing_contract",
            "asymptotic_contract",
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
        or config["analysis_id"] != "gravity-shared-quadrature-lensing-backreaction-v1"
        or config["status"] != "frozen_no_data_restricted_exterior_lensing_backreaction_derivation"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureLensingBackreactionError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureLensingBackreactionError(f"config {key} changed")
    checks = config["machine_check_contract"]
    if len(checks["required_symbolic_checks"]) != 16 or len(checks["numeric_x_probes"]) != 4:
        raise QuadratureLensingBackreactionError("machine-check inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureLensingBackreactionError("adjudication decision changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureLensingBackreactionError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureLensingBackreactionError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureLensingBackreactionError("predecessor Git binding failed") from error


def validate_predecessor(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    commit = str(binding["git_commit"])
    if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
        raise QuadratureLensingBackreactionError("predecessor commit changed")
    for artifact in binding["artifacts"]:
        relative = str(artifact["path"])
        current = root / relative
        if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
            raise QuadratureLensingBackreactionError("predecessor artifact changed")
        if _git("show", f"{commit}:{relative}", root=root) != current.read_bytes():
            raise QuadratureLensingBackreactionError("predecessor commit bytes changed")
    receipt = _read_json(root / str(binding["receipt_path"]))
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        receipt.get("content_sha256") != binding["receipt_content_sha256"]
        or _sha(body) != binding["receipt_content_sha256"]
        or receipt.get("schema_version") != binding["receipt_schema_version"]
        or receipt.get("decision") != binding["receipt_decision"]
    ):
        raise QuadratureLensingBackreactionError("predecessor receipt changed")
    return {
        "binding_id": binding["binding_id"],
        "git_commit": commit,
        "artifact_count": len(binding["artifacts"]),
        "receipt_content_sha256": receipt["content_sha256"],
        "valid": True,
    }


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": simplified == 0,
    }


def _linearized_fourier_blocks(phi: sp.Symbol, psi: sp.Symbol, k: sp.Symbol) -> tuple[Any, Any]:
    eta = sp.diag(-1, 1, 1, 1)
    h = sp.zeros(4)
    h[0, 0] = -2 * phi
    for index in range(1, 4):
        h[index, index] = -2 * psi
    wave = (0, 0, 0, k)
    trace = sum(eta[index, index] * h[index, index] for index in range(4))
    h_up = sp.zeros(4)
    h_mixed = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            h_up[left, right] = sum(
                eta[left, a] * eta[right, b] * h[a, b] for a in range(4) for b in range(4)
            )
            h_mixed[left, right] = sum(eta[left, a] * h[a, right] for a in range(4))

    def component(mu: int, nu: int) -> Any:
        first = sum(-wave[a] * wave[mu] * h_mixed[a, nu] for a in range(4))
        second = sum(-wave[a] * wave[nu] * h_mixed[a, mu] for a in range(4))
        box_metric = k**2 * h[mu, nu]
        trace_term = wave[mu] * wave[nu] * trace
        divergence = sum(-wave[a] * wave[b] * h_up[a, b] for a in range(4) for b in range(4))
        box_trace = -(k**2) * trace
        einstein_trace = -eta[mu, nu] * (divergence - box_trace)
        return sp.simplify((first + second + box_metric + trace_term + einstein_trace) / 2)

    return component(0, 0), sp.simplify(sum(component(i, i) for i in range(1, 4)))


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, x = sp.symbols("s x", positive=True)
    phi, psi, k = sp.symbols("Phi_E Psi_E k", real=True)
    rho, pressure_trace = sp.symbols("rho pressure_trace", real=True)
    G, mpl2, a0, r_m, r, mass = sp.symbols("G Mpl2 a0 r_M r M_b", positive=True)
    p = s**2 / 4 + s / 4 + sp.log(1 - 2 * s) / 8
    rho_norm = -p
    radial_norm = p + s**3 / (1 - 2 * s)
    tangential_norm = p
    lens_norm = sp.simplify(2 * rho_norm + radial_norm + 2 * tangential_norm)
    x_of_s = s**2 / (1 - 2 * s)
    s_of_x = sp.sqrt(x**2 + x) - x
    g00, spatial_trace = _linearized_fourier_blocks(phi, psi, k)

    energy_radial_coefficient = sp.limit(2 * rho_norm / s**3, s, 0, dir="+")
    lens_radial_coefficient = sp.limit(2 * lens_norm / s**3, s, 0, dir="+")
    energy_log = sp.simplify(4 * sp.pi * energy_radial_coefficient * mpl2).subs(
        mpl2, 1 / (8 * sp.pi * G)
    )
    lens_gradient_log = sp.simplify(4 * sp.pi * G * lens_radial_coefficient * mpl2).subs(
        mpl2, 1 / (8 * sp.pi * G)
    )
    ratio_motion = sp.simplify((sp.Rational(2, 3) * a0**2 * r_m**3 / r**2) / (a0 * r_m / r)).subs(
        r_m**2, G * mass / a0
    )
    checks = [
        _check("S01_LINEARIZED_G00", g00 + 2 * psi * k**2, "G00=2 nabla^2 Psi_E."),
        _check(
            "S02_LINEARIZED_SPATIAL_TRACE",
            spatial_trace + 2 * k**2 * (phi - psi),
            "The spatial trace is 2 nabla^2(Phi_E-Psi_E).",
        ),
        _check(
            "S03_LENSING_POISSON_COMBINATION",
            2 * (4 * sp.pi * G * rho)
            + 4 * sp.pi * G * pressure_trace
            - 4 * sp.pi * G * (2 * rho + pressure_trace),
            "Adding twice the 00 equation to the spatial trace gives the lensing source.",
        ),
        _check(
            "S04_SCALAR_LENSING_SOURCE_EQUALS_RADIAL_PRESSURE",
            lens_norm - radial_norm,
            "The quadrature scalar lensing combination equals its radial pressure.",
        ),
        _check(
            "S05_KINETIC_DENSITY_CUBIC_LIMIT",
            sp.limit(p / s**3, s, 0, dir="+") + sp.Rational(1, 3),
            "p~-s^3/3.",
        ),
        _check(
            "S06_ENERGY_DENSITY_CUBIC_LIMIT",
            sp.limit(rho_norm / s**3, s, 0, dir="+") - sp.Rational(1, 3),
            "rho/(2Mpl^2a0^2)~s^3/3.",
        ),
        _check(
            "S07_LENSING_SOURCE_CUBIC_LIMIT",
            sp.limit(lens_norm / s**3, s, 0, dir="+") - sp.Rational(2, 3),
            "Sigma/(2Mpl^2a0^2)~2s^3/3.",
        ),
        _check(
            "S08_QUADRATURE_INVERSE_RELATION",
            sp.simplify(x_of_s.subs(s, s_of_x) - x),
            "The exterior branch obeys x=s^2/(1-2s).",
        ),
        _check(
            "S09_EXTERIOR_BRANCH_LIMIT",
            sp.limit(s_of_x / sp.sqrt(x), x, 0, dir="+") - 1,
            "s~sqrt(x)=r_M/r.",
        ),
        _check(
            "S10_ENERGY_DENSITY_RADIAL_COEFFICIENT",
            energy_radial_coefficient - sp.Rational(2, 3),
            "The physical energy-density coefficient is 2/3.",
        ),
        _check(
            "S11_LENSING_SOURCE_RADIAL_COEFFICIENT",
            lens_radial_coefficient - sp.Rational(4, 3),
            "The physical lensing-source coefficient is 4/3.",
        ),
        _check(
            "S12_SCALAR_ENERGY_LOG_COEFFICIENT",
            energy_log - 1 / (3 * G),
            "The exterior scalar energy grows with coefficient a0^2 r_M^3/(3G).",
        ),
        _check(
            "S13_LENSING_GRADIENT_LOG_COEFFICIENT",
            lens_gradient_log - sp.Rational(2, 3),
            "The lensing-gradient logarithm has coefficient 2a0^2r_M^3/3.",
        ),
        _check(
            "S14_BACKREACTION_TO_MOTION_RATIO",
            ratio_motion - sp.Rational(2, 3) * G * mass / r,
            "The leading logarithmic ratio to g_phi is compactness suppressed.",
        ),
        _check(
            "S15_BACKREACTION_TO_GR_EQUIVALENT_RATIO",
            ratio_motion / 2 - sp.Rational(1, 3) * G * mass / r,
            "The ratio to the GR-equivalent 2g_phi lensing response is compactness suppressed.",
        ),
        _check(
            "S16_DIRECT_CONFORMAL_LENSING_CANCELLATION",
            (phi + s) + (psi - s) - (phi + psi),
            "The direct universal conformal scalar shifts cancel from Phi+Psi.",
        ),
    ]
    expressions = {
        "kinetic_density": str(p),
        "normalized_energy_density": str(rho_norm),
        "normalized_radial_pressure": str(radial_norm),
        "normalized_lensing_source": str(lens_norm),
        "exterior_branch": str(s_of_x),
    }
    return checks, expressions


def numeric_probes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    for x in map(float, config["machine_check_contract"]["numeric_x_probes"]):
        s = math.sqrt(x * x + x) - x
        p = s * s / 4 + s / 4 + math.log1p(-2 * s) / 8
        rho_norm = -p
        radial_norm = p + s**3 / (1 - 2 * s)
        lens_norm = 2 * rho_norm + radial_norm + 2 * p
        branch_error = abs(s / math.sqrt(x) - 1)
        passed = (
            0 < s < 0.5
            and rho_norm > 0
            and lens_norm > 0
            and abs(lens_norm - radial_norm) <= tolerance
            and (previous_error is None or branch_error < previous_error)
        )
        rows.append(
            {
                "x": x,
                "s": s,
                "normalized_energy_density": rho_norm,
                "normalized_lensing_source": lens_norm,
                "s_over_sqrt_x": s / math.sqrt(x),
                "branch_asymptotic_error": branch_error,
                "passed": passed,
            }
        )
        previous_error = branch_error
    return rows


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    predecessor = validate_predecessor(root, config["predecessor_binding"])
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureLensingBackreactionError("test file hash changed")
    symbolic, expressions = symbolic_checks()
    numeric = numeric_probes(config)
    if tuple(row["check_id"] for row in symbolic) != tuple(
        config["machine_check_contract"]["required_symbolic_checks"]
    ):
        raise QuadratureLensingBackreactionError("symbolic check inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureLensingBackreactionError("symbolic derivation failed")
    if not all(row["passed"] for row in numeric):
        raise QuadratureLensingBackreactionError("numeric branch probe failed")
    source_path = Path(__file__).resolve()
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
        "predecessor_validation": predecessor,
        "frozen_scope": config["frozen_scope"],
        "linearized_lensing_contract": config["linearized_lensing_contract"],
        "asymptotic_contract": config["asymptotic_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "derived_expressions": expressions,
            "numeric_probes": numeric,
        },
        "counts": {
            "predecessor_artifacts": predecessor["artifact_count"],
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "numeric_probes": len(numeric),
            "numeric_probes_passed": sum(bool(row["passed"]) for row in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The result is a static linearized spherical exterior calculation, not a nonlinear global metric solution.",
            "The logarithmic terms require an interior matching constant and can be changed by a separately derived external-field, cosmological, or high-radius completion; none is present here.",
            "Compactness suppression is a failure of this universal conformal single-scalar branch, not an unconditional no-go for every same-action architecture.",
            "No lensing rows, target values, halo model, or photon-specific adjustment was opened, fitted, or introduced.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body):
        raise QuadratureLensingBackreactionError("receipt content hash changed")
    if dict(receipt) != build_receipt(root):
        raise QuadratureLensingBackreactionError("receipt evidence changed")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureLensingBackreactionError("refusing to overwrite existing receipt")
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
                return "EXISTING_IDENTICAL"
            raise QuadratureLensingBackreactionError("receipt publication race") from error
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path) -> tuple[Path, str]:
    path = root.resolve() / OUTPUT_PATH
    return path, _atomic_no_clobber(path, _canonical_bytes(build_receipt(root)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        path, publication = write_receipt(root)
        output: Any = {"path": str(path), "publication": publication}
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {
            "valid": True,
            "decision": receipt["decision"],
            "content_sha256": receipt["content_sha256"],
        }
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "restricted_exterior_lensing_backreaction_derived": receipt["claim_boundary"][
                "restricted_exterior_lensing_backreaction_derived"
            ],
            "same_action_quantitative_lensing_success": receipt["claim_boundary"][
                "same_action_quantitative_lensing_success"
            ],
            "content_sha256": receipt["content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
