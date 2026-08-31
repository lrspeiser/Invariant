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

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_flrw_bounded_witness_v1.json")
CONFIG_CANONICAL_SHA256 = "97d7073e72896b42990882456fb3ce4fe9ed77b5d85430deede08d9f9ddef2d6"


class KineticGateFlrwWitnessError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise KineticGateFlrwWitnessError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["content_sha256"] = ""
    return _content_sha256(payload)


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise KineticGateFlrwWitnessError(f"missing predecessor git object: {commit}:{relative}")
    return result.stdout


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = _read_json(base / CONFIG_PATH)
    if _content_sha256(config) != CONFIG_CANONICAL_SHA256:
        raise KineticGateFlrwWitnessError("FLRW witness config semantics changed")
    if (
        config.get("schema_version")
        != "invariant-gravity-matter-lensing-kinetic-gate-flrw-bounded-witness-1.0"
        or config.get("status") != "APPEND_ONLY_COUPLED_ON_SHELL_BOUNDED_DOMAIN_WITNESS"
    ):
        raise KineticGateFlrwWitnessError("unsupported FLRW witness config")
    for predecessor in config["predecessors"]:
        for role in ("config", "module", "test", "receipt"):
            relative = predecessor[f"{role}_path"]
            expected = predecessor[f"{role}_sha256"]
            path = base / relative
            if not path.is_file() or _sha256_file(path) != expected:
                raise KineticGateFlrwWitnessError(f"predecessor {predecessor['id']} {role} changed")
            if _sha256_bytes(_git_show(base, predecessor["commit"], relative)) != expected:
                raise KineticGateFlrwWitnessError(
                    f"predecessor {predecessor['id']} {role} commit binding changed"
                )
        receipt = _read_json(base / predecessor["receipt_path"])
        if receipt.get("content_sha256") != predecessor["receipt_content_sha256"]:
            raise KineticGateFlrwWitnessError(
                f"predecessor {predecessor['id']} receipt content changed"
            )
    if config["claim_boundary"]["publication_ready"] is not False:
        raise KineticGateFlrwWitnessError("claim ceiling changed")
    return config


def gate_terms(x_value: float, beta: float) -> dict[str, float]:
    if not math.isfinite(x_value) or x_value <= 0.0:
        raise KineticGateFlrwWitnessError("X must be finite and positive")
    if not math.isfinite(beta) or beta <= 0.0:
        raise KineticGateFlrwWitnessError("beta must be finite and positive")
    u_value = beta * x_value**2
    z_value = (1.0 + u_value) ** 2
    zx_value = 4.0 * beta * x_value * (1.0 + u_value)
    zxx_value = 4.0 * beta * (1.0 + 3.0 * u_value)
    h_value = zx_value + 2.0 * x_value * zxx_value
    m_value = 12.0 * u_value * (1.0 + u_value) ** 2 * (1.0 - 3.0 * u_value) / x_value
    q_value = 2.0 * u_value / (1.0 + u_value)
    return {
        "u": u_value,
        "Z": z_value,
        "Z_X": zx_value,
        "Z_XX": zxx_value,
        "H_gate": h_value,
        "M": m_value,
        "q": q_value,
    }


def initial_currents(x_value: float, y_value: float, beta: float) -> tuple[float, float]:
    if not math.isfinite(y_value) or y_value <= 0.0:
        raise KineticGateFlrwWitnessError("Y must be finite and positive")
    terms = gate_terms(x_value, beta)
    c_value = 1.0 + y_value * terms["Z_X"]
    return c_value * math.sqrt(2.0 * x_value), terms["Z"] * math.sqrt(2.0 * y_value)


def _y_from_chi_current(x_value: float, scale_factor: float, j_chi: float, beta: float) -> float:
    z_value = gate_terms(x_value, beta)["Z"]
    return j_chi**2 / (2.0 * scale_factor**6 * z_value**2)


def solve_invariants(
    scale_factor: float,
    j_phi: float,
    j_chi: float,
    beta: float,
    iterations: int,
) -> tuple[float, float]:
    if not math.isfinite(scale_factor) or scale_factor < 1.0:
        raise KineticGateFlrwWitnessError("witness scale factor must be finite and at least one")
    if iterations < 80:
        raise KineticGateFlrwWitnessError("insufficient root iterations")

    def residual(x_value: float) -> float:
        if x_value == 0.0:
            return -j_phi
        terms = gate_terms(x_value, beta)
        y_value = _y_from_chi_current(x_value, scale_factor, j_chi, beta)
        c_value = 1.0 + y_value * terms["Z_X"]
        return scale_factor**3 * c_value * math.sqrt(2.0 * x_value) - j_phi

    lower = 0.0
    upper = math.nextafter(math.sqrt(1.0 / (3.0 * beta)), 0.0)
    if residual(upper) <= 0.0:
        raise KineticGateFlrwWitnessError(
            "current has no root inside the bounded positive-mixing interval"
        )
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    x_value = 0.5 * (lower + upper)
    y_value = _y_from_chi_current(x_value, scale_factor, j_chi, beta)
    if gate_terms(x_value, beta)["u"] >= 1.0 / 3.0:
        raise KineticGateFlrwWitnessError("root escaped the bounded positive-mixing interval")
    return x_value, y_value


def _solve_derivatives(x_value: float, y_value: float, beta: float) -> tuple[float, float]:
    terms = gate_terms(x_value, beta)
    z_value = terms["Z"]
    zx_value = terms["Z_X"]
    zxx_value = terms["Z_XX"]
    c_value = 1.0 + y_value * zx_value
    a11 = 0.5 / x_value + y_value * zxx_value / c_value
    a12 = zx_value / c_value
    a21 = zx_value / z_value
    a22 = 0.5 / y_value
    determinant = a11 * a22 - a12 * a21
    if determinant <= 0.0:
        raise KineticGateFlrwWitnessError("implicit current Jacobian is not positive")
    dx_d_n = 3.0 * (a12 - a22) / determinant
    dy_d_n = 3.0 * (a21 - a11) / determinant
    return dx_d_n, dy_d_n


def _symmetric_eigenvalues(a11: float, a12: float, a22: float) -> tuple[float, float]:
    discriminant = math.sqrt((a11 - a22) ** 2 + 4.0 * a12**2)
    return 0.5 * (a11 + a22 - discriminant), 0.5 * (a11 + a22 + discriminant)


def _sound_speed_eigenvalues(
    gradient_phi: float,
    gradient_chi: float,
    kinetic_phi: float,
    kinetic_cross: float,
    kinetic_chi: float,
) -> tuple[float, float]:
    determinant = kinetic_phi * kinetic_chi - kinetic_cross**2
    linear = gradient_phi * kinetic_chi + gradient_chi * kinetic_phi
    constant = gradient_phi * gradient_chi
    discriminant = linear**2 - 4.0 * determinant * constant
    scale = max(linear**2, abs(4.0 * determinant * constant), 1.0)
    if discriminant < -1.0e-13 * scale:
        raise KineticGateFlrwWitnessError("complex generalized sound speeds")
    root = math.sqrt(max(discriminant, 0.0))
    return (linear - root) / (2.0 * determinant), (linear + root) / (2.0 * determinant)


def evaluate_point(
    scale_factor: float,
    x_value: float,
    y_value: float,
    beta: float,
    m_pl: float,
    j_phi: float,
    j_chi: float,
) -> dict[str, float]:
    terms = gate_terms(x_value, beta)
    z_value = terms["Z"]
    zx_value = terms["Z_X"]
    c_value = 1.0 + y_value * zx_value
    kinetic_phi = 1.0 + y_value * terms["H_gate"]
    kinetic_cross = 2.0 * zx_value * math.sqrt(x_value * y_value)
    kinetic_chi = z_value
    kinetic_min, kinetic_max = _symmetric_eigenvalues(kinetic_phi, kinetic_cross, kinetic_chi)
    kinetic_determinant = kinetic_phi * kinetic_chi - kinetic_cross**2
    sound_min, sound_max = _sound_speed_eigenvalues(
        c_value, z_value, kinetic_phi, kinetic_cross, kinetic_chi
    )
    pressure = x_value + z_value * y_value
    energy_density = x_value + z_value * y_value + 2.0 * x_value * y_value * zx_value
    hubble = math.sqrt(energy_density / (3.0 * m_pl**2))
    dx_d_n, dy_d_n = _solve_derivatives(x_value, y_value, beta)
    rho_x = 1.0 + 3.0 * y_value * zx_value + 2.0 * x_value * y_value * terms["Z_XX"]
    rho_y = z_value + 2.0 * x_value * zx_value
    drho_d_n = rho_x * dx_d_n + rho_y * dy_d_n
    phi_current = scale_factor**3 * c_value * math.sqrt(2.0 * x_value)
    chi_current = scale_factor**3 * z_value * math.sqrt(2.0 * y_value)
    phi_eom_residual = abs(
        3.0
        + (0.5 / x_value + y_value * terms["Z_XX"] / c_value) * dx_d_n
        + (zx_value / c_value) * dy_d_n
    )
    chi_eom_residual = abs(3.0 + (zx_value / z_value) * dx_d_n + (0.5 / y_value) * dy_d_n)
    continuity_scale = max(abs(drho_d_n), abs(3.0 * (energy_density + pressure)), 1.0)
    continuity_residual = abs(drho_d_n + 3.0 * (energy_density + pressure)) / continuity_scale
    raychaudhuri_residual = abs(
        drho_d_n / (6.0 * m_pl**2) + (x_value * c_value + y_value * z_value) / m_pl**2
    )
    friedmann_residual = abs(3.0 * m_pl**2 * hubble**2 - energy_density)
    return {
        "a": scale_factor,
        "X": x_value,
        "Y": y_value,
        **terms,
        "C": c_value,
        "kinetic_phi": kinetic_phi,
        "kinetic_cross": kinetic_cross,
        "kinetic_chi": kinetic_chi,
        "kinetic_determinant": kinetic_determinant,
        "kinetic_min_eigenvalue": kinetic_min,
        "kinetic_max_eigenvalue": kinetic_max,
        "gradient_min_eigenvalue": min(c_value, z_value),
        "sound_speed_squared_min": sound_min,
        "sound_speed_squared_max": sound_max,
        "pressure": pressure,
        "energy_density": energy_density,
        "equation_of_state": pressure / energy_density,
        "H": hubble,
        "dX_dln_a": dx_d_n,
        "dY_dln_a": dy_d_n,
        "phi_current_relative_residual": abs(phi_current - j_phi) / j_phi,
        "chi_current_relative_residual": abs(chi_current - j_chi) / j_chi,
        "phi_eom_residual": phi_eom_residual,
        "chi_eom_residual": chi_eom_residual,
        "continuity_relative_residual": continuity_residual,
        "friedmann_absolute_residual": friedmann_residual,
        "raychaudhuri_absolute_residual": raychaudhuri_residual,
    }


def build_trajectory(config: Mapping[str, Any]) -> list[dict[str, float]]:
    witness = config["frozen_witness"]
    beta = float(witness["beta"])
    m_pl = float(witness["M_Pl"])
    x_initial = math.sqrt(float(witness["initial_u"]) / beta)
    y_initial = float(witness["initial_Y"])
    j_phi, j_chi = initial_currents(x_initial, y_initial, beta)
    count = int(witness["grid_points"])
    log_final = math.log(float(witness["final_scale_factor"]))
    trajectory: list[dict[str, float]] = []
    for index in range(count):
        scale_factor = math.exp(log_final * index / (count - 1))
        x_value, y_value = solve_invariants(
            scale_factor,
            j_phi,
            j_chi,
            beta,
            int(witness["root_iterations"]),
        )
        trajectory.append(evaluate_point(scale_factor, x_value, y_value, beta, m_pl, j_phi, j_chi))
    return trajectory


def symbolic_checks() -> dict[str, bool]:
    x, y, beta = sp.symbols("x y beta", positive=True)
    amplitude = sp.symbols("A", positive=True)
    u = beta * x**2
    z = (1 + u) ** 2
    zx = sp.diff(z, x)
    zxx = sp.diff(zx, x)
    lagrangian = x + z * y
    c = sp.diff(lagrangian, x)
    rho = 2 * x * sp.diff(lagrangian, x) + 2 * y * sp.diff(lagrangian, y) - lagrangian
    mixing = sp.expand(z * (zx + 2 * x * zxx) - 4 * x * zx**2)
    expected_mixing = 12 * u * (1 + u) ** 2 * (1 - 3 * u) / x
    kinetic_det = sp.expand(z * (1 + y * (zx + 2 * x * zxx)) - 4 * x * y * zx**2)
    reduced_current = sp.sqrt(2 * x) * (1 + amplitude * zx / z**2)
    expected_current_derivative = (1 + amplitude * mixing / z**3) / sp.sqrt(2 * x)
    phi_velocity, chi_velocity, phi_gradient, chi_gradient = sp.symbols(
        "phi_velocity chi_velocity phi_gradient chi_gradient", real=True
    )
    x_jet = (phi_velocity**2 - phi_gradient**2) / 2
    y_jet = (chi_velocity**2 - chi_gradient**2) / 2
    z_jet = (1 + beta * x_jet**2) ** 2
    jet_lagrangian = x_jet + z_jet * y_jet
    jet_background = {
        phi_velocity: sp.sqrt(2 * x),
        chi_velocity: sp.sqrt(2 * y),
        phi_gradient: 0,
        chi_gradient: 0,
    }
    velocity_hessian = sp.hessian(jet_lagrangian, (phi_velocity, chi_velocity)).subs(jet_background)
    negative_gradient_hessian = -sp.hessian(jet_lagrangian, (phi_gradient, chi_gradient)).subs(
        jet_background
    )
    expected_velocity_hessian = sp.Matrix(
        [
            [1 + y * (zx + 2 * x * zxx), 2 * zx * sp.sqrt(x * y)],
            [2 * zx * sp.sqrt(x * y), z],
        ]
    )
    expected_gradient_hessian = sp.diag(c, z)
    adm_kinetic_matrix = sp.Matrix(
        [
            [c + 2 * x * sp.diff(c, x), 2 * sp.sqrt(x * y) * sp.diff(z, x)],
            [2 * sp.sqrt(x * y) * sp.diff(z, x), z],
        ]
    )
    adm_gradient_matrix = sp.diag(c, z)
    velocity_hessian = velocity_hessian.applyfunc(sp.simplify)
    negative_gradient_hessian = negative_gradient_hessian.applyfunc(sp.simplify)
    return {
        "W02_COVARIANT_TO_FLRW_STRESS": sp.simplify(rho - (x + z * y + 2 * x * y * zx)) == 0,
        "W03_SCALAR_CURRENT_REDUCTION": sp.simplify(c - (1 + y * zx)) == 0,
        "W04_EXACT_GATE_DERIVATIVES": sp.simplify(zx - 4 * beta * x * (1 + u)) == 0
        and sp.simplify(zxx - 4 * beta * (1 + 3 * u)) == 0,
        "W05_EXACT_M_FACTOR": sp.simplify(mixing - expected_mixing) == 0,
        "W13_KINETIC_MATRIX_POSITIVE": sp.simplify(kinetic_det - (z + y * mixing)) == 0,
        "W19_UNIQUE_POSITIVE_CURRENT_BRANCH": sp.simplify(
            sp.diff(reduced_current, x) - expected_current_derivative
        )
        == 0,
        "W20_INDEPENDENT_QUADRATIC_HESSIAN": velocity_hessian.equals(expected_velocity_hessian)
        and negative_gradient_hessian.equals(expected_gradient_hessian),
        "W21_GENERAL_MULTIFIELD_ADM_MAPPING": adm_kinetic_matrix.equals(expected_velocity_hessian)
        and adm_gradient_matrix.equals(expected_gradient_hessian),
    }


def _trajectory_digest(trajectory: Sequence[Mapping[str, float]]) -> str:
    return _content_sha256(trajectory)


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    trajectory = build_trajectory(config)
    witness = config["frozen_witness"]
    tolerance = float(witness["residual_tolerance"])
    margin = float(witness["strict_health_margin"])
    symbolic = symbolic_checks()

    maxima = {
        "phi_current_relative_residual": max(
            row["phi_current_relative_residual"] for row in trajectory
        ),
        "chi_current_relative_residual": max(
            row["chi_current_relative_residual"] for row in trajectory
        ),
        "phi_eom_residual": max(row["phi_eom_residual"] for row in trajectory),
        "chi_eom_residual": max(row["chi_eom_residual"] for row in trajectory),
        "continuity_relative_residual": max(
            row["continuity_relative_residual"] for row in trajectory
        ),
        "friedmann_absolute_residual": max(
            row["friedmann_absolute_residual"] for row in trajectory
        ),
        "raychaudhuri_absolute_residual": max(
            row["raychaudhuri_absolute_residual"] for row in trajectory
        ),
    }
    extrema = {
        "u_min": min(row["u"] for row in trajectory),
        "u_max": max(row["u"] for row in trajectory),
        "M_min": min(row["M"] for row in trajectory),
        "gradient_min_eigenvalue": min(row["gradient_min_eigenvalue"] for row in trajectory),
        "kinetic_min_eigenvalue": min(row["kinetic_min_eigenvalue"] for row in trajectory),
        "kinetic_determinant_min": min(row["kinetic_determinant"] for row in trajectory),
        "sound_speed_squared_min": min(row["sound_speed_squared_min"] for row in trajectory),
        "sound_speed_squared_max": max(row["sound_speed_squared_max"] for row in trajectory),
        "equation_of_state_min": min(row["equation_of_state"] for row in trajectory),
        "equation_of_state_max": max(row["equation_of_state"] for row in trajectory),
    }
    current_ok = (
        maxima["phi_current_relative_residual"] < tolerance
        and maxima["chi_current_relative_residual"] < tolerance
        and maxima["phi_eom_residual"] < tolerance
        and maxima["chi_eom_residual"] < tolerance
    )
    checks = {
        "W01_PREDECESSOR_BYTES_AND_RECEIPTS": True,
        **symbolic,
        "W06_INITIAL_DATA_RECONSTRUCTION": math.isclose(
            trajectory[0]["u"], float(witness["initial_u"]), rel_tol=0.0, abs_tol=tolerance
        )
        and math.isclose(
            trajectory[0]["Y"], float(witness["initial_Y"]), rel_tol=0.0, abs_tol=tolerance
        ),
        "W07_CURRENT_CONSERVATION": current_ok,
        "W08_CONTINUITY_EQUATION": maxima["continuity_relative_residual"] < tolerance,
        "W09_FRIEDMANN_EQUATION": maxima["friedmann_absolute_residual"] < tolerance,
        "W10_RAYCHAUDHURI_EQUATION": maxima["raychaudhuri_absolute_residual"] < tolerance,
        "W11_BOUNDED_POSITIVE_M_DOMAIN": extrema["u_max"] < 1.0 / 3.0 and extrema["M_min"] > margin,
        "W12_GRADIENT_MATRIX_POSITIVE": extrema["gradient_min_eigenvalue"] > margin,
        "W13_KINETIC_MATRIX_POSITIVE": symbolic["W13_KINETIC_MATRIX_POSITIVE"]
        and extrema["kinetic_min_eigenvalue"] > margin
        and extrema["kinetic_determinant_min"] > margin,
        "W14_SOUND_SPEED_EIGENVALUES_POSITIVE": extrema["sound_speed_squared_min"] > margin,
        "W15_EXPANDING_BRANCH_MOVES_AWAY_FROM_THRESHOLD": all(
            trajectory[index]["u"] > trajectory[index + 1]["u"]
            for index in range(len(trajectory) - 1)
        ),
        "W16_CONSTANT_KINETIC_ESCAPE_BOUND": config["escape_architecture"][
            "healthy_complete_theory_claim"
        ]
        is False,
        "W17_CLAIM_CEILING": config["claim_boundary"][
            "coupled_metric_and_scalar_background_on_shell"
        ]
        is True
        and config["claim_boundary"]["publication_ready"] is False
        and config["claim_boundary"]["historical_novelty_established"] is False
        and config["claim_boundary"]["metric_cone_subluminality"] is False,
        "W18_SUPERLUMINAL_WARNING_RETAINED": extrema["sound_speed_squared_max"] > 1.0
        and config["frozen_witness"]["expected"]["all_sound_speeds_metric_subluminal"] is False,
    }
    if set(checks) != set(config["required_checks"]):
        raise KineticGateFlrwWitnessError("required check inventory changed")
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise KineticGateFlrwWitnessError(f"FLRW witness checks failed: {failed}")

    checkpoints = [trajectory[index] for index in (0, 20, 40, 60, 80)]
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-matter-lensing-kinetic-gate-flrw-bounded-witness-receipt-1.0",
        "analysis_id": config["analysis_id"],
        "status": "PASS_COUPLED_EINSTEIN_SCALAR_ON_SHELL_POSITIVE_BLOCK_WITNESS_SUPERLUMINAL_WARNING",
        "implementation_binding": {
            "module_path": config["implementation"]["module_path"],
            "module_sha256": _sha256_file(base / config["implementation"]["module_path"]),
            "test_path": config["implementation"]["test_path"],
            "test_sha256": _sha256_file(base / config["implementation"]["test_path"]),
        },
        "predecessor_receipt_content_sha256": {
            predecessor["id"]: predecessor["receipt_content_sha256"]
            for predecessor in config["predecessors"]
        },
        "covariant_action": config["covariant_action"],
        "flrw_reduction": config["flrw_reduction"],
        "principal_health": config["principal_health"],
        "primary_literature_positioning": config["primary_literature_positioning"],
        "witness": {
            "parameters": config["frozen_witness"],
            "trajectory_points": len(trajectory),
            "trajectory_sha256": _trajectory_digest(trajectory),
            "checkpoints": checkpoints,
            "residual_maxima": maxima,
            "health_extrema": extrema,
        },
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "escape_architecture": config["escape_architecture"],
        "claim_boundary": config["claim_boundary"],
        "publication_value": config["publication_value"],
        "zero_access": config["zero_access"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return "EXISTING_IDENTICAL"
        raise KineticGateFlrwWitnessError(f"refusing to replace existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    return _atomic_no_clobber(base / config["output_path"], build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    stored = _read_json(base / config["output_path"])
    expected = build_receipt(base)
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise KineticGateFlrwWitnessError("stored FLRW witness receipt changed")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    else:
        receipt = check_receipt()
        if args.command == "check":
            print("VALID")
        else:
            print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
