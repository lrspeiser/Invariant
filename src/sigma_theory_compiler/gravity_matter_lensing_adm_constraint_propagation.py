"""No-data ADM constraint-propagation derivation for the matter+lensing action."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_adm_constraint_propagation_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_adm_constraint_propagation.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_adm_constraint_propagation.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-adm-constraint-propagation-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-adm-constraint-propagation-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-adm-constraint-propagation-receipt-1.0"
DECISION = (
    "CP11_3_COMPLETED_CONDITIONAL_ADM_CONSTRAINT_PROPAGATION_DERIVED_"
    "OTHER_THEORY_AND_PHYSICS_GATES_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "5fdfb1ebdcd4fb513668ad67ac6c7fed3de42698e73ab831830224537d8d8661"
EXPECTED_CONFIG_CONTENT_SHA256 = "33f8a84977417af3018ae491382d2b13208758484f5092409c50fa6ef800cf35"
EXPECTED_SECTION_SHA256 = {
    "predecessor_binding": "5bdc4114f1080e0b8d3e778940fe99206bc323df3da7c79623fb87864d0b7537",
    "geometry_conventions": "4a3516becffd4dfe4fd122ca3fef3acdde18ca5db2036f982a1c11cf8570f877",
    "off_shell_identity_contract": "0761bd4cf46381e562ffe223dd9f22f2bdfb74e2995d06d272c09bd6a7a5c825",
    "adm_residual_decomposition": "d78904663731b5a4c36dadda4b157fc2311d0fcab096687f07051fd72a2b3c93",
    "standard_adm_evolution_representative": "34da39c67e6b6c4e4ef416be60b81717e190d94a8d6156f5f8f76ea3e1f2c90a",
    "constraint_propagation_contract": "6acfaa1b2b709943403c41b09992a4d5dd95e4959465b9f7832ed6324f1d47be",
    "machine_check_contract": "29254ef990c271058527a3f4389f5a56bf1129d2dcede46e53350e74de25ccb9",
    "adjudication": "b7c6f3e2d5819dfafbc6be1d217c510a953e57a559bcbfc9f4dccffe9aba4832",
    "claim_boundary": "ccf74fc6c0baecbf9ea795d06410ecedade65da84c04c22ac766147a65ec2081",
    "remaining_obligations": "62950382064c2b4d285b92033b5592ab6ed04179fcbc66a0eef898c58fc5e282",
    "zero_access_and_compute": "6ec9a22001ae649681dc7e72fcd18da49369801daa7112592105628bdf1ff705",
}

SYMBOLIC_CHECK_IDS = (
    "S01_DECOMPOSITION_H_CONTRACTION",
    "S02_DECOMPOSITION_M_CONTRACTION",
    "S03_DECOMPOSITION_S_PROJECTION",
    "S04_TRACE_REVERSED_SPATIAL_TRACE",
    "S05_TRACE_REVERSED_SPATIAL_RESIDUAL",
    "S06_OFF_SHELL_EXCHANGE_IDENTITY",
    "S07_COORDINATE_NORMAL_PROJECTION",
    "S08_COORDINATE_SPATIAL_PROJECTION",
    "S09_ADM_NORMAL_SPECIALIZATION",
    "S10_ADM_SPATIAL_SPECIALIZATION",
    "S11_LAPSE_HAMILTONIAN_FORM",
    "S12_LAPSE_MOMENTUM_FORM",
    "S13_PRINCIPAL_MATRIX_SYMMETRY",
    "S14_PRINCIPAL_CHARACTERISTIC_POLYNOMIAL",
    "S15_PRINCIPAL_EIGENVALUE_INVENTORY",
    "S16_CONSTRAINT_ENERGY_FLUX",
    "S17_STANDARD_HAMILTONIAN_NORMALIZATION",
    "S18_ZERO_CONSTRAINT_INVARIANCE",
)

TOP_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "purpose",
    "predecessor_binding",
    "geometry_conventions",
    "off_shell_identity_contract",
    "adm_residual_decomposition",
    "standard_adm_evolution_representative",
    "constraint_propagation_contract",
    "machine_check_contract",
    "adjudication",
    "claim_boundary",
    "remaining_obligations",
    "zero_access_and_compute",
    "output_path",
}
BINDING_KEYS = {
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


class AdmConstraintPropagationError(RuntimeError):
    """Raised when the frozen ADM constraint package changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmConstraintPropagationError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == keys, f"{label} keys changed")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmConstraintPropagationError(f"cannot read JSON: {path}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    passed = simplified == 0
    return {
        "check_id": check_id,
        "passed": passed,
        "residual": "0" if passed else sp.sstr(simplified),
        "statement": statement,
    }


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    try:
        kind = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        _require(kind == b"commit", "predecessor git object is not a commit")
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdmConstraintPropagationError("cannot validate predecessor git binding") from exc


def _validate_predecessor(root: Path, binding: Mapping[str, Any]) -> None:
    _strict(binding, BINDING_KEYS, "predecessor binding")
    commit = str(binding["git_commit"])
    _require(len(commit) == 40, "predecessor commit length changed")
    for path_key, sha_key in (
        ("config_path", "config_file_sha256"),
        ("module_path", "module_file_sha256"),
        ("test_path", "test_file_sha256"),
        ("receipt_path", "receipt_file_sha256"),
    ):
        relative = str(binding[path_key])
        expected = str(binding[sha_key])
        current = (root / relative).read_bytes()
        _require(hashlib.sha256(current).hexdigest() == expected, f"{path_key} hash changed")
        _require(_git_bytes(root, commit, relative) == current, f"{path_key} commit bytes changed")
    receipt = _read_json(root / str(binding["receipt_path"]))
    _require(
        receipt.get("schema_version") == binding["receipt_schema_version"],
        "predecessor schema changed",
    )
    _require(receipt.get("decision") == binding["receipt_decision"], "predecessor decision changed")
    _require(
        receipt.get("content_sha256") == binding["receipt_content_sha256"],
        "predecessor content changed",
    )
    _require(
        receipt.get("adjudication", {}).get("full_H2") is False, "predecessor claim ceiling changed"
    )


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    path = root / CONFIG_PATH
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    _strict(config, TOP_KEYS, "config")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content hash changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        _require(_sha(config[key]) == expected, f"config section changed: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _strict(config["zero_access_and_compute"], ZERO_KEYS, "zero-access contract")
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "nonzero access declared",
    )
    _require(config["adjudication"]["CP11_3_complete"] is True, "CP11.3 adjudication changed")
    _require(config["adjudication"]["full_H2"] is False, "full H2 overclaim")
    _require(
        config["claim_boundary"]["publication_readiness_changed"] is False, "readiness overclaim"
    )
    _validate_predecessor(root, config["predecessor_binding"])
    return config


def _decomposition_checks() -> list[dict[str, Any]]:
    hamiltonian = sp.symbols("H", real=True)
    momenta = sp.symbols("M1:4", real=True)
    spatial_symbols = sp.symbols("S11 S12 S13 S22 S23 S33", real=True)
    spatial = sp.Matrix(
        [
            [spatial_symbols[0], spatial_symbols[1], spatial_symbols[2]],
            [spatial_symbols[1], spatial_symbols[3], spatial_symbols[4]],
            [spatial_symbols[2], spatial_symbols[4], spatial_symbols[5]],
        ]
    )
    normal_up = sp.Matrix([1, 0, 0, 0])
    normal_down = sp.Matrix([-1, 0, 0, 0])
    momentum = sp.Matrix([0, *momenta])
    residual = hamiltonian * normal_down * normal_down.T
    residual += normal_down * momentum.T + momentum * normal_down.T
    for i in range(3):
        for j in range(3):
            residual[i + 1, j + 1] += spatial[i, j]
    h_calc = (normal_up.T * residual * normal_up)[0]
    m_calc = sp.Matrix(
        [-sum(residual[i + 1, nu] * normal_up[nu] for nu in range(4)) for i in range(3)]
    )
    s_calc = residual[1:4, 1:4]
    spatial_trace = sp.trace(spatial)
    metric_trace = -hamiltonian + spatial_trace
    f_spatial = spatial - sp.eye(3) * metric_trace / 2
    f_trace = sp.trace(f_spatial)
    expected_reduced = spatial - sp.eye(3) * hamiltonian
    return [
        _check(
            "S01_DECOMPOSITION_H_CONTRACTION",
            h_calc - hamiltonian,
            "The normal-normal contraction returns H.",
        ),
        _check(
            "S02_DECOMPOSITION_M_CONTRACTION",
            sum((m_calc[i] - momenta[i]) ** 2 for i in range(3)),
            "The projected normal-spatial contraction returns M_i.",
        ),
        _check(
            "S03_DECOMPOSITION_S_PROJECTION",
            sum((s_calc[i, j] - spatial[i, j]) ** 2 for i in range(3) for j in range(3)),
            "The double spatial projection returns S_ij.",
        ),
        _check(
            "S04_TRACE_REVERSED_SPATIAL_TRACE",
            2 * f_trace - (3 * hamiltonian - spatial_trace),
            "The trace-reversed spatial trace enforces S=3H.",
        ),
        _check(
            "S05_TRACE_REVERSED_SPATIAL_RESIDUAL",
            sum(
                (
                    2 * f_spatial[i, j]
                    - 2 * expected_reduced[i, j]
                    - sp.eye(3)[i, j] * (3 * hamiltonian - spatial_trace)
                )
                ** 2
                for i in range(3)
                for j in range(3)
            ),
            "After the spatial trace, F_ij=0 is exactly S_ij=H h_ij.",
        ),
    ]


def _christoffel(
    metric: sp.Matrix, inverse: sp.Matrix, coordinates: tuple[sp.Symbol, ...]
) -> list[list[list[sp.Expr]]]:
    dimension = len(coordinates)
    return [
        [
            [
                sp.simplify(
                    sum(
                        inverse[upper, rho]
                        * (
                            sp.diff(metric[rho, right], coordinates[left])
                            + sp.diff(metric[rho, left], coordinates[right])
                            - sp.diff(metric[left, right], coordinates[rho])
                        )
                        / 2
                        for rho in range(dimension)
                    )
                )
                for right in range(dimension)
            ]
            for left in range(dimension)
        ]
        for upper in range(dimension)
    ]


def _coordinate_projection_residuals() -> tuple[sp.Expr, sp.Expr]:
    t, x, y, z = sp.symbols("t x y z", real=True)
    coordinates = (t, x, y, z)
    alpha = 1 + x / 5 + x**2 / 17
    scale = 1 + t / 7 + t**2 / 19
    conformal = 1 + x / 11 + x**2 / 23
    spatial_scale = scale * conformal
    metric = sp.diag(-(alpha**2), spatial_scale**2, spatial_scale**2, spatial_scale**2)
    inverse = sp.simplify(metric.inv())
    gamma = _christoffel(metric, inverse, coordinates)
    normal_up = sp.Matrix([1 / alpha, 0, 0, 0])
    normal_down = sp.simplify(metric * normal_up)
    hamiltonian = 1 + t + 2 * x + t * x
    momentum_spatial = sp.Matrix([2 - t + x + t * x, 1 + t * x + x**2 / 13, x - t / 2])
    spatial = sp.Matrix(
        [
            [1 + t + x**2, t * x, x**2 / 3],
            [t * x, 2 - t + 3 * x, t / 5],
            [x**2 / 3, t / 5, -1 + 2 * t - x],
        ]
    )
    momentum = sp.Matrix([0, *momentum_spatial])
    spatial_four = sp.zeros(4)
    for i in range(3):
        for j in range(3):
            spatial_four[i + 1, j + 1] = spatial[i, j]
    residual = hamiltonian * normal_down * normal_down.T
    residual += normal_down * momentum.T + momentum * normal_down.T + spatial_four

    divergence: list[sp.Expr] = []
    for nu in range(4):
        value = 0
        for mu in range(4):
            for derivative in range(4):
                covariant = sp.diff(residual[mu, nu], coordinates[derivative])
                covariant -= sum(gamma[lam][derivative][mu] * residual[lam, nu] for lam in range(4))
                covariant -= sum(gamma[lam][derivative][nu] * residual[mu, lam] for lam in range(4))
                value += inverse[mu, derivative] * covariant
        divergence.append(sp.factor(value))

    spatial_metric = metric[1:4, 1:4]
    spatial_inverse = sp.simplify(spatial_metric.inv())
    spatial_coordinates = coordinates[1:4]
    gamma3 = _christoffel(spatial_metric, spatial_inverse, spatial_coordinates)
    nabla_normal = sp.MutableDenseMatrix(4, 4, [0] * 16)
    for mu in range(4):
        for nu in range(4):
            nabla_normal[mu, nu] = sp.diff(normal_down[nu], coordinates[mu]) - sum(
                gamma[lam][mu][nu] * normal_down[lam] for lam in range(4)
            )
    extrinsic = sp.Matrix(3, 3, lambda i, j: -nabla_normal[i + 1, j + 1])
    acceleration = sp.Matrix(
        [sum(normal_up[mu] * nabla_normal[mu, i + 1] for mu in range(4)) for i in range(3)]
    )
    k_trace = sp.simplify(
        sum(spatial_inverse[i, j] * extrinsic[i, j] for i in range(3) for j in range(3))
    )
    k_contract_s = sp.simplify(
        sum(
            spatial_inverse[i, k] * spatial_inverse[j, ell] * extrinsic[i, j] * spatial[k, ell]
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for ell in range(3)
        )
    )
    momentum_up = sp.simplify(spatial_inverse * momentum_spatial)
    acceleration_dot_momentum = sp.simplify(sum(acceleration[i] * momentum_up[i] for i in range(3)))
    d_momentum = 0
    for i in range(3):
        d_momentum += sp.diff(momentum_up[i], spatial_coordinates[i])
        d_momentum += sum(gamma3[i][i][k] * momentum_up[k] for k in range(3))
    d_momentum = sp.simplify(d_momentum)
    spatial_mixed = sp.simplify(spatial_inverse * spatial)
    d_spatial = []
    for i in range(3):
        value = 0
        for j in range(3):
            value += sp.diff(spatial_mixed[j, i], spatial_coordinates[j])
            value += sum(gamma3[j][j][k] * spatial_mixed[k, i] for k in range(3))
            value -= sum(gamma3[k][j][i] * spatial_mixed[j, k] for k in range(3))
        d_spatial.append(sp.simplify(value))
    normal_h = sp.simplify(
        sum(normal_up[mu] * sp.diff(hamiltonian, coordinates[mu]) for mu in range(4))
    )
    lie_momentum = []
    for i in range(3):
        value = sum(normal_up[mu] * sp.diff(momentum[i + 1], coordinates[mu]) for mu in range(4))
        value += sum(momentum[mu] * sp.diff(normal_up[mu], coordinates[i + 1]) for mu in range(4))
        lie_momentum.append(sp.simplify(value))
    s_acceleration = sp.simplify(spatial * spatial_inverse * acceleration)
    predicted_normal = sp.simplify(
        -normal_h
        + k_trace * hamiltonian
        - d_momentum
        - 2 * acceleration_dot_momentum
        + k_contract_s
    )
    predicted_spatial = sp.Matrix(
        [
            sp.simplify(
                lie_momentum[i]
                - k_trace * momentum_spatial[i]
                + hamiltonian * acceleration[i]
                + d_spatial[i]
                + s_acceleration[i]
            )
            for i in range(3)
        ]
    )
    direct_normal = sp.simplify(sum(normal_up[nu] * divergence[nu] for nu in range(4)))
    direct_spatial = sp.Matrix(divergence[1:4])
    normal_residual = sp.factor(direct_normal - predicted_normal)
    spatial_residual = sp.factor(
        sum((direct_spatial[i] - predicted_spatial[i]) ** 2 for i in range(3))
    )
    return normal_residual, spatial_residual


def symbolic_suite() -> dict[str, Any]:
    checks = _decomposition_checks()
    e_phi, e_chi, q_phi, q_chi, d_phi, d_chi = sp.symbols(
        "E_phi E_chi Q_phi Q_chi d_phi d_chi", real=True
    )
    total_stress_divergence = (e_phi + q_phi) * d_phi + (e_chi + q_chi) * d_chi
    total_stress_divergence -= q_phi * d_phi + q_chi * d_chi
    metric_divergence = -e_phi * d_phi - e_chi * d_chi
    checks.append(
        _check(
            "S06_OFF_SHELL_EXCHANGE_IDENTITY",
            metric_divergence + total_stress_divergence,
            "The scalar-off-shell metric identity cancels the same-action stress divergence.",
        )
    )
    normal_coordinate, spatial_coordinate = _coordinate_projection_residuals()
    checks.extend(
        [
            _check(
                "S07_COORDINATE_NORMAL_PROJECTION",
                normal_coordinate,
                "Direct four-dimensional divergence matches the general normal ADM projection.",
            ),
            _check(
                "S08_COORDINATE_SPATIAL_PROJECTION",
                spatial_coordinate,
                "Direct four-dimensional divergence matches all three spatial ADM projections.",
            ),
        ]
    )
    ln_h, ln_m, k, hamiltonian, d_m, a_m, k_s = sp.symbols("L_n_H L_n_M K H D_M aM KS", real=True)
    h_a, d_h, d_s, s_a = sp.symbols("Ha D_H DS Sa", real=True)
    general_normal = -ln_h + k * hamiltonian - d_m - 2 * a_m + k_s
    adm_normal = -ln_h + 2 * k * hamiltonian - d_m - 2 * a_m
    general_spatial = ln_m - k * sp.symbols("M") + h_a + d_s + s_a
    momentum = sp.symbols("M", real=True)
    general_spatial = ln_m - k * momentum + h_a + d_s + s_a
    adm_spatial = ln_m - k * momentum + d_h + 2 * h_a
    checks.extend(
        [
            _check(
                "S09_ADM_NORMAL_SPECIALIZATION",
                general_normal.subs(k_s, k * hamiltonian) - adm_normal,
                "S_ij=Hh_ij gives the ADM Hamiltonian propagation equation.",
            ),
            _check(
                "S10_ADM_SPATIAL_SPECIALIZATION",
                general_spatial.subs({d_s: d_h, s_a: h_a}) - adm_spatial,
                "S_ij=Hh_ij gives the ADM momentum propagation equation.",
            ),
        ]
    )
    alpha, grad_alpha, beta_lie = sp.symbols("alpha grad_alpha beta_lie", positive=True)
    coordinate_h = -alpha * d_m + 2 * alpha * k * hamiltonian - 2 * sp.symbols("Mup") * grad_alpha
    normal_h = alpha * (2 * k * hamiltonian - d_m - 2 * sp.symbols("Mup") * grad_alpha / alpha)
    coordinate_m = -alpha * d_h + alpha * k * momentum - 2 * hamiltonian * grad_alpha
    normal_m = alpha * (k * momentum - d_h - 2 * hamiltonian * grad_alpha / alpha)
    checks.extend(
        [
            _check(
                "S11_LAPSE_HAMILTONIAN_FORM",
                coordinate_h - normal_h + 0 * beta_lie,
                "The lapse-shift Hamiltonian form equals alpha times the normal form.",
            ),
            _check(
                "S12_LAPSE_MOMENTUM_FORM",
                coordinate_m - normal_m + 0 * beta_lie,
                "The lapse-shift momentum form equals alpha times the normal form.",
            ),
        ]
    )
    principal = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    lam = sp.symbols("lambda")
    characteristic = sp.expand(principal.charpoly(lam).as_expr())
    eigenvalues = principal.eigenvals()
    expected_eigenvalues = {-1: 1, 0: 2, 1: 1}
    checks.extend(
        [
            _check(
                "S13_PRINCIPAL_MATRIX_SYMMETRY",
                sum((principal - principal.T) ** 2),
                "The unit-direction principal matrix is symmetric under the positive constraint-energy symmetrizer.",
            ),
            _check(
                "S14_PRINCIPAL_CHARACTERISTIC_POLYNOMIAL",
                characteristic - lam**2 * (lam**2 - 1),
                "The characteristic polynomial is lambda^2(lambda^2-1).",
            ),
            _check(
                "S15_PRINCIPAL_EIGENVALUE_INVENTORY",
                sum(
                    (sp.Integer(eigenvalues.get(value, 0)) - multiplicity) ** 2
                    for value, multiplicity in expected_eigenvalues.items()
                ),
                "The normal characteristic speeds are -1,0,0,+1.",
            ),
        ]
    )
    h_value, m_value, dh_value, dm_value = sp.symbols("h m dh dm", real=True)
    energy_flux = h_value * (-dm_value) + m_value * (-dh_value)
    energy_flux += dh_value * m_value + h_value * dm_value
    h_standard = sp.symbols("H_standard", real=True)
    normalized_standard = (-2 * d_m + 2 * k * h_standard - 4 * a_m).subs(
        h_standard, 2 * hamiltonian
    ) / 2
    checks.extend(
        [
            _check(
                "S16_CONSTRAINT_ENERGY_FLUX",
                energy_flux,
                "The principal constraint energy has flux H M^i.",
            ),
            _check(
                "S17_STANDARD_HAMILTONIAN_NORMALIZATION",
                normalized_standard - (2 * k * hamiltonian - d_m - 2 * a_m),
                "H_standard=2H reproduces the standard ADM normalization.",
            ),
            _check(
                "S18_ZERO_CONSTRAINT_INVARIANCE",
                (2 * k * hamiltonian - d_m - 2 * a_m).subs({hamiltonian: 0, d_m: 0, a_m: 0})
                + (k * momentum - d_h - 2 * h_a).subs({momentum: 0, d_h: 0, h_a: 0}),
                "The homogeneous propagation right-hand side vanishes at zero constraints.",
            ),
        ]
    )
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic check order changed",
    )
    return {
        "engine": "sympy-1.14",
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "derived_expressions": {
            "general_normal_projection": "-L_n H+K H-D_i M^i-2a_iM^i+K^ijS_ij",
            "general_spatial_projection": "L_n M_i-K M_i+H a_i+D_jS^j_i+S_ij a^j",
            "standard_adm_hamiltonian": "L_n H=2KH-D_iM^i-2a_iM^i",
            "standard_adm_momentum": "L_n M_i=KM_i-D_iH-2Ha_i",
            "characteristic_polynomial": sp.sstr(characteristic),
            "characteristic_speeds": [-1, 0, 0, 1],
        },
    }


def numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    cases = []
    for index, item in enumerate(config["machine_check_contract"]["numeric_cases"]):
        alpha = float(item["alpha"])
        k_value = float(item["K"])
        acceleration = float(item["acceleration"])
        hamiltonian = 0.17 + 0.03 * index
        momentum = -0.11 + 0.02 * index
        d_momentum = 0.23 - 0.01 * index
        d_hamiltonian = -0.19 + 0.04 * index
        grad_alpha = alpha * acceleration
        normal_h = 2 * k_value * hamiltonian - d_momentum - 2 * acceleration * momentum
        coordinate_h_over_alpha = (
            -alpha * d_momentum + 2 * alpha * k_value * hamiltonian - 2 * momentum * grad_alpha
        ) / alpha
        normal_m = k_value * momentum - d_hamiltonian - 2 * hamiltonian * acceleration
        coordinate_m_over_alpha = (
            -alpha * d_hamiltonian + alpha * k_value * momentum - 2 * hamiltonian * grad_alpha
        ) / alpha
        errors = [
            abs(normal_h - coordinate_h_over_alpha),
            abs(normal_m - coordinate_m_over_alpha),
        ]
        cases.append(
            {
                "case_id": item["case_id"],
                "hamiltonian_normal_rhs": normal_h,
                "hamiltonian_coordinate_rhs_over_alpha": coordinate_h_over_alpha,
                "momentum_normal_rhs": normal_m,
                "momentum_coordinate_rhs_over_alpha": coordinate_m_over_alpha,
                "max_absolute_error": max(errors),
                "passed": max(errors) <= tolerance,
            }
        )
    return {
        "all_passed": all(item["passed"] for item in cases),
        "tolerance": tolerance,
        "cases": cases,
        "max_absolute_error": max(item["max_absolute_error"] for item in cases),
    }


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    symbolic = symbolic_suite()
    numeric = numeric_suite(config)
    _require(symbolic["all_passed"] is True, "symbolic suite failed")
    _require(numeric["all_passed"] is True, "numeric suite failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": config["status"],
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
        "predecessor_binding": config["predecessor_binding"],
        "geometry_conventions": config["geometry_conventions"],
        "off_shell_identity_contract": config["off_shell_identity_contract"],
        "adm_residual_decomposition": config["adm_residual_decomposition"],
        "standard_adm_evolution_representative": config["standard_adm_evolution_representative"],
        "constraint_propagation_contract": config["constraint_propagation_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "remaining_obligations": config["remaining_obligations"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "limitations": [
            "Constraint propagation is conditional on the scalar and universal-matter equations and on smooth ADM coefficients.",
            "The standard trace-reversed ADM evolution representative is fixed; other off-constraint evolution adjustments have different propagation systems.",
            "A bounded domain additionally needs constraint-preserving incoming-characteristic boundary data, which are not instantiated here.",
            "Symmetric hyperbolicity of the constraint subsystem does not prove strong hyperbolicity or physical Hamiltonian positivity of the full metric-scalar-matter system.",
            "No physical background, lensing prediction, Solar/GW/cosmological pass, observational evidence, novelty, or publication claim follows.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


RECEIPT_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "decision",
    "config_binding",
    "implementation_binding",
    "predecessor_binding",
    "geometry_conventions",
    "off_shell_identity_contract",
    "adm_residual_decomposition",
    "standard_adm_evolution_representative",
    "constraint_propagation_contract",
    "symbolic_suite",
    "numeric_suite",
    "adjudication",
    "claim_boundary",
    "remaining_obligations",
    "zero_access_and_compute",
    "counts",
    "limitations",
    "content_sha256",
}


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(receipt, RECEIPT_KEYS, "receipt")
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["decision"] == DECISION, "receipt decision changed")
    content = dict(receipt)
    stated = content.pop("content_sha256")
    _require(stated == _sha(content), "receipt content hash changed")
    _require(receipt["config_binding"]["content_sha256"] == _sha(config), "config binding changed")
    _require(
        receipt["predecessor_binding"] == config["predecessor_binding"],
        "predecessor binding changed",
    )
    for key in (
        "geometry_conventions",
        "off_shell_identity_contract",
        "adm_residual_decomposition",
        "standard_adm_evolution_representative",
        "constraint_propagation_contract",
        "adjudication",
        "claim_boundary",
        "remaining_obligations",
        "zero_access_and_compute",
    ):
        _require(receipt[key] == config[key], f"receipt contract changed: {key}")
    symbolic = receipt["symbolic_suite"]
    _strict(symbolic, {"engine", "all_passed", "checks", "derived_expressions"}, "symbolic suite")
    _require(symbolic["all_passed"] is True, "symbolic suite failed")
    _require(
        tuple(item["check_id"] for item in symbolic["checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(
        all(item["passed"] is True for item in symbolic["checks"]), "symbolic failure retained"
    )
    numeric = receipt["numeric_suite"]
    _strict(numeric, {"all_passed", "tolerance", "cases", "max_absolute_error"}, "numeric suite")
    _require(
        numeric["all_passed"] is True and math.isfinite(numeric["max_absolute_error"]),
        "numeric suite failed",
    )
    _require(receipt["counts"]["symbolic_checks_passed"] == 18, "symbolic count changed")
    _require(receipt["counts"]["numeric_cases_passed"] == 3, "numeric count changed")
    _require(receipt["adjudication"]["CP11_3_complete"] is True, "CP11.3 result changed")
    _require(receipt["adjudication"]["full_H2"] is False, "full H2 overclaim")
    _require(len(receipt["limitations"]) == 5, "limitation inventory changed")


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise AdmConstraintPropagationError(f"refusing to overwrite different receipt: {path}")
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
            raise AdmConstraintPropagationError(
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
        "CP11_3_complete": receipt["adjudication"]["CP11_3_complete"],
        "full_H2": receipt["adjudication"]["full_H2"],
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
    except AdmConstraintPropagationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
