"""Restricted combined aether-tetrad and quadrature-scalar hyperbolicity gate."""

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

from sigma_theory_compiler import adm_aether
from sigma_theory_compiler import (
    gravity_shared_quadrature_reduced_principal_factorization as reduced_factorization,
)

CONFIG_PATH = Path("configs/gravity_shared_quadrature_combined_tetrad_hyperbolicity_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_combined_tetrad_hyperbolicity.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-combined-tetrad-hyperbolicity-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-combined-tetrad-hyperbolicity-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-combined-tetrad-hyperbolicity-receipt-1.0"
STATUS = "restricted_static_W_zero_combined_tetrad_scalar_symmetric_hyperbolicity_no_data"
DECISION = (
    "RESTRICTED_STATIC_W_ZERO_COMBINED_TETRAD_SCALAR_SYMMETRIC_HYPERBOLICITY_DERIVED_"
    "AETHER_CHERENKOV_SAFE_LOCUS_SCALAR_TRANSVERSE_CHERENKOV_AND_GLOBAL_PHYSICAL_GATES_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "829b427663bf476f2bc2262a12af1f864399c7b22160ab5e501395377b11c4ec"
EXPECTED_TEST_FILE_SHA256 = "c04c0167e77d6c035989af228911ec3507c9f66847909bda95ee403cee54ec0b"
EXPECTED_SECTION_HASHES = {
    "predecessor_bindings": "9f8d26289cf7d12ead0d114d0daf0b01d49105fff2d841d573d652b2f1ea6a23",
    "frozen_branch_contract": "55d77701f2e5c555a775a5f257f9958a1d165635c5c815e9c5633cf10ec85525",
    "symmetric_hyperbolic_locus_contract": "78bc65c33331bd8d05ec4a4e1c5f44deec7ca1c0beb7a848f69962ae7dbd2a00",
    "scalar_first_order_contract": "b463df982365d72086abc5a0fa6ff1fc34b4e32f70192b71aa300753c4358b02",
    "combined_principal_contract": "0309e0f295b4501aeaa6dba8e29be3b50d2b4ff317aa1079f842559e6272838b",
    "ppn_and_physical_gate_boundary": "d4a8d54587cca5773e1e521703d399d6afeedd41078210f8cef62fcfcdeb5d6a",
    "obstruction_contract": "24b8ed5a34ca4d33f68258e265c2f4e5ced16c3e9118ef19085c43daa8aa80ac",
    "machine_check_contract": "8b925c6ffb814746cd79c244c66c1a38d1c82da85e451485726b3d52d9113c6b",
    "adjudication": "6bb8ad56a88a4c7e1eb5f95c0761896b48775d0583ef3e63b5f9eeaeee058a9a",
    "claim_boundary": "a3268d705148ecb9e46d5af0ea5a7626cb0db09186f860c75b91788d4e23af73",
    "zero_access_and_compute": "d49dd6f61c1704a662f2a82623b63bcb487c43018af38bad97d092c982b74e94",
}


class QuadratureCombinedHyperbolicityError(RuntimeError):
    """Raised when a frozen combined-hyperbolicity contract changes."""


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
        raise QuadratureCombinedHyperbolicityError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureCombinedHyperbolicityError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureCombinedHyperbolicityError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "frozen_branch_contract",
            "symmetric_hyperbolic_locus_contract",
            "scalar_first_order_contract",
            "combined_principal_contract",
            "ppn_and_physical_gate_boundary",
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
        or config["analysis_id"] != "gravity-shared-quadrature-combined-tetrad-hyperbolicity-v1"
        or config["status"]
        != "frozen_no_data_restricted_static_W_zero_combined_tetrad_symmetric_hyperbolicity"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureCombinedHyperbolicityError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureCombinedHyperbolicityError(f"config {key} changed")
    machine = config["machine_check_contract"]
    if len(machine["required_symbolic_checks"]) != 37 or len(machine["numeric_cases"]) != 4:
        raise QuadratureCombinedHyperbolicityError("machine inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureCombinedHyperbolicityError("adjudication changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureCombinedHyperbolicityError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureCombinedHyperbolicityError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureCombinedHyperbolicityError("predecessor Git binding failed") from error


def validate_predecessors(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        commit = str(binding["git_commit"])
        if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
            raise QuadratureCombinedHyperbolicityError("predecessor commit changed")
        for artifact in binding["artifacts"]:
            relative = str(artifact["path"])
            current = root / relative
            if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureCombinedHyperbolicityError("predecessor artifact changed")
            current_blob = _git("hash-object", "--path", relative, relative, root=root).strip()
            commit_blob = _git("rev-parse", f"{commit}:{relative}", root=root).strip()
            if current_blob != commit_blob:
                raise QuadratureCombinedHyperbolicityError("predecessor commit bytes changed")
        row = {
            "binding_id": binding["binding_id"],
            "git_commit": commit,
            "artifact_count": len(binding["artifacts"]),
            "valid": True,
        }
        if "receipt_path" in binding:
            receipt = _read_json(root / str(binding["receipt_path"]))
            body = {key: value for key, value in receipt.items() if key != "content_sha256"}
            if (
                receipt.get("content_sha256") != binding["receipt_content_sha256"]
                or _sha(body) != binding["receipt_content_sha256"]
                or receipt.get("schema_version") != binding["receipt_schema_version"]
                or receipt.get("decision") != binding["receipt_decision"]
            ):
                raise QuadratureCombinedHyperbolicityError("predecessor receipt changed")
            row["receipt_content_sha256"] = receipt["content_sha256"]
        else:
            if (
                binding.get("runtime_control")
                != "sigma_theory_compiler.adm_aether.einstein_aether_covariant_strong_hyperbolicity_control"
            ):
                raise QuadratureCombinedHyperbolicityError("formal runtime control changed")
            control = adm_aether.einstein_aether_covariant_strong_hyperbolicity_control()
            if (
                not control["passed"]
                or control["primary_source"] != binding["primary_source"]
                or control["physical_mode_count"] != 5
            ):
                raise QuadratureCombinedHyperbolicityError("formal runtime control failed")
            row["runtime_control_passed"] = True
            row["primary_source"] = control["primary_source"]
        validated.append(row)
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
    Delta, t, r, q, alpha = sp.symbols("Delta t r q alpha", positive=True, finite=True)
    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    varphi = sp.symbols("varphi", real=True, finite=True)
    omega, k_parallel, ky, kz = sp.symbols("omega k_parallel ky kz", real=True)

    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    vector_numerator = 2 * c1 - c1**2 + c3**2
    trace_factor = 2 + c13 + 3 * c2
    speed2 = sp.factor(1 / (1 - c13))
    speed1 = sp.factor(vector_numerator / (2 * c14 * (1 - c13)))
    speed0 = sp.factor(c123 * (2 - c14) / (c14 * (1 - c13) * trace_factor))
    energy1 = sp.factor(vector_numerator / (1 - c13))
    energy0 = sp.factor(c14 * (2 - c14))
    spin0_kinetic = sp.factor(c14**2 * (1 - c13) * trace_factor / c123)
    alpha1 = sp.factor(-8 * (c3**2 + c1 * c4) / vector_numerator)
    alpha2 = sp.factor(
        alpha1 / 2 - (c1 + 2 * c3 - c4) * (2 * c1 + 3 * c2 + c3 + c4) / (c123 * (2 - c14))
    )

    Gamma = sp.factor(1 / (1 - 2 * Delta))
    s1_squared = sp.factor(1 / (1 - Delta) ** 2)
    sigma = sp.factor(3 * Gamma * (1 + Delta * Gamma))
    source_s1_squared = sp.factor(
        Gamma
        * (2 + 3 * Delta * Gamma)
        / (1 + Delta * Gamma + (1 - Delta) * (1 + 3 * Delta * Gamma + 3 * Delta**2 * Gamma**2))
    )
    locus = {
        c1: sp.factor(Delta * s1_squared),
        c2: sp.factor(Delta * Gamma),
        c3: sp.factor(-Delta * s1_squared),
        c4: sp.factor(Delta - Delta * s1_squared),
    }
    Delta_parameterization = t / (40000 * (1 + t))
    symmetry_relation_one = sp.factor(Gamma + (Gamma - sigma) / (2 + 3 * Delta * Gamma))
    symmetry_relation_two = sp.factor(
        Gamma / s1_squared
        - (1 + Delta * Gamma)
        + Delta * (1 + Gamma + Delta * sigma) / (2 + 3 * Delta * Gamma)
    )
    aether_symmetrizer_weight = sp.factor(1 / s1_squared)

    a_coef = sp.factor(2 * alpha**2 * r * (1 - r) / (1 - 2 * r) ** 2)
    k_coef = a_coef
    c_coef = sp.factor(alpha**2 * r / (1 - 2 * r))
    a0 = sp.diag(a_coef, k_coef, c_coef, c_coef)
    a_parallel = sp.zeros(4)
    a_y = sp.zeros(4)
    a_z = sp.zeros(4)
    a_parallel[0, 1] = a_parallel[1, 0] = -k_coef
    a_y[0, 2] = a_y[2, 0] = -c_coef
    a_z[0, 3] = a_z[3, 0] = -c_coef
    first_order_symbol = omega * a0 + k_parallel * a_parallel + ky * a_y + kz * a_z
    expected_first_order_determinant = sp.factor(
        c_coef**2
        * k_coef
        * omega**2
        * (a_coef * omega**2 - k_coef * k_parallel**2 - c_coef * (ky**2 + kz**2))
    )
    scalar_parameterization = q / (2 * (1 + q))

    aether_theorem = adm_aether.einstein_aether_covariant_strong_hyperbolicity_control()
    reduced_checks, _ = reduced_factorization.symbolic_checks()
    reduced_by_id = {row["check_id"]: row for row in reduced_checks}
    cross_ids = (
        "S02_SCALAR_TO_DERIVATIVE_AETHER_ZERO",
        "S03_SCALAR_TO_DERIVATIVE_METRIC_ZERO",
        "S04_AETHER_TO_DERIVATIVE_SCALAR_ZERO",
        "S05_METRIC_TO_DERIVATIVE_SCALAR_ZERO",
    )

    locus_speed2 = sp.factor(speed2.subs(locus))
    locus_speed1 = sp.factor(speed1.subs(locus))
    locus_speed0 = sp.factor(speed0.subs(locus))
    locus_energy1 = sp.factor(energy1.subs(locus))
    locus_energy0 = sp.factor(energy0.subs(locus))
    locus_spin0_kinetic = sp.factor(spin0_kinetic.subs(locus))
    locus_alpha1 = sp.factor(alpha1.subs(locus))
    locus_alpha2 = sp.factor(alpha2.subs(locus))
    physical_tensor_ratio = sp.factor(locus_speed2 / sp.exp(4 * varphi))
    scalar_transverse = sp.factor(c_coef / a_coef)

    aether_not_subluminal = (
        locus_speed2 == 1
        and locus_speed0 == 1
        and _positive_residual((locus_speed1 - 1).subs(Delta, Delta_parameterization)) == 0
    )
    scalar_transverse_subluminal = all(
        _positive_residual(value) == 0
        for value in (
            scalar_transverse.subs(r, scalar_parameterization),
            (1 - scalar_transverse).subs(r, scalar_parameterization),
        )
    )
    common_time = all(
        _positive_residual(value) == 0
        for value in (
            aether_symmetrizer_weight.subs(Delta, Delta_parameterization),
            *tuple(item.subs(r, scalar_parameterization) for item in a0.diagonal()),
        )
    )
    combined_all_mode_cherenkov_safe = aether_not_subluminal and not scalar_transverse_subluminal
    combined_direct_sum = (
        symmetry_relation_one == 0
        and symmetry_relation_two == 0
        and all(reduced_by_id[item]["passed"] for item in cross_ids)
        and common_time
    )

    checks = [
        _check(
            "S01_COMMITTED_AETHER_TETRAD_THEOREM",
            0
            if (
                aether_theorem["passed"]
                and aether_theorem["primary_source"] == "https://arxiv.org/abs/1902.05130"
            )
            else 1,
            "The committed primary-source aether-tetrad theorem and source binding pass.",
        ),
        _check(
            "S02_COMMITTED_REDUCED_FACTORIZATION",
            0 if all(row["passed"] for row in reduced_checks) else 1,
            "The committed static W=0 derivative-order factorization passes.",
        ),
        _check("S03_C13_ZERO", c13.subs(locus), "The exact locus keeps c13 zero."),
        _check(
            "S04_C14_DELTA",
            c14.subs(locus) - Delta,
            "The primary-source parameter Delta equals c14.",
        ),
        _check(
            "S05_C123_DELTA_GAMMA",
            c123.subs(locus) - Delta * Gamma,
            "The scalar aether combination equals Delta times Gamma.",
        ),
        _check(
            "S06_PRIMARY_SOURCE_S1_RELATION",
            s1_squared - source_s1_squared,
            "The exact vector speed satisfies primary-source equation 86.",
        ),
        _check("S07_SPIN2_LUMINAL", locus_speed2 - 1, "The tensor cone is g-null."),
        _check(
            "S08_SPIN1_SPEED",
            locus_speed1 - 1 / (1 - Delta) ** 2,
            "The aether-vector squared speed is the exact superluminal expression.",
        ),
        _check("S09_SPIN0_LUMINAL", locus_speed0 - 1, "The aether-scalar cone is g-null."),
        _check(
            "S10_SIGMA_LOCUS",
            sigma - 3 * (1 - Delta) / (1 - 2 * Delta) ** 2,
            "The constraint-combination coefficient sigma is exact.",
        ),
        _check(
            "S11_SYMMETRY_RELATION_ONE",
            symmetry_relation_one,
            "The first primary-source symmetrizer relation vanishes.",
        ),
        _check(
            "S12_SYMMETRY_RELATION_TWO",
            symmetry_relation_two,
            "The second primary-source symmetrizer relation vanishes.",
        ),
        _check(
            "S13_AETHER_SYMMETRIZER_WEIGHT_POSITIVE",
            _positive_residual(aether_symmetrizer_weight.subs(Delta, Delta_parameterization)),
            "The direction-independent aether symmetrizer weight is positive.",
        ),
        _check(
            "S14_SPIN1_SUPERLUMINAL_PARAMETERIZATION",
            _positive_residual((locus_speed1 - 1).subs(Delta, Delta_parameterization)),
            "The aether-vector mode is strictly faster than the photon cone.",
        ),
        _check(
            "S15_SPIN1_ENERGY_POSITIVE_PARAMETERIZATION",
            _positive_residual(locus_energy1.subs(Delta, Delta_parameterization)),
            "The spin-1 energy coefficient is positive.",
        ),
        _check(
            "S16_SPIN0_ENERGY_POSITIVE_PARAMETERIZATION",
            _positive_residual(locus_energy0.subs(Delta, Delta_parameterization)),
            "The spin-0 energy coefficient is positive.",
        ),
        _check(
            "S17_SPIN0_KINETIC_POSITIVE_PARAMETERIZATION",
            _positive_residual(locus_spin0_kinetic.subs(Delta, Delta_parameterization)),
            "The spin-0 kinetic normalization is positive.",
        ),
        _check(
            "S18_ALPHA1_LOCUS",
            locus_alpha1 + 4 * Delta,
            "The finite locus has alpha1=-4Delta.",
        ),
        _check("S19_ALPHA2_ZERO", locus_alpha2, "The locus has alpha2 exactly zero."),
        _check(
            "S20_ALPHA1_CITED_BOUND_MARGIN_POSITIVE",
            _positive_residual(
                (sp.Rational(1, 10000) + locus_alpha1).subs(Delta, Delta_parameterization)
            ),
            "The primary-source |alpha1| necessary-bound margin is positive.",
        ),
        _check(
            "S21_GAMMA_CITED_BOUND_MARGIN_POSITIVE",
            _positive_residual(
                (sp.Rational(1, 125) - (Gamma - 1)).subs(Delta, Delta_parameterization)
            ),
            "The primary-source |Gamma-1| necessary-bound margin is positive.",
        ),
        _check(
            "S22_AETHER_EXACT_LIMIT_LOSES_KINETIC_MARGIN",
            sp.limit(locus_energy1 + locus_energy0, Delta, 0, dir="+"),
            "The exact preferred-frame-free limit loses the aether kinetic margins.",
        ),
        _check(
            "S23_SCALAR_TIME_SYMMETRIZER_POSITIVE_PARAMETERIZATION",
            sum(
                _positive_residual(item.subs(r, scalar_parameterization)) for item in a0.diagonal()
            ),
            "The quadrature-scalar time symmetrizer is positive definite.",
        ),
        _check(
            "S24_SCALAR_PARALLEL_FLUX_SYMMETRIC",
            a_parallel - a_parallel.T,
            "The parallel quadrature-scalar flux is symmetric.",
        ),
        _check(
            "S25_SCALAR_Y_FLUX_SYMMETRIC",
            a_y - a_y.T,
            "The first transverse quadrature-scalar flux is symmetric.",
        ),
        _check(
            "S26_SCALAR_Z_FLUX_SYMMETRIC",
            a_z - a_z.T,
            "The second transverse quadrature-scalar flux is symmetric.",
        ),
        _check(
            "S27_SCALAR_FIRST_ORDER_DETERMINANT",
            first_order_symbol.det() - expected_first_order_determinant,
            "The quadrature-scalar first-order determinant reproduces its physical cone.",
        ),
        _check(
            "S28_SCALAR_LONGITUDINAL_SPEED",
            k_coef / a_coef - 1,
            "The quadrature scalar is photon-luminal along its frozen gradient.",
        ),
        _check(
            "S29_SCALAR_TRANSVERSE_SPEED_PARAMETERIZATION",
            scalar_transverse.subs(r, scalar_parameterization) - 1 / (q + 2),
            "The transverse quadrature-scalar squared speed is strictly subluminal.",
        ),
        _check(
            "S30_COMMITTED_PRINCIPAL_CROSS_BLOCKS_ZERO",
            0 if all(reduced_by_id[item]["passed"] for item in cross_ids) else 1,
            "All committed scalar-to-aether/metric degree-two cross blocks vanish at W=0.",
        ),
        _check(
            "S31_COMBINED_PHYSICAL_MODE_COUNT",
            aether_theorem["physical_mode_count"] + 1 - 6,
            "Five physical aether modes plus one quadrature scalar give six modes.",
        ),
        _check(
            "S32_TENSOR_PHOTON_CONE_ALIGNMENT",
            physical_tensor_ratio.subs(varphi, 0) - 1,
            "The tensor and universal physical photon cones align on the frozen frame.",
        ),
        _check(
            "S33_COMMON_LOCAL_CAUCHY_TIME",
            0 if common_time else 1,
            "Both positive symmetrizers use the aligned aether time as a common local Cauchy time.",
        ),
        _check(
            "S34_AETHER_MODES_NOT_SUBLUMINAL",
            0 if aether_not_subluminal else 1,
            "No physical aether mode is slower than the photon cone on the exact locus.",
        ),
        _check(
            "S35_SCALAR_TRANSVERSE_MODE_SUBLUMINAL",
            0 if scalar_transverse_subluminal else 1,
            "The quadrature scalar remains strictly subluminal in transverse directions.",
        ),
        _check(
            "S36_COMBINED_ALL_MODE_CHERENKOV_SAFETY_REJECTED",
            0 if not combined_all_mode_cherenkov_safe else 1,
            "The full six-mode Cherenkov-safety claim is correctly rejected.",
        ),
        _check(
            "S37_COMBINED_SYMMETRIZER_DIRECT_SUM",
            0 if combined_direct_sum else 1,
            "The exact W=0 combined principal symmetrizer is a positive direct sum.",
        ),
    ]
    expressions = {
        "coupling_locus": {str(key): str(value) for key, value in locus.items()},
        "Gamma": str(Gamma),
        "s1_squared": str(s1_squared),
        "sigma": str(sigma),
        "symmetry_relation_one": str(symmetry_relation_one),
        "symmetry_relation_two": str(symmetry_relation_two),
        "aether_symmetrizer_weight": str(aether_symmetrizer_weight),
        "spin_2_speed_squared": str(locus_speed2),
        "spin_1_speed_squared": str(locus_speed1),
        "spin_0_speed_squared": str(locus_speed0),
        "spin_1_energy": str(locus_energy1),
        "spin_0_energy": str(locus_energy0),
        "spin_0_kinetic": str(locus_spin0_kinetic),
        "alpha1": str(locus_alpha1),
        "alpha2": str(locus_alpha2),
        "scalar_A0": str(a0),
        "scalar_A_parallel": str(a_parallel),
        "scalar_A_y": str(a_y),
        "scalar_A_z": str(a_z),
        "scalar_characteristic_determinant": str(sp.factor(first_order_symbol.det())),
        "scalar_longitudinal_speed_squared": str(sp.factor(k_coef / a_coef)),
        "scalar_transverse_speed_squared": str(scalar_transverse),
        "combined_all_mode_cherenkov_safe": combined_all_mode_cherenkov_safe,
        "aether_formulation": aether_theorem["formulation"],
        "aether_primary_source": aether_theorem["primary_source"],
        "aether_primary_source_scope": "equations 78 and 84-86",
    }
    return checks, expressions


def numeric_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    Delta, r = sp.symbols("Delta r", positive=True, finite=True)
    Gamma = sp.factor(1 / (1 - 2 * Delta))
    speed1 = sp.factor(1 / (1 - Delta) ** 2)
    speed0 = sp.Integer(1)
    energy1 = sp.factor(2 * Delta / (1 - Delta) ** 2)
    energy0 = sp.factor(Delta * (2 - Delta))
    spin0_kinetic = energy0
    aether_symmetrizer_weight = sp.factor((1 - Delta) ** 2)
    alpha1 = -4 * Delta
    alpha2 = sp.Integer(0)
    scalar_transverse = sp.factor((1 - 2 * r) / (2 * (1 - r)))
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(config["machine_check_contract"]["numeric_cases"]):
        Delta_value = sp.Rational(str(raw["Delta"]))
        r_value = sp.Rational(str(raw["r"]))
        values = {
            "Gamma": sp.factor(Gamma.subs(Delta, Delta_value)),
            "spin_2_speed_squared": sp.Integer(1),
            "spin_1_speed_squared": sp.factor(speed1.subs(Delta, Delta_value)),
            "spin_0_speed_squared": speed0,
            "scalar_longitudinal_speed_squared": sp.Integer(1),
            "scalar_transverse_speed_squared": sp.factor(scalar_transverse.subs(r, r_value)),
            "spin_1_energy": sp.factor(energy1.subs(Delta, Delta_value)),
            "spin_0_energy": sp.factor(energy0.subs(Delta, Delta_value)),
            "spin_0_kinetic": sp.factor(spin0_kinetic.subs(Delta, Delta_value)),
            "aether_symmetrizer_weight": sp.factor(
                aether_symmetrizer_weight.subs(Delta, Delta_value)
            ),
            "alpha1": sp.factor(alpha1.subs(Delta, Delta_value)),
            "alpha2": alpha2,
        }
        aether_not_subluminal = all(
            values[key] >= 1
            for key in (
                "spin_2_speed_squared",
                "spin_1_speed_squared",
                "spin_0_speed_squared",
            )
        )
        scalar_subluminal = 0 < values["scalar_transverse_speed_squared"] < 1
        common_time = (
            values["aether_symmetrizer_weight"] > 0
            and values["spin_1_energy"] > 0
            and values["spin_0_energy"] > 0
            and values["spin_0_kinetic"] > 0
        )
        theorem_entry = aether_not_subluminal and common_time
        combined_all_mode_cherenkov_safe = aether_not_subluminal and not scalar_subluminal
        passed = (
            0 < Delta_value < sp.Rational(1, 40000)
            and 0 < r_value < sp.Rational(1, 2)
            and values["alpha1"] < 0
            and values["alpha2"] == 0
            and -values["alpha1"] < sp.Rational(1, 10000)
            and values["Gamma"] - 1 < sp.Rational(1, 125)
            and theorem_entry
            and scalar_subluminal
            and not combined_all_mode_cherenkov_safe
        )
        rows.append(
            {
                "case_id": f"N{index + 1:02d}",
                "Delta": str(Delta_value),
                "r": str(r_value),
                **{key: str(value) for key, value in values.items()},
                "aether_primary_symmetric_theorem_entry": bool(theorem_entry),
                "common_local_Cauchy_time": bool(common_time),
                "aether_modes_not_subluminal": bool(aether_not_subluminal),
                "scalar_transverse_mode_subluminal": bool(scalar_subluminal),
                "all_mode_cherenkov_safety": bool(combined_all_mode_cherenkov_safe),
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
        raise QuadratureCombinedHyperbolicityError("symbolic inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureCombinedHyperbolicityError("symbolic derivation failed")
    numeric = numeric_cases(config)
    if not all(row["passed"] for row in numeric):
        raise QuadratureCombinedHyperbolicityError("numeric derivation failed")
    source_path = Path(__file__).resolve()
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureCombinedHyperbolicityError("test file hash changed")
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
        "frozen_branch_contract": config["frozen_branch_contract"],
        "symmetric_hyperbolic_locus_contract": config["symmetric_hyperbolic_locus_contract"],
        "scalar_first_order_contract": config["scalar_first_order_contract"],
        "combined_principal_contract": config["combined_principal_contract"],
        "ppn_and_physical_gate_boundary": config["ppn_and_physical_gate_boundary"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "numeric_cases": numeric,
            "expressions": expressions,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(bool(row["passed"]) for row in numeric),
            "aether_physical_modes": 5,
            "combined_physical_modes": 6,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls_by_builder": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "limitations": [
            config["obstruction_contract"]["aether_ppn"],
            config["obstruction_contract"]["scalar_cherenkov"],
            config["obstruction_contract"]["low_gradient"],
            config["obstruction_contract"]["endpoint"],
            config["obstruction_contract"]["background"],
            config["obstruction_contract"]["matter"],
            config["obstruction_contract"]["health"],
            config["obstruction_contract"]["observations"],
        ],
        "zero_access_and_compute": config["zero_access_and_compute"],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise QuadratureCombinedHyperbolicityError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != _sha(body):
        raise QuadratureCombinedHyperbolicityError("receipt content hash changed")
    expected = build_receipt(root)
    if receipt != expected:
        raise QuadratureCombinedHyperbolicityError("stored receipt differs from exact rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise QuadratureCombinedHyperbolicityError("refusing to replace existing receipt")
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
            raise QuadratureCombinedHyperbolicityError(
                "concurrent receipt creator published different bytes"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path) -> Path:
    receipt = build_receipt(root)
    payload = _canonical_bytes(receipt)
    output = root.resolve() / OUTPUT_PATH
    _atomic_no_clobber(output, payload)
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
        "restricted_combined_symmetric_hyperbolicity": receipt["adjudication"][
            "restricted_combined_local_symmetric_hyperbolicity"
        ],
        "full_covariant_health": receipt["claim_boundary"]["full_covariant_health_established"],
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
