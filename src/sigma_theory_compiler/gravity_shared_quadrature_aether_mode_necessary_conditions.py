"""Restricted Einstein-aether mode and PPN conditions for the quadrature metric."""

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

from sigma_theory_compiler.adm_aether import (
    einstein_aether_linearized_energy_control,
    einstein_aether_reduced_principal_domain_control,
)

CONFIG_PATH = Path("configs/gravity_shared_quadrature_aether_mode_necessary_conditions_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_aether_mode_necessary_conditions.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-aether-mode-necessary-conditions-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-aether-mode-necessary-conditions-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-aether-mode-necessary-conditions-receipt-1.0"
STATUS = "restricted_aether_modes_and_ppn_necessary_conditions_derived_no_data"
DECISION = (
    "RESTRICTED_AETHER_MODE_AND_PPN_CONDITIONS_DERIVED_EXACT_GW_AND_PPN_ZERO_"
    "INTERSECTION_IS_SINGULAR_FINITE_TOLERANCE_ALL_LUMINAL_LOCUS_EXISTS_FULL_"
    "VECTOR_SCALAR_THEORY_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "0090fd57c93bcae073c14c1c5406fe07c38fa555cde440ef2b9b16e4933491f1"
EXPECTED_TEST_FILE_SHA256 = "5ee61198dcd23ca4588f03f34d9c319f64b55f23dda839ad49b2a09fef7526ad"
EXPECTED_SECTION_HASHES = {
    "predecessor_bindings": "0ed99917163bad3575086364e8c450bd0dee9af628d9091f9bfde3bf0e3c63a5",
    "aether_mode_contract": "fffe50a2546fdc7f5b6ab65e68d7f6dcb59358d59c006d7661db10b7429d7fe8",
    "ppn_contract": "3a78bc29ce38c3b11d5314e8a599ea4cc8371520c5c5a6f5ea9a2b056ee503fe",
    "specialization_contract": "18b3413b8ff32d0c83970fe21f62f7e0cb79d61de5d14824525d8eb9bb3f1f52",
    "physical_cone_contract": "c38777f9fa7f6d9f65915d6405fe20ad826dd0f835902e4c8401971c5113dd44",
    "obstruction_contract": "9cf7f26421e05b45ba35791b2cc6e9e4102cc427d85b33ab51cbdeb8124018d0",
    "machine_check_contract": "40b37db5441cc194981e4b0d3ff5577197b8a7108694cb78d6d644283a3fb42d",
    "adjudication": "8ec8fa1d19cf8472242ae479eca168dfa9029a418336362ff8275b96d75c32fc",
    "claim_boundary": "c61bb6fd3f19c357e392350633e2c5fd557bfc1c4b44f3c86799f002a1fad89a",
    "zero_access_and_compute": "d49dd6f61c1704a662f2a82623b63bcb487c43018af38bad97d092c982b74e94",
}


class QuadratureAetherModeError(RuntimeError):
    """Raised when a frozen aether-mode contract or derivation changes."""


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
        raise QuadratureAetherModeError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureAetherModeError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureAetherModeError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "aether_mode_contract",
            "ppn_contract",
            "specialization_contract",
            "physical_cone_contract",
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
        or config["analysis_id"] != "gravity-shared-quadrature-aether-mode-necessary-conditions-v1"
        or config["status"] != "frozen_no_data_restricted_aether_mode_and_ppn_necessary_conditions"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureAetherModeError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureAetherModeError(f"config {key} changed")
    machine = config["machine_check_contract"]
    if len(machine["required_symbolic_checks"]) != 25 or len(machine["epsilon_cases"]) != 3:
        raise QuadratureAetherModeError("machine-check inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureAetherModeError("adjudication changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureAetherModeError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureAetherModeError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureAetherModeError("predecessor Git binding failed") from error


def validate_predecessors(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        commit = str(binding["git_commit"])
        if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
            raise QuadratureAetherModeError("predecessor commit changed")
        for artifact in binding["artifacts"]:
            relative = str(artifact["path"])
            current = root / relative
            if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureAetherModeError("predecessor artifact changed")
            current_blob = _git("hash-object", "--path", relative, relative, root=root).strip()
            commit_blob = _git("rev-parse", f"{commit}:{relative}", root=root).strip()
            if current_blob != commit_blob:
                raise QuadratureAetherModeError("predecessor commit bytes changed")
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
                raise QuadratureAetherModeError("predecessor receipt changed")
            row["receipt_content_sha256"] = receipt["content_sha256"]
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


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    principal_control = einstein_aether_reduced_principal_domain_control()
    energy_control = einstein_aether_linearized_energy_control()

    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    epsilon, tau, q = sp.symbols("epsilon tau q", positive=True, finite=True)
    varphi = sp.symbols("varphi", real=True, finite=True)
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
    scalar_kinetic = sp.factor(c14**2 * (1 - c13) * trace_factor / c123)

    alpha1 = sp.factor(-8 * (c3**2 + c1 * c4) / (2 * c1 - c1**2 + c3**2))
    alpha2 = sp.factor(
        alpha1 / 2 - (c1 + 2 * c3 - c4) * (2 * c1 + 3 * c2 + c3 + c4) / (c123 * (2 - c14))
    )
    ppn_c4 = -(c3**2) / c1
    ppn_c2 = (-2 * c1**2 - c1 * c3 + c3**2) / (3 * c1)
    locus = {
        c1: epsilon,
        c2: epsilon / (1 - 2 * epsilon),
        c3: -epsilon,
        c4: 0,
    }
    exact_zero = {c3: -c1, c4: -c1, c2: 0}
    positive_parameterization = epsilon - q / (2 * (1 + q))
    photon_speed = sp.exp(2 * varphi)
    mode_to_photon_ratio = sp.simplify(1 / photon_speed)

    checks = [
        _check(
            "S01_COMMITTED_REDUCED_PRINCIPAL_CONTROL",
            0 if principal_control["passed"] else 1,
            "The committed five-mode reduced-principal control passes.",
        ),
        _check(
            "S02_COMMITTED_LINEAR_ENERGY_CONTROL",
            0 if energy_control["passed"] else 1,
            "The committed physical-mode energy control passes.",
        ),
        _check("S03_C13_ZERO_C3", c13.subs(locus), "The locus has c13=0."),
        _check("S04_SPIN2_LUMINAL", speed2.subs(locus) - 1, "The spin-2 mode is g-luminal."),
        _check(
            "S05_VECTOR_NUMERATOR_LOCUS",
            vector_numerator.subs(locus) - 2 * epsilon,
            "The spin-1 gradient numerator is 2 epsilon.",
        ),
        _check(
            "S06_SPIN1_LUMINAL_LOCUS", speed1.subs(locus) - 1, "Both spin-1 modes are g-luminal."
        ),
        _check(
            "S07_C123_LOCUS",
            c123.subs(locus) - epsilon / (1 - 2 * epsilon),
            "The locus has positive c123 on 0<epsilon<1/2.",
        ),
        _check(
            "S08_SCALAR_TRACE_LOCUS",
            trace_factor.subs(locus) - (2 - epsilon) / (1 - 2 * epsilon),
            "The scalar trace factor has the frozen positive form.",
        ),
        _check("S09_SPIN0_LUMINAL_LOCUS", speed0.subs(locus) - 1, "The spin-0 mode is g-luminal."),
        _check(
            "S10_SPIN1_ENERGY_LOCUS",
            energy1.subs(locus) - 2 * epsilon,
            "The spin-1 energy coefficient is positive for epsilon>0.",
        ),
        _check(
            "S11_SPIN0_ENERGY_LOCUS",
            energy0.subs(locus) - epsilon * (2 - epsilon),
            "The spin-0 energy coefficient is positive on the locus.",
        ),
        _check(
            "S12_PPN_ALPHA1_C13_ZERO",
            alpha1.subs(c3, -c1) + 4 * (c1 + c4),
            "With c13=0, alpha1=-4c14.",
        ),
        _check(
            "S13_PPN_ALPHA1_LUMINAL_LOCUS",
            alpha1.subs(locus) + 4 * epsilon,
            "The finite luminal locus has alpha1=-4epsilon.",
        ),
        _check(
            "S14_PPN_ALPHA2_LUMINAL_LOCUS",
            alpha2.subs(locus),
            "The finite luminal locus has alpha2=0.",
        ),
        _check(
            "S15_EXACT_PPN_C4_UNDER_C13_ZERO",
            ppn_c4.subs(c3, -c1) + c1,
            "Exact alpha1 cancellation under c13=0 gives c4=-c1.",
        ),
        _check(
            "S16_EXACT_PPN_C2_UNDER_C13_ZERO",
            ppn_c2.subs(c3, -c1),
            "The exact PPN-zero family gives c2=0 under c13=0.",
        ),
        _check(
            "S17_EXACT_PPN_C14_SINGULAR",
            c14.subs(exact_zero),
            "Exact GW+PPN cancellation has c14=0.",
        ),
        _check(
            "S18_EXACT_PPN_C123_SINGULAR",
            c123.subs(exact_zero),
            "Exact GW+PPN cancellation has c123=0.",
        ),
        _check(
            "S19_VECTOR_KINETIC_LIMIT_ZERO",
            sp.limit(2 * c14.subs(locus), epsilon, 0, dir="+"),
            "The spin-1 kinetic normalization vanishes in the exact-PPN limit.",
        ),
        _check(
            "S20_SCALAR_KINETIC_LIMIT_ZERO",
            sp.limit(scalar_kinetic.subs(locus), epsilon, 0, dir="+"),
            "The spin-0 kinetic normalization vanishes in the exact-PPN limit.",
        ),
        _check(
            "S21_ALPHA1_TOLERANCE_BOUNDARY",
            (-alpha1.subs(locus)).subs(epsilon, tau / 4) - tau,
            "epsilon=tau/4 saturates an abstract |alpha1| tolerance tau.",
        ),
        _check(
            "S22_PHYSICAL_CONE_RATIO",
            mode_to_photon_ratio - sp.exp(-2 * varphi),
            "Every g-luminal aether mode has speed ratio exp(-2varphi) to photons.",
        ),
        _check(
            "S23_PHYSICAL_CONE_ALIGNMENT",
            mode_to_photon_ratio.subs(varphi, 0) - 1,
            "varphi_infinity=0 aligns the pure-aether and physical photon cones.",
        ),
        _check(
            "S24_LUMINAL_LOCUS_POSITIVE_DOMAIN",
            (1 - 2 * epsilon).subs(epsilon, q / (2 * (1 + q))) - 1 / (1 + q),
            "epsilon=q/[2(1+q)] parameterizes the full 0<epsilon<1/2 domain.",
        ),
        _check(
            "S25_NO_UNIFORM_POSITIVE_KINETIC_MARGIN",
            sp.limit((2 * epsilon) + epsilon * (2 - epsilon), epsilon, 0, dir="+"),
            "Both positive vector residues lose their margin as epsilon approaches zero.",
        ),
    ]
    expressions = {
        "spin_2_speed_squared": str(speed2),
        "spin_1_speed_squared": str(speed1),
        "spin_0_speed_squared": str(speed0),
        "spin_1_energy": str(energy1),
        "spin_0_energy": str(energy0),
        "alpha1": str(alpha1),
        "alpha2": str(alpha2),
        "luminal_locus": "c1=epsilon,c2=epsilon/(1-2epsilon),c3=-epsilon,c4=0",
        "mode_to_photon_speed_ratio": str(mode_to_photon_ratio),
        "positive_parameterization_residual": str(positive_parameterization),
    }
    inherited = {
        "principal_mode_count": principal_control["mode_count"],
        "principal_regular_domain": principal_control["necessary_and_sufficient_regular_domain"],
        "principal_primary_source": principal_control["primary_source"],
        "energy_primary_sources": energy_control["primary_sources"],
        "principal_negative_controls_pass": all(
            row["rejected"] for row in principal_control["negative_controls"].values()
        ),
        "energy_speed_only_negative_controls_pass": all(
            row["positive_speed_negative_energy"]
            for row in energy_control["speed_only_negative_controls"].values()
        ),
    }
    return checks, expressions, inherited


def epsilon_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in config["machine_check_contract"]["epsilon_cases"]:
        epsilon = sp.Rational(text)
        c1 = epsilon
        c2 = epsilon / (1 - 2 * epsilon)
        c3 = -epsilon
        c4 = sp.Integer(0)
        c13 = c1 + c3
        c14 = c1 + c4
        c123 = c1 + c2 + c3
        vector_numerator = 2 * c1 - c1**2 + c3**2
        trace_factor = 2 + c13 + 3 * c2
        speeds = (
            1 / (1 - c13),
            vector_numerator / (2 * c14 * (1 - c13)),
            c123 * (2 - c14) / (c14 * (1 - c13) * trace_factor),
        )
        energy1 = vector_numerator / (1 - c13)
        energy0 = c14 * (2 - c14)
        alpha1 = -4 * epsilon
        alpha2 = sp.Integer(0)
        rows.append(
            {
                "epsilon": str(epsilon),
                "couplings": {
                    "c1": str(c1),
                    "c2": str(c2),
                    "c3": str(c3),
                    "c4": str(c4),
                },
                "speed_squared": [str(sp.factor(value)) for value in speeds],
                "spin_1_energy": str(sp.factor(energy1)),
                "spin_0_energy": str(sp.factor(energy0)),
                "alpha1": str(alpha1),
                "alpha2": str(alpha2),
                "passed": bool(
                    0 < epsilon < sp.Rational(1, 2)
                    and all(value == 1 for value in speeds)
                    and energy1 > 0
                    and energy0 > 0
                    and alpha1 < 0
                    and alpha2 == 0
                ),
            }
        )
    return rows


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    predecessors = validate_predecessors(root, config["predecessor_bindings"])
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureAetherModeError("test file hash changed")
    symbolic, expressions, inherited = symbolic_checks()
    numeric = epsilon_cases(config)
    if tuple(row["check_id"] for row in symbolic) != tuple(
        config["machine_check_contract"]["required_symbolic_checks"]
    ):
        raise QuadratureAetherModeError("symbolic inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureAetherModeError("symbolic derivation failed")
    if not all(row["passed"] for row in numeric):
        raise QuadratureAetherModeError("epsilon case failed")
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
        "predecessor_validation": predecessors,
        "aether_mode_contract": config["aether_mode_contract"],
        "ppn_contract": config["ppn_contract"],
        "specialization_contract": config["specialization_contract"],
        "physical_cone_contract": config["physical_cone_contract"],
        "obstruction_contract": config["obstruction_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "derived_expressions": expressions,
            "inherited_committed_controls": inherited,
            "epsilon_cases": numeric,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "inherited_pure_aether_modes": inherited["principal_mode_count"],
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "epsilon_cases": len(numeric),
            "epsilon_cases_passed": sum(bool(row["passed"]) for row in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls_by_builder": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The inherited Einstein-aether reduced symbol is exact around aligned Minkowski aether, not an on-shell nonzero-gradient quadrature background.",
            "The all-luminal finite locus is a necessary pure-aether parameter specialization, not a full scalar-vector-metric characteristic or nonlinear health proof.",
            "Exact tensor and preferred-frame cancellation reaches a singular kinetic surface; finite tolerance approaches that surface and may lower the strong-coupling scale.",
            "No Solar-System, GW, binary-pulsar, cosmological, lensing, confirmation, or independent observation was opened or passed.",
            "No healthy gravity theory, quantitative lensing prediction, historical novelty, or publication-readiness change is claimed.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body):
        raise QuadratureAetherModeError("receipt hash changed")
    if dict(receipt) != build_receipt(root):
        raise QuadratureAetherModeError("receipt evidence changed")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureAetherModeError("refusing to overwrite receipt")
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
            raise QuadratureAetherModeError("receipt publication race") from error
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
            "finite_luminal_locus_exists": receipt["adjudication"][
                "finite_positive_all_g_luminal_locus_exists"
            ],
            "exact_ppn_zero_regular": receipt["adjudication"][
                "exact_c13_alpha1_alpha2_zero_is_regular"
            ],
            "full_covariant_health_established": receipt["claim_boundary"][
                "full_covariant_health_established"
            ],
            "content_sha256": receipt["content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
