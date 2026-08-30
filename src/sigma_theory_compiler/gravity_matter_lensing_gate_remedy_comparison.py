"""No-data structural comparison of five remedies to kinetic-gate mixing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_gate_remedy_comparison_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_gate_remedy_comparison.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_gate_remedy_comparison.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-gate-remedy-comparison-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-gate-remedy-comparison-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-gate-remedy-comparison-receipt-1.0"
DECISION = (
    "STRUCTURAL_REMEDY_COMPARISON_COMPLETE_B_CONSTANT_Y_LEADS_CONDITIONALLY_"
    "NO_ARCHITECTURE_HEALTH_OR_NOVELTY_ESTABLISHED"
)
EXPECTED_CONFIG_FILE_SHA256 = "e95c9d3e6742dc3c30fe048a618aed3d7315e9d4c6e2285b2b78e37f8072015e"
EXPECTED_CONFIG_CONTENT_SHA256 = "e855f30ca5df4cf7244216819880c9f9b6b7e0896a6a1ec768ef494314fa0c93"

ARCHITECTURE_IDS = (
    "A_SOURCE_AMPLITUDE_GATE",
    "B_SPLIT_KINETIC_MASS_GATES",
    "C_POSITIVE_FIELD_SPACE_METRIC",
    "D_AUXILIARY_CHANNEL",
    "E_DYNAMIC_XCHI_SUPPRESSION",
)
ORIGIN_IDS = (
    "SOURCE_ONLY_GATE",
    "SPLIT_KINETIC_AND_MASS_GATES",
    "ORTHOGONAL_FIELD_SPACE_METRIC",
    "AUXILIARY_NONPROPAGATING_CHANNEL",
    "DYNAMIC_XCHI_ENVELOPE",
)
RANKING = (
    "B_SPLIT_KINETIC_MASS_GATES",
    "A_SOURCE_AMPLITUDE_GATE",
    "C_POSITIVE_FIELD_SPACE_METRIC",
    "D_AUXILIARY_CHANNEL",
    "E_DYNAMIC_XCHI_SUPPRESSION",
)
SYMBOLIC_CHECK_IDS = (
    "S01_A_CANONICAL_CHI_SYMBOL",
    "S02_A_MASS_IS_LOWER_ORDER",
    "S03_B_GENERAL_PRINCIPAL_DETERMINANT",
    "S04_B_CONSTANT_Y_CROSS_VANISHES",
    "S05_B_LOCAL_RANGE_DISPERSION",
    "S06_C_FIELD_SPACE_DETERMINANT",
    "S07_C_SCHUR_BOUNDARY",
    "S08_D_AUXILIARY_EOM",
    "S09_D_AUXILIARY_ELIMINATION",
    "S10_D_ZERO_AUXILIARY_PRINCIPAL_RANK",
    "S11_E_MULTIPLIER_VELOCITY_HESSIAN_SINGULAR",
    "S12_UNIVERSAL_CONFORMAL_NULL_CONE",
)


class RemedyComparisonError(RuntimeError):
    """Raised when a binding, derivation, or publication guard fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RemedyComparisonError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemedyComparisonError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    if isinstance(residual, sp.MatrixBase):
        passed = all(sp.simplify(item) == 0 for item in residual)
        rendered: Any = "ZERO_MATRIX" if passed else sp.sstr(residual.applyfunc(sp.simplify))
    else:
        simplified = sp.simplify(residual)
        passed = simplified == 0
        rendered = "0" if passed else sp.sstr(simplified)
    return {
        "check_id": check_id,
        "passed": passed,
        "residual": rendered,
        "statement": statement,
    }


def run_symbolic_suite() -> dict[str, Any]:
    metric = sp.diag(-1, 1)
    q0, q1, chi, mass = sp.symbols("q0 q1 chi m", real=True)
    gradient = sp.Matrix([q0, q1])
    x_chi = -sp.Rational(1, 2) * (gradient.T * metric * gradient)[0]
    canonical = x_chi - mass**2 * chi**2 / 2
    canonical_symbol = -sp.hessian(canonical, (q0, q1))

    c, c_x, y, y_x, k2, vk, wk = sp.symbols("C C_X Y Y_X k2 vk wk", real=True)
    symbol_b = sp.Matrix([[c * k2 - c_x * vk**2, -y_x * vk * wk], [-y_x * vk * wk, y * k2]])
    determinant_b = c * y * k2**2 - c_x * y * k2 * vk**2 - y_x**2 * vk**2 * wk**2
    omega, wave, z = sp.symbols("omega k Z", positive=True)
    dispersion = y * (-(omega**2) + wave**2) + z * mass**2
    omega_squared = wave**2 + z * mass**2 / y

    p_y, p_yy, k11, k12, k22 = sp.symbols("P_y P_yy K_11 K_12 K_22", real=True)
    symbol_c = sp.Matrix([[(p_y + k11) * k2 - p_yy * vk**2, k12 * k2], [k12 * k2, k22 * k2]])
    determinant_c = k22 * k2 * ((p_y + k11) * k2 - p_yy * vk**2) - k12**2 * k2**2
    schur_spatial = (p_y + k11) * k22 - k12**2

    aux, aux_mass, current = sp.symbols("a M_a J", nonzero=True, real=True)
    auxiliary_lagrangian = -(aux_mass**2) * aux**2 / 2 + aux * current
    auxiliary_solution = current / aux_mass**2
    effective_auxiliary = sp.simplify(auxiliary_lagrangian.subs(aux, auxiliary_solution))
    phi_symbol = sp.symbols("P_phi", nonzero=True)
    auxiliary_principal = sp.diag(phi_symbol, 0)

    phi_dot, chi_dot, lambda_dot, multiplier = sp.symbols(
        "phi_dot chi_dot lambda_dot lambda", real=True
    )
    frozen_f, frozen_z = sp.symbols("E Z_gate", real=True)
    constrained_kinetic = (frozen_z + multiplier) * chi_dot**2 / 2 - multiplier * frozen_f
    constrained_hessian = sp.hessian(constrained_kinetic, (phi_dot, chi_dot, lambda_dot))

    conformal = sp.symbols("A", positive=True)
    covector = sp.Matrix(sp.symbols("n0:4", real=True))
    inverse_metric = sp.diag(-1, 1, 1, 1)
    base_norm = (covector.T * inverse_metric * covector)[0]
    physical_inverse = inverse_metric / conformal**2
    physical_norm = (covector.T * physical_inverse * covector)[0]

    checks = [
        _check(
            "S01_A_CANONICAL_CHI_SYMBOL",
            canonical_symbol - metric,
            "The canonical finite-range chi gradient Hessian gives the metric principal tensor.",
        ),
        _check(
            "S02_A_MASS_IS_LOWER_ORDER",
            sp.diff(canonical_symbol, mass),
            "The canonical mass does not enter the principal gradient Hessian.",
        ),
        _check(
            "S03_B_GENERAL_PRINCIPAL_DETERMINANT",
            sp.det(symbol_b) - determinant_b,
            "The split-gate principal determinant contains derivative mixing only through Y_X.",
        ),
        _check(
            "S04_B_CONSTANT_Y_CROSS_VANISHES",
            symbol_b.subs(y_x, 0)[0, 1],
            "Freezing Y constant removes the derivative cross-principal block.",
        ),
        _check(
            "S05_B_LOCAL_RANGE_DISPERSION",
            dispersion.subs(omega**2, omega_squared),
            "A constant local background has m_eff^2=m_chi^2*Z/Y.",
        ),
        _check(
            "S06_C_FIELD_SPACE_DETERMINANT",
            sp.det(symbol_c) - determinant_c,
            "The field-only two-field metric has the exact frozen Schur determinant.",
        ),
        _check(
            "S07_C_SCHUR_BOUNDARY",
            schur_spatial.subs(k12**2, (p_y + k11) * k22),
            "The frozen cross-term equality is the spatial Schur-complement boundary.",
        ),
        _check(
            "S08_D_AUXILIARY_EOM",
            sp.diff(auxiliary_lagrangian, aux).subs(aux, auxiliary_solution),
            "The algebraic auxiliary equation gives a=J/M_a^2.",
        ),
        _check(
            "S09_D_AUXILIARY_ELIMINATION",
            effective_auxiliary - current**2 / (2 * aux_mass**2),
            "Exact auxiliary elimination adds J^2/(2*M_a^2).",
        ),
        _check(
            "S10_D_ZERO_AUXILIARY_PRINCIPAL_RANK",
            auxiliary_principal.rank() - 1,
            "The two-variable block has rank one: the auxiliary channel has no independent second-order principal eigenvalue.",
        ),
        _check(
            "S11_E_MULTIPLIER_VELOCITY_HESSIAN_SINGULAR",
            sp.det(constrained_hessian),
            "The lambda velocity row is zero, so the unconstrained velocity Hessian is singular.",
        ),
        _check(
            "S12_UNIVERSAL_CONFORMAL_NULL_CONE",
            physical_norm - base_norm / conformal**2,
            "One conformal physical metric preserves the base null condition for all coupled photons.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "checks": checks,
        "all_passed": True,
        "derived_expressions": {
            "B_principal_determinant": "C*Y*k2^2-C_X*Y*k2*(v.k)^2-Y_X^2*(v.k)^2*(w.k)^2",
            "B_local_mass_squared": "m_chi^2*Z/Y",
            "C_spatial_schur": "(P_y+K_11)*K_22-K_12^2",
            "D_effective_term": "J^2/(2*M_a^2)",
            "E_velocity_hessian_determinant": "0",
            "universal_conformal_null_norm": "tilde_g_inverse(n,n)=g_inverse(n,n)/A^2",
        },
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_receipt_bindings",
            "shared_requirements",
            "architectures",
            "ranking_contract",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-gate-remedy-comparison-v1",
        "analysis identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    bindings = config["predecessor_receipt_bindings"]
    _require(
        tuple(item["binding_id"] for item in bindings)
        == (
            "theory_preflight",
            "external_metric_principal_symbol",
            "conditional_kinetic_gate_no_go",
        ),
        "predecessor inventory changed",
    )
    binding_keys = {
        "binding_id",
        "git_commit",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "receipt_schema_version",
        "receipt_decision",
    }
    for binding in bindings:
        _strict(binding, binding_keys, f"predecessor {binding.get('binding_id')}")
    requirements = config["shared_requirements"]
    _require(
        "same single physical metric" in requirements["universal_metric"],
        "universal metric weakened",
    )
    _require(
        "higher derivatives" in requirements["derivative_coupling_warning"],
        "derivative-coupling warning removed",
    )
    _require("second multiplier" in requirements["forbidden_shortcut"], "photon shortcut enabled")
    architectures = config["architectures"]
    _require(
        tuple(item["architecture_id"] for item in architectures) == ARCHITECTURE_IDS,
        "architecture inventory changed",
    )
    _require(
        tuple(item["origin_remedy_id"] for item in architectures) == ORIGIN_IDS,
        "origin remedy mapping changed",
    )
    architecture_keys = {
        "architecture_id",
        "origin_remedy_id",
        "minimal_action_term",
        "degrees_of_freedom",
        "external_metric_principal_symbol",
        "conservation_and_universal_metric",
        "higher_derivative_risk",
        "range_and_amplitude",
        "repair_target",
        "universal_metric_preserved",
        "independent_photon_multiplier",
        "structural_rank",
        "structural_disposition",
        "healthy_claim",
        "novelty_claim",
    }
    for index, architecture in enumerate(architectures, start=1):
        _strict(architecture, architecture_keys, architecture["architecture_id"])
        _require(architecture["universal_metric_preserved"] is True, "universal metric lost")
        _require(
            architecture["independent_photon_multiplier"] is False, "photon multiplier enabled"
        )
        _require(
            architecture["healthy_claim"] is False and architecture["novelty_claim"] is False,
            "architecture overclaimed",
        )
        _require(
            architecture["structural_rank"] == RANKING.index(architecture["architecture_id"]) + 1,
            "rank inconsistent",
        )
        _require(
            all(
                architecture[key]
                for key in architecture_keys
                - {"healthy_claim", "novelty_claim", "independent_photon_multiplier"}
            ),
            f"architecture field empty: {index}",
        )
    _require(tuple(config["ranking_contract"]["order"]) == RANKING, "ranking changed")
    _require(
        config["ranking_contract"]["leading_branch"].startswith("B with Y=constant>0"),
        "leading branch changed",
    )
    _require(
        tuple(config["machine_check_contract"]["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic contract changed",
    )
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["universal_metric_required"] is True
        and adjudication["independent_photon_multiplier_allowed"] is False,
        "universal-metric gate changed",
    )
    _require(
        adjudication["healthy_architecture_identified"] is False
        and adjudication["novel_architecture_identified"] is False,
        "health or novelty overclaimed",
    )
    claims = config["claim_boundary"]
    _require(
        claims["structural_comparison_completed"] is True
        and claims["bounded_algebra_machine_verified"] is True,
        "bounded result disabled",
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key not in {"structural_comparison_completed", "bounded_algebra_machine_verified"}
        ),
        "claim boundary overstated",
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "access state changed",
    )


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    _require(path.is_file(), "config missing")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_predecessors(root: Path, config: Mapping[str, Any]) -> None:
    for binding in config["predecessor_receipt_bindings"]:
        path = root / binding["receipt_path"]
        _require(path.is_file(), f"predecessor missing: {binding['binding_id']}")
        _require(
            _file_sha(path) == binding["receipt_file_sha256"],
            f"predecessor changed: {binding['binding_id']}",
        )
        receipt = _read_json(path)
        _require(
            receipt.get("schema_version") == binding["receipt_schema_version"],
            f"predecessor schema changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            f"predecessor content changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("decision") == binding["receipt_decision"],
            f"predecessor decision changed: {binding['binding_id']}",
        )


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _validate_predecessors(root, config)
    _require((root / SOURCE_PATH).is_file(), "implementation missing")
    _require((root / TEST_PATH).is_file(), "test missing")
    symbolic = run_symbolic_suite()
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "five_remedies_structurally_compared_no_health_or_novelty_claim",
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
        "predecessor_receipt_bindings": config["predecessor_receipt_bindings"],
        "shared_requirements": config["shared_requirements"],
        "architectures": config["architectures"],
        "ranking_contract": config["ranking_contract"],
        "symbolic_suite": symbolic,
        "adjudication": config["adjudication"],
        "counts": {
            "architectures_compared": len(config["architectures"]),
            "universal_metric_architectures": sum(
                item["universal_metric_preserved"] for item in config["architectures"]
            ),
            "independent_photon_multipliers": sum(
                item["independent_photon_multiplier"] for item in config["architectures"]
            ),
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "healthy_architectures_established": 0,
            "novel_architectures_established": 0,
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The ranking is a target-free structural triage, not evidence that B or any other architecture is healthy, novel, or phenomenologically successful.",
            "All principal symbols freeze the gravitational metric and matter fields; full metric constraints, universal-metric matter characteristics, and on-shell backgrounds remain absent.",
            "A source gate depending on X_phi inside the physical metric can add matter-dependent scalar principal terms or higher derivatives; only the non-derivative source gate receives conditional structural promise.",
            "B with constant Y avoids the specific kinetic-gate cross term, but a mass gate Z(X_phi) still changes the phi self-principal tensor when chi is nonzero.",
            "C controls a field-space Schur complement but lacks an environmental gate; D loses a propagating finite-range channel; E has unresolved constraint closure and degree-of-freedom count.",
            "One universal physical metric is mandatory; no independent photon or lensing multiplier has been introduced or authorized.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(
        receipt,
        {
            "schema_version",
            "analysis_id",
            "status",
            "decision",
            "config_binding",
            "implementation_binding",
            "predecessor_receipt_bindings",
            "shared_requirements",
            "architectures",
            "ranking_contract",
            "symbolic_suite",
            "adjudication",
            "counts",
            "claim_boundary",
            "zero_access_and_compute",
            "limitations",
            "content_sha256",
        },
        "receipt",
    )
    body = dict(receipt)
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(
        receipt["schema_version"] == RECEIPT_SCHEMA and receipt["decision"] == DECISION,
        "receipt identity changed",
    )
    _require(
        receipt["config_binding"]["content_sha256"] == _sha(config),
        "receipt config binding changed",
    )
    _require(
        receipt["predecessor_receipt_bindings"] == config["predecessor_receipt_bindings"],
        "receipt predecessors changed",
    )
    _require(
        receipt["architectures"] == config["architectures"]
        and receipt["ranking_contract"] == config["ranking_contract"],
        "comparison changed",
    )
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claims changed",
    )
    counts = receipt["counts"]
    _require(
        counts["architectures_compared"] == counts["universal_metric_architectures"] == 5,
        "architecture count changed",
    )
    _require(counts["independent_photon_multipliers"] == 0, "photon multiplier introduced")
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 12,
        "symbolic count changed",
    )
    _require(
        counts["healthy_architectures_established"]
        == counts["novel_architectures_established"]
        == 0,
        "health or novelty overclaimed",
    )
    _require(
        all(
            counts[key] == 0
            for key in (
                "observational_files_opened",
                "network_calls",
                "model_or_paid_calls",
                "gpu_calls",
            )
        ),
        "receipt access changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise RemedyComparisonError(f"refusing to overwrite different receipt: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RemedyComparisonError(
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
        "architectures_compared": receipt["counts"]["architectures_compared"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "leading_branch": receipt["adjudication"]["leading_structural_branch"],
        "healthy_architecture_identified": receipt["adjudication"][
            "healthy_architecture_identified"
        ],
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
    except RemedyComparisonError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
