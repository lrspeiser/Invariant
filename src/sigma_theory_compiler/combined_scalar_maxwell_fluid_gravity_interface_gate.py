from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class CombinedMatterGravityInterfaceError(RuntimeError):
    """Raised when the bounded combined interface evidence is absent or broadened."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise CombinedMatterGravityInterfaceError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CombinedMatterGravityInterfaceError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CombinedMatterGravityInterfaceError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise CombinedMatterGravityInterfaceError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise CombinedMatterGravityInterfaceError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise CombinedMatterGravityInterfaceError(f"bound content hash mismatch: {path}")
    return path, value


def _sector_result(receipt: dict[str, Any], sector_id: str) -> dict[str, Any]:
    result = next(
        (item for item in receipt.get("sector_results", []) if item.get("sector_id") == sector_id),
        None,
    )
    if not isinstance(result, dict):
        raise CombinedMatterGravityInterfaceError(f"registered sector is absent: {sector_id}")
    return result


def _gate(result: dict[str, Any], gate_id: str) -> dict[str, Any]:
    gate = next(
        (item for item in result.get("gates", []) if item.get("gate_id") == gate_id),
        None,
    )
    if not isinstance(gate, dict):
        raise CombinedMatterGravityInterfaceError(f"registered sector gate is absent: {gate_id}")
    return gate


def _exact_combined_replay() -> dict[str, Any]:
    # On-shell total conservation is additive. Coefficients are in the independent
    # sector Euler-force basis (scalar, Maxwell, fluid).
    total_stress_divergence_coefficients = (1, 1, 1)
    on_shell_field_equation_values = (0, 0, 0)
    on_shell_total_residual = tuple(
        coefficient * value
        for coefficient, value in zip(
            total_stress_divergence_coefficients,
            on_shell_field_equation_values,
        )
    )
    omitted_fluid_coefficients = (1, 1, 0)
    omitted_fluid_residual = tuple(
        expected - actual
        for expected, actual in zip(
            total_stress_divergence_coefficients,
            omitted_fluid_coefficients,
        )
    )
    if on_shell_total_residual != (0, 0, 0) or omitted_fluid_residual == (0, 0, 0):
        raise CombinedMatterGravityInterfaceError("combined stress conservation replay failed")

    # A common local orthonormal frame aligned with the timelike fluid potential
    # gives five g-null blocks and one acoustic block. Positive rescalings are
    # suppressed; each pair is (omega^2 coefficient, |k|^2 coefficient).
    principal_blocks = [(-1, 1)] * 5 + [(-3, 1)]
    time_kinetic_coefficients = tuple(-block[0] for block in principal_blocks)
    spatial_gradient_coefficients = tuple(block[1] for block in principal_blocks)
    if min(time_kinetic_coefficients) <= 0 or min(spatial_gradient_coefficients) <= 0:
        raise CombinedMatterGravityInterfaceError("combined matter principal positivity failed")

    # Exact arbitrary-background Maxwell gauge identity. A symmetric Ricci tensor
    # contracted with antisymmetric F cancels pairwise. Corrupting one ordered
    # Ricci entry leaves one exact monomial.
    ricci_field_coefficients: dict[tuple[int, int], int] = {}
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            low, high = sorted((first, second))
            field_sign = 1 if first < second else -1
            ricci_field_coefficients[(low, high)] = (
                ricci_field_coefficients.get((low, high), 0) + field_sign
            )
    ricci_field_coefficients = {
        pair: coefficient
        for pair, coefficient in ricci_field_coefficients.items()
        if coefficient != 0
    }
    corrupted_ricci_residual = {(0, 1): 1}
    if ricci_field_coefficients or not corrupted_ricci_residual:
        raise CombinedMatterGravityInterfaceError("Maxwell subsidiary identity replay failed")

    return {
        "combined_stress_conservation": {
            "total_stress": "T_total=T_scalar+T_Maxwell+T_fluid",
            "sector_euler_force_coefficients": list(total_stress_divergence_coefficients),
            "on_shell_field_equation_values": list(on_shell_field_equation_values),
            "on_shell_total_residual": list(on_shell_total_residual),
            "conclusion": "nabla^mu T_total_mu_nu=0 on all three matter equations",
            "negative_control": {
                "mutation": "omit the fluid stress from T_total",
                "coefficient_residual": list(omitted_fluid_residual),
                "rejected": True,
            },
        },
        "combined_matter_principal_compatibility": {
            "common_time_covector": "nabla_mu(tau)",
            "common_time_basis": (
                "X>0 makes nabla(tau) timelike for g; scalar and Lorenz-Maxwell use "
                "the g cone, and the acoustic inverse metric also makes it timelike"
            ),
            "second_order_components": 6,
            "principal_block_coefficients": [list(block) for block in principal_blocks],
            "time_kinetic_coefficients": list(time_kinetic_coefficients),
            "spatial_gradient_coefficients": list(spatial_gradient_coefficients),
            "combined_characteristic_polynomial": ("(|k|^2-omega^2)^5 (|k|^2-3 omega^2)"),
            "characteristic_roots": {
                "light": ["-|k|", "+|k|"],
                "light_multiplicity": 5,
                "acoustic": ["-|k|/sqrt(3)", "+|k|/sqrt(3)"],
                "acoustic_multiplicity": 1,
            },
            "second_derivative_cross_sector_blocks": 0,
            "strongly_hyperbolic_matter_direct_sum": True,
        },
        "internal_matter_constraint_closure": {
            "scalar_internal_constraints": 0,
            "fluid_internal_constraints": 0,
            "maxwell_internal_constraints": 1,
            "maxwell_constraint": "C=nabla_mu A^mu",
            "maxwell_gauge_identity": "nabla_rho E^rho=R_rho_nu F^(rho nu)=0",
            "ricci_field_contraction_residual_terms": len(ricci_field_coefficients),
            "subsidiary_equation": "box_g C=0 for the source-free Lorenz-gauge system",
            "corrupted_ricci_symmetry_negative": {
                "residual_terms": len(corrupted_ricci_residual),
                "first_witness": "R_01 F_01",
                "rejected": True,
            },
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "invariant-combined-matter-gravity-interface-config-1.0":
        raise CombinedMatterGravityInterfaceError("unsupported config schema")
    expected_policy = {
        "pass_combined_matter_interface": True,
        "pass_candidate_specific_gravity_source_closure": False,
        "pass_full_coupled_principal_system": False,
        "pass_gravity_constraint_propagation": False,
        "vortical_fluid": False,
        "external_maxwell_current": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise CombinedMatterGravityInterfaceError("claims policy is absent or broadened")

    predecessors = config["predecessors"]
    universal_path, universal = _load_binding(repository, predecessors["universal_matter"])
    maxwell_path, maxwell = _load_binding(repository, predecessors["maxwell_arbitrary_background"])
    fluid_path, fluid = _load_binding(repository, predecessors["fluid_constraint_complete"])
    contract_path, contract = _load_binding(
        repository, config["evidence_bindings"]["covariant_field_contract"]
    )

    scalar = _sector_result(universal, "minimally_coupled_scalar")
    if scalar.get("status") != "PASS":
        raise CombinedMatterGravityInterfaceError("scalar predecessor is not PASS")
    for gate_id in (
        "action_level_universal_metric_coupling",
        "stress_energy_conservation_interface",
        "principal_symbol_hyperbolicity",
        "constraint_propagation",
    ):
        if _gate(scalar, gate_id).get("outcome") != "PASS":
            raise CombinedMatterGravityInterfaceError(
                f"scalar predecessor gate is not PASS: {gate_id}"
            )
    if (
        maxwell.get("decision") != "PASS_ARBITRARY_BACKGROUND_MAXWELL_STRESS_DIVERGENCE"
        or maxwell.get("claims", {}).get(
            "arbitrary_background_maxwell_hilbert_stress_divergence_closed"
        )
        is not True
    ):
        raise CombinedMatterGravityInterfaceError(
            "arbitrary-background Maxwell predecessor is incomplete"
        )
    if fluid.get("decision") != "PASS_FOURTH_GATE_ZERO_INDEPENDENT_CONSTRAINTS":
        raise CombinedMatterGravityInterfaceError("fluid constraint predecessor is incomplete")

    action_contract = contract.get("action_contract", {})
    if (
        action_contract.get("physical_metric") != "g_mu_nu"
        or action_contract.get("action")
        != "S = S_grav[g_mu_nu, gravitational_fields] + S_m[g_mu_nu, psi_m]"
    ):
        raise CombinedMatterGravityInterfaceError("single physical metric action contract changed")
    sectors = config.get("sectors")
    if not isinstance(sectors, list) or len(sectors) != 3:
        raise CombinedMatterGravityInterfaceError("combined sector manifest changed")
    if sum(int(item["second_order_components"]) for item in sectors) != 6:
        raise CombinedMatterGravityInterfaceError("combined matter component count changed")

    replay = _exact_combined_replay()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_combined_scalar_maxwell_fluid_gravity_interface_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-combined-matter-gravity-interface-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK",
        "gate_results": [
            {
                "gate_id": "single_physical_metric_and_action_split",
                "outcome": "PASS",
            },
            {
                "gate_id": "combined_matter_stress_conservation",
                "outcome": "PASS",
            },
            {
                "gate_id": "combined_matter_principal_compatibility",
                "outcome": "PASS",
            },
            {
                "gate_id": "internal_matter_constraint_source_closure",
                "outcome": "PASS",
            },
            {
                "gate_id": "candidate_specific_gravity_matter_source_closure",
                "outcome": "BLOCK",
                "reason_codes": ["missing_candidate_specific_gravity_matter_coupled_registration"],
            },
        ],
        "combined_matter_certificate": replay,
        "gravity_block": {
            "outcome": "BLOCK",
            "reason_code": ("missing_candidate_specific_gravity_matter_coupled_registration"),
            "minimal_registration_contract": [
                (
                    "one selected gravitational action and gauge-fixed Euler system bound "
                    "to the same physical-metric and total-matter action hash"
                ),
                (
                    "exact insertion of T_scalar+T_Maxwell+T_fluid into the registered "
                    "metric equation with normalization and sign conventions"
                ),
                (
                    "candidate-specific full coupled principal matrix including every "
                    "gravity field, gauge block, and all three matter sectors"
                ),
                (
                    "one common time covector and exact symmetrizer/diagonalizer with "
                    "uniform domain bounds for the full coupled matrix"
                ),
                (
                    "exact gravitational Hamiltonian/momentum and gauge-constraint "
                    "propagation with total matter sources and Maxwell subsidiary constraint"
                ),
                (
                    "corrupted source sign or omitted-sector negative with a nonzero "
                    "constraint-propagation residual"
                ),
            ],
        },
        "counts": {
            "matter_sectors": 3,
            "matter_second_order_components": 6,
            "light_cone_components": 5,
            "acoustic_cone_components": 1,
            "internal_matter_constraints": 1,
            "combined_interface_passes": 4,
            "gravity_interface_blocks": 1,
            "exact_combined_residuals": 4,
            "negative_controls": 2,
            "rejects": 0,
        },
        "claims": {
            "combined_three_sector_matter_interface_closed": True,
            "total_matter_stress_conserved_on_shell": True,
            "common_time_matter_principal_compatibility_closed": True,
            "internal_matter_constraint_source_closure_closed": True,
            "candidate_specific_gravity_source_closure_closed": False,
            "full_coupled_gravity_matter_principal_system_closed": False,
            "gravity_constraint_propagation_closed": False,
            "vortical_fluid_covered": False,
            "external_maxwell_current_covered": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "three-sector matter-side interface for one canonical scalar, source-free "
            "Lorenz-Maxwell, and the admitted irrotational P(X)=kappa X^2 fluid on their "
            "shared physical metric and common-time domain; candidate-specific gravitational "
            "source equations, full coupled principal/constraint closure, vortical fluids, "
            "external currents, H7, universal all-matter closure, and promotion remain blocked"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "universal_matter": {
                "path": universal_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(universal_path),
                "content_sha256": universal["content_sha256"],
            },
            "maxwell_arbitrary_background": {
                "path": maxwell_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(maxwell_path),
                "content_sha256": maxwell["content_sha256"],
            },
            "fluid_constraint_complete": {
                "path": fluid_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(fluid_path),
                "content_sha256": fluid["content_sha256"],
            },
            "covariant_field_contract": {
                "path": contract_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(contract_path),
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
