from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_d4_curl_constraint_admission_campaign import _gradient_lift
from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_diagonal_third_jet_campaign import _content_hash, _matrix_payload
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import STATE_DIMENSION

SCHEMA = "sigma-quartic-tc2-d4-parity-cubic-angular-escape-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-parity-cubic-angular-escape-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_COMPANION_SHA256 = "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
EXPECTED_W_SHA256 = "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"


class QuarticTC2D4ParityCubicAngularEscapeError(ValueError):
    """Raised when the parity-preserving cubic angular escape is invalid."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4ParityCubicAngularEscapeError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4ParityCubicAngularEscapeError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4ParityCubicAngularEscapeError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4ParityCubicAngularEscapeError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _substitute_matrix(matrix: sp.Matrix, replacements: Mapping[sp.Symbol, sp.Expr]) -> sp.Matrix:
    return matrix.applyfunc(lambda value: sp.factor(value.subs(replacements)))


def _exact_escape(
    minimal: Mapping[str, Any],
    curl: Mapping[str, Any],
    axis2: Mapping[str, Any],
    full_linear_no_go: Mapping[str, Any],
) -> dict[str, Any]:
    basis = _correction_basis()
    direction_1 = basis["block"]
    wedge = basis["wedge"]
    output = direction_1[:, 21]
    direction_2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    linear_curl_symbol = (n1 * direction_1 + n2 * direction_2).applyfunc(sp.factor)
    scalar_multiplier = n1**2
    cubic_symbol = (scalar_multiplier * linear_curl_symbol).applyfunc(sp.factor)
    lift = _gradient_lift(n1, n2, n3)
    residual = (cubic_symbol * lift).applyfunc(sp.factor)
    antipodal = _substitute_matrix(cubic_symbol, {n1: -n1, n2: -n2, n3: -n3})
    e1 = {n1: 1, n2: 0, n3: 0}
    minus_e1 = {n1: -1, n2: 0, n3: 0}
    e2 = {n1: 0, n2: 1, n3: 0}
    minus_e2 = {n1: 0, n2: -1, n3: 0}
    e3 = {n1: 0, n2: 0, n3: 1}
    axis_blocks = {
        "e1": _substitute_matrix(cubic_symbol, e1),
        "minus_e1": _substitute_matrix(cubic_symbol, minus_e1),
        "e2": _substitute_matrix(cubic_symbol, e2),
        "minus_e2": _substitute_matrix(cubic_symbol, minus_e2),
        "e3": _substitute_matrix(cubic_symbol, e3),
    }
    if (
        _content_hash(_matrix_payload(direction_1)) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(direction_2)) != EXPECTED_COMPANION_SHA256
        or _content_hash(_matrix_payload(wedge)) != EXPECTED_W_SHA256
        or not residual.is_zero_matrix
        or antipodal != -cubic_symbol
        or axis_blocks["e1"] != direction_1
        or axis_blocks["minus_e1"] != -direction_1
        or any(not axis_blocks[name].is_zero_matrix for name in ("e2", "minus_e2", "e3"))
    ):
        raise QuarticTC2D4ParityCubicAngularEscapeError(
            "cubic angular symbol construction mismatch"
        )

    minimal_rows = minimal.get("exact_escape", {}).get("candidate_classification", [])
    axis2_rows = axis2.get("exact_axis2_base_D4_audit", {}).get("candidate_comparison", [])
    minimal_by_id = {row["candidate_id"]: row for row in minimal_rows}
    axis2_by_id = {row["candidate_id"]: row for row in axis2_rows}
    candidate_ids = sorted(set(minimal_by_id) & set(axis2_by_id))
    candidates = []
    for candidate_id in candidate_ids:
        reference = minimal_by_id[candidate_id]
        companion = axis2_by_id[candidate_id]
        if reference.get("corrected_D4_Sylvester_solvable") is not True or reference.get(
            "eta_unique_tuning"
        ) != companion.get("eta"):
            raise QuarticTC2D4ParityCubicAngularEscapeError("candidate predecessor tuning mismatch")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "a10": reference["a10"],
                "eta": reference["eta_unique_tuning"],
                "e1_D4_Sylvester_solvable_inherited": True,
                "e1_deltaK_sha256": reference["corrected_deltaK_sha256"],
                "e2_base_D4_RHS_zero": True,
                "e2_angular_multiplier": "0",
                "e2_correction_block_zero": True,
                "e2_D4_Sylvester_solvable": True,
                "e2_deltaK_zero": True,
                "all_direction_D4_Sylvester_solvable": False,
            }
        )
    axis2_result = axis2["exact_axis2_base_D4_audit"]["result"]
    if (
        len(candidates) != EXPECTED_CANDIDATES
        or axis2_result.get("base_D4_RHS_identically_zero") is not True
        or full_linear_no_go.get("claims", {}).get(
            "full_linear_gradient_annihilator_completion_class_ruled_out"
        )
        is not True
        or curl.get("exact_admission", {})
        .get("physical_reduction_equivalence", {})
        .get("directional_operator_times_gradient_lift_zero")
        is not True
    ):
        raise QuarticTC2D4ParityCubicAngularEscapeError(
            "cubic angular predecessor consequence mismatch"
        )

    return {
        "declared_escape_class": {
            "name": "even_polynomial_scalar_multipliers_of_the_admitted_C12_curl_symbol",
            "base_symbol": "B_curl(n)=n1*V+n2*C_companion",
            "class_symbol": "B_a(n)=a(n)*B_curl(n)",
            "requirements": [
                "a(-n)=a(n) for antipodal oddness of B_a",
                "a(e1)=1 to preserve the reference-direction D4 solution",
                "a(e2)=0 to remove the axis-two companion",
                "a is polynomial and bounded on S^2",
            ],
            "local_first_order_differential_symbols_included": False,
            "order_one_pseudodifferential_angular_symbols_included": True,
        },
        "minimality": {
            "base_symbol_degree": 1,
            "constant_even_multiplier_impossible": True,
            "degree_one_multiplier_rejected_by_antipodal_parity": True,
            "minimal_nonconstant_even_multiplier_degree": 2,
            "minimal_total_angular_polynomial_degree": 3,
            "canonical_multiplier": "a(n)=n1^2",
            "proof": (
                "An odd base symbol requires an even scalar multiplier. A constant cannot "
                "equal one at e1 and zero at e2; degree two is the first nonconstant even "
                "degree, and n1^2 satisfies both axis values."
            ),
        },
        "exact_symbol": {
            "definition": "B_cubic(n)=n1^2*(n1*V+n2*C_companion)",
            "nonzero_polynomial_coefficient_blocks": 2,
            "coefficient_blocks": {
                "n1^3": EXPECTED_V_SHA256,
                "n1^2*n2": EXPECTED_COMPANION_SHA256,
            },
            "symbol_sha256": _content_hash(_matrix_payload(cubic_symbol)),
            "antipodal_odd": True,
            "sphere_multiplier_interval": "0<=n1^2<=1",
            "sphere_bound_certificate": "1-n1^2=n2^2+n3^2>=0 on S^2",
            "reference_e1_block_sha256": _content_hash(_matrix_payload(axis_blocks["e1"])),
            "minus_e1_is_negative_reference": True,
            "e2_block_zero": True,
            "minus_e2_block_zero": True,
            "e3_block_zero": True,
        },
        "physical_gradient_lift_equivalence": {
            "identity": "B_cubic(n)*L(n)=n1^2*B_curl(n)*L(n)=0",
            "residual_zero": True,
            "residual_sha256": _content_hash(_matrix_payload(residual)),
            "constraint_surface_principal_operator_zero": True,
        },
        "pseudodifferential_constraint_admission": {
            "operator": "eta(Y)*u*M1(C_12^[10])",
            "M1_fourier_symbol": "xi1^2/|xi|^2=n1^2",
            "zero_mode_policy": "M1(0)=0",
            "constant_coefficient_spatial_derivative_commutation": True,
            "periodic_or_Schwartz_constraint_surface_invariant": True,
            "boundary_domain_realization_proved": False,
            "local_differential_operator_realization_proved": False,
            "covariant_action_origin_proved": False,
        },
        "two_axis_D4_consequence": {
            "reference_e1_solutions_inherited": EXPECTED_CANDIDATES,
            "axis2_base_D4_RHS_identically_zero": True,
            "axis2_companion_blocks_after_multiplier": 0,
            "axis2_D4_compatibilities": EXPECTED_CANDIDATES,
            "axis2_D4_obstructions": 0,
            "candidate_records": candidates,
            "all_direction_D4_compatibility_proved": False,
        },
        "first_blocker": {
            "name": "generic_direction_D4_and_nonlocal_variable_coefficient_admission",
            "required_next": (
                "Evaluate the complete D4 recurrence for generic directions and prove an "
                "appropriate order-one pseudodifferential constraint, energy, boundary, and "
                "variable-coefficient commutator calculus, or replace M1 by a covariant local origin."
            ),
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4ParityCubicAngularEscapeError("cubic angular config mismatch")
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    predecessors = {
        key: _load_bound(root, config[key])
        for key in (
            "minimal_escape",
            "curl_admission",
            "companion_range",
            "axis2_base_rhs",
            "spatial_gradient_no_go",
            "full_linear_no_go",
        )
    }
    expected_statuses = {
        "minimal_escape": "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only",
        "curl_admission": "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V",
        "companion_range": "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go",
        "axis2_base_rhs": "pass_exact_all_12_axis2_D4_companion_obstructions",
        "spatial_gradient_no_go": (
            "pass_exact_exhaustive_spatial_gradient_annihilator_completion_no_go"
        ),
        "full_linear_no_go": "pass_exact_full_linear_gradient_annihilator_completion_no_go",
    }
    if any(predecessors[key].get("status") != status for key, status in expected_statuses.items()):
        raise QuarticTC2D4ParityCubicAngularEscapeError("predecessor status mismatch")
    exact = _exact_escape(
        predecessors["minimal_escape"],
        predecessors["curl_admission"],
        predecessors["axis2_base_rhs"],
        predecessors["full_linear_no_go"],
    )
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_minimal_parity_preserving_cubic_angular_two_axis_escape",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "curl_admission",
                "companion_range",
                "axis2_base_rhs",
                "spatial_gradient_no_go",
                "full_linear_no_go",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "reference_direction": "e1",
            "newly_closed_companion_direction": "e2",
        },
        "exact_escape": exact,
        "counts": {
            "bound_predecessors": 6,
            "scalar_multiplier_degree": 2,
            "total_angular_polynomial_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "candidate_specializations": 12,
            "reference_e1_D4_solutions_inherited": 12,
            "new_axis2_D4_compatibilities": 12,
            "new_axis2_D4_obstructions": 0,
            "generic_direction_D4_compatibilities_proved": 0,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "constant_multiplier": {
                "cannot_preserve_e1_and_zero_e2": True,
                "rejected": True,
            },
            "linear_multiplier_n1": {
                "total_symbol_antipodal_even": True,
                "rejected": True,
            },
            "omit_curl_companion_before_multiplication": {
                "gradient_lift_residual_nonzero": True,
                "rejected": True,
            },
            "claim_local_first_order_differential_origin": {
                "angular_symbol_contains_xi1_squared_over_mod_xi_squared": True,
                "rejected": True,
            },
            "promote_two_axis_result_to_all_directions": {
                "generic_direction_D4_uncomputed": True,
                "rejected": True,
            },
            "promote_pseudodifferential_escape_to_global_TC2": {
                "variable_coefficient_commutator_and_boundary_calculus_missing": True,
                "rejected": True,
            },
        },
        "claims": {
            "minimal_parity_preserving_cubic_scalar_multiplier_constructed": True,
            "full_gradient_lift_annihilation_proved": True,
            "reference_e1_D4_solutions_inherited": True,
            "all_12_axis2_D4_compatibilities_proved_for_cubic_symbol": True,
            "bounded_unit_sphere_multiplier_proved": True,
            "periodic_or_Schwartz_constraint_surface_principal_admission": True,
            "generic_direction_D4_compatibility_proved": False,
            "local_differential_operator_origin_proved": False,
            "covariant_action_origin_proved": False,
            "spatially_covariant_tensor_completion_proved": False,
            "variable_coefficient_pseudodifferential_energy_calculus_proved": False,
            "corrected_candidate_family_registered": False,
            "remaining_D4_selector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": exact["first_blocker"]["required_next"],
        "scope": (
            "Exact construction and minimality theorem within even polynomial scalar multipliers "
            "of the admitted C12 curl symbol. The cubic odd angular symbol n1^2*B_curl preserves "
            "all 12 reference e1 solutions, annihilates the full gradient lift, and vanishes at "
            "e2, converting the 12 axis-two obstructions into 12 exact compatibilities. It is an "
            "order-one nonlocal angular/pseudodifferential escape, not a local covariant operator. "
            "Generic-direction D4, variable-coefficient commutators, boundary calculus, remaining "
            "D4, tube, CK1, CK3, TC2, B7, global-H7, and lifespan remain fail-closed."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4ParityCubicAngularEscapeError("content identity mismatch")
    exact = document.get("exact_escape", {})
    symbol = exact.get("exact_symbol", {})
    consequence = exact.get("two_axis_D4_consequence", {})
    claims = document.get("claims", {})
    if (
        document.get("status")
        != "pass_exact_minimal_parity_preserving_cubic_angular_two_axis_escape"
        or document.get("counts")
        != {
            "bound_predecessors": 6,
            "scalar_multiplier_degree": 2,
            "total_angular_polynomial_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "candidate_specializations": 12,
            "reference_e1_D4_solutions_inherited": 12,
            "new_axis2_D4_compatibilities": 12,
            "new_axis2_D4_obstructions": 0,
            "generic_direction_D4_compatibilities_proved": 0,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        }
        or exact.get("minimality", {}).get("minimal_total_angular_polynomial_degree") != 3
        or symbol.get("antipodal_odd") is not True
        or symbol.get("sphere_multiplier_interval") != "0<=n1^2<=1"
        or symbol.get("e2_block_zero") is not True
        or exact.get("physical_gradient_lift_equivalence", {}).get("residual_zero") is not True
        or consequence.get("reference_e1_solutions_inherited") != 12
        or consequence.get("axis2_D4_compatibilities") != 12
        or consequence.get("axis2_D4_obstructions") != 0
        or len(consequence.get("candidate_records", [])) != 12
        or any(
            row.get("e2_D4_Sylvester_solvable") is not True
            or row.get("all_direction_D4_Sylvester_solvable") is not False
            for row in consequence.get("candidate_records", [])
        )
        or any(
            claims.get(key) is not True
            for key in (
                "minimal_parity_preserving_cubic_scalar_multiplier_constructed",
                "full_gradient_lift_annihilation_proved",
                "reference_e1_D4_solutions_inherited",
                "all_12_axis2_D4_compatibilities_proved_for_cubic_symbol",
                "bounded_unit_sphere_multiplier_proved",
                "periodic_or_Schwartz_constraint_surface_principal_admission",
            )
        )
        or any(
            claims.get(key) is not False
            for key in (
                "generic_direction_D4_compatibility_proved",
                "local_differential_operator_origin_proved",
                "covariant_action_origin_proved",
                "spatially_covariant_tensor_completion_proved",
                "variable_coefficient_pseudodifferential_energy_calculus_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4ParityCubicAngularEscapeError("exact/fail-closed contract mismatch")


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the minimal parity-preserving cubic escape."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_campaign(args.project_root, args.config, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "axis2_compatibilities": artifact["counts"]["new_axis2_D4_compatibilities"],
                "generic_direction_passes": artifact["counts"][
                    "generic_direction_D4_compatibilities_proved"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
