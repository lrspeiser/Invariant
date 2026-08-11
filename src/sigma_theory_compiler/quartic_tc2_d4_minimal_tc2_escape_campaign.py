from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _matrix_payload,
)
from .quartic_tc2_fourth_jet_parallel_kernel import _polarized_fourth_payload
from .quartic_tc2_mixed_third_jet_chunk_campaign import _solve_sylvester
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import (
    STATE_DIMENSION,
    _reference_and_first_jet_packet,
)

SCHEMA = "sigma-quartic-tc2-d4-minimal-tc2-escape-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-minimal-tc2-escape-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_CANONICAL_RHS_SHA256 = (
    "47cb50c6c9e882626e4b5eba3f548be8ed162d076ed275f0edb7b230970d4850"
)
EXPECTED_CANONICAL_COMPRESSION_SHA256 = (
    "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
)
CORRECTION_BLOCK_SHA256 = (
    "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
)
CORRECTION_WEDGE_SHA256 = (
    "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"
)
CORRECTED_SYMBOLIC_DELTA_SHA256 = (
    "4df7e57e8235108dbee4c0cdf9559755de22a87492724db0ebe0cea1a6dbc039"
)
CANDIDATE_DELTA_HASHES = {
    "-1": "8a7d4874498255aa47acc68492a5bfe477bbad4eedd2a34ec2a0b15d1c5ad3a0",
    "-1/2": "dd980f4500438246e5b58822dd2fdbe066cb0a1679e7afbad25d77d0d5bdd338",
    "1/2": "6435012a1778ed41ba9e6ad90ade7a7caebac57bf21eabd6309a29cf1c947a37",
    "1": "79a59d7a622e296ef5db98d8b116a2140acf1a8f5a29b9187829d3b7cab2040a",
}


class QuarticTC2D4MinimalTC2EscapeCampaignError(ValueError):
    """Raised when the bounded TC2 obstruction-escape campaign is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _sparse_payload(matrix: sp.Matrix) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _correction_basis() -> dict[str, sp.Matrix]:
    reference = _reference_and_first_jet_packet()
    energy0 = reference["energy0"]
    left = sp.zeros(STATE_DIMENSION, 1)
    left[16] = 1
    left[28] = 1
    right = sp.zeros(STATE_DIMENSION, 1)
    right[21] = 1
    wedge = left * right.T - right * left.T
    block = (energy0.inv() * left * right.T).applyfunc(sp.factor)
    induced = (energy0 * block - block.T * energy0).applyfunc(sp.factor)
    zero_projector = reference["projectors"][sp.S.Zero]
    if (
        not energy0.equals(energy0.T)
        or energy0.det() == 0
        or block.rank() != 1
        or sum(value != 0 for value in block) != 6
        or induced != wedge
        or wedge.rank() != 2
        or (zero_projector.T * wedge * zero_projector).applyfunc(sp.factor) != wedge
        or _content_hash(_matrix_payload(block)) != CORRECTION_BLOCK_SHA256
        or _content_hash(_matrix_payload(wedge)) != CORRECTION_WEDGE_SHA256
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "minimal rank-one TC2 correction basis mismatch"
        )
    return {"block": block, "wedge": wedge, "left": left, "right": right}


def _canonical_payload(campaign: dict[str, Any]) -> dict[str, sp.Matrix]:
    directions = _active_directions()
    basis = [
        directions[position] for position in campaign["selector"]["active_positions"]
    ]
    payload, evaluations = _polarized_fourth_payload(ACTIVE_INDICES, basis)
    if (
        evaluations != 15
        or _content_hash(_matrix_payload(payload["fourth_Sylvester_RHS"]))
        != EXPECTED_CANONICAL_RHS_SHA256
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "canonical D4 polarization replay mismatch"
        )
    return payload


def _exact_escape(
    payload: dict[str, sp.Matrix], candidates: dict[str, Any]
) -> dict[str, Any]:
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha")
    if alpha is None:
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "canonical obstruction lost alpha"
        )
    eta = sp.Symbol("eta", real=True)
    basis = _correction_basis()
    wedge = basis["wedge"]
    scalar = sp.Rational(34816, 15) * alpha**5
    reference = _reference_and_first_jet_packet()
    compressions = {
        eigenvalue: (
            projector.T * (rhs + eta * wedge) * projector
        ).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    if (
        not (
            compressions[sp.S.Zero] - (scalar + eta) * wedge
        ).applyfunc(sp.factor).is_zero_matrix
        or any(
            not matrix.is_zero_matrix
            for eigenvalue, matrix in compressions.items()
            if eigenvalue != 0
        )
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "induced correction cokernel map mismatch"
        )
    tuned_eta = -scalar
    corrected_rhs = (rhs + tuned_eta * wedge).applyfunc(sp.factor)
    solvable, delta, audit = _solve_sylvester(corrected_rhs)
    if (
        not solvable
        or not audit["residual_zero"]
        or audit["nonzero_equal_eigenspace_compressions"]
        or not delta.equals(delta.T)
        or delta.rank() != 2
        or sum(value != 0 for value in delta) != 4
        or _content_hash(_matrix_payload(delta)) != CORRECTED_SYMBOLIC_DELTA_SHA256
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "symbolically tuned D4 Sylvester solve failed"
        )
    candidate_rows: list[dict[str, Any]] = []
    for certificate in sorted(
        candidates["certificates"], key=lambda row: row["candidate_id"]
    ):
        candidate_id = certificate["candidate_id"]
        alpha_value = sp.sympify(certificate["coefficients"]["a10"])
        eta_value = sp.factor(tuned_eta.subs(alpha, alpha_value))
        candidate_rhs = (
            rhs.subs(alpha, alpha_value) + eta_value * wedge
        ).applyfunc(sp.factor)
        candidate_solvable, candidate_delta, candidate_audit = _solve_sylvester(
            candidate_rhs
        )
        delta_sha256 = _content_hash(_matrix_payload(candidate_delta))
        if (
            not candidate_solvable
            or not candidate_audit["residual_zero"]
            or candidate_audit["nonzero_equal_eigenspace_compressions"]
            or not candidate_delta.equals(candidate_delta.T)
            or candidate_delta.rank() != 2
            or sum(value != 0 for value in candidate_delta) != 4
            or delta_sha256 != CANDIDATE_DELTA_HASHES[str(alpha_value)]
        ):
            raise QuarticTC2D4MinimalTC2EscapeCampaignError(
                f"candidate-specific escape failed: {candidate_id}"
            )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "a10": str(alpha_value),
                "c20": str(sp.sympify(certificate["coefficients"]["c20"])),
                "eta_unique_tuning": str(eta_value),
                "corrected_equal_eigenspace_compressions_zero": True,
                "corrected_D4_Sylvester_solvable": True,
                "corrected_deltaK_Hermitian": True,
                "corrected_deltaK_rank": 2,
                "corrected_deltaK_nonzero_entries": 4,
                "corrected_deltaK_sha256": delta_sha256,
                "corrected_D4_Sylvester_residual_zero": True,
                "covariant_operator_origin_proved": False,
            }
        )
    eta_values = sorted(
        {sp.sympify(row["eta_unique_tuning"]) for row in candidate_rows}
    )
    if eta_values != [
        -sp.Rational(34816, 15),
        -sp.Rational(1088, 15),
        sp.Rational(1088, 15),
        sp.Rational(34816, 15),
    ]:
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "candidate tuning value classification mismatch"
        )
    wrong_sign = (rhs + scalar * wedge).applyfunc(sp.factor)
    wrong_solvable, _, wrong_audit = _solve_sylvester(wrong_sign)
    if wrong_solvable or not wrong_audit["nonzero_equal_eigenspace_compressions"]:
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "wrong-sign negative control did not obstruct"
        )
    return {
        "correction_ansatz": {
            "name": "rank-one quartic stationary-sector TC2 counterterm",
            "active_monomial": "Y_0*Y_2*Y_3*Y_9",
            "lower_derivatives_orders_0_through_3_zero": True,
            "normalized_D4_block": "eta*V",
            "V_definition": "K55(0)^(-1)*(e_16+e_28)*e_21^T",
            "V_sparse": _sparse_payload(basis["block"]),
            "V_rank": 1,
            "V_nonzero_entries": 6,
            "V_sha256": CORRECTION_BLOCK_SHA256,
            "energy_skew_definition": "K55(0)*V-V^T*K55(0)=W",
            "W_sparse": _sparse_payload(wedge),
            "W_rank": 2,
            "W_nonzero_entries": 4,
            "W_sha256": CORRECTION_WEDGE_SHA256,
            "correction_parameter_dimension": 1,
            "candidate_bound_parameter": True,
            "covariant_or_action_derived": False,
        },
        "induced_cokernel_map": {
            "domain_basis": ["eta"],
            "image_basis": ["W"],
            "formula": "eta -> eta*W",
            "rank": 1,
            "image_dimension": 1,
            "canonical_obstruction_line_dimension": 1,
            "canonical_obstruction_in_image": True,
            "corrected_zero_eigenspace_compression": (
                "((34816/15)*alpha^5+eta)*W"
            ),
            "unique_solvability_condition": "eta=-(34816/15)*alpha^5",
            "all_other_equal_eigenspace_compressions_zero": True,
        },
        "minimality": {
            "zero_parameter_correction_can_escape": False,
            "minimum_parameter_dimension": 1,
            "rank_zero_block_can_escape": False,
            "minimum_block_rank": 1,
            "rank_one_block_sufficient": True,
            "target_wedge_rank": 2,
            "proof": (
                "The canonical cokernel target is a nonzero one-dimensional line spanned "
                "by rank-two W. A zero-dimensional or rank-zero correction has zero image. "
                "The displayed rank-one V has energy-skew exactly W, so both lower bounds "
                "are attained."
            ),
        },
        "symbolic_tuned_solution": {
            "eta": str(tuned_eta),
            "deltaK_sparse": _sparse_payload(delta),
            "deltaK_rank": 2,
            "deltaK_nonzero_entries": 4,
            "deltaK_Hermitian": True,
            "deltaK_sha256": CORRECTED_SYMBOLIC_DELTA_SHA256,
            "equal_eigenspace_compressions_zero": True,
            "Sylvester_residual_zero": True,
        },
        "candidate_classification": candidate_rows,
        "distinct_candidate_eta_values": [str(value) for value in eta_values],
        "wrong_sign_control": {
            "eta": "+(34816/15)*alpha^5",
            "resulting_compression": "(69632/15)*alpha^5*W",
            "solvable_for_registered_nonzero_alpha": False,
            "rejected": True,
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
        or config.get("global_claim_policy") != "fail_closed"
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "minimal TC2 escape config mismatch"
        )
    for local_key in ("campaign_source", "campaign_test"):
        binding = config[local_key]
        if _file_sha256((project_root / binding["path"]).read_bytes()) != binding["file_sha256"]:
            raise QuarticTC2D4MinimalTC2EscapeCampaignError(
                f"local binding mismatch: {local_key}"
            )
    obstruction = _load_bound(project_root, config["obstruction_certificate"])
    homogeneous = _load_bound(project_root, config["homogeneous_freedom_reduction"])
    campaign = _load_bound(project_root, config["fourth_campaign"])
    candidates = _load_bound(project_root, config["candidate_source"])
    if (
        obstruction.get("exact_symbolic_certificate", {})
        .get("equal_eigenspace_compressions", {})
        .get("zero_eigenspace", {})
        .get("sha256")
        != EXPECTED_CANONICAL_COMPRESSION_SHA256
        or homogeneous.get("homogeneous_freedom_reduction", {}).get(
            "induced_D4_zero_eigenspace_map_rank"
        )
        != 0
        or homogeneous.get("claims", {}).get(
            "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244"
        )
        is not True
        or len(candidates.get("certificates", [])) != EXPECTED_CANDIDATES
        or campaign.get("selector", {}).get("records", [])[OBLIGATION_OFFSET].get(
            "active_indices"
        )
        != list(ACTIVE_INDICES)
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "bound predecessor contract mismatch"
        )
    exact = _exact_escape(_canonical_payload(campaign), candidates)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: {
                "path": config[key]["path"],
                "file_sha256": config[key]["file_sha256"],
                **(
                    {"content_sha256": config[key]["content_sha256"]}
                    if "content_sha256" in config[key]
                    else {}
                ),
            }
            for key in (
                "obstruction_certificate",
                "homogeneous_freedom_reduction",
                "fourth_campaign",
                "candidate_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "active_positions": [0, 2, 4, 15],
            "multiplicity_partition": "ABCD",
            "canonical_rhs_sha256": EXPECTED_CANONICAL_RHS_SHA256,
            "canonical_compression_sha256": EXPECTED_CANONICAL_COMPRESSION_SHA256,
        },
        "exact_escape": exact,
        "counts": {
            "selector_obligations_touched": 1,
            "correction_basis_dimension": 1,
            "correction_block_rank": 1,
            "induced_cokernel_map_rank": 1,
            "target_cokernel_line_dimension": 1,
            "candidate_specializations_checked": EXPECTED_CANDIDATES,
            "candidate_D4_solutions_after_tuning": EXPECTED_CANDIDATES,
            "candidate_D4_obstructions_after_tuning": 0,
            "distinct_candidate_tunings": 4,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "zero_correction": {
                "eta": "0",
                "remaining_compression": "(34816/15)*alpha^5*W",
                "registered_candidates_solvable": 0,
                "rejected": True,
            },
            "wrong_sign": exact["wrong_sign_control"],
            "one_universal_eta_for_all_candidates": {
                "distinct_required_values": exact["distinct_candidate_eta_values"],
                "single_eta_closes_all_12": False,
                "rejected": True,
            },
            "homogeneous_lower_jet_repair": {
                "induced_cokernel_map_rank": 0,
                "rejected": True,
            },
            "infer_covariant_origin": {
                "state_space_block_constructed": True,
                "covariant_action_term_constructed": False,
                "gauge_constraint_compatibility_proved": False,
                "rejected": True,
            },
            "promote_one_D4_escape_to_global_closure": {
                "later_selector_obligations_evaluated": False,
                "tube_or_remainder_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "obligation_244_minimal_algebraic_TC2_escape_constructed": True,
            "candidate_specific_tuned_D4_compatibility_count": 12,
            "single_universal_eta_closes_all_12": False,
            "correction_covariant_or_action_derived": False,
            "correction_gauge_constraint_compatible": False,
            "corrected_candidate_family_registered": False,
            "all_3060_fourth_jet_obligations_evaluated": False,
            "full_fourth_jet_range_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": (
            "Derive the rank-one quartic state-space counterterm from a covariant action or "
            "admissible gauge-fixed TC2 operator and prove constraint compatibility, or reject "
            "the algebraic escape as nonphysical. Only then register corrected candidates and "
            "recompute their affected jet obligations."
        ),
        "scope": (
            "This campaign proves the smallest algebraic state-space TC2 ansatz capable of "
            "canceling the invariant obligation-244 cokernel witness. The tuned coefficient is "
            "candidate-specific. No covariant/action origin, gauge compatibility, corrected "
            "candidate registration, remaining D4 selector pass, tube theorem, CK1, CK3, TC2, "
            "B7, global-H7, or lifespan result is inferred."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "minimal TC2 escape content identity mismatch"
        )
    exact = document.get("exact_escape", {})
    ansatz = exact.get("correction_ansatz", {})
    induced = exact.get("induced_cokernel_map", {})
    symbolic = exact.get("symbolic_tuned_solution", {})
    rows = exact.get("candidate_classification", [])
    expected_claims = {
        "obligation_244_minimal_algebraic_TC2_escape_constructed": True,
        "candidate_specific_tuned_D4_compatibility_count": 12,
        "single_universal_eta_closes_all_12": False,
        "correction_covariant_or_action_derived": False,
        "correction_gauge_constraint_compatible": False,
        "corrected_candidate_family_registered": False,
        "all_3060_fourth_jet_obligations_evaluated": False,
        "full_fourth_jet_range_closed": False,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }
    if (
        document.get("status")
        != "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only"
        or document.get("claims") != expected_claims
        or document.get("counts")
        != {
            "selector_obligations_touched": 1,
            "correction_basis_dimension": 1,
            "correction_block_rank": 1,
            "induced_cokernel_map_rank": 1,
            "target_cokernel_line_dimension": 1,
            "candidate_specializations_checked": 12,
            "candidate_D4_solutions_after_tuning": 12,
            "candidate_D4_obstructions_after_tuning": 0,
            "distinct_candidate_tunings": 4,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        }
        or ansatz.get("V_rank") != 1
        or ansatz.get("V_nonzero_entries") != 6
        or ansatz.get("V_sha256") != CORRECTION_BLOCK_SHA256
        or ansatz.get("W_rank") != 2
        or ansatz.get("W_sha256") != CORRECTION_WEDGE_SHA256
        or ansatz.get("covariant_or_action_derived") is not False
        or induced.get("rank") != 1
        or induced.get("image_dimension") != 1
        or induced.get("canonical_obstruction_in_image") is not True
        or induced.get("unique_solvability_condition")
        != "eta=-(34816/15)*alpha^5"
        or symbolic.get("deltaK_sha256") != CORRECTED_SYMBOLIC_DELTA_SHA256
        or symbolic.get("equal_eigenspace_compressions_zero") is not True
        or symbolic.get("Sylvester_residual_zero") is not True
        or len(rows) != EXPECTED_CANDIDATES
        or len({row.get("candidate_id") for row in rows}) != EXPECTED_CANDIDATES
        or any(
            row.get("corrected_D4_Sylvester_solvable") is not True
            or row.get("corrected_deltaK_Hermitian") is not True
            or row.get("corrected_D4_Sylvester_residual_zero") is not True
            or row.get("covariant_operator_origin_proved") is not False
            or row.get("corrected_deltaK_sha256")
            != CANDIDATE_DELTA_HASHES.get(row.get("a10"))
            for row in rows
        )
        or set(document.get("negative_controls", {}))
        != {
            "zero_correction",
            "wrong_sign",
            "one_universal_eta_for_all_candidates",
            "homogeneous_lower_jet_repair",
            "infer_covariant_origin",
            "promote_one_D4_escape_to_global_closure",
        }
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4MinimalTC2EscapeCampaignError(
            "minimal TC2 escape exact/fail-closed contract mismatch"
        )


def run_campaign(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the minimal algebraic TC2 escape for D4 obligation 244."
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
                "correction_dimension": artifact["counts"][
                    "correction_basis_dimension"
                ],
                "candidate_D4_solutions": artifact["counts"][
                    "candidate_D4_solutions_after_tuning"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
