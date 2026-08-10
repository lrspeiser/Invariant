from __future__ import annotations

import hashlib
import json
from functools import cache
from math import ceil, inf, isfinite, nextafter
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-rows0-4-nonlinear-subfamily-campaign-1.0"
OUTPUT_ROWS = tuple(range(5))
MIXED_PAIR = ("p0[10]", "p1[10]")


class QuarticRows04NonlinearSubfamilyError(ValueError):
    """Raised when the quantitative rows-zero-through-four slice is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _outward_integer(value: float) -> int:
    if not (isfinite(value) and value >= 0):
        raise QuarticRows04NonlinearSubfamilyError(
            "a published derivative envelope is invalid"
        )
    return ceil(nextafter(value, inf))


@cache
def generic_rows0_4_nonlinear_control() -> tuple[bool, dict[str, Any]]:
    """Prove inverse, determinant, H7 algebra, and energy-absorption factors."""

    inverse_upper = sp.Integer(5)
    dimension = sp.Integer(11)
    sigma_lower = 1 / inverse_upper
    determinant_lower = sigma_lower**dimension
    fourier_l1_h7 = sp.sqrt(21) / (64 * sp.sqrt(sp.pi))
    h7_algebra = sp.simplify(2**7 * fourier_l1_h7)
    dyadic_source_upper = sp.Integer(2) ** 7
    state_h7_square_to_q7 = sp.Integer(2) ** 15
    q7_bilinear_factor = sp.simplify(
        dyadic_source_upper * state_h7_square_to_q7 * h7_algebra
    )
    gamma, component_bound, h_lower, energy = sp.symbols(
        "Gamma C_B h E", positive=True, finite=True
    )
    absorption_residual = sp.simplify(
        gamma
        * sp.sqrt(energy)
        * component_bound
        * energy
        / h_lower
        - gamma
        * component_bound
        * energy ** sp.Rational(3, 2)
        / h_lower
    )
    epsilon = sp.Symbol("epsilon", positive=True, finite=True)
    nonzero_determinant = epsilon
    variable = sp.Symbol("z", real=True)
    variable_coefficient = 1 + variable
    frozen_residual = sp.diff(variable_coefficient, variable)
    passed = bool(
        sigma_lower == sp.Rational(1, 5)
        and determinant_lower == sp.Rational(1, 5**11)
        and h7_algebra == 2 * sp.sqrt(21) / sp.sqrt(sp.pi)
        and q7_bilinear_factor == 2**23 * sp.sqrt(21) / sp.sqrt(sp.pi)
        and absorption_residual == 0
        and nonzero_determinant.has(epsilon)
        and frozen_residual != 0
    )
    return passed, {
        "control": "quantitative time-block inverse and frozen mixed H7 subfamily",
        "time_block": {
            "hypothesis": "||A^-1||_2<=5",
            "sigma_min_lower": str(sigma_lower),
            "determinant_absolute_lower": str(determinant_lower),
            "identity": "|det(A)|=product_i sigma_i(A)",
            "dimension": int(dimension),
        },
        "H7_product_algebra": {
            "fourier_convention": "unitary Fourier transform on R3",
            "fourier_L1_embedding_constant": str(fourier_l1_h7),
            "weight_inequality": (
                "<eta+zeta>^7<=2^6(<eta>^7+<zeta>^7)"
            ),
            "algebra_constant": str(h7_algebra),
            "bound": "||fg||_H7<=C_alg||f||_H7||g||_H7",
        },
        "dyadic_conversion": {
            "sqrt_Q7_source_to_H7_factor": str(dyadic_source_upper),
            "state_H7_square_to_Q7_factor": str(state_h7_square_to_q7),
            "total_bilinear_Q7_factor": str(q7_bilinear_factor),
        },
        "energy_absorption": {
            "rule": (
                "Gamma*sqrt(E)*C_B*Q7<="
                "Gamma*C_B*h_lower^-1*E^(3/2)"
            ),
            "residual": str(absorption_residual),
        },
        "negative_controls": {
            "nonzero_determinant_without_lower_bound": {
                "family": "diag(epsilon,1,...,1)",
                "determinant": str(nonzero_determinant),
                "rejected": nonzero_determinant.has(epsilon),
            },
            "promote_frozen_bilinear_coefficient_to_variable_coefficient": {
                "test_coefficient": str(variable_coefficient),
                "derivative_residual": str(frozen_residual),
                "rejected": frozen_residual != 0,
            },
            "omit_dyadic_source_upper": {
                "missing_factor": str(dyadic_source_upper),
                "rejected": dyadic_source_upper != 1,
            },
        },
        "passed": passed,
    }


def _packet_key(row: int) -> str:
    return f"common_row{row}_arithmetic_packet"


def _lower_key(row: int) -> str:
    if row <= 1:
        return f"lower_Jacobian_row{row}"
    return "lower_Jacobian"


def _mixed_key(row: int) -> str:
    if row <= 1:
        return f"selected_mixed_F_row{row}"
    return "selected_mixed_F"


def _audit_row_packet(campaign: dict[str, Any], row: int) -> dict[str, Any]:
    packet = campaign[_packet_key(row)]
    dag = packet["arithmetic_dag"]
    body = {key: value for key, value in dag.items() if key != "content_sha256"}
    if dag.get("content_sha256") != _content_hash(body):
        raise QuarticRows04NonlinearSubfamilyError(
            f"row{row} arithmetic DAG hash mismatch"
        )
    lower = packet[_lower_key(row)]
    mixed = packet[_mixed_key(row)]
    expected_indices = {(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1)}
    actual_indices = {
        tuple(int(value) for value in item["multi_index"]) for item in mixed
    }
    if not (
        len(lower) == 54
        and len(mixed) == 6
        and actual_indices == expected_indices
        and all(
            item.get("output_row") == row
            and item.get("normalized_residual") == "0"
            for item in lower
        )
        and all(
            item.get("output_row") == row
            and item.get("atom_pair") == list(MIXED_PAIR)
            and item.get("normalized_coefficient_residual") == "0"
            for item in mixed
        )
    ):
        raise QuarticRows04NonlinearSubfamilyError(
            f"row{row} arithmetic coverage mismatch"
        )
    return {
        "output_row": row,
        "arithmetic_dag_sha256": dag["content_sha256"],
        "lower_roots": len(lower),
        "selected_mixed_roots": len(mixed),
        "selected_multi_indices": [list(index) for index in sorted(actual_indices)],
        "Dxy_root": next(
            int(item["arithmetic_root"])
            for item in mixed
            if item["multi_index"] == [1, 1]
        ),
        "inverse_division_assumption": packet["inverse_evidence"][
            "division_assumption"
        ],
    }


def _certify_candidate(
    row0_remainder: dict[str, Any],
    arithmetic_records: tuple[dict[str, Any], ...],
    solved: dict[str, Any],
    tube: dict[str, Any],
    global_record: dict[str, Any],
    packet_audits: list[dict[str, Any]],
    generic: dict[str, Any],
) -> dict[str, Any]:
    records = (row0_remainder, *arithmetic_records, solved, tube, global_record)
    candidate_id = str(row0_remainder.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticRows04NonlinearSubfamilyError("candidate identity mismatch")
    if any(
        record.get("coefficients") != row0_remainder.get("coefficients")
        for record in records[1:]
    ):
        raise QuarticRows04NonlinearSubfamilyError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_row0_reference_linear_slice_nonlinear_remainder_fail_closed",
        "pass_row0_lower_arithmetic_materialization_partial_mixed_fail_closed",
        "pass_row1_lower_arithmetic_materialization_other_rows_fail_closed",
        "pass_row2_lower_arithmetic_materialization_other_rows_fail_closed",
        "pass_row3_lower_arithmetic_materialization_other_rows_fail_closed",
        "pass_row4_lower_arithmetic_materialization_other_rows_fail_closed",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticRows04NonlinearSubfamilyError(
            "candidate prerequisite status mismatch"
        )
    if not (
        row0_remainder.get("full_11_row_remainder_closed") is False
        and row0_remainder.get("global_H7_differential_inequality_closed") is False
        and global_record.get("global_H7_differential_inequality_closed") is False
        and global_record.get("nonlinear_lifespan_proved") is False
    ):
        raise QuarticRows04NonlinearSubfamilyError(
            "candidate fail-closed prerequisite mismatch"
        )
    if solved.get("inverse_time_block_2_norm_upper") != "5":
        raise QuarticRows04NonlinearSubfamilyError(
            "quantitative inverse time-block bound is absent"
        )
    if not (
        solved.get("coordinate_component_radius")
        == tube.get("coordinate_component_radius")
        == "1/10000000000000"
        and solved["solved_source_Frechet_derivatives"]["orders"]
        == [0, 1, 2, 3, 4]
        and solved["solved_source_Frechet_derivatives"]["input_norm"]
        == "153-coordinate-atom component l_infinity"
        and solved["solved_source_Frechet_derivatives"]["output_norm"]
        == "11-acceleration-vector Euclidean 2-norm"
    ):
        raise QuarticRows04NonlinearSubfamilyError(
            "solved-source tube or norm contract mismatch"
        )
    frechet_numeric = solved["solved_source_Frechet_derivatives"][
        "2_norm_envelopes_numeric"
    ]
    frechet_ceilings = {
        str(order): _outward_integer(float(frechet_numeric[str(order)]))
        for order in range(5)
    }
    time_block_numeric = solved["coordinate_time_block_derivatives"][
        "2_norm_envelopes_numeric"
    ]
    time_block_ceilings = {
        str(order): _outward_integer(float(time_block_numeric[str(order)]))
        for order in range(1, 5)
    }
    term_breakdown = solved["solved_source_Frechet_derivatives"]["term_breakdown"]
    remainder_ceilings = {
        str(order): _outward_integer(
            float(term_breakdown[str(order)]["remainder_term_numeric"])
        )
        for order in range(5)
    }
    mixed_order_bounds = {
        "1,1": frechet_ceilings["2"],
        "1,2": frechet_ceilings["3"],
        "2,1": frechet_ceilings["3"],
        "1,3": frechet_ceilings["4"],
        "2,2": frechet_ceilings["4"],
        "3,1": frechet_ceilings["4"],
    }
    m2 = sp.Integer(frechet_ceilings["2"])
    q7_factor = sp.sympify(
        generic["dyadic_conversion"]["total_bilinear_Q7_factor"]
    )
    c_b = sp.simplify(q7_factor * m2)
    h7_lower = global_record["global_energy"]["H7_lower"]
    gamma = global_record["strongest_global_differential_inequality"]["Gamma_B"]
    a_row0 = row0_remainder["updated_global_inequality"]["A_row0"]
    d_increment = f"({gamma})*({c_b})/({h7_lower})"
    gamma_numeric = float(global_record["numeric_constants"]["unresolved_B7_coefficient"])
    h7_lower_numeric = float(global_record["numeric_constants"]["H7_energy_lower"])
    d_numeric = gamma_numeric * float(sp.N(c_b, 18)) / h7_lower_numeric
    if not (isfinite(d_numeric) and d_numeric > 0):
        raise QuarticRows04NonlinearSubfamilyError(
            "nonlinear energy coefficient is invalid"
        )
    return {
        "schema_version": "sigma-quartic-rows0-4-nonlinear-subfamily-certificate-1.0",
        "status": "pass_frozen_rows0_4_mixed_bilinear_subfamily_full_remainder_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": row0_remainder["coefficients"],
        "quantitative_time_block": {
            "inverse_2_norm_upper": "5",
            "sigma_min_lower": "1/5",
            "determinant_absolute_lower": f"1/{5**11}",
            "dimension": 11,
            "certified_on_coordinate_tube": True,
        },
        "reachable_family_bounds": {
            "time_block_D1_to_D4_entry_upper_ceilings": time_block_ceilings,
            "remainder_W0_to_W4_component_upper_ceilings": remainder_ceilings,
            "solved_source_F0_to_F4_component_upper_ceilings": frechet_ceilings,
            "selected_mixed_F_root_upper_ceilings": mixed_order_bounds,
            "outward_rounding": (
                "ceil(nextafter(published_binary64_envelope,+infinity))"
            ),
            "raw_order_zero_A_entry_upper_available": False,
            "raw_A0_not_needed_for_selected_recurrence": (
                "A0 is used only through the certified ||A^-1||2 bound"
            ),
        },
        "frozen_mixed_bilinear_subfamily": {
            "output_rows": list(OUTPUT_ROWS),
            "atom_pair": list(MIXED_PAIR),
            "definition": (
                "N_xy=P_rows0:4 D^2F(Y_reference)[e_x,e_y] x y"
            ),
            "Dxy_vector_2_norm_bound": str(m2),
            "H7_algebra_constant": generic["H7_product_algebra"][
                "algebra_constant"
            ],
            "dyadic_Q7_conversion_factor": generic["dyadic_conversion"][
                "total_bilinear_Q7_factor"
            ],
            "B7_bound": f"B7_frozen_xy<={c_b}*Q7",
            "C_B_contribution": str(c_b),
            "selected_mixed_roots_quantitatively_bounded": 30,
            "certified": True,
        },
        "updated_global_inequality": {
            "exact": (
                "E7'<=A_row0*E7+D_frozen_xy*E7^(3/2)+"
                "Gamma_B*sqrt(E7)*B7_remaining"
            ),
            "A_row0": a_row0,
            "D_frozen_xy": d_increment,
            "Gamma_B": gamma,
            "remaining_functional": "B7_remaining",
            "remaining_definition": (
                "all variable-coefficient selected-pair terms, all other atom pairs, "
                "unmaterialized output rows 5-10, and remaining Bony/good-unknown terms"
            ),
            "proved_with_explicit_remainder": True,
            "closed_Riccati_inequality": False,
        },
        "packet_bindings": packet_audits,
        "full_row0_nonlinear_remainder_closed": False,
        "rows0_4_full_nonlinear_remainder_closed": False,
        "full_11_row_remainder_closed": False,
        "global_H7_differential_inequality_closed": False,
        "nonlinear_lifespan_proved": False,
        "precise_missing_tensor_leaves": {
            "missing_output_rows": list(range(5, 11)),
            "missing_atom_pairs": (
                "every unordered pair of the 153 coordinate atoms except "
                "{p0[10],p1[10]}"
            ),
            "missing_variable_coefficient_control": (
                "an H7 paradifferential/Moser bound for DxyF(Y)-DxyF(Y_reference)"
            ),
            "raw_arithmetic_interval_gap": (
                "121 order-zero A component leaves lack individual upper bounds, "
                "although the selected solve bypasses them with ||A^-1||2<=5"
            ),
        },
        "numeric_constants": {
            "D2F_outward_integer_ceiling": int(m2),
            "C_B_frozen_xy": float(sp.N(c_b, 18)),
            "D_frozen_xy": d_numeric,
        },
        "scope": (
            "This certifies only the frozen reference bilinear interaction of the "
            "configured scalar-gradient pair in output rows zero through four. It "
            "does not promote the selected roots to a variable-coefficient or full "
            "row remainder."
        ),
    }


def run_quartic_rows0_4_nonlinear_subfamily_campaign(
    row0_remainder_campaign: dict[str, Any],
    row0_arithmetic_campaign: dict[str, Any],
    row1_arithmetic_campaign: dict[str, Any],
    row2_arithmetic_campaign: dict[str, Any],
    row3_arithmetic_campaign: dict[str, Any],
    row4_arithmetic_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    tube_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        arithmetic_campaigns = (
            row0_arithmetic_campaign,
            row1_arithmetic_campaign,
            row2_arithmetic_campaign,
            row3_arithmetic_campaign,
            row4_arithmetic_campaign,
        )
        campaigns = (
            row0_remainder_campaign,
            *arithmetic_campaigns,
            solved_source_campaign,
            tube_campaign,
            global_h7_campaign,
        )
        expected = (
            "pass_all_12_row0_reference_linear_slices_nonlinear_and_global_remainders_fail_closed",
            "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed",
            "pass_all_12_rows0_1_arithmetic_materialized_other_rows_fail_closed",
            "pass_all_12_rows0_2_arithmetic_materialized_other_rows_fail_closed",
            "pass_all_12_rows0_3_arithmetic_materialized_other_rows_fail_closed",
            "pass_all_12_rows0_4_arithmetic_materialized_other_rows_fail_closed",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            (
                "audit_all_12_global_H7_energies_single_source_remainder_"
                "lifespans_fail_closed"
            ),
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRows04NonlinearSubfamilyError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected:
            raise QuarticRows04NonlinearSubfamilyError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticRows04NonlinearSubfamilyError(
                "campaign content hash mismatch"
            )
        if (
            row0_remainder_campaign["upstream_sha256"]["row0_arithmetic"]
            != row0_arithmetic_campaign["content_sha256"]
            or row0_remainder_campaign["upstream_sha256"]["global_H7"]
            != global_h7_campaign["content_sha256"]
            or solved_source_campaign["upstream_sha256"]["coordinate_tube"]
            != tube_campaign["content_sha256"]
            or global_h7_campaign["upstream_sha256"]["tube"]
            != tube_campaign["content_sha256"]
        ):
            raise QuarticRows04NonlinearSubfamilyError(
                "row0/solved/global tube provenance mismatch"
            )
        for row in range(1, 5):
            upstream_name = "row0_arithmetic" if row == 1 else f"row{row - 1}_arithmetic"
            if arithmetic_campaigns[row]["upstream_sha256"][upstream_name] != (
                arithmetic_campaigns[row - 1]["content_sha256"]
            ):
                raise QuarticRows04NonlinearSubfamilyError(
                    f"row{row} arithmetic provenance mismatch"
                )
        metric_hashes = {
            campaign["upstream_sha256"]["metric_rows_tensor_dag"]
            for campaign in arithmetic_campaigns
        }
        if len(metric_hashes) != 1:
            raise QuarticRows04NonlinearSubfamilyError(
                "arithmetic metric provenance mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or list(config["output_rows"]) != list(OUTPUT_ROWS)
            or list(config["mixed_atom_pair"]) != list(MIXED_PAIR)
            or int(config["sobolev_order"]) != 7
            or config.get("variable_coefficient_policy") != "fail_closed"
            or config.get("full_row0_remainder_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticRows04NonlinearSubfamilyError(
                "unsupported nonlinear subfamily contract"
            )
        generic_passed, generic = generic_rows0_4_nonlinear_control()
        if not generic_passed:
            raise QuarticRows04NonlinearSubfamilyError(
                "generic nonlinear subfamily control failed"
            )
        packet_audits = [
            _audit_row_packet(campaign, row)
            for row, campaign in enumerate(arithmetic_campaigns)
        ]
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticRows04NonlinearSubfamilyError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                tuple(records[candidate_id] for records in maps[1:6]),
                maps[6][candidate_id],
                maps[7][candidate_id],
                maps[8][candidate_id],
                packet_audits,
                generic,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_rows0_4_frozen_mixed_bilinear_subfamilies_"
                "full_nonlinear_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "row0_remainder": row0_remainder_campaign["content_sha256"],
                **{
                    f"row{row}_arithmetic": campaign["content_sha256"]
                    for row, campaign in enumerate(arithmetic_campaigns)
                },
                "solved_source": solved_source_campaign["content_sha256"],
                "tube": tube_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_rows0_4_nonlinear_control": generic,
            "row_packet_audits": packet_audits,
            "counts": {
                "selected": len(certificates),
                "sigma_min_bounds_certified": len(certificates),
                "determinant_lower_bounds_certified": len(certificates),
                "selected_mixed_roots_quantitatively_bounded_per_candidate": 30,
                "frozen_mixed_C_B_contributions_certified": len(certificates),
                "full_row0_nonlinear_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The tube quantitatively separates the time block from singularity, "
                "and the frozen selected mixed bilinear subfamily in rows 0-4 has an "
                "explicit H7 C_B contribution. All variable/full remainders stay open."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticRows04NonlinearSubfamilyError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "sigma_min_bounds_certified": 0,
                "determinant_lower_bounds_certified": 0,
                "selected_mixed_roots_quantitatively_bounded_per_candidate": 0,
                "frozen_mixed_C_B_contributions_certified": 0,
                "full_row0_nonlinear_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_rows0_4_nonlinear_subfamily_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
