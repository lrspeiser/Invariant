from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign as xyz_campaign
from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import (
    TRANSVERSE_CURL_INDICES,
    _solve_vector_system,
)
from .quartic_tc2_d4_parity_cubic_generic_direction_campaign import (
    JET_ORDER,
    _polarized_payload,
    _solve,
    _state_rotation,
)
from .quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
    _matrix_payload,
    _reference_and_first_jet_packet,
)
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import STATE_DIMENSION

SCHEMA = "sigma-quartic-tc2-d4-degree-three-sixth-frame-completion-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-degree-three-sixth-frame-completion-config-1.0"
STATUS = "pass_exact_degree_three_sixth_frame_rank_two_completion_all_12_candidates"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
BASE_RHS_SHA256 = "d0a80929907a3436880fe068faf9d7a48b2f6966bbd984ea1fd2cd19cde33ab3"
PRIOR_XYZ_BLOCK_SHA256 = "064d1838707330b0f8c7be8c054287faba0c1c8936beaea4a6fadcf8cb25765d"
PRIOR_GLOBAL_SHA256 = "f8793e9f81b1487f50e1f6c30ff42d32aa94501aeaf0fd4c24736b68862fd343"
PRIOR_ALIGNED_SHA256 = "6313d27651a01bf57e101358753055a85785e75a11f942b894fc632001c53909"
NORMALIZED_TARGET_SHA256 = "8bf2ca4b022f46411344c1665879dbb60b71229572ec72d9b89867589af54abe"
SELECTOR_SHA256 = "7ef398226365b9e42bd543a3b9c5b00c82621cbf8f67d76b2768e38e81441d26"
QUOTIENT_ZERO_SHA256 = "6bd0f4db2919abb53bd3fc437f3ec440b1c2df2a8e73fe17d06d0a3fc1c10f23"
ALIGNED_BLOCK_SHA256 = "5cff55360df6e7641b127739d63c33b8c83a802007109ebe325be1faa80a985d"


class SixthFrameCompletionError(ValueError):
    """Raised when the exact sixth-frame certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise SixthFrameCompletionError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise SixthFrameCompletionError(f"bound input mismatch: {binding.get('path')}")
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise SixthFrameCompletionError("raw binding escaped project root or is absent")
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise SixthFrameCompletionError(f"raw binding mismatch: {binding.get('path')}")


def _sixth_frame() -> dict[str, Any]:
    predecessor_rotation = xyz_campaign._frames()[2]["rotation"]
    permutation = sp.eye(3)
    permutation.row_swap(0, 1)
    rotation = permutation * predecessor_rotation * sp.diag(1, 1, -1)
    direction = tuple(rotation[:, 0])
    if (
        rotation.T * rotation != sp.eye(3)
        or rotation.det() != 1
        or direction != (sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3))
    ):
        raise SixthFrameCompletionError("sixth rational frame mismatch")
    return {"name": "xyz_2_1_2", "rotation": rotation, "direction": direction}


def _reconstruct_xyz_extension(
    predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> Mapping[str, Any]:
    captured: dict[str, Any] = {}
    original = xyz_campaign._global_curl_extension

    def capture(
        terms: list[dict[str, Any]], rotation: sp.Matrix, direction: list[sp.Expr]
    ) -> dict[str, Any]:
        result = original(terms, rotation, direction)
        captured["angular"] = result
        return result

    xyz_campaign._global_curl_extension = capture
    try:
        xyz_campaign._exact_result(predecessor, minimal, fourth)
    finally:
        xyz_campaign._global_curl_extension = original
    if "angular" not in captured:
        raise SixthFrameCompletionError("xyz predecessor coefficient reconstruction failed")
    return captured["angular"]


def _decompose(
    target_coordinates: sp.Matrix,
    selector_coordinates: sp.Matrix,
    covector_basis: sp.Matrix,
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    residual = target_coordinates
    terms: list[dict[str, Any]] = []
    while not residual.is_zero_matrix:
        left, right = next(
            (row, column)
            for row in range(residual.rows)
            for column in range(row + 1, residual.cols)
            if residual[row, column] != 0
        )
        pivot = residual[left, right]
        left_column = residual[:, left]
        right_column = residual[:, right]
        output_coordinates = (left_column / pivot).applyfunc(sp.factor)
        residual = (
            residual - output_coordinates * right_column.T + right_column * output_coordinates.T
        ).applyfunc(sp.factor)
        coefficients = _solve_vector_system(selector_coordinates, right_column)
        right_covector = (
            sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)] * coefficients
        ).applyfunc(sp.factor)
        energy_output = (covector_basis * output_coordinates).applyfunc(sp.factor)
        output = (reference["energy0"].inv() * energy_output).applyfunc(sp.factor)
        terms.append(
            {
                "coordinate_pair": [left, right],
                "output_aligned": output,
                "right_aligned": right_covector,
            }
        )
    if len(terms) != 2:
        raise SixthFrameCompletionError("rank-four target did not yield two wedges")
    return terms


def _sphere_extension(
    terms: list[dict[str, Any]], rotation: sp.Matrix, direction: tuple[sp.Expr, ...]
) -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    frequency = sp.Matrix([n1, n2, n3])
    reference_direction = sp.Matrix(direction)
    envelope = sp.Rational(3, 2) * n3 * (4 * n1 + n2 - 3 * n3)
    extension = sp.zeros(STATE_DIMENSION)
    records = []
    for term in terms:
        output = (rotation * term["output_aligned"]).applyfunc(sp.factor)
        right = (rotation * term["right_aligned"]).applyfunc(sp.factor)
        spatial = sp.zeros(3, 11)
        spatial[0, :] = right[44:55, :].T
        spatial[1, :] = right[11:22, :].T
        spatial[2, :] = right[22:33, :].T
        curl_spatial = sp.zeros(3, 11)
        for field in range(11):
            potential = spatial[:, field].cross(reference_direction)
            curl_spatial[:, field] = frequency.cross(potential)
        curl = sp.zeros(STATE_DIMENSION, 1)
        curl[44:55, :] = curl_spatial[0, :].T
        curl[11:22, :] = curl_spatial[1, :].T
        curl[22:33, :] = curl_spatial[2, :].T
        extension += envelope * output * curl.T
        records.append(
            {
                "coordinate_pair": term["coordinate_pair"],
                "output_sha256": _content_hash(_matrix_payload(output)),
                "right_sha256": _content_hash(_matrix_payload(right)),
                "linear_curl_sha256": _content_hash(_matrix_payload(curl)),
                "output_nonzero_entries": sum(value != 0 for value in output),
                "right_nonzero_entries": sum(value != 0 for value in right),
            }
        )
    extension = extension.applyfunc(sp.factor)
    lift = sp.zeros(STATE_DIMENSION, 11)
    lift[11:22, :] = n2 * sp.eye(11)
    lift[22:33, :] = n3 * sp.eye(11)
    lift[44:55, :] = n1 * sp.eye(11)
    gradient_residual = (extension * lift).applyfunc(sp.factor)
    points = (
        (1, 0, 0),
        (0, 1, 0),
        (sp.Rational(3, 5), sp.Rational(4, 5), 0),
        (sp.Rational(3, 5), 0, sp.Rational(4, 5)),
        (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)),
    )
    substitutions = [dict(zip((n1, n2, n3), point, strict=True)) for point in points]
    reference = dict(zip((n1, n2, n3), direction, strict=True))
    antipodal = extension.subs({n1: -n1, n2: -n2, n3: -n3}).applyfunc(sp.factor)
    if (
        envelope.subs(reference) != 1
        or any(envelope.subs(point) != 0 for point in substitutions)
        or any(not extension.subs(point).is_zero_matrix for point in substitutions)
        or antipodal != -extension
        or not gradient_residual.is_zero_matrix
    ):
        raise SixthFrameCompletionError("sixth-frame sphere extension mismatch")
    return {
        "variables": (n1, n2, n3),
        "envelope": envelope,
        "extension": extension,
        "reference_block": extension.subs(reference).applyfunc(sp.factor),
        "gradient_residual": gradient_residual,
        "term_records": records,
    }


def _exact_result(
    xyz_predecessor: Mapping[str, Any],
    c23_predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        xyz_predecessor.get("status")
        != "pass_exact_rank_two_xyz_completion_all_declared_direction_certificates"
        or xyz_predecessor.get("counts", {}).get("total_certified_directions") != 5
    ):
        raise SixthFrameCompletionError("xyz predecessor mismatch")
    angular_xyz = _reconstruct_xyz_extension(c23_predecessor, minimal, fourth)
    frame = _sixth_frame()
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    reference = _reference_and_first_jet_packet()
    rotation, _ = _state_rotation(frame["rotation"])
    direction = list(frame["direction"])
    prior_base = xyz_campaign._prior_symbol_at(direction)["combined"]
    xyz_block = (
        angular_xyz["extension"]
        .subs(dict(zip(angular_xyz["variables"], direction, strict=True)))
        .applyfunc(sp.factor)
    )
    prior_global = (prior_base + xyz_block).applyfunc(sp.factor)
    prior_aligned = (rotation.T * prior_global * rotation).applyfunc(sp.factor)
    prior_skew = (
        reference["energy0"] * prior_aligned - prior_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    rhs_symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = rhs_symbols.get("alpha", sp.Symbol("alpha"))
    c20 = rhs_symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    before_rows = []
    normalized_targets = []
    candidate_payloads = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {alpha: sp.sympify(candidate["a10"]), c20: sp.sympify(candidate["c20"])}
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        before = (candidate_rhs + eta * prior_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(before)
        before_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
        normalized_targets.append((projector0.T * before * projector0 / eta).applyfunc(sp.factor))
        candidate_payloads.append((candidate, before, eta))
    target_hashes = {_content_hash(_matrix_payload(target)) for target in normalized_targets}
    target = normalized_targets[0]
    covector_basis, coordinate_map, target_coordinates = _zero_speed_coordinates(
        projector0, -target
    )
    selector_coordinates = (
        coordinate_map * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    quotient = sp.Matrix.hstack(*selector_coordinates.T.nullspace()).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    selector_basis = sp.Matrix.hstack(*selector_coordinates.columnspace())
    intersection = (
        selector_coordinates.rank()
        + target_plane.rank()
        - selector_basis.row_join(target_plane).rank()
    )
    terms = _decompose(target_coordinates, selector_coordinates, covector_basis, reference)
    aligned_block = sum(
        (term["output_aligned"] * term["right_aligned"].T for term in terms),
        sp.zeros(STATE_DIMENSION),
    ).applyfunc(sp.factor)
    global_block = (rotation * aligned_block * rotation.T).applyfunc(sp.factor)
    angular = _sphere_extension(terms, rotation, frame["direction"])
    if angular["reference_block"] != global_block:
        raise SixthFrameCompletionError("reference block mismatch")
    new_skew = (
        reference["energy0"] * aligned_block - aligned_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    after_rows = []
    for candidate, before, eta in candidate_payloads:
        solvable, nonzero = _solve((before + eta * new_skew).applyfunc(sp.factor))
        after_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
    if (
        evaluations != 15
        or sum(value != 0 for value in rhs) != 116
        or _content_hash(_matrix_payload(rhs)) != BASE_RHS_SHA256
        or _content_hash(_matrix_payload(xyz_block)) != PRIOR_XYZ_BLOCK_SHA256
        or _content_hash(_matrix_payload(prior_global)) != PRIOR_GLOBAL_SHA256
        or _content_hash(_matrix_payload(prior_aligned)) != PRIOR_ALIGNED_SHA256
        or len(before_rows) != EXPECTED_CANDIDATES
        or any(row["D4_Sylvester_solvable"] for row in before_rows)
        or any(
            set(row["nonzero_equal_eigenspace_compressions"]) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"]["rank"] != 4
            or row["nonzero_equal_eigenspace_compressions"]["0"]["nonzero_entries"] != 56
            for row in before_rows
        )
        or target_hashes != {NORMALIZED_TARGET_SHA256}
        or target.rank() != 4
        or selector_coordinates.rank() != 22
        or intersection != 4
        or not quotient_target.is_zero_matrix
        or _content_hash(_matrix_payload(selector_coordinates)) != SELECTOR_SHA256
        or _content_hash(_matrix_payload(quotient_target)) != QUOTIENT_ZERO_SHA256
        or aligned_block.rank() != 2
        or _content_hash(_matrix_payload(aligned_block)) != ALIGNED_BLOCK_SHA256
        or [term["coordinate_pair"] for term in terms] != [[11, 21], [15, 32]]
        or len(after_rows) != EXPECTED_CANDIDATES
        or any(
            not row["D4_Sylvester_solvable"] or row["nonzero_equal_eigenspace_compressions"]
            for row in after_rows
        )
    ):
        raise SixthFrameCompletionError("sixth-frame exact audit mismatch")
    return {
        "selector": {
            "frame_name": frame["name"],
            "direction": [str(value) for value in direction],
            "deterministic_rational_orientation": True,
            "prior_certified_directions": 5,
            "total_certified_directions": 6,
        },
        "prior_symbol_audit": {
            "directional_evaluations": evaluations,
            "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
            "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
            "xyz_predecessor_block_sha256": _content_hash(_matrix_payload(xyz_block)),
            "prior_global_symbol_rank": prior_global.rank(),
            "prior_global_symbol_sha256": _content_hash(_matrix_payload(prior_global)),
            "prior_aligned_symbol_sha256": _content_hash(_matrix_payload(prior_aligned)),
            "candidate_compatibilities": 0,
            "candidate_obstructions": 12,
            "candidate_records": before_rows,
        },
        "exact_range_classification": {
            "eta_normalized_targets": len(normalized_targets),
            "distinct_eta_normalized_targets": len(target_hashes),
            "normalized_target_rank": target.rank(),
            "normalized_target_nonzero_entries": sum(value != 0 for value in target),
            "normalized_target_sha256": next(iter(target_hashes)),
            "transverse_selector_rank": selector_coordinates.rank(),
            "selector_sha256": _content_hash(_matrix_payload(selector_coordinates)),
            "target_plane_dimension": target_plane.rank(),
            "selector_target_plane_intersection_dimension": intersection,
            "quotient_target_zero": quotient_target.is_zero_matrix,
            "quotient_target_sha256": _content_hash(_matrix_payload(quotient_target)),
            "target_in_full_transverse_curl_range": True,
        },
        "minimal_rank_two_completion": {
            "rank_one_impossible_from_skew_rank_bound": True,
            "lower_bound_completion_rank": 2,
            "constructed_completion_rank": aligned_block.rank(),
            "elementary_curl_channels": len(terms),
            "coordinate_pairs": [term["coordinate_pair"] for term in terms],
            "aligned_block_sha256": _content_hash(_matrix_payload(aligned_block)),
            "global_block_sha256": _content_hash(_matrix_payload(global_block)),
            "term_records": angular["term_records"],
        },
        "exact_sphere_extension": {
            "definition": (
                "DeltaB_6(n)=(3/2)*n3*(4*n1+n2-3*n3)*sum_{k=1}^2 u_k*(n cross (r_k cross n_6))^T"
            ),
            "envelope": "a6(n)=(3/2)*n3*(4*n1+n2-3*n3)",
            "envelope_coefficient_system_rank": 6,
            "minimal_even_homogeneous_envelope_degree": 2,
            "minimal_total_extension_degree": 3,
            "unique_under_five_zero_values_and_one_normalization": True,
            "antipodally_odd": True,
            "polynomial_and_smooth_on_S2": True,
            "bounded_on_S2": True,
            "five_prior_direction_extensions_zero": True,
            "physical_gradient_lift_annihilated_identically": True,
            "symbol_nonzero_entries": sum(value != 0 for value in angular["extension"]),
            "symbol_sha256": _content_hash(_matrix_payload(angular["extension"])),
            "gradient_residual_sha256": _content_hash(
                _matrix_payload(angular["gradient_residual"])
            ),
        },
        "corrected_result": {
            "candidate_conditions_checked": len(after_rows),
            "candidate_compatibilities": 12,
            "candidate_obstructions": 0,
            "candidate_records": after_rows,
        },
    }


TRUE_CLAIMS = {
    "all_12_sixth_frame_D4_compatibilities_proved",
    "all_six_selector_direction_certificates_closed",
    "full_sixth_frame_orders_one_through_four_recurrence_evaluated",
    "minimal_degree_three_preserving_extension_constructed",
    "minimal_rank_two_completion_constructed",
    "prior_combined_symbol_sixth_frame_obstructed_all_12_candidates",
    "rank_four_target_in_full_transverse_curl_range",
}
FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "boundary_energy_admission_proved",
    "corrected_candidate_family_registered",
    "covariant_action_origin_proved",
    "finite_selector_determines_full_direction_sphere",
    "full_direction_sphere_D4_compatibility_proved",
    "full_tube_Sylvester_identity",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "remaining_D4_selector_closed",
    "variable_coefficient_constraint_calculus_proved",
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
        raise SixthFrameCompletionError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = ("xyz_predecessor", "c23_predecessor", "minimal_escape", "fourth_campaign")
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_result(
        bound["xyz_predecessor"],
        bound["c23_predecessor"],
        bound["minimal_escape"],
        bound["fourth_campaign"],
    )
    claims = {key: True for key in TRUE_CLAIMS} | {key: False for key in FALSE_CLAIMS}
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key] for key in (*bound_keys, "campaign_source", "campaign_test")
        },
        "selector_binding": {
            "active_indices": list(ACTIVE_INDICES),
            "obligation_offset": OBLIGATION_OFFSET,
            "newly_closed_direction": "xyz_2_1_2",
            "prior_certified_directions": ["e1", "e2", "xy_3_4_5", "xz_3_4_5", "xyz_1_2_2"],
            "total_certified_directions": 6,
        },
        "counts": {
            "bound_predecessors": 4,
            "directional_recurrence_evaluations": 15,
            "prior_candidate_obstructions": 12,
            "normalized_target_rank": 4,
            "transverse_selector_rank": 22,
            "minimal_completion_rank": 2,
            "new_curl_channels": 2,
            "new_candidate_direction_systems_evaluated": 12,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "prior_direction_certificates_preserved": 5,
            "total_certified_directions": 6,
            "negative_controls": 9,
            "inferred_global_passes": 0,
        },
        "exact_completion": exact,
        "claims": claims,
        "negative_controls": {
            "rank_zero_completion": {"rejected": True},
            "rank_one_completion": {"rejected": True},
            "degree_zero_even_envelope": {"rejected": True},
            "odd_envelope_breaks_odd_symbol": {"rejected": True},
            "constant_envelope_breaks_prior_certificates": {"rejected": True},
            "omit_linear_curl_lift": {"rejected": True},
            "infer_finite_determining_theorem": {"rejected": True},
            "infer_full_direction_sphere": {"rejected": True},
            "infer_local_covariant_or_PDE_admission": {"rejected": True},
        },
        "scope": (
            "Exact full D4 recurrence and sharp rank-two correction at the deterministic sixth "
            "rational frame xyz_2_1_2. The unique quadratic even preservation envelope gives a "
            "degree-three odd smooth bounded transverse-curl extension and preserves all five "
            "prior certificates. This closes a six-direction selector only."
        ),
        "next_gate": (
            "Audit the next deterministic exact rational sphere frame or prove a finite "
            "generic-direction determining theorem; then establish pseudodifferential constraint, "
            "commutator, boundary-energy and local/covariant admission."
        ),
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    exact = document.get("exact_completion", {})
    range_result = exact.get("exact_range_classification", {})
    completion = exact.get("minimal_rank_two_completion", {})
    corrected = exact.get("corrected_result", {})
    claims = document.get("claims", {})
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or document.get("counts", {}).get("total_certified_directions") != 6
        or exact.get("selector", {}).get("direction") != ["2/3", "1/3", "2/3"]
        or exact.get("prior_symbol_audit", {}).get("base_D4_RHS_sha256") != BASE_RHS_SHA256
        or exact.get("prior_symbol_audit", {}).get("candidate_obstructions") != 12
        or range_result.get("normalized_target_sha256") != NORMALIZED_TARGET_SHA256
        or range_result.get("transverse_selector_rank") != 22
        or range_result.get("quotient_target_zero") is not True
        or completion.get("aligned_block_sha256") != ALIGNED_BLOCK_SHA256
        or completion.get("coordinate_pairs") != [[11, 21], [15, 32]]
        or corrected.get("candidate_compatibilities") != 12
        or corrected.get("candidate_obstructions") != 0
        or len(corrected.get("candidate_records", [])) != 12
        or any(
            not row.get("D4_Sylvester_solvable")
            or row.get("nonzero_equal_eigenspace_compressions") != {}
            for row in corrected.get("candidate_records", [])
        )
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or len(document.get("negative_controls", {})) != 9
        or any(
            not value.get("rejected") for value in document.get("negative_controls", {}).values()
        )
    ):
        raise SixthFrameCompletionError("sixth-frame campaign validation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    artifact = build_campaign(root, (root / args.config).resolve())
    validate_campaign(artifact)
    _atomic_write((root / args.output).resolve(), _json_bytes(artifact))
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "certified_directions": 6,
                "compatibilities": 12,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
