from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _content_hash_matches,
)
from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    ACTIVE_DIRECTION_COUNT,
    TOTAL_MIXED_TRIPLES,
    _mixed_selector,
    _record_hash_matches,
    _triple_kind,
)

SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-basis-reduction-campaign-1.0"
SYMMETRIC_CUBIC_DIMENSION = 680
STABLE_PREFIX_COUNT = 576


class QuarticTC2MixedThirdJetBasisReductionError(ValueError):
    """Raised when the exact active-direction reduction is not fully bound."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise QuarticTC2MixedThirdJetBasisReductionError(
            "bound artifact escapes repository root"
        ) from error
    if not path.is_file() or _sha256_file(path) != binding["file_sha256"]:
        raise QuarticTC2MixedThirdJetBasisReductionError("bound artifact file hash mismatch")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        not _content_hash_matches(artifact)
        or artifact["content_sha256"] != binding["content_sha256"]
    ):
        raise QuarticTC2MixedThirdJetBasisReductionError("bound artifact content hash mismatch")
    return artifact


def _all_global_claims_false(ledger: dict[str, Any]) -> bool:
    return all(
        ledger.get(key) is False
        for key in (
            "all_12_300_mixed_third_jets_closed",
            "full_tube_Sylvester_identity",
            "CK1_closed",
            "CK3_closed",
            "TC2_closed",
            "B7_closed",
            "global_H7_closed",
            "lifespan_proved",
        )
    )


def _validate_stable_evidence(
    diagonal: dict[str, Any], chunks: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[tuple[int, int, int]], str]:
    expected_diagonal_status = (
        "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_"
        "mixed_triples_full_tube_global_H7_fail_closed"
    )
    if (
        diagonal.get("status") != expected_diagonal_status
        or diagonal.get("counts", {}).get("diagonal_direction_packets") != 41
        or diagonal.get("counts", {}).get("symbolic_parameter_diagonal_third_jet_passes")
        != 41
        or diagonal.get("counts", {}).get("candidate_direction_evaluations") != 492
        or diagonal.get("counts", {}).get("candidate_direction_solvable") != 492
        or diagonal.get("counts", {}).get("candidate_direction_obstructed") != 0
        or len(diagonal.get("direction_records", [])) != ACTIVE_DIRECTION_COUNT
    ):
        raise QuarticTC2MixedThirdJetBasisReductionError("diagonal evidence mismatch")
    if any(
        record.get("symbolic_equal_eigenspace_compressions_zero") is not True
        or len(record.get("candidate_results", [])) != 12
        or any(candidate.get("solvable") is not True for candidate in record["candidate_results"])
        for record in diagonal["direction_records"]
    ):
        raise QuarticTC2MixedThirdJetBasisReductionError("diagonal pass record mismatch")

    selector = _mixed_selector()
    expected_offset = 0
    prior_tip: str | None = None
    stable_triples: list[tuple[int, int, int]] = []
    for chunk in chunks:
        contract = chunk.get("chunk_contract", {})
        manifest = chunk.get("triple_manifest", [])
        counts = chunk.get("counts", {})
        if (
            int(contract.get("chunk_offset", -1)) != expected_offset
            or int(contract.get("requested_chunk_size", -1)) != 64
            or int(contract.get("processed_count", -1)) != 64
            or int(contract.get("next_offset", -1)) != expected_offset + 64
            or contract.get("prior_resume_sha256") != prior_tip
            or contract.get("stopped_early") is not False
            or len(manifest) != 64
            or chunk.get("first_exact_obstruction") is not None
            or counts.get("selected") != 64
            or counts.get("symbolic_parameter_compatible") != 64
            or counts.get("candidate_evaluations") != 768
            or counts.get("candidate_solvable") != 768
            or counts.get("candidate_obstructed") != 0
            or not _all_global_claims_false(chunk.get("closure_ledger", {}))
        ):
            raise QuarticTC2MixedThirdJetBasisReductionError("stable chunk contract mismatch")
        previous = contract.get("resume_seed_sha256")
        for local_index, record in enumerate(manifest):
            selector_index = expected_offset + local_index
            triple = tuple(int(value) for value in record.get("active_position_triple", []))
            if (
                record.get("selector_index") != selector_index
                or triple != selector[selector_index]
                or record.get("previous_record_sha256") != previous
                or not _record_hash_matches(record)
                or record.get("symbolic_parameter_compatible") is not True
                or record.get("obstructed_candidate_ids") != []
                or len(record.get("candidate_results", [])) != 12
                or any(
                    candidate.get("solvable") is not True
                    for candidate in record["candidate_results"]
                )
            ):
                raise QuarticTC2MixedThirdJetBasisReductionError(
                    "stable chunk record-chain mismatch"
                )
            previous = record["record_sha256"]
            stable_triples.append(triple)
        if previous != contract.get("resume_tip_sha256"):
            raise QuarticTC2MixedThirdJetBasisReductionError("stable chunk resume tip mismatch")
        prior_tip = previous
        expected_offset += 64
    if (
        expected_offset != int(config["stable_predecessor_prefix_count"])
        or expected_offset != STABLE_PREFIX_COUNT
        or prior_tip != config["stable_predecessor_resume_tip_sha256"]
    ):
        raise QuarticTC2MixedThirdJetBasisReductionError("stable predecessor boundary mismatch")
    return stable_triples, str(prior_tip)


@cache
def _exact_reduction_packet() -> dict[str, Any]:
    directions = _active_directions()
    coordinate_names = tuple(
        sorted({name for direction in directions for name in direction["direction"]})
    )
    direction_matrix = sp.Matrix(
        [
            [direction["direction"].get(name, sp.S.Zero) for direction in directions]
            for name in coordinate_names
        ]
    )
    _, pivot_positions = direction_matrix.rref()
    rank = len(pivot_positions)
    basis_matrix = direction_matrix[:, pivot_positions]
    _, independent_rows = basis_matrix.T.rref()
    square_minor = basis_matrix.extract(independent_rows, range(rank))
    coordinates = square_minor.inv() * direction_matrix.extract(
        independent_rows, range(ACTIVE_DIRECTION_COUNT)
    )
    if not (basis_matrix * coordinates - direction_matrix).is_zero_matrix:
        raise QuarticTC2MixedThirdJetBasisReductionError("active-direction span residual")

    cubic_basis = tuple(itertools.combinations_with_replacement(range(rank), 3))
    cubic_index = {triple: index for index, triple in enumerate(cubic_basis)}
    if len(cubic_basis) != SYMMETRIC_CUBIC_DIMENSION:
        raise QuarticTC2MixedThirdJetBasisReductionError("symmetric-cubic dimension mismatch")

    def functional_row(triple: tuple[int, int, int]) -> list[sp.Expr]:
        row = [sp.S.Zero] * len(cubic_basis)
        for left in range(rank):
            left_coefficient = coordinates[left, triple[0]]
            if left_coefficient == 0:
                continue
            for middle in range(rank):
                middle_coefficient = coordinates[middle, triple[1]]
                if middle_coefficient == 0:
                    continue
                for right in range(rank):
                    right_coefficient = coordinates[right, triple[2]]
                    if right_coefficient == 0:
                        continue
                    key = tuple(sorted((left, middle, right)))
                    index = cubic_index[key]
                    row[index] = sp.expand(
                        row[index]
                        + left_coefficient * middle_coefficient * right_coefficient
                    )
        return row

    diagonal_triples = [(index, index, index) for index in range(ACTIVE_DIRECTION_COUNT)]
    stable_mixed_triples = list(_mixed_selector()[:STABLE_PREFIX_COUNT])
    diagonal_matrix = sp.Matrix([functional_row(triple) for triple in diagonal_triples])
    prefix_matrix = sp.Matrix([functional_row(triple) for triple in stable_mixed_triples])
    evidence_matrix = diagonal_matrix.col_join(prefix_matrix)
    _, evidence_pivot_coordinates = evidence_matrix.rref()
    complement_coordinates = [
        index
        for index in range(len(cubic_basis))
        if index not in evidence_pivot_coordinates
    ]
    obligation_basis_triples = [cubic_basis[index] for index in complement_coordinates]
    obligation_active_triples = [
        tuple(pivot_positions[index] for index in triple)
        for triple in obligation_basis_triples
    ]
    completion = evidence_matrix.col_join(
        sp.Matrix(
            [
                [sp.S.One if column == index else sp.S.Zero for column in range(len(cubic_basis))]
                for index in complement_coordinates
            ]
        )
    )
    if completion.rank() != len(cubic_basis):
        raise QuarticTC2MixedThirdJetBasisReductionError("reduced selector is incomplete")
    if len(complement_coordinates) > 1 and completion[:-1, :].rank() != len(cubic_basis) - 1:
        raise QuarticTC2MixedThirdJetBasisReductionError("reduced selector minimality mismatch")

    support_counts: Counter[int] = Counter()
    direction_coordinates: list[dict[str, Any]] = []
    for active_position, direction in enumerate(directions):
        support = [
            {
                "basis_active_position": int(pivot_positions[row]),
                "coefficient": str(sp.factor(coordinates[row, active_position])),
            }
            for row in range(rank)
            if coordinates[row, active_position] != 0
        ]
        support_counts[len(support)] += 1
        direction_coordinates.append(
            {
                "active_position": active_position,
                "atom_index": int(direction["atom_index"]),
                "atom": str(direction["atom"]),
                "basis_support": support,
            }
        )

    selector_index = {triple: index for index, triple in enumerate(_mixed_selector())}
    obligations: list[dict[str, Any]] = []
    previous = _content_hash(
        {
            "basis_active_positions": list(pivot_positions),
            "stable_prefix_count": STABLE_PREFIX_COUNT,
            "evidence_functional_rank": len(evidence_pivot_coordinates),
        }
    )
    for obligation_index, triple in enumerate(obligation_active_triples):
        body = {
            "obligation_index": obligation_index,
            "global_selector_index": selector_index[triple],
            "triple_kind": _triple_kind(triple),
            "active_position_triple": list(triple),
            "atom_index_triple": [int(directions[index]["atom_index"]) for index in triple],
            "atom_triple": [str(directions[index]["atom"]) for index in triple],
            "previous_obligation_sha256": previous,
        }
        record = {**body, "obligation_sha256": _content_hash(body)}
        obligations.append(record)
        previous = record["obligation_sha256"]

    return {
        "coordinate_names": list(coordinate_names),
        "ambient_coordinate_count": len(coordinate_names),
        "active_direction_count": ACTIVE_DIRECTION_COUNT,
        "active_direction_rank": rank,
        "basis_active_positions": list(pivot_positions),
        "basis_atom_indices": [int(directions[index]["atom_index"]) for index in pivot_positions],
        "basis_atoms": [str(directions[index]["atom"]) for index in pivot_positions],
        "span_residual_zero": True,
        "direction_coordinate_support_counts": {
            str(key): value for key, value in sorted(support_counts.items())
        },
        "direction_coordinates": direction_coordinates,
        "symmetric_cubic_dimension": len(cubic_basis),
        "diagonal_evidence_functional_rank": diagonal_matrix.rank(),
        "stable_prefix_functional_rank": prefix_matrix.rank(),
        "combined_evidence_functional_rank": len(evidence_pivot_coordinates),
        "reduced_obligation_count": len(obligations),
        "reduced_obligation_kind_counts": dict(
            sorted(Counter(record["triple_kind"] for record in obligations).items())
        ),
        "reduced_obligation_first_selector_index": obligations[0]["global_selector_index"],
        "reduced_obligation_last_selector_index": obligations[-1]["global_selector_index"],
        "reduced_obligation_seed_sha256": obligations[0]["previous_obligation_sha256"],
        "reduced_obligation_tip_sha256": previous,
        "reduced_obligations": obligations,
        "completion_rank": completion.rank(),
        "drop_final_obligation_rank": completion[:-1, :].rank(),
    }


def run_quartic_tc2_mixed_third_jet_basis_reduction_campaign(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2MixedThirdJetBasisReductionError("unsupported schema_version")
        if any(
            config.get(key) != "fail_closed"
            for key in (
                "unevaluated_reduced_obligation_policy",
                "unprocessed_mixed_third_jet_policy",
                "full_tube_policy",
                "CK1_policy",
                "CK3_policy",
                "TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        ):
            raise QuarticTC2MixedThirdJetBasisReductionError("closure policy mismatch")
        if config.get("reduction_theorem") != (
            "symmetric_trilinear_tensor_determined_by_exact_basis_coordinate_functionals"
        ):
            raise QuarticTC2MixedThirdJetBasisReductionError("reduction theorem mismatch")
        diagonal = _load_bound_json(root, config["diagonal_evidence"])
        chunks = [_load_bound_json(root, item) for item in config["stable_chunk_evidence"]]
        stable_triples, stable_tip = _validate_stable_evidence(diagonal, chunks, config)
        packet = _exact_reduction_packet()
        expected = config["expected_reduction"]
        actual = {
            "ambient_coordinate_count": packet["ambient_coordinate_count"],
            "active_direction_rank": packet["active_direction_rank"],
            "basis_active_positions": packet["basis_active_positions"],
            "symmetric_cubic_dimension": packet["symmetric_cubic_dimension"],
            "combined_evidence_functional_rank": packet["combined_evidence_functional_rank"],
            "reduced_obligation_count": packet["reduced_obligation_count"],
        }
        if actual != expected:
            raise QuarticTC2MixedThirdJetBasisReductionError("configured reduction mismatch")
        if stable_triples != list(_mixed_selector()[:STABLE_PREFIX_COUNT]):
            raise QuarticTC2MixedThirdJetBasisReductionError("stable selector prefix mismatch")
        obligations = packet["reduced_obligations"]
        if any(record["global_selector_index"] < STABLE_PREFIX_COUNT for record in obligations):
            raise QuarticTC2MixedThirdJetBasisReductionError(
                "reduced obligation overlaps stable prefix"
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_exact_15_direction_basis_reduction_560_obligations_"
                "no_inferred_passes_global_closure_fail_closed"
            ),
            "errors": [],
            "config_sha256": _content_hash(config),
            "stable_evidence": {
                "diagonal_content_sha256": diagonal["content_sha256"],
                "chunk_content_sha256": [chunk["content_sha256"] for chunk in chunks],
                "diagonal_records": 41,
                "mixed_prefix_records": len(stable_triples),
                "candidate_evaluations": 41 * 12 + len(stable_triples) * 12,
                "candidate_obstructions": 0,
                "stable_prefix_resume_tip_sha256": stable_tip,
            },
            "exact_active_direction_reduction": {
                key: value
                for key, value in packet.items()
                if key not in {"reduced_obligations"}
            },
            "reduced_obligation_selector": {
                "selector": "basis_coordinate_complement_of_stable_evidence_row_space",
                "global_mixed_selector": (
                    "lexicographic_active_direction_multisets_excluding_AAA"
                ),
                "stable_remaining_mixed_triples": TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT,
                "exact_obligations": packet["reduced_obligation_count"],
                "candidate_evaluations_if_all_obligations_are_run": (
                    packet["reduced_obligation_count"] * 12
                ),
                "unevaluated_obligations_counted_as_passes": 0,
                "remaining_active_triples_counted_as_inferred_passes": 0,
                "obligations": obligations,
            },
            "theorem": {
                "statement": (
                    "For every vector-valued symmetric trilinear map T on the exact span "
                    "of the 41 active directions, the 120 stable evaluation functionals "
                    "together with the 560 listed basis-coordinate functionals form a basis "
                    "of Sym^3(V)^*. Hence evaluating exactly those 560 obligations determines "
                    "all active-sector cubic values. Equal-eigenspace compression, coefficient "
                    "specialization, and the off-eigenspace Sylvester inverse are linear in the "
                    "third-order RHS, so the same reduction applies to D3P55, D3K55, D3TC2, "
                    "compatibility, and deltaK_ABC after every obligation is actually proved."
                ),
                "exact_field": "Q(sqrt(2))",
                "direction_span_residual_zero": packet["span_residual_zero"],
                "completion_rank": packet["completion_rank"],
                "drop_final_obligation_rank": packet["drop_final_obligation_rank"],
                "minimal_complement": (
                    packet["completion_rank"] - packet["combined_evidence_functional_rank"]
                    == packet["reduced_obligation_count"]
                ),
            },
            "negative_controls": {
                "remove_one_reduced_obligation": {
                    "resulting_rank": packet["drop_final_obligation_rank"],
                    "full_rank": packet["symmetric_cubic_dimension"],
                    "rejected": packet["drop_final_obligation_rank"]
                    != packet["symmetric_cubic_dimension"],
                },
                "count_unevaluated_obligations_as_passes": {
                    "inferred_pass_count": 0,
                    "rejected": True,
                },
                "promote_finite_third_jet_to_full_tube": {
                    "missing": "fourth-and-higher residual jets or a nonlinear range theorem",
                    "rejected": True,
                },
            },
            "counts": {
                "stable_mixed_triples_evaluated": STABLE_PREFIX_COUNT,
                "stable_mixed_triples_remaining": TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT,
                "reduced_exact_obligations": packet["reduced_obligation_count"],
                "reduced_obligations_evaluated": 0,
                "reduced_obligations_passed": 0,
                "remaining_active_triples_inferred_passed": 0,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "closure_ledger": {
                "basis_reduction_theorem_proved": True,
                "all_560_reduced_obligations_closed": False,
                "all_12_300_mixed_third_jets_closed": False,
                "full_tube_Sylvester_identity": False,
                "CK1_closed": False,
                "CK3_closed": False,
                "TC2_closed": False,
                "B7_closed": False,
                "global_H7_closed": False,
                "lifespan_proved": False,
            },
            "claim": (
                "Exact active-direction rank and symmetric-cubic row-space reduction replace "
                "11,724 remaining brute-force mixed evaluations by 560 explicit obligations."
            ),
            "scope": (
                "This artifact proves only the reduction theorem and selector. None of the 560 "
                "obligations is evaluated or counted as passed here; mixed closure, full tube, "
                "CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "counts": {
                "stable_mixed_triples_evaluated": 0,
                "stable_mixed_triples_remaining": TOTAL_MIXED_TRIPLES,
                "reduced_exact_obligations": 0,
                "reduced_obligations_evaluated": 0,
                "reduced_obligations_passed": 0,
                "remaining_active_triples_inferred_passed": 0,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_mixed_third_jet_basis_reduction_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
