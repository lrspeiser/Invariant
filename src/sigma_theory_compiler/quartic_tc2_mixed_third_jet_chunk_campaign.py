from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _content_hash_matches,
    _directional_taylor_packet,
    _directional_taylor_packet_cached,
    _matrix_payload,
    generic_diagonal_third_jet_control,
)
from .quartic_tc2_quadratic_deltak_extension_campaign import _collect_records
from .quartic_tc2_variable_sylvester_campaign import (
    STATE_DIMENSION,
    _reference_and_first_jet_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-chunk-campaign-1.0"
DEFAULT_CHUNK_SIZE = 64
ACTIVE_DIRECTION_COUNT = 41
TOTAL_ACTIVE_SYMMETRIC_TRIPLES = 12_341
TOTAL_MIXED_TRIPLES = 12_300


class QuarticTC2MixedThirdJetChunkError(ValueError):
    """Raised when a mixed third-jet chunk is incomplete or overstated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _record_hash_matches(record: dict[str, Any]) -> bool:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return record.get("record_sha256") == _content_hash(body)


def _mixed_selector() -> tuple[tuple[int, int, int], ...]:
    triples = tuple(
        triple
        for triple in itertools.combinations_with_replacement(range(ACTIVE_DIRECTION_COUNT), 3)
        if len(set(triple)) > 1
    )
    if len(triples) != TOTAL_MIXED_TRIPLES:
        raise QuarticTC2MixedThirdJetChunkError("mixed selector cardinality mismatch")
    return triples


def _triple_kind(triple: tuple[int, int, int]) -> str:
    left, middle, right = triple
    if left == middle < right:
        return "AAB"
    if left < middle == right:
        return "ABB"
    if left < middle < right:
        return "ABC"
    raise QuarticTC2MixedThirdJetChunkError("selector emitted a diagonal triple")


def _combine_directions(
    terms: tuple[tuple[sp.Expr, dict[str, sp.Expr]], ...],
) -> dict[str, sp.Expr]:
    result: dict[str, sp.Expr] = {}
    for scale, direction in terms:
        for name, value in direction.items():
            result[name] = sp.factor(result.get(name, sp.S.Zero) + scale * value)
    return {name: value for name, value in result.items() if value != 0}


def _direction_payload(direction: dict[str, sp.Expr]) -> dict[str, sp.Matrix]:
    direction_key = tuple(sorted((name, str(value)) for name, value in direction.items()))
    packet = _directional_taylor_packet(
        {
            "atom_index": -1,
            "atom": "polarization_direction:" + _content_hash(direction_key)[:16],
            "direction": direction,
        }
    )
    if len(packet["orders"]) != 3 or not all(
        order["solvable"] and order["residual_zero"] for order in packet["orders"]
    ):
        raise QuarticTC2MixedThirdJetChunkError("directional recurrence failed before polarization")
    return {
        "D3P55": (6 * packet["physical"][3]).applyfunc(sp.factor),
        "D3K55": (6 * packet["energy"][3]).applyfunc(sp.factor),
        "D3TC2": (6 * packet["block"][3]).applyfunc(sp.factor),
        "third_Sylvester_RHS": (6 * packet["orders"][2]["rhs"]).applyfunc(sp.factor),
    }


def _linear_combination(
    payloads: tuple[tuple[sp.Expr, dict[str, sp.Matrix]], ...], denominator: int
) -> dict[str, sp.Matrix]:
    result: dict[str, sp.Matrix] = {}
    for key in payloads[0][1]:
        result[key] = (
            sum(
                (scale * payload[key] for scale, payload in payloads),
                sp.zeros(*payloads[0][1][key].shape),
            )
            / denominator
        ).applyfunc(sp.factor)
    return result


def _polarized_third_payload(
    triple: tuple[int, int, int], directions: list[dict[str, Any]]
) -> dict[str, sp.Matrix]:
    left, middle, right = (directions[index]["direction"] for index in triple)
    kind = _triple_kind(triple)
    if kind == "AAB":
        plus = _direction_payload(_combine_directions(((sp.S.One, left), (sp.S.One, right))))
        minus = _direction_payload(_combine_directions(((sp.S.One, left), (-sp.S.One, right))))
        singleton = _direction_payload(right)
        return _linear_combination(
            ((sp.S.One, plus), (-sp.S.One, minus), (-sp.Integer(2), singleton)),
            6,
        )
    if kind == "ABB":
        plus = _direction_payload(_combine_directions(((sp.S.One, left), (sp.S.One, right))))
        minus = _direction_payload(_combine_directions(((sp.S.One, left), (-sp.S.One, right))))
        singleton = _direction_payload(left)
        return _linear_combination(
            ((sp.S.One, plus), (sp.S.One, minus), (-sp.Integer(2), singleton)),
            6,
        )
    plus = _direction_payload(
        _combine_directions(((sp.S.One, left), (sp.S.One, middle), (sp.S.One, right)))
    )
    right_minus = _direction_payload(
        _combine_directions(((sp.S.One, left), (sp.S.One, middle), (-sp.S.One, right)))
    )
    middle_minus = _direction_payload(
        _combine_directions(((sp.S.One, left), (-sp.S.One, middle), (sp.S.One, right)))
    )
    left_minus = _direction_payload(
        _combine_directions(((-sp.S.One, left), (sp.S.One, middle), (sp.S.One, right)))
    )
    return _linear_combination(
        (
            (sp.S.One, plus),
            (-sp.S.One, right_minus),
            (-sp.S.One, middle_minus),
            (-sp.S.One, left_minus),
        ),
        24,
    )


def generic_mixed_third_polarization_control() -> tuple[bool, dict[str, Any]]:
    x, y, z = sp.symbols("x y z", real=True)
    polynomial = 2 * x**3 + 3 * x**2 * y + 5 * x * y**2 + 7 * y**3 + 11 * x * y * z

    def value(vector: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
        return sp.expand(polynomial.subs(dict(zip((x, y, z), vector, strict=True))))

    a = (sp.Integer(1), sp.Integer(0), sp.Integer(0))
    b = (sp.Integer(0), sp.Integer(1), sp.Integer(0))
    c = (sp.Integer(0), sp.Integer(0), sp.Integer(1))

    def vector_sum(*terms: tuple[sp.Expr, tuple[sp.Expr, ...]]) -> tuple[sp.Expr, ...]:
        return tuple(
            sp.expand(sum(scale * vector[index] for scale, vector in terms)) for index in range(3)
        )

    aab = sp.expand(
        (value(vector_sum((1, a), (1, b))) - value(vector_sum((1, a), (-1, b))) - 2 * value(b)) / 6
    )
    abb = sp.expand(
        (value(vector_sum((1, a), (1, b))) + value(vector_sum((1, a), (-1, b))) - 2 * value(a)) / 6
    )
    abc = sp.expand(
        (
            value(vector_sum((1, a), (1, b), (1, c)))
            - value(vector_sum((1, a), (1, b), (-1, c)))
            - value(vector_sum((1, a), (-1, b), (1, c)))
            - value(vector_sum((-1, a), (1, b), (1, c)))
        )
        / 24
    )
    expected_aab = sp.diff(polynomial, x, x, y).subs({x: 0, y: 0, z: 0}) / 6
    expected_abb = sp.diff(polynomial, x, y, y).subs({x: 0, y: 0, z: 0}) / 6
    expected_abc = sp.diff(polynomial, x, y, z).subs({x: 0, y: 0, z: 0}) / 6
    corrupt = sp.expand(abc + value(vector_sum((-1, a), (1, b), (1, c))) / 24)
    diagonal_passed, diagonal_control = generic_diagonal_third_jet_control()
    passed = bool(
        diagonal_passed
        and aab == expected_aab
        and abb == expected_abb
        and abc == expected_abc
        and corrupt != expected_abc
    )
    return passed, {
        "control": "exact cubic polarization for AAB, ABB, and ABC coefficients",
        "factorial_normalization": (
            "directional payloads are third derivatives; polarization returns the mixed "
            "third derivative tensor with no additional factorial"
        ),
        "AAB_control": str(sp.factor(aab - expected_aab)),
        "ABB_control": str(sp.factor(abb - expected_abb)),
        "ABC_control": str(sp.factor(abc - expected_abc)),
        "diagonal_recurrence_control_sha256": _content_hash(diagonal_control),
        "negative_controls": {
            "omit_fourth_ABC_sign_term": {
                "residual": str(sp.factor(corrupt - expected_abc)),
                "rejected": corrupt != expected_abc,
            },
            "infer_full_mixed_sector_from_one_chunk": {
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "remaining_after_full_chunk": TOTAL_MIXED_TRIPLES - DEFAULT_CHUNK_SIZE,
                "rejected": True,
            },
            "promote_third_jet_to_full_tube": {
                "missing": "fourth-and-higher remainder or nonlinear range theorem",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _solve_sylvester(rhs: sp.Matrix) -> tuple[bool, sp.Matrix, dict[str, Any]]:
    reference = _reference_and_first_jet_packet()
    compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
    delta = sp.zeros(STATE_DIMENSION)
    if solvable:
        for left, left_projector in reference["projectors"].items():
            for right, right_projector in reference["projectors"].items():
                if left != right:
                    delta += left_projector.T * rhs * right_projector / (left - right)
        delta = delta.applyfunc(sp.factor)
    residual = (delta * reference["physical0"] - reference["physical0"].T * delta + rhs).applyfunc(
        sp.factor
    )
    compression_summary = {
        str(eigenvalue): {
            "nonzero_entries": sum(value != 0 for value in matrix),
            "sha256": _content_hash(_matrix_payload(matrix)),
        }
        for eigenvalue, matrix in compressions.items()
        if not matrix.is_zero_matrix
    }
    return (
        solvable,
        delta,
        {
            "nonzero_equal_eigenspace_compressions": compression_summary,
            "residual_zero": residual.is_zero_matrix,
        },
    )


def _candidate_result(
    rhs: sp.Matrix,
    alpha: sp.Symbol,
    c20: sp.Symbol,
    coefficients: dict[str, str],
) -> dict[str, Any]:
    candidate_rhs = rhs.subs(
        {
            alpha: sp.sympify(coefficients["a10"]),
            c20: sp.sympify(coefficients["c20"]),
        }
    ).applyfunc(sp.factor)
    solvable, delta, audit = _solve_sylvester(candidate_rhs)
    return {
        "solvable": solvable,
        "equal_eigenspace_compressions_zero": solvable,
        "nonzero_equal_eigenspace_compressions": audit["nonzero_equal_eigenspace_compressions"],
        "deltaK_ABC_Hermitian": delta.equals(delta.T) if solvable else False,
        "deltaK_ABC_nonzero_entries": sum(value != 0 for value in delta),
        "deltaK_ABC_rank": delta.rank() if solvable else None,
        "deltaK_ABC_sha256": (_content_hash(_matrix_payload(delta)) if solvable else None),
        "third_Sylvester_residual_zero": audit["residual_zero"],
    }


def _pair_bindings(
    triple: tuple[int, int, int],
    directions: list[dict[str, Any]],
    pair_packets: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    positions = sorted(set(triple))
    pairs = sorted(
        {(min(left, right), max(left, right)) for left in positions for right in positions}
    )
    bindings: list[dict[str, Any]] = []
    for left_position, right_position in pairs:
        left_atom = int(directions[left_position]["atom_index"])
        right_atom = int(directions[right_position]["atom_index"])
        key = (min(left_atom, right_atom), max(left_atom, right_atom))
        packet = pair_packets.get(key)
        if packet is None:
            raise QuarticTC2MixedThirdJetChunkError("prior D2 pair packet is absent")
        bindings.append(
            {
                "active_position_pair": [left_position, right_position],
                "atom_index_pair": list(key),
                "D2_pair_packet_content_sha256": packet["content_sha256"],
            }
        )
    return bindings


def run_quartic_tc2_mixed_third_jet_chunk_campaign(
    diagonal_campaign: dict[str, Any],
    quadratic_campaign: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2MixedThirdJetChunkError("unsupported schema_version")
        if not _content_hash_matches(diagonal_campaign) or not _content_hash_matches(
            quadratic_campaign
        ):
            raise QuarticTC2MixedThirdJetChunkError("upstream content hash mismatch")
        expected_upstream = config["expected_upstream_content_sha256"]
        actual_upstream = {
            "diagonal_third_jet": diagonal_campaign["content_sha256"],
            "quadratic_deltaK": quadratic_campaign["content_sha256"],
        }
        if actual_upstream != expected_upstream:
            raise QuarticTC2MixedThirdJetChunkError("configured provenance mismatch")
        if diagonal_campaign.get("status") != (
            "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_"
            "mixed_triples_full_tube_global_H7_fail_closed"
        ) or quadratic_campaign.get("status") != (
            "pass_all_12_complete_reference_quadratic_deltaK_two_jets_full_identity_fail_closed"
        ):
            raise QuarticTC2MixedThirdJetChunkError("upstream status mismatch")
        if (
            diagonal_campaign["upstream_sha256"]["quadratic_deltaK"]
            != quadratic_campaign["content_sha256"]
        ):
            raise QuarticTC2MixedThirdJetChunkError("diagonal/quadratic chain mismatch")
        if any(
            config.get(key) != "fail_closed"
            for key in (
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
            raise QuarticTC2MixedThirdJetChunkError("closure policy mismatch")
        if (
            config.get("selector") != "lexicographic_active_direction_multisets_excluding_AAA"
            or int(config["chunk_size"]) != DEFAULT_CHUNK_SIZE
            or int(config["chunk_offset"]) < 0
            or int(config["chunk_offset"]) >= TOTAL_MIXED_TRIPLES
            or config.get("stop_on_first_obstruction") is not True
            or config.get("resume_policy") != "record_sha256_chain"
        ):
            raise QuarticTC2MixedThirdJetChunkError("chunk contract mismatch")
        generic_passed, generic = generic_mixed_third_polarization_control()
        if not generic_passed:
            raise QuarticTC2MixedThirdJetChunkError("generic polarization control failed")
        records, packets, artifact_hashes = _collect_records(
            canonical_artifacts,
            selector_key="selector_pair_index",
            expected_count=861,
        )
        if artifact_hashes != diagonal_campaign["canonical_second_pair_artifact_content_sha256"]:
            raise QuarticTC2MixedThirdJetChunkError("canonical D2 artifact sequence mismatch")
        pair_packets: dict[tuple[int, int], dict[str, Any]] = {}
        for record in records:
            key = (
                int(record["left_atom_index"]),
                int(record["right_atom_index"]),
            )
            packet = packets.get(str(record["symbolic_pair_packet_sha256"]))
            if packet is None:
                raise QuarticTC2MixedThirdJetChunkError("D2 packet lookup failed")
            pair_packets[key] = packet
        if len(pair_packets) != 861:
            raise QuarticTC2MixedThirdJetChunkError("D2 pair binding coverage mismatch")
        directions = _active_directions()
        diagonal_records = {
            int(record["atom_index"]): record for record in diagonal_campaign["direction_records"]
        }
        if len(directions) != ACTIVE_DIRECTION_COUNT or len(diagonal_records) != 41:
            raise QuarticTC2MixedThirdJetChunkError("active direction binding mismatch")
        coefficients = {
            str(item["candidate_id"]): item["coefficients"]
            for item in diagonal_campaign["certificates"]
        }
        if len(coefficients) != 12:
            raise QuarticTC2MixedThirdJetChunkError("candidate count mismatch")
        selector = _mixed_selector()
        offset = int(config["chunk_offset"])
        selected = selector[offset : offset + DEFAULT_CHUNK_SIZE]
        if len(selected) != DEFAULT_CHUNK_SIZE:
            raise QuarticTC2MixedThirdJetChunkError("short final chunks are not enabled")
        seed_body = {
            "upstream": actual_upstream,
            "canonical_D2_artifact_sequence_sha256": _content_hash(artifact_hashes),
            "selector": config["selector"],
            "chunk_offset": offset,
            "prior_resume_sha256": config.get("expected_prior_resume_sha256"),
        }
        seed = _content_hash(seed_body)
        previous = seed
        manifest: list[dict[str, Any]] = []
        first_obstruction: dict[str, Any] | None = None
        alpha, c20 = sp.symbols("alpha c20")
        for chunk_index, triple in enumerate(selected):
            selector_index = offset + chunk_index
            payload = _polarized_third_payload(triple, directions)
            rhs = payload["third_Sylvester_RHS"]
            free_symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
            alpha = free_symbols.get("alpha", alpha)
            c20 = free_symbols.get("c20", c20)
            symbolic_solvable, symbolic_delta, symbolic_audit = _solve_sylvester(rhs)
            candidate_results: list[dict[str, Any]] = []
            obstructed_candidates: list[str] = []
            for candidate_id, candidate_coefficients in sorted(coefficients.items()):
                candidate = _candidate_result(rhs, alpha, c20, candidate_coefficients)
                if not candidate["solvable"]:
                    obstructed_candidates.append(candidate_id)
                candidate_results.append({"candidate_id": candidate_id, **candidate})
            unique_positions = sorted(set(triple))
            record_body = {
                "selector_index": selector_index,
                "chunk_index": chunk_index,
                "triple_kind": _triple_kind(triple),
                "active_position_triple": list(triple),
                "atom_index_triple": [int(directions[index]["atom_index"]) for index in triple],
                "atom_triple": [str(directions[index]["atom"]) for index in triple],
                "prior_bindings": {
                    "D1_variable_sylvester_content_sha256": diagonal_campaign["upstream_sha256"][
                        "variable_sylvester"
                    ],
                    "D2_pair_packets": _pair_bindings(triple, directions, pair_packets),
                    "diagonal_D3_direction_record_sha256": [
                        _content_hash(diagonal_records[int(directions[index]["atom_index"])])
                        for index in unique_positions
                    ],
                    "diagonal_third_jet_artifact_content_sha256": diagonal_campaign[
                        "content_sha256"
                    ],
                },
                "D3P55_nonzero_entries": sum(value != 0 for value in payload["D3P55"]),
                "D3P55_sha256": _content_hash(_matrix_payload(payload["D3P55"])),
                "D3K55_nonzero_entries": sum(value != 0 for value in payload["D3K55"]),
                "D3K55_sha256": _content_hash(_matrix_payload(payload["D3K55"])),
                "D3TC2_nonzero_entries": sum(value != 0 for value in payload["D3TC2"]),
                "D3TC2_sha256": _content_hash(_matrix_payload(payload["D3TC2"])),
                "third_Sylvester_RHS_sha256": _content_hash(_matrix_payload(rhs)),
                "symbolic_parameter_compatible": symbolic_solvable,
                "symbolic_nonzero_equal_eigenspace_compressions": symbolic_audit[
                    "nonzero_equal_eigenspace_compressions"
                ],
                "symbolic_deltaK_ABC_Hermitian": (
                    symbolic_delta.equals(symbolic_delta.T) if symbolic_solvable else False
                ),
                "symbolic_deltaK_ABC_nonzero_entries": sum(value != 0 for value in symbolic_delta),
                "symbolic_deltaK_ABC_rank": (symbolic_delta.rank() if symbolic_solvable else None),
                "symbolic_deltaK_ABC_sha256": (
                    _content_hash(_matrix_payload(symbolic_delta)) if symbolic_solvable else None
                ),
                "candidate_results": candidate_results,
                "obstructed_candidate_ids": obstructed_candidates,
                "previous_record_sha256": previous,
            }
            record = {
                **record_body,
                "record_sha256": _content_hash(record_body),
            }
            if not _record_hash_matches(record):
                raise QuarticTC2MixedThirdJetChunkError("record hash mismatch")
            manifest.append(record)
            previous = record["record_sha256"]
            _directional_taylor_packet_cached.cache_clear()
            if obstructed_candidates:
                first_obstruction = {
                    "selector_index": selector_index,
                    "record_sha256": record["record_sha256"],
                    "active_position_triple": list(triple),
                    "atom_triple": record["atom_triple"],
                    "obstructed_candidate_ids": obstructed_candidates,
                    "gate": "equal-eigenspace compatibility of mixed third Sylvester RHS",
                }
                break
        processed = len(manifest)
        passed_records = sum(not record["obstructed_candidate_ids"] for record in manifest)
        candidate_evaluations = sum(len(record["candidate_results"]) for record in manifest)
        candidate_obstructions = sum(len(record["obstructed_candidate_ids"]) for record in manifest)
        kinds = Counter(record["triple_kind"] for record in manifest)
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "stop_first_exact_mixed_third_jet_obstruction"
                if first_obstruction
                else "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": actual_upstream,
            "canonical_D2_artifact_content_sha256": artifact_hashes,
            "canonical_D2_artifact_sequence_sha256": _content_hash(artifact_hashes),
            "config_sha256": _content_hash(config),
            "generic_mixed_third_polarization_control": generic,
            "chunk_contract": {
                "selector": config["selector"],
                "global_mixed_triple_count": TOTAL_MIXED_TRIPLES,
                "chunk_offset": offset,
                "requested_chunk_size": DEFAULT_CHUNK_SIZE,
                "processed_count": processed,
                "next_offset": offset + processed,
                "stop_on_first_obstruction": True,
                "stopped_early": first_obstruction is not None,
                "resume_policy": config["resume_policy"],
                "prior_resume_sha256": config.get("expected_prior_resume_sha256"),
                "resume_seed_sha256": seed,
                "resume_tip_sha256": previous,
            },
            "counts": {
                "selected": processed,
                "triple_kind_counts": dict(sorted(kinds.items())),
                "symbolic_parameter_compatible": sum(
                    record["symbolic_parameter_compatible"] for record in manifest
                ),
                "candidate_evaluations": candidate_evaluations,
                "candidate_solvable": candidate_evaluations - candidate_obstructions,
                "candidate_obstructed": candidate_obstructions,
                "mixed_triples_remaining": TOTAL_MIXED_TRIPLES - offset - passed_records,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "first_exact_obstruction": first_obstruction,
            "triple_manifest": manifest,
            "closure_ledger": {
                "processed_mixed_third_jets_closed": passed_records,
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
                f"The first {processed} lexicographic mixed active-sector triples were "
                "evaluated with exact cubic polarization and candidate-specific Sylvester "
                "compatibility, stopping immediately if a candidate obstruction appeared."
            ),
            "scope": (
                "This restartable chunk is only a subset of the 12,300 polarized mixed "
                "triples. Unprocessed triples, full tube identity, CK1, CK3, TC2, B7, "
                "global H7, dyadic closure, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2MixedThirdJetChunkError) as error:
        _directional_taylor_packet_cached.cache_clear()
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "triple_manifest": [],
            "first_exact_obstruction": None,
            "counts": {
                "selected": 0,
                "candidate_evaluations": 0,
                "candidate_solvable": 0,
                "candidate_obstructed": 0,
                "mixed_triples_remaining": TOTAL_MIXED_TRIPLES,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_mixed_third_jet_chunk_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
