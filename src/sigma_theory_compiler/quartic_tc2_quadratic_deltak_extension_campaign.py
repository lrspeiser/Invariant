from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_second_atom_chunk64_campaign import _candidate_coefficients

SCHEMA_VERSION = "sigma-quartic-tc2-quadratic-deltak-extension-campaign-1.0"
ATOM_DIMENSION = 153
TOTAL_UNORDERED_PAIRS = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2


class QuarticTC2QuadraticDeltaKError(ValueError):
    """Raised when the reference quadratic deltaK jet is not exact and complete."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(value: dict[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return value.get("content_sha256") == _content_hash(body)


def generic_quadratic_sylvester_jet_control() -> tuple[bool, dict[str, Any]]:
    y, h, k = sp.symbols("y h k", real=True)
    d0, d1, d2 = sp.symbols("d0 d1 d2", real=True)
    quadratic = d0 + d1 * y + d2 * y**2 / 2
    passed = bool(
        quadratic.subs(y, 0) == d0
        and sp.diff(quadratic, y).subs(y, 0) == d1
        and sp.diff(quadratic, y, 2).subs(y, 0) == d2
        and sp.diff(quadratic, y, 3) == 0
        and h * k != 0
    )
    return passed, {
        "control": "quadratic Taylor realization versus a full nonlinear Sylvester solution",
        "ansatz": (
            "deltaK[2](Y)=a10*deltaK0+a10^2*sum_A Y_A deltaK_A"
            "+1/2*sum_AB Y_A Y_B deltaK_AB"
        ),
        "reference_jet_orders_closed": [0, 1, 2],
        "full_tube_Sylvester_identity_closed": False,
        "norm_convention": (
            "component-l_infinity state directions and Frobenius matrix norm; "
            "the same constants bound coordinate-l2 directions because ||h||inf<=||h||2"
        ),
        "negative_controls": {
            "omit_mixed_pair_multiplicity": {
                "missing": "deltaK_AB h_A k_B + deltaK_AB h_B k_A for A<B",
                "rejected": True,
            },
            "infer_full_identity_from_two_jet": {
                "missing": "third and higher Sylvester residual jets or a nonlinear solve",
                "rejected": True,
            },
            "infer_positivity_without_ell10_smallness": {
                "missing": "Weyl perturbation condition",
                "rejected": True,
            },
            "promote_reference_quadratic_jet_to_TC2": {
                "missing": "tube-uniform full Sylvester identity and CK1/CK3 ledger",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any], *, expected_offset: int, expected_tip: str
) -> None:
    if (
        not _content_hash_matches(checkpoint)
        or int(checkpoint.get("next_offset", -1)) != expected_offset
        or checkpoint.get("prior_resume_sha256") != expected_tip
        or checkpoint.get("permanently_stopped")
    ):
        raise QuarticTC2QuadraticDeltaKError("completed checkpoint integrity mismatch")
    claims = checkpoint.get("claims", {})
    if not claims or any(bool(value) for value in claims.values()):
        raise QuarticTC2QuadraticDeltaKError("upstream checkpoint promoted a global claim")


def _validate_artifact(artifact: dict[str, Any]) -> None:
    if not _content_hash_matches(artifact) or artifact.get("errors"):
        raise QuarticTC2QuadraticDeltaKError("second-pair artifact integrity mismatch")
    if artifact.get("first_exact_obstruction") is not None:
        raise QuarticTC2QuadraticDeltaKError("second-pair artifact contains an obstruction")
    for packet in artifact.get("symbolic_pair_packets", []):
        if not _content_hash_matches(packet):
            raise QuarticTC2QuadraticDeltaKError("symbolic pair packet hash mismatch")


def _record_hash_matches(record: dict[str, Any]) -> bool:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return record.get("record_sha256") == _content_hash(body)


def _collect_records(
    artifacts: list[dict[str, Any]], *, selector_key: str, expected_count: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    packets: dict[str, dict[str, Any]] = {}
    artifact_hashes: list[str] = []
    for artifact in artifacts:
        _validate_artifact(artifact)
        artifact_hashes.append(str(artifact["content_sha256"]))
        for packet in artifact.get("symbolic_pair_packets", []):
            packet_hash = str(packet["content_sha256"])
            if packet_hash in packets and packets[packet_hash] != packet:
                raise QuarticTC2QuadraticDeltaKError("symbolic packet hash collision")
            packets[packet_hash] = packet
        for source_record in artifact.get("pair_manifest", []):
            record = source_record
            if selector_key not in record and selector_key == "selector_pair_index":
                record = dict(source_record)
                record[selector_key] = int(artifact["chunk_contract"]["chunk_offset"]) + int(
                    source_record["chunk_index"]
                )
            records.append(record)
    records.sort(key=lambda item: int(item[selector_key]))
    if (
        len(records) != expected_count
        or [int(record[selector_key]) for record in records] != list(range(expected_count))
        or not all(
            _record_hash_matches(
                {key: value for key, value in record.items() if key != selector_key}
                if selector_key == "selector_pair_index"
                and "chunk_index" in record
                and "selector_pair_index" in record
                else record
            )
            for record in records
        )
    ):
        raise QuarticTC2QuadraticDeltaKError("second-pair selector coverage mismatch")
    return records, packets, artifact_hashes


def _hermitian(entries: list[dict[str, Any]]) -> bool:
    matrix = {
        (int(entry["row"]), int(entry["column"])): str(entry["value"])
        for entry in entries
    }
    return all(
        value == matrix.get((column, row))
        for (row, column), value in matrix.items()
    )


def _candidate_quadratic_envelopes(
    records: list[dict[str, Any]],
    packets: dict[str, dict[str, Any]],
    coefficients: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    alpha, c20 = sp.symbols("alpha c20")
    expression_ceiling: dict[str, int] = {}
    packet_ceiling: dict[str, int] = {}
    for packet_hash, packet in packets.items():
        entries = packet["deltaK_AB_entries"]
        if not _hermitian(entries):
            raise QuarticTC2QuadraticDeltaKError("symbolic deltaK_AB is not Hermitian")
        ceiling = 0
        for entry in entries:
            value_text = str(entry["value"])
            if value_text not in expression_ceiling:
                polynomial = sp.Poly(sp.sympify(value_text), alpha, c20)
                coefficient_l1 = sum((sp.Abs(item) for item in polynomial.coeffs()), sp.S.Zero)
                expression_ceiling[value_text] = int(sp.ceiling(coefficient_l1))
            ceiling += expression_ceiling[value_text]
        packet_ceiling[packet_hash] = ceiling
    common_c2 = 0
    nonzero_symbolic_pairs = 0
    norm_histogram: Counter[str] = Counter()
    candidate_matrix_hashes: dict[str, list[str]] = {key: [] for key in coefficients}
    for record in records:
        packet_hash = str(
            record.get("symbolic_pair_packet_sha256")
            or record.get("symbolic_packet_sha256")
        )
        packet = packets.get(packet_hash)
        if packet is None:
            raise QuarticTC2QuadraticDeltaKError("record packet is absent")
        weight = 1 if record["left_atom_index"] == record["right_atom_index"] else 2
        common_c2 += weight * packet_ceiling[packet_hash]
        if packet["deltaK_AB_entries"]:
            nonzero_symbolic_pairs += 1
        norm_histogram[str(packet_ceiling[packet_hash])] += 1
        results = {
            str(item["candidate_id"]): item for item in record["candidate_results"]
        }
        if set(results) != set(coefficients):
            raise QuarticTC2QuadraticDeltaKError("candidate set mismatch in pair record")
        for candidate_id, result in results.items():
            if not all(
                bool(result.get(key))
                for key in ("solvable", "Hermitian", "second_Sylvester_residual_zero")
            ) or not result.get("deltaK_AB_sha256"):
                raise QuarticTC2QuadraticDeltaKError("candidate pair is not exactly solved")
            candidate_matrix_hashes[candidate_id].append(str(result["deltaK_AB_sha256"]))
    aggregate: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_coefficients in sorted(coefficients.items()):
        if (
            sp.Abs(sp.sympify(candidate_coefficients["a10"])) > 1
            or sp.Abs(sp.sympify(candidate_coefficients["c20"])) > 1
        ):
            raise QuarticTC2QuadraticDeltaKError("candidate escaped unit coefficient box")
        aggregate[candidate_id] = {
            "candidate_id": candidate_id,
            "coefficients": candidate_coefficients,
            "D2_deltaK_coordinate_linf_to_Frobenius_integer_ceiling": common_c2,
            "bound": "||D2 deltaK[2](Y)[h,k]||F <= C2 ||h||inf ||k||inf",
            "envelope_method": (
                "symbolic entry coefficient-l1 on |a10|,|c20|<=1, then matrix entry-l1, "
                "with factor two for mixed unordered coordinate pairs"
            ),
            "nonzero_symbolic_pair_coefficients": nonzero_symbolic_pairs,
            "evaluated_pair_coefficients": len(records),
            "candidate_matrix_sequence_sha256": _content_hash(
                candidate_matrix_hashes[candidate_id]
            ),
            "outward_norm_histogram_sha256": _content_hash(dict(sorted(norm_histogram.items()))),
        }
    return aggregate


def _certify_candidate(
    variable_certificate: dict[str, Any],
    quadratic: dict[str, Any],
    radius: sp.Rational,
) -> dict[str, Any]:
    coefficients = variable_certificate["coefficients"]
    alpha = sp.sympify(coefficients["a10"])
    positivity = variable_certificate["affine_deltaK_positivity"]
    c0 = sp.sympify(positivity["C0"])
    c1 = sp.sympify(positivity["C1"])
    c2 = sp.Integer(quadratic["D2_deltaK_coordinate_linf_to_Frobenius_integer_ceiling"])
    d1_bound = sp.Abs(alpha) ** 2 * c1 + radius * c2
    value_bound = sp.Abs(alpha) * c0 + sp.Abs(alpha) ** 2 * radius * c1 + radius**2 * c2 / 2
    lambda_k = sp.sympify(positivity["lambda_K55"])
    ell10_max = sp.factor(lambda_k / (2 * value_bound))
    if not all(bool(value > 0) for value in (c0, c1, c2, d1_bound, value_bound, ell10_max)):
        raise QuarticTC2QuadraticDeltaKError("nonpositive quadratic envelope")
    return {
        "schema_version": "sigma-quartic-tc2-quadratic-deltak-certificate-1.0",
        "status": "pass_reference_quadratic_Hermitian_two_jet_full_identity_fail_closed",
        "candidate_id": variable_certificate["candidate_id"],
        "coefficients": coefficients,
        "quadratic_deltaK": {
            "Hermitian_on_tube": True,
            "coordinate_component_radius": str(radius),
            "D1_coordinate_linf_to_Frobenius_bound": str(d1_bound),
            "D1_coordinate_linf_to_Frobenius_numeric_upper": float(sp.N(d1_bound, 16)),
            "D2_coordinate_linf_to_Frobenius_integer_ceiling": str(c2),
            "value_perturbation_from_K55_bound": str(value_bound),
            "reference_Sylvester_residual_derivatives_zero": [0, 1, 2],
            "full_tube_Sylvester_identity": False,
        },
        "positivity": {
            "lambda_K55": str(lambda_k),
            "sufficient_condition": "|ell10| <= ell10_max_at_declared_radius",
            "ell10_max_at_declared_radius": str(ell10_max),
            "ell10_max_numeric_approximation": float(sp.N(ell10_max, 16)),
            "Weyl_margin": "lambda_K55/2",
            "closed_for_quadratic_reference_extension": True,
        },
        "closure_ledger": {
            "complete_reference_deltaK_two_jet": True,
            "quadratic_D1_D2_bounds": True,
            "quadratic_Hermiticity": True,
            "quadratic_positivity_under_explicit_ell10_condition": True,
            "full_tube_Sylvester_identity": False,
            "variable_CK1_all_terms_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "first_remaining_blocker": {
            "gate": "third Sylvester jet / nonlinear range condition",
            "required": (
                "D3K55, D3P55, D3TC2 and deltaK_ABC (or a tube-uniform nonlinear "
                "Sylvester-range theorem) with equal-eigenspace compatibility"
            ),
            "why_D1_D2_bounds_do_not_close_it": (
                "the quadratic ansatz has no cubic coefficient, while products with "
                "variable P55 generate an uncontrolled cubic Sylvester residual"
            ),
            "closed": False,
        },
    }


def run_quartic_tc2_quadratic_deltak_extension_campaign(
    variable_campaign: dict[str, Any],
    classification_campaign: dict[str, Any],
    canonical_checkpoint: dict[str, Any],
    obligation_checkpoint: dict[str, Any],
    p55_tube_campaign: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    obligation_artifacts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2QuadraticDeltaKError("unsupported schema_version")
        upstream = (
            variable_campaign,
            classification_campaign,
            p55_tube_campaign,
        )
        if not all(_content_hash_matches(item) for item in upstream):
            raise QuarticTC2QuadraticDeltaKError("upstream content hash mismatch")
        expected_hashes = config["expected_upstream_content_sha256"]
        actual_hashes = {
            "variable_sylvester": variable_campaign["content_sha256"],
            "excluded_pair_classification": classification_campaign["content_sha256"],
            "canonical_checkpoint": canonical_checkpoint["content_sha256"],
            "obligation_checkpoint": obligation_checkpoint["content_sha256"],
            "P55_tube": p55_tube_campaign["content_sha256"],
        }
        if actual_hashes != expected_hashes:
            raise QuarticTC2QuadraticDeltaKError("configured provenance mismatch")
        if any(
            config.get(key) != "fail_closed"
            for key in ("CK1_policy", "CK3_policy", "TC2_policy", "B7_policy", "global_H7_policy", "lifespan_policy")
        ):
            raise QuarticTC2QuadraticDeltaKError("global closure policy mismatch")
        _validate_checkpoint(
            canonical_checkpoint,
            expected_offset=861,
            expected_tip=str(config["expected_canonical_tip"]),
        )
        _validate_checkpoint(
            obligation_checkpoint,
            expected_offset=2675,
            expected_tip=str(config["expected_obligation_tip"]),
        )
        canonical, canonical_packets, canonical_hashes = _collect_records(
            canonical_artifacts, selector_key="selector_pair_index", expected_count=861
        )
        obligations, obligation_packets, obligation_hashes = _collect_records(
            obligation_artifacts,
            selector_key="obligation_selector_index",
            expected_count=2675,
        )
        classification = classification_campaign["excluded_pair_manifest"]
        zero_records = [item for item in classification if item["rigorously_discharged"]]
        obligation_indices = {int(item["global_pair_index"]) for item in obligations}
        classified_obligations = {
            int(item["global_pair_index"])
            for item in classification
            if not item["rigorously_discharged"]
        }
        all_indices = (
            {int(item["global_pair_index"]) for item in canonical}
            | {int(item["global_pair_index"]) for item in classification}
        )
        if (
            len(classification) != 10920
            or len(zero_records) != 8245
            or obligation_indices != classified_obligations
            or all_indices != set(range(TOTAL_UNORDERED_PAIRS))
            or not classification_campaign["entrywise_zero_chain_rule_control"]["passed"]
        ):
            raise QuarticTC2QuadraticDeltaKError("11,781-pair partition mismatch")
        passed, generic = generic_quadratic_sylvester_jet_control()
        if not passed:
            raise QuarticTC2QuadraticDeltaKError("generic quadratic control failed")
        coefficients = _candidate_coefficients(variable_campaign)
        if len(coefficients) != 12:
            raise QuarticTC2QuadraticDeltaKError("candidate count mismatch")
        packets = {**canonical_packets, **obligation_packets}
        records = canonical + obligations
        quadratic = _candidate_quadratic_envelopes(records, packets, coefficients)
        variable_certificates = {
            str(item["candidate_id"]): item for item in variable_campaign["certificates"]
        }
        radius = sp.Rational(str(config["coordinate_component_radius"]))
        certificates = [
            _certify_candidate(variable_certificates[candidate_id], quadratic[candidate_id], radius)
            for candidate_id in sorted(coefficients)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_complete_reference_quadratic_deltaK_two_jets_full_identity_fail_closed",
            "errors": [],
            "upstream_sha256": actual_hashes,
            "second_pair_artifact_content_sha256": {
                "canonical": canonical_hashes,
                "obligations": obligation_hashes,
            },
            "second_pair_artifact_sequence_sha256": {
                "canonical": _content_hash(canonical_hashes),
                "obligations": _content_hash(obligation_hashes),
            },
            "config_sha256": _content_hash(config),
            "generic_quadratic_sylvester_jet_control": generic,
            "pair_partition": {
                "total_unordered_coordinate_pairs": TOTAL_UNORDERED_PAIRS,
                "canonical_active_exact_pairs": len(canonical),
                "excluded_exact_obligations": len(obligations),
                "entrywise_zero_chain_rule_pairs": len(zero_records),
                "coverage_complete": True,
                "global_pair_index_set_sha256": _content_hash(sorted(all_indices)),
            },
            "quadratic_D2_envelopes": [quadratic[key] for key in sorted(quadratic)],
            "consequences_for_variable_energy": {
                "CK1": {
                    "closed": False,
                    "proved_step": (
                        "the quadratic surrogate now has tube-uniform D1/D2 coefficient "
                        "bounds that can be combined with the hash-bound coordinate "
                        "DP55/D2P55 envelopes"
                    ),
                    "remaining": (
                        "CK1 for the actual symmetrizer requires replacing the surrogate "
                        "by a full Sylvester solution and bounding its higher remainder"
                    ),
                },
                "CK3": {
                    "closed": False,
                    "proved_step": "all first and second coefficient differentiations of the surrogate are bounded",
                    "remaining": (
                        "the first derivative of the cubic Sylvester residual needs D3 data "
                        "or a nonlinear implicit-solution estimate"
                    ),
                },
                "TC2": {
                    "closed": False,
                    "proved_step": "TC2 Sylvester cancellation is exact at Y=0 through two state derivatives",
                    "remaining": (
                        "no tube bound or cancellation is proved for the order-three-and-higher "
                        "equal-eigenspace residual"
                    ),
                },
                "P55_tube_provenance_sha256": p55_tube_campaign["content_sha256"],
            },
            "counts": {
                "selected": len(certificates),
                "reference_two_jets_closed": len(certificates),
                "full_tube_Sylvester_identities": 0,
                "full_variable_CK1_closures": 0,
                "CK3_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 11,781 unordered coordinate-pair coefficients define a complete, "
                "Hermitian reference quadratic deltaK two-jet with explicit D1/D2 and "
                "conditional positivity bounds for each of 12 candidates."
            ),
            "scope": (
                "This is an exact order-two Taylor certificate, not a tube-uniform full "
                "Sylvester solution. Variable CK1, CK3, TC2, B7, global H7, dyadic "
                "summation, and lifespan remain fail-closed at the third-order range gate."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2QuadraticDeltaKError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "reference_two_jets_closed": 0,
                "full_tube_Sylvester_identities": 0,
                "full_variable_CK1_closures": 0,
                "CK3_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 1,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_quadratic_deltak_extension_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
