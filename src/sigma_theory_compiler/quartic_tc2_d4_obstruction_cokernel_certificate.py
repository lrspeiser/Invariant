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

SCHEMA = "sigma-quartic-tc2-d4-obstruction-cokernel-certificate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-obstruction-cokernel-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_RHS_SHA256 = "47cb50c6c9e882626e4b5eba3f548be8ed162d076ed275f0edb7b230970d4850"
EXPECTED_COMPRESSION_SHA256 = (
    "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
)
WITNESS_POSITIONS = ((16, 21, 1), (21, 16, -1), (21, 28, -1), (28, 21, 1))
CANDIDATE_PARAMETERS = {
    "quartic-symbol-06e267a9215345b6": ("-1/2", "-1"),
    "quartic-symbol-076dc0ba965ab63a": ("1/2", "1"),
    "quartic-symbol-317e5395817a432b": ("-1", "1"),
    "quartic-symbol-50f184dfe1a814bf": ("-1/2", "0"),
    "quartic-symbol-5455cad9e42a0dbc": ("-1", "-1"),
    "quartic-symbol-561de1410d6cb21f": ("1/2", "-1"),
    "quartic-symbol-8fd254934d778c28": ("1", "1"),
    "quartic-symbol-9e65901e5299a514": ("1", "0"),
    "quartic-symbol-e4a6a9193316a6ff": ("1/2", "0"),
    "quartic-symbol-ef832e4c3b71ee42": ("1", "-1"),
    "quartic-symbol-f31a234e2bf7b97f": ("-1/2", "1"),
    "quartic-symbol-fb5c20c15ce6d778": ("-1", "0"),
}
SPECIALIZATION_CERTIFICATES = {
    "-1": (
        "-34816/15",
        "34816/15",
        "e03a81da200375e9058446c494409461dbedea0fbc4f5545fd83d401f954430e",
    ),
    "-1/2": (
        "-1088/15",
        "1088/15",
        "5988648c9ccc7feefb28428b03795eb87e19b0e3e4431e67e48ed2eba8f189ce",
    ),
    "1/2": (
        "1088/15",
        "1088/15",
        "d39ed0a639e6a85d74cacd94a370f2f19eb9988e8c016874e7014098be110783",
    ),
    "1": (
        "34816/15",
        "34816/15",
        "3f3728ac7c1618847dd2a33d32fbd53a5346d10fa58ef371317a5d6988b509af",
    ),
}


class QuarticTC2D4ObstructionCokernelCertificateError(ValueError):
    """Raised when the exact D4 obstruction certificate does not close."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
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


def _validate_inputs(
    chunk: dict[str, Any], campaign: dict[str, Any], candidates: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    obstruction = chunk.get("first_exact_obstruction", {})
    manifest = chunk.get("obligation_manifest", [])
    selector = campaign.get("selector", {}).get("records", [])
    certificates = candidates.get("certificates", [])
    if (
        chunk.get("status") != "stop_first_exact_fourth_jet_obstruction"
        or chunk.get("chunk_contract", {}).get("next_obligation_offset") != 245
        or chunk.get("chunk_contract", {}).get("processed_count") != 21
        or chunk.get("chunk_contract", {}).get(
            "records_after_first_obstruction_committed_or_inferred"
        )
        != 0
        or obstruction.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(obstruction.get("active_indices", ())) != ACTIVE_INDICES
        or len(manifest) != 21
        or manifest[-1].get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(manifest[-1].get("active_indices", ())) != ACTIVE_INDICES
        or manifest[-1].get("record_sha256") != obstruction.get("record_sha256")
        or manifest[-1].get("fourth_Sylvester_RHS_sha256") != EXPECTED_RHS_SHA256
        or manifest[-1]
        .get("symbolic_nonzero_equal_eigenspace_compressions", {})
        .get("0", {})
        .get("sha256")
        != EXPECTED_COMPRESSION_SHA256
        or len(selector) != 3060
        or selector[OBLIGATION_OFFSET].get("record_sha256")
        != obstruction.get("selector_record_sha256")
        or tuple(selector[OBLIGATION_OFFSET].get("active_indices", ()))
        != ACTIVE_INDICES
        or len(certificates) != EXPECTED_CANDIDATES
        or candidates.get("counts", {}).get("selected") != EXPECTED_CANDIDATES
        or sorted(obstruction.get("obstructed_candidate_ids", []))
        != sorted(certificate["candidate_id"] for certificate in certificates)
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "authoritative D4 obstruction inputs are inconsistent"
        )
    return manifest[-1], selector[OBLIGATION_OFFSET]


def _exact_certificate(
    campaign: dict[str, Any], candidates: dict[str, Any], obstruction_record: dict[str, Any]
) -> dict[str, Any]:
    active_positions = campaign["selector"]["active_positions"]
    directions = _active_directions()
    basis_directions = [directions[position] for position in active_positions]
    payload, directional_evaluations = _polarized_fourth_payload(
        ACTIVE_INDICES, basis_directions
    )
    rhs = payload["fourth_Sylvester_RHS"]
    if (
        directional_evaluations != 15
        or _content_hash(_matrix_payload(rhs)) != EXPECTED_RHS_SHA256
        or _content_hash(_matrix_payload(payload["D4K55"]))
        != obstruction_record["D4K55_sha256"]
        or _content_hash(_matrix_payload(payload["D4P55"]))
        != obstruction_record["D4P55_sha256"]
        or _content_hash(_matrix_payload(payload["D4TC2"]))
        != obstruction_record["D4TC2_sha256"]
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "exact fourth-polarization replay mismatch"
        )
    reference = _reference_and_first_jet_packet()
    compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    nonzero = {key: value for key, value in compressions.items() if not value.is_zero_matrix}
    if set(nonzero) != {sp.S.Zero}:
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "unexpected nonzero equal-eigenspace compression"
        )
    compression = nonzero[sp.S.Zero]
    alpha_symbols = [symbol for symbol in compression.free_symbols if str(symbol) == "alpha"]
    if len(alpha_symbols) != 1:
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "D4 compression does not have the expected alpha parameter"
        )
    alpha = alpha_symbols[0]
    scalar = sp.Rational(34816, 15) * alpha**5
    witness = sp.zeros(STATE_DIMENSION)
    for row, column, sign in WITNESS_POSITIONS:
        witness[row, column] = sign
    if (
        compression != scalar * witness
        or _content_hash(_matrix_payload(compression)) != EXPECTED_COMPRESSION_SHA256
        or compression.rank() != 2
        or compression.T != -compression
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "rank-two cokernel factorization mismatch"
        )
    candidate_rows: list[dict[str, Any]] = []
    observed_hashes = {
        row["candidate_id"]: row["nonzero_equal_eigenspace_compressions"]["0"][
            "sha256"
        ]
        for row in obstruction_record["candidate_results"]
    }
    for certificate in sorted(candidates["certificates"], key=lambda row: row["candidate_id"]):
        candidate_id = certificate["candidate_id"]
        alpha_value = sp.sympify(certificate["coefficients"]["a10"])
        candidate_compression = compression.subs(alpha, alpha_value)
        candidate_sha256 = _content_hash(_matrix_payload(candidate_compression))
        if (
            alpha_value == 0
            or candidate_compression.rank() != 2
            or candidate_sha256 != observed_hashes[candidate_id]
        ):
            raise QuarticTC2D4ObstructionCokernelCertificateError(
                f"candidate obstruction mismatch: {candidate_id}"
            )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "a10": str(alpha_value),
                "c20": str(sp.sympify(certificate["coefficients"]["c20"])),
                "witness_scalar": str(sp.factor(scalar.subs(alpha, alpha_value))),
                "absolute_witness_entry": str(abs(scalar.subs(alpha, alpha_value))),
                "compression_rank": 2,
                "compression_nonzero_entries": 4,
                "compression_sha256": candidate_sha256,
                "compatible": False,
            }
        )
    zero_rhs = rhs.subs(alpha, 0).applyfunc(sp.factor)
    zero_solvable, zero_delta, zero_audit = _solve_sylvester(zero_rhs)
    if (
        not zero_solvable
        or not zero_rhs.is_zero_matrix
        or not zero_delta.is_zero_matrix
        or not zero_audit["residual_zero"]
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "alpha-zero exact negative control failed"
        )
    absolute_entries = [sp.sympify(row["absolute_witness_entry"]) for row in candidate_rows]
    return {
        "directional_evaluations": directional_evaluations,
        "rhs": {
            "shape": [STATE_DIMENSION, STATE_DIMENSION],
            "nonzero_entries": sum(value != 0 for value in rhs),
            "free_parameters": sorted(str(symbol) for symbol in rhs.free_symbols),
            "sha256": EXPECTED_RHS_SHA256,
        },
        "equal_eigenspace_compressions": {
            "nonzero_eigenvalues": ["0"],
            "zero_eigenspace": {
                "factorization": "(34816/15)*alpha^5*W",
                "scalar": str(scalar),
                "witness_matrix_W_sparse": _sparse_payload(witness),
                "compression_sparse": _sparse_payload(compression),
                "nonzero_entries": 4,
                "generic_rank": 2,
                "skew_symmetric": True,
                "sha256": EXPECTED_COMPRESSION_SHA256,
            },
            "all_other_reference_eigenspaces_zero": True,
        },
        "range_certificate": {
            "sylvester_operator": "S(X)=X*P55(0)-P55(0)^T*X",
            "annihilator_identity": "Pi_0^T*S(X)*Pi_0=0 for every 55x55 X",
            "witness_functional": "e_16^T*Pi_0^T*R4*Pi_0*e_21",
            "witness_value": str(scalar),
            "compatibility_polynomial": "alpha^5",
            "compatibility_iff_over_Q_or_R": "alpha=0",
            "independent_of_c20": True,
            "conclusion": (
                "The authoritative canonical order-one-through-three extension has no "
                "order-four Sylvester solution on obligation 244 whenever alpha is nonzero."
            ),
        },
        "candidate_classification": candidate_rows,
        "exact_candidate_gap": {
            "minimum_absolute_nonzero_witness_entry": str(min(absolute_entries)),
            "maximum_absolute_nonzero_witness_entry": str(max(absolute_entries)),
            "interval": "[1088/15,34816/15]",
            "interval_is_exact_rational": True,
            "zero_excluded": True,
        },
        "alpha_zero_control": {
            "rhs_zero": True,
            "compression_zero": True,
            "sylvester_solvable": True,
            "residual_zero": True,
            "registered_candidate_has_alpha_zero": False,
        },
    }


def build_certificate(project_root: Path, config_path: Path) -> dict[str, Any]:
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
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "D4 cokernel config mismatch"
        )
    for local_key in ("campaign_source", "campaign_test"):
        binding = config[local_key]
        if _file_sha256((project_root / binding["path"]).read_bytes()) != binding["file_sha256"]:
            raise QuarticTC2D4ObstructionCokernelCertificateError(
                f"local binding mismatch: {local_key}"
            )
    chunk = _load_bound(project_root, config["obstruction_chunk"])
    campaign = _load_bound(project_root, config["fourth_campaign"])
    candidates = _load_bound(project_root, config["candidate_source"])
    obstruction_record, selector_record = _validate_inputs(chunk, campaign, candidates)
    exact = _exact_certificate(campaign, candidates, obstruction_record)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
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
                "obstruction_chunk",
                "fourth_campaign",
                "candidate_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "active_positions": selector_record["active_positions"],
            "multiplicity_partition": selector_record["multiplicity_partition"],
            "selector_record_sha256": selector_record["record_sha256"],
            "obstruction_record_sha256": obstruction_record["record_sha256"],
        },
        "exact_symbolic_certificate": exact,
        "counts": {
            "selector_obligations_classified": 1,
            "directional_polarization_evaluations": 15,
            "candidate_specializations_checked": EXPECTED_CANDIDATES,
            "candidate_obstructions_certified": EXPECTED_CANDIDATES,
            "candidate_compatibilities_certified": 0,
            "nonzero_reference_eigenspace_compressions": 1,
            "compression_nonzero_entries": 4,
            "compression_generic_rank": 2,
            "negative_controls": 4,
            "inferred_passes": 0,
        },
        "negative_controls": {
            "discard_zero_eigenspace_projection": {
                "false_conclusion": "R4 is in range because only unequal eigenspaces are tested",
                "nonzero_witness": "(34816/15)*alpha^5",
                "rejected": True,
            },
            "cancel_with_c20": {
                "false_conclusion": "choose c20 to cancel the obstruction",
                "compression_free_symbols": ["alpha"],
                "rejected": True,
            },
            "extrapolate_alpha_zero_control": {
                "false_conclusion": "the alpha=0 solution closes a registered candidate",
                "registered_alpha_values": ["-1", "-1/2", "1/2", "1"],
                "rejected": True,
            },
            "promote_canonical_obstruction_to_all_lower_jet_gauges": {
                "false_conclusion": "all possible homogeneous lower-jet completions are obstructed",
                "missing_gate": (
                    "parameterize the order-one-through-three equal-eigenspace homogeneous "
                    "kernel, or prove the canonical zero-block normalization mandatory"
                ),
                "rejected": True,
            },
        },
        "claims": {
            "canonical_D4_obligation_244_classified": True,
            "canonical_D4_obligation_244_compatible": False,
            "all_12_registered_candidates_canonically_obstructed": True,
            "c20_can_remove_canonical_obstruction": False,
            "alternative_lower_jet_homogeneous_completion_ruled_out": False,
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
            "Parameterize the order-one-through-three equal-eigenspace homogeneous Sylvester "
            "kernel and test whether its induced order-four correction can cancel the rank-two "
            "zero-eigenspace witness; alternatively prove the canonical zero-block normalization "
            "is required by the target symmetrizer construction."
        ),
        "scope": (
            "This certificate classifies the exact permanent obstruction produced by the "
            "authoritative canonical D4 service. It does not infer failure for unparameterized "
            "lower-jet homogeneous completions and does not promote any tube, CK1, CK3, TC2, "
            "B7, global-H7, or lifespan claim."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_certificate(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "certificate content identity mismatch"
        )
    symbolic = document.get("exact_symbolic_certificate", {})
    zero = symbolic.get("equal_eigenspace_compressions", {}).get("zero_eigenspace", {})
    gap = symbolic.get("exact_candidate_gap", {})
    selector = document.get("selector_binding", {})
    rows = symbolic.get("candidate_classification", [])
    expected_witness_sparse = [
        {"row": row, "column": column, "value": str(sign)}
        for row, column, sign in WITNESS_POSITIONS
    ]
    expected_compression_sparse = [
        {
            "row": row,
            "column": column,
            "value": ("" if sign == 1 else "-") + "34816*alpha**5/15",
        }
        for row, column, sign in WITNESS_POSITIONS
    ]
    row_map = {row.get("candidate_id"): row for row in rows}
    candidate_rows_exact = len(row_map) == len(rows) == EXPECTED_CANDIDATES
    for candidate_id, (a10, c20) in CANDIDATE_PARAMETERS.items():
        row = row_map.get(candidate_id, {})
        scalar, absolute, compression_sha256 = SPECIALIZATION_CERTIFICATES[a10]
        candidate_rows_exact = candidate_rows_exact and row == {
            "candidate_id": candidate_id,
            "a10": a10,
            "c20": c20,
            "witness_scalar": scalar,
            "absolute_witness_entry": absolute,
            "compression_rank": 2,
            "compression_nonzero_entries": 4,
            "compression_sha256": compression_sha256,
            "compatible": False,
        }
    expected_claims = {
        "canonical_D4_obligation_244_classified": True,
        "canonical_D4_obligation_244_compatible": False,
        "all_12_registered_candidates_canonically_obstructed": True,
        "c20_can_remove_canonical_obstruction": False,
        "alternative_lower_jet_homogeneous_completion_ruled_out": False,
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
        document.get("claims") != expected_claims
        or document.get("status")
        != "pass_exact_canonical_d4_obstruction_cokernel_classification"
        or selector
        != {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "active_positions": [0, 2, 4, 15],
            "multiplicity_partition": "ABCD",
            "selector_record_sha256": (
                "337daa86bf740ae9e66dbef0829df30297c02e22b8baeb6b90328d608fa66c87"
            ),
            "obstruction_record_sha256": (
                "7c309eec9d225f4c0813f0696e9806d7e5c2c9802528ade40d1d92c5f13d4c56"
            ),
        }
        or document.get("counts")
        != {
            "selector_obligations_classified": 1,
            "directional_polarization_evaluations": 15,
            "candidate_specializations_checked": 12,
            "candidate_obstructions_certified": 12,
            "candidate_compatibilities_certified": 0,
            "nonzero_reference_eigenspace_compressions": 1,
            "compression_nonzero_entries": 4,
            "compression_generic_rank": 2,
            "negative_controls": 4,
            "inferred_passes": 0,
        }
        or zero.get("factorization") != "(34816/15)*alpha^5*W"
        or zero.get("sha256") != EXPECTED_COMPRESSION_SHA256
        or zero.get("generic_rank") != 2
        or zero.get("nonzero_entries") != 4
        or zero.get("skew_symmetric") is not True
        or zero.get("witness_matrix_W_sparse") != expected_witness_sparse
        or zero.get("compression_sparse") != expected_compression_sparse
        or symbolic.get("rhs")
        != {
            "shape": [55, 55],
            "nonzero_entries": 8,
            "free_parameters": ["alpha"],
            "sha256": EXPECTED_RHS_SHA256,
        }
        or gap.get("interval") != "[1088/15,34816/15]"
        or gap.get("zero_excluded") is not True
        or not candidate_rows_exact
        or symbolic.get("alpha_zero_control")
        != {
            "rhs_zero": True,
            "compression_zero": True,
            "sylvester_solvable": True,
            "residual_zero": True,
            "registered_candidate_has_alpha_zero": False,
        }
        or set(document.get("negative_controls", {}))
        != {
            "discard_zero_eigenspace_projection",
            "cancel_with_c20",
            "extrapolate_alpha_zero_control",
            "promote_canonical_obstruction_to_all_lower_jet_gauges",
        }
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4ObstructionCokernelCertificateError(
            "certificate exact/fail-closed contract mismatch"
        )


def run_certificate(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_certificate(project_root, config_path)
    validate_certificate(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify the exact D4 cokernel obstruction.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_certificate(args.project_root, args.config, args.output)
    print(json.dumps({
        "status": artifact["status"],
        "content_sha256": artifact["content_sha256"],
        "candidate_obstructions": artifact["counts"]["candidate_obstructions_certified"],
        "next_gate": artifact["next_gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
