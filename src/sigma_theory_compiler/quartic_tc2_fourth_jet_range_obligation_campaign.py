from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations_with_replacement
from math import comb, factorial
from pathlib import Path
from typing import Any

from .quartic_tc2_diagonal_third_jet_campaign import _active_directions, _content_hash
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)

SCHEMA = "sigma-quartic-tc2-fourth-jet-range-obligation-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-fourth-jet-range-obligation-config-1.0"
ACTIVE_DIMENSION = 15
JET_ORDER = 4
EXPECTED_SELECTOR_COUNT = 3060
EXPECTED_CANDIDATE_COUNT = 12


class QuarticTC2FourthJetRangeObligationCampaignError(ValueError):
    """Raised when the fourth-jet range-obligation contract is not exact."""


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _partition_name(indices: tuple[int, ...]) -> str:
    multiplicities = sorted(Counter(indices).values(), reverse=True)
    names = {
        (4,): "AAAA",
        (3, 1): "AAAB",
        (2, 2): "AABB",
        (2, 1, 1): "AABC",
        (1, 1, 1, 1): "ABCD",
    }
    try:
        return names[tuple(multiplicities)]
    except KeyError as exc:
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "unexpected fourth-order multiplicity partition"
        ) from exc


def _selector_seed(
    checkpoint: dict[str, Any], reduction: dict[str, Any], candidates: dict[str, Any]
) -> str:
    return _content_hash(
        {
            "completed_third_jet_checkpoint": checkpoint["content_sha256"],
            "reranked_reduction": reduction["content_sha256"],
            "candidate_source": candidates["content_sha256"],
            "active_dimension": ACTIVE_DIMENSION,
            "jet_order": JET_ORDER,
        }
    )


def _fourth_selector(
    active_positions: list[int],
    basis_directions: list[dict[str, Any]],
    seed: str,
) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    prior = seed
    active_names = [
        f"basis_{index}:atom_{direction['atom_index']}"
        for index, direction in enumerate(basis_directions)
    ]
    for offset, indices in enumerate(
        combinations_with_replacement(range(ACTIVE_DIMENSION), JET_ORDER)
    ):
        counts = Counter(indices)
        multi_index = [counts.get(index, 0) for index in range(ACTIVE_DIMENSION)]
        alpha_factorial = 1
        for multiplicity in multi_index:
            alpha_factorial *= factorial(multiplicity)
        body = {
            "selector_offset": offset,
            "active_indices": list(indices),
            "active_positions": [active_positions[index] for index in indices],
            "basis_labels": [active_names[index] for index in indices],
            "atom_indices": [
                int(basis_directions[index]["atom_index"]) for index in indices
            ],
            "multi_index": multi_index,
            "alpha_factorial": alpha_factorial,
            "multiplicity_partition": _partition_name(indices),
            "normalization": "Y^alpha/alpha! so D^alpha at zero equals one",
            "prior_record_sha256": prior,
        }
        record = {**body, "record_sha256": _content_hash(body)}
        records.append(record)
        prior = record["record_sha256"]
    return records, prior


def _claims() -> dict[str, bool]:
    return {
        "reference_mixed_third_jet_closed": True,
        "fourth_jet_minimal_selector_constructed": True,
        "fourth_jet_range_compatibility_closed": False,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }


def _validate_upstream(
    checkpoint: dict[str, Any], reduction: dict[str, Any], candidates: dict[str, Any]
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    expected_checkpoint_claims = {
        "full_mixed_sector_closed": True,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }
    reranking = reduction.get("exact_reranking", {})
    certificates = candidates.get("certificates", [])
    if (
        checkpoint.get("completed_chunks") != 7
        or checkpoint.get("next_obligation_offset") != 447
        or checkpoint.get("remaining_obligations") != 0
        or checkpoint.get("permanently_stopped") is not False
        or checkpoint.get("claims") != expected_checkpoint_claims
        or reranking.get("active_direction_rank") != ACTIVE_DIMENSION
        or reranking.get("symmetric_cubic_dimension") != 680
        or reranking.get("completion_rank") != 680
        or len(reranking.get("basis_active_positions", [])) != ACTIVE_DIMENSION
        or len(reranking.get("coordinate_names", [])) != 16
        or len(certificates) != EXPECTED_CANDIDATE_COUNT
        or candidates.get("counts", {}).get("selected") != EXPECTED_CANDIDATE_COUNT
        or candidates.get("counts", {}).get("first_order_variable_extensions")
        != EXPECTED_CANDIDATE_COUNT
        or any(certificate.get("first_obstruction") is not None for certificate in certificates)
    ):
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "upstream third-jet/candidate contract mismatch"
        )
    return (
        list(reranking["basis_active_positions"]),
        list(reranking["coordinate_names"]),
        certificates,
    )


def build_fourth_jet_range_obligation_campaign(
    project_root: Path, config_path: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if config.get("schema_version") != CONFIG_SCHEMA or not _hash_matches(config):
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "config hash/schema mismatch"
        )
    if (
        config.get("expected_active_dimension") != ACTIVE_DIMENSION
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or config.get("expected_fourth_selector_count") != EXPECTED_SELECTOR_COUNT
        or config.get("fourth_jet_policy")
        != "construct_exact_minimal_selector_no_inferred_passes"
        or config.get("full_tube_policy") != "fail_closed"
    ):
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "unsupported campaign contract"
        )
    checkpoint = _load_bound(project_root, config["completed_third_jet_checkpoint"])
    reduction = _load_bound(project_root, config["reranked_reduction"])
    candidates = _load_bound(project_root, config["candidate_source"])
    active_positions, coordinate_names, certificates = _validate_upstream(
        checkpoint, reduction, candidates
    )
    directions = _active_directions()
    basis_directions = [directions[position] for position in active_positions]
    basis_direction_ledger = [
        {
            "basis_index": index,
            "active_position": active_positions[index],
            "atom_index": int(direction["atom_index"]),
            "atom": str(direction["atom"]),
            "direction": {
                name: str(value)
                for name, value in sorted(direction["direction"].items())
            },
        }
        for index, direction in enumerate(basis_directions)
    ]
    seed = _selector_seed(checkpoint, reduction, candidates)
    selector, tip = _fourth_selector(active_positions, basis_directions, seed)
    partition_counts = Counter(
        record["multiplicity_partition"] for record in selector
    )
    expected_partition_counts = {
        "AAAA": 15,
        "AAAB": 210,
        "AABB": 105,
        "AABC": 1365,
        "ABCD": 1365,
    }
    if (
        len(selector) != EXPECTED_SELECTOR_COUNT
        or comb(ACTIVE_DIMENSION + JET_ORDER - 1, JET_ORDER)
        != EXPECTED_SELECTOR_COUNT
        or dict(sorted(partition_counts.items()))
        != dict(sorted(expected_partition_counts.items()))
        or len({record["record_sha256"] for record in selector})
        != EXPECTED_SELECTOR_COUNT
    ):
        raise QuarticTC2FourthJetRangeObligationCampaignError(
            "fourth-jet selector construction mismatch"
        )
    candidate_ledger = [
        {
            "candidate_id": certificate["candidate_id"],
            "coefficients": certificate["coefficients"],
            "reference_first_order_extension_closed": True,
            "reference_mixed_third_jet_closed": True,
            "fourth_jet_obligations": EXPECTED_SELECTOR_COUNT,
            "fourth_jet_obligations_evaluated": 0,
            "fourth_jet_obligations_passed": 0,
            "fourth_jet_range_compatibility_closed": False,
        }
        for certificate in certificates
    ]
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_fourth_jet_minimal_selector_manifest_no_evaluations_tube_fail_closed",
        "config_sha256": config["content_sha256"],
        "upstream_sha256": {
            "completed_third_jet_checkpoint": checkpoint["content_sha256"],
            "reranked_reduction": reduction["content_sha256"],
            "candidate_source": candidates["content_sha256"],
        },
        "claim": (
            "The completed third-jet evidence fixes a 15-dimensional active sector. "
            "Without an additional symmetry or structural identity, all 3060 normalized "
            "Sym^4 monomials are necessary fourth-order cokernel obligations."
        ),
        "theorem": {
            "name": "finite-jet insufficiency and worst-case minimal fourth selector",
            "active_dimension": ACTIVE_DIMENSION,
            "jet_order": JET_ORDER,
            "selector_dimension_formula": "binomial(15+4-1,4)=3060",
            "proof": (
                "For every omitted multi-index alpha, add "
                "epsilon*(Y^alpha/alpha!)*c to the Sylvester right-hand side, with "
                "nonzero c in the reference cokernel. This perturbation has identical "
                "jets through order three and zero fourth derivatives on every retained "
                "selector entry, while D^alpha equals epsilon*c and is obstructed."
            ),
            "consequence": (
                "No positive tube radius follows from the completed third jet alone, and "
                "no fourth-order selector entry may be inferred from the others absent a "
                "new exact symmetry theorem."
            ),
        },
        "exact_remainder_constants": {
            "fourth_order_operator_norm_taylor_factor": "1/24",
            "active_coordinate_count_fourth_power": 50625,
            "coordinatewise_normalized_derivative_sum_factor": "16875/8",
            "identity": "sum_{|alpha|=4} 1/alpha! = 15^4/4! = 16875/8",
            "range_residual_bound": "||Q R_4(Y)|| <= M4*||Y||^4/24",
            "bound_is_not_vanishing": True,
        },
        "selector": {
            "basis": "normalized monomials Y^alpha/alpha! in Sym^4(R^15)",
            "active_positions": active_positions,
            "ambient_coordinate_names": coordinate_names,
            "basis_directions": basis_direction_ledger,
            "seed_sha256": seed,
            "tip_sha256": tip,
            "partition_counts": expected_partition_counts,
            "records": selector,
        },
        "candidate_ledger": candidate_ledger,
        "negative_controls": {
            "hidden_fourth_order_cokernel_perturbation": {
                "perturbation": "epsilon*(Y_0^4/4!)*c, epsilon != 0, c in coker(L0)",
                "derivatives_at_zero_orders_0_through_3": ["0", "0", "0", "0"],
                "normalized_fourth_derivative": "epsilon*c != 0",
                "same_completed_third_jet_evidence": True,
                "tube_solution_for_any_positive_radius": False,
                "rejected": True,
            },
            "single_omitted_selector_entry": {
                "omitted_entry_witnesses": EXPECTED_SELECTOR_COUNT,
                "retained_fourth_derivatives_zero": True,
                "omitted_normalized_fourth_derivative": "epsilon*c != 0",
                "proper_subset_sufficient_without_new_symmetry": False,
                "rejected": True,
            },
            "small_remainder_bound_as_identity": {
                "false_implication": "M4*r^4/24 small implies Q R_4 identically zero",
                "counterexample_exists_for_every_r_positive": True,
                "rejected": True,
            },
        },
        "counts": {
            "candidates": EXPECTED_CANDIDATE_COUNT,
            "active_directions": ACTIVE_DIMENSION,
            "fourth_selector_records": EXPECTED_SELECTOR_COUNT,
            "candidate_fourth_jet_obligations": (
                EXPECTED_CANDIDATE_COUNT * EXPECTED_SELECTOR_COUNT
            ),
            "fourth_jet_obligations_evaluated": 0,
            "fourth_jet_obligations_passed": 0,
            "fourth_jet_obligations_inferred_passed": 0,
            "negative_controls": 3,
        },
        "claims": _claims(),
        "next_gate": {
            "exact_obligation": (
                "Generate D4K55, D4P55, and D4TC2 on the 3060-record normalized selector; "
                "project each right-hand side to the equal-eigenspace cokernel for all 12 "
                "candidates and stop on the first exact obstruction."
            ),
            "after_D4": (
                "A tube theorem still requires either an exact structural range identity "
                "or an all-order/analytic remainder theorem; a finite M4 bound alone is "
                "not range compatibility."
            ),
        },
        "scope": (
            "This campaign constructs and proves worst-case minimality of the next exact "
            "fourth-jet selector. It evaluates no fourth derivative and does not promote "
            "full tube, CK1, CK3, TC2, B7, global H7, or lifespan."
        ),
        "errors": [],
    }
    return _with_hash(body)


def run_fourth_jet_range_obligation_campaign(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    artifact = build_fourth_jet_range_obligation_campaign(project_root, config_path)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact minimal fourth-jet Sylvester range selector."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_fourth_jet_range_obligation_campaign(
        args.project_root, args.config, args.output
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "selector_records": artifact["counts"]["fourth_selector_records"],
                "candidate_obligations": artifact["counts"][
                    "candidate_fourth_jet_obligations"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
