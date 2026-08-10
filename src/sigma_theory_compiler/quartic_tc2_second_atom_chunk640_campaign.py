from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quartic_tc2_second_atom_continuation_engine import (
    generic_contiguous_boundary_control,
    run_second_atom_continuation,
)

SCHEMA_VERSION = "sigma-quartic-tc2-second-atom-chunk640-campaign-1.0"
CHUNK_OFFSET = 640
EXPECTED_PRIOR_STATUS = (
    "pass_cumulative_640_second_atom_pairs_no_obstruction_remaining_fail_closed"
)
SUCCESS_STATUS = (
    "pass_cumulative_704_second_atom_pairs_no_obstruction_remaining_fail_closed"
)
OBSTRUCTION_STATUS = (
    "exact_second_atom_Sylvester_obstruction_found_in_chunk640_global_H7_fail_closed"
)


def generic_second_atom_chunk640_boundary_control() -> tuple[bool, dict[str, Any]]:
    return generic_contiguous_boundary_control(CHUNK_OFFSET)


def run_quartic_tc2_second_atom_chunk640_campaign(
    prior_chunk_campaign: dict[str, Any],
    variable_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return run_second_atom_continuation(
        prior_chunk_campaign,
        variable_campaign,
        config,
        schema_version=SCHEMA_VERSION,
        chunk_offset=CHUNK_OFFSET,
        expected_prior_status=EXPECTED_PRIOR_STATUS,
        success_status=SUCCESS_STATUS,
        obstruction_status=OBSTRUCTION_STATUS,
    )


def write_quartic_tc2_second_atom_chunk640_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
