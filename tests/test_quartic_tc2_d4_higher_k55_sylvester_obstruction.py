from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_d4_higher_k55_sylvester_obstruction import (
    CONFIG_PATH,
    OUTPUT_PATH,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_equal_eigenspace_obstruction_blocks_k55() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(campaign, ROOT)
    counts = campaign["counts"]
    assert counts["input_residual_nonzero_polynomial_entries"] == 120
    assert counts["equal_plus_one_projection_nonzero_polynomial_entries"] == 192
    assert counts["equal_minus_one_projection_nonzero_polynomial_entries"] == 192
    assert counts["other_equal_eigenspace_projection_nonzero_polynomial_entries"] == 0
    assert counts["unequal_eigenspace_forcing_nonzero_polynomial_entries"] == 0
    assert counts["canonical_Sylvester_correction_nonzero_polynomial_entries"] == 0
    assert counts["manifest_registered_after"] == 154


def test_all_seven_equal_eigenspaces_are_serialized() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    packets = campaign["projected_equal_eigenspace_packets"]
    assert [row["eigenvalue"] for row in packets] == ["0", "1", "-1", "1/2", "-1/2", "1/3", "-1/3"]
    assert [row["nonzero_polynomial_entries"] for row in packets] == [0, 192, 192, 0, 0, 0, 0]
    assert all("content_sha256" in row["packet"] for row in packets)


def test_obstruction_replay_is_deterministic() -> None:
    assert build_campaign(ROOT, ROOT / CONFIG_PATH) == build_campaign(ROOT, ROOT / CONFIG_PATH)
