from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_d4_physical_metric_transport_no_go import (
    CONFIG_PATH,
    OUTPUT_PATH,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_rational_direction_proves_metric_transport_no_go() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(campaign, ROOT)
    witness = campaign["exact_unit_direction_witness"]
    assert witness["direction"] == ["3/5", "4/5", "0"]
    assert witness["unit_sphere_residual"] == "(3/5)^2+(4/5)^2-1=0"
    for sign in witness["physical_sign_witnesses"]:
        assert sign["physical_diagonal_companion1_nonzero_entries"] == 0
        assert sign["symmetric_metric_domain_dimension"] == 253
        assert sign["transport_linear_map_rank"] == 0
        assert sign["projected_target_nonzero_entries"] == 32
        assert sign["augmented_rank"] == 1
        assert sign["consistent"] is False
        assert sign["left_nullspace_coordinate_witness"] == {
            "row": 4,
            "column": 10,
            "linear_map_value_for_every_symmetric_metric": "0",
            "required_target_value": "4096/46875",
        }


def test_no_go_keeps_every_downstream_claim_false() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert campaign["counts"]["manifest_registered_after"] == 154
    assert campaign["counts"]["emitted_output_rows"] == 0
    assert all(value is False for value in campaign["claims"].values())


def test_no_go_replay_is_deterministic() -> None:
    assert build_campaign(ROOT, ROOT / CONFIG_PATH) == build_campaign(ROOT, ROOT / CONFIG_PATH)
