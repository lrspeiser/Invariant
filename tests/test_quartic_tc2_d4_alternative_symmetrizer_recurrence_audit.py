from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_alternative_symmetrizer_recurrence_audit import (
    CONFIG_PATH,
    OUTPUT_PATH,
    AlternativeSymmetrizerRecurrenceAuditError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ordered_audit_finds_only_the_alternative_recurrence_route() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(campaign, ROOT)
    audit = campaign["audit_order"]
    assert [row["alternative"] for row in audit] == [
        "cross_cluster_symmetric_transport",
        "nonsymmetric_form",
        "change_P55_or_action_authority",
        "alternative_symmetrizer_recurrence",
    ]
    assert [row["result"] for row in audit] == [
        "REJECT_LOWER_ORDER_IDENTITY",
        "REJECT_NOT_A_SYMMETRIZER",
        "REJECT_CURRENT_SOURCE_AUTHORITY",
        "PASS_WITNESS_LOCAL_CONSTRUCTION",
    ]
    for sign in audit[0]["exact_sign_results"]:
        assert sign["unrestricted_symmetric_map_rank"] == 2
        assert sign["unrestricted_augmented_rank"] == 2
        assert sign["lower_order_preserving_map_rank"] == 0
        assert sign["lower_order_preserving_augmented_rank"] == 1
    for sign in audit[1]["exact_sign_results"]:
        assert sign["lower_order_preserving_nonsymmetric_variables"] == 82
        assert sign["transport_map_rank"] == 0
        assert sign["augmented_rank"] == 1
    assert audit[2]["post_mutation_companion_residual_entries"] == 0


def test_alternative_recurrence_is_exact_but_does_not_advance_manifest() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    recurrence = campaign["exact_witness_local_recurrence"]
    assert recurrence["symmetric_equal_eigenspace_order_one_variables"] == 52
    assert recurrence["order_three_equal_block_equations"] == 30
    assert recurrence["order_three_recurrence_map_rank"] == 4
    assert recurrence["order_three_augmented_rank"] == 4
    assert recurrence["canonical_nonzero_spectral_coefficients"] == [
        {
            "eigenvalue": "1",
            "spectral_row": 0,
            "spectral_column": 2,
            "coefficient": "64/1875",
        },
        {
            "eigenvalue": "-1",
            "spectral_row": 3,
            "spectral_column": 5,
            "coefficient": "-64/1875",
        },
    ]
    assert recurrence["symmetry_remainder_entries"] == [0, 0, 0]
    assert recurrence["companion_Taylor_identity_remainder_entries"] == [0, 0, 0]
    assert campaign["counts"] == {
        "alternatives_audited_in_declared_order": 4,
        "constructive_witness_local_alternatives": 1,
        "global_coordinate_free_alternatives": 0,
        "manifest_registered_before": 154,
        "manifest_registered_after": 154,
        "remaining_packets": 150,
        "emitted_rows": 0,
        "remaining_rows": 117180,
    }
    assert campaign["decision"] == "BLOCK_SERIALIZATION"


def test_replay_and_tamper_fail_closed() -> None:
    campaign = build_campaign(ROOT, ROOT / CONFIG_PATH)
    assert campaign == build_campaign(ROOT, ROOT / CONFIG_PATH)
    tampered = json.loads(json.dumps(campaign))
    tampered["counts"]["manifest_registered_after"] = 155
    with pytest.raises(AlternativeSymmetrizerRecurrenceAuditError, match="replay mismatch"):
        validate_campaign(tampered, ROOT)
