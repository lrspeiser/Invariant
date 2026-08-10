import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_high_atom_d2_good_unknown_campaign import (
    generic_high_atom_d2_good_unknown_control,
    run_quartic_high_atom_d2_good_unknown_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
    RUNS / "quartic-paradifferential-good-unknown-campaign" / "campaign.json",
    RUNS / "quartic-h7-resonant-remedy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_high_atom_d2_good_unknown_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-high-atom-d2-good-unknown-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_implicit_d2_identity_reference_binding_and_negatives_are_exact() -> None:
    passed, control = generic_high_atom_d2_good_unknown_control()
    assert passed
    assert control["known_answer"]["residual_zero"]
    assert control["coordinate_to_covariant_binding"]["residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_representative_d2_obstruction_is_exact_and_global_h7_stays_open() -> None:
    result = run_quartic_high_atom_d2_good_unknown_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_exact_representative_D2_obstructions_"
        "named_good_unknown_cancellation_refuted_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "representative_high_atom_D2_contractions_materialized": 12,
        "arithmetic_DAG_direct_matches": 12,
        "named_good_unknown_cancellations_proved": 0,
        "named_good_unknown_cancellations_refuted": 12,
        "nonzero_obstructions": 12,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert result["actual_reference_audit"]["A_reference_determinant"] == "6561/4096"
    assert result["actual_reference_audit"]["nonzero_entry_count"] == 4
    assert result["representative_D2_arithmetic_packet"]["arithmetic_dag"][
        "content_sha256"
    ]
    first = result["certificates"][0]
    assert first["representative_slice"]["component_D2_value"] in {"-1", "1", "-2", "2"}
    assert first["named_good_unknown_comparison"][
        "cancellation_refuted_for_this_slice"
    ]
    assert not first["connection_to_B7_global_H7"]["B7_fully_replaced"]
    assert result == _load(ARTIFACT)


def test_hash_tamper_and_false_global_promotion_contracts_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[1]))
    corrupt["upstream_sha256"]["principal_source"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_high_atom_d2_good_unknown_campaign(
        inputs[0], corrupt, *inputs[2:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    wrong_atom = dict(config)
    wrong_atom["representative_high_atom"] = "s02[10]"
    result = run_quartic_high_atom_d2_good_unknown_campaign(*inputs, wrong_atom)
    assert result["status"] == "reject"
    assert "unsupported D2 audit contract" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_high_atom_d2_good_unknown_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
