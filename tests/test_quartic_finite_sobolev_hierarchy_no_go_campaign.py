from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_finite_sobolev_hierarchy_no_go_campaign import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    EXPECTED_DATA_SEALS,
    FIRST_BLOCKER,
    _order_ledger,
    _validate_config,
    _validate_result,
    build_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_finite_sobolev_hierarchy_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-campaign/campaign.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    value["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_campaign(CONFIG)


def test_exact_rebuild_counts_blocker_and_artifact(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_all_finite_order_exponent_recurrence_is_exact(
    rebuilt: dict[str, object],
) -> None:
    for order in (4, 7, 8, 9, 12, 31):
        ledger = _order_ledger(order)
        assert ledger["coefficient_H_order_minus_1_exponent"] == 0
        assert ledger["high_low_output_H_order_exponent"] == 1
        assert ledger["conditional_coefficient_H_order_exponent"] == 0
        assert ledger["restarted_high_low_output_exponent"] == 1
    theorem = rebuilt["theorem"]
    assert theorem["exact_high_shell_lower_bound"].endswith(">= N*c_packet/2")
    assert "any finite integer s>=4" in theorem["conclusion"]
    assert "does not rule out a full tensor cancellation" in theorem["scope_limit"]
    controls = rebuilt["exact_controls"]["positive_exponent_replay"]
    assert controls["input_exponents_zero"]
    assert controls["output_exponents_one"]
    assert controls["restarted_output_exponents_one"]


def test_recovery_chain_supplies_no_unregistered_derivative_gain(
    rebuilt: dict[str, object],
) -> None:
    audit = rebuilt["recovery_chain_audit"]
    assert audit["anti_Wick_composition"]["spatial_derivative_gain"] == 0
    assert audit["annular_C6_constants"]["spatial_derivative_gain"] == 0
    assert audit["bounded_frequency_defect"]["high_shell_value_on_witness"] == 0
    assert audit["dyadic_localization"]["witness_growth_exponent"] == 1
    controls = rebuilt["exact_controls"]
    for label in (
        "promote_conditional_H8_to_autonomous_H8",
        "promote_annular_C6_to_spatial_smoothing",
        "use_compact_defect_to_cancel_high_shell",
        "erase_candidate_D2_coupling",
        "promote_slice_no_go_to_all_modified_energies",
    ):
        assert controls[label]["rejected"] is True


def test_candidate_D2_multipliers_are_nonzero_and_decisions_remain_blocked(
    rebuilt: dict[str, object],
) -> None:
    records = rebuilt["candidate_records"]
    assert len(records) == 12
    assert {row["representative_D2_value"] for row in records} == {
        "-2",
        "-1",
        "1",
        "2",
    }
    assert {row["absolute_growth_multiplier"] for row in records} == {"1", "2"}
    assert all(row["decision"] == "blocked" for row in records)
    assert all(row["candidate_rejection_authorized"] is False for row in records)
    assert all(row["full_tensor_cancellation_proved"] is False for row in records)


@pytest.mark.parametrize(
    "mutation",
    [
        "predecessor_hash",
        "candidate_pass",
        "open_claim",
        "promote_H8",
        "erase_secondary_blocker",
        "extra_key",
        "local_source_hash",
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    rebuilt: dict[str, object], mutation: str
) -> None:
    value = copy.deepcopy(rebuilt)
    if mutation == "predecessor_hash":
        value["source_bindings"]["dyadic_localization"]["content_sha256"] = "0" * 64
    elif mutation == "candidate_pass":
        value["candidate_records"][0]["decision"] = "pass"
    elif mutation == "open_claim":
        value["claim_seals"]["autonomous_H8_energy_closed"] = True
    elif mutation == "promote_H8":
        value["exact_controls"]["promote_conditional_H8_to_autonomous_H8"][
            "rejected"
        ] = False
    elif mutation == "erase_secondary_blocker":
        value["secondary_blockers"].pop()
    elif mutation == "extra_key":
        value["invented_global_H7_pass"] = True
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    _reseal(value)
    with pytest.raises(ValueError, match="finite-Sobolev"):
        _validate_result(value)


def test_config_and_source_bindings_are_closed_world(
    rebuilt: dict[str, object],
) -> None:
    config = _load(CONFIG)
    _validate_config(config)
    changed = copy.deepcopy(config)
    changed["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary changed"):
        _validate_config(changed)

    bindings = rebuilt["source_bindings"]
    for label in ("config", "source", "test"):
        assert not Path(bindings[label]["path"]).is_absolute()
        assert _file_sha(ROOT / bindings[label]["path"]) == bindings[label]["file_sha256"]
    assert rebuilt["claim_seals"] == EXPECTED_CLAIM_SEALS
    assert rebuilt["data_seals"] == EXPECTED_DATA_SEALS
    assert not any(rebuilt["claim_seals"].values())
    assert not any(rebuilt["data_seals"].values())


def test_source_has_no_runtime_data_or_gpu_surface() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/quartic_finite_sobolev_hierarchy_no_go_campaign.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "cupy",
        "torch",
        "os.kill",
        "popen",
    ):
        assert forbidden not in lowered
