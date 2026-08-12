from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_full_tensor_good_unknown_reconciliation_gate import (
    EXPECTED_CLAIM_SEALS,
    EXPECTED_COUNTS,
    EXPECTED_DATA_SEALS,
    FIRST_BLOCKER,
    _slice_entries,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_full_tensor_good_unknown_reconciliation_gate.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-full-tensor-good-unknown-reconciliation-gate/"
    "campaign.json"
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
    return build_gate(CONFIG)


def test_exact_rebuild_counts_and_first_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision"] == "representative_slice_cancelled_full_D2_identity_blocked"
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert artifact["gate_counts"] == EXPECTED_COUNTS
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_all_four_source_entries_cancel_for_every_candidate(
    rebuilt: dict[str, object],
) -> None:
    records = rebuilt["candidate_records"]
    assert len(records) == 12
    for record in records:
        source = record["two_channel_modified_slice"]["source_entries"]
        correction = record["two_channel_modified_slice"]["correction_entries"]
        assert source == [
            [row["row"], row["column"], row["value"]]
            for row in _slice_entries(record["a10"])
        ]
        assert len(source) == 4
        assert record["two_channel_modified_slice"]["source_matrix_rank"] == 2
        assert record["two_channel_modified_slice"]["residual_entries"] == []
        assert all(left[:2] == right[:2] for left, right in zip(source, correction, strict=True))
        assert record["two_channel_modified_slice"]["all_four_entries_cancelled"]


def test_D1_coverage_does_not_promote_to_complete_D2(
    rebuilt: dict[str, object],
) -> None:
    theorem = rebuilt["coverage_theorem"]
    assert theorem["registered_D1_shape"] == [11, 153]
    assert theorem["registered_D1_entries_per_candidate"] == 11 * 153
    assert theorem["closed_world_ordered_D2_shape"] == [11, 153, 153]
    assert theorem["closed_world_ordered_D2_entries_per_candidate"] == 11 * 153 * 153
    assert theorem["complete_ordered_D2_entries_registered"] == 0
    assert theorem["representative_exact_slice"]["residual_entries"] == 0
    lineage = theorem["topology_lineage_audit"]
    assert lineage["current_dyadic_binding_matches"] is False
    assert lineage["topology_snapshot_dyadic_content_sha256"] != (
        lineage["current_finite_gate_dyadic_content_sha256"]
    )
    assert "do not determine" in theorem["non_promotion"]


def test_unmodified_no_go_and_modified_slice_are_not_conflated(
    rebuilt: dict[str, object],
) -> None:
    for record in rebuilt["candidate_records"]:
        assert record["unmodified_slice"]["direct_Hs_estimate_closed"] is False
        assert record["two_channel_modified_slice"]["reference_s01_H01_slice_removed"]
        coverage = record["coverage_boundary"]
        assert coverage["complete_ordered_D2_manifest_registered"] is False
        assert coverage["all_99_second_atom_families_cancelled"] is False
        assert coverage["all_induced_terms_bounded"] is False
        assert record["candidate_decision"] == "blocked"
        assert record["candidate_rejection_authorized"] is False


def test_negative_controls_and_seals_fail_closed(rebuilt: dict[str, object]) -> None:
    assert all(control["rejected"] for control in rebuilt["exact_controls"].values())
    assert rebuilt["claim_seals"] == EXPECTED_CLAIM_SEALS
    assert rebuilt["data_seals"] == EXPECTED_DATA_SEALS
    assert not any(rebuilt["claim_seals"].values())
    assert not any(rebuilt["data_seals"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        "predecessor_hash",
        "candidate_pass",
        "open_H7",
        "promote_D2",
        "change_residual",
        "remove_blocker",
        "extra_key",
        "source_hash",
    ],
)
def test_resealed_semantic_tampering_rejects(
    rebuilt: dict[str, object], mutation: str
) -> None:
    value = copy.deepcopy(rebuilt)
    if mutation == "predecessor_hash":
        value["source_bindings"]["full_source_jacobian"]["content_sha256"] = "0" * 64
    elif mutation == "candidate_pass":
        value["candidate_records"][0]["candidate_decision"] = "pass"
    elif mutation == "open_H7":
        value["claim_seals"]["global_H7_energy_closed"] = True
    elif mutation == "promote_D2":
        value["coverage_theorem"]["complete_ordered_D2_entries_registered"] = 257499
    elif mutation == "change_residual":
        value["candidate_records"][0]["two_channel_modified_slice"][
            "residual_entries"
        ] = [{"row": 0, "column": 10, "value": "1"}]
    elif mutation == "remove_blocker":
        value["secondary_blockers"].pop()
    elif mutation == "extra_key":
        value["global_H7_pass"] = True
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    _reseal(value)
    with pytest.raises(ValueError, match="full-tensor"):
        _validate_result(value)


def test_config_and_local_source_bindings_are_exact(
    rebuilt: dict[str, object],
) -> None:
    config = _load(CONFIG)
    _validate_config(config)
    changed = copy.deepcopy(config)
    changed["policies"]["full_D2_promotion"] = "pass"
    with pytest.raises(ValueError, match="config boundary changed"):
        _validate_config(changed)

    for label in ("config", "source", "test"):
        binding = rebuilt["source_bindings"][label]
        assert not Path(binding["path"]).is_absolute()
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]


def test_source_has_no_runtime_data_or_gpu_surface() -> None:
    source = (
        ROOT
        / "src/sigma_theory_compiler/quartic_full_tensor_good_unknown_reconciliation_gate.py"
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
