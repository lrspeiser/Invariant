from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_all_polarization_alternative_k55_recurrence import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    AllPolarizationRecurrenceError,
    _content_hash,
    _load_bound,
    _validate_contract,
    _with_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_campaign(ROOT, CONFIG)


def test_exact_rebuild_matches_checked_campaign(rebuilt: dict[str, object]) -> None:
    checked = _load(ARTIFACT)
    assert rebuilt == checked
    validate_campaign(checked, ROOT)


def test_first_exact_incompatibility_is_subset_02(rebuilt: dict[str, object]) -> None:
    assert rebuilt["first_exact_incompatibility"] == {
        "evaluation_id": "subset_02",
        "joint_coefficient_rank": 4,
        "joint_augmented_rank": 5,
        "first_missing_primitive": "source_bound_equal_eigenspace_G1_transport_solution",
    }
    records = rebuilt["evaluation_records"]
    assert [record["evaluation_id"] for record in records] == [
        "subset_0",
        "subset_1",
        "subset_2",
        "subset_3",
        "subset_01",
        "subset_02",
    ]
    assert records[-1]["pulled_back_residual_nonzero_entries"] == 80
    assert records[-1]["order_two_admissibility_rank"] == 4
    assert records[-1]["joint_coefficient_rank"] == 4
    assert records[-1]["joint_augmented_rank"] == 5
    assert not records[-1]["compatible"]


def test_subset_2_reproduces_prior_canonical_witness(rebuilt: dict[str, object]) -> None:
    record = rebuilt["evaluation_records"][2]
    assert record["evaluation_id"] == "subset_2"
    assert record["pulled_back_residual_nonzero_entries"] == 16
    assert record["order_two_admissibility_rank"] == 0
    assert record["joint_coefficient_rank"] == record["joint_augmented_rank"] == 4
    assert record["canonical_nonzero_spectral_coefficients"] == [
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
    assert record["companion_Taylor_remainder_entries"] == [0, 0, 0]


def test_every_compatible_canonical_solution_replays_exactly(
    rebuilt: dict[str, object],
) -> None:
    records = rebuilt["evaluation_records"]
    assert [record["compatible"] for record in records] == [True] * 5 + [False]
    for record in records[:-1]:
        assert record["companion_Taylor_remainder_entries"] == [0, 0, 0]
        for key in ("delta_G1", "delta_G2", "delta_G3"):
            packet = record[key]
            assert packet["content_sha256"] == _content_hash(packet)


def test_atomic_manifest_admission_remains_closed(rebuilt: dict[str, object]) -> None:
    assert rebuilt["decision"] == "BLOCK_SERIALIZATION"
    assert rebuilt["counts"] == {
        "required_evaluations": 15,
        "evaluations_solved": 5,
        "evaluations_audited": 6,
        "55_state_lifts_passed": 0,
        "manifest_registered_before": 154,
        "manifest_registered_after": 154,
        "registered_packets": 0,
        "remaining_packets": 150,
    }
    assert rebuilt["lift_records"] == []
    assert not rebuilt["positive_tube_proved"]
    assert not any(rebuilt["claims"].values())
    assert rebuilt["atomic_admission_contract"]["partial_manifest_advance_forbidden"]
    assert not rebuilt["atomic_admission_contract"]["satisfied"]


def test_remaining_evaluations_are_explicitly_uncomputed_not_zero(
    rebuilt: dict[str, object],
) -> None:
    assert rebuilt["remaining_unaudited_evaluations"] == [
        "subset_03",
        "subset_12",
        "subset_13",
        "subset_23",
        "subset_012",
        "subset_013",
        "subset_023",
        "subset_123",
        "subset_0123",
    ]
    assert not rebuilt["claims"]["inferred_missing_packets_as_zero"]


def test_all_declared_negative_controls_are_executed(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["negative_controls"]
    assert set(controls) == {
        "drop_first_incompatibility",
        "promote_witness_direction_to_coordinate_free_packet",
        "advance_154_manifest_before_atomic_admission",
        "infer_uncomputed_packet_as_zero",
        "claim_global_H7_from_finite_direction_algebra",
    }
    assert all(value == {"rejected": True} for value in controls.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "erase_incompatibility",
        "equalize_ranks",
        "advance_manifest",
        "infer_zero",
        "promote_global",
        "claim_positive_tube",
        "drop_matrix_entry",
        "change_witness_coefficient",
        "mark_all_15",
    ],
)
def test_resealed_campaign_tampering_fails_closed(
    rebuilt: dict[str, object], mutation: str
) -> None:
    changed = copy.deepcopy(rebuilt)
    if mutation == "erase_incompatibility":
        changed["first_exact_incompatibility"] = None
    elif mutation == "equalize_ranks":
        changed["evaluation_records"][-1]["joint_augmented_rank"] = 4
    elif mutation == "advance_manifest":
        changed["counts"]["manifest_registered_after"] = 304
    elif mutation == "infer_zero":
        changed["claims"]["inferred_missing_packets_as_zero"] = True
    elif mutation == "promote_global":
        changed["claims"]["global_H7_claim"] = True
    elif mutation == "claim_positive_tube":
        changed["positive_tube_proved"] = True
    elif mutation == "drop_matrix_entry":
        changed["evaluation_records"][2]["delta_G1"]["entries"].pop()
        changed["evaluation_records"][2]["delta_G1"] = _with_hash(
            changed["evaluation_records"][2]["delta_G1"]
        )
    elif mutation == "change_witness_coefficient":
        changed["evaluation_records"][2]["canonical_nonzero_spectral_coefficients"][0][
            "coefficient"
        ] = "0"
    else:
        changed["all_15_equal_eigenspace_transport_systems_pass"] = True
    changed = _with_hash(changed)
    with pytest.raises(AllPolarizationRecurrenceError):
        _validate_contract(changed)


@pytest.mark.parametrize(
    "source_name",
    [
        "alternative_witness",
        "K55_frontier",
        "higher_P55",
        "P55_order_one",
        "higher_H_star",
        "H_star_order_one",
        "flat_P55",
        "flat_action_metric",
        "projector_recipes",
    ],
)
def test_each_upstream_semantic_corruption_rejects(tmp_path: Path, source_name: str) -> None:
    config = _load(CONFIG)
    binding = config["upstreams"][source_name]
    authority = _load(ROOT / binding["path"])
    authority["synthetic_corruption"] = source_name
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(AllPolarizationRecurrenceError):
        _load_bound(
            tmp_path.resolve(),
            {"path": "authority.json", "content_sha256": binding["content_sha256"]},
        )


def test_config_tamper_rejects_before_exact_solve(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["manifest"]["registered_after_on_block"] = 304
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(AllPolarizationRecurrenceError, match="configuration seal changed"):
        build_campaign(ROOT, changed)


def test_local_bindings_are_path_free_and_materializer_stable(
    rebuilt: dict[str, object], tmp_path: Path
) -> None:
    for binding in rebuilt["local_bindings"].values():
        assert not Path(binding["path"]).is_absolute()
        path = ROOT / binding["path"]
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert hashlib.sha256(normalized.encode()).hexdigest() == binding["normalized_text_sha256"]
        crlf = tmp_path / Path(binding["path"]).name
        crlf.write_bytes(normalized.replace("\n", "\r\n").encode())
        materialized = crlf.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert (
            hashlib.sha256(materialized.encode()).hexdigest() == binding["normalized_text_sha256"]
        )


def test_source_has_no_network_gpu_or_subprocess_surface() -> None:
    source = (ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "socket", "subprocess", "multiprocessing", "cuda"):
        assert token not in source
    assert TEST_PATH in source
