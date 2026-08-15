from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_independent_g2_alternative_k55_recurrence import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    IndependentG2RecurrenceError,
    _content_hash,
    _validate_contract,
    _with_hash,
    build_campaign,
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
    _validate_contract(checked)


def test_all_fifteen_broader_transport_systems_close(rebuilt: dict[str, object]) -> None:
    records = rebuilt["evaluation_records"]
    assert len(records) == 15
    assert rebuilt["all_15_broader_transport_systems_pass"]
    assert rebuilt["first_exact_broader_transport_incompatibility"] is None
    assert rebuilt["remaining_unaudited_evaluations"] == []
    assert all(record["compatible"] for record in records)
    assert all(
        record["canonical_route_coefficient_rank"] == record["canonical_route_augmented_rank"]
        for record in records
    )
    assert all(record["companion_Taylor_remainder_entries"] == [0, 0, 0] for record in records)


def test_subset_02_is_constructed_by_independent_equal_G2(rebuilt: dict[str, object]) -> None:
    record = rebuilt["evaluation_records"][5]
    assert record["evaluation_id"] == "subset_02"
    assert record["pulled_back_residual_nonzero_entries"] == 80
    assert record["canonical_route_coefficient_rank"] == 4
    assert record["canonical_route_augmented_rank"] == 4
    assert record["canonical_G1_nonzero_spectral_coefficients"] == []
    assert record["canonical_equal_G2_nonzero_spectral_coefficients"] == [
        {
            "eigenvalue": "1",
            "spectral_row": 0,
            "spectral_column": 2,
            "coefficient": "22528*sqrt(2)/140625",
        },
        {
            "eigenvalue": "-1",
            "spectral_row": 3,
            "spectral_column": 5,
            "coefficient": "-22528*sqrt(2)/140625",
        },
    ]


def test_subset_2_preserves_prior_G1_witness(rebuilt: dict[str, object]) -> None:
    record = rebuilt["evaluation_records"][2]
    assert record["evaluation_id"] == "subset_2"
    assert record["canonical_G1_nonzero_spectral_coefficients"] == [
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
    assert record["canonical_equal_G2_nonzero_spectral_coefficients"] == []


def test_each_canonical_matrix_is_sealed(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["evaluation_records"]:
        for key in ("delta_G1", "delta_G2", "delta_G3"):
            packet = record[key]
            assert packet["content_sha256"] == _content_hash(packet)
            assert packet["nonzero_entries"] == len(packet["entries"])


def test_first_exact_broader_no_go_is_55_state_lift(rebuilt: dict[str, object]) -> None:
    assert rebuilt["lift_records"] == [
        {
            "evaluation_id": "subset_0",
            "K55_symmetry_remainder_entries": [0, 0, 0],
            "K55_symmetrizer_remainder_entries": [0, 0, 0],
            "pass": True,
        },
        {
            "evaluation_id": "subset_1",
            "K55_symmetry_remainder_entries": [0, 0, 0],
            "K55_symmetrizer_remainder_entries": [0, 0, 0],
            "pass": True,
        },
        {
            "evaluation_id": "subset_2",
            "K55_symmetry_remainder_entries": [0, 0, 0],
            "K55_symmetrizer_remainder_entries": [0, 0, 72],
            "pass": False,
        },
    ]
    assert rebuilt["first_exact_55_state_lift_failure"] == {
        "evaluation_id": "subset_2",
        "K55_symmetry_remainder_entries": [0, 0, 0],
        "K55_symmetrizer_remainder_entries": [0, 0, 72],
        "first_missing_primitive": (
            "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
        ),
    }
    assert not rebuilt["all_15_exact_55_state_lifts_pass"]


def test_positive_tube_is_not_attempted_after_lift_failure(rebuilt: dict[str, object]) -> None:
    assert rebuilt["positive_tube_records"] == []
    assert not rebuilt["positive_tube_proved"]
    assert rebuilt["counts"]["positive_tubes_proved"] == 0


def test_atomic_manifest_remains_154_of_304(rebuilt: dict[str, object]) -> None:
    assert rebuilt["decision"] == "BLOCK_SERIALIZATION"
    assert rebuilt["counts"] == {
        "required_evaluations": 15,
        "transport_evaluations_audited": 15,
        "transport_evaluations_solved": 15,
        "55_state_lifts_passed": 2,
        "positive_tubes_proved": 0,
        "manifest_registered_before": 154,
        "manifest_registered_after": 154,
        "registered_packets": 0,
        "remaining_packets": 150,
    }
    assert rebuilt["atomic_admission_contract"]["partial_manifest_advance_forbidden"]
    assert not rebuilt["atomic_admission_contract"]["satisfied"]


def test_sealed_action_and_P55_authorities_are_not_mutated(rebuilt: dict[str, object]) -> None:
    assert not rebuilt["preserved_authorities"]["P55_or_action_mutated"]
    assert rebuilt["claims"] == {
        "all_15_broader_transports_proved": True,
        "all_15_exact_55_state_lifts_proved": False,
        "positive_symmetrizer_tube_proved": False,
        "higher_K55_registered": False,
        "manifest_advanced": False,
        "missing_packets_inferred_as_zero": False,
        "global_H7_claim": False,
    }


def test_all_declared_negative_controls_are_executed(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["negative_controls"]
    assert len(controls) == 6
    assert all(value == {"rejected": True} for value in controls.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_transport",
        "break_subset_02_rank",
        "erase_subset_02_G2",
        "tamper_matrix",
        "erase_lift_failure",
        "erase_72_remainder",
        "claim_all_lifts",
        "claim_positive_tube",
        "advance_manifest",
        "register_packets",
        "mutate_action",
        "infer_zero",
        "claim_H7",
    ],
)
def test_resealed_campaign_tampering_fails_closed(
    rebuilt: dict[str, object], mutation: str
) -> None:
    changed = copy.deepcopy(rebuilt)
    if mutation == "drop_transport":
        changed["evaluation_records"].pop()
    elif mutation == "break_subset_02_rank":
        changed["evaluation_records"][5]["canonical_route_augmented_rank"] = 5
    elif mutation == "erase_subset_02_G2":
        changed["evaluation_records"][5]["canonical_equal_G2_nonzero_spectral_coefficients"] = []
    elif mutation == "tamper_matrix":
        packet = changed["evaluation_records"][5]["delta_G2"]
        packet["entries"].pop()
        changed["evaluation_records"][5]["delta_G2"] = _with_hash(packet)
    elif mutation == "erase_lift_failure":
        changed["first_exact_55_state_lift_failure"] = None
    elif mutation == "erase_72_remainder":
        changed["lift_records"][2]["K55_symmetrizer_remainder_entries"] = [0, 0, 0]
    elif mutation == "claim_all_lifts":
        changed["all_15_exact_55_state_lifts_pass"] = True
    elif mutation == "claim_positive_tube":
        changed["positive_tube_proved"] = True
    elif mutation == "advance_manifest":
        changed["counts"]["manifest_registered_after"] = 304
    elif mutation == "register_packets":
        changed["counts"]["registered_packets"] = 150
    elif mutation == "mutate_action":
        changed["preserved_authorities"]["P55_or_action_mutated"] = True
    elif mutation == "infer_zero":
        changed["claims"]["missing_packets_inferred_as_zero"] = True
    else:
        changed["claims"]["global_H7_claim"] = True
    changed = _with_hash(changed)
    with pytest.raises(IndependentG2RecurrenceError):
        _validate_contract(changed)


def test_config_tamper_rejects_before_exact_solve(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["manifest"]["registered_after_on_block"] = 304
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(IndependentG2RecurrenceError, match="configuration seal changed"):
        build_campaign(ROOT, changed)


def test_local_bindings_are_path_free_and_crlf_stable(
    rebuilt: dict[str, object], tmp_path: Path
) -> None:
    for binding in rebuilt["local_bindings"].values():
        assert not Path(binding["path"]).is_absolute()
        path = ROOT / binding["path"]
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert hashlib.sha256(normalized.encode()).hexdigest() == binding["normalized_text_sha256"]
        materialized = tmp_path / Path(binding["path"]).name
        materialized.write_bytes(normalized.replace("\n", "\r\n").encode())
        replay = materialized.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert hashlib.sha256(replay.encode()).hexdigest() == binding["normalized_text_sha256"]


def test_source_has_no_network_gpu_or_subprocess_surface() -> None:
    source = (ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "socket", "subprocess", "multiprocessing", "cuda"):
        assert token not in source
    assert TEST_PATH in source
