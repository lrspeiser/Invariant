from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    quartic_fitted_output_connection_action_jet_nonidentifiability_gate as gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / gate.CONFIG_PATH
OUTPUT = ROOT / gate.OUTPUT_PATH


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = _sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_checked_in_artifact_matches_exact_rebuild() -> None:
    expected = gate.build_gate(CONFIG)
    actual = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert actual == expected
    gate._validate_result(actual, root=ROOT)


def test_exact_null_polynomial_value_first_and_second_jet_certificate() -> None:
    result = gate.build_gate(CONFIG)
    certificate = result["null_polynomial_certificate"]

    assert certificate == {
        "expanded_null_polynomial": "g^4-5/4*g^2+1/4",
        "first_derivative": "4*g^3-5/2*g",
        "second_derivative": "12*g^2-5/2",
        "grid_samples": [
            {
                "G4_X": "-1",
                "null_value": "0",
                "first_jet_lambda_coefficient": "-3/2",
                "second_jet_lambda_coefficient": "19/2",
            },
            {
                "G4_X": "-1/2",
                "null_value": "0",
                "first_jet_lambda_coefficient": "3/4",
                "second_jet_lambda_coefficient": "1/2",
            },
            {
                "G4_X": "1/2",
                "null_value": "0",
                "first_jet_lambda_coefficient": "-3/4",
                "second_jet_lambda_coefficient": "1/2",
            },
            {
                "G4_X": "1",
                "null_value": "0",
                "first_jet_lambda_coefficient": "3/2",
                "second_jet_lambda_coefficient": "19/2",
            },
        ],
    }


def test_all_22_coordinates_receive_independent_exact_ambiguity_parameters() -> None:
    result = gate.build_gate(CONFIG)
    rows = result["coordinate_ambiguity_records"]

    assert len(rows) == 22
    assert [row["coordinate_ordinal"] for row in rows] == list(range(22))
    assert (
        len({(row["direction"], row["output_row"], row["input_row"], row["atom"]) for row in rows})
        == 22
    )
    assert all(f"lambda_{index}" in row["extension_family"] for index, row in enumerate(rows))
    assert all(row["registered_value_equalities"] == 4 for row in rows)
    assert all(row["first_jet_ambiguities"] == 4 for row in rows)
    assert all(row["second_jet_ambiguities"] == 4 for row in rows)
    assert all(row["jet_identified"] is False for row in rows)


def test_counts_and_candidate_disposition_remain_fail_closed() -> None:
    result = gate.build_gate(CONFIG)

    assert result["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert result["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    assert result["gate_counts"] == {
        "selected": 12,
        "fitted_connection_coordinates": 22,
        "registered_G4_X_grid_points": 4,
        "registered_value_equalities_replayed": 88,
        "independent_ambiguity_parameters": 22,
        "nonidentified_first_jet_samples": 88,
        "nonidentified_second_jet_samples": 88,
        "registered_covariant_derivation_functors": 0,
        "registered_corrected_second_source_jet_entries": 0,
        "cross_slice_D2F_entries_admitted": 0,
        "complete_ordered_D2F_tensors_registered": 0,
        "full_high_atom_good_unknown_identities_proved": 0,
        "global_H7_closures": 0,
        "nonlinear_PDE_closures": 0,
        "lifespans_proved": 0,
    }


def test_global_claims_and_unsafe_controls_are_closed() -> None:
    result = gate.build_gate(CONFIG)
    claims = result["claim_seals"]

    for key in (
        "covariant_output_connection_derivation_registered",
        "corrected_second_source_jet_registered",
        "cross_slice_D2F_entries_admitted",
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert claims[key] is False
    assert all(control["rejected"] is True for control in result["exact_controls"].values())
    assert (
        "not a no-go theorem for a covariant action derivation"
        in result["jet_nonidentifiability_theorem"]["boundary"]
    )


def test_local_and_predecessor_file_bindings_are_live() -> None:
    result = gate.build_gate(CONFIG)
    bindings = result["source_bindings"]

    for label in ("source", "config", "test"):
        binding = bindings[label]
        assert (
            binding["file_sha256"]
            == hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
        )
    for binding in bindings["predecessor"].values():
        assert (
            binding["file_sha256"]
            == hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
        )


def test_unknown_top_level_key_and_resealed_count_tamper_fail() -> None:
    value = gate.build_gate(CONFIG)
    value["unknown"] = True
    _reseal(value)
    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["gate_counts"]["nonidentified_first_jet_samples"] = 87
    _reseal(value)
    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("claim_seals", "corrected_second_source_jet_registered"), True),
        (("claim_seals", "candidate_theory_rejected"), True),
        (("downstream_admission_counts", "pass"), 1),
        (("decision_counts", "reject"), 1),
    ],
)
def test_resealed_claim_and_disposition_tampers_fail(
    path: tuple[str, str], replacement: object
) -> None:
    value = gate.build_gate(CONFIG)
    value[path[0]][path[1]] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("registered_value_equalities", 3),
        ("first_jet_ambiguities", 0),
        ("second_jet_ambiguities", 0),
        ("jet_identified", True),
        ("extension_family", "beta*g"),
    ],
)
def test_resealed_coordinate_certificate_tampers_fail(field: str, replacement: object) -> None:
    value = gate.build_gate(CONFIG)
    value["coordinate_ambiguity_records"][0][field] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


def test_resealed_null_polynomial_and_jet_sample_tampers_fail() -> None:
    value = gate.build_gate(CONFIG)
    value["null_polynomial_certificate"]["expanded_null_polynomial"] = "0"
    _reseal(value)
    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["null_polynomial_certificate"]["grid_samples"][0]["second_jet_lambda_coefficient"] = "0"
    _reseal(value)
    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


def test_config_unknown_key_and_predecessor_binding_tamper_fail(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["unknown"] = True
    with pytest.raises(ValueError, match="config boundary changed"):
        gate._validate_config(config)

    value = gate.build_gate(CONFIG)
    tampered = copy.deepcopy(value)
    tampered["source_bindings"]["predecessor"]["artifact"]["file_sha256"] = "0" * 64
    _reseal(tampered)
    with pytest.raises(ValueError, match="predecessor binding changed"):
        gate._validate_result(tampered, root=ROOT)


def test_content_hash_and_local_binding_tamper_fail() -> None:
    value = gate.build_gate(CONFIG)
    value["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    _reseal(value)
    with pytest.raises(ValueError, match="local source binding changed"):
        gate._validate_result(value, root=ROOT)
