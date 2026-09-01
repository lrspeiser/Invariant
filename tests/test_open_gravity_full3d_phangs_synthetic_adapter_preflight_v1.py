from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.open_gravity_full3d_phangs_synthetic_adapter_preflight_v1 import (
    _MECHANISM_ENTRYPOINTS,
    LEDGER_PATH,
    PACKETS_PATH,
    RECEIPT_PATH,
    derive_release,
    lane6_newton_adapter,
    load_config,
    validate_config,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import array_sha256
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release() -> tuple[dict, bytes, bytes]:
    return derive_release()


def test_three_primary_sources_and_nine_common_abi_adapters_are_complete(release) -> None:
    receipt, packets, ledger = release
    assert receipt["object_count"] == receipt["primary_source_cell_count"] == 3
    assert receipt["grid_shape"] == [17, 17, 17]
    assert receipt["mechanism_count"] == 9
    assert receipt["adapter_executions"] == 27
    assert receipt["replay_entry_count"] == 54
    assert receipt["target_free_confusion_cells"] == 243
    assert receipt["target_free_self_injections_recovered"] == 27
    assert receipt["target_free_self_injections_total"] == 27
    assert receipt["predecessor_source_gate_failures_retained"] == 1
    assert set(receipt["mechanism_ids"]) == set(_MECHANISM_ENTRYPOINTS)
    assert hashlib.sha256(packets).hexdigest() == receipt["packets_jsonl_sha256"]
    assert hashlib.sha256(ledger).hexdigest() == receipt["ledger_json_sha256"]


def test_every_source_value_prediction_and_confusion_cell_is_typed_and_hashed(release) -> None:
    _, packets, _ = release
    rows = [json.loads(line) for line in packets.decode("utf-8").splitlines()]
    assert len(rows) == 3
    prediction_count = 0
    confusion_count = 0
    for row in rows:
        scenario = row["scenario"]
        assert scenario["geometry_mode"] == "nonspherical3d"
        assert scenario["expected_predictions"][0]["shape"] == [17, 17, 17, 3]
        assert scenario["expected_predictions"][0]["axes"] == ["x", "y", "z", "component"]
        assert scenario["expected_predictions"][0]["unit"] == "m s^-2"
        references = {
            reference["element_id"]: reference
            for partition in ("formula_features", "scoring_responses", "hidden_truth")
            for reference in scenario[partition]
        }
        for element_id, reference in references.items():
            value = np.asarray(row["values"][element_id], dtype=reference["dtype"])
            assert list(value.shape) == reference["shape"]
            assert array_sha256(value) == reference["value_sha256"]
        uncertainty = scenario["uncertainties"][0]
        variance = np.asarray(
            row["values"]["uncertainty.synthetic-acceleration-variance"], dtype=np.float64
        )
        assert array_sha256(variance) == uncertainty["artifact_sha256"]
        assert len(row["predictions"]) == 9
        for prediction in row["predictions"]:
            artifact = prediction["artifact"]
            value = np.asarray(prediction["value"], dtype=artifact["dtype"])
            assert artifact["axes"] == ["x", "y", "z", "component"]
            assert artifact["unit"] == "m s^-2"
            assert artifact["frame"] == "solver-source"
            assert list(value.shape) == artifact["shape"] == [17, 17, 17, 3]
            assert array_sha256(value) == artifact["value_sha256"]
            assert prediction["output_sha256"] == canonical_sha256(
                {"prediction.vector.acceleration": artifact}
            )
            prediction_count += 1
        for truth in row["target_free_confusion"]:
            assert truth["self_recovered"]
            assert truth["truth_id"] in truth["winner_ids"]
            assert len(truth["candidate_distances"]) == 9
            confusion_count += len(truth["candidate_distances"])
    assert prediction_count == 27
    assert confusion_count == 243


def test_exact_source_provenance_and_response_blind_claim_ceiling(release) -> None:
    receipt, _, _ = release
    bound_paths = {binding["path"] for binding in receipt["source_bindings"]}
    assert len(bound_paths) == 18
    assert {
        "src/sigma_theory_compiler/open_gravity_lane6_same_grid_nonspherical_predictions_v1.py",
        "src/sigma_theory_compiler/open_gravity_phangs_things_model_lifted_3d_source_builder_v1.py",
        "src/sigma_theory_compiler/open_gravity_phangs_things_full3d_source_systematics_v1.py",
        "src/sigma_theory_compiler/open_gravity_phangs_things_full3d_solver_bridge_v1.py",
        "src/sigma_theory_compiler/open_gravity_3d_newton_aqual_qumond_baselines_v1.py",
        "src/sigma_theory_compiler/open_gravity_disk_polar_escape_load_v1.py",
    } <= bound_paths
    assert receipt["raw_fits_inventory"]["count"] == 21
    assert receipt["raw_fits_inventory"]["bytes"] == 74_030_400
    assert len(receipt["raw_fits_inventory"]["verified_rows"]) == 21
    assert receipt["source_density_hashes"] == {
        "NGC2903": "0c79affca436ec4ca6d95e87ad773c3ee59e117e580d44f95c718283047280c8",
        "NGC3351": "62a8f04cf1b6cd3d8ba921cde061726b511211152ecbbfa2fcc94bdaaa821255",
        "NGC3627": "6c51d479c992ef6c4fa1da05b4e17ca10199a5360ee2233a3fa98c91cb309574",
    }
    access = receipt["access_accounting"]
    assert access["scientific_response_files_opened"] == 0
    assert access["scientific_response_rows_opened"] == 0
    assert access["lensing_response_files_opened"] == 0
    assert access["real_scores_computed"] == 0
    assert not access["response_calibrated"]

    config = load_config()
    forged = {**config, "claim_class": "EMPIRICAL_SUPPORT"}
    with pytest.raises(SchemaViolation, match="claim ceiling"):
        validate_config(forged, verify_sources=False)

    altered_dpel = {
        **config,
        "dpel01_frozen_adapter": {
            **config["dpel01_frozen_adapter"],
            "steps": 101,
        },
    }
    with pytest.raises(SchemaViolation, match="frozen adapter parameters"):
        validate_config(altered_dpel, verify_sources=False)

    with pytest.raises(SchemaViolation, match="no free parameters"):
        lane6_newton_adapter({}, {"undeclared": 1.0})


def test_stored_release_is_exactly_the_fresh_deterministic_derivation(release) -> None:
    receipt, packets, ledger = release
    assert (ROOT / PACKETS_PATH).read_bytes() == packets
    assert (ROOT / LEDGER_PATH).read_bytes() == ledger
    assert json.loads((ROOT / RECEIPT_PATH).read_text()) == receipt
