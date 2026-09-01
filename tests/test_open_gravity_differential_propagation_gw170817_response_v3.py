from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_response_v2 as frozen,
)
from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_response_v3 as response,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return response.load_config()


def _copy(path: Path, base: Path) -> None:
    target = base / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / path, target)


def _copy_freeze_package(tmp_path: Path) -> Path:
    config = json.loads((ROOT / response.CONFIG_PATH).read_text(encoding="utf-8"))
    for path in (response.CONFIG_PATH, response.MODULE_PATH, response.TEST_PATH):
        _copy(path, tmp_path)
    predecessor = config["predecessor_v2"]
    for role in ("config", "module", "test", "prediction"):
        _copy(Path(predecessor[f"{role}_path"]), tmp_path)
    v2_config = json.loads((ROOT / Path(predecessor["config_path"])).read_text(encoding="utf-8"))
    for predecessor_row in v2_config["predecessors"].values():
        for role in ("config", "module", "test", "receipt"):
            _copy(Path(predecessor_row[f"{role}_path"]), tmp_path)
    return tmp_path


def test_v3_is_append_only_dq_cadence_repair() -> None:
    config = _config()
    assert config["repair"] == {
        "v2_failure": "DQ sample count changed",
        "observed_hdf_fact": (
            "strain is 4096 Hz with 16777216 samples; simple DQmask is 1 Hz "
            "with 4096 samples and Xspacing=1.0 s"
        ),
        "strain_rate_hz": 4096,
        "dq_rate_hz": 1,
        "dq_sample_count": 4096,
        "dq_slice_rule": (
            "index DQmask by integer GPS seconds independently of strain indices; "
            "a strain interval is admissible only when every covered DQ second passes "
            "DATA and CBC_CAT1"
        ),
        "scientific_formula_or_grid_changed": False,
    }
    assert config["claim_boundary"]["v2_preserved"] is True
    assert config["claim_boundary"]["theorem_unchanged"] is True
    assert config["claim_boundary"]["empirical_support_claim"] is False


def test_science_sections_and_target_free_controls_are_exact_v2_replays() -> None:
    config = _config()
    science = response._science_config(config, ROOT)
    assert (
        response._sha256_bytes(response._canonical(science["preprocessing"]))
        == (config["science_freeze"]["preprocessing_sha256"])
    )
    assert (
        response._sha256_bytes(response._canonical(science["gr_control"]))
        == (config["science_freeze"]["gr_control_sha256"])
    )
    assert (
        response._sha256_bytes(response._canonical(science["response_likelihood"]))
        == config["science_freeze"]["response_likelihood_sha256"]
    )
    replay = frozen.target_free_controls(science)
    assert replay["control_sha256"] == config["predecessor_v2"]["target_free_control_sha256"]
    assert all(row["passed"] for row in replay["zero_noise"])


def test_dq_slicing_is_one_hz_and_independent_of_strain_indices() -> None:
    dq = np.arange(4096, dtype=np.uint32)
    observed = response._slice_dq(dq, 1187006834, 1187008756, 128)
    assert observed.shape == (128,)
    assert observed[0] == 1922
    assert observed[-1] == 2049
    with pytest.raises(response.GW170817ResponseV3Error, match="outside payload"):
        response._slice_dq(dq, 1187006834, 1187006800, 128)


def test_v3_prediction_freezes_repair_code_and_replayed_controls(tmp_path: Path) -> None:
    base = _copy_freeze_package(tmp_path)
    assert response.freeze(base) == "CREATED"
    assert response.freeze(base) == "EXISTING_IDENTICAL"
    config = response.load_config(base)
    prediction = response.validate_prediction(config, base)
    assert prediction["status"] == (
        "FROZEN_AFTER_HEADER_ONLY_FAILURE_BEFORE_ANY_STRAIN_OR_DQ_VALUE_ACCESS"
    )
    assert prediction["repair"]["scientific_formula_or_grid_changed"] is False
    assert (
        prediction["target_free_controls"]["control_sha256"]
        == (config["predecessor_v2"]["target_free_control_sha256"])
    )
    assert prediction["access_at_freeze"]["dq_values_read"] == 0
    assert prediction["access_at_freeze"]["strain_values_read"] == 0


def test_mutations_fail_closed() -> None:
    config = _config()

    wrong_cadence = copy.deepcopy(config)
    wrong_cadence["repair"]["dq_rate_hz"] = 4096
    with pytest.raises(response.GW170817ResponseV3Error, match="DQ cadence repair"):
        response.validate_config(wrong_cadence)

    changed_science = copy.deepcopy(config)
    changed_science["science_freeze"]["gr_control_sha256"] = "0" * 64
    with pytest.raises(response.GW170817ResponseV3Error, match="frozen science section"):
        response.validate_config(changed_science, ROOT)

    hidden_access = copy.deepcopy(config)
    hidden_access["access_before_v3_freeze"]["dq_values_read"] = 1
    with pytest.raises(response.GW170817ResponseV3Error, match="access ledger"):
        response.validate_config(hidden_access)


def test_real_data_claims_remain_separate_and_narrow() -> None:
    boundary = _config()["claim_boundary"]
    assert boundary["source_result_separate"] is True
    assert boundary["real_data_result_separate"] is True
    assert boundary["fundamental_parameter_posterior"] is False
    assert boundary["publication_ready"] is False
