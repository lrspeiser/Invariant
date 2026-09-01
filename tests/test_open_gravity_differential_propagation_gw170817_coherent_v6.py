from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v6 as module,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v6_labels_v5_failure_and_separates_gates() -> None:
    config = module.load_config(ROOT)
    assert config["strict_audit_of_v5"]["label"] == ("FAIL_OPTIMIZER_AND_REGISTERED_POWER_GATE")
    assert "optimizer_recovery_gate" in config
    assert "registered_branch_power_gate" in config
    assert "reservoir_power_calibration" in config
    assert config["freeze_boundary"]["gw190425_status"] == ("SEALED_NOT_ACQUIRED_NOT_OPENED")
    assert config["freeze_boundary"]["real_response_authorized"] is False


def test_sample_count_and_parent_alias_fail_closed() -> None:
    config = module.load_config(ROOT)
    mutated = copy.deepcopy(config)
    mutated["sample_count_hardening"]["analysis_sample_count"] += 1
    with pytest.raises(module.CoherentV6Error, match="sample count mutation"):
        module.validate_config(mutated)
    mutated = copy.deepcopy(config)
    mutated["package"]["module_path"] = "src/../bad.py"
    with pytest.raises(module.CoherentV6Error, match="parent alias"):
        module.validate_config(mutated)


def test_projected_snr_is_data_projection_not_template_norm() -> None:
    result = module._projected_snr_from_components(
        {"H1": 12.0, "L1": 20.0, "V1": -1.0},
        {"H1": 9.0, "L1": 16.0, "V1": 1.0},
    )
    assert result["per_detector"] == {"H1": 4.0, "L1": 5.0, "V1": -1.0}
    assert result["coherent_network"] == pytest.approx(31.0 / (26.0**0.5))
    assert result["coherent_network"] != pytest.approx(26.0**0.5)


def test_predecessor_source_runtime_and_schema_audits_without_strain() -> None:
    pytest.importorskip("lal", reason="exact LALSuite runtime is isolated under WSL")
    config = module.load_config(ROOT)
    science, audit = module._static_rebuild(config, ROOT)
    assert science["preprocessing"]["analysis_sample_count"] == 1048576
    assert audit["predecessors"]["status"] == "PASS_V4_V5_EXACT_AND_REPLAYED"
    assert audit["runtime"]["status"] == "PASS_WHEEL_AND_INSTALLED_RUNTIME_BYTES"
    assert audit["hdf_and_dq"]["strain_values_read"] == 0
    assert audit["hdf_and_dq"]["dq_values_read"] == 3840
