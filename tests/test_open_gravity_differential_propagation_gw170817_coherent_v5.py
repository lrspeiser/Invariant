from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v5 as module,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v5_repairs_only_preprocessing_and_keeps_holdout_sealed() -> None:
    config = module.load_config(ROOT)
    audit = module._validate_predecessor(config, ROOT)
    inheritance = module._inheritance_audit(config, ROOT)
    science = module.compose_science_config(config, ROOT)
    assert audit["status"] == "PASS_BLOCKED_V4_REPLAYED_EXACT"
    assert inheritance["status"] == "PASS_ONLY_PREPROCESSING_CHANGED"
    assert science["preprocessing"]["analysis_duration_seconds"] == 256
    assert science["freeze_boundary"]["gw190425_status"] == "SEALED_NOT_ACQUIRED_NOT_OPENED"


def test_source_schemas_are_dimensionally_bound() -> None:
    config = module.load_config(ROOT)
    science = module.compose_science_config(config, ROOT)
    audit = module._source_schema_audit(config, science, ROOT)
    assert audit["status"] == "PASS_PSD_IS_POWER_PER_HZ_AND_CALIBRATION_PHASE_IS_RADIANS"
    assert audit["published_psd"]["value_semantics"].endswith("not amplitude spectral density")


def test_hdf_headers_dq_support_and_nyquist_without_strain() -> None:
    config = module.load_config(ROOT)
    science = module.compose_science_config(config, ROOT)
    hdf = module._hdf_and_dq_audit(config, science, ROOT)
    support = module._support_and_nyquist_audit(science)
    assert hdf["status"] == "PASS_EXACT_HDF_HEADERS_AND_DQ_WITHOUT_STRAIN_VALUES"
    assert hdf["strain_values_read"] == 0
    assert hdf["dq_values_read"] == 3840
    assert support["frequency_bins_per_detector"] == 16200
    assert support["last_frequency_hz"] == 2047.875
    assert support["support_checks"] == {
        "signal_start_after_flat_start": True,
        "coalescence_before_flat_end": True,
        "post_coalescence_flat_margin": True,
    }
    assert all(support["nyquist_checks"].values())
