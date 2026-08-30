from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_quadrature_scalar_cherenkov_obstruction as obstruction,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / obstruction.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_derivation_is_exact() -> None:
    checks, expressions = obstruction.symbolic_checks()
    assert len(checks) == 22
    assert all(row["passed"] for row in checks)
    assert expressions["massless_source_charge"] == "2*E"
    assert expressions["transverse_speed_squared"] == "(2*s - 1)/(2*(s - 1))"
    assert expressions["kinetic_factor_parameterization"] == "2/(q*(q + 2))"


def test_numeric_cases_preserve_open_and_closed_regions() -> None:
    rows = obstruction.numeric_cases(_config())
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert [row["open_emission_region"] for row in rows] == [True, True, False, True]
    assert all(not row["exact_longitudinal_open_emission"] for row in rows)


def test_config_and_predecessors_are_exact() -> None:
    config = obstruction.load_config(ROOT)
    rows = obstruction.validate_predecessors(ROOT, config["predecessor_bindings"])
    assert [row["binding_id"] for row in rows] == [
        "quadrature_universal_vector_metric",
        "quadrature_combined_tetrad_hyperbolicity",
    ]
    assert all(row["valid"] for row in rows)
    assert sum(row["artifact_count"] for row in rows) == 8


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("frozen_branch_contract", "dispersion", "forged"),
        ("universal_metric_source_contract", "ultrarelativistic_limit", "forged"),
        ("anisotropic_cherenkov_contract", "existence_criterion", "forged"),
        ("kinetic_normalization_contract", "alpha_limit", "forged"),
        ("obstruction_contract", "rate", "forged"),
        ("adjudication", "radiation_rate_derived", True),
        ("claim_boundary", "observational_scalar_cherenkov_exclusion_established", True),
    ],
)
def test_nested_scientific_mutations_fail_closed(
    section: str, key: str, replacement: object
) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = replacement
    with pytest.raises(obstruction.QuadratureScalarCherenkovError):
        obstruction.validate_config(config)


def test_primary_source_and_machine_mutations_fail_closed() -> None:
    config = copy.deepcopy(_config())
    config["primary_source_context"][0]["url"] = "https://example.invalid/forged"
    with pytest.raises(obstruction.QuadratureScalarCherenkovError):
        obstruction.validate_config(config)

    config = copy.deepcopy(_config())
    config["machine_check_contract"]["numeric_cases"][0]["expected_open_emission"] = False
    with pytest.raises(obstruction.QuadratureScalarCherenkovError):
        obstruction.validate_config(config)


def test_primary_sources_are_version_pinned_and_claim_is_narrow() -> None:
    config = _config()
    assert [row["url"] for row in config["primary_source_context"]] == [
        "https://arxiv.org/abs/hep-ph/0106220v2",
        "https://arxiv.org/abs/1508.07007v1",
    ]
    assert config["adjudication"]["anisotropic_cherenkov_phase_space_derived"] is True
    assert config["adjudication"]["radiation_rate_derived"] is False
    assert config["adjudication"]["cosmic_ray_survival_test_passed"] is False
    assert config["claim_boundary"]["observational_support"] is False


def test_zero_access_contract_is_literal_zero() -> None:
    config = _config()
    assert config["zero_access_and_compute"]
    assert set(config["zero_access_and_compute"].values()) == {0}


def test_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / obstruction.OUTPUT_PATH).read_text(encoding="utf-8"))
    expected = obstruction.build_receipt(ROOT)
    assert stored == expected
    obstruction.validate_receipt(stored, ROOT)
    assert stored["counts"]["symbolic_checks_passed"] == 22
    assert stored["counts"]["numeric_cases_passed"] == 4
    assert stored["counts"]["observational_rows_opened"] == 0


def test_receipt_tamper_is_rejected() -> None:
    stored = json.loads((ROOT / obstruction.OUTPUT_PATH).read_text(encoding="utf-8"))
    stored["adjudication"]["all_mode_cherenkov_safety"] = True
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    stored["content_sha256"] = obstruction._sha(body)
    with pytest.raises(obstruction.QuadratureScalarCherenkovError):
        obstruction.validate_receipt(stored, ROOT)


def test_atomic_writer_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "receipt.json"
    output.write_bytes(b"do-not-replace")
    with pytest.raises(obstruction.QuadratureScalarCherenkovError):
        obstruction._atomic_no_clobber(output, b"different")
    assert output.read_bytes() == b"do-not-replace"


def test_second_write_is_identical() -> None:
    before = (ROOT / obstruction.OUTPUT_PATH).read_bytes()
    path = obstruction.write_receipt(ROOT)
    assert path.read_bytes() == before


def test_builder_has_no_network_or_observational_loader() -> None:
    source = Path(obstruction.__file__).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "pandas" not in source
    assert "astropy" not in source
    assert 'observational_rows_opened": 0' in source
