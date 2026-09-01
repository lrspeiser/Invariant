from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_dynamic_source_memory_kernels_v2 as memory

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return memory.load_config()


def _driver(config: dict[str, object], driver_id: str) -> dict[str, object]:
    drivers = config["drivers"]
    assert isinstance(drivers, list)
    return next(row for row in drivers if row["id"] == driver_id)


def _copy_package(tmp_path: Path) -> Path:
    config = json.loads((ROOT / memory.CONFIG_PATH).read_text(encoding="utf-8"))
    predecessor = config["predecessor"]
    relatives = [
        memory.CONFIG_PATH,
        memory.MODULE_PATH,
        memory.TEST_PATH,
        Path(predecessor["receipt_path"]),
        Path(predecessor["config_path"]),
        Path(predecessor["module_path"]),
        Path(predecessor["test_path"]),
        Path(predecessor["source_path"]),
    ]
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_twenty_driver_programs_execute_with_dimensions_and_unique_outputs() -> None:
    config = _config()
    assert tuple(row["id"] for row in config["drivers"]) == memory.DRIVER_IDS
    executions = [memory.execute_driver(config, row) for row in config["drivers"]]
    assert len(executions) == 20
    assert len({row["program_sha256"] for row in executions}) == 20
    assert len({row["output_sha256"] for row in executions}) == 20
    assert all(row["output_unit"] == [0, 0, 0] for row in executions)
    assert all(row["declared_variables_exactly_used"] is True for row in executions)
    assert all(row["output_min"] >= -1.0 for row in executions)
    assert all(row["output_max"] <= 1.0 for row in executions)


def test_every_counted_concept_executes_driver_bytes_as_kernel_input() -> None:
    config = _config()
    legacy_config = memory._legacy_config(ROOT)
    drivers, pipelines = memory.execute_pipelines(config, legacy_config)
    assert len(drivers) == 20
    assert len(pipelines) == 120
    assert len({row["pipeline_program_sha256"] for row in pipelines}) == 120
    assert all(row["driver_output_sha256"] == row["kernel_input_sha256"] for row in pipelines)
    assert all(row["status"] == "EXECUTED_DRIVER_OUTPUT_AS_KERNEL_INPUT" for row in pipelines)
    assert config["claim_boundary"]["unique_empirical_theories_claimed"] is False


def test_valid_driver_mutation_changes_downstream_and_dimension_mutation_rejects() -> None:
    config = _config()
    legacy_config = memory._legacy_config(ROOT)
    original = _driver(config, "D01_ACC")
    baseline = memory.execute_driver(config, original)
    changed = copy.deepcopy(original)
    reference = next(row for row in changed["variables"] if row["name"] == "g_ref")
    reference["fixture"]["value"] = 1.7
    changed_execution = memory.execute_driver(config, changed)
    assert changed_execution["output_sha256"] != baseline["output_sha256"]
    kernel = memory.legacy._kernel_map(legacy_config)["K02_EXPONENTIAL"]
    baseline_response = memory.legacy.simulate_kernel(
        "K02_EXPONENTIAL", baseline["times"], baseline["output"], kernel["parameters"]
    )
    changed_response = memory.legacy.simulate_kernel(
        "K02_EXPONENTIAL",
        changed_execution["times"],
        changed_execution["output"],
        kernel["parameters"],
    )
    assert memory._array_sha256(changed_response) != memory._array_sha256(baseline_response)

    bad_dimension = copy.deepcopy(original)
    bad_reference = next(row for row in bad_dimension["variables"] if row["name"] == "g_ref")
    bad_reference["unit"] = [0, 0, 1]
    with pytest.raises(memory.DynamicMemoryV2Error, match="tanh requires dimensionless"):
        memory.execute_driver(config, bad_dimension)


def test_unused_or_response_derived_driver_inputs_fail_closed() -> None:
    config = _config()
    unused = copy.deepcopy(_driver(config, "D08_BAL"))
    extra = copy.deepcopy(unused["variables"][0])
    extra["name"] = "unused_source"
    unused["variables"].append(extra)
    with pytest.raises(memory.DynamicMemoryV2Error, match="not exactly used"):
        memory.execute_driver(config, unused)

    leaked = copy.deepcopy(_driver(config, "D08_BAL"))
    leaked["variables"][0]["role"] = "observational_response_residual"
    with pytest.raises(memory.DynamicMemoryV2Error, match="response-derived"):
        memory.execute_driver(config, leaked)


def test_computed_structural_triage_has_no_empirical_rank() -> None:
    config = _config()
    triage = memory._structural_triage(config, memory._legacy_config(ROOT))
    assert [row["structural_order"] for row in triage] == list(range(1, 7))
    assert all(row["empirical_rank"] is None for row in triage)
    assert all(row["computed_sort_key"] for row in triage)
    assert triage[-1]["kernel_id"] == "K01_RETARDED"
    assert triage[-1]["single_event_identifiable"] is False
    k02 = next(row for row in triage if row["kernel_id"] == "K02_EXPONENTIAL")
    k06 = next(row for row in triage if row["kernel_id"] == "K06_STOCHASTIC_OU")
    assert k02["conditional_mean_fingerprint_sha256"] == k06["conditional_mean_fingerprint_sha256"]
    assert k02["conditional_mean_equivalence_class_size"] == 2
    assert k06["computed_discriminator_channels"]["conditional_variance"] is True
    assert all(
        row["status"] == "NOT_EVALUATED_RESPONSE_UNOPENED"
        for row in config["future_heldout_metrics"]
    )


def test_delay_lti_source_ringdown_and_ou_countermodels_are_executed() -> None:
    config = _config()
    rows = memory._countermodel_executions(config, memory._legacy_config(ROOT))
    assert [row["id"] for row in rows] == [
        "C01_FREE_DELAY",
        "C02_SINGLE_LTI",
        "C03_DOUBLE_LTI",
        "C04_SOURCE_RINGDOWN",
        "C05_OU_NOISE",
    ]
    assert all(row["status"] == "EXECUTED_TARGET_FREE_COUNTERMODEL" for row in rows)
    assert all(len(row["program_sha256"]) == 64 for row in rows)
    ringdown = next(row for row in rows if row["id"] == "C04_SOURCE_RINGDOWN")
    assert ringdown["post_source_rms"] > 0.0


def test_gw150914_preflight_is_exact_but_source_blocked() -> None:
    config = _config()
    preflight = config["observational_preflight"]
    assert preflight["status"] == "SOURCE_BLOCKED_MISSING_PAYLOAD_HASHES_AND_SCHEMA_RECEIPTS"
    assert [row["content_length"] for row in preflight["products"]] == [1040592, 1007420]
    assert [row["etag"] for row in preflight["products"]] == [
        "fe0d0-56d5f9a35d6f9",
        "f5f3c-56d5f9a35f638",
    ]
    assert all(row["payload_sha256"] is None for row in preflight["products"])
    assert preflight["metadata_receipt"]["method"] == "HEAD_ONLY"
    assert preflight["metadata_receipt"]["response_body_bytes"] == 0
    assert set(preflight["preprocessing"]) == {
        "analysis_interval_gps",
        "psd_intervals_gps",
        "dq_rule",
        "detrend",
        "analysis_window",
        "psd",
        "band_hz",
        "fft",
        "resampling",
    }
    assert "Whittle" in preflight["likelihood"]["kind"]
    assert preflight["likelihood"]["optimizer"].startswith("exhaustive declared")
    assert preflight["likelihood"]["nuisance_grid"]["t0_seconds"]["step"] == pytest.approx(
        1.0 / 4096.0
    )
    assert preflight["likelihood"]["validation_suite"]["noise_realizations_per_cell"] == 128
    assert config["access_contract"]["observational_response_rows_read"] == 0


def test_source_ready_or_access_overclaim_rejects() -> None:
    config = _config()
    source_ready = copy.deepcopy(config)
    source_ready["observational_preflight"]["status"] = "EXECUTABLE"
    with pytest.raises(memory.DynamicMemoryV2Error, match="source gate"):
        memory.validate_config(source_ready)

    accessed = copy.deepcopy(config)
    accessed["access_contract"]["response_body_bytes"] = 1
    with pytest.raises(memory.DynamicMemoryV2Error, match="access contract"):
        memory.validate_config(accessed)


def test_v1_predecessor_is_preserved_and_bound() -> None:
    config = _config()
    predecessor = config["predecessor"]
    assert (
        memory._sha256_file(ROOT / predecessor["receipt_path"]) == predecessor["receipt_raw_sha256"]
    )
    assert (
        memory._sha256_file(ROOT / predecessor["config_path"]) == predecessor["config_raw_sha256"]
    )
    assert (
        memory._sha256_file(ROOT / predecessor["module_path"]) == predecessor["module_raw_sha256"]
    )
    assert memory._sha256_file(ROOT / predecessor["test_path"]) == predecessor["test_raw_sha256"]
    assert memory._sha256_file(ROOT / predecessor["source_path"]) == predecessor["source_sha256"]


def test_deterministic_build_check_replay_and_tamper_gate(tmp_path: Path) -> None:
    base = _copy_package(tmp_path)
    assert memory.build(base) == "CREATED"
    assert memory.check(base) == "VALID"
    assert memory.build(base) == "EXISTING_IDENTICAL"
    receipt = json.loads((base / memory.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert receipt["counts"]["executable_dimensioned_drivers"] == 20
    assert receipt["counts"]["executed_driver_kernel_pipelines"] == 120
    assert receipt["counts"]["unique_empirical_theories_claimed"] == 0
    assert receipt["counts"]["observational_response_rows"] == 0
    assert receipt["observational_preflight_status"].startswith("SOURCE_BLOCKED")
    artifact = base / memory.ARTIFACT_DIR / "driver-programs-and-executions.json"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(memory.DynamicMemoryV2Error, match="artifact differs"):
        memory.check(base)
