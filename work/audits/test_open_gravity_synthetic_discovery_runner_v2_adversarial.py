"""Independent adversarial audit of frozen generic discovery runner v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 as adapter_registry
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import AdapterRegistration
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v2 import (
    run_discovery_matrix_v2,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import DiscoveryStatus
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    UncertaintyRef,
    array_sha256,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "runner_v2_audit_fixtures",
    ROOT / "tests/test_open_gravity_synthetic_discovery_runner_v1.py",
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture)


def _comparison(
    uncertainty_id: str = "response.diagonal-covariance",
) -> tuple[ObservableComparison, ...]:
    return (
        ObservableComparison(
            "prediction.vector.acceleration",
            "response.vector.acceleration",
            uncertainty_id,
        ),
    )


def _parameter(cell_id: str, scale: int) -> ParameterCell:
    return ParameterCell(
        cell_id,
        {"scale_denominator": 1, "scale_numerator": scale},
    )


def _default_bindings():
    comparator = fixture._binding(
        "binding.comparator.v1", "scale-comparator", BindingStatus.EXECUTABLE
    )
    control = fixture._binding(
        "binding.control.v1", "scale-control", BindingStatus.EXECUTABLE
    )
    return comparator, control


def _invoke(
    *,
    scenarios=None,
    scenario_values=None,
    truths=None,
    bindings=None,
    parameter_cells=None,
    comparisons=None,
    release=None,
    distinct_gap: float = 0.1,
    ledger_id: str = "gravity.synthetic.runner.v2.independent-audit",
):
    if scenarios is None:
        scenario, values = fixture._scenario_and_values()
        scenarios = (scenario,)
        scenario_values = {scenario.scenario_id: values}
    else:
        scenarios = tuple(scenarios)
        assert scenario_values is not None
    if bindings is None:
        bindings = _default_bindings()
    bindings = tuple(bindings)
    if parameter_cells is None:
        comparator, control = bindings
        parameter_cells = {
            comparator.binding_id: (_parameter("scale.one", 1),),
            control.binding_id: (_parameter("scale.two", 2),),
        }
    if truths is None:
        truths = {scenario.scenario_id: "scale-control" for scenario in scenarios}
    if comparisons is None:
        comparisons = {scenario.scenario_id: _comparison() for scenario in scenarios}
    executable = tuple(row for row in bindings if row.status is BindingStatus.EXECUTABLE)
    return run_discovery_matrix_v2(
        catalogue=fixture._catalogue(),
        release=release or fixture._release(),
        scenarios=scenarios,
        scenario_values=scenario_values,
        truth_formula_by_scenario=truths,
        bindings=bindings,
        adapters=tuple(
            AdapterRegistration.create(f"adapter.audit.{index}.v2", binding)
            for index, binding in enumerate(executable)
        ),
        parameter_cells=parameter_cells,
        comparisons=comparisons,
        distinct_gap=distinct_gap,
        ledger_id=ledger_id,
    )


def _scenario_with_uncertainty(uncertainty: np.ndarray, uncertainty_id: str):
    scenario, values = fixture._scenario_and_values()
    reference = UncertaintyRef(
        uncertainty_id,
        "response.vector.acceleration",
        "covariance" if uncertainty.ndim == 2 and uncertainty.shape == (6, 6) else "diagonal-covariance",
        f"uncertainty/{uncertainty_id}.npy",
        array_sha256(uncertainty),
    )
    scenario = replace(scenario, uncertainties=(reference,))
    values = ScenarioRuntimeValues(
        values.formula_values,
        values.response_values,
        values.truth_values,
        {uncertainty_id: uncertainty},
    )
    return scenario, values


def _contains_native_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_native_float(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_native_float(item) for item in value)
    return False


def test_frozen_subject_receipt_and_predecessor_bindings_are_exact() -> None:
    config_path = ROOT / "configs/open_gravity_synthetic_discovery_runner_v2.json"
    receipt_path = ROOT / "runs/gravity/open-gravity-synthetic-discovery-runner-v2/receipt.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == (
        "f09ee8bb3e85fb7bc7081028b142483e86284ea34ed4ccf31b0e9bd5ba4304c0"
    )
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        "850ba4f2e6c2268a97c485deb013f351ab22ccd49dcb85f9d34abf770ed9adea"
    )
    content_sha256 = receipt.pop("content_sha256")
    assert canonical_sha256(receipt) == content_sha256 == (
        "6d1d8095c7b58b521d441e2951e18cd1c42a6c0992f90a80a0c27b258237fdc1"
    )
    for group, pairs in (
        (config["subject"], (("module_path", "module_sha256"), ("test_path", "test_sha256"))),
        (
            config["predecessor_v1"],
            (
                ("module_path", "module_sha256"),
                ("test_path", "test_sha256"),
                ("audit_path", "audit_sha256"),
            ),
        ),
    ):
        for path_key, hash_key in pairs:
            assert hashlib.sha256((ROOT / group[path_key]).read_bytes()).hexdigest() == group[hash_key]


def test_two_scenario_cartesian_matrix_and_exact_eligible_ledger_identity() -> None:
    first, first_values = fixture._scenario_and_values()
    second_id = "galaxy.runner.fixture.v2"
    second_object = "synthetic.object.002"
    second = replace(
        first,
        scenario_id=second_id,
        object_id=second_object,
        seed_lineage=replace(
            first.seed_lineage,
            scenario_id=second_id,
            object_id=second_object,
            nuisance_draw=1,
        ),
    )
    comparator, control = _default_bindings()
    result = _invoke(
        scenarios=(first, second),
        scenario_values={first.scenario_id: first_values, second.scenario_id: first_values},
        bindings=(comparator, control),
        parameter_cells={
            comparator.binding_id: (
                _parameter("scale.one", 1),
                _parameter("scale.three", 3),
            ),
            control.binding_id: (_parameter("scale.two", 2),),
        },
    )
    expected_cells = {
        (scenario.scenario_id, binding.binding_id, cell.parameter_cell_id)
        for scenario in (first, second)
        for binding, cells in (
            (comparator, (_parameter("scale.one", 1), _parameter("scale.three", 3))),
            (control, (_parameter("scale.two", 2),)),
        )
        for cell in cells
    }
    assert result.scenario_count == 2
    assert result.attempted_cell_count == 6
    assert result.scored_cell_count == 6
    assert {(row.scenario_id, row.binding_id, row.parameter_cell_id) for row in result.cells} == expected_cells
    assert len(result.ledger.entries) == 12
    assert len({row.ledger_entry_sha256 for row in result.cells}) == 6
    by_hash = {entry.entry_sha256: entry for entry in result.ledger.entries}
    binding_by_id = {row.binding_id: row for row in (comparator, control)}
    for cell in result.cells:
        completed = by_hash[cell.ledger_entry_sha256]
        eligible = result.ledger.entries[completed.sequence - 1]
        assert completed.parameter_cell_id == cell.parameter_cell_id
        assert completed.scenario_id == cell.scenario_id
        assert completed.binding_sha256 == binding_by_id[cell.binding_id].content_sha256
        assert completed.prior_entry_sha256 == eligible.entry_sha256
        assert eligible.status is DiscoveryStatus.ELIGIBLE_NOT_RUN
        assert eligible.binding_sha256 == completed.binding_sha256


def test_exact_joint_tie_and_single_candidate_are_nondistinct() -> None:
    comparator, control = _default_bindings()
    tied = _invoke(
        bindings=(comparator, control),
        parameter_cells={
            comparator.binding_id: (_parameter("scale.comparator", 2),),
            control.binding_id: (_parameter("scale.control", 2),),
        },
        distinct_gap=0.0,
    )
    assert sum(cell.winner for cell in tied.cells) == 2
    assert tied.truth_recovery_count == 1
    assert tied.distinct_truth_recovery_count == 0
    assert {cell.discovery_status for cell in tied.cells} == {DiscoveryStatus.UNDERPOWERED.value}

    single = _invoke(
        bindings=(control,),
        parameter_cells={control.binding_id: (_parameter("scale.control", 2),)},
        distinct_gap=0.0,
    )
    assert single.truth_recovery_count == 1
    assert single.distinct_truth_recovery_count == 0
    assert single.cells[0].winner is True
    assert single.cells[0].distinct is False
    assert single.cells[0].discovery_status == DiscoveryStatus.UNDERPOWERED.value


def test_multi_parameter_ineligible_cells_and_nonexecutables_have_no_pseudoscore() -> None:
    comparator, control = _default_bindings()
    incompatible = replace(
        comparator,
        binding_id="binding.incompatible.v1",
        formula_id="incompatible-law",
        geometry_support=("spherical1d",),
    )
    theory = fixture._binding("binding.theory.v1", "theory-law", BindingStatus.THEORY_ONLY)
    bindings = (control, incompatible, theory)
    result = _invoke(
        bindings=bindings,
        parameter_cells={
            control.binding_id: (_parameter("scale.truth", 2),),
            incompatible.binding_id: (
                _parameter("scale.one", 1),
                _parameter("scale.two", 2),
            ),
            theory.binding_id: (),
        },
    )
    assert result.attempted_cell_count == 4
    assert result.scored_cell_count == 1
    blocked = [cell for cell in result.cells if cell.binding_id == incompatible.binding_id]
    assert {cell.parameter_cell_id for cell in blocked} == {"scale.one", "scale.two"}
    assert len({cell.ledger_entry_sha256 for cell in blocked}) == 2
    assert all(cell.whitened_rmse is None and cell.result_sha256 is None for cell in blocked)
    theory_cell = next(cell for cell in result.cells if cell.binding_id == theory.binding_id)
    assert theory_cell.parameter_cell_id is None
    assert theory_cell.whitened_rmse is None
    assert theory_cell.result_sha256 is None
    assert theory_cell.winner is False
    entry_by_hash = {entry.entry_sha256: entry for entry in result.ledger.entries}
    assert all(entry_by_hash[cell.ledger_entry_sha256].formula_id == cell.formula_id for cell in result.cells)
    assert len({cell.ledger_entry_sha256 for cell in result.cells}) == len(result.cells)


def test_unregistered_truth_and_response_calibrated_release_fail_closed() -> None:
    scenario, values = fixture._scenario_and_values()
    comparator, control = _default_bindings()
    common = {
        "scenarios": (scenario,),
        "scenario_values": {scenario.scenario_id: values},
        "bindings": (comparator, control),
        "parameter_cells": {
            comparator.binding_id: (_parameter("scale.one", 1),),
            control.binding_id: (_parameter("scale.two", 2),),
        },
    }
    with pytest.raises(SchemaViolation, match="truth formula"):
        _invoke(**common, truths={scenario.scenario_id: "not-a-candidate"})
    with pytest.raises(SchemaViolation, match="response-calibrated"):
        _invoke(**common, release=replace(fixture._release(), response_calibrated=True))


def test_covariance_target_must_be_the_selected_response() -> None:
    scenario, values = fixture._scenario_and_values()
    wrong = UncertaintyRef(
        "response.diagonal-covariance",
        "source.vector.acceleration",
        "diagonal-covariance",
        scenario.uncertainties[0].artifact_path,
        scenario.uncertainties[0].artifact_sha256,
    )
    forged = replace(scenario, uncertainties=(wrong,))
    with pytest.raises(SchemaViolation, match="covariance targets"):
        _invoke(
            scenarios=(forged,),
            scenario_values={forged.scenario_id: values},
            comparisons={forged.scenario_id: _comparison()},
        )


def test_diagonal_and_valid_full_psd_whitening() -> None:
    diagonal = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    scenario, values = _scenario_with_uncertainty(diagonal, "response.diagonal-audit")
    diagonal_result = _invoke(
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        comparisons={scenario.scenario_id: _comparison("response.diagonal-audit")},
    )
    assert next(cell for cell in diagonal_result.cells if cell.formula_id == "scale-control").whitened_rmse == 0.0
    assert all(cell.discovery_status != DiscoveryStatus.NUMERICAL_INVALID.value for cell in diagonal_result.cells)

    full = np.diag(np.arange(1, 7, dtype=np.float64))
    scenario, values = _scenario_with_uncertainty(full, "response.full-psd-audit")
    full_result = _invoke(
        scenarios=(scenario,),
        scenario_values={scenario.scenario_id: values},
        comparisons={scenario.scenario_id: _comparison("response.full-psd-audit")},
    )
    assert next(cell for cell in full_result.cells if cell.formula_id == "scale-control").whitened_rmse == 0.0
    assert all(cell.discovery_status != DiscoveryStatus.NUMERICAL_INVALID.value for cell in full_result.cells)


def test_non_psd_full_covariance_is_rejected_independent_of_residual_direction() -> None:
    non_psd = np.diag(np.asarray((-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
    scenario, values = _scenario_with_uncertainty(non_psd, "response.non-psd-audit")
    _comparator, control = _default_bindings()
    with pytest.raises(SchemaViolation, match="not positive semidefinite"):
        _invoke(
            scenarios=(scenario,),
            scenario_values={scenario.scenario_id: values},
            bindings=(control,),
            parameter_cells={control.binding_id: (_parameter("scale.two", 2),)},
            comparisons={scenario.scenario_id: _comparison("response.non-psd-audit")},
        )


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ArithmeticError])
def test_broad_adapter_exceptions_are_retained_per_cell(monkeypatch, error_type) -> None:
    def unavailable(_features, _parameters):
        raise error_type("independent-audit adapter failure")

    monkeypatch.setattr(adapter_registry, "vector_scale_control", unavailable)
    result = _invoke()
    assert result.scored_cell_count == 0
    assert result.truth_recovery_count == 0
    assert result.distinct_truth_recovery_count == 0
    assert not any(cell.winner for cell in result.cells)
    assert {cell.discovery_status for cell in result.cells} == {
        DiscoveryStatus.NUMERICAL_INVALID.value
    }
    assert all(cell.result_sha256 is not None for cell in result.cells)
    assert len(result.ledger.entries) == 4


def test_formula_adapter_receives_only_declared_source_features(monkeypatch) -> None:
    observed: list[set[str]] = []
    original = adapter_registry.vector_scale_control

    def spy(features, parameters):
        observed.append(set(features))
        return original(features, parameters)

    monkeypatch.setattr(adapter_registry, "vector_scale_control", spy)
    _invoke()
    assert len(observed) == 4
    assert all(keys == {"source.vector.acceleration"} for keys in observed)


def test_all_invalid_is_retained_without_winner_or_pseudo_recovery(monkeypatch) -> None:
    def fail(_features, _parameters):
        raise SchemaViolation("independent-audit all-invalid")

    monkeypatch.setattr(adapter_registry, "vector_scale_control", fail)
    result = _invoke()
    assert result.attempted_cell_count == 2
    assert result.scored_cell_count == 0
    assert result.truth_recovery_count == 0
    assert result.distinct_truth_recovery_count == 0
    assert not any(cell.winner or cell.truth_recovered or cell.distinct for cell in result.cells)
    assert len({cell.ledger_entry_sha256 for cell in result.cells}) == 2


def test_hash_payload_has_no_native_float_and_replay_is_deterministic() -> None:
    first = _invoke()
    second = _invoke()
    assert first.to_dict() == second.to_dict()
    assert first.ledger.to_dict() == second.ledger.to_dict()
    assert first.content_sha256 == second.content_sha256
    assert first.content_sha256 == canonical_sha256(first._body())
    assert not _contains_native_float(first.to_dict())
    assert all(
        cell.to_dict()["whitened_rmse_hex"] is None
        or isinstance(cell.to_dict()["whitened_rmse_hex"], str)
        for cell in first.cells
    )


def test_result_mutations_fail_closed() -> None:
    result = _invoke()
    with pytest.raises(FrozenInstanceError):
        result.scored_cell_count = 99  # type: ignore[misc]
    with pytest.raises(SchemaViolation, match="result hash changed"):
        replace(result, scored_cell_count=result.scored_cell_count + 1)
    forged_cell = replace(result.cells[0], ledger_entry_sha256="0" * 64)
    with pytest.raises(SchemaViolation, match="exact replay-ledger entry"):
        replace(result, cells=(forged_cell, *result.cells[1:]))
    flipped = replace(result.cells[0], winner=not result.cells[0].winner)
    with pytest.raises(SchemaViolation, match="result hash changed"):
        replace(result, cells=(flipped, *result.cells[1:]))
