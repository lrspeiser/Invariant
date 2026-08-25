from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.archimedes_first_principles import (
    RECEIPT_PATH,
    ArchimedesDiscoveryError,
    build_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(ROOT)


def test_recovers_archimedes_law_from_noisy_target_blind_data(receipt: dict) -> None:
    result = receipt["result"]
    discovery = receipt["buoyancy_discovery"]
    assert result == {
        "discovered_formula": "buoyant_force = fluid_density * gravity * displaced_volume",
        "training_rows": 2187,
        "shifted_holdout_rows": 128,
        "hydrostatic_sensor_rows": 243,
        "status": "PASS_BOUNDED_FIRST_PRINCIPLES_DISCOVERY_CONTROL",
    }
    assert discovery["winner"]["exponents"] == [1, -1, -1, -1, 0, 0, 0, 0]
    assert discovery["winner"]["expression"] == (
        "buoyant_force/(fluid_density*gravity*displaced_volume)"
    )
    assert discovery["independent_exact_evaluators_agree"] is True
    assert discovery["exact_constant_in_interval"] is True


def test_shift_nuisance_mutations_and_identifiability_are_explicit(receipt: dict) -> None:
    discovery = receipt["buoyancy_discovery"]
    assert discovery["shifted_holdout_relative_span"] == discovery["winner"]["relative_span"]
    assert {item["status"] for item in discovery["nuisance_interventions"]} == {"pass"}
    assert {item["status"] for item in discovery["mutation_controls"]} == {"rejected"}
    assert discovery["identifiability_control"]["status"] == "blocked_unidentifiable"
    assert discovery["identifiability_control"]["best_score_tie_count_in_top_five"] >= 2


def test_derivation_connects_pressure_data_to_force_law(receipt: dict) -> None:
    hydrostatic = receipt["hydrostatic_discovery"]
    derivation = receipt["derivation"]
    assert hydrostatic["winner"]["expression"] == (
        "delta_pressure/(fluid_density*gravity*delta_height)"
    )
    assert hydrostatic["winner"]["relative_span"] == "0"
    assert derivation["status"] == "pass"
    assert derivation["proof_plan"] == "hydrostatic_gradient_then_closed_surface_divergence"
    assert derivation["rectangular_prism_symbolic_control"]["status"] == "pass"
    assert derivation["steps"][-1]["claim"] == (
        "buoyant_force = fluid_density * gravity * displaced_volume"
    )


def test_committed_receipt_replays_exactly(receipt: dict) -> None:
    committed = json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    assert committed == receipt
    assert validate_receipt(committed, ROOT)["status"] == (
        "PASS_BOUNDED_FIRST_PRINCIPLES_DISCOVERY_CONTROL"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["result"].__setitem__("discovered_formula", "F = rho * V"),
        lambda value: value["buoyancy_discovery"]["winner"].__setitem__(
            "exponents", [1, 0, -1, -1, 0, 0, -1, 0]
        ),
        lambda value: value["derivation"].__setitem__("status", "pass_without_divergence"),
        lambda value: value.__setitem__("content_sha256", "0" * 64),
    ],
)
def test_resealed_or_semantically_mutated_receipts_fail_closed(receipt: dict, mutation) -> None:
    changed = deepcopy(receipt)
    mutation(changed)
    with pytest.raises(ArchimedesDiscoveryError, match="does not replay exactly"):
        validate_receipt(changed, ROOT)
