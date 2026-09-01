from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_holmberg_ii_2d_publication_synthesis_v1 as synthesis,
)


def test_synthesis_reaches_retrospective_six_object_threshold() -> None:
    receipt = synthesis.build_receipt(synthesis.load_config(verify_package=False))
    gate = receipt["retrospective_six_object_gate"]
    assert gate["threshold_met"] is True
    assert gate["signal_objects"] == ["NGC2976", "NGC3521", "UGC04305"]
    assert gate["new_response_blind_signal_objects"] == ["NGC3521", "UGC04305"]
    assert gate["retrospective_not_confirmation"] is True


def test_external_holmberg_cell_is_fixed_and_beats_all_controls() -> None:
    receipt = synthesis.build_receipt(synthesis.load_config(verify_package=False))
    cell = receipt["holmberg_external_cell"]
    assert cell["cell_score_id"] == "UGC04305__IRAC1_FIXED_ML0P6__I27P0__ROBUST"
    assert cell["winner"] == "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
    assert cell["all_three_comparators_beaten"] is True
    assert (
        cell["rmse_m_s"]["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"]
        < cell["rmse_m_s"]["NEWTON_3D_DST"]
    )
    assert (
        cell["rmse_m_s"]["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"]
        < cell["rmse_m_s"]["MOND_STANDARD_MU_ON_NEWTON_3D"]
    )
    assert (
        cell["rmse_m_s"]["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"]
        < cell["rmse_m_s"]["RAR_2016_ON_NEWTON_3D"]
    )


def test_novelty_is_bounded_and_not_global_priority() -> None:
    receipt = synthesis.build_receipt(synthesis.load_config(verify_package=False))
    audit = receipt["bounded_literature_audit"]
    assert audit["primary_rg_papers"] == 6
    assert audit["prior_2d_velocity_map_rg_test_identified"] is False
    assert audit["prior_holmberg_ii_rg_test_identified"] is False
    assert audit["global_priority_established"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_rebuild_uses_no_raw_response_or_network() -> None:
    receipt = synthesis.build_receipt(synthesis.load_config(verify_package=False))
    access = receipt["access_accounting"]
    assert access["sealed_aggregate_receipts_opened"] == 3
    assert access["response_files_opened"] == 0
    assert access["response_pixels_decoded"] == 0
    assert access["network_calls_during_deterministic_rebuild"] == 0


def test_material_config_mutations_fail_closed() -> None:
    config = synthesis.load_config(verify_package=False)
    mutations = []
    changed = copy.deepcopy(config)
    changed["claim_boundary"]["publication_ready"] = True
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["novelty_contract"]["global_priority_or_exhaustive_literature_claim"] = True
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["fixed_program_contract"]["epsilon_0"] = 0.5
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["retrospective_synthesis_contract"]["confirmation"] = True
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(synthesis.PublicationSynthesisError):
            synthesis.validate_config(mutation)


def test_predecessor_hash_mutation_fails() -> None:
    config = synthesis.load_config(verify_package=False)
    changed = copy.deepcopy(config)
    changed["predecessor_bindings"][0]["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(synthesis.PublicationSynthesisError):
        synthesis.build_receipt(changed)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert synthesis._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert synthesis._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(synthesis.PublicationSynthesisError):
        synthesis._atomic_no_clobber(path, b"two\n")


def test_stored_receipt_matches_rebuild_when_present() -> None:
    path = synthesis._repo_path(synthesis.OUTPUT_PATH)
    if path.exists():
        assert synthesis.check_receipt() == "VALID"
