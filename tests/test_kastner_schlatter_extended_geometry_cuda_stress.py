from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.kastner_schlatter_extended_geometry_cuda_stress import (
    _load_predecessor,
    deterministic_inputs,
    deterministic_sources,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_extended_geometry_cuda_stress.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-extended-geometry-cuda-stress.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(document: dict) -> None:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_artifact_validates_and_exact_counters_close() -> None:
    artifact = _load(ARTIFACT)
    validate_campaign(artifact, CONFIG)
    assert artifact["counts"] == {
        "synthetic_geometry_classes": 4,
        "source_resolutions": 5,
        "geometry_resolution_cases": 20,
        "evaluation_radii_per_case": 2048,
        "unique_source_evaluation_interactions": 44_695_552,
        "gpu_warmup_repetitions": 2,
        "gpu_measured_repetitions": 64,
        "gpu_kernel_dispatches": 1320,
        "gpu_measured_source_evaluation_interactions": 2_860_515_328,
        "extended_source_laws_registered": 0,
        "lensing_cases_executed": 0,
        "observational_records_accessed": 0,
        "physical_or_theory_passes": 0,
    }


@pytest.mark.parametrize("geometry", ["thin_shell", "thin_ring", "disk_like", "spherical_volume"])
def test_deterministic_antipodal_unit_mass_sources(geometry: str) -> None:
    first_positions, first_mass = deterministic_sources(geometry, 64)
    second_positions, second_mass = deterministic_sources(geometry, 64)
    assert first_positions.tobytes() == second_positions.tobytes()
    assert first_mass.tobytes() == second_mass.tobytes()
    assert np.array_equal(first_positions[:32], -first_positions[32:])
    assert float(np.sum(first_mass)) == 1.0
    assert float(np.max(np.linalg.norm(first_positions, axis=1))) <= 1.0 + 1e-15


def test_manifest_is_deterministic() -> None:
    config = _load(CONFIG)
    first = deterministic_inputs(config)
    second = deterministic_inputs(config)
    assert first["evaluation_radii"].tobytes() == second["evaluation_radii"].tobytes()
    for left, right in zip(first["cases"], second["cases"], strict=True):
        assert left["positions"].tobytes() == right["positions"].tobytes()
        assert left["masses"].tobytes() == right["masses"].tobytes()


def test_hypotheses_are_separate_from_paper_and_fail_closed() -> None:
    artifact = _load(ARTIFACT)
    boundary = artifact["paper_boundary"]
    assert boundary["extended_source_operator_registered"] is False
    assert boundary["covariant_extended_metric_registered"] is False
    assert boundary["lensing_deflection_operator_registered"] is False
    enclosed = artifact["completion_hypotheses"]["H_enclosed_mass"]
    local = artifact["completion_hypotheses"]["H_local_superposition"]
    assert enclosed["source_status"] == "unproved_hypothesis_not_in_paper_equation_graph"
    assert enclosed["decision"] == "blocked_not_a_registered_extended_source_law"
    assert local["source_status"] == "unproved_hypothesis_not_in_paper_equation_graph"
    assert local["point_mass_aggregation_invariant"] is False
    assert local["continuum_discretization_invariant"] is False
    assert local["unequal_pair_matter_force_control"]["action_reaction_balance"] is False
    assert artifact["lensing_rotation_consistency_gate"]["executed"] is False


def test_gpu_cpu_and_far_field_bounds_close() -> None:
    artifact = _load(ARTIFACT)
    control = artifact["gpu_cpu_crosscheck"]
    assert control["maximum_absolute_component_error"] <= 1e-10
    assert control["maximum_far_coefficient_relative_error_to_sqrt_N"] <= 0.0005
    records = artifact["completion_hypotheses"]["H_local_superposition"][
        "far_field_case_records"
    ]
    assert len(records) == 20
    assert all(record["point_mass_asymptote_recovered"] is False for record in records)


def test_claim_data_and_host_path_seals() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["synthetic_only"] is True
    for key in (
        "observations_opened",
        "physical_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        assert artifact[key] is False
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    assert "c:" + "\\users\\" not in text
    assert "/" + "home/" not in text
    for marker in ("api" + "_key", "bear" + "er ", "s" + "k-"):
        assert marker not in text


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("physical_pass", True), "claim or data seal"),
        (
            lambda value: value["paper_boundary"].__setitem__(
                "extended_source_operator_registered", True
            ),
            "extended source operator was invented",
        ),
        (
            lambda value: value["lensing_rotation_consistency_gate"].__setitem__(
                "executed", True
            ),
            "unsupported lensing gate was executed",
        ),
        (
            lambda value: value["completion_hypotheses"]["H_local_superposition"].__setitem__(
                "point_mass_aggregation_invariant", True
            ),
            "local-superposition obstruction changed",
        ),
        (
            lambda value: value["deterministic_manifest"]["array_sha256"].__setitem__(
                "evaluation_radii", "0" * 64
            ),
            "deterministic geometry manifest changed",
        ),
    ],
)
def test_rehashed_tampering_fails_closed(mutation, message: str) -> None:
    artifact = copy.deepcopy(_load(ARTIFACT))
    mutation(artifact)
    _rehash(artifact)
    with pytest.raises(ValueError, match=message):
        validate_campaign(artifact, CONFIG)


def test_predecessor_hash_tamper_fails_before_cuda() -> None:
    config = _load(CONFIG)
    binding = config["predecessors"]["equation_graph"]
    original = binding["file_sha256"]
    binding["file_sha256"] = (
        ("0" if original[0] != "0" else "1") + original[1:]
    )
    with pytest.raises(ValueError, match="equation_graph file hash mismatch"):
        _load_predecessor(ROOT, "equation_graph", binding)
