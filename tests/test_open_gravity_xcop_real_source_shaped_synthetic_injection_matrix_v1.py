from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_xcop_real_source_shaped_synthetic_injection_matrix_v1 as subject,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage


def test_config_and_adapter_inventory_are_frozen() -> None:
    config = subject.load_config()
    subject.validate_config(config)
    assert config["clusters"] == [
        "A1644",
        "A1795",
        "A2142",
        "A2255",
        "A2319",
        "A3266",
        "A85",
        "ZW1215",
    ]
    assert len(config["mechanisms"]) == 8
    assert config["adapter_blocks"][0]["formula_id"] == "DPEL01_DISK_POLAR_ESCAPE_LOAD"


def test_common_abi_bindings_are_cluster_typed_and_dpel_is_blocked() -> None:
    config = subject.load_config()
    catalogue = subject._catalogue(config)
    bindings = subject._bindings(config)
    assert len(bindings) == 9
    assert sum(row.status is BindingStatus.EXECUTABLE for row in bindings) == 8
    dpel = next(row for row in bindings if row.formula_id == subject._DPEL)
    assert dpel.status is BindingStatus.UNADAPTED
    assert dpel.entrypoint is None
    assert all(row.domains == ("cluster",) for row in bindings)
    assert catalogue.content_sha256


def test_source_lift_opens_only_declared_baryonic_profiles_and_preserves_mass() -> None:
    config = subject.load_config()
    items, opened, _ = subject._source_items(config)
    assert len(items) == 8
    assert len(opened) == 13
    assert {row["role"] for row in opened} == {"density", "stellar_mass"}
    assert sum(row["bytes"] for row in opened) == 308160
    for item in items:
        metadata = item["source_metadata"]
        density = item["values"]["source.scalar.mass-density"]
        spacing = float(item["values"]["geometry.scalar.grid-spacing-normalized"][0])
        half_box = float(item["values"]["geometry.scalar.half-box-length"][0])
        lifted_mass = float(np.sum(density) * (spacing * half_box) ** 3)
        assert math.isclose(
            lifted_mass,
            metadata["outer_baryonic_mass_kg"],
            rel_tol=2.0e-15,
            abs_tol=0.0,
        )


def test_noise_families_are_seeded_finite_and_response_blind() -> None:
    config = subject.load_config()
    truth = np.full((17, 17, 17, 3), 2.0e-11, dtype=np.float64)
    radius = np.zeros((17, 17, 17), dtype=np.float64)
    for index, family in enumerate(config["noise_families"]):
        lineage = SeedLineage(
            config["suite_seed"],
            f"cluster.synthetic.test.noise-{family}.v1",
            "test-cluster",
            "truth.newton",
            index,
            0,
        )
        first = subject._noise_response(truth, family, lineage, 0.1, radius, config)
        second = subject._noise_response(truth, family, lineage, 0.1, radius, config)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert np.all(np.isfinite(first[0]))
        assert np.all(first[1] > 0.0)
    zero_lineage = SeedLineage(
        config["suite_seed"],
        "cluster.synthetic.test.noise-zero-noise.v1",
        "test-cluster",
        "truth.newton",
        2,
        0,
    )
    zero, _, _ = subject._noise_response(truth, "zero-noise", zero_lineage, 0.1, radius, config)
    assert np.array_equal(zero, truth)


def test_npz_serialization_is_deterministic() -> None:
    arrays = {
        "b": np.arange(5, dtype=np.float64),
        "a": np.arange(3, dtype=np.int64),
    }
    assert subject._npz_bytes(arrays) == subject._npz_bytes(dict(reversed(list(arrays.items()))))


def test_frozen_receipt_never_claims_empirical_support_if_present() -> None:
    path = subject._ROOT / subject.RECEIPT_PATH
    if not path.is_file():
        pytest.skip("append-only receipt has not been frozen yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["access_accounting"]["measured_response_rows_opened"] == 0
    assert receipt["dpel_adapter_block_count"] == receipt["scenario_count"]
