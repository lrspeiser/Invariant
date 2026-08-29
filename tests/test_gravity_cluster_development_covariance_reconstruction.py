from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_cluster_development_covariance_reconstruction as covariance,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MATRIX_HASHES = {
    "A1644": "ec778e52e8c6671980e6cfa95b2225479d9ce2c7706024c6d45e0c0d7971bb5c",
    "A1795": "df12eb7f403efa0511aa5ffbf2c6ce70fbdc5e09c913469e7ce233e19eef2539",
    "A2142": "b4b5b897fb68aebc65f13cd5ed39d8f7a4c686fcb3ac8f731183c1b0033e1c79",
    "A2255": "afd8b219d960ab4e52afcc4833efa5e32c2e22e217f2301901c66d7d5f0e4449",
    "A2319": "1f4fe21549d35d52acb54c80141e1c279a739182e669faa9bd2ce2773a374247",
    "A3266": "ab737b56feba84c25b5da768f8098278fb326da419644311f233d06ba8aa9d49",
    "A85": "f237f237e590b4e0afd926c6f04549e5b5493f47540efc95de70e18d81edf641",
    "ZW1215": "6b51cd92738709762930e19265ce6aaab97285167643bb291e7fedb2228a5150",
}


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return covariance.build_receipt(ROOT)


def test_reconstruction_is_development_only_and_numerically_valid() -> None:
    result = covariance.reconstruct_pressure_covariances(ROOT)
    assert tuple(result) == covariance.DEVELOPMENT_CLUSTERS
    assert not (set(result) & set(covariance.EXCLUDED_CLUSTERS))
    assert sum(row["bins"] for row in result.values()) == 82
    for cluster, row in result.items():
        matrix = row["matrix"]
        correlation = row["correlation"]
        assert matrix.shape == (row["bins"], row["bins"])
        assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-14)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-30
        assert np.allclose(np.diag(correlation), 1.0, rtol=1e-12, atol=1e-12)
        assert row["reconstructed_covariance_sha256"] == EXPECTED_MATRIX_HASHES[cluster]


def test_receipt_advances_only_the_pressure_development_pilot(
    receipt: dict[str, object],
) -> None:
    assert receipt["decision"] == (
        "DEVELOPMENT_PRESSURE_COVARIANCE_PILOT_RECONSTRUCTIBLE_CP5_REMAINS_PARTIAL"
    )
    assert receipt["advanced_goal_evidence"] == {
        "CP5.1": (
            "released_planck_pressure_correlation_reconstructible_for_8_"
            "already_exposed_development_clusters"
        )
    }
    assert receipt["completed_goal_evidence"] == {}
    assert receipt["counts"] == {
        "development_clusters_reconstructed": 8,
        "pressure_covariance_matrices": 8,
        "pressure_covariance_bins": 82,
        "same_release_confirmation_members_opened": 0,
        "independent_target_rows_opened": 0,
        "temperature_covariance_matrices": 0,
        "density_covariance_matrices": 0,
        "shared_or_cross_instrument_covariance_matrices": 0,
        "paid_model_calls": 0,
    }
    claims = receipt["claims"]
    assert claims["development_pressure_covariance_reconstruction_supported"]
    assert not claims["pressure_covariance_scored"]
    assert not claims["CP5_1_through_CP5_6_complete"]
    assert not claims["independent_replication"]
    assert not claims["scientific_result_emitted"]


def test_auxiliary_inventory_records_assets_without_inventing_covariance(
    receipt: dict[str, object],
) -> None:
    audit = receipt["auxiliary_archive_audit"]
    assert audit["available_counts"] == {
        "background_mosaics": 8,
        "exposure_mosaics": 8,
        "science_mosaics": 8,
        "spectral_fit_summaries": 8,
    }
    assert set(audit["absent_filename_token_counts"]) == {
        "psf",
        "beam",
        "arf",
        "rmf",
        "response",
        "event",
        "attitude",
        "mask",
        "noise",
        "calib",
    }
    assert not any(audit["absent_filename_token_counts"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["population_boundary"].__setitem__(
            "independent_target_rows_allowed", 1
        ),
        lambda value: value["population_boundary"]["development_clusters"].append(
            "A2029"
        ),
        lambda value: value["pressure_covariance_reconstruction"][
            "member_bindings"
        ][0].__setitem__("cluster", "A2029"),
        lambda value: value["claim_boundary"].__setitem__(
            "CP5_1_through_CP5_6_complete", True
        ),
        lambda value: value["component_dispositions"]["CP5.2"].__setitem__(
            "status", "COMPLETE"
        ),
        lambda value: value["implementation_binding"].__setitem__(
            "path", "src/sigma_theory_compiler/gravity_cluster_independent_data_contract.py"
        ),
    ],
)
def test_target_access_and_claim_promotion_mutations_fail_closed(mutation: object) -> None:
    config = copy.deepcopy(covariance.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(covariance.GravityClusterDevelopmentCovarianceError):
        covariance.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / covariance.OUTPUT_PATH).read_text(encoding="utf-8"))
    covariance.validate_receipt(stored, ROOT)
    assert stored == covariance.build_receipt(ROOT)
