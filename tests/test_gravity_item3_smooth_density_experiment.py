from __future__ import annotations

from pathlib import Path

import numpy as np

from sigma_theory_compiler import gravity_item3_smooth_density_experiment as experiment
from sigma_theory_compiler import gravity_item3_smooth_density_profiles as source

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 0
    for domain_index, domain in enumerate(("galaxy", "cluster")):
        for object_index in range(5):
            for radial_index in range(4):
                a = -2.0 + 0.2 * radial_index + 0.05 * object_index
                m = -1.0 + domain_index + 0.1 * radial_index
                c = 0.2 + 0.03 * object_index
                rows.append(
                    {
                        "row_id": row_id,
                        "variant": "primary",
                        "domain": domain,
                        "name": f"{domain}-{object_index}",
                        "object_key": f"{domain}:{object_index}",
                        "radius_kpc": float(radial_index + 1),
                        "response_log10_ratio": 0.3 * a + 0.4 * m * c,
                        "a": a,
                        "a2": a**2,
                        "a3": a**3,
                        "s": m + c / 2,
                        "v": m - c / 2,
                        "m": m,
                        "c": c,
                        "m_x_c": m * c,
                        "a_x_m": a * m,
                        "a_x_c": a * c,
                        "transition_product": 0.01 * (radial_index + 1),
                        "transition_balance": 0.1 * domain_index,
                        "population_proxy": float(domain_index),
                    }
                )
                row_id += 1
    return rows


def test_balanced_weights_equalize_domains_and_objects() -> None:
    rows = _synthetic_rows()
    weights = experiment._balanced_weights(rows)
    assert np.isclose(weights.sum(), 1.0)
    for domain in ("galaxy", "cluster"):
        indices = [index for index, row in enumerate(rows) if row["domain"] == domain]
        assert np.isclose(weights[indices].sum(), 0.5)
    for key in {str(row["object_key"]) for row in rows}:
        indices = [index for index, row in enumerate(rows) if row["object_key"] == key]
        assert np.isclose(weights[indices].sum(), 0.1)


def test_folds_hold_out_whole_objects_in_each_domain() -> None:
    rows = _synthetic_rows()
    assignments = experiment._object_folds(rows, salt="test", folds=5)
    assert set(assignments.values()) == set(range(5))
    assert len(assignments) == 10
    for fold in range(5):
        domains = {
            row["domain"]
            for row in rows
            if assignments[str(row["object_key"])] == fold
        }
        assert domains == {"galaxy", "cluster"}


def test_density_permutation_keeps_targets_and_gbar_features() -> None:
    rows = _synthetic_rows()
    permuted = experiment._permuted_density_rows(rows, seed=9)
    assert [row["response_log10_ratio"] for row in permuted] == [
        row["response_log10_ratio"] for row in rows
    ]
    assert [row["a"] for row in permuted] == [row["a"] for row in rows]
    assert all(
        np.isclose(float(row["m_x_c"]), float(row["m"]) * float(row["c"]))
        for row in permuted
    )


def test_real_sources_have_no_confirmation_access() -> None:
    config = source.load_config(ROOT)
    manifest = experiment._load_source_manifest(ROOT)
    primary, stellar = experiment._load_rows(ROOT)
    assert manifest["reserved_confirmation_profiles_opened"] == 0
    assert len(primary) == 375
    assert len(stellar) == 39
    assert set(config["cluster_lane"]["reserved_confirmation_objects"]).isdisjoint(
        {row["name"] for row in primary}
    )
