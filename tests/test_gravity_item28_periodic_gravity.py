from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item28_periodic_gravity import (
    _admissible_candidates,
    _build_term_matrix,
    _candidate_manifest,
    _common_rule,
    _contract_digest,
    _curve_summary,
    _row_physics,
    _suggest_injections,
    _vizier_rows,
    generate_raw_candidates,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _predictor() -> dict[str, object]:
    return {
        "identity": "UGC99999",
        "disk_scale_kpc": 2.0,
        "log_stellar_mass_proxy": 10.2,
        "bulge_fraction_proxy": 0.2,
        "bulge_re_kpc": 0.7,
        "bulge_sersic_n": 2.0,
        "radius_kpc": 3.5,
        "radius_disk_scale": 1.75,
        "disk_mu0_R": 21.0,
        "inclination_deg": 55.0,
        "ttype": 5.0,
        "bar_component_count": 1,
        "disk_break_present": 0,
        "seeing_arcsec": 2.0,
        "disk_scale_arcsec": 10.0,
        "side_fractional_difference": 0.05,
    }


def _curve_body() -> bytes:
    lines = [
        "# response fixture",
        "Name\tr\te_r\tr2\te_r2\tVrot\te_Vrot\tNbins\tSide",
        " \tkpc\tkpc\tarcsec\tarcsec\tkm/s\tkm/s\t \t",
        "---------\t-----\t-----\t-----\t----\t---\t---\t--\t-",
    ]
    radii = [0.4, 0.6, 0.9, 1.1, 1.4, 1.6, 1.9, 2.1, 2.4, 2.6, 2.9, 3.1, 3.4, 3.6]
    for side, offset in (("a", -3.0), ("r", 3.0)):
        for x in radii:
            radius = 2.0 * x
            velocity = 90.0 + 25.0 * math.log1p(x) + offset
            lines.append(
                f"UGC 99999\t{radius:.3f}\t0.05\t{x * 10:.1f}\t0.2\t{velocity:.3f}\t5.0\t4\t{side}"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _curve_body_ghasp_vi_alias() -> bytes:
    return _curve_body().replace(b"Nbins", b"NBins")


def test_item28_contract_has_equal_viability_and_raw_capacity() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["niche"]) == 262144
    assert [int(np.count_nonzero(raw["niche"] == niche)) for niche in range(4)] == [
        65536,
        65536,
        65536,
        65536,
    ]
    assert config["discovery_policy"]["equal_initial_viability"] is True
    assert config["discovery_policy"]["age_or_history_is_not_privileged"] is True
    assert config["discovery_policy"]["partial_results_are_not_pruned"] is True
    assert config["discovery_policy"]["paper_claim_requires_unchanged_fresh_replication"] is True


def test_item28_candidate_generation_is_deterministic() -> None:
    config = load_config(ROOT)
    first = generate_raw_candidates(config)
    second = generate_raw_candidates(config)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_item28_admissible_candidates_pass_physics_and_retain_every_niche() -> None:
    config = load_config(ROOT)
    arrays, audit = _admissible_candidates(config)
    assert len(arrays["niche"]) == audit["admissible_candidates"]
    assert all(int(audit["admissible_niche_counts"][str(niche)]) > 50000 for niche in range(4))
    assert audit["maximum_admitted_local_fractional_response"] <= 1e-5
    assert audit["minimum_admitted_mu"] >= 0.05
    assert audit["maximum_admitted_mu"] <= 20.0


def test_item28_frozen_injections_are_target_blind_and_cover_all_niches() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    injections = config["candidate_generator"]["synthetic_injection_admissible_indices"]
    assert injections == _suggest_injections(arrays)
    assert [int(arrays["niche"][index]) for index in injections] == [0, 1, 2, 3]
    assert _candidate_manifest(config)["synthetic_injection_admissible_indices"] == injections


def test_item28_vizier_parser_ignores_metadata_units_and_separator() -> None:
    rows = _vizier_rows(
        b"# metadata\nName\tValue\n \tunit\n--------\t-----\nUGC 1\t3.5\n"
    )
    assert rows == [{"Name": "UGC 1", "Value": "3.5"}]


def test_item28_curve_parser_requires_two_sides_and_builds_frozen_grid() -> None:
    config = load_config(ROOT)
    records, audit = _curve_summary(_curve_body(), _predictor(), config)
    assert audit["failure"] is None
    assert audit["approaching_raw_points"] == 14
    assert audit["receding_raw_points"] == 14
    assert len(records) == 6
    assert [record["radius_disk_scale"] for record in records] == [
        0.75,
        1.25,
        1.75,
        2.25,
        2.75,
        3.25,
    ]
    assert all(record["approaching_velocity_km_s"] < record["receding_velocity_km_s"] for record in records)


def test_item28_curve_parser_accepts_documented_ghasp_vi_nbins_alias() -> None:
    config = load_config(ROOT)
    records, audit = _curve_summary(_curve_body_ghasp_vi_alias(), _predictor(), config)
    assert audit["failure"] is None
    assert len(records) == 6


def test_item28_baryonic_proxy_and_all_periodic_terms_are_finite() -> None:
    config = load_config(ROOT)
    arrays, _ = _admissible_candidates(config)
    row = _predictor()
    velocity, acceleration, orbital_myr = _row_physics(row, config)
    assert velocity > 0.0
    assert acceleration > 0.0
    assert orbital_myr > 0.0
    selected = np.asarray(config["candidate_generator"]["synthetic_injection_admissible_indices"])
    subset = {key: value[selected] for key, value in arrays.items()}
    terms = _build_term_matrix(config, subset, [row])
    assert terms.shape == (4, 1)
    assert np.all(np.isfinite(terms))


def test_item28_common_rule_requires_stable_niche_wave_and_phase() -> None:
    config = load_config(ROOT)
    stable = [
        {"niche": 2, "wave": wave, "phase_rad": phase}
        for wave, phase in ((4.0, 0.0), (5.0, 0.3), (6.0, 0.6), (7.0, 0.8), (16.0, 3.0))
    ]
    audit = _common_rule(stable, config)
    assert audit["same_niche_folds"] == 5
    assert audit["common_wave_folds"] >= 4
    assert audit["common_phase_folds"] >= 4
    assert audit["pass"] is True
    unstable = [
        {"niche": niche, "wave": float(index + 1), "phase_rad": float(index)}
        for index, niche in enumerate((0, 1, 2, 3, 0))
    ]
    assert _common_rule(unstable, config)["pass"] is False


def test_item28_contract_digest_ignores_only_commit_bindings() -> None:
    config = load_config(ROOT)
    rebound = dict(config)
    rebound["scientific_freeze_commit"] = "a" * 40
    rebound["sample_freeze_commit"] = "b" * 40
    assert _contract_digest(config) == _contract_digest(rebound)
    changed = dict(config)
    changed["hypothesis"] = "changed"
    assert _contract_digest(config) != _contract_digest(changed)


def test_item28_result_validates_when_present() -> None:
    config = load_config(ROOT)
    result = ROOT / str(config["paths"]["result"])
    if not result.exists():
        return
    from sigma_theory_compiler.gravity_item28_periodic_gravity import validate_result

    assert validate_result(ROOT) == result
