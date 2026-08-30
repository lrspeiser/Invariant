from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_campaign_v1_postrun_adjudicator as post

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_source_outputs_are_exactly_bound() -> None:
    config = post.load_config(ROOT)
    source = config["source_campaign"]
    for relative, expected in source["package_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for path_key, hash_key in (
        ("access_intent_path", "access_intent_raw_sha256"),
        ("result_path", "result_raw_sha256"),
        ("adjudication_path", "adjudication_raw_sha256"),
    ):
        assert (
            hashlib.sha256((ROOT / source[path_key]).read_bytes()).hexdigest() == source[hash_key]
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("diagnosis", "relative_tolerance"), 1.0),
        (("diagnosis", "classification_changes_allowed"), 1),
        (("access_scope", "new_scores_computed"), 1),
        (("claim_ceiling", "cross_scale_survivor_claim"), True),
        (("source_campaign", "result_raw_sha256"), "0" * 64),
        (("output_path",), "C:/attacker.json"),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = post.load_config(ROOT)
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(post.OpenGravityPostrunAdjudicatorError):
        post.validate_config(mutated)


def test_receipt_rebuild_is_target_free_and_zero_survivor() -> None:
    receipt = post.build_receipt(ROOT)
    assert receipt["status"] == "PASS_STABLE_REAGGREGATION_ZERO_SURVIVORS"
    assert receipt["classifications"] == {
        "published_galaxy_passes": 0,
        "stable_galaxy_passes": 0,
        "published_cluster_passes": 0,
        "stable_cluster_passes": 0,
        "published_cross_domain_survivors": 0,
        "stable_cross_domain_survivors": 0,
    }
    assert receipt["best_development_cells"] == {
        "GALAXIES": "GP01L-n1",
        "CLUSTERS": "GP01L-n1",
    }
    assert receipt["diagnosis"]["changed_float_count"] == 6151
    assert receipt["diagnosis"]["max_absolute_delta"] <= 1e-12
    assert receipt["diagnosis"]["max_relative_delta"] <= 1e-12
    assert receipt["access_accounting"]["raw_scientific_source_files_opened"] == 0
    assert receipt["access_accounting"]["formula_evaluations"] == 0
    assert receipt["access_accounting"]["new_scores_computed"] == 0


def test_three_rebuilds_are_byte_semantic_identical() -> None:
    first = post.build_receipt(ROOT)
    assert first == post.build_receipt(ROOT) == post.build_receipt(ROOT)


def test_unexpected_float_path_is_rejected() -> None:
    config = post.load_config(ROOT)
    stats = {
        "changed_float_count": 0,
        "max_absolute_delta": 0.0,
        "max_relative_delta": 0.0,
        "changed_path_sha256": [],
    }
    with pytest.raises(post.OpenGravityPostrunAdjudicatorError):
        post._compare_normalized(
            {"raw_object_loss": 1.0},
            {"raw_object_loss": 1.0 + 1e-15},
            config,
            "global-cell-ledger.json.galaxies.0",
            stats,
        )


def test_only_allowlisted_derived_float_paths_can_normalize() -> None:
    assert post._allowed_float_delta(
        ".global-cell-ledger.json.galaxy_adjudication.0.maximum_subgroup_loss_ratio"
    )
    assert post._allowed_float_delta(
        ".global-cell-ledger.json.cluster_adjudication.0.scenario_evidence.0."
        "pilot_stage.candidate_mean_loss"
    )
    assert not post._allowed_float_delta(
        ".global-cell-ledger.json.galaxies.0.scenario_results.0.objects.0.loss"
    )


def test_module_has_no_raw_scientific_loader_or_scoring_call() -> None:
    source = inspect.getsource(post)
    for forbidden in (
        "_load_sparc_responses(",
        "_load_xcop_responses(",
        "_score_sparc_candidates(",
        "_score_xcop_candidates(",
        "_compute()",
    ):
        assert forbidden not in source


def test_cli_has_no_path_or_root_override() -> None:
    with pytest.raises(SystemExit):
        post.main(["status", "--root", "C:/attacker"])
    with pytest.raises(SystemExit):
        post.main(["check", "--output", "C:/attacker.json"])
