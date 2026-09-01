from __future__ import annotations

import csv
import io
from collections import Counter
from copy import deepcopy

import pytest

from sigma_theory_compiler import open_gravity_campaign_v1_near_miss_atlas_v1 as atlas


@pytest.fixture(scope="module")
def config():
    return atlas.load_config()


@pytest.fixture(scope="module")
def packet(config):
    payloads, summary = atlas.build_artifacts(config)
    return payloads, summary, atlas.build_receipt(config)


def _csv_rows(payload: bytes):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def test_config_and_package_pins(config):
    atlas.validate_config(config)
    assert atlas.file_sha256(atlas._repo_path(atlas.CONFIG_PATH)) == atlas._CONFIG_RAW_SHA256
    assert atlas.content_sha256(config) == atlas._CONFIG_CONTENT_SHA256
    assert (
        atlas.module_semantic_sha256(atlas._repo_path(atlas.MODULE_PATH))
        == atlas._MODULE_SEMANTIC_SHA256
    )
    assert atlas.file_sha256(atlas._repo_path(atlas.TEST_PATH)) == atlas._TEST_RAW_SHA256


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("analysis_contract", "registered_live_candidates"), 408),
        (("analysis_contract", "strict_threshold_fraction"), 0.0),
        (("analysis_contract", "new_response_scoring"), 1),
        (("artifact_contract", "artifacts", 0, "rows"), 1847),
        (("claim_boundary", "physical_time_dilation_tested"), True),
        (("claim_boundary", "publication_ready"), True),
        (("output_path",), "runs/forged.json"),
    ],
)
def test_material_config_mutations_fail(config, path, value):
    forged = deepcopy(config)
    cursor = forged
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(atlas.NearMissAtlasError):
        atlas.validate_config(forged)


def test_exact_artifact_inventory_and_counts(packet):
    payloads, summary, receipt = packet
    assert sorted(payloads) == [
        "blocked-and-unscored-ideas.csv",
        "cell-comparator-atlas.csv",
        "comparator-ladder.csv",
        "concept-domain-atlas.csv",
        "figure-near-miss-ladder.svg",
        "near-miss-summary.md",
    ]
    assert summary["planned_domain_cells"] == 1848
    assert summary["valid_domain_cells"] == 1822
    assert summary["invalid_source_gate_cells"] == 26
    assert summary["strict_domain_survivors"] == 0
    assert summary["domain_concept_rows"] == 189
    assert summary["blocked_or_unscored_rows"] == 285
    assert len(receipt["summary"]["artifact_index"]) == 6


def test_cell_atlas_distinguishes_failure_from_baseline_gains(packet):
    payloads, summary, _ = packet
    rows = _csv_rows(payloads["cell-comparator-atlas.csv"])
    assert len(rows) == 1848
    assert len({(row["domain"], row["cell_id"]) for row in rows}) == 1848
    categories = Counter((row["domain"], row["category"]) for row in rows)
    assert categories[("GALAXIES", "INVALID_SOURCE_GATE")] == 8
    assert categories[("CLUSTERS", "INVALID_SOURCE_GATE")] == 18
    assert categories[("GALAXIES", "NUISANCE_CONDITIONAL_OVER_RAR")] == 1
    assert categories[("GALAXIES", "ROBUST_OVER_BARYON_ONLY")] == 75
    assert categories[("CLUSTERS", "ROBUST_OVER_BARYON_ONLY")] == 1458
    assert summary["galaxy_baryon_robust_cells"] == 76
    assert summary["cluster_baryon_robust_cells"] == 1458


def test_gp01_l_n1_is_conditional_not_a_survivor(packet):
    payloads, summary, _ = packet
    rows = _csv_rows(payloads["cell-comparator-atlas.csv"])
    galaxy = next(
        row for row in rows if row["domain"] == "GALAXIES" and row["cell_id"] == "GP01L-n1"
    )
    cluster = next(
        row for row in rows if row["domain"] == "CLUSTERS" and row["cell_id"] == "GP01L-n1"
    )
    assert galaxy["category"] == "NUISANCE_CONDITIONAL_OVER_RAR"
    assert galaxy["strict_domain_pass"] == "false"
    assert float(galaxy["rar_worst_fractional_improvement"]) == pytest.approx(-0.317387282499)
    assert float(galaxy["rar_best_fractional_improvement"]) == pytest.approx(0.0849871909391)
    assert cluster["rar_outcome"] == "LOSS_ALL_SCENARIOS"
    assert float(cluster["rar_worst_fractional_improvement"]) == pytest.approx(-0.258473326254)
    assert float(cluster["rar_best_fractional_improvement"]) == pytest.approx(-0.216704043586)
    assert summary["galaxy_rar_conditional_cells"] == 1
    assert summary["cluster_rar_any_win_cells"] == 0


def test_comparator_ladder_preserves_context(packet):
    payloads, _, _ = packet
    rows = _csv_rows(payloads["comparator-ladder.csv"])
    assert len(rows) == 15
    galaxy_nfw = next(
        row
        for row in rows
        if row["domain"] == "GALAXIES" and row["comparator_id"] == "GR_PLUS_NFW_CONTEXTUAL_CEILING"
    )
    cluster_cross = next(
        row
        for row in rows
        if row["domain"] == "CLUSTERS" and row["comparator_id"] == "PREVIOUS_CROSS_SCALE"
    )
    galaxy_baryon = next(
        row for row in rows if row["domain"] == "GALAXIES" and row["comparator_id"] == "BARYON_ONLY"
    )
    assert galaxy_nfw["candidate_cells_with_any_scenario_win"] == "0"
    assert cluster_cross["candidate_cells_with_any_scenario_win"] == "0"
    assert galaxy_baryon["candidate_cells_winning_all_scenarios"] == "76"
    assert galaxy_baryon["candidate_cells_winning_all_by_two_percent"] == "66"


def test_blocked_ledger_is_not_mislabeled_as_falsified(packet):
    payloads, _, _ = packet
    rows = _csv_rows(payloads["blocked-and-unscored-ideas.csv"])
    assert len(rows) == 285
    campaign_rows = [row for row in rows if row["source"] == "CAMPAIGN_BLOCKED_IDEA_LEDGER"]
    statuses = Counter(row["candidate_status"] for row in campaign_rows)
    assert statuses == {
        "REGISTERED_THEORY_ONLY": 274,
        "SOURCE_BLOCKED": 3,
        "KNOWN_REWRITE_NONINDEPENDENT": 1,
        "QUARANTINED_REVISION_REQUIRED": 1,
    }
    for row in rows:
        assert row["physical_time_dilation_derived"] == "false"
        assert row["light_propagation_derived"] == "false"
        assert row["redshift_closure_derived"] == "false"
        assert row["capture_or_dissipation_derived"] == "false"
        assert row["tensor_or_quantum_gravity_derived"] == "false"


def test_summary_states_executable_depth_and_untested_scope(packet):
    payloads, _, receipt = packet
    text = payloads["near-miss-summary.md"].decode("utf-8")
    assert "not equivalent to 407 complete physical theories" in text
    assert "static radial matter-response closures" in text
    assert "does not eliminate the broader time-well" in text
    assert "No valid candidate cell beat the strongest executable comparator" in text
    assert receipt["claim_boundary"]["response_free_reaggregation_complete"] is True
    assert receipt["claim_boundary"]["unique_theory_established"] is False
    assert receipt["claim_boundary"]["physical_time_dilation_tested"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_figure_and_csv_outputs_are_response_free(packet):
    payloads, _, _ = packet
    figure = payloads["figure-near-miss-ladder.svg"].decode("utf-8")
    assert figure.startswith("<svg")
    assert figure.count("<rect class=") == 15
    combined = b"\n".join(payloads.values()).lower()
    assert b".fits" not in combined
    assert b"vobs" not in combined
    assert b"response_asset_sha256" not in combined


def test_receipt_forgery_fails(config, packet):
    _, _, receipt = packet
    forged = deepcopy(receipt)
    forged["claim_boundary"]["publication_ready"] = True
    forged["content_sha256"] = atlas.content_sha256({**forged, "content_sha256": ""})
    with pytest.raises(atlas.NearMissAtlasError):
        atlas.validate_receipt(config, forged)


def test_build_is_deterministic(config, packet):
    payloads, summary, receipt = packet
    payloads_again, summary_again = atlas.build_artifacts(config)
    assert payloads_again == payloads
    assert summary_again == summary
    assert atlas.build_receipt(config) == receipt
