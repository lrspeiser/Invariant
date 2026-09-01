from __future__ import annotations

import csv
import io
from copy import deepcopy

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_seven_galaxy_2d_manuscript_evidence_v1 as manuscript,
)


@pytest.fixture(scope="module")
def config():
    return manuscript.load_config()


@pytest.fixture(scope="module")
def packet(config):
    payloads, summary = manuscript.build_artifacts(config)
    return payloads, summary, manuscript.build_receipt(config)


def test_config_and_package_pins(config):
    manuscript.validate_config(config)
    assert (
        manuscript.file_sha256(manuscript._repo_path(manuscript.CONFIG_PATH))
        == manuscript._CONFIG_RAW_SHA256
    )
    assert manuscript.content_sha256(config) == manuscript._CONFIG_CONTENT_SHA256
    assert (
        manuscript.module_semantic_sha256(manuscript._repo_path(manuscript.MODULE_PATH))
        == manuscript._MODULE_SEMANTIC_SHA256
    )
    assert (
        manuscript.file_sha256(manuscript._repo_path(manuscript.TEST_PATH))
        == manuscript._TEST_RAW_SHA256
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("artifact_contract", "raw_response_maps_reopened"), True),
        (("artifact_contract", "artifacts", 0, "rows"), 8),
        (("claim_boundary", "universal_rg_replication"), True),
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
    with pytest.raises(manuscript.ManuscriptEvidenceError):
        manuscript.validate_config(forged)


def test_exact_artifact_inventory_and_counts(packet):
    payloads, summary, receipt = packet
    assert sorted(payloads) == [
        "figure-1-seven-primary-rmse.svg",
        "figure-2-rg-vs-best-comparator-all-cells.svg",
        "manuscript-summary.md",
        "table-1-seven-primary-cells.csv",
        "table-2-all-48-sensitivity-cells.csv",
        "table-3-six-leave-one-out-reaggregations.csv",
    ]
    assert summary["primary_objects"] == 7
    assert summary["all_score_cells"] == 48
    assert summary["leave_one_out_reaggregations"] == 6
    assert summary["external_primary_rg_wins"] == 0
    assert summary["external_rg_winning_cells"] == 3
    assert summary["ic2574_rg_winning_cells"] == 3
    assert summary["loo_rg_rank_one"] == 1
    assert len(receipt["evidence_summary"]["artifact_index"]) == 6


def _csv_rows(payload: bytes):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def test_primary_table_contains_exact_seven_objects(packet):
    payloads, _, _ = packet
    rows = _csv_rows(payloads["table-1-seven-primary-cells.csv"])
    assert [row["object_id"] for row in rows] == list(manuscript._OBJECT_ORDER)
    assert all(row["primary_cell"] == "true" for row in rows)
    assert sum(row["winner"] == manuscript._MODEL_KEYS["RG"] for row in rows) == 0
    ic2574 = next(row for row in rows if row["object_id"] == "IC2574")
    assert float(ic2574["newton_rmse_km_s"]) == pytest.approx(16.0987514134)
    assert float(ic2574["rg_rmse_km_s"]) == pytest.approx(16.5104999901)


def test_all_cells_and_loo_tables_are_deterministic(packet):
    payloads, _, _ = packet
    all_rows = _csv_rows(payloads["table-2-all-48-sensitivity-cells.csv"])
    loo = _csv_rows(payloads["table-3-six-leave-one-out-reaggregations.csv"])
    assert len(all_rows) == 48
    assert len({row["cell_score_id"] for row in all_rows}) == 48
    assert sum(row["rg_wins_cell"] == "true" for row in all_rows) == 9
    assert len(loo) == 6
    assert sum(row["rg_rank"] == "1" for row in loo) == 1


def test_figures_are_vector_summaries_not_response_maps(packet):
    payloads, _, _ = packet
    primary = payloads["figure-1-seven-primary-rmse.svg"].decode()
    sensitivity = payloads["figure-2-rg-vs-best-comparator-all-cells.svg"].decode()
    assert primary.startswith("<svg")
    assert "Seven preregistered primary cells" in primary
    assert primary.count("<polyline") == 4
    assert sensitivity.startswith("<svg")
    assert sensitivity.count("<circle") == 50
    assert ".fits" not in primary.lower() + sensitivity.lower()
    assert "response_asset_sha256" not in primary + sensitivity


def test_summary_keeps_claim_boundary(packet):
    payloads, _, receipt = packet
    text = payloads["manuscript-summary.md"].decode()
    assert "0/6" in text
    assert "MODEL_LIFTED_2P5D" in text
    assert "No p-values" in text
    assert "not publication-ready" in text
    assert "bounded-corpus" in text
    assert receipt["claim_boundary"]["deterministic_manuscript_artifacts_built"] is True
    assert receipt["claim_boundary"]["universal_rg_replication"] is False
    assert receipt["claim_boundary"]["unique_theory_established"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_receipt_forgery_fails(config, packet):
    _, _, receipt = packet
    forged = deepcopy(receipt)
    forged["claim_boundary"]["publication_ready"] = True
    forged["content_sha256"] = manuscript.content_sha256({**forged, "content_sha256": ""})
    with pytest.raises(manuscript.ManuscriptEvidenceError):
        manuscript.validate_receipt(config, forged)


def test_build_is_deterministic(config, packet):
    payloads, summary, receipt = packet
    payloads_again, summary_again = manuscript.build_artifacts(config)
    assert payloads_again == payloads
    assert summary_again == summary
    assert manuscript.build_receipt(config) == receipt
