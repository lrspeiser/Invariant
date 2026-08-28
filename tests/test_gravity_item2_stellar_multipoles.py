from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.gravity_item1_effective_dimension as item1
import sigma_theory_compiler.gravity_item2_clash_stellar_multipoles as stellar
import sigma_theory_compiler.gravity_item2_stellar_multipole_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / experiment.OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def _member_row(*, x: float, y: float, radius_mpc: float, mass: float) -> dict[str, str]:
    return {
        "PointS": "0",
        "photoflag": "0",
        "nfobs": "16",
        "nfdet": "16",
        "s2n": "20",
        "Stell_Mass": str(mass),
        "clusterz": "0.4",
        "SpeczValue": "0.4",
        "SpeczQual": "0",
        "zb_1": "0.4",
        "zb_Min_1": "0.35",
        "zb_Max_1": "0.45",
        "Odds_1": "0.9",
        "PhyDistBCG": str(radius_mpc),
        "x": str(x),
        "y": str(y),
        "BCG_pos_RA": "1000",
        "BCG_pos_Dec": "1000",
    }


def test_target_blind_contract_and_sources_are_frozen() -> None:
    config = stellar.load_config(ROOT)
    assert len(config["sources"]["clash_molino_catalogs"]) == 20
    assert config["target_blind_extraction"]["common_aperture_kpc"] == 150
    assert config["target_blind_extraction"]["primary_weight_power"] == 1
    assert config["target_blind_extraction"]["lensing_corrected_columns_used"] == []
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["sparc_confirmation_evaluator_accesses_allowed"] == 0
    source = (ROOT / stellar.SOURCE_PATH).read_text(encoding="utf-8")
    assert "per_object_diagnostics" not in source
    assert "oracle_beta" not in source
    assert "gravity_item1" not in source


def test_member_selection_and_synthetic_multipoles_are_physical() -> None:
    config = stellar.load_config(ROOT)
    selection = config["target_blind_extraction"]["member_selection"]
    rows = [
        _member_row(x=1100, y=1000, radius_mpc=0.05, mass=10.5),
        _member_row(x=900, y=1000, radius_mpc=0.05, mass=10.5),
        _member_row(x=1000, y=1150, radius_mpc=0.10, mass=10.2),
        _member_row(x=1000, y=850, radius_mpc=0.10, mass=10.2),
    ]
    assert all(stellar.is_member(row, selection) == (True, "member") for row in rows)
    rejected = dict(rows[0], PointS="1")
    assert stellar.is_member(rejected, selection) == (False, "point_source")
    features = stellar.measure_catalog(
        rows,
        bcg_mass_solar=4.0e11,
        aperture_kpc=150,
        central_exclusion_kpc=5,
        weight_power=1,
        selection=selection,
    )
    assert features["member_count_including_bcg"] == 5
    assert 0 < features["effective_member_count"] <= 5
    assert 0 <= features["concentration_c20"] <= 1
    assert 0 <= features["quadrupole_amplitude"] <= 1
    assert features["m3_aperture_amplitude"] >= 0
    assert features["m4_aperture_amplitude"] >= 0


def test_sealed_representation_passes_without_gravity_or_lensing_fields() -> None:
    manifest = stellar.validate_extraction(ROOT)
    assert manifest["decision"] == "PASS_TARGET_BLIND_REPRESENTATION_GATE"
    assert all(manifest["checks"].values())
    assert manifest["counts"]["catalogs"] == 20
    assert manifest["counts"]["feature_rows"] == 60
    assert manifest["counts"]["minimum_primary_members_including_bcg"] == 5
    assert manifest["counts"]["gravity_response_receipts_read"] == 0
    assert manifest["counts"]["lensing_corrected_fields_used"] == 0
    primary = manifest["external_xray_validation"]["1"]
    assert all(float(value) > 0 for value in primary["components"].values())
    assert float(primary["joint"]["one_sided_p_value"]) <= 0.05
    assert float(primary["position_axis"]["fraction_within_30_deg"]) >= 0.4


def test_feature_table_contains_three_weightings_and_twenty_primary_clusters() -> None:
    manifest = stellar.validate_extraction(ROOT)
    with (ROOT / manifest["feature_file"]["path"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 60
    assert {float(row["weight_power"]) for row in rows} == {0.0, 0.5, 1.0}
    primary = [row for row in rows if float(row["weight_power"]) == 1.0]
    assert len(primary) == len({row["slug"] for row in primary}) == 20
    assert min(float(row["eight_sector_footprint_kpc"]) for row in primary) >= 150


def test_join_happens_after_representation_and_covers_expected_real_objects() -> None:
    config = experiment.load_config(ROOT)
    objects, labels, representation = experiment.prepare_objects(ROOT, config)
    assert representation["decision"] == "PASS_TARGET_BLIND_REPRESENTATION_GATE"
    assert len(objects) == len(labels) == 88
    assert sum(row["domain"] == "galaxy" for row in objects) == 68
    assert sum(row["domain"] == "cluster" for row in objects) == 20
    assert all("observed" not in row["features"] for row in objects)
    assert all(
        row["shape_provenance"]["weight_power"] == 1
        for row in objects
        if row["domain"] == "cluster"
    )


def test_folds_hold_out_whole_objects_and_balance_both_populations() -> None:
    config = experiment.load_config(ROOT)
    objects, _, _ = experiment.prepare_objects(ROOT, config)
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(objects, salt=cv["fold_salt"], folds=cv["outer_folds"])
    assert set(assignments) == {row["key"] for row in objects}
    for fold in range(5):
        heldout = [row for row in objects if assignments[row["key"]] == fold]
        assert sum(row["domain"] == "cluster" for row in heldout) == 4
        assert sum(row["domain"] == "galaxy" for row in heldout) in {13, 14}


def test_receipt_rebuilds_exactly_and_preserves_measured_claim_boundary() -> None:
    stored = _load(OUTPUT)
    rebuilt = experiment.build_receipt(ROOT)
    assert rebuilt == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["claims"]["alternative_to_gr_established"] is False
    assert stored["claims"]["direct_lensing_test_completed"] is False
    assert stored["counts"]["paid_model_calls"] == 0
    assert stored["counts"]["sparc_confirmation_evaluator_accesses"] == 0


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
        "stellar_tracer_is_complete_baryonic_mass_map",
    ],
)
def test_resealed_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["claims"][claim] = True
    with pytest.raises(experiment.GravityItem2StellarMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_proxy_admission_and_decision_drift_are_rejected() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["model_results"]["linear_support_dimension_proxy"][
        "qualifying_universal_stellar_multipole_model"
    ] = True
    with pytest.raises(experiment.GravityItem2StellarMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)

    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["decision"] = "PASS_ITEM2_STELLAR_MULTIPOLE_DEVELOPMENT_GATE"
    with pytest.raises(experiment.GravityItem2StellarMultipoleExperimentError):
        experiment.validate_receipt(_reseal(receipt), root=ROOT)
