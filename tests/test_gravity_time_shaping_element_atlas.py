from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_time_shaping_element_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_all_source_bindings_are_exact() -> None:
    config = atlas.load_config(ROOT)
    atlas.validate_config(config, ROOT)
    bindings = atlas.validate_bindings(config, ROOT)
    assert bindings["valid"] is True
    assert bindings["element_count"] == 20
    assert bindings["unique_element_evidence_files"] == 19
    assert bindings["top_level_bindings"][-1]["git_commit"] == (
        "0df01cbe54f3df2921bebaded6842170f4392428"
    )


def test_variant_registry_is_exact_and_collision_free() -> None:
    config = atlas.load_config(ROOT)
    variants = atlas.generate_variants(config)
    assert len(variants) == 3520
    assert sum(row["arity"] == 1 for row in variants) == 480
    assert sum(row["arity"] == 2 for row in variants) == 3040
    assert len({row["variant_id"] for row in variants}) == 3520


def test_every_element_occurs_in_all_frozen_single_and_pair_slots() -> None:
    config = atlas.load_config(ROOT)
    variants = atlas.generate_variants(config)
    for element in config["element_ontology"]:
        relevant = [row for row in variants if element["id"] in row["elements"]]
        assert len(relevant) == 328
        assert sum(row["arity"] == 1 for row in relevant) == 24
        assert sum(row["arity"] == 2 for row in relevant) == 304


def test_symbolic_clock_derivations_pass() -> None:
    checks = atlas.symbolic_checks()
    assert len(checks) == 9
    assert all(row["passed"] for row in checks)
    assert checks[2]["check_id"] == "S03_SOLAR_RESPONSE"


def test_all_variants_are_finite_positive_and_solar_screened() -> None:
    config = atlas.load_config(ROOT)
    result = atlas.probe_variants(atlas.generate_variants(config))
    assert result["variant_count"] == 3520
    assert result["probe_evaluations"] == 14080
    assert result["all_finite_positive"] is True
    assert result["all_high_acceleration_screened"] is True
    assert float(result["max_high_acceleration_nu_minus_one"]) < 1e-12


def test_real_evidence_replay_preserves_positive_and_negative_results() -> None:
    replay = atlas.replay_real_evidence(ROOT)
    assert replay["signal_count"] == 7
    assert replay["strongest_signal_id"] == "stellar_age_x_density"
    assert float(replay["signals"][0]["directional_improvement"]) > 0.18
    assert replay["signals"][1]["status"] == "rejected_as_explanation_age_signal_persists"
    assert all(float(row["directional_improvement"]) <= 0 for row in replay["signals"][4:])
    assert replay["fresh_raw_rows_opened"] == 0
    assert replay["confirmation_rows_opened"] == 0


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("variant_grammar", "total_variant_count", 3519),
        ("derivation_contract", "target_independence", "after response"),
        ("real_data_replay_contract", "confirmation_rows_opened", 1),
        ("next_execution_contract", "authorization_state", "authorized"),
        ("claim_boundary", "all_3520_structures_fresh_real_data_scored", True),
        ("zero_new_access", "new_real_rows_opened", 1),
    ],
)
def test_nested_mutations_fail_closed(section: str, key: str, value: object) -> None:
    forged = copy.deepcopy(atlas.load_config(ROOT))
    forged[section][key] = value
    with pytest.raises(atlas.TimeShapingAtlasError):
        atlas.validate_config(forged, ROOT)


def test_element_evidence_mutation_fails_closed() -> None:
    forged = copy.deepcopy(atlas.load_config(ROOT))
    forged["element_ontology"][12]["disposition"] = "publication_pass"
    with pytest.raises(atlas.TimeShapingAtlasError):
        atlas.validate_config(forged, ROOT)


def test_build_receipt_has_honest_ceiling() -> None:
    receipt = atlas.build_receipt(ROOT)
    assert receipt["decision"] == atlas.DECISION
    assert receipt["variant_registry"]["total_variant_count"] == 3520
    assert receipt["variant_prefilter"]["probe_evaluations"] == 14080
    assert receipt["claim_boundary"]["existing_real_data_evidence_replayed"] is True
    assert receipt["claim_boundary"]["all_3520_structures_fresh_real_data_scored"] is False
    assert receipt["claim_boundary"]["scientific_claim_allowed"] is False
    assert all(value == 0 for value in receipt["zero_new_access"].values())


def test_stored_receipt_matches_exact_rebuild() -> None:
    receipt = atlas.check_receipt(ROOT)
    assert receipt == atlas.build_receipt(ROOT)
    payload = dict(receipt)
    content = payload.pop("content_sha256")
    assert content == atlas._content_sha(payload)


def test_tampered_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = atlas.build_receipt(ROOT)
    receipt["claim_boundary"]["scientific_claim_allowed"] = True
    payload = dict(receipt)
    payload.pop("content_sha256")
    receipt["content_sha256"] = atlas._content_sha(payload)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(atlas, "OUTPUT_PATH", output.relative_to(tmp_path))
    monkeypatch.setattr(atlas, "build_receipt", lambda root: receipt | {"decision": "different"})
    with pytest.raises(atlas.TimeShapingAtlasError):
        atlas.check_receipt(tmp_path)


def test_atomic_no_clobber_race_retains_one_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"

    def publish(payload: bytes) -> str:
        try:
            return atlas._atomic_no_clobber(target, payload)
        except atlas.TimeShapingAtlasError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert target.read_bytes() in {b"first", b"second"}
    assert outcomes.count("CREATED") == 1
    assert outcomes.count("REFUSED") == 1


def test_status_is_explicitly_not_a_fresh_real_score() -> None:
    result = atlas.status(ROOT)
    assert result["valid"] is True
    assert result["elements"] == 20
    assert result["variants"] == 3520
    assert result["strongest_signal"] == "stellar_age_x_density"
    assert result["fresh_real_score"] is False
    assert result["authorization"] == "not_authorized_by_this_atlas"
