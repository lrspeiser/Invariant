from __future__ import annotations

import copy
import json

import pytest

from sigma_theory_compiler import open_gravity_dissipative_capture_enlarged_systems_v2 as capture


def test_old_receipt_is_pinned_and_explicitly_superseded() -> None:
    config = capture.load_config()
    assert (
        capture.file_sha256(capture.Path(config["supersedes"]["path"]))
        == config["supersedes"]["file_sha256"]
    )
    assert config["supersedes"]["audit_status"].startswith("BLOCKED_")


def test_receiver_dynamics_derive_invariants_from_state() -> None:
    config = capture.load_config()
    row = capture.receiver_dynamics_fixture(
        config, "single", config["dynamics_fixture"]["single"]["post_pass_initial_h"]
    )
    assert row["receiver_is_posthoc_deficit"] is False
    assert len(row["initial_state"]) == len(row["final_state"]) == 10
    assert row["final_invariants"]["receiver_internal_energy"] > 0.0
    assert row["final_invariants"]["receiver_entropy"] > 0.0
    assert abs(row["state_derived_errors"]["total_energy"]) < 1.0e-8
    assert row["state_derived_errors"]["linear_momentum_norm"] < 1.0e-14
    assert abs(row["state_derived_errors"]["angular_momentum"]) < 1.0e-12
    assert abs(row["state_derived_errors"]["entropy_identity"]) < 1.0e-12


def test_same_matter_state_memory_history_changes_capture() -> None:
    config = capture.load_config()
    quiet = capture.receiver_dynamics_fixture(
        config, "single", config["dynamics_fixture"]["single"]["quiet_initial_h"]
    )
    post = capture.receiver_dynamics_fixture(
        config, "single", config["dynamics_fixture"]["single"]["post_pass_initial_h"]
    )
    assert quiet["initial_state"][:9] == post["initial_state"][:9]
    assert quiet["initial_relative_energy"] == pytest.approx(post["initial_relative_energy"])
    assert quiet["captured"] is False
    assert post["captured"] is True
    assert (
        post["final_invariants"]["receiver_internal_energy"]
        > quiet["final_invariants"]["receiver_internal_energy"]
    )


def test_bimodal_is_executable_but_retains_extra_identifiability_gap() -> None:
    config = capture.load_config()
    quiet = capture.receiver_dynamics_fixture(
        config, "bimodal", config["dynamics_fixture"]["bimodal"]["quiet_initial_h"]
    )
    post = capture.receiver_dynamics_fixture(
        config, "bimodal", config["dynamics_fixture"]["bimodal"]["post_pass_initial_h"]
    )
    assert len(quiet["final_state"]) == len(post["final_state"]) == 11
    assert quiet["captured"] is False
    assert post["captured"] is True
    card = next(row for row in config["mechanisms"] if row["id"].startswith("DC06_"))
    assert "extra kernel mode requires identifiability" in card["gaps"]
    assert "NESTED_EXTENSION" in card["novelty_grade"]


def test_compression_gate_has_state_receiver_and_time_arrow() -> None:
    result = capture.receiver_dynamics_fixture(capture.load_config(), "compression")
    assert result["final_invariants"]["receiver_internal_energy"] > 0.0
    assert result["final_invariants"]["receiver_entropy"] > 0.0
    assert result["gate_energy_warning"] is None


def test_zero_gamma_recovers_conservative_limit() -> None:
    config = capture.load_config()
    conservative = capture.receiver_dynamics_fixture(config, "conservative")
    modified = copy.deepcopy(config)
    modified["dynamics_fixture"]["gamma"] = 0.0
    memory = capture.receiver_dynamics_fixture(
        modified, "single", modified["dynamics_fixture"]["single"]["post_pass_initial_h"]
    )
    assert memory["final_relative_energy"] == pytest.approx(
        conservative["final_relative_energy"], abs=1.0e-11
    )
    assert memory["final_invariants"]["receiver_internal_energy"] == 0.0


def test_explicit_conservative_third_body_captures_visible_pair() -> None:
    result = capture.three_body_countermodel_fixture(capture.load_config())
    assert result["initial_visible_pair_energy"] > 0.0
    assert result["final_visible_pair_energy"] < 0.0
    assert result["visible_pair_became_bound"] is True
    assert result["entropy_change"] == 0.0
    assert abs(result["state_derived_errors"]["total_energy"]) < 5.0e-8
    assert result["state_derived_errors"]["linear_momentum_norm"] < 1.0e-12
    assert abs(result["state_derived_errors"]["angular_momentum"]) < 1.0e-12


def test_receiver_demotions_are_not_silently_counted_as_dynamics() -> None:
    config = capture.load_config()
    cards = {row["id"][:4]: row for row in config["mechanisms"]}
    for key in ("DC02", "DC03", "DC04"):
        assert cards[key]["structural_metrics"]["executable_states"] == 0
        assert cards[key]["execution_grade"] in {
            "CONSTITUTIVE_JUMP_MAP_ONLY",
            "EFFECTIVE_FORCE_ONLY",
            "FAR_ZONE_FLUX_FORMULA_ONLY",
        }
        assert cards[key]["gaps"]


def test_memory_gate_energy_failure_is_retained() -> None:
    config = capture.load_config()
    for key in ("DC05", "DC06"):
        card = next(row for row in config["mechanisms"] if row["id"].startswith(key))
        assert any("stress energy undefined" in value for value in card["gaps"])
        assert "NOT_COVARIANTLY_CLOSED" in card["theory_grade"]


def test_ranking_is_computed_and_never_called_a_data_score() -> None:
    config = capture.load_config()
    ranking = capture.mechanism_triage(config)
    assert [row["rank"] for row in ranking] == list(range(1, 9))
    assert all(row["is_data_or_model_score"] is False for row in ranking)
    candidates = [row for row in ranking if row["candidate_eligible"]]
    assert candidates[0]["mechanism_id"] == "DC05_TIMEWELL_SINGLE_MEMORY_BATH"
    assert (
        next(row for row in ranking if row["mechanism_id"].startswith("DC06"))[
            "theory_triage_score"
        ]
        < candidates[0]["theory_triage_score"]
    )
    assert "EXPERT_STRUCTURAL_TRIAGE" in config["triage_policy"]["label"]


def test_tng_manifest_is_exact_where_resolved_and_honestly_source_blocked() -> None:
    manifest = capture.load_config()["tng_manifest"]
    assert manifest["status"] == "SOURCE_BLOCKED_API_AUTH_AND_PAYLOAD_CHECKSUMS_UNAVAILABLE"
    assert [row["snap"] for row in manifest["snapshot_grid"]] == [50, 59, 67, 72, 78, 84, 91, 99]
    assert manifest["metadata_pages"][0]["simulation"] == "TNG100-1"
    assert manifest["metadata_pages"][1]["simulation"] == "TNG100-1-Dark"
    assert manifest["endpoint_templates"]["matching_filename"] == "subhalo_matching_to_dark.hdf5"
    assert "Student-t" in manifest["likelihood"]["model"]
    assert "collisionless violent relaxation" in manifest["likelihood"]["nulls"]
    assert "payload byte sizes and SHA256 hashes" in manifest["unresolved_before_source_open"]


def test_countermodels_retain_violent_relaxation_and_ordinary_physics() -> None:
    rows = capture.counterexamples(capture.load_config())
    ids = {row["id"] for row in rows}
    assert ids == {
        "CM01_EXPLICIT_CONSERVATIVE_THREE_BODY_CAPTURE",
        "CM02_COLLISIONLESS_VIOLENT_RELAXATION",
        "CM03_ORDINARY_GAS_SHOCK_COOLING",
        "CM04_ESTABLISHED_FRICTION_AND_FEEDBACK",
    }


def test_adversarial_entropy_and_source_mutations_fail_closed() -> None:
    config = capture.load_config()
    mutations = (
        lambda value: value["dynamics_fixture"].__setitem__("gamma", -0.1),
        lambda value: value["dynamics_fixture"]["single"].__setitem__("weights", [-1.0]),
        lambda value: value["access_contract"].__setitem__("raw_scientific_rows", 1),
        lambda value: value["access_contract"].__setitem__("new_real_data_scores", 1),
        lambda value: value["tng_manifest"].__setitem__("status", "READY"),
        lambda value: value["triage_policy"].__setitem__("retain_all_failures", False),
    )
    for mutation in mutations:
        forged = copy.deepcopy(config)
        mutation(forged)
        with pytest.raises(capture.DissipativeCaptureV2Error):
            capture.validate_config(forged)


def test_receipt_has_split_pass_block_and_requests_independent_reaudit() -> None:
    receipt = capture.build_receipt()
    assert receipt["status"] == "PASS_THEORY_REPAIR_BLOCK_TNG_SOURCE"
    assert receipt["receiver_audit"]["posthoc_receiver_deficits"] == 0
    assert receipt["receiver_audit"]["state_derived_conservation_pass"] is True
    assert receipt["same_state_history_conditioned_capture"] is True
    assert receipt["conservative_three_body_countermodel_pass"] is True
    assert receipt["access_accounting"]["raw_scientific_rows"] == 0
    assert receipt["decision"] == "REQUEST_INDEPENDENT_REAUDIT_BEFORE_ANY_TNG_RESPONSE_ACCESS"


def test_artifacts_are_deterministic_and_report_source_block() -> None:
    config = capture.load_config()
    first = capture.build_artifacts(config)
    second = capture.build_artifacts(config)
    assert first == second
    assert set(first) == {
        "theory-cards.jsonl",
        "executable-state-fixtures.json",
        "computed-theory-triage.csv",
        "countermodels.json",
        "tng100-hydro-dmo-source-manifest.json",
        "report.md",
    }
    assert len(first["theory-cards.jsonl"].splitlines()) == 8
    assert b"Theory repair PASS; real-data source BLOCKED" in first["report.md"]
    assert json.loads(first["tng100-hydro-dmo-source-manifest.json"])["status"].startswith(
        "SOURCE_BLOCKED"
    )


def test_atomic_no_clobber(tmp_path) -> None:
    path = tmp_path / "sealed.bin"
    assert capture._atomic_no_clobber(path, b"one") == "CREATED"
    with pytest.raises(capture.DissipativeCaptureV2Error, match="existing artifact differs"):
        capture._atomic_no_clobber(path, b"two")
