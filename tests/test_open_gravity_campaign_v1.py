from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
from sigma_theory_compiler import open_gravity_campaign_v1 as campaign

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_dependency_bindings_are_exact() -> None:
    config = campaign.load_config()
    verified = campaign.verify_dependency_bindings(ROOT, config)
    assert verified["verified_files"] >= 18


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("authority", "campaign_count"), 2),
        (("candidate_contract", "exact_parameter_cells"), 2485),
        (("nuisance_cases", 3, "missing_stellar_to_gas_mass_ratio"), 0.05),
        (("metrics", "minimum_meaningful_improvement"), 0.0),
        (("claim_ceiling", "new_theory_claim"), True),
        (("output_paths", "result"), "work/attacker.json"),
        (("scientific_input_bindings", "sparc", "dataset_raw_sha256"), "0" * 64),
        (("scientific_input_bindings", "xcop", "raw_root"), "C:/forged"),
        (("preparation_access_disclosure", "response_rows_decoded_or_inspected"), 1),
        (("artifact_contract", "in_process_monkeypatch_resistance_claim"), True),
    ],
)
def test_config_mutations_fail(path: tuple[object, ...], value: object) -> None:
    config = campaign.load_config()
    mutated = copy.deepcopy(config)
    cursor: object = mutated
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(campaign.OpenGravityCampaignError):
        campaign.validate_config(mutated)


def test_manifest_is_complete_and_registry_validated() -> None:
    manifest, context = campaign.build_manifest()
    assert len(manifest["candidate_versions"]) == 407
    assert len(manifest["parameter_cells"]) == 2486
    assert len(context["cards"]) == 407
    assert len(campaign._eligible_cells(manifest, "GALAXIES")) == 179
    assert len(campaign._eligible_cells(manifest, "CLUSTERS")) == 1669
    assert {row["lane"] for row in manifest["candidate_versions"]} == {
        "CORE",
        "ADJACENT",
        "ORTHOGONAL",
        "RIVALS_CONTROLS",
        "WILDCARD",
    }


def test_terminal_ledger_is_exact_and_session_terminal() -> None:
    manifest, _ = campaign.build_manifest()
    ledger = campaign.build_terminal_ledger(manifest)
    campaign.validate_terminal_ledger(ledger, manifest)
    assert ledger["session_terminal"] is True
    assert ledger["automatic_second_campaign_allowed"] is False
    mutated = copy.deepcopy(ledger)
    mutated["campaign_ordinal"] = 2
    mutated["ledger_content_sha256"] = campaign._self_hash(mutated, "ledger_content_sha256")
    with pytest.raises(campaign.OpenGravityCampaignError):
        campaign.validate_terminal_ledger(mutated, manifest)


def test_preflight_rebuild_is_zero_access() -> None:
    preflight = campaign.build_preflight()
    assert preflight["manifest"]["candidate_count"] == 407
    assert preflight["manifest"]["parameter_cell_count"] == 2486
    assert preflight["access"] == campaign.ZERO_ACCESS
    assert (
        preflight["preparation_access_disclosure"]
        == campaign.load_config()["preparation_access_disclosure"]
    )
    assert preflight["preparation_access_disclosure"]["response_rows_decoded_or_inspected"] == 0
    assert not campaign.ACCESS_INTENT_PATH.exists()
    assert not campaign.RESULT_PATH.exists()


def test_scientific_input_contracts_are_exact_without_payload_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = campaign.load_config()
    original = clock.load_config(ROOT)
    calls = {"git": 0}

    def no_payload_git(_root: Path, _commit: str, _relative: str, expected_sha256: str) -> None:
        assert expected_sha256 == config["scientific_input_bindings"]["sparc"]["dataset_raw_sha256"]
        calls["git"] += 1

    monkeypatch.setattr(campaign, "_require_git_payload_hash", no_payload_git)
    monkeypatch.setattr(clock, "load_config", lambda _root: copy.deepcopy(original))
    validated = campaign._verify_scientific_input_contracts(ROOT, config)
    assert validated["input_contract"] == original["input_contract"]
    assert calls == {"git": 1}

    for mutator in (
        lambda value: value["input_contract"].__setitem__("raw_root", "C:/FORGED"),
        lambda value: value["input_contract"]["files"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["input_contract"]["files"][0].__setitem__("member", "../escaped.fits"),
        lambda value: value["input_contract"]["files"].pop(),
    ):
        mutated = copy.deepcopy(original)
        mutator(mutated)
        monkeypatch.setattr(clock, "load_config", lambda _root, value=mutated: value)
        with pytest.raises(campaign.OpenGravityCampaignError):
            campaign._verify_scientific_input_contracts(ROOT, config)


def test_committed_sparc_blob_must_match_the_frozen_raw_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_git_show", lambda *_args: b"wrong committed response bytes")
    with pytest.raises(campaign.OpenGravityCampaignError, match="committed scientific input bytes"):
        campaign._require_git_payload_hash(
            ROOT,
            "92bc8bfcdc31714d9b9f69b86b44dc3920613350",
            "configs/sparc_rotation_curves_full_v1.json",
            "dde80c7fc72974358b1370e1978726b87fe1a4048f0880ae79cf513e260a7cf1",
        )


def test_scientific_input_verification_is_after_access_intent() -> None:
    source = campaign.MODULE_PATH.read_text(encoding="utf-8")
    execute = source[source.index("def execute_campaign()") : source.index("def check_result()")]
    assert execute.index("_atomic_no_clobber(root / ACCESS_INTENT_PATH") < execute.index(
        "_verify_scientific_input_contracts(root, config)"
    )
    assert execute.index("_verify_scientific_input_contracts(root, config)") < execute.index(
        "_load_sparc_responses(root, context, config)"
    )


def test_validation_threat_model_is_fresh_official_cli_only() -> None:
    contract = campaign.load_config()["artifact_contract"]
    assert contract["supported_validation_entrypoint"] == "FRESH_PYTHON_PROCESS_OFFICIAL_CLI_ONLY"
    assert contract["in_process_monkeypatch_resistance_claim"] is False


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert campaign._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert campaign._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    assert campaign._atomic_no_clobber(path, b"two\n") == "EXISTING_DIFFERENT"
    assert path.read_bytes() == b"one\n"


def _galaxy() -> SimpleNamespace:
    return SimpleNamespace(
        name="SYNTHETIC",
        radius=(Fraction(1), Fraction(2), Fraction(4), Fraction(8)),
        v_gas=(Fraction(20), Fraction(22), Fraction(24), Fraction(25)),
        v_disk=(Fraction(40), Fraction(45), Fraction(48), Fraction(50)),
        v_bul=(Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        v_obs=(Fraction(50), Fraction(55), Fraction(60), Fraction(62)),
        e_v_obs=(Fraction(2), Fraction(2), Fraction(3), Fraction(3)),
        count=4,
    )


def test_sparc_source_never_passes_response_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}
    original = campaign.adapter.compile_sparc_source_drivers

    def wrapped(radius: object, gas: object, stars: object) -> dict[str, object]:
        seen["arguments"] = 3
        return original(radius, gas, stars)

    monkeypatch.setattr(campaign.adapter, "compile_sparc_source_drivers", wrapped)
    source = campaign._sparc_source(
        _galaxy(),
        {"cell_id": "SYNTHETIC", "disk_ml": 0.5, "bulge_ml": 0.7},
    )
    assert seen == {"arguments": 3}
    assert source["bundle"]["response_inputs"] == 0
    assert source["rows"] == 4


def test_static_and_gp01_factors_are_finite() -> None:
    manifest, context = campaign.build_manifest()
    source = campaign._sparc_source(
        _galaxy(),
        {"cell_id": "SYNTHETIC", "disk_ml": 0.5, "bulge_ml": 0.7},
    )
    concept_packet = {row["concept_id"]: row for row in context["packet"]["twell_rows"]}
    for candidate, cell in campaign._eligible_cells(manifest, "GALAXIES")[:3]:
        factor, _ = campaign._factor_for_cell(
            candidate, cell, concept_packet, source["bundle"], "SPARC"
        )
        assert factor.shape == (257,)
        assert np.all(np.isfinite(factor))
        assert np.all(factor > 0)


def test_loss_rows_equalizes_observable_groups() -> None:
    rows = [
        {
            "row_id": "p1",
            "observed": 2.0,
            "error": 0.1,
            "observable": "pressure",
            "radius_kpc": 1.0,
        },
        {
            "row_id": "p2",
            "observed": 2.0,
            "error": 0.1,
            "observable": "pressure",
            "radius_kpc": 2.0,
        },
        {
            "row_id": "t1",
            "observed": 4.0,
            "error": 0.2,
            "observable": "temperature",
            "radius_kpc": 1.5,
        },
    ]
    score = campaign._loss_rows(
        {"p1": 2.0, "p2": 2.0, "t1": 4.0},
        rows,
        minimum_fractional_error=0.05,
    )
    assert score["loss"] == 0.0
    assert score["by_observable"] == {"pressure": 0.0, "temperature": 0.0}


def test_domain_adjudication_can_retain_zero_survivors() -> None:
    config = campaign.load_config()
    _manifest, context = campaign.build_manifest()
    candidate = {
        "cell_id": "C1",
        "concept_id": "X",
        "scenario_results": [
            {
                "scenario_id": row["cell_id"],
                "mean_loss": 2.0,
                "objects": [
                    {"object": name, "loss": 2.0}
                    for name in context["source_predecessor"]["objects"]["XCOP"]
                ],
            }
            for row in campaign._scenario_rows(config, "CLUSTERS")
        ],
    }
    comparator = {
        "scenario_results": [
            {
                "scenario_id": row["cell_id"],
                "comparator_id": "CONTROL",
                "mean_loss": 1.0,
                "objects": [
                    {"object": name, "loss": 1.0}
                    for name in context["source_predecessor"]["objects"]["XCOP"]
                ],
            }
            for row in campaign._scenario_rows(config, "CLUSTERS")
        ]
    }
    result = campaign._adjudicate_domain("CLUSTERS", [candidate], comparator, context, config)
    assert result[0]["passes"] is False
    assert campaign._cross_domain_adjudication([], result) == []


def test_result_adjudication_is_self_hashing() -> None:
    result = {
        "campaign_id": "OPEN-GRAVITY-CAMPAIGN-v1",
        "result_content_sha256": "a" * 64,
        "counts": {"dashboards": 147},
        "cross_domain_survivors": [],
    }
    adjudication = campaign._result_adjudication(result, [])
    assert adjudication["adjudication_content_sha256"] == campaign._self_hash(
        adjudication, "adjudication_content_sha256"
    )


def test_result_artifact_contract_is_exact_and_empty_index_fails_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest, context = campaign.build_manifest()
    expected = campaign._expected_artifact_paths(context)
    assert len(expected) == 156
    assert len(set(expected)) == 156
    assert (
        sum(
            path.startswith(f"{campaign.ARTIFACT_DIRECTORY.as_posix()}/dashboards/")
            for path in expected
        )
        == 147
    )

    reads = {"count": 0}

    def forbidden_read(_path: Path) -> dict[str, object]:
        reads["count"] += 1
        raise AssertionError("empty index must fail before an artifact read")

    monkeypatch.setattr(campaign, "_read_json", forbidden_read)
    with pytest.raises(campaign.OpenGravityCampaignError, match="artifact index count"):
        campaign._load_result_artifacts(tmp_path, {"artifact_index": []}, context)
    assert reads == {"count": 0}


def test_empty_artifact_payload_bundle_cannot_validate() -> None:
    manifest, context = campaign.build_manifest()
    config = campaign.load_config()
    forged_result = {
        "best_development_cells": {"GALAXIES": "NONE", "CLUSTERS": "NONE"},
        "counts": {"dashboards": 147, "artifacts": 156},
        "cross_domain_adjudication": [],
        "cross_domain_survivors": [],
    }
    with pytest.raises(campaign.OpenGravityCampaignError, match="artifact payload set"):
        campaign._validate_result_artifacts(forged_result, {}, manifest, context, config)


def test_dashboard_status_and_caveats_are_rebuilt_not_trusted() -> None:
    manifest, _context = campaign.build_manifest()
    config = campaign.load_config()
    cell = manifest["parameter_cells"][0]
    cell_id = str(cell["cell_id"])
    candidate_results = [
        {
            "cell_id": cell_id,
            "concept_id": str(cell["exact_value_or_rule"]["concept_id"]),
            "scenario_results": [
                {"objects": [{"object": "OBJECT", "loss": 2.0}]} for _ in range(3)
            ],
        }
    ]
    comparator_summary = {
        "scenario_results": [
            {
                "comparator_id": "CONTROL",
                "scenario_id": f"S{index}",
                "objects": [{"object": "OBJECT", "loss": 1.0}],
            }
            for index in range(3)
        ]
    }
    evidence = {
        "source_profile": [],
        "candidate_state": [],
        "nominal_radial_prediction": [],
    }
    rebuilt = campaign._rebuild_dashboard(
        evidence,
        "GALAXIES",
        "OBJECT",
        {"best_development_cells": {"GALAXIES": cell_id}},
        manifest,
        candidate_results,
        comparator_summary,
        config,
    )
    assert rebuilt["status"] == "COUNTEREXAMPLE"
    assert rebuilt["environment"] == "SOURCE_BLOCKED_NO_ADMITTED_ENVIRONMENT_MAP"
    forged = copy.deepcopy(rebuilt)
    forged["status"] = "SUPPORTS"
    assert forged != rebuilt
    with pytest.raises(campaign.OpenGravityCampaignError, match="legacy submitted-dashboard"):
        campaign._validate_dashboard("FORGED", forged, "GALAXIES", "OBJECT", {}, {}, [], {})


def test_dashboard_evidence_rejects_negative_gain_before_score_use() -> None:
    evidence = {
        "source_profile": [{"radius_kpc": 1.0, "baryonic_acceleration_m_s2": 1.0e-10}],
        "candidate_state": [{"radius_kpc": 1.0, "gain_factor": -1.0}],
        "nominal_radial_prediction": [
            {"radius_kpc": 1.0, "observed": 1.0, "error": 0.1, "predicted": 1.0}
        ],
    }
    with pytest.raises(campaign.OpenGravityCampaignError, match="source/state values"):
        campaign._validate_dashboard_evidence(
            "GALAXIES-OBJECT",
            evidence,
            "GALAXIES",
            "OBJECT",
            {"best_development_cells": {"GALAXIES": "NONE"}},
            [],
        )


def test_campaign_config_is_json_object() -> None:
    value = json.loads((ROOT / campaign.CONFIG_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
