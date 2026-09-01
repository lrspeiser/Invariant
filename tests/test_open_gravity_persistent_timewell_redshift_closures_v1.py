from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_persistent_timewell_redshift_closures_v1 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def test_package_pins_are_exact() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


def test_config_closures_dimensions_sources_and_controls_are_frozen() -> None:
    config = subject.load_config()
    assert [row["id"] for row in config["closures"]] == list(subject._CLOSURE_IDS)
    assert len(config["published_neighbors"]) == 7
    assert len(config["mandatory_controls"]) == 12
    assert config["dimensions"]["tidal_sqrtK"] == "L^-2"
    assert config["dimensions"]["partial_t_psi_dt"] == "1"
    assert config["real_data_preflight"]["response_status"] == "NOT_OPENED_NOT_SCORED"
    assert config["access_contract"]["scientific_response_rows_scored"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "OPEN"),
        (("access_contract", "network_calls_by_builder"), 1),
        (("claim_boundary", "real_data_scored"), True),
        (("real_data_preflight", "frozen_tau_seconds"), [1.0]),
        (("outputs", "receipt"), "elsewhere.json"),
    ],
)
def test_semantic_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(subject.PersistentRedshiftError):
        subject.validate_config(config)


def test_synthetic_design_has_only_endpoint_exact_gradient_degeneracy() -> None:
    signatures, pairs = subject._synthetic_signatures()
    assert len(signatures) == 15 * 9
    assert len(pairs) == 36
    identical = [row for row in pairs if not row["distinguishable"]]
    assert len(identical) == 1
    assert {identical[0]["left"], identical[0]["right"]} == {
        "C00_GR_ENDPOINT",
        "C01_EXACT_GRADIENT_PATH",
    }


def test_gauge_chromatic_history_lens_and_round_trip_falsifiers() -> None:
    fixtures = subject._fixtures()
    by_key = {(row["fixture"], row["case"]): row for row in fixtures}

    gauge_base = by_key[("F03_POTENTIAL_ZERO_SHIFT", "base")]
    gauge_shift = by_key[("F03_POTENTIAL_ZERO_SHIFT", "add_constant")]
    assert subject._predict("C00_GR_ENDPOINT", gauge_base) == subject._predict(
        "C00_GR_ENDPOINT", gauge_shift
    )
    assert subject._predict("C02_POTENTIAL_COLUMN", gauge_base) != subject._predict(
        "C02_POTENTIAL_COLUMN", gauge_shift
    )

    low = by_key[("F05_TWO_FREQUENCIES", "low")]
    high = by_key[("F05_TWO_FREQUENCIES", "high")]
    low_shift = subject._predict("C04_DISPERSIVE_PERSISTENT_MEDIUM", low)[0]
    high_shift = subject._predict("C04_DISPERSIVE_PERSISTENT_MEDIUM", high)[0]
    assert low_shift / high_shift == pytest.approx(16.0)
    assert subject._predict("C06_PATH_MEMORY_OPACITY", low) == subject._predict(
        "C06_PATH_MEMORY_OPACITY", high
    )

    immediate = by_key[("F04_SOURCE_SWITCH_OFF_HISTORY", "immediate")]
    late = by_key[("F04_SOURCE_SWITCH_OFF_HISTORY", "late")]
    assert subject._predict("C05_ENDPOINT_MEMORY", immediate) != subject._predict(
        "C05_ENDPOINT_MEMORY", late
    )

    image_a = by_key[("F06_LENS_TWO_IMAGES", "image_A")]
    image_b = by_key[("F06_LENS_TWO_IMAGES", "image_B")]
    assert subject._predict("C00_GR_ENDPOINT", image_a) == subject._predict(
        "C00_GR_ENDPOINT", image_b
    )
    assert subject._predict("C03_TIDAL_CURVATURE_COLUMN", image_a) != subject._predict(
        "C03_TIDAL_CURVATURE_COLUMN", image_b
    )

    outbound = by_key[("F07_ROUND_TRIP", "outbound")]
    returning = by_key[("F07_ROUND_TRIP", "return")]
    assert subject._predict("C00_GR_ENDPOINT", outbound)[0] + subject._predict(
        "C00_GR_ENDPOINT", returning
    )[0] == pytest.approx(0.0)
    assert (
        subject._predict("C06_PATH_MEMORY_OPACITY", outbound)[0]
        + subject._predict("C06_PATH_MEMORY_OPACITY", returning)[0]
        < 0.0
    )


def test_expansion_is_retained_as_control_not_replaced() -> None:
    expansion = next(
        row for row in subject._fixtures() if row["fixture"] == "F08_EXPANSION_TIME_DILATION"
    )
    for closure_id in subject._CLOSURE_IDS:
        residual, stretch = subject._predict(closure_id, expansion)
        assert residual == 0.0
        assert stretch == 1.5


def test_ranking_retains_failures_and_known_comparators() -> None:
    ledger = subject._closure_ledger(subject.load_config())
    ranked = sorted(
        (int(row["candidate_rank"]), row["closure_id"])
        for row in ledger
        if row["candidate_rank"] != ""
    )
    assert ranked == [
        (1, "C08_CAUSAL_METRIC_MEMORY_PATH"),
        (2, "C05_ENDPOINT_MEMORY"),
        (3, "C03_TIDAL_CURVATURE_COLUMN"),
        (4, "C06_PATH_MEMORY_OPACITY"),
        (5, "C04_DISPERSIVE_PERSISTENT_MEDIUM"),
    ]
    dispositions = {row["closure_id"]: row["disposition"] for row in ledger}
    assert dispositions["C01_EXACT_GRADIENT_PATH"] == "FAIL_NOT_DISTINCT"
    assert dispositions["C02_POTENTIAL_COLUMN"] == "FAIL_GAUGE_ENERGY_RECIPROCITY"
    assert dispositions["C07_TIME_VARYING_METRIC_PATH"] == "KNOWN_COMPARATOR"


def test_receipt_and_artifacts_are_deterministic_and_claim_bounded() -> None:
    first, first_payloads = subject.build_receipt()
    second, second_payloads = subject.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert first["summary"]["closures"] == 9
    assert first["summary"]["synthetic_signature_rows"] == 135
    assert first["summary"]["pairwise_comparisons"] == 36
    assert first["summary"]["exact_degeneracies"] == 1
    assert first["summary"]["real_response_rows_scored"] == 0
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["gravity_discovery_established"] is False
    report = first_payloads["report.md"].decode()
    assert "C08 causal metric-memory path" in report
    assert "BLOCKED for an empirical gravity claim" in report
    assert "not replacements for cosmological expansion" in report


def test_written_canonical_package_checks_and_forgery_rejects() -> None:
    observed = subject.check_package()
    assert observed["status"].startswith("PASS_RESPONSE_FREE")
    forged = copy.deepcopy(observed)
    forged["summary"]["real_response_rows_scored"] = 1
    assert forged != subject.build_receipt()[0]


def test_no_raw_or_paid_access_and_no_payload_files_are_bound() -> None:
    config = subject.load_config()
    assert set(config["access_contract"].values()) == {0}
    for binding in config["local_bindings"]:
        assert Path(binding["path"]).suffix == ".json"
        assert "raw" not in Path(binding["path"]).name.lower()
    assert config["real_data_preflight"]["source_status"].endswith(
        "EXACT_PAYLOAD_URLS_AND_HASHES_NOT_YET_RECEIPTED"
    )
