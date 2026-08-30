from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_gravity_light_quantum_cards_v1 as cards


@pytest.fixture(scope="module")
def packet() -> tuple[dict, list[dict], list[dict]]:
    config = cards.load_config()
    probes = cards.run_target_free_probes(config["cards"])
    gate_ids = [f"TG{i:02d}_PLACEHOLDER" for i in range(1, 26)]
    rows = cards.gate_projection(config["cards"], gate_ids)
    return config, probes, rows


def test_config_and_exact_committed_bindings(packet: tuple) -> None:
    config, _probes, _rows = packet
    cards.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "REGISTRY",
        "THEORY_GATE_MATRIX",
        "CLASSICAL_3D_CONTROLS",
    ]
    assert all(len(binding["commit"]) == 40 for binding in config["bindings"])
    assert sum(len(binding["artifacts"]) for binding in config["bindings"]) == 6


def test_exact_thirteen_registry_cards(packet: tuple) -> None:
    config, _probes, _rows = packet
    assert [card["id"] for card in config["cards"]] == [f"QG{i:02d}" for i in range(1, 14)]
    assert {card["id"]: card["name"] for card in config["cards"]} == cards._EXPECTED_CARD_NAMES
    assert len(config["primary_sources"]) == 11


def test_every_card_has_exact_typed_schema(packet: tuple) -> None:
    config, _probes, _rows = packet
    expected = set(config["card_fields"])
    assert len(expected) == 21
    assert all(set(card) == expected for card in config["cards"])
    assert all(card["falsifiers"] for card in config["cards"])
    assert all(card["required_next_artifacts"] for card in config["cards"])


def test_matter_photon_clock_and_radiation_are_separate(packet: tuple) -> None:
    config, _probes, _rows = packet
    for card in config["cards"][:-1]:
        values = [
            card["matter_observable"],
            card["photon_lensing_observable"],
            card["clock_redshift_observable"],
            card["radiation_observable"],
        ]
        assert all(type(value) is str and value for value in values)
        assert len(set(values)) == 4


def test_only_wildcard_is_quarantined_and_undefined(packet: tuple) -> None:
    config, probes, _rows = packet
    wildcard = config["cards"][-1]
    assert wildcard["id"] == "QG13"
    assert wildcard["card_status"] == "INCOMPLETE_QUARANTINE"
    assert wildcard["state_space"] == "UNDEFINED"
    assert probes[-1] == {
        "card_id": "QG13",
        "probe_status": "INCOMPLETE_QUARANTINE",
        "metrics": {"execution_attempts": 0},
    }
    assert all(cards._all_text_defined(card) for card in config["cards"][:-1])


def test_twelve_bounded_target_free_probes_pass(packet: tuple) -> None:
    _config, probes, _rows = packet
    assert len(probes) == 13
    assert Counter(row["probe_status"] for row in probes) == {
        "PASS_TARGET_FREE": 12,
        "INCOMPLETE_QUARANTINE": 1,
    }
    assert all(row["metrics"] for row in probes)


def test_classical_tensor_wave_probe_is_transverse_traceless(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG02")["metrics"]
    assert result["trace"] == 0.0
    assert result["longitudinal_norm"] == 0.0
    assert result["dispersion_residual"] == 0.0


def test_quantum_state_probe_preserves_trace_and_purity(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG04")["metrics"]
    assert result["trace"] == pytest.approx(1.0, abs=1.0e-15)
    assert result["purity"] == pytest.approx(1.0, abs=1.0e-15)
    assert result["density_determinant"] == 0.0


def test_massive_carrier_probe_has_finite_range_and_subluminal_group_speed(
    packet: tuple,
) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG05")["metrics"]
    assert 0.0 < result["group_speed_over_c"] < 1.0
    assert result["yukawa_local_limit_error"] < 1.1e-7


def test_extra_mode_probe_retains_designed_ghost_failure(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG06")["metrics"]
    assert result["healthy_kinetic_minimum"] > 0.0
    assert result["designed_ghost_minimum"] < 0.0


def test_entanglement_probe_distinguishes_nonlocal_from_additive_phase(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG07")["metrics"]
    assert result["entangling_concurrence"] == pytest.approx(1.0, abs=1.0e-15)
    assert result["additive_local_phase_concurrence"] < 1.0e-15


def test_stochastic_probe_requires_positive_covariance(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG08")["metrics"]
    assert result["covariance_determinant"] > 0.0
    assert result["covariance_trace"] > 0.0
    assert result["zero_noise_limit_residual"] == 0.0


def test_entropic_probe_marks_assumptions_not_microscopic_derivation(packet: tuple) -> None:
    config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG09")["metrics"]
    card = next(row for row in config["cards"] if row["id"] == "QG09")
    assert result["normalized_entropic_force_residual"] < 1.0e-15
    assert result["assumed_relations"] == 2
    assert "not counted as a microscopic derivation" in card["claim_boundary"]


def test_discrete_probe_is_relabeling_invariant(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG10")["metrics"]
    assert result["events"] == 4
    assert result["causal_relations"] == result["relabeling_relation_count"] == 5


def test_medium_probe_rejects_negative_sound_speed_squared(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG11")["metrics"]
    assert result["sound_speed_squared"] > 0.0
    assert result["designed_bad_sound_speed_squared"] < 0.0


def test_memory_probe_is_causal_normalized_and_has_local_limit(packet: tuple) -> None:
    _config, probes, _rows = packet
    result = next(row for row in probes if row["card_id"] == "QG12")["metrics"]
    assert result["normalization_residual"] < 1.0e-15
    assert result["advanced_support_weight"] == 0.0
    assert result["short_memory_present_weight"] > result["broad_present_weight"]


def test_exact_thirteen_by_twenty_five_projection_uses_real_gate_ids(packet: tuple) -> None:
    config, _probes, _rows = packet
    gate_config = cards._read_json(
        cards._ROOT / "configs/open_gravity_theory_gate_matrix_v1.json", "gate config"
    )
    gate_ids = [row["id"] for row in gate_config["gate_vocabulary"]]
    rows = cards.gate_projection(config["cards"], gate_ids)
    assert len(rows) == 325
    observed: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        observed[row["card_id"]].append(row["gate_id"])
    assert len(observed) == 13
    assert all(values == gate_ids for values in observed.values())


def test_quantum_unitarity_gate_is_required_only_for_quantum_cards(packet: tuple) -> None:
    config, _probes, _rows = packet
    gate_ids = [f"TG{i:02d}_PLACEHOLDER" for i in range(1, 26)]
    gate_ids[16] = "TG17_QUANTUM_UNITARITY"
    rows = cards.gate_projection(config["cards"], gate_ids)
    statuses = {
        row["card_id"]: row["evidence_status"]
        for row in rows
        if row["gate_id"] == "TG17_QUANTUM_UNITARITY"
    }
    assert {card_id for card_id, status in statuses.items() if status == "REQUIRED_UNRUN"} == (
        cards._QUANTUM_IDS
    )
    assert statuses["QG01"] == "NOT_APPLICABLE_CURRENT_SCOPE"
    assert statuses["QG13"] == "INCOMPLETE_QUARANTINE"


def test_real_source_and_campaign_gates_stay_closed(packet: tuple) -> None:
    config, _probes, _rows = packet
    gate_ids = ["TG24_REAL_3D_SOURCE", "TG25_REAL_DATA_CAMPAIGN"]
    rows = cards.gate_projection(config["cards"], gate_ids)
    for row in rows:
        if row["card_id"] == "QG13":
            assert row["evidence_status"] == "INCOMPLETE_QUARANTINE"
        elif row["gate_id"] == "TG24_REAL_3D_SOURCE":
            assert row["evidence_status"] == "BLOCKED_MISSING_SOURCE"
        else:
            assert row["evidence_status"] == "BLOCKED_UPSTREAM_GATES"


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "primary_sources",
        "card_fields",
        "cards",
        "probe_contract",
        "status_policy",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple, section: str) -> None:
    config, _probes, _rows = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(cards.TheoryCardError, match="changed"):
        cards.validate_config(changed)


def test_coherent_observational_authority_rebind_is_rejected(packet: tuple) -> None:
    config, _probes, _rows = packet
    changed = copy.deepcopy(config)
    changed["probe_contract"]["observational_passes"] = 13
    with pytest.raises(cards.TheoryCardError, match="probe authority changed|semantics changed"):
        cards.validate_config(changed)


def test_coherent_wildcard_promotion_is_rejected(packet: tuple) -> None:
    config, _probes, _rows = packet
    changed = copy.deepcopy(config)
    changed["cards"][-1]["card_status"] = "ESTABLISHED"
    changed["cards"][-1]["state_space"] = "anything"
    with pytest.raises(cards.TheoryCardError, match="wildcard promoted|semantics changed"):
        cards.validate_config(changed)


def test_gate_contract_hash_is_deterministic(packet: tuple) -> None:
    config, _probes, _rows = packet
    gate_ids = [f"TG{i:02d}_PLACEHOLDER" for i in range(1, 26)]
    first = cards.gate_projection(config["cards"], gate_ids)
    second = cards.gate_projection(config["cards"], gate_ids)
    assert first == second
    assert cards._stream_root(first) == cards._stream_root(second)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(cards, "OUTPUT_PATH", tmp_path / "response.json")
    monkeypatch.setattr(cards, "_read_json", forbidden)
    with pytest.raises(cards.TheoryCardError, match="output path changed"):
        cards.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple) -> None:
    _config, _probes, _rows = packet
    receipt = cards.build_receipt()
    cards.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["gate_projection"]["observational_passes"] = 13
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = cards.content_sha256(body)
    with pytest.raises(cards.TheoryCardError, match="not reproducible"):
        cards.validate_receipt_payload(forged)


def test_receipt_zero_access_and_narrow_claim(packet: tuple) -> None:
    _config, _probes, _rows = packet
    receipt = cards.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert receipt["gate_projection"]["observational_passes"] == 0
    assert "a new fundamental theory" in receipt["claim_boundary"]["does_not_establish"]
    assert "a graviton detection" in receipt["claim_boundary"]["does_not_establish"]
