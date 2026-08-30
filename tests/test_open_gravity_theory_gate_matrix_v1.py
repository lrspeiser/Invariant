from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_theory_gate_matrix_v1 as gates


@pytest.fixture(scope="module")
def packet() -> tuple[dict, list[dict], list[dict]]:
    config, _predecessor, catalog = gates.load_inputs()
    rows = list(gates.iter_gate_rows(config, catalog))
    return config, catalog, rows


def test_config_and_committed_bindings(packet: tuple) -> None:
    config, _catalog, _rows = packet
    gates.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "REGISTRY",
        "SOURCE_CATALOG",
        "FULL_3D_FOUNDATION",
        "NEWTON_AQUAL_QUMOND_BASELINES",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_exact_420_by_25_matrix(packet: tuple) -> None:
    config, catalog, rows = packet
    assert len(catalog) == 420
    assert len(config["gate_vocabulary"]) == 25
    assert len(rows) == 10_500
    assert rows[0]["mechanism_id"] == "TW2-A01-D01"
    assert rows[0]["gate_id"] == "TG01_DIMENSIONS_LIMITS"
    assert rows[-1]["mechanism_id"] == "QG13"
    assert rows[-1]["gate_id"] == "TG25_REAL_DATA_CAMPAIGN"


def test_every_mechanism_has_every_gate_once(packet: tuple) -> None:
    config, _catalog, rows = packet
    expected = {row["id"] for row in config["gate_vocabulary"]}
    observed: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        observed[row["mechanism_id"]].append(row["gate_id"])
    assert len(observed) == 420
    assert all(len(values) == 25 and set(values) == expected for values in observed.values())


def test_exact_row_schema_and_status_vocabulary(packet: tuple) -> None:
    config, _catalog, rows = packet
    schema = set(config["matrix_contract"]["row_fields"])
    allowed = set(config["status_vocabulary"])
    assert all(set(row) == schema for row in rows)
    assert {row["evidence_status"] for row in rows} <= allowed


def test_ontology_class_partition_is_total(packet: tuple) -> None:
    _config, _catalog, rows = packet
    mechanism_classes: dict[str, str] = {}
    for row in rows:
        mechanism_classes.setdefault(row["mechanism_id"], row["ontology_class"])
        assert mechanism_classes[row["mechanism_id"]] == row["ontology_class"]
    counts = Counter(mechanism_classes.values())
    assert sum(counts.values()) == 420
    assert counts["GP01"] == 7
    assert counts["GRAVITY_LIGHT_ONTOLOGY"] == 13
    assert counts["STOCHASTIC"] == 21


def test_aqual_has_only_bound_primary_and_target_free_passes(packet: tuple) -> None:
    _config, _catalog, rows = packet
    aqual = {row["gate_id"]: row for row in rows if row["mechanism_id"] == "GP01-AQUAL"}
    assert aqual["TG04_EQUATIONS_OPERATOR"]["evidence_status"] == (
        "PASS_PRIMARY_SOURCE_NOT_INDEPENDENTLY_REDERIVED"
    )
    assert aqual["TG07_FULL_3D_SOLVER"]["evidence_status"] == "PASS_TARGET_FREE"
    assert aqual["TG19_PHOTON_LENSING"]["evidence_status"] == ("BLOCKED_MISSING_DEFINITION")
    assert aqual["TG24_REAL_3D_SOURCE"]["evidence_status"] == "BLOCKED_MISSING_SOURCE"


def test_static_formula_does_not_overclaim_fundamental_health(packet: tuple) -> None:
    _config, _catalog, rows = packet
    sample = {row["gate_id"]: row for row in rows if row["mechanism_id"] == "TW2-A01-D01"}
    assert sample["TG01_DIMENSIONS_LIMITS"]["evidence_status"] == "PASS_REGISTERED"
    assert sample["TG07_FULL_3D_SOLVER"]["evidence_status"] == "BLOCKED_MISSING_SOLVER"
    assert sample["TG11_PRINCIPAL_SYMBOL"]["required_for_current_claim"] is False
    assert sample["TG11_PRINCIPAL_SYMBOL"]["evidence_status"] == ("NOT_APPLICABLE_CURRENT_SCOPE")


def test_quantum_gate_is_required_only_for_quantum_claims(packet: tuple) -> None:
    _config, _catalog, rows = packet
    qg03 = next(
        row
        for row in rows
        if row["mechanism_id"] == "QG03" and row["gate_id"] == "TG17_QUANTUM_UNITARITY"
    )
    classical = next(
        row
        for row in rows
        if row["mechanism_id"] == "QG01" and row["gate_id"] == "TG17_QUANTUM_UNITARITY"
    )
    assert qg03["required_for_current_claim"] is True
    assert qg03["evidence_status"] == "REQUIRED_UNRUN"
    assert classical["required_for_current_claim"] is False
    assert classical["evidence_status"] == "NOT_APPLICABLE_CURRENT_SCOPE"


def test_classical_gr_and_wave_nodes_are_external_controls_not_local_rederivations(
    packet: tuple,
) -> None:
    _config, _catalog, rows = packet
    for mechanism_id in ("QG01", "QG02"):
        selected = [
            row
            for row in rows
            if row["mechanism_id"] == mechanism_id
            and row["gate_id"] in {"TG02_FIELD_STATE", "TG04_EQUATIONS_OPERATOR"}
        ]
        assert {row["evidence_status"] for row in selected} == {
            "PASS_PRIMARY_SOURCE_NOT_INDEPENDENTLY_REDERIVED"
        }


def test_incomplete_action_is_quarantined_at_every_gate(packet: tuple) -> None:
    _config, _catalog, rows = packet
    action = [row for row in rows if row["mechanism_id"] == "GP01-ACTION_PLACEHOLDER"]
    assert len(action) == 25
    assert {row["evidence_status"] for row in action} == {"INCOMPLETE_QUARANTINE"}


def test_real_source_and_campaign_gates_are_fail_closed_for_every_mechanism(
    packet: tuple,
) -> None:
    _config, _catalog, rows = packet
    source = [row for row in rows if row["gate_id"] == "TG24_REAL_3D_SOURCE"]
    campaign = [row for row in rows if row["gate_id"] == "TG25_REAL_DATA_CAMPAIGN"]
    assert len(source) == len(campaign) == 420
    assert {row["evidence_status"] for row in source} <= {
        "BLOCKED_MISSING_SOURCE",
        "INCOMPLETE_QUARANTINE",
    }
    assert {row["evidence_status"] for row in campaign} <= {
        "BLOCKED_UPSTREAM_GATES",
        "INCOMPLETE_QUARANTINE",
    }


def test_no_observational_pass_or_score_exists(packet: tuple) -> None:
    _config, _catalog, rows = packet
    assert not any("OBSERV" in row["evidence_status"] for row in rows)
    assert not any("SCORE" in row["evidence_status"] for row in rows)


def test_gate_contract_hash_is_deterministic(packet: tuple) -> None:
    config, catalog, rows = packet
    again = list(gates.iter_gate_rows(config, catalog))
    assert rows == again
    assert gates._stream_root(rows) == gates._stream_root(again)


@pytest.mark.parametrize(
    "section",
    (
        "bindings",
        "gate_vocabulary",
        "status_vocabulary",
        "ontology_classes",
        "route_rules",
        "matrix_contract",
        "access_contract",
        "claim_boundary",
    ),
)
def test_each_semantic_section_rejects_mutation(packet: tuple, section: str) -> None:
    config, _catalog, _rows = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(gates.TheoryGateError, match=f"section {section} changed"):
        gates.validate_config(changed)


def test_coherent_authority_rebind_is_rejected(packet: tuple) -> None:
    config, _catalog, _rows = packet
    changed = copy.deepcopy(config)
    changed["matrix_contract"]["observational_passes_allowed"] = 420
    changed["section_sha256"]["matrix_contract"] = gates.content_sha256(changed["matrix_contract"])
    with pytest.raises(gates.TheoryGateError, match="observational authority changed"):
        gates.validate_config(changed)


def test_noncanonical_output_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(gates, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(gates, "_read_json", forbidden)
    with pytest.raises(gates.TheoryGateError, match="output path changed"):
        gates.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_forgery_rejection(packet: tuple) -> None:
    _config, _catalog, _rows = packet
    receipt = gates.build_receipt()
    gates.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["matrix"]["observational_passes"] = 420
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = gates.content_sha256(body)
    with pytest.raises(gates.TheoryGateError, match="not reproducible"):
        gates.validate_receipt_payload(forged)


def test_zero_access_and_narrow_claim(packet: tuple) -> None:
    config, _catalog, _rows = packet
    receipt = gates.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert config["matrix_contract"]["observational_passes_allowed"] == 0
    assert (
        "that routed gates have been executed unless explicitly bound"
        in receipt["claim_boundary"]["does_not_establish"]
    )
