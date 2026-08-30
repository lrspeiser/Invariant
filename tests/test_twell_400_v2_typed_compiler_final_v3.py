from __future__ import annotations

import copy
import os
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_registry_foundation_v1 as registry
from sigma_theory_compiler import twell_400_v2_typed_compiler_final_v3 as twell
from sigma_theory_compiler.twell_400_v2_typed_compiler_final_v3 import (
    CARDS_PATH,
    CONFIG_PATH,
    EXPECTED_CARDS_PATH,
    EXPECTED_CONFIG_CANONICAL_SHA256,
    EXPECTED_RECEIPT_PATH,
    EXPECTED_SECTION_SEALS,
    EXPECTED_UNSEALED_ROOT_SHA256,
    MODULE_PATH,
    RECEIPT_PATH,
    TEST_PATH,
    FinalTwellCompilerError,
    _atomic_packet_no_clobber,
    _canonical_packet_paths,
    _sha256_bytes,
    build_packet,
    cards_bytes,
    check_packet,
    content_sha256,
    load_config,
    main,
    ordered_concept_ids,
    receipt_content_sha256,
    stream_root,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def packet() -> tuple[list[dict], bytes, dict]:
    return build_packet(ROOT)


def _row(rows: list[dict], concept_id: str) -> dict:
    return next(row for row in rows if row["concept_id"] == concept_id)


def test_final_config_and_all_sections_are_exactly_sealed() -> None:
    config = load_config(ROOT)
    assert content_sha256(config) == EXPECTED_CONFIG_CANONICAL_SHA256
    assert config["section_seals"] == {
        **EXPECTED_SECTION_SEALS,
        "unsealed_root_sha256": EXPECTED_UNSEALED_ROOT_SHA256,
    }


@pytest.mark.parametrize("section", tuple(EXPECTED_SECTION_SEALS))
def test_every_final_section_mutation_fails_closed(section: str) -> None:
    config = copy.deepcopy(load_config(ROOT))
    value = config[section]
    if isinstance(value, list):
        if "sha256" in value[0]:
            value[0]["sha256"] = "0" * 64
        else:
            value[0]["status"] += "_MUTATED"
    else:
        first = next(iter(value))
        if isinstance(value[first], bool):
            value[first] = not value[first]
        elif isinstance(value[first], str):
            value[first] += "_MUTATED"
        else:
            raise TypeError(section)
    with pytest.raises(FinalTwellCompilerError, match="sealed section changed"):
        validate_config(config)


def test_coordinated_mutation_fails_hardcoded_seal() -> None:
    config = copy.deepcopy(load_config(ROOT))
    config["lane_contract"]["expected_lane_counts"]["CORE"] += 1
    config["section_seals"]["lane_contract"] = content_sha256(config["lane_contract"])
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    config["section_seals"]["unsealed_root_sha256"] = content_sha256(unsealed)
    with pytest.raises(FinalTwellCompilerError, match="sealed section changed"):
        validate_config(config)


def test_final_packet_retains_exact_400_cards_and_1184_computed_cells(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, receipt = packet
    assert [row["concept_id"] for row in rows] == ordered_concept_ids()
    assert receipt["enumeration"] == {
        "atomic_count": 380,
        "compound_count": 20,
        "total_count": 400,
        "parameter_cell_count": 1184,
        "cartesian_cell_count": 1182,
        "compound_override_evidence_count": 2,
        "ordered_concept_ids_sha256": (
            "7388f8982c5014ef6c365d00aa780ba2ecb8b8b3f6786658fb3db36b64c29c5f"
        ),
    }
    assert receipt["exact_cell_evidence"]["passed_card_count"] == 400
    assert receipt["exact_cell_evidence"]["passed_cell_count"] == 1184
    assert receipt["exact_cell_evidence"]["failed_cell_count"] == 0
    assert receipt["exact_cell_evidence"]["maximum_computed_operator_residual"] <= 1e-10


def test_every_live_card_is_schema_valid_exact_initial_registration(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, _receipt = packet
    schema = registry.load_schemas(ROOT)["mechanism_card"]
    for row in rows:
        card = row["card"]
        assert registry.schema_errors(card, schema) == []
        assert registry.mechanism_card_admission(card, schema) == {
            "eligible": True,
            "status": "READY_FOR_THEORY_GATES",
            "errors": [],
        }
        assert card["card_id"] == f"{row['concept_id']}@2.2.0"
        assert card["semantic_version"] == "2.2.0"
        assert card["hashes"]["formula_sha256"] == registry.mechanism_formula_sha256(card)
        assert card["parents"] == []
        assert card["version_change"] == {
            "kind": "INITIAL_REGISTRATION",
            "previous_card_id": None,
            "previous_card_sha256": None,
            "changed_facets": [],
            "prior_result_retained": False,
            "replay_all_affected": False,
        }
        assert (
            sum(
                artifact.startswith("COUNTEREVIDENCE_ONLY:")
                for artifact in card["provenance"]["origin_artifacts"]
            )
            == 6
        )
        assert all(isinstance(cell["unit"], str) for cell in card["parameter_cells"])


def test_build_never_invokes_or_claims_registry_version_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_transition(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("initial admission must not invoke version-transition validation")

    monkeypatch.setattr(registry, "validate_version_transition", forbidden_transition)
    rows, _payload, receipt = build_packet(ROOT)
    assert len(rows) == 400
    assert receipt["registration_semantics"] == {
        "card_semantic_version": "2.2.0",
        "initial_registration_count": 400,
        "all_parents_empty": True,
        "all_previous_card_ids_null": True,
        "all_previous_card_hashes_null": True,
        "all_changed_facets_empty": True,
        "all_prior_result_retained_false": True,
        "interim_packets_committed_to_registry": False,
        "ordinary_successor_claimed": False,
        "rewrite_or_migration_claimed": False,
        "registry_validate_version_transition_invoked": False,
    }
    assert receipt["claim_boundary"]["ordinary_successor_claimed"] is False
    assert receipt["claim_boundary"]["rewrite_or_migration_claimed"] is False
    assert receipt["counterevidence"]["artifacts_are_registered_parents"] is False
    assert receipt["counterevidence"]["artifacts_are_revision_predecessors"] is False


def test_manifest_input_rows_bind_complete_live_cards_and_all_five_lanes(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, receipt = packet
    candidates = [row["manifest_input"] for row in rows]
    cards = [row["card"] for row in rows]
    assert Counter(row["lane"] for row in candidates) == {
        "CORE": 140,
        "ADJACENT": 10,
        "ORTHOGONAL": 160,
        "RIVALS_CONTROLS": 80,
        "WILDCARD": 10,
    }
    assert {row["lane"] for row in candidates} == set(registry.LANES)
    assert all(row["candidate_status"] == "REGISTERED_THEORY_ONLY" for row in candidates)
    assert all(
        execution["execution_disposition"] == "THEORY_ONLY"
        and execution["eligible"] is False
        and execution["scored"] is False
        for row in candidates
        for execution in row["domain_execution"].values()
    )
    assert (
        registry.mechanism_card_set_sha256(cards)
        == receipt["registry_manifest_input"]["mechanism_card_set_sha256"]
    )
    assert (
        registry.campaign_equivalence_ledger_sha256(candidates)
        == receipt["registry_manifest_input"]["equivalence_ledger_sha256"]
    )
    for row, card in zip(candidates, cards, strict=True):
        assert row["candidate_id"] == card["stable_concept_id"]
        assert row["card_sha256"] == registry.content_sha256(card)
        assert row["formula_sha256"] == card["hashes"]["formula_sha256"]
        assert row["equivalence_fingerprint_sha256"] == (
            registry.equivalence_fingerprint_sha256(card)
        )


def test_qg_analogy_and_rival_rows_remain_theory_only_catalogs(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    _rows, _payload, receipt = packet
    assert receipt["theory_only_catalogs"] == {
        "qg_ontology_rows": 13,
        "light_gravity_analogy_rows": 13,
        "rival_comparator_rows": 11,
        "catalog_rows_are_not_duplicated_or_scored_cards": True,
    }


def test_source_matrix_and_adapter_are_deferred_and_no_domain_is_planned(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, receipt = packet
    assert receipt["source_gate"]["source_availability_v1_used_for_readiness"] is False
    assert receipt["source_gate"]["static_adapter_used_for_readiness"] is False
    assert receipt["claim_boundary"]["source_availability_final_bound"] is False
    assert receipt["claim_boundary"]["campaign_manifest_frozen"] is False
    assert receipt["claim_boundary"]["campaign_execution_authority"] is False
    assert all(
        domain["execution_disposition"] == "THEORY_ONLY"
        for row in rows
        for domain in row["manifest_input"]["domain_execution"].values()
    )


def test_preflight_no_anchor_fact_is_hard_bound_without_empirical_failure_claim(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    _rows, _payload, receipt = packet
    assert (
        "T1/T2 transport is source blocked on all eight"
        in receipt["source_gate"]["gp01_preflight_fact"]
    )
    assert receipt["claim_boundary"]["scientific_validity_claimed"] is False
    assert receipt["claim_boundary"]["response_scoring_authorized"] is False


def test_parameter_branches_remain_distinct_in_final_packet(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, _receipt = packet
    cases = [
        ("TW2-A06-D01", "ell"),
        ("TW2-A10-D01", "u_c"),
        ("TW2-A11-D01", "s_c"),
        ("TW2-A12-D01", "mu"),
        ("TW2-A14-D01", "k"),
        ("TW2-A16-D01", "tau"),
        ("TW2-A18-D01", "sigma"),
        ("TW2-A19-D01", "kappa"),
    ]
    for concept_id, parameter in cases:
        row = _row(rows, concept_id)
        cells = [cell for cell in row["cell_results"] if cell["parameters"]["lambda"] == 0.25]
        grouped = {
            cell["parameters"][parameter]: next(
                fixture["output_sha256"]
                for fixture in cell["fixture_results"]
                if fixture["fixture"] == "SMOOTH_BOUNDED_SOURCE"
            )
            for cell in cells
        }
        assert len(grouped) == 2
        assert len(set(grouped.values())) == 2


def test_build_reads_only_hard_bound_metadata_and_never_deferred_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT)
    allowed = {
        (ROOT / CONFIG_PATH).resolve(),
        (ROOT / MODULE_PATH).resolve(),
        (ROOT / TEST_PATH).resolve(),
    }
    for row in config["hard_bindings"]:
        path = Path(row["path"])
        allowed.add((path if path.is_absolute() else ROOT / path).resolve())
    deferred = {(ROOT / row["path"]).resolve() for row in config["deferred_bindings"]}
    original = Path.read_bytes
    opened: list[Path] = []

    def traced(path: Path) -> bytes:
        resolved = path.resolve()
        opened.append(resolved)
        assert resolved in allowed
        assert resolved not in deferred
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", traced)
    rows, payload, receipt = build_packet(ROOT)
    assert len(rows) == len(payload.splitlines()) == 400
    assert set(opened) == allowed
    assert receipt["access_audit"]["deferred_source_or_adapter_files_opened"] == 0
    assert receipt["access_audit"]["astronomy_response_payloads_opened"] == 0


def test_stream_and_receipt_are_deterministic_and_rooted(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, payload, receipt = packet
    assert payload == cards_bytes(rows)
    assert receipt["stream"]["ordered_line_root_sha256"] == stream_root(rows)
    assert receipt["stream"]["file_sha256"] == _sha256_bytes(payload)
    assert receipt["receipt_content_sha256"] == receipt_content_sha256(receipt)
    assert build_packet(ROOT)[1:] == (payload, receipt)


def test_atomic_two_file_no_clobber_and_adversarial_receipt_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = tmp_path / "packet" / "cards.jsonl"
    receipt = tmp_path / "packet" / "receipt.json"
    _atomic_packet_no_clobber(cards, b"cards\n", receipt, b"receipt\n")
    with pytest.raises(FinalTwellCompilerError, match="refusing to overwrite"):
        _atomic_packet_no_clobber(cards, b"changed\n", receipt, b"changed\n")
    assert cards.read_bytes() == b"cards\n"
    assert receipt.read_bytes() == b"receipt\n"

    race_cards = tmp_path / "race" / "cards.jsonl"
    race_receipt = tmp_path / "race" / "receipt.json"
    real_link = os.link
    calls = 0

    def racing_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_link(source, destination)
            return
        race_receipt.write_bytes(b"competitor\n")
        raise FileExistsError("receipt race")

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FinalTwellCompilerError, match="refusing to overwrite"):
        _atomic_packet_no_clobber(race_cards, b"our cards\n", race_receipt, b"our receipt\n")
    assert not race_cards.exists()
    assert race_receipt.read_bytes() == b"competitor\n"


def _named(item_id: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "definition": f"frozen theory-only definition for {item_id}",
        "implementation_sha256": content_sha256(item_id),
    }


def _partition(partition_id: str, role: str, object_ids: list[str]) -> dict[str, object]:
    return {
        "partition_id": partition_id,
        "role": role,
        "anonymous_object_ids": object_ids,
        "object_ledger_sha256": registry.partition_object_ledger_sha256(object_ids),
        "data_contract_sha256": content_sha256(f"data-contract:{partition_id}"),
    }


def _manifest_for_registry_validator(rows: list[dict]) -> dict[str, object]:
    config = registry.load_config(ROOT)
    cards = [row["card"] for row in rows]
    candidates = [row["manifest_input"] for row in rows]
    formula_families = {row["formula_sha256"] for row in candidates}
    equivalence_families = {row["equivalence_family_id"] for row in candidates}
    current = {
        "response_scored_campaigns": 0,
        "response_planned_campaigns": 1,
        "adaptive_generations": 1,
        "concepts": 400,
        "registered_candidate_rows": 400,
        "equivalence_families": len(equivalence_families),
        "formula_variants": len(formula_families),
        "parameter_cells": 1,
        "hyperparameter_cells": 1,
        "nuisance_scenarios": 1,
        "transformations": 1,
        "object_subsets": 2,
        "observables": 2,
        "metrics": 1,
        "repairs": 1,
        "stopping_decisions": 1,
        "residual_inspired_branches": 1,
        "selection_stages": 2,
        "response_planned_formula_variants": 0,
        "response_planned_domain_executions": 0,
        "response_scored_formula_variants": 0,
        "response_scored_domain_executions": 0,
    }
    before = {dimension: 0 for dimension in registry.MULTIPLICITY_DIMENSIONS}
    blind = config["target_blind_contract"]
    registry_receipt = registry.build_receipt(ROOT)
    manifest: dict[str, object] = {
        "schema_version": "invariant-open-gravity-campaign-manifest-1.0",
        "manifest_id": "TWELL-FINAL-THEORY-INPUT-VALIDATION@1.0.0",
        "campaign_id": "TWELL-FINAL-THEORY-INPUT-VALIDATION",
        "semantic_version": "1.0.0",
        "manifest_state": "FROZEN_UNRUN",
        "frozen_at_utc": "2026-08-30T00:00:00Z",
        "frozen_before_response_access": True,
        "response_scored_campaign": True,
        "registry_binding": {
            "registry_id": registry.REGISTRY_ID,
            "semantic_version": registry.REGISTRY_VERSION,
            "foundation_receipt_sha256": registry_receipt["content_sha256"],
            "mechanism_card_set_sha256": registry.mechanism_card_set_sha256(cards),
            "equivalence_ledger_sha256": registry.campaign_equivalence_ledger_sha256(candidates),
            "trusted_session_contract_sha256": registry.trusted_session_contract_sha256(config),
            "twell_400_ids_sha256": registry.TWELL_IDS_SHA256,
        },
        "candidate_versions": candidates,
        "data_roles_and_splits": {
            "source_partitions": [_partition("SOURCE-ONLY", "SOURCE_ONLY", ["O0001"])],
            "response_partitions": [
                _partition("PILOT", "DEVELOPMENT_PILOT", ["O0001"]),
                _partition("FULL", "DEVELOPMENT_FULL", ["O0002"]),
                _partition("CONFIRMATION", "CONFIRMATION_SEALED", ["O0003"]),
                _partition("INDEPENDENT", "INDEPENDENT_SEALED", ["O0004"]),
            ],
            "pilot_full_relation": "DISJOINT",
            "confirmation_forbidden_in_campaign": True,
        },
        "parameter_cells": [
            {"cell_id": "P1", "exact_value_or_rule": "theory-only", "frozen": True}
        ],
        "hyperparameter_cells": [{"cell_id": "H1", "exact_value_or_rule": "none", "frozen": True}],
        "nuisance_cases": [{"cell_id": "N1", "exact_value_or_rule": "none", "frozen": True}],
        "adaptive_generation_ids": ["NONE-THEORY-ONLY"],
        "transformations": [_named("TRANSFORM-1")],
        "object_subsets": [_named("SUBSET-1"), _named("SUBSET-2")],
        "observables": [_named("OBS-1"), _named("OBS-2")],
        "metrics": [_named("METRIC-1")],
        "comparators": [_named("COMPARATOR-1")],
        "repairs": [_named("REPAIR-1")],
        "stopping_decisions": [_named("STOP-1")],
        "residual_inspired_branch_ids": ["NONE-THEORY-ONLY"],
        "selection_stages": [_named("STAGE-1"), _named("STAGE-2")],
        "correction_method": {
            "method_id": "GLOBAL-SEQUENTIAL-1",
            "exact_rule": "no response-planned formula in this validation fixture",
            "selection_adjusted_reporting": True,
            "global_sequential_evidence_budget_units": 1,
        },
        "global_multiplicity_ledger": {
            "ledger_id": registry.MULTIPLICITY_LEDGER_ID,
            "campaign_sequence": 1,
            "previous_manifest_sha256": "GENESIS",
            "never_resets": True,
            "counts_before": before,
            "counts_this_campaign": current,
            "counts_after": copy.deepcopy(current),
        },
        "promotion_thresholds": {
            "shared_constants_required": True,
            "minimum_meaningful_improvement": 0.01,
            "selection_adjusted_evidence_threshold": 0.95,
            "leave_one_object_out_minimum": 0.0,
            "minimum_object_breadth": 1,
            "minimum_domain_breadth": 1,
        },
        "worst_case_and_subgroup_ceilings": [
            {"ceiling_id": "W1", "metric_id": "METRIC-1", "scope": "all", "maximum": 10.0}
        ],
        "budgets": {
            "lane_candidate_limits": {
                "CORE": 140,
                "ADJACENT": 10,
                "ORTHOGONAL": 160,
                "RIVALS_CONTROLS": 80,
                "WILDCARD": 10,
            },
            "revision_limit": 0,
            "compute_cost_ceiling": "local-zero-external-cost",
            "network_cost_ceiling": 0,
            "model_cost_ceiling": 0,
            "paid_cost_ceiling": 0,
        },
        "confirmation": {
            "K": 1,
            "evidence_budget_units_total": 1,
            "slots": [
                {
                    "slot_id": "K1",
                    "evidence_budget_units": 1,
                    "sealed": True,
                    "opened": False,
                    "candidate_id": None,
                }
            ],
            "opened_slots": 0,
            "independent_reduction_required": True,
        },
        "target_blind_contract": {
            "anonymous_formula_identifiers": blind["anonymous_formula_identifiers"],
            "anonymous_object_identifiers": blind["anonymous_object_identifiers"],
            "target_class_switches_allowed": blind["target_class_switches_allowed"],
            "object_specific_gravity_tuning_allowed": blind[
                "object_specific_gravity_tuning_allowed"
            ],
            "score_feedback_to_formula_authors_before_batch_close": blind[
                "score_feedback_to_formula_authors_before_batch_close"
            ],
            "theory_author_may_adjudicate_own_candidate": blind[
                "theory_author_may_adjudicate_own_candidate"
            ],
            "legitimate_source_metadata_visible": blind["legitimate_source_metadata_visible"],
            "forbidden_generation_inputs": blind["forbidden_generation_inputs"],
            "sealed_formula_mapping_sha256": content_sha256("formula-map"),
            "sealed_object_mapping_sha256": content_sha256("object-map"),
        },
        "session_terminal_contract": {
            "session_id": registry.TRUSTED_SESSION_ID,
            "trusted_session_contract_sha256": registry.trusted_session_contract_sha256(config),
            "terminal_ledger_contract_sha256": registry.terminal_ledger_contract_sha256(config),
            "campaign_execution_authority": "WITHHELD_PENDING_PERSISTED_TERMINAL_LEDGER",
            "response_scored_campaign_limit": 1,
            "response_scored_campaign_ordinal": 1,
            "automatic_second_campaign_allowed": False,
            "on_adjudication": "SESSION_TERMINAL",
            "post_freeze_new_or_repaired_idea_destination": registry.IDEA_RESERVOIR_ID,
            "zero_survivors_allowed": True,
        },
        "zero_access_at_freeze": {field: 0 for field in registry.ZERO_ACCESS_FIELDS},
        "manifest_content_sha256": content_sha256("placeholder"),
    }
    manifest["manifest_content_sha256"] = registry.campaign_manifest_sha256(manifest)
    return manifest


def test_complete_packet_is_consumable_by_registry_campaign_manifest_validator(
    packet: tuple[list[dict], bytes, dict],
) -> None:
    rows, _payload, _receipt = packet
    manifest = _manifest_for_registry_validator(rows)
    schema = registry.load_schemas(ROOT)["campaign_manifest"]
    config = registry.load_config(ROOT)
    registry.validate_campaign_manifest(
        manifest,
        schema,
        config,
        mechanism_cards=[row["card"] for row in rows],
        root=ROOT,
    )


def test_check_is_hard_bound_to_canonical_paths_before_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trap_cards = tmp_path / "attacker-cards.jsonl"
    trap_receipt = tmp_path / "attacker-receipt.json"
    trap_cards.write_bytes(b"{}\n")
    trap_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError):
        check_packet(ROOT, trap_cards, trap_receipt)  # type: ignore[call-arg]

    monkeypatch.setattr(twell, "CARDS_PATH", Path("arbitrary/cards.jsonl"))

    def forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("path mismatch must fail before any file read")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(FinalTwellCompilerError, match="cards path constant changed"):
        check_packet(ROOT)


@pytest.mark.parametrize("command", ["build", "check"])
@pytest.mark.parametrize("flag", ["--cards", "--receipt"])
def test_cli_rejects_arbitrary_packet_paths_without_reading_them(
    command: str,
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trap = tmp_path / "attacker-selected-packet.json"
    trap.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        assert path.resolve() != trap.resolve()
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        assert path.resolve() != trap.resolve()
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(SystemExit) as error:
        main([command, flag, str(trap)])
    assert error.value.code == 2


def test_canonical_packet_paths_are_exact() -> None:
    assert _canonical_packet_paths(ROOT) == (
        ROOT / Path(EXPECTED_CARDS_PATH),
        ROOT / Path(EXPECTED_RECEIPT_PATH),
    )
    assert CARDS_PATH.as_posix() == EXPECTED_CARDS_PATH
    assert RECEIPT_PATH.as_posix() == EXPECTED_RECEIPT_PATH


def test_stored_final_packet_matches_exact_rebuild() -> None:
    assert check_packet(ROOT) == build_packet(ROOT)[2]
