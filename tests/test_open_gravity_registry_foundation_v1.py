from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_registry_foundation_v1 as registry

ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _hashed(label: str) -> str:
    return registry.content_sha256({"label": label})


def _card(
    stable_id: str = "TEST-CARD",
    *,
    identity_class: str = "NEW_CONCEPT",
    kind: str = "FIELD_EQUATIONS",
    executable: bool = True,
    scientific_status: str = "H_HYPOTHESIS",
) -> dict[str, object]:
    card: dict[str, object] = {
        "schema_version": "invariant-open-gravity-mechanism-card-1.0",
        "card_id": f"{stable_id}@1.0.0",
        "stable_concept_id": stable_id,
        "semantic_version": "1.0.0",
        "identity_class": identity_class,
        "parents": [],
        "author_agent": "unit-test-discovery-agent",
        "provenance": {
            "created_at_utc": "2026-08-30T00:00:00Z",
            "origin_timing": "PRE_RESPONSE",
            "origin_artifacts": ["synthetic:no-data-seed"],
            "residual_access_lineage": [],
        },
        "lay_mechanism": "A causal source-only field keeps a finite record of prior baryons.",
        "novelty_claim": "The exact bounded state equation is the claimed differentiator.",
        "ontology": ["QG12"],
        "scientific_status": scientific_status,
        "operational_variables": [
            {
                "symbol": "q",
                "operational_definition": "dimensionless source-derived memory state",
                "dimension": "1",
                "observable_or_latent": "LATENT_FIELD",
            }
        ],
        "source": f"declared baryonic density only for {stable_id}",
        "coupling": "universal scalar coupling",
        "action_or_equations": {
            "kind": kind,
            "exact_expressions": [
                "S_to_be_completed" if kind == "ACTION_PLACEHOLDER" else "tau*dq/dt+q=u_source"
            ],
            "executable": executable,
        },
        "initial_conditions": ["q=0 at the frozen initial slice"],
        "boundaries": ["q=0 at the shared outer boundary"],
        "degrees_of_freedom": {
            "fields": ["q"],
            "spin_helicity": "spin-0 classical effective field",
            "mass": "not assigned",
            "statistics": "not applicable",
            "state": "classical deterministic",
            "quantum_applicability": "NOT_APPLICABLE",
        },
        "propagation": {
            "speed": "bounded by c",
            "dispersion": "none in the frozen seed",
            "polarization": "scalar",
            "attenuation": "finite relaxation",
            "range": "universal L",
            "static_limit": "q=u_source",
        },
        "state_rule": {"mode": "TEMPORAL_MEMORY", "exact_rule": "tau*dq/dt+q=u_source"},
        "closures": {
            "matter": "source-derived scalar response",
            "photon": "L0_NO_LIGHT_CLAIM",
            "gravitational_wave": "GR recovery required; no new claim",
            "quantum_laboratory": "no effect claimed",
            "capture": "C0_ISOLATED_CONSERVATIVE",
            "cosmology": "no effect claimed",
        },
        "ledgers": {
            "energy": "conservative seed; action still required before promotion",
            "momentum": "total momentum must be derived",
            "entropy": "no entropy production claimed",
            "information": "state history is finite and causal",
        },
        "structure": {
            "symmetries": ["spatial rotations in the synthetic control"],
            "covariance_or_frame": "declared preferred source frame",
            "equivalence_behavior": "universal matter coupling required",
            "causal_structure": "retarded source state with speed at most c",
        },
        "dimensions": ["[tau*dq/dt]=[q]=1"],
        "parameter_cells": [],
        "priors": [],
        "screens": [],
        "limiting_cases": ["tau->0 recovers the local source state"],
        "source_only_data_contract": {
            "allowed_inputs": ["baryonic_density"],
            "forbidden_response_inputs": ["motion", "pressure", "temperature", "lensing"],
            "construction_before_response": True,
            "missing_data_action": "SOURCE_BLOCKED",
        },
        "synthetic_falsifier": "Reject if the retarded injected source is not recovered.",
        "real_data_discriminator": "Frozen held-out local-versus-history difference.",
        "prior_art": [{"citation": "synthetic:test", "relationship": "control only"}],
        "equivalence_fingerprint": {
            "canonical_symbolic_sha256": _hashed(f"symbolic:{stable_id}"),
            "analytic_limits_sha256": _hashed(f"limits:{stable_id}"),
            "synthetic_fingerprint_sha256": _hashed(f"synthetic:{stable_id}"),
            "observable_fingerprint_sha256": _hashed(f"observable:{stable_id}"),
        },
        "version_change": {
            "kind": "INITIAL_REGISTRATION",
            "previous_card_id": None,
            "previous_card_sha256": None,
            "changed_facets": [],
            "prior_result_retained": True,
            "replay_all_affected": False,
        },
        "hashes": {
            "code_sha256": _hashed(f"code:{stable_id}"),
            "data_sha256": _hashed(f"data:{stable_id}"),
            "environment_sha256": _hashed(f"environment:{stable_id}"),
            "configuration_sha256": _hashed(f"configuration:{stable_id}"),
            "formula_sha256": SHA_A,
        },
    }
    card["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(card)
    return card


def _revision(previous: dict[str, object], version: str, kind: str, facets: list[str]) -> dict:
    current = copy.deepcopy(previous)
    previous_sha = registry.content_sha256(previous)
    current["semantic_version"] = version
    current["card_id"] = f"{current['stable_concept_id']}@{version}"
    current["parents"] = [
        {
            "card_id": previous["card_id"],
            "card_sha256": previous_sha,
            "relation": "REPAIRS" if kind == "PATCH" else "SUPERSEDES",
        }
    ]
    current["version_change"] = {
        "kind": kind,
        "previous_card_id": previous["card_id"],
        "previous_card_sha256": previous_sha,
        "changed_facets": facets,
        "prior_result_retained": True,
        "replay_all_affected": kind == "PATCH",
    }
    return current


def _domain_rows(disposition: str) -> dict[str, dict[str, object]]:
    planned = disposition == "SEALED_UNOPENED_FOR_SCORING"
    return {
        domain: {
            "eligible": planned,
            "execution_disposition": disposition,
            "scored": False,
            "source_contract_sha256": _hashed(f"source-contract:{domain}"),
        }
        for domain in registry.DOMAINS
    }


def _candidate(card: dict[str, object], lane: str, index: int, status: str) -> dict[str, object]:
    disposition = {
        "REGISTERED_THEORY_ONLY": "THEORY_ONLY",
        "SOURCE_BLOCKED": "SOURCE_BLOCKED",
        "READY_FOR_RESPONSE_SCORING": "SEALED_UNOPENED_FOR_SCORING",
        "QUARANTINED_REVISION_REQUIRED": "QUARANTINED",
        "KNOWN_REWRITE_NONINDEPENDENT": "KNOWN_REWRITE_NONINDEPENDENT",
    }[status]
    domains = _domain_rows(disposition)
    if status == "READY_FOR_RESPONSE_SCORING":
        for domain in registry.DOMAINS[1:]:
            domains[domain] = {
                "eligible": False,
                "execution_disposition": "NOT_APPLICABLE",
                "scored": False,
                "source_contract_sha256": _hashed(f"source-contract:{domain}"),
            }
    return {
        "candidate_id": card["stable_concept_id"],
        "card_id": card["card_id"],
        "semantic_version": card["semantic_version"],
        "anonymous_formula_id": f"F{index:04d}",
        "lane": lane,
        "candidate_status": status,
        "scientific_status": card["scientific_status"],
        "identity_class": card["identity_class"],
        "mechanism_kind": card["action_or_equations"]["kind"],
        "mechanism_executable": card["action_or_equations"]["executable"],
        "card_sha256": registry.content_sha256(card),
        "formula_sha256": card["hashes"]["formula_sha256"],
        "configuration_sha256": card["hashes"]["configuration_sha256"],
        "equivalence_family_id": f"EQ-{index}",
        "equivalence_fingerprint_sha256": registry.equivalence_fingerprint_sha256(card),
        "domain_execution": domains,
    }


def _cards() -> list[dict[str, object]]:
    return [
        _card("CANDIDATE-1"),
        _card("CANDIDATE-2"),
        _card("CANDIDATE-3", executable=False),
        _card("CANDIDATE-4", kind="ACTION_PLACEHOLDER", executable=False),
        _card("CANDIDATE-5", identity_class="KNOWN_REWRITE"),
    ]


def _named(item_id: str) -> dict[str, str]:
    return {
        "item_id": item_id,
        "definition": f"frozen definition for {item_id}",
        "implementation_sha256": _hashed(f"implementation:{item_id}"),
    }


def _partition(partition_id: str, role: str, object_ids: list[str]) -> dict[str, object]:
    return {
        "partition_id": partition_id,
        "role": role,
        "anonymous_object_ids": object_ids,
        "object_ledger_sha256": registry.partition_object_ledger_sha256(object_ids),
        "data_contract_sha256": _hashed(f"data-contract:{partition_id}"),
    }


def _campaign(config: dict[str, object], cards: list[dict[str, object]]) -> dict[str, object]:
    statuses = [
        "READY_FOR_RESPONSE_SCORING",
        "REGISTERED_THEORY_ONLY",
        "SOURCE_BLOCKED",
        "QUARANTINED_REVISION_REQUIRED",
        "KNOWN_REWRITE_NONINDEPENDENT",
    ]
    candidates = [
        _candidate(card, lane, index, status)
        for index, (card, lane, status) in enumerate(
            zip(cards, registry.LANES, statuses, strict=True), start=1
        )
    ]
    current = {
        "response_scored_campaigns": 0,
        "response_planned_campaigns": 1,
        "adaptive_generations": 1,
        "concepts": 5,
        "registered_candidate_rows": 5,
        "equivalence_families": 5,
        "formula_variants": 5,
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
        "response_planned_formula_variants": 1,
        "response_planned_domain_executions": 1,
        "response_scored_formula_variants": 0,
        "response_scored_domain_executions": 0,
    }
    before = {dimension: 0 for dimension in registry.MULTIPLICITY_DIMENSIONS}
    blind = config["target_blind_contract"]
    receipt = registry.build_receipt(ROOT)
    manifest: dict[str, object] = {
        "schema_version": "invariant-open-gravity-campaign-manifest-1.0",
        "manifest_id": "OPEN-GRAVITY-CAMPAIGN-TEST@1.0.0",
        "campaign_id": "OPEN-GRAVITY-CAMPAIGN-TEST",
        "semantic_version": "1.0.0",
        "manifest_state": "FROZEN_UNRUN",
        "frozen_at_utc": "2026-08-30T00:00:00Z",
        "frozen_before_response_access": True,
        "response_scored_campaign": True,
        "registry_binding": {
            "registry_id": registry.REGISTRY_ID,
            "semantic_version": registry.REGISTRY_VERSION,
            "foundation_receipt_sha256": receipt["content_sha256"],
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
        "parameter_cells": [{"cell_id": "P1", "exact_value_or_rule": "fixed", "frozen": True}],
        "hyperparameter_cells": [{"cell_id": "H1", "exact_value_or_rule": "fixed", "frozen": True}],
        "nuisance_cases": [{"cell_id": "N1", "exact_value_or_rule": "nominal", "frozen": True}],
        "adaptive_generation_ids": ["ADAPTIVE-1"],
        "transformations": [_named("TRANSFORM-1")],
        "object_subsets": [_named("SUBSET-1"), _named("SUBSET-2")],
        "observables": [_named("OBS-1"), _named("OBS-2")],
        "metrics": [_named("METRIC-1")],
        "comparators": [_named("COMPARATOR-1")],
        "repairs": [_named("REPAIR-1")],
        "stopping_decisions": [_named("STOP-1")],
        "residual_inspired_branch_ids": ["RESIDUAL-BRANCH-1"],
        "selection_stages": [_named("STAGE-1"), _named("STAGE-2")],
        "correction_method": {
            "method_id": "GLOBAL-SEQUENTIAL-1",
            "exact_rule": "spend one preallocated unit per declared selection leaf",
            "selection_adjusted_reporting": True,
            "global_sequential_evidence_budget_units": 100,
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
            "lane_candidate_limits": {lane: 1 for lane in registry.LANES},
            "revision_limit": 0,
            "compute_cost_ceiling": "local-zero-external-cost",
            "network_cost_ceiling": 0,
            "model_cost_ceiling": 0,
            "paid_cost_ceiling": 0,
        },
        "confirmation": {
            "K": 1,
            "evidence_budget_units_total": 100,
            "slots": [
                {
                    "slot_id": "K1",
                    "evidence_budget_units": 100,
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
            "sealed_formula_mapping_sha256": SHA_D,
            "sealed_object_mapping_sha256": SHA_E,
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
        "manifest_content_sha256": SHA_A,
    }
    manifest["manifest_content_sha256"] = registry.campaign_manifest_sha256(manifest)
    return manifest


def _rehash_manifest(manifest: dict[str, object]) -> None:
    manifest["manifest_content_sha256"] = registry.campaign_manifest_sha256(manifest)


def _entry(idea_id: str, version: str) -> dict[str, object]:
    return {
        "idea_id": idea_id,
        "entry_version": version,
        "created_at_utc": "2026-08-30T00:00:00Z",
        "created_by": "unit-test-discovery-agent",
        "lineage": ["OPEN-GRAVITY-CAMPAIGN-TEST"],
        "origin_timing": "ADAPTIVE_DEVELOPMENT",
        "residual_source_campaigns": ["OPEN-GRAVITY-CAMPAIGN-TEST"],
        "summary": "An adaptive idea retained for a later campaign.",
        "hypothesis": "A new boundary state may explain an exposed residual pattern.",
        "intended_ontology": ["QG12"],
        "mechanism_card_id": None,
        "current_state": "INCOMPLETE_QUARANTINE",
        "eligible_campaign": "FUTURE_ONLY",
        "current_campaign_scoring_allowed": False,
    }


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return registry.load_config(ROOT)


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict[str, object]]:
    return registry.load_schemas(ROOT)


@pytest.fixture()
def bundle(config: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    cards = _cards()
    return cards, _campaign(config, cards)


def test_foundation_content_and_every_schema_semantic_byte_are_hard_pinned(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    assert registry.content_sha256(config) == registry.EXPECTED_CONFIG_CONTENT_SHA256
    assert {
        name: registry.content_sha256(schema) for name, schema in schemas.items()
    } == registry.EXPECTED_SCHEMA_CONTENT_SHA256
    assert len(registry.twell_concept_ids()) == len(set(registry.twell_concept_ids())) == 400
    assert registry.content_sha256(registry.twell_concept_ids()) == registry.TWELL_IDS_SHA256
    assert [row["id"] for row in config["grammar"]["ontology_nodes"]] == [
        f"QG{index:02d}" for index in range(1, 14)
    ]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("purpose",), "CONFIRMED DISCOVERY"),
        (("version_rules", "patch_change_facets"), ["anything"]),
        (("grammar", "ontology_nodes", 0, "name"), "bananas"),
        (("grammar", "light_gravity_axes", 0, "light"), "anything"),
    ],
)
def test_coordinated_config_semantic_mutations_fail_closed(
    config: dict[str, object], path: tuple[str | int, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor: object = mutated
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(registry.OpenGravityRegistryError, match="immutable foundation"):
        registry.validate_foundation_config(mutated)


@pytest.mark.parametrize("schema_name", ["mechanism_card", "idea_reservoir", "campaign_manifest"])
def test_nested_schema_weakening_under_the_same_id_fails_closed(
    schemas: dict[str, dict[str, object]], schema_name: str
) -> None:
    mutated = copy.deepcopy(schemas[schema_name])
    mutated["title"] = "weakened schema with unchanged ID"
    with pytest.raises(registry.OpenGravityRegistryError, match="immutable schema semantics"):
        registry._validate_schema_document(
            mutated,
            {
                "mechanism_card": "urn:invariant:open-gravity:mechanism-card:1.0",
                "idea_reservoir": "urn:invariant:open-gravity:idea-reservoir:1.0",
                "campaign_manifest": "urn:invariant:open-gravity:campaign-manifest:1.0",
            }[schema_name],
            registry.EXPECTED_SCHEMA_CONTENT_SHA256[schema_name],
        )


def test_card_admission_distinguishes_ready_blocked_placeholder_rewrite_and_established(
    schemas: dict[str, dict[str, object]],
) -> None:
    schema = schemas["mechanism_card"]
    assert registry.mechanism_card_admission(_card(), schema)["status"] == "READY_FOR_THEORY_GATES"
    assert (
        registry.mechanism_card_admission(_card(executable=False), schema)["status"]
        == "SOURCE_BLOCKED"
    )
    assert (
        registry.mechanism_card_admission(
            _card(kind="ACTION_PLACEHOLDER", executable=False), schema
        )["status"]
        == "QUARANTINED_REVISION_REQUIRED"
    )
    assert (
        registry.mechanism_card_admission(_card(identity_class="KNOWN_REWRITE"), schema)["status"]
        == "KNOWN_REWRITE_NONINDEPENDENT"
    )
    established = registry.mechanism_card_admission(
        _card(scientific_status="E_ESTABLISHED"), schema
    )
    assert established["status"] == "INCOMPLETE_QUARANTINE"
    assert any("not self-attestable" in error for error in established["errors"])


def test_incomplete_and_formula_rebound_cards_fail_closed(
    schemas: dict[str, dict[str, object]],
) -> None:
    schema = schemas["mechanism_card"]
    incomplete = _card()
    incomplete.pop("ledgers")
    assert registry.mechanism_card_admission(incomplete, schema)["eligible"] is False
    rebound = _card()
    rebound["hashes"]["formula_sha256"] = SHA_A
    admission = registry.mechanism_card_admission(rebound, schema)
    assert admission["eligible"] is False
    assert any("canonical formula" in error for error in admission["errors"])


def test_semver_exact_major_minor_patch_and_metadata_transitions(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    previous = _card()
    schema = schemas["mechanism_card"]
    minor = _revision(previous, "1.1.0", "MINOR", ["equations"])
    minor["action_or_equations"]["exact_expressions"] = ["tau*dq/dt+q=u_source+u_boundary"]
    minor["equivalence_fingerprint"]["canonical_symbolic_sha256"] = SHA_B
    minor["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(minor)
    assert registry.validate_version_transition(previous, minor, config, schema) == "MINOR"

    patch = _revision(previous, "1.0.1", "PATCH", ["numerical_bug_fix"])
    patch["hashes"]["code_sha256"] = SHA_B
    assert registry.validate_version_transition(previous, patch, config, schema) == "PATCH"

    major = _revision(previous, "2.0.0", "MAJOR", ["observable_closure"])
    major["closures"]["photon"] = "L1_SINGLE_METRIC"
    major["equivalence_fingerprint"]["canonical_symbolic_sha256"] = SHA_C
    major["equivalence_fingerprint"]["observable_fingerprint_sha256"] = SHA_A
    major["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(major)
    assert registry.validate_version_transition(previous, major, config, schema) == "MAJOR"

    metadata = _revision(previous, "1.0.1", "METADATA_ONLY", ["nonsemantic_provenance_note"])
    metadata["prior_art"][0]["relationship"] = "corrected nonsemantic citation note"
    assert (
        registry.validate_version_transition(previous, metadata, config, schema) == "METADATA_ONLY"
    )


def test_version_transition_revalidates_exact_config_and_schema_semantics(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    previous = _card()
    current = _revision(previous, "1.1.0", "MINOR", ["physical_meaning"])
    current["lay_mechanism"] = "materially different physical mechanism"
    mutated_config = copy.deepcopy(config)
    mutated_config["version_rules"]["major_change_facets"].remove("physical_meaning")
    mutated_config["version_rules"]["minor_change_facets"].append("physical_meaning")
    with pytest.raises(registry.OpenGravityRegistryError, match="immutable foundation"):
        registry.validate_version_transition(
            previous,
            current,
            mutated_config,
            schemas["mechanism_card"],
        )

    mutated_schema = copy.deepcopy(schemas["mechanism_card"])
    mutated_schema["title"] = "coherently weakened under an unchanged schema ID"
    with pytest.raises(registry.OpenGravityRegistryError, match="immutable schema semantics"):
        registry.validate_version_transition(previous, current, config, mutated_schema)


def test_formula_bearing_scientific_revision_rebinds_canonical_symbolic_fingerprint(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    previous = _card()
    current = _revision(previous, "1.1.0", "MINOR", ["equations"])
    current["action_or_equations"]["exact_expressions"] = ["changed equation"]
    current["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(current)
    with pytest.raises(registry.OpenGravityRegistryError, match="canonical-symbolic"):
        registry.validate_version_transition(previous, current, config, schemas["mechanism_card"])


@pytest.mark.parametrize(
    ("kind", "hash_name"),
    [
        ("PATCH", "formula_sha256"),
        ("PATCH", "configuration_sha256"),
        ("PATCH", "data_sha256"),
        ("PATCH", "environment_sha256"),
        ("METADATA_ONLY", "code_sha256"),
        ("METADATA_ONLY", "data_sha256"),
        ("METADATA_ONLY", "environment_sha256"),
    ],
)
def test_patch_and_metadata_cannot_rebind_protected_hashes(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    kind: str,
    hash_name: str,
) -> None:
    previous = _card()
    facet = "numerical_bug_fix" if kind == "PATCH" else "nonsemantic_provenance_note"
    current = _revision(previous, "1.0.1", kind, [facet])
    if kind == "PATCH":
        current["hashes"]["code_sha256"] = SHA_B
    else:
        current["prior_art"][0]["relationship"] = "metadata correction"
    current["hashes"][hash_name] = SHA_E
    with pytest.raises(registry.OpenGravityRegistryError):
        registry.validate_version_transition(previous, current, config, schemas["mechanism_card"])


def test_semver_skips_and_declared_noop_facets_fail_closed(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    previous = _card()
    skipped = _revision(previous, "1.3.0", "MINOR", ["equations"])
    skipped["action_or_equations"]["exact_expressions"] = ["changed"]
    skipped["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(skipped)
    with pytest.raises(registry.OpenGravityRegistryError, match="MINOR"):
        registry.validate_version_transition(previous, skipped, config, schemas["mechanism_card"])
    noop = _revision(previous, "1.1.0", "MINOR", ["equations"])
    with pytest.raises(registry.OpenGravityRegistryError, match="no changed field"):
        registry.validate_version_transition(previous, noop, config, schemas["mechanism_card"])


def test_reservoir_global_chain_and_same_idea_semver_are_append_only(
    schemas: dict[str, dict[str, object]],
) -> None:
    reservoir = registry._load_json(ROOT / registry.RESERVOIR_PATH)
    schema = schemas["idea_reservoir"]
    first = registry.append_reservoir_entry(
        reservoir, _entry("IDEA-ADAPTIVE-MEMORY", "1.0.0"), schema
    )
    second = registry.append_reservoir_entry(first, _entry("IDEA-ADAPTIVE-MEMORY", "1.1.0"), schema)
    registry.validate_idea_reservoir(second, schema, previous=first)
    assert (
        second["entries"][1]["previous_same_idea_entry_sha256"]
        == first["entries"][0]["entry_sha256"]
    )
    with pytest.raises(registry.OpenGravityRegistryError, match="not monotonic"):
        registry.append_reservoir_entry(second, _entry("IDEA-ADAPTIVE-MEMORY", "1.0.1"), schema)
    mutated = copy.deepcopy(second)
    mutated["entries"][0]["summary"] = "silently changed"
    with pytest.raises(registry.OpenGravityRegistryError, match="entry hash"):
        registry.validate_idea_reservoir(mutated, schema)


def test_campaign_binds_live_receipt_cards_equivalence_and_domain_dispositions(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    registry.validate_campaign_manifest(
        manifest,
        schemas["campaign_manifest"],
        config,
        mechanism_cards=cards,
        root=ROOT,
    )
    current = manifest["global_multiplicity_ledger"]["counts_this_campaign"]
    assert current["response_planned_campaigns"] == 1
    assert current["response_scored_campaigns"] == 0
    assert current["registered_candidate_rows"] == 5
    assert current["formula_variants"] == 5
    assert current["response_planned_formula_variants"] == 1
    assert current["response_planned_domain_executions"] == 1
    assert current["response_scored_formula_variants"] == 0
    assert current["response_scored_domain_executions"] == 0


def test_coherent_duplicate_formula_rows_count_once_but_rows_remain_registered(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    cards = _cards()
    for field_name in registry.FORMULA_PAYLOAD_FIELDS:
        cards[1][field_name] = copy.deepcopy(cards[0][field_name])
    cards[1]["equivalence_fingerprint"] = copy.deepcopy(cards[0]["equivalence_fingerprint"])
    cards[1]["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(cards[1])
    manifest = _campaign(config, cards)
    manifest["candidate_versions"][1]["equivalence_family_id"] = "EQ-1"
    manifest["registry_binding"]["equivalence_ledger_sha256"] = (
        registry.campaign_equivalence_ledger_sha256(manifest["candidate_versions"])
    )
    current = manifest["global_multiplicity_ledger"]["counts_this_campaign"]
    after = manifest["global_multiplicity_ledger"]["counts_after"]
    current["formula_variants"] = after["formula_variants"] = 4
    current["equivalence_families"] = after["equivalence_families"] = 4
    _rehash_manifest(manifest)
    registry.validate_campaign_manifest(
        manifest,
        schemas["campaign_manifest"],
        config,
        mechanism_cards=cards,
        root=ROOT,
    )
    assert current["registered_candidate_rows"] == 5
    assert current["formula_variants"] == 4
    assert current["equivalence_families"] == 4


def test_arbitrary_foundation_hash_and_live_card_tamper_fail_closed(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    arbitrary = copy.deepcopy(manifest)
    arbitrary["registry_binding"]["foundation_receipt_sha256"] = SHA_A
    _rehash_manifest(arbitrary)
    with pytest.raises(registry.OpenGravityRegistryError, match="live registry"):
        registry.validate_campaign_manifest(
            arbitrary,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )
    tampered_cards = copy.deepcopy(cards)
    tampered_cards[0]["lay_mechanism"] = "silently mutated live card"
    with pytest.raises(registry.OpenGravityRegistryError):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=tampered_cards,
            root=ROOT,
        )


def test_identical_executable_formulas_cannot_split_equivalence_families(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    cards = _cards()
    for field_name in registry.FORMULA_PAYLOAD_FIELDS:
        cards[1][field_name] = copy.deepcopy(cards[0][field_name])
    cards[1]["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(cards[1])
    manifest = _campaign(config, cards)
    with pytest.raises(registry.OpenGravityRegistryError, match="split across equivalence"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


@pytest.mark.parametrize(
    ("candidate_index", "bad_status"),
    [
        (2, "READY_FOR_RESPONSE_SCORING"),
        (3, "READY_FOR_RESPONSE_SCORING"),
        (4, "READY_FOR_RESPONSE_SCORING"),
    ],
)
def test_nonexecutable_placeholder_and_rewrite_candidates_never_enter_ready_slots(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
    candidate_index: int,
    bad_status: str,
) -> None:
    cards, manifest = bundle
    candidate = manifest["candidate_versions"][candidate_index]
    candidate["candidate_status"] = bad_status
    candidate["domain_execution"] = _domain_rows("SEALED_UNOPENED_FOR_SCORING")
    manifest["registry_binding"]["equivalence_ledger_sha256"] = (
        registry.campaign_equivalence_ledger_sha256(manifest["candidate_versions"])
    )
    _rehash_manifest(manifest)
    with pytest.raises(
        registry.OpenGravityRegistryError, match="status overclaims|planned scoring slot"
    ):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_candidate_domain_eligibility_disposition_and_scored_must_agree(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    manifest["candidate_versions"][0]["domain_execution"]["GALAXIES"]["eligible"] = False
    _rehash_manifest(manifest)
    with pytest.raises(registry.OpenGravityRegistryError, match="eligibility flag"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


@pytest.mark.parametrize("mutation", ["scored_true", "scored_disposition"])
def test_frozen_unrun_manifest_cannot_claim_postrun_scoring(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
    mutation: str,
) -> None:
    cards, manifest = bundle
    domain = manifest["candidate_versions"][0]["domain_execution"]["GALAXIES"]
    if mutation == "scored_true":
        domain["scored"] = True
    else:
        domain["execution_disposition"] = "SCORED"
    _rehash_manifest(manifest)
    with pytest.raises(registry.OpenGravityRegistryError, match="schema errors"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


@pytest.mark.parametrize(
    "actual_dimension",
    [
        "response_scored_campaigns",
        "response_scored_formula_variants",
        "response_scored_domain_executions",
    ],
)
def test_frozen_manifest_separates_precharged_plans_from_actual_scored_counters(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
    actual_dimension: str,
) -> None:
    cards, manifest = bundle
    ledger = manifest["global_multiplicity_ledger"]
    ledger["counts_this_campaign"][actual_dimension] = 1
    ledger["counts_after"][actual_dimension] = 1
    _rehash_manifest(manifest)
    with pytest.raises(registry.OpenGravityRegistryError, match="multiplicity counts"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


@pytest.mark.parametrize(
    "missing_key",
    ["candidate_status", "scientific_status", "domain_execution"],
)
def test_candidate_lifecycle_and_domain_contract_fields_are_mandatory(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
    missing_key: str,
) -> None:
    cards, manifest = bundle
    del manifest["candidate_versions"][0][missing_key]
    _rehash_manifest(manifest)
    with pytest.raises(registry.OpenGravityRegistryError, match="schema errors"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_unallowlisted_established_candidate_cannot_enter_campaign(
    config: dict[str, object], schemas: dict[str, dict[str, object]]
) -> None:
    cards = _cards()
    cards[0]["scientific_status"] = "E_ESTABLISHED"
    manifest = _campaign(config, cards)
    with pytest.raises(registry.OpenGravityRegistryError, match="incomplete|E_ESTABLISHED"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_pilot_full_membership_is_computed_for_disjoint_and_nested_relations(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    nested = copy.deepcopy(manifest)
    full = next(
        row
        for row in nested["data_roles_and_splits"]["response_partitions"]
        if row["role"] == "DEVELOPMENT_FULL"
    )
    full["anonymous_object_ids"] = ["O0001", "O0002"]
    full["object_ledger_sha256"] = registry.partition_object_ledger_sha256(
        full["anonymous_object_ids"]
    )
    nested["data_roles_and_splits"]["pilot_full_relation"] = (
        "PROPERLY_NESTED_WITH_FULL_PROCEDURE_CORRECTION"
    )
    _rehash_manifest(nested)
    registry.validate_campaign_manifest(
        nested,
        schemas["campaign_manifest"],
        config,
        mechanism_cards=cards,
        root=ROOT,
    )

    overlap = copy.deepcopy(nested)
    overlap["data_roles_and_splits"]["pilot_full_relation"] = "DISJOINT"
    _rehash_manifest(overlap)
    with pytest.raises(registry.OpenGravityRegistryError, match="not disjoint"):
        registry.validate_campaign_manifest(
            overlap,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )

    identical = copy.deepcopy(manifest)
    full = next(
        row
        for row in identical["data_roles_and_splits"]["response_partitions"]
        if row["role"] == "DEVELOPMENT_FULL"
    )
    full["anonymous_object_ids"] = ["O0001"]
    full["object_ledger_sha256"] = registry.partition_object_ledger_sha256(
        full["anonymous_object_ids"]
    )
    _rehash_manifest(identical)
    with pytest.raises(registry.OpenGravityRegistryError, match="not disjoint"):
        registry.validate_campaign_manifest(
            identical,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


@pytest.mark.parametrize("dimension", registry.MULTIPLICITY_DIMENSIONS)
def test_every_global_multiplicity_dimension_is_exactly_derived(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
    dimension: str,
) -> None:
    cards, manifest = bundle
    ledger = manifest["global_multiplicity_ledger"]
    delta = -1 if ledger["counts_this_campaign"][dimension] else 1
    ledger["counts_this_campaign"][dimension] += delta
    ledger["counts_after"][dimension] += delta
    _rehash_manifest(manifest)
    with pytest.raises(registry.OpenGravityRegistryError, match="multiplicity counts"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_nonfinite_thresholds_fail_closed_before_hash_rebinding(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    manifest["promotion_thresholds"]["minimum_meaningful_improvement"] = float("nan")
    with pytest.raises(registry.OpenGravityRegistryError, match="schema errors|noncanonical"):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_trusted_session_id_cannot_be_evaded_by_relabeling(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    relabeled = copy.deepcopy(manifest)
    relabeled["session_terminal_contract"]["session_id"] = "OPEN-GRAVITY-SESSION-NEXT"
    _rehash_manifest(relabeled)
    with pytest.raises(registry.OpenGravityRegistryError, match="schema errors|trusted-session"):
        registry.validate_campaign_manifest(
            relabeled,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )
    with pytest.raises(registry.OpenGravityRegistryError, match="relabeled"):
        registry.validate_session_campaign_set([relabeled], config)

    another = copy.deepcopy(manifest)
    another["campaign_id"] = "OPEN-GRAVITY-CAMPAIGN-SECOND"
    another["manifest_id"] = "OPEN-GRAVITY-CAMPAIGN-SECOND@1.0.0"
    _rehash_manifest(another)
    with pytest.raises(registry.OpenGravityRegistryError, match="more than one"):
        registry.validate_session_campaign_set([manifest, another], config)


def test_separately_validated_genesis_manifests_have_no_execution_authority(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, first = bundle
    second = copy.deepcopy(first)
    second["campaign_id"] = "OPEN-GRAVITY-CAMPAIGN-SECOND"
    second["manifest_id"] = "OPEN-GRAVITY-CAMPAIGN-SECOND@1.0.0"
    _rehash_manifest(second)
    for manifest in (first, second):
        registry.validate_campaign_manifest(
            manifest,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )
        with pytest.raises(registry.OpenGravityRegistryError, match="authority is withheld"):
            registry.assert_campaign_execution_authority(
                manifest,
                terminal_ledger_receipt={"self_attested": True},
            )
    assert config["one_campaign_terminal_rule"]["foundation_campaign_execution_authority"] is False
    assert config["claim_boundary"]["campaign_execution_authority_granted"] is False


def test_target_blindness_confirmation_budget_and_zero_access_stay_exact(
    config: dict[str, object],
    schemas: dict[str, dict[str, object]],
    bundle: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    cards, manifest = bundle
    leaked = copy.deepcopy(manifest)
    leaked["target_blind_contract"]["target_class_switches_allowed"] = True
    _rehash_manifest(leaked)
    with pytest.raises(registry.OpenGravityRegistryError):
        registry.validate_campaign_manifest(
            leaked,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )
    budget = copy.deepcopy(manifest)
    budget["confirmation"]["slots"][0]["evidence_budget_units"] = 99
    _rehash_manifest(budget)
    with pytest.raises(registry.OpenGravityRegistryError, match="exactly preallocated"):
        registry.validate_campaign_manifest(
            budget,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )
    nonzero = copy.deepcopy(manifest)
    nonzero["zero_access_at_freeze"]["scientific_response_rows_opened"] = 1
    _rehash_manifest(nonzero)
    with pytest.raises(registry.OpenGravityRegistryError):
        registry.validate_campaign_manifest(
            nonzero,
            schemas["campaign_manifest"],
            config,
            mechanism_cards=cards,
            root=ROOT,
        )


def test_receipt_builder_reads_only_allowlisted_governance_metadata(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
) -> None:
    seen: list[Path] = []
    original = Path.read_bytes

    def audited_read_bytes(path: Path) -> bytes:
        seen.append(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    receipt = registry.build_receipt(ROOT)
    expected = {
        (ROOT / registry.CONFIG_PATH).resolve(),
        (ROOT / registry.MECHANISM_SCHEMA_PATH).resolve(),
        (ROOT / registry.RESERVOIR_SCHEMA_PATH).resolve(),
        (ROOT / registry.CAMPAIGN_SCHEMA_PATH).resolve(),
        (ROOT / registry.RESERVOIR_PATH).resolve(),
        (ROOT / registry.MODULE_PATH).resolve(),
        (ROOT / registry.TEST_PATH).resolve(),
        registry.OPEN_GOAL_PATH.resolve(),
        registry.BASELINE_GOAL_PATH.resolve(),
    }
    assert set(seen) == expected
    assert receipt["access_accounting"]["governance_metadata_files_opened"] == len(expected)
    assert receipt["campaign_governance"]["trusted_session_id"] == registry.TRUSTED_SESSION_ID
    assert receipt["campaign_governance"]["registered_concepts_are_not_scored_variants"] is True
    assert receipt["campaign_governance"]["campaign_execution_authority_granted"] is False
    assert receipt["campaign_governance"][
        "terminal_ledger_contract_sha256"
    ] == registry.terminal_ledger_contract_sha256(config)
    for field in registry.ZERO_ACCESS_FIELDS:
        assert receipt["access_accounting"][field] == 0


def test_coherently_rehashed_foundation_receipt_forgery_fails_closed() -> None:
    receipt = copy.deepcopy(registry.build_receipt(ROOT))
    receipt["status"] = "CONFIRMED_DISCOVERY"
    receipt.pop("content_sha256")
    receipt["content_sha256"] = registry.content_sha256(receipt)
    with pytest.raises(registry.OpenGravityRegistryError, match="exact rebuild"):
        registry.validate_foundation_receipt(receipt, ROOT)


def test_atomic_receipt_writer_is_deterministic_and_refuses_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert registry._atomic_no_clobber(output, b"same\n") == "CREATED"
    assert registry._atomic_no_clobber(output, b"same\n") == "EXISTING_IDENTICAL"
    with pytest.raises(registry.OpenGravityRegistryError, match="refusing to overwrite"):
        registry._atomic_no_clobber(output, b"different\n")
    assert output.read_bytes() == b"same\n"


def test_stored_receipt_matches_exact_rebuild_and_no_clobber() -> None:
    receipt = registry.check_receipt(ROOT)
    assert receipt == registry.build_receipt(ROOT)
    assert registry.write_receipt(ROOT) == "EXISTING_IDENTICAL"
    assert receipt["claim_boundary"]["scientific_claim_allowed"] is False
    assert receipt["access_accounting"]["scientific_response_rows_opened"] == 0
