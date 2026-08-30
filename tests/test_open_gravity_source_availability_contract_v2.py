from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_source_availability_contract_v2 as source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs() -> tuple[dict, dict]:
    return source.load_inputs()


def _matrix_rows_by_key(
    config: dict, predecessor: dict, wanted: set[tuple[str, str, str]]
) -> dict[tuple[str, str, str], dict]:
    found: dict[tuple[str, str, str], dict] = {}
    for row in source.iter_matrix_rows(config, predecessor):
        key = (row["mechanism_id"], row["object_id"], row["observable_id"])
        if key in wanted:
            found[key] = row
    return found


def test_predecessor_is_exact_append_only_counterevidence(inputs: tuple[dict, dict]) -> None:
    config, predecessor = inputs
    binding = config["predecessor_counterevidence"]
    assert binding["preserved_unchanged"] is True
    assert binding["receipt_payload_opened_by_v2_generator"] is False
    assert source.file_sha256(ROOT / binding["path"]) == binding["file_sha256"]
    assert source.content_sha256(predecessor) == binding["semantic_content_sha256"]
    assert len(binding["counterevidence_findings"]) == 4
    assert len(predecessor["incident_ledger"]) == 2


def test_committed_registry_and_gp01_hashes_are_exact(inputs: tuple[dict, dict]) -> None:
    config, _ = inputs
    bindings = {row["binding_id"]: row for row in config["committed_bindings"]}
    registry = bindings["OPEN_GRAVITY_REGISTRY_FOUNDATION"]
    assert registry["commit"] == "74cf64129787163cbead8dccb243fa4faf86fbe1"
    registry_artifacts = {row["role"]: row for row in registry["artifacts"]}
    assert registry_artifacts["REGISTRY_CONFIG"]["sha256"] == (
        "4b65e4dc919d51462ca78e47c9b1314aa2ac3cf5b8c158b2cc19d758a4214e0d"
    )
    assert registry_artifacts["MECHANISM_CARD_SCHEMA"]["sha256"] == (
        "5c14dc4b4b5e5e457e80410f8e19cf92b575f527ef8b07a00958825c60605396"
    )
    assert registry_artifacts["ZERO_RESPONSE_RECEIPT_HASH_ONLY"]["sha256"] == (
        "58ae29ad74b982af91c63c4c444af30d8528421cb2e0315dfa2501456ab9c23a"
    )

    assert bindings["GP01_FOUNDATION"]["commit"] == ("35f70938f158c81971b2e1b838371b09d9fcee2c")
    preflight = bindings["GP01_XCOP_SOURCE_PREFLIGHT"]
    assert preflight["commit"] == "ed2988546fb1165d9efe5e62d52cddebc7b1a79d"
    assert preflight["receipt_status"] == "SOURCE_ONLY_PREFLIGHT_COMPLETE_ZERO_RESPONSE_ACCESS"
    assert preflight["receipt_decision"] == (
        "SOURCE_ONLY_PREFLIGHT_LOCAL_AND_ELLIPTIC_READY_T1_T2_BLOCKED_NO_Y100_ANCHOR"
    )
    receipt = {row["role"]: row for row in preflight["artifacts"]}[
        "PASS_SOURCE_ONLY_RECEIPT_HASH_ONLY"
    ]
    assert receipt["sha256"] == "3d4d83980d1e1cec45ca5a6353ee827370f33940f5cbe1d9510f83e0cc44de34"


@pytest.mark.parametrize("section", source.STRICT_SECTIONS)
def test_every_strict_section_rejects_mutation(section: str) -> None:
    config = json.loads((ROOT / source.CONFIG_PATH).read_text(encoding="utf-8"))
    mutated = copy.deepcopy(config)
    if section == "authority_boundary":
        mutated[section]["may_claim_DATA_ELIGIBLE"] = True
    elif section == "predecessor_counterevidence":
        mutated[section]["path"] += ".mutated"
    elif section == "inherited_sections":
        mutated[section]["objects"] = "0" * 64
    elif section == "incident_enforcement":
        mutated[section]["retained_incident_count"] = 1
    elif section == "committed_bindings":
        mutated[section][0]["commit"] = "0" * 40
    elif section == "final_bind_gates":
        mutated[section]["TWELL_SUCCESSOR"]["status"] += "_MUTATED"
    elif section == "disposition_contract":
        mutated[section]["static_ready_architectures"].pop()
    elif section == "gp01_dispositions":
        mutated[section]["GP01-L"]["SPARC"]["concept"] = "SOURCE_BLOCKED"
    elif section == "ontology_disposition":
        mutated[section]["nodes"].pop()
    elif section == "comparator_policy":
        mutated[section]["literature_name_never_substitutes_for_source_and_solver"] = False
    elif section == "discriminator_source_block":
        mutated[section]["GROUPS"] = "SOURCE_READY"
    elif section == "manifest_projection_contract":
        mutated[section]["concept_domain_slot_count"] += 1
    elif section == "matrix_contract":
        mutated[section]["expanded_tuple_count"] += 1
    elif section == "claim_boundary":
        mutated[section]["DATA_ELIGIBLE_claimed"] = True
    else:  # pragma: no cover - the parameterization and branch list must stay aligned
        raise AssertionError(section)
    with pytest.raises(
        source.SourceAvailabilityV2Error, match=f"strict section hash changed: {section}"
    ):
        source.validate_config(mutated)


def test_whole_config_seal_rejects_unsectioned_mutation() -> None:
    config = json.loads((ROOT / source.CONFIG_PATH).read_text(encoding="utf-8"))
    config["contract_id"] = "MUTATED"
    with pytest.raises(source.SourceAvailabilityV2Error, match="v2 config semantics changed"):
        source.validate_config(config)


def test_exact_catalog_object_and_observable_axes(inputs: tuple[dict, dict]) -> None:
    config, predecessor = inputs
    catalog = source.mechanism_catalog(predecessor)
    assert len(catalog) == 420
    assert Counter(row["mechanism_family"] for row in catalog) == {
        "TWELL_ATOMIC": 380,
        "TWELL_COMPOUND": 20,
        "GP01": 7,
        "GRAVITY_LIGHT_ONTOLOGY": 13,
    }
    assert source.content_sha256(source.twell_concept_ids()) == source.EXPECTED_TWELL_IDS_SHA256
    assert len(predecessor["objects"]["SPARC"]) == 139
    assert len(predecessor["objects"]["XCOP"]) == 8
    assert len(predecessor["objects"]["XCOP_stellar_profile_available"]) == 5
    assert len(predecessor["objects"]["XCOP_stellar_profile_missing"]) == 3
    assert config["matrix_contract"]["object_observable_slots"] == 155


def test_twell_source_partition_is_exact_at_concept_and_cell_levels(
    inputs: tuple[dict, dict],
) -> None:
    config, predecessor = inputs
    catalog = source.mechanism_catalog(predecessor)[:400]
    by_domain: dict[str, list[tuple]] = {}
    for domain in ("SPARC", "XCOP"):
        dispositions = [source._mechanism_disposition(config, row, domain) for row in catalog]
        by_domain[domain] = dispositions
    assert sum(row[0].startswith("SOURCE_READY") for row in by_domain["SPARC"]) == 60
    assert sum(row[0].startswith("SOURCE_READY") for row in by_domain["XCOP"]) == 126
    assert sum(row[2] for row in by_domain["SPARC"] if row[0].startswith("SOURCE_READY")) == 176
    assert sum(row[2] for row in by_domain["XCOP"] if row[0].startswith("SOURCE_READY")) == 370

    for architecture in ("A15_RETARDED", "A16_MEMORY", "A17_RESONANCE", "A18_STOCHASTIC"):
        rows = [row for row in catalog if row["architecture"] == architecture]
        assert rows
        for domain in ("SPARC", "XCOP"):
            assert all(
                source._mechanism_disposition(config, row, domain)[0] == "SOURCE_BLOCKED"
                for row in rows
            )

    compounds = {
        row["mechanism_id"]: row for row in catalog if row["mechanism_family"] == "TWELL_COMPOUND"
    }
    xcop_ready = {
        mechanism_id
        for mechanism_id, row in compounds.items()
        if source._mechanism_disposition(config, row, "XCOP")[0].startswith("SOURCE_READY")
    }
    assert xcop_ready == {"X01", "X05", "X10", "X13", "X17", "X18"}
    assert not any(
        source._mechanism_disposition(config, row, "SPARC")[0].startswith("SOURCE_READY")
        for row in compounds.values()
    )
    assert source._mechanism_disposition(config, compounds["X01"], "SPARC")[4] == (
        "SOURCE_BLOCKED_NO_HONEST_SPHERICAL_MASS_HISTORY"
    )


def test_matrix_is_deterministic_stream_of_exact_65100_rows(inputs: tuple[dict, dict]) -> None:
    config, predecessor = inputs
    first = source.matrix_summary(config, predecessor)
    second = source.matrix_summary(config, predecessor)
    assert first == second
    assert first["expanded_tuple_count"] == 65_100
    assert first["canonical_row_stream_sha256"] == (
        "bc4f3dbe4d972f098ddc4aac242692fc5b768e47ae70354aaa46eb7ff3ddb154"
    )
    assert first["domain_tuple_counts"] == {"SPARC": 58_380, "XCOP": 6_720}
    assert first["concept_readiness_counts"] == {
        "KNOWN_REWRITE_NONINDEPENDENT": 16,
        "QUARANTINED": 155,
        "SOURCE_BLOCKED": 52_387,
        "SOURCE_READY_STATIC_RADIAL_CONCEPT": 8_479,
        "SOURCE_READY_STATIC_SPHERICAL_RADIAL_CONCEPT": 2_048,
        "THEORY_ONLY": 2_015,
    }
    assert first["parameter_cell_readiness_counts"] == {
        "KNOWN_REWRITE_NONINDEPENDENT_SCORE_ONCE": 16,
        "QUARANTINED_NO_PARAMETER_CELL_EXECUTION": 155,
        "SOURCE_BLOCKED_NO_PARAMETER_CELL_EXECUTION": 52_387,
        "SOURCE_READY_COMPILED_PARAMETER_CELLS_NO_DATA_ELIGIBILITY": 10_527,
        "THEORY_ONLY_NO_PARAMETER_CELLS": 2_015,
    }
    assert first["rows_materialized_in_receipt"] == 0


def test_gp01_and_ontology_dispositions_are_not_overclaimed(inputs: tuple[dict, dict]) -> None:
    config, predecessor = inputs
    wanted = {
        ("GP01-L", "CamB", "ROTATION_CURVE"),
        ("GP01-L", "A85", "PRESSURE_PROFILE"),
        ("GP01-AQUAL", "CamB", "ROTATION_CURVE"),
        ("GP01-AQUAL", "A85", "TEMPERATURE_PROFILE"),
        ("GP01-T1", "A85", "PRESSURE_PROFILE"),
        ("GP01-T2", "A3266", "TEMPERATURE_PROFILE"),
        ("GP01-ELLIPTIC", "CamB", "ROTATION_CURVE"),
        ("GP01-ELLIPTIC", "A85", "PRESSURE_PROFILE"),
        ("GP01-TELEGRAPH", "A85", "PRESSURE_PROFILE"),
        ("GP01-ACTION_PLACEHOLDER", "A85", "PRESSURE_PROFILE"),
        ("QG01", "CamB", "ROTATION_CURVE"),
        ("QG13", "A85", "TEMPERATURE_PROFILE"),
    }
    rows = _matrix_rows_by_key(config, predecessor, wanted)
    assert set(rows) == wanted
    assert rows[("GP01-L", "CamB", "ROTATION_CURVE")]["concept_readiness"] == (
        "SOURCE_READY_STATIC_RADIAL_CONCEPT"
    )
    assert rows[("GP01-L", "A85", "PRESSURE_PROFILE")]["concept_readiness"] == (
        "SOURCE_READY_STATIC_SPHERICAL_RADIAL_CONCEPT"
    )
    assert rows[("GP01-AQUAL", "CamB", "ROTATION_CURVE")]["concept_readiness"] == "SOURCE_BLOCKED"
    assert rows[("GP01-AQUAL", "A85", "TEMPERATURE_PROFILE")]["concept_readiness"] == (
        "KNOWN_REWRITE_NONINDEPENDENT"
    )
    for key in (
        ("GP01-T1", "A85", "PRESSURE_PROFILE"),
        ("GP01-T2", "A3266", "TEMPERATURE_PROFILE"),
    ):
        assert rows[key]["concept_readiness"] == "SOURCE_BLOCKED"
        assert "Y100_ANCHOR" in rows[key]["disposition_reason"]
    assert (
        rows[("GP01-ELLIPTIC", "CamB", "ROTATION_CURVE")]["concept_readiness"] == "SOURCE_BLOCKED"
    )
    elliptic = rows[("GP01-ELLIPTIC", "A85", "PRESSURE_PROFILE")]
    assert elliptic["concept_readiness"] == "SOURCE_READY_STATIC_SPHERICAL_RADIAL_CONCEPT"
    assert elliptic["parameter_cell_readiness"] in {
        "PENDING_EXACT_STATIC_RADIAL_ADAPTER_BIND",
        "SOURCE_READY_COMPILED_PARAMETER_CELLS_NO_DATA_ELIGIBILITY",
    }
    assert (
        rows[("GP01-TELEGRAPH", "A85", "PRESSURE_PROFILE")]["concept_readiness"] == "SOURCE_BLOCKED"
    )
    assert (
        rows[("GP01-ACTION_PLACEHOLDER", "A85", "PRESSURE_PROFILE")]["concept_readiness"]
        == "QUARANTINED"
    )
    assert rows[("QG01", "CamB", "ROTATION_CURVE")]["concept_readiness"] == "THEORY_ONLY"
    assert rows[("QG13", "A85", "TEMPERATURE_PROFILE")]["concept_readiness"] == "THEORY_ONLY"
    assert all(row["DATA_ELIGIBLE_claimed"] is False for row in rows.values())
    assert all(row["campaign_authority_granted"] is False for row in rows.values())


def test_xcop_stellar_split_and_shared_xray_ancestry_are_explicit(
    inputs: tuple[dict, dict],
) -> None:
    config, predecessor = inputs
    wanted = {
        ("GP01-L", "A85", "PRESSURE_PROFILE"),
        ("GP01-L", "A85", "TEMPERATURE_PROFILE"),
        ("GP01-L", "A3266", "PRESSURE_PROFILE"),
        ("GP01-L", "A3266", "TEMPERATURE_PROFILE"),
    }
    rows = _matrix_rows_by_key(config, predecessor, wanted)
    assert (
        rows[("GP01-L", "A85", "PRESSURE_PROFILE")]["stellar_source_status"] == "SOURCE_AVAILABLE"
    )
    assert (
        rows[("GP01-L", "A3266", "PRESSURE_PROFILE")]["stellar_source_status"] == "SOURCE_MISSING"
    )
    assert all(row["shared_xray_measurement_ancestry"] is True for row in rows.values())
    assert {row["observable_source_status"] for row in rows.values()} == {
        "SOURCE_AVAILABLE_SHARED_MEASUREMENT_ANCESTRY"
    }


def test_partitions_comparators_legacy_floor_and_discriminator_are_retained(
    inputs: tuple[dict, dict],
) -> None:
    config, predecessor = inputs
    partition = predecessor["partition_design"]
    assert len(partition["SPARC_pilot"]) == 28
    assert partition["SPARC_validation_count"] == 111
    assert partition["XCOP_pilot"] == ["A85", "A3266"]
    assert len(partition["XCOP_validation"]) == 6
    assert partition["SPARC_pilot_rank_order_sha256"] == (
        "79ca09d283ff059230e65a11b3215f8a9163a336057906539bcdf59090780e8c"
    )
    comparator = {row["id"]: row for row in predecessor["comparator_inventory"]}
    for comparator_id in config["comparator_policy"]["source_or_solver_blocked"]:
        assert comparator_id in comparator
        assert comparator[comparator_id]["SPARC"] in {"SOURCE_MISSING", "UNKNOWN_SOURCE_BLOCKED"}
    assert (
        predecessor["legacy_multiplicity_lower_bound"]["minimum_known_candidate_galaxy_evaluations"]
        == 450_000_000
    )
    assert predecessor["legacy_multiplicity_lower_bound"]["nonoverlap_not_proven"] is True
    discriminator = config["discriminator_source_block"]
    assert {discriminator[domain] for domain in ("SPARC", "XCOP", "GROUPS")} == {"SOURCE_BLOCKED"}
    assert discriminator["group_response_rows_opened"] == 0


def test_manifest_concept_domain_projection_has_stable_hash_for_every_slot(
    inputs: tuple[dict, dict],
) -> None:
    config, predecessor = inputs
    rows = list(source.iter_manifest_concept_domain_rows(config, predecessor))
    assert len(rows) == 420 * 4 == 1680
    assert Counter(row["registry_domain"] for row in rows) == {
        "GALAXIES": 420,
        "GROUPS": 420,
        "CLUSTERS": 420,
        "LENSING": 420,
    }
    assert len({row["source_contract_sha256"] for row in rows}) == len(rows)
    assert all(source.SHA256_RE.fullmatch(row["source_contract_sha256"]) for row in rows)
    assert all(
        row["domain_execution"]["source_contract_sha256"] == row["source_contract_sha256"]
        for row in rows
    )
    assert all(row["domain_execution"]["eligible"] is False for row in rows)
    assert all(row["domain_execution"]["scored"] is False for row in rows)
    assert not any(
        row["domain_execution"]["execution_disposition"] == "SEALED_UNOPENED_FOR_SCORING"
        for row in rows
    )
    assert all(row["candidate_status"] != "READY_FOR_RESPONSE_SCORING" for row in rows)


def test_manifest_parameter_cell_projection_is_exact_and_deterministic(
    inputs: tuple[dict, dict],
) -> None:
    config, predecessor = inputs
    cells = list(source.iter_exact_parameter_cells(config, predecessor))
    assert len(cells) == 2486
    assert len({cell_id for _, cell_id in cells}) == 2486
    assert sum(mechanism["mechanism_family"].startswith("TWELL_") for mechanism, _ in cells) == 1184
    assert sum(mechanism["mechanism_id"] == "GP01-ELLIPTIC" for mechanism, _ in cells) == 1296
    first = source.manifest_projection_summary(config, predecessor)
    second = source.manifest_projection_summary(config, predecessor)
    assert first == second
    assert first["concept_domain_slots"]["slot_count"] == 1680
    assert first["concept_domain_slots"]["canonical_slot_stream_sha256"] == (
        "89f70dace9233b47ceef3e3ec9c71158accf623fa4f5661c786ea6cb86737a03"
    )
    assert first["parameter_cell_domain_slots"]["slot_count"] == 2486 * 4 == 9944
    assert first["parameter_cell_domain_slots"]["canonical_slot_stream_sha256"] == (
        "0ff2ccddfb40f0886f7e7e598f785af80efa90f97256636ffd56caa04d731b5f"
    )
    assert first["parameter_cell_domain_slots"]["unique_source_contract_sha256_count"] == 9944
    assert first["DATA_ELIGIBLE_claimed"] is False
    assert first["campaign_authority_granted"] is False


def test_unrepresentable_cards_and_lanes_are_reported(inputs: tuple[dict, dict]) -> None:
    config, predecessor = inputs
    rows = list(source.iter_manifest_concept_domain_rows(config, predecessor))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mechanism_family"]].append(row)
    assert not any(row["manifest_candidate_representable"] for row in grouped["GP01"])
    assert not any(
        row["manifest_candidate_representable"] for row in grouped["GRAVITY_LIGHT_ONTOLOGY"]
    )
    representability = config["manifest_projection_contract"]["representability"]
    assert representability["ADJACENT_LANE"] == "EMPTY_NO_CARDS"
    assert "MECHANISM_CARDS" in representability["GP01_VARIANTS"]
    assert representability["QG01_QG13"].startswith("ONTOLOGY_NODES_THEORY_ONLY")


def test_dependency_gates_retain_failed_predecessors_and_promote_only_audited_replacements(
    inputs: tuple[dict, dict],
) -> None:
    config, _ = inputs
    assert config["status"] == (
        "SEALED_FINAL_DEPENDENCIES_AUDITED_ZERO_RESPONSE_ACCESS_NO_CAMPAIGN_AUTHORITY"
    )
    gates = config["final_bind_gates"]
    assert set(gates) == {"TWELL_SUCCESSOR", "PRIMARY_SOURCE_PRIOR_ART", "STATIC_RADIAL_ADAPTER"}
    for gate in gates.values():
        assert gate["status"] == source.FINAL_GATE_STATUS
        assert gate["replacement_required"] is False
        assert gate["independent_audit_status"] == gate["required_independent_audit_status"]
        assert {row["role"] for row in gate["artifacts"]} == set(gate["required_roles"])
        assert gate["failed_predecessor_artifacts"]
        assert {row["role"] for row in gate["failed_predecessor_artifacts"]} == set(
            gate["required_roles"]
        )
        assert source.COMMIT_RE.fullmatch(gate["commit"])
    assert gates["TWELL_SUCCESSOR"]["failed_predecessor_audit_findings"] == [
        "INVALID_400_CARD_SEMANTIC_TRANSITIONS",
        "ARBITRARY_CHECK_PATH_LOADER",
    ]
    assert gates["PRIMARY_SOURCE_PRIOR_ART"]["failed_predecessor_audit_findings"] == [
        "ARBITRARY_CHECK_PATH_LOADER"
    ]
    adapter = gates["STATIC_RADIAL_ADAPTER"]
    assert adapter["commit"] == "440ff1fb6d3c2bb586f66466ba6c9fe4c97d3817"
    assert adapter["independent_audit_status"] == "PASS_ADAPTER_OPERATOR_EQUIVALENCE_ONLY"
    assert adapter["failed_predecessor_audit_findings"] == [
        "EXACT_RHO0_MISMATCH",
        "CALLER_BYPASSABLE_CONVERGENCE_TOLERANCE",
        "ARBITRARY_RECEIPT_PATH_LOADER",
    ]
    assert adapter["active_card_root_sha256"] == (
        "3802a35194c618b54679dd04fcfcdbbf55f2339d6fcb3098a2e5c1e7198b1d5d"
    )
    assert adapter["active_GP01_cards_sha256"] == (
        "ab15131a731e9fddff9fead5b8ed0b732ac48626ca6bb7d9970babe414797531"
    )
    assert adapter["active_program_root_sha256"] == (
        "042dbf102ff51a0e3734d340b6a6d1f33b5b6155f4c3352d23d6c8678e85378d"
    )
    twell = gates["TWELL_SUCCESSOR"]
    prior = gates["PRIMARY_SOURCE_PRIOR_ART"]
    assert twell["commit"] == prior["commit"] == "c7fe76fb34fb00c62286394c6a1903ae0722efd0"
    assert twell["active_program_root_sha256"] == (
        "c78139d1dc837ee75913f2e2448cea111c39196e00a6c423ef962ae624f56db3"
    )
    assert twell["active_card_set_root_sha256"] == (
        "b0196cd9c09b1926eace922ca3fd2f1c7e3eca4be440c68e7ff7a63d5706865d"
    )
    assert twell["active_candidate_root_sha256"] == (
        "fe88c17fa2f6367b01b933d55acb61d30ca792cbf423034a9bb6ecf01a59ce4d"
    )
    assert twell["active_equivalence_root_sha256"] == (
        "714422f3440d05e347095b62a084dd077f3628a338cb5d0919cf0b9cb39a4e7c"
    )


def test_generator_does_not_open_any_upstream_receipt_payload(
    inputs: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor = inputs
    forbidden = {
        str((ROOT / config["predecessor_counterevidence"]["receipt_path"]).resolve()),
        *{
            str((ROOT / artifact["path"]).resolve())
            for binding in config["committed_bindings"]
            for artifact in binding["artifacts"]
            if "RECEIPT" in artifact["role"]
        },
        *{
            str((ROOT / artifact["path"]).resolve())
            for gate in config["final_bind_gates"].values()
            for ledger_key in ("failed_predecessor_artifacts", "artifacts")
            for artifact in gate.get(ledger_key, [])
            if artifact["role"] == "RECEIPT" and "path" in artifact
        },
    }
    real_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if str(path.resolve()) in forbidden:
            raise AssertionError(f"upstream receipt payload opened: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert source.matrix_summary(config, predecessor)["expanded_tuple_count"] == 65100
    assert (
        source.manifest_projection_summary(config, predecessor)["concept_domain_slots"][
            "slot_count"
        ]
        == 1680
    )


def test_final_seal_is_impossible_while_any_agent_hash_gate_is_pending(
    inputs: tuple[dict, dict],
) -> None:
    config, _ = inputs
    if source._all_final_gates_bound(config):
        receipt = source.build_receipt()
        assert receipt["campaign_manifest"]["DATA_ELIGIBLE_claimed"] is False
    else:
        with pytest.raises(
            source.SourceAvailabilityV2Error, match="final bind gates remain pending"
        ):
            source.build_receipt()


def test_append_only_atomic_writer_refuses_nonidentical_content(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert source._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert source._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(source.SourceAvailabilityV2Error, match="refusing to overwrite"):
        source._atomic_no_clobber(path, b"two\n")
    assert path.read_bytes() == b"one\n"


def test_sealed_receipt_is_zero_response_and_contains_no_stream_rows(
    inputs: tuple[dict, dict],
) -> None:
    config, _ = inputs
    if not source._all_final_gates_bound(config):
        pytest.skip("final agent-reported hash gates are intentionally pending")
    receipt = source.build_receipt()
    assert set(receipt["zero_access"].values()) == {0}
    assert receipt["matrix"]["rows_materialized_in_receipt"] == 0
    assert (
        receipt["manifest_projection"]["concept_domain_slots"]["rows_materialized_in_receipt"] == 0
    )
    assert (
        receipt["manifest_projection"]["parameter_cell_domain_slots"][
            "rows_materialized_in_receipt"
        ]
        == 0
    )
    assert receipt["campaign_manifest"] == {
        "status": "NOT_CREATED_BY_THIS_CONTRACT",
        "DATA_ELIGIBLE_claimed": False,
        "response_execution_authorized": False,
    }
    encoded = json.dumps(receipt, sort_keys=True)
    assert '"rows"' not in encoded
    assert "READY_FOR_RESPONSE_SCORING" not in encoded
    assert "SEALED_UNOPENED_FOR_SCORING" not in encoded


def _canonical_artifact_bytes() -> tuple[bytes, bytes, bytes]:
    return (
        (ROOT / source.CONFIG_PATH).read_bytes(),
        (ROOT / source.MODULE_PATH).read_bytes(),
        (ROOT / source.TEST_PATH).read_bytes(),
    )


def _reseal_forged_receipt(
    receipt: dict, config_bytes: bytes, module_bytes: bytes, test_bytes: bytes
) -> None:
    receipt["bindings"]["config"]["sha256"] = source.bytes_sha256(config_bytes)
    receipt["bindings"]["config"]["semantic_sha256"] = source.content_sha256(
        json.loads(config_bytes)
    )
    receipt["bindings"]["module"]["sha256"] = source.bytes_sha256(module_bytes)
    receipt["bindings"]["module"]["semantic_sha256"] = source.implementation_semantic_sha256(
        module_bytes
    )
    receipt["bindings"]["test"]["sha256"] = source.bytes_sha256(test_bytes)
    receipt["bindings"]["test"]["semantic_sha256"] = source.text_semantic_sha256(test_bytes)
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = source.content_sha256(receipt)


def test_artifact_raw_and_semantic_seals_are_exact() -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    assert source.bytes_sha256(config_bytes) == source.EXPECTED_CONFIG_FILE_SHA256
    assert source.content_sha256(json.loads(config_bytes)) == (
        source.EXPECTED_CONFIG_CONTENT_SHA256
    )
    assert source.implementation_semantic_sha256(module_bytes) == (
        source.EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256
    )
    assert source.bytes_sha256(test_bytes) == source.EXPECTED_TEST_FILE_SHA256
    assert source.text_semantic_sha256(test_bytes) == source.EXPECTED_TEST_SEMANTIC_SHA256


def test_coordinated_module_test_and_receipt_forgery_is_rejected() -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    forged = copy.deepcopy(source.build_receipt())
    tampered_module = module_bytes.replace(
        b"The module reads only its metadata contracts.",
        b"The module reads just its metadata contracts.",
        1,
    )
    tampered_test = test_bytes + b"\n# coordinated forgery\n"
    assert tampered_module != module_bytes
    _reseal_forged_receipt(forged, config_bytes, tampered_module, tampered_test)
    with pytest.raises(
        source.SourceAvailabilityV2Error,
        match="implementation semantic hash changed",
    ):
        source.validate_receipt_payload(
            forged,
            config_bytes=config_bytes,
            module_bytes=tampered_module,
            test_bytes=tampered_test,
        )


def test_coordinated_test_and_receipt_forgery_is_rejected() -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    forged = copy.deepcopy(source.build_receipt())
    tampered_test = test_bytes + b"\n# forged test mutation\n"
    _reseal_forged_receipt(forged, config_bytes, module_bytes, tampered_test)
    with pytest.raises(source.SourceAvailabilityV2Error, match="test raw file hash changed"):
        source.validate_receipt_payload(
            forged,
            config_bytes=config_bytes,
            module_bytes=module_bytes,
            test_bytes=tampered_test,
        )


def test_test_semantic_seal_rejects_mutation_after_raw_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    tampered_test = test_bytes + b"\n# semantic-layer forgery\n"
    monkeypatch.setattr(source, "EXPECTED_TEST_FILE_SHA256", source.bytes_sha256(tampered_test))
    with pytest.raises(source.SourceAvailabilityV2Error, match="test semantic hash changed"):
        source.validate_artifact_integrity(config_bytes, module_bytes, tampered_test)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload + b" ", "config raw file hash changed"),
        (
            lambda payload: payload.replace(
                b'"contract_id": "OPEN-GRAVITY-SOURCE-AVAILABILITY-002"',
                b'"contract_id": "FORGED-GRAVITY-SOURCE-AVAILABILITY"',
                1,
            ),
            "config raw file hash changed",
        ),
    ),
)
def test_config_raw_seal_rejects_coherent_receipt_mutation(mutation, message: str) -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    forged = copy.deepcopy(source.build_receipt())
    tampered_config = mutation(config_bytes)
    assert tampered_config != config_bytes
    _reseal_forged_receipt(forged, tampered_config, module_bytes, test_bytes)
    with pytest.raises(source.SourceAvailabilityV2Error, match=message):
        source.validate_receipt_payload(
            forged,
            config_bytes=tampered_config,
            module_bytes=module_bytes,
            test_bytes=test_bytes,
        )


def test_config_semantic_seal_rejects_mutation_after_raw_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    tampered_config = config_bytes.replace(
        b'"contract_id": "OPEN-GRAVITY-SOURCE-AVAILABILITY-002"',
        b'"contract_id": "FORGED-GRAVITY-SOURCE-AVAILABILITY"',
        1,
    )
    assert tampered_config != config_bytes
    monkeypatch.setattr(
        source,
        "EXPECTED_CONFIG_FILE_SHA256",
        source.bytes_sha256(tampered_config),
    )
    with pytest.raises(source.SourceAvailabilityV2Error, match="config semantic hash changed"):
        source.validate_artifact_integrity(tampered_config, module_bytes, test_bytes)


def test_output_path_is_rejected_before_any_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source, "OUTPUT_PATH", Path("attacker/receipt.json"))

    def forbidden_read(*args, **kwargs):
        raise AssertionError("filesystem read occurred before output-path validation")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(source.SourceAvailabilityV2Error, match="canonical output path changed"):
        source.validate_receipt()


def test_cli_rejects_arbitrary_root_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def intercepted_validator() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(source, "validate_receipt", intercepted_validator)
    with pytest.raises(SystemExit) as exc_info:
        source.main(["validate", "--root", "C:/attacker-root"])
    assert exc_info.value.code == 2
    assert called is False


def test_canonical_receipt_validates() -> None:
    source.validate_receipt()
