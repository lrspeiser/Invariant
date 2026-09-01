from __future__ import annotations

import copy
import os
from collections import Counter
from pathlib import Path

import pytest

import sigma_theory_compiler.twell_400_v2_typed_compiler_packet as twell
from sigma_theory_compiler.twell_400_v2_typed_compiler_packet import (
    CARDS_PATH,
    CONFIG_PATH,
    EXPECTED_CONFIG_CANONICAL_SHA256,
    EXPECTED_SECTION_SEALS,
    EXPECTED_UNSEALED_ROOT_SHA256,
    FAILED_CARDS_PATH,
    MODULE_PATH,
    RECEIPT_PATH,
    TEST_PATH,
    TwellCompilerError,
    _atomic_packet_no_clobber,
    _failed_card_hashes,
    _internal_card_errors,
    _mixed_boundary_helmholtz,
    _neumann_helmholtz,
    build_packet,
    cards_bytes,
    check_packet,
    compile_rows,
    content_sha256,
    load_config,
    ordered_concept_ids,
    receipt_content_sha256,
    stream_root,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _compile() -> list[dict]:
    return compile_rows(
        load_config(ROOT),
        {
            "code_sha256": SHA,
            "data_sha256": SHA,
            "environment_sha256": SHA,
            "configuration_sha256": SHA,
        },
        _failed_card_hashes((ROOT / FAILED_CARDS_PATH).read_bytes()),
    )


@pytest.fixture(scope="module")
def compiled() -> list[dict]:
    return _compile()


def _row(rows: list[dict], concept_id: str) -> dict:
    return next(row for row in rows if row["concept_id"] == concept_id)


def _cell(row: dict, **parameters: float) -> dict:
    return next(
        cell
        for cell in row["cell_results"]
        if cell["cell_kind"] == "CARTESIAN"
        and all(cell["parameters"][key] == value for key, value in parameters.items())
    )


def _fixture_digest(cell: dict, fixture: str = "SMOOTH_BOUNDED_SOURCE") -> str:
    return next(
        result["output_sha256"]
        for result in cell["fixture_results"]
        if result["fixture"] == fixture
    )


def test_enumeration_is_exactly_20_by_19_plus_20_in_frozen_order() -> None:
    config = load_config(ROOT)
    identifiers = ordered_concept_ids()
    assert len(config["driver_catalog"]) == 20
    assert len(config["architecture_catalog"]) == 19
    assert len(config["compound_catalog"]) == 20
    assert len(identifiers) == 400
    assert identifiers[0] == "TW2-A01-D01"
    assert identifiers[19] == "TW2-A01-D20"
    assert identifiers[20] == "TW2-A02-D01"
    assert identifiers[379] == "TW2-A19-D20"
    assert identifiers[380:] == [f"X{index:02d}" for index in range(1, 21)]
    assert content_sha256(identifiers) == (
        "7388f8982c5014ef6c365d00aa780ba2ecb8b8b3f6786658fb3db36b64c29c5f"
    )


def test_x02_through_x20_operations_are_explicitly_frozen() -> None:
    compounds = {row["id"]: row for row in load_config(ROOT)["compound_catalog"]}
    expected = {
        "X02": "u=u1*u2",
        "X03": "u=u1*(1-abs(u2))",
        "X04": "u=(u1+u2+u1*u2)/3",
        "X05": "u=(u1-u2)/2",
        "X06": "u=u1*u2",
        "X07": "u=sign(u1*u2)*sqrt(abs(u1*u2))",
        "X08": "u=(u1*u2)/(1+abs(u1*u2))",
        "X09": "u=(2*u1+u2)/3",
        "X10": "u=u1/(1+abs(u2))",
        "X11": "u=tanh(u1+u2)",
        "X12": "u=u1*cos(pi*u2)",
        "X13": "u=(u1+2*u2)/3",
        "X14": "u=u1*(1+u2)/2",
        "X15": "u=u2/(1+abs(u1))",
        "X16": "u=u1*u2",
        "X17": "u=u1/(1+abs(u2))",
        "X18": "u=(u1+u2)/2",
        "X19": "u=u1; sigma_eff=sigma_cell*abs(u2)",
        "X20": "u=u2; kappa_eff=kappa_cell*abs(u1)",
    }
    assert {key: compounds[key]["operation"] for key in expected} == expected
    assert all(compounds[key]["operation_id"] for key in expected)


def test_full_cartesian_parameter_grid_has_exactly_1184_cells(
    compiled: list[dict],
) -> None:
    assert sum(row["parameter_cell_count"] for row in compiled) == 1184
    assert (
        sum(cell["cell_kind"] == "CARTESIAN" for row in compiled for cell in row["cell_results"])
        == 1182
    )
    assert (
        sum(
            cell["cell_kind"] == "COMPOUND_OVERRIDE_EVIDENCE"
            for row in compiled
            for cell in row["cell_results"]
        )
        == 2
    )
    for row in compiled:
        identifiers = [cell["cell_id"] for cell in row["cell_results"]]
        assert len(identifiers) == len(set(identifiers)) == row["parameter_cell_count"]
        assert len(row["card"]["parameter_cells"]) == row["parameter_cell_count"]


def test_every_cell_has_computed_evidence_and_exact_probe_pass(
    compiled: list[dict],
) -> None:
    assert Counter(row["probe_status"] for row in compiled) == {
        "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES": 400
    }
    cells = [cell for row in compiled for cell in row["cell_results"]]
    assert Counter(cell["status"] for cell in cells) == {
        "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES": 1184
    }
    for cell in cells:
        assert cell["execution_class"] == "EXACT_FROZEN_OPERATOR"
        dimension = cell["computed_dimension_evidence"]
        assert dimension["computed"] is True
        assert dimension["status"] == "PASS"
        assert dimension["residual_count"] == 0
        assert len(cell["fixture_results"]) == 3
        for fixture in cell["fixture_results"]:
            assert fixture["finite_violation_count"] == 0
            assert fixture["deterministic_replay_residual"] == 0
            assert fixture["deterministic_evidence_replay_residual"] == 0
            assert fixture["computed_operator_residual"] <= 1e-10
            assert fixture["computed_boundary_or_initial_residual"] <= 1e-10
            assert fixture["computed_analytic_lambda_zero_residual"] <= 1e-12
            assert fixture["operator_evidence"]["computed"] is True


def test_a06_exact_neumann_solver_uses_both_ell_cells(compiled: list[dict]) -> None:
    row = _row(compiled, "TW2-A06-D01")
    first = _cell(row, **{"lambda": 0.25, "ell": 0.1})
    second = _cell(row, **{"lambda": 0.25, "ell": 0.25})
    assert _fixture_digest(first) != _fixture_digest(second)
    values = [index / 32 for index in range(33)]
    state_01, residual_01 = _neumann_helmholtz(values, 0.1)
    state_025, residual_025 = _neumann_helmholtz(values, 0.25)
    assert state_01 != state_025
    assert residual_01["operator_residual"] <= 1e-10
    assert residual_025["operator_residual"] <= 1e-10
    assert residual_01["boundary_residual"] == 0
    assert residual_025["boundary_residual"] == 0


def test_a12_exact_mixed_boundary_solver_uses_both_mu_cells(
    compiled: list[dict],
) -> None:
    row = _row(compiled, "TW2-A12-D01")
    first = _cell(row, **{"lambda": 0.25, "mu": 1.0})
    second = _cell(row, **{"lambda": 0.25, "mu": 4.0})
    assert _fixture_digest(first) != _fixture_digest(second)
    values = [index / 32 for index in range(33)]
    state_1, residual_1 = _mixed_boundary_helmholtz(values, 1.0)
    state_4, residual_4 = _mixed_boundary_helmholtz(values, 4.0)
    assert state_1 != state_4
    assert state_1[-1] == state_4[-1] == 0
    assert residual_1["operator_residual"] <= 1e-10
    assert residual_4["operator_residual"] <= 1e-10
    assert residual_1["boundary_residual"] == 0
    assert residual_4["boundary_residual"] == 0


@pytest.mark.parametrize(
    ("concept_id", "first", "second"),
    [
        (
            "TW2-A10-D01",
            {"lambda": 0.25, "u_c": 0.25},
            {"lambda": 0.25, "u_c": 0.5},
        ),
        (
            "TW2-A11-D01",
            {"lambda": 0.25, "s_c": 0.5},
            {"lambda": 0.25, "s_c": 1.0},
        ),
        (
            "TW2-A14-D01",
            {"lambda": 0.25, "k": 1.0},
            {"lambda": 0.25, "k": 2.0},
        ),
        (
            "TW2-A16-D01",
            {"lambda": 0.25, "tau": 0.1},
            {"lambda": 0.25, "tau": 0.5},
        ),
        (
            "TW2-A18-D01",
            {"lambda": 0.25, "sigma": 0.0},
            {"lambda": 0.25, "sigma": 0.05},
        ),
        (
            "TW2-A19-D01",
            {"lambda": 0.25, "kappa": 0.0},
            {"lambda": 0.25, "kappa": 0.5},
        ),
    ],
)
def test_each_previously_ignored_parameter_changes_the_exact_probe(
    compiled: list[dict], concept_id: str, first: dict, second: dict
) -> None:
    row = _row(compiled, concept_id)
    assert _fixture_digest(_cell(row, **first)) != _fixture_digest(_cell(row, **second))


def test_a13_kernel_is_defined_as_exact_a06_inverse(compiled: list[dict]) -> None:
    config = load_config(ROOT)
    architecture = next(
        row for row in config["architecture_catalog"] if row["id"] == "A13_MIXED_MODE"
    )
    assert "K_ell is exactly the A06 Neumann finite-difference inverse" in " ".join(
        architecture["canonical_expressions"]
    )
    row = _row(compiled, "TW2-A13-D01")
    first = _cell(row, **{"lambda": 0.25, "theta": 0.0, "ell": 0.25})
    second = _cell(row, **{"lambda": 0.25, "theta": 0.7853981633974483, "ell": 0.25})
    assert _fixture_digest(first) != _fixture_digest(second)


def test_all_lambda_and_compound_override_cells_are_executed(
    compiled: list[dict],
) -> None:
    assert all(
        {cell["parameters"]["lambda"] for cell in row["cell_results"]} == {0.0, 0.25}
        for row in compiled
    )
    for concept_id, effective_key in (
        ("X19", "sigma_eff_digest_sha256"),
        ("X20", "kappa_eff_digest_sha256"),
    ):
        row = _row(compiled, concept_id)
        override = next(
            cell
            for cell in row["cell_results"]
            if cell["cell_kind"] == "COMPOUND_OVERRIDE_EVIDENCE"
        )
        assert override["status"] == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES"
        for fixture in override["fixture_results"]:
            assert effective_key in fixture["operator_evidence"]["effective_parameters"]


def test_cards_use_internal_contract_and_append_only_failed_lineage(
    compiled: list[dict],
) -> None:
    old_hashes = _failed_card_hashes((ROOT / FAILED_CARDS_PATH).read_bytes())
    for row in compiled:
        card = row["card"]
        assert _internal_card_errors(card) == []
        assert card["semantic_version"] == "2.1.0"
        assert card["action_or_equations"]["executable"] is True
        assert card["version_change"]["previous_card_sha256"] == old_hashes[row["concept_id"]]
        assert card["version_change"]["prior_result_retained"] is True
        assert card["version_change"]["replay_all_affected"] is True
        assert card["closures"]["photon"] == "L0_NO_LIGHT_CLAIM"
        assert card["closures"]["capture"] == "C0_ISOLATED_CONSERVATIVE"


def test_failed_probe_is_formula_basis_only_and_never_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = twell._evaluate_operator

    def injected_failure(architecture_id: str, *args: object, **kwargs: object) -> object:
        if architecture_id == "A06_SPATIAL_KERNEL":
            raise TwellCompilerError("synthetic exact-operator failure")
        return original(architecture_id, *args, **kwargs)

    monkeypatch.setattr(twell, "_evaluate_operator", injected_failure)
    rows = _compile()
    failed = [row for row in rows if row["architecture_id"] == "A06_SPATIAL_KERNEL"]
    assert len(failed) == 22
    assert all(row["compiler_status"] == "INCOMPLETE_QUARANTINE" for row in failed)
    assert all(row["execution_class"] == "FORMULA_BASIS_ONLY" for row in failed)
    assert all(row["card"]["action_or_equations"]["executable"] is False for row in failed)
    assert all("READY_FOR_THEORY_GATES" not in row["domain_admission"].values() for row in failed)


def test_source_partition_is_exact_at_concept_and_parameter_cell_levels(
    compiled: list[dict],
) -> None:
    assert Counter(row["compiler_status"] for row in compiled) == {
        "READY_FOR_THEORY_GATES": 143,
        "SOURCE_BLOCKED": 257,
    }
    assert Counter(row["domain_admission"]["SPARC"] for row in compiled) == {
        "READY_FOR_THEORY_GATES": 68,
        "SOURCE_BLOCKED": 332,
    }
    assert Counter(row["domain_admission"]["XCOP"] for row in compiled) == {
        "READY_FOR_THEORY_GATES": 143,
        "SOURCE_BLOCKED": 257,
    }
    assert Counter(row["compiler_status"] for row in compiled for _cell in row["cell_results"]) == {
        "READY_FOR_THEORY_GATES": 422,
        "SOURCE_BLOCKED": 762,
    }
    assert Counter(
        row["domain_admission"]["SPARC"] for row in compiled for _cell in row["cell_results"]
    ) == {"READY_FOR_THEORY_GATES": 200, "SOURCE_BLOCKED": 984}
    assert Counter(
        row["domain_admission"]["XCOP"] for row in compiled for _cell in row["cell_results"]
    ) == {"READY_FOR_THEORY_GATES": 422, "SOURCE_BLOCKED": 762}


def test_history_architectures_pass_synthetic_operator_but_stay_source_blocked(
    compiled: list[dict],
) -> None:
    history = [row for row in compiled if row["architecture_id"] in {"A15_RETARDED", "A16_MEMORY"}]
    assert len(history) == 42
    assert all(row["probe_status"] == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" for row in history)
    assert all(row["compiler_status"] == "SOURCE_BLOCKED" for row in history)


def test_equivalence_audit_separates_exact_rewrite_from_probe_degeneracy(
    compiled: list[dict],
) -> None:
    exact = Counter(row["equivalence_family_id"] for row in compiled)
    probes = Counter(row["probe_digest_sha256"] for row in compiled)
    assert len(exact) == 400
    assert all(count == 1 for count in exact.values())
    assert any(count > 1 for count in probes.values())
    for driver in range(1, 21):
        identifiers = [f"TW2-A{architecture:02d}-D{driver:02d}" for architecture in (1, 2, 8)]
        rows = [row for row in compiled if row["concept_id"] in identifiers]
        assert len({row["probe_digest_sha256"] for row in rows}) == 1
        assert len({row["equivalence_family_id"] for row in rows}) == 3


def test_mechanism_schema_registry_and_gp01_are_deferred_non_authorizing() -> None:
    config = load_config(ROOT)
    deferred = config["governance_bindings"]["deferred_informational"]
    assert {row["binding_id"] for row in deferred} == {
        "MECHANISM-CARD-SCHEMA-FINAL",
        "REGISTRY-FOUNDATION-FINAL-RECEIPT",
        "GP01-FOUNDATION-FINAL-RECEIPT",
    }
    assert all("sha256" not in row for row in deferred)
    assert all(row["may_authorize_campaign"] is False for row in deferred)
    assert config["governance_bindings"]["campaign_manifest_frozen"] is False
    assert config["claim_boundary"]["campaign_manifest_frozen"] is False


def _mutate(config: dict, section: str) -> None:
    if section == "identity":
        config[section]["status"] += "_MUTATED"
    elif section == "governance_bindings":
        config[section]["hard_bound"][0]["sha256"] = "0" * 64
    elif section == "access_contract":
        config[section]["zero_access"]["scientific_response_rows_opened"] = 1
    elif section == "driver_catalog":
        config[section][0]["reference_value"] = 2e-10
    elif section == "architecture_catalog":
        config[section][0]["canonical_expressions"][0] += "_MUTATED"
    elif section == "compound_catalog":
        config[section][1]["operation"] += "_MUTATED"
    elif section == "compiler_contract":
        config[section]["deterministic_seed"] += 1
    elif section == "probe_contract":
        config[section]["expected_concept_parameter_cell_count"] += 1
    elif section == "status_contract":
        config[section]["ready_rule"] += " mutated"
    elif section == "output_contract":
        config[section]["cards_path"] += ".mutated"
    elif section == "claim_boundary":
        config[section]["scientific_validity_claimed"] = True
    else:  # pragma: no cover
        raise AssertionError(section)


@pytest.mark.parametrize("section", tuple(EXPECTED_SECTION_SEALS))
def test_every_semantic_section_mutation_fails_closed(section: str) -> None:
    config = copy.deepcopy(load_config(ROOT))
    _mutate(config, section)
    with pytest.raises(TwellCompilerError, match="sealed section changed"):
        validate_config(config)


def test_coordinated_content_and_seal_mutation_fails_hardcoded_seal() -> None:
    config = copy.deepcopy(load_config(ROOT))
    config["compound_catalog"][1]["operation"] += "_COORDINATED"
    config["section_seals"]["compound_catalog"] = content_sha256(config["compound_catalog"])
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    config["section_seals"]["unsealed_root_sha256"] = content_sha256(unsealed)
    with pytest.raises(TwellCompilerError, match="sealed section changed"):
        validate_config(config)


def test_config_and_section_seals_are_exact() -> None:
    config = load_config(ROOT)
    assert content_sha256(config) == EXPECTED_CONFIG_CANONICAL_SHA256
    assert config["section_seals"] == {
        **EXPECTED_SECTION_SEALS,
        "unsealed_root_sha256": EXPECTED_UNSEALED_ROOT_SHA256,
    }


def test_build_reads_only_allowlisted_metadata_and_failed_cards_counterevidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT)
    allowed = {
        (ROOT / CONFIG_PATH).resolve(),
        (ROOT / MODULE_PATH).resolve(),
        (ROOT / TEST_PATH).resolve(),
        (ROOT / FAILED_CARDS_PATH).resolve(),
    }
    for row in config["governance_bindings"]["hard_bound"]:
        path = Path(row["path"])
        allowed.add((path if path.is_absolute() else ROOT / path).resolve())
    forbidden_deferred = {
        (ROOT / row["path"]).resolve()
        for row in config["governance_bindings"]["deferred_informational"]
    }
    failed_receipt = (
        ROOT / config["output_contract"]["failed_packet_retained"]["receipt_path"]
    ).resolve()
    original = Path.read_bytes
    opened: list[Path] = []

    def traced(path: Path) -> bytes:
        resolved = path.resolve()
        opened.append(resolved)
        assert resolved in allowed
        assert resolved not in forbidden_deferred
        assert resolved != failed_receipt
        if "runs\\gravity" in str(resolved).lower():
            assert resolved == (ROOT / FAILED_CARDS_PATH).resolve()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", traced)
    rows, payload, receipt = build_packet(ROOT)
    assert len(rows) == 400
    assert len(payload.splitlines()) == 400
    assert set(opened) == allowed
    assert receipt["mechanism_schema_status"]["read_or_hashed"] is False
    audit = receipt["access_audit"]
    assert audit["runs_gravity_result_receipts_opened"] == 0
    assert audit["astronomy_response_payloads_opened"] == 0
    for field in config["access_contract"]["zero_access"]:
        assert audit[field] == 0


def test_receipt_records_exact_cell_and_status_counts() -> None:
    rows, payload, receipt = build_packet(ROOT)
    assert receipt["semantic_version"] == "1.1.0"
    assert receipt["parameter_cell_summary"] == {
        "total_count": 1184,
        "base_cartesian_count": 1182,
        "compound_override_evidence_count": 2,
        "probe_status_counts": {"PASS_TARGET_FREE_EXACT_OPERATOR_PROBES": 1184},
        "compiler_status_counts": {
            "READY_FOR_THEORY_GATES": 422,
            "SOURCE_BLOCKED": 762,
        },
        "domain_admission_counts": {
            "SPARC": {"READY_FOR_THEORY_GATES": 200, "SOURCE_BLOCKED": 984},
            "XCOP": {"READY_FOR_THEORY_GATES": 422, "SOURCE_BLOCKED": 762},
        },
    }
    assert receipt["probe_summary"]["passed_card_count"] == 400
    assert receipt["probe_summary"]["failed_card_count"] == 0
    assert receipt["probe_summary"]["maximum_computed_operator_residual"] <= 1e-10
    assert receipt["stream"]["file_sha256"] == twell._sha256_bytes(payload)
    assert len(rows) == 400


def test_stream_is_canonical_deterministic_and_rooted() -> None:
    rows, payload, receipt = build_packet(ROOT)
    assert payload == cards_bytes(rows)
    assert receipt["stream"]["ordered_line_root_sha256"] == stream_root(rows)
    assert receipt["stream"]["line_count"] == 400
    assert receipt["receipt_content_sha256"] == receipt_content_sha256(receipt)
    assert build_packet(ROOT)[1:] == (payload, receipt)


def test_atomic_two_file_writer_is_no_clobber(tmp_path: Path) -> None:
    cards = tmp_path / "packet" / "cards.jsonl"
    receipt = tmp_path / "packet" / "receipt.json"
    _atomic_packet_no_clobber(cards, b"{}\n", receipt, b"{}\n")
    assert cards.read_bytes() == b"{}\n"
    assert receipt.read_bytes() == b"{}\n"
    with pytest.raises(TwellCompilerError, match="refusing to overwrite"):
        _atomic_packet_no_clobber(cards, b"changed\n", receipt, b"changed\n")
    assert cards.read_bytes() == b"{}\n"
    assert receipt.read_bytes() == b"{}\n"


def test_receipt_race_rolls_back_only_our_link_even_when_receipt_now_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = tmp_path / "race" / "cards.jsonl"
    receipt = tmp_path / "race" / "receipt.json"
    real_link = os.link
    calls = 0

    def racing_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_link(source, destination)
            return
        receipt.write_bytes(b"competitor-receipt\n")
        raise FileExistsError("adversarial receipt publication")

    monkeypatch.setattr(twell.os, "link", racing_link)
    with pytest.raises(TwellCompilerError, match="refusing to overwrite"):
        _atomic_packet_no_clobber(cards, b"our-cards\n", receipt, b"our-receipt\n")
    assert not cards.exists()
    assert receipt.read_bytes() == b"competitor-receipt\n"


def test_retained_failed_packet_is_unchanged_counterevidence() -> None:
    config = load_config(ROOT)
    retained = config["output_contract"]["failed_packet_retained"]
    assert (
        twell._sha256_bytes((ROOT / retained["cards_path"]).read_bytes())
        == retained["cards_sha256"]
    )
    assert (
        twell._sha256_bytes((ROOT / retained["receipt_path"]).read_bytes())
        == retained["receipt_sha256"]
    )
    assert retained["audit_status"] == "BLOCKED_SUPERSEDED_RETAINED_AS_COUNTEREVIDENCE"


def test_stored_packet_matches_exact_rebuild() -> None:
    stored = check_packet(ROOT, ROOT / CARDS_PATH, ROOT / RECEIPT_PATH)
    assert stored == build_packet(ROOT)[2]
