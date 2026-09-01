from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.open_gravity_primary_source_comparator_contract_v1 import (
    ANALOGY_IDS_REQUIRED,
    COMPARATOR_IDS_REQUIRED,
    CONFIG_PATH,
    EXPECTED_CONFIG_CANONICAL_SHA256,
    EXPECTED_SECTION_SEALS,
    EXPECTED_UNSEALED_ROOT_SHA256,
    GP01_FINDING_IDS_REQUIRED,
    MODULE_PATH,
    OUTPUT_PATH,
    QG_IDS_REQUIRED,
    SOURCE_IDS_REQUIRED,
    TEST_PATH,
    ZERO_ACCESS_FIELDS,
    OpenGravityPriorArtError,
    _atomic_no_clobber,
    build_receipt,
    check_receipt,
    content_sha256,
    load_config,
    receipt_content_sha256,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows}


def test_contract_inventory_and_exact_source_versions_are_frozen() -> None:
    config = load_config(ROOT)
    assert content_sha256(config) == EXPECTED_CONFIG_CANONICAL_SHA256
    assert config["section_seals"] == {
        **EXPECTED_SECTION_SEALS,
        "unsealed_root_sha256": EXPECTED_UNSEALED_ROOT_SHA256,
    }

    sources = _by_id(config["primary_sources"], "source_id")
    assert set(sources) == SOURCE_IDS_REQUIRED
    assert sources["SRC-AQUAL-1984"]["url"] == (
        "https://articles.adsabs.harvard.edu/pdf/1984ApJ...286....7B"
    )
    assert sources["SRC-AQUAL-1984"]["exact_version"] == (
        "Astrophysical Journal 286 (1984), pages 7-14, version of record"
    )
    assert sources["SRC-PENNER-2026"]["url"] == "https://arxiv.org/abs/2602.09249v1"
    assert sources["SRC-RG-2016"]["url"] == "https://arxiv.org/abs/1603.04943v1"
    assert sources["SRC-EMOND-2012"]["url"] == "https://arxiv.org/abs/1207.6232v1"
    assert sources["SRC-MOG-2016"]["url"] == "https://arxiv.org/abs/1610.06909v2"
    assert sources["SRC-NONLOCAL-2012"]["url"] == "https://arxiv.org/abs/1111.4702v2"
    assert sources["SRC-NFW-1997"]["doi"] == "10.1086/304888"
    assert sources["SRC-EINASTO-2004"]["doi"] == "10.1111/j.1365-2966.2004.07586.x"
    for source in sources.values():
        assert source["url"].startswith("https://")
        assert source["exact_version"]
        assert source["operational_anchor"]
        assert source["parameters_and_boundaries"]
        assert source["claimed_domain"]


def test_gp01_novelty_boundary_preserves_all_required_negative_claims() -> None:
    config = load_config(ROOT)
    boundary = config["gp01_novelty_boundary"]
    findings = _by_id(boundary["frozen_findings"], "finding_id")
    assert set(findings) == GP01_FINDING_IDS_REQUIRED
    assert boundary["historical_novelty_status"] == (
        "OPEN_REQUIRES_INDEPENDENT_HUMAN_PRIOR_ART_REVIEW"
    )
    assert boundary["human_review_required_before_novelty_language"] is True
    assert "not evidence" in boundary["no_negative_search_inference"]
    assert findings["GP01-LOCAL-AQUAL-BOUNDARY"]["relationship"] == (
        "KNOWN_HIGH_SYMMETRY_EQUIVALENCE_NOT_GLOBAL_REWRITE"
    )
    assert "curl field" in findings["GP01-LOCAL-AQUAL-BOUNDARY"]["exact_statement"]
    assert "external vector field" in findings["GP01-AQUAL-EFE-BOUNDARY"]["exact_statement"]
    assert findings["GP01-T1-INTEGRABILITY-BOUNDARY"]["relationship"] == (
        "LOCAL_REWRITE_WHEN_GATE_DEPENDS_ONLY_ON_LOCAL_F"
    )
    assert findings["GP01-PDE-PERMITTIVITY-BOUNDARY"]["relationship"] == (
        "FORMAL_PERMITTIVITY_EQUIVALENCE_IF_GAMMA_IS_PRESCRIBED"
    )
    assert "not temporal memory" in findings["GP01-HELMHOLTZ-NOT-TEMPORAL"]["exact_statement"]
    assert findings["GP01-PENNER-NEIGHBOR"]["relationship"] == (
        "CLOSEST_PUBLISHED_BEHAVIORAL_NEIGHBOR_NO_EXACT_REWRITE_FOUND"
    )
    assert "never score" in findings["GP01-ACTION-QUARANTINE"]["promotion_consequence"]


def test_dynamical_comparator_matrix_is_complete_and_fail_closed() -> None:
    config = load_config(ROOT)
    comparators = _by_id(config["dynamical_comparators"], "comparator_id")
    assert set(comparators) == COMPARATOR_IDS_REQUIRED
    for row in comparators.values():
        assert row["source_ids"]
        assert row["exact_equation_or_definition"]
        assert row["parameters_and_boundaries"]
        assert row["claimed_domain"]
        assert row["closest_equivalence"]
        assert row["never_substitute_name_only"] is True
        for domain in ("sparc", "xcop"):
            status = row[f"{domain}_status"]
            missing = row[f"{domain}_missing_requirements"]
            if status in {"SOURCE_AND_SOLVER_BLOCKED", "SOLVER_BLOCKED"}:
                assert missing
            else:
                assert status == "IMPLEMENTED_PREDECESSOR_REQUIRES_CAMPAIGN_REBIND"
                assert missing == []

    aqual_efe = comparators["CMP-AQUAL-EFE"]
    assert aqual_efe["sparc_status"] == "SOURCE_AND_SOLVER_BLOCKED"
    assert aqual_efe["xcop_status"] == "SOURCE_AND_SOLVER_BLOCKED"
    assert "independent_external_vector_field" in aqual_efe["sparc_missing_requirements"]
    assert "independent_external_vector_field" in aqual_efe["xcop_missing_requirements"]
    assert comparators["CMP-PENNER-2026"]["sparc_status"] == "SOURCE_AND_SOLVER_BLOCKED"
    assert comparators["CMP-REFRACTED-GRAVITY"]["xcop_status"] == "SOLVER_BLOCKED"
    assert comparators["CMP-EMOND"]["xcop_status"] == "SOURCE_AND_SOLVER_BLOCKED"
    assert comparators["CMP-MOG-STVG"]["xcop_status"] == "SOLVER_BLOCKED"
    assert comparators["CMP-MASHHOON-NONLOCAL"]["xcop_status"] == "SOLVER_BLOCKED"
    assert comparators["CMP-GR-EINASTO"]["sparc_status"] == "SOLVER_BLOCKED"
    assert comparators["CMP-GR-EINASTO"]["xcop_status"] == (
        "IMPLEMENTED_PREDECESSOR_REQUIRES_CAMPAIGN_REBIND"
    )


def test_qg01_through_qg13_and_all_light_gravity_analogies_are_explicit() -> None:
    config = load_config(ROOT)
    ontologies = config["ontology_prior_art"]
    analogies = config["light_gravity_analogies"]
    assert tuple(row["ontology_id"] for row in ontologies) == QG_IDS_REQUIRED
    assert tuple(row["analogy_id"] for row in analogies) == ANALOGY_IDS_REQUIRED
    for row in ontologies[1:]:
        assert row["sparc_status"] == "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR"
        assert row["xcop_status"] == "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR"
        assert row["radial_substitution_rule"]
    for row in analogies:
        assert row["executable_status"] == (
            "ANALOGY_ONLY_SOURCE_BLOCKED_UNTIL_TYPED_EQUATIONS_SOLVER_AND_BOUNDARIES"
        )
        assert row["operational_seed"]
        assert row["required_caution"]


def test_implementation_bindings_match_local_files_without_opening_receipts() -> None:
    config = load_config(ROOT)
    for binding in config["implementation_bindings"]:
        for kind in ("config", "module"):
            path = ROOT / binding[f"{kind}_path"]
            import hashlib

            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding[f"{kind}_sha256"]
        assert binding["campaign_rebind_required"] is True
        assert binding["response_receipt_read_forbidden"] is True
        assert "receipt" not in binding["config_path"].lower()
        assert "receipt" not in binding["module_path"].lower()


def _mutate_section(config: dict, section: str) -> None:
    if section == "identity":
        config[section]["status"] += "_MUTATED"
    elif section == "purpose":
        config[section]["statement"] += " mutated"
    elif section == "governance_bindings":
        config[section]["open_goal"]["sha256"] = "0" * 64
    elif section == "access_contract":
        config[section]["authoring_primary_source_web_tool_invocations"] += 1
    elif section == "primary_sources":
        config[section][0]["operational_anchor"] += " mutated"
    elif section == "gp01_novelty_boundary":
        config[section]["frozen_findings"][0]["exact_statement"] += " mutated"
    elif section in {"dynamical_comparators", "ontology_prior_art"}:
        config[section][0]["closest_equivalence"] += " mutated"
    elif section == "light_gravity_analogies":
        config[section][0]["required_caution"] += " mutated"
    elif section == "implementation_bindings":
        config[section][0]["module_sha256"] = "0" * 64
    elif section == "controlled_vocabularies":
        config[section]["source_block_rule"] += " mutated"
    elif section == "claim_boundary":
        config[section]["required_next_step"] += " mutated"
    elif section == "output_path":
        config[section] += ".mutated"
    else:  # pragma: no cover - the parametrization is the frozen inventory
        raise AssertionError(section)


@pytest.mark.parametrize("section", tuple(EXPECTED_SECTION_SEALS))
def test_every_nested_semantic_section_mutation_fails_closed(section: str) -> None:
    config = copy.deepcopy(load_config(ROOT))
    _mutate_section(config, section)
    with pytest.raises(OpenGravityPriorArtError, match="sealed section changed"):
        validate_config(config)


def test_mutating_a_seal_fails_closed() -> None:
    config = copy.deepcopy(load_config(ROOT))
    config["section_seals"]["primary_sources"] = "0" * 64
    with pytest.raises(OpenGravityPriorArtError, match="sealed section changed"):
        validate_config(config)


def test_coordinated_content_and_config_seal_mutation_still_fails_hardcoded_seal() -> None:
    config = copy.deepcopy(load_config(ROOT))
    config["primary_sources"][0]["operational_anchor"] += " coordinated mutation"
    config["section_seals"]["primary_sources"] = content_sha256(config["primary_sources"])
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    config["section_seals"]["unsealed_root_sha256"] = content_sha256(unsealed)
    with pytest.raises(OpenGravityPriorArtError, match="sealed section changed"):
        validate_config(config)


def test_receipt_rebuild_reads_only_exact_metadata_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT)
    allowed = {
        (ROOT / CONFIG_PATH).resolve(),
        (ROOT / MODULE_PATH).resolve(),
        (ROOT / TEST_PATH).resolve(),
    }
    for binding in config["governance_bindings"].values():
        path = Path(binding["path"])
        allowed.add((path if path.is_absolute() else ROOT / path).resolve())
    for binding in config["implementation_bindings"]:
        allowed.add((ROOT / binding["config_path"]).resolve())
        allowed.add((ROOT / binding["module_path"]).resolve())

    original = Path.read_bytes
    opened: list[Path] = []

    def traced(path: Path) -> bytes:
        opened.append(path.resolve())
        assert path.resolve() in allowed
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", traced)
    receipt = build_receipt(ROOT)
    assert set(opened) == allowed
    assert receipt["access_audit"]["response_bearing_receipts_opened"] == 0
    assert all(
        "receipt" not in row["path"].lower()
        for row in receipt["access_audit"]["allowlisted_metadata_files_opened"]
    )
    for field in ZERO_ACCESS_FIELDS:
        assert receipt["access_audit"][field] == 0
    assert receipt["access_audit"]["network_calls_during_receipt_rebuild"] == 0
    assert receipt["access_audit"]["model_calls_during_receipt_rebuild"] == 0
    assert receipt["access_audit"]["paid_calls_during_receipt_rebuild"] == 0


def test_receipt_is_deterministic_and_self_hashed() -> None:
    first = build_receipt(ROOT)
    second = build_receipt(ROOT)
    assert first == second
    assert first["receipt_content_sha256"] == receipt_content_sha256(first)
    assert first["config_canonical_sha256"] == EXPECTED_CONFIG_CANONICAL_SHA256
    assert first["comparator_inventory"]["family_name_is_never_an_executable_substitute"]
    assert first["gp01_novelty_boundary"]["human_review_required_before_novelty_language"]


def test_atomic_writer_is_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "receipt.json"
    first = b'{"first":true}\n'
    _atomic_no_clobber(target, first)
    assert target.read_bytes() == first
    with pytest.raises(OpenGravityPriorArtError, match="refusing to overwrite"):
        _atomic_no_clobber(target, b'{"second":true}\n')
    assert target.read_bytes() == first


def test_stored_receipt_is_exact_deterministic_rebuild() -> None:
    stored = check_receipt(ROOT, ROOT / OUTPUT_PATH)
    assert stored == build_receipt(ROOT)
    assert json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8")) == stored
