from __future__ import annotations

import copy
import io
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.blind_planetary_law_rediscovery_campaign import (
    CANDIDATE_SELECTION_RULE,
    CONFIG_PATH,
    FORBIDDEN_VOCABULARY,
    OUTPUT_PATH,
    PAIR_PREDICTOR_EXPONENT_BOUND,
    PAIR_RESPONSE_EXPONENT_BOUND,
    STATIC_CLAIMS,
    TARGETS_PATH,
    TRIPLE_PREDICTOR_EXPONENT_BOUND,
    TRIPLE_RESPONSE_EXPONENT_BOUND,
    VIEW_FAMILIES,
    WORLD_IDS,
    BlindPlanetaryLawError,
    _compare_candidate,
    _run_world_phase_a,
    _SealedTargetsGuard,
    _unseal_targets,
    build_artifacts,
    config_vocabulary_violations,
    validate_artifacts,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))

EXPECTED_VERDICTS = {
    "alternative_exclusion_reference": "REDISCOVERED_EXACT",
    "einstein_perihelion_advance": "REDISCOVERED_EXACT",
    "kepler_harmonic_law": "REDISCOVERED_EXACT",
    "newton_inverse_square_law": "REDISCOVERED_EXACT",
}
EXPECTED_LABELS = {
    "alternative_exclusion_reference": "A",
    "einstein_perihelion_advance": "G",
    "kepler_harmonic_law": "K",
    "newton_inverse_square_law": "N",
}
EXPECTED_INVARIANTS = {
    "alternative_exclusion_reference": "x_response*x1^2",
    "einstein_perihelion_advance": "x_response*x1*(1 - x2^2)",
    "kepler_harmonic_law": "x_response^2/x1^3",
    "newton_inverse_square_law": "x_response*x1^2",
}
EXPECTED_EXPONENTS = {
    "alternative_exclusion_reference": "-2",
    "einstein_perihelion_advance": None,
    "kepler_harmonic_law": "3/2",
    "newton_inverse_square_law": "-2",
}
EXPECTED_ADMITTED_VIEWS = {
    "alternative_exclusion_reference": ["u=-2;v=1"],
    "einstein_perihelion_advance": ["v=1;i=1;j=1;k=0", "v=2;i=2;j=2;k=0"],
    "kepler_harmonic_law": ["u=3;v=2"],
    "newton_inverse_square_law": ["u=-2;v=1"],
}
DECLARED_SEARCH_SPACE = {"power_pair": 26, "power_triple": 250}
TOTAL_DECLARED_VIEWS = 26 + 26 + 250 + 26


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return build_artifacts(ROOT)


@pytest.fixture(scope="module")
def checked(artifacts: dict) -> dict:
    validate_artifacts(artifacts["campaign"], artifacts["worlds"], root=ROOT)
    return artifacts["campaign"]


def _by_id(campaign: dict) -> dict[str, dict]:
    return {row["classical_id"]: row for row in campaign["world_results"]}


def _pair_candidate(u: int, v: int, constant: Fraction) -> dict:
    return {
        "constant": {"numerator": constant.numerator, "denominator": constant.denominator},
        "exponent": str(Fraction(u, v)),
        "kind": "power_law",
        "sympy_expression": (
            f"(Rational({constant.numerator}, {constant.denominator}))**Rational(1, {v})"
            f"*x1**Rational({u}, {v})"
        ),
        "view_exponents": {"u": u, "v": v},
        "view_family": "power_pair",
    }


def _pair_target(u: int, v: int, constant: Fraction, expression: str) -> dict:
    return {
        "expression": expression,
        "kind": "power_law",
        "parameters": {
            "constant": f"{constant.numerator}/{constant.denominator}",
            "exponent": str(Fraction(u, v)),
            "view_exponents": {"u": u, "v": v},
            "view_family": "power_pair",
        },
    }


# ---------------------------------------------------------------------------
# The verdict table
# ---------------------------------------------------------------------------


def test_every_world_rediscovered_with_measured_verdict_table(checked: dict) -> None:
    rows = _by_id(checked)
    assert set(rows) == set(EXPECTED_VERDICTS)
    assert {key: row["verdict"] for key, row in rows.items()} == EXPECTED_VERDICTS
    assert {key: row["world_label"] for key, row in rows.items()} == EXPECTED_LABELS
    assert checked["decision"] == "PASS"
    assert checked["counts"]["rediscovered_exact"] == 4
    assert checked["counts"]["missed"] == 0
    assert checked["counts"]["partial"] == 0
    for row in rows.values():
        assert row["comparison_detail"]["structure_match"] is True
        assert row["comparison_detail"]["constant_match"] is True
        assert row["comparison_detail"]["residual"] == "0"


def test_recovered_exponents_and_derived_columns(checked: dict) -> None:
    rows = _by_id(checked)
    assert {key: row["discovered_exponent"] for key, row in rows.items()} == EXPECTED_EXPONENTS
    assert {key: row["discovered_invariant"] for key, row in rows.items()} == EXPECTED_INVARIANTS
    # The exponents are the whole point: 3/2 for the harmonic law, -2 for inverse square.
    assert Fraction(rows["kepler_harmonic_law"]["discovered_exponent"]) == Fraction(3, 2)
    assert Fraction(rows["newton_inverse_square_law"]["discovered_exponent"]) == -2
    # World G had to construct a(1 - e^2) before any law was visible.
    relativity = rows["einstein_perihelion_advance"]
    assert relativity["discovered_invariant"] == "x_response*x1*(1 - x2^2)"
    assert relativity["comparison_detail"]["derived_column_match"] is True


def test_every_world_confirms_on_untouched_holdout(checked: dict) -> None:
    rows = _by_id(checked)
    # Ten public rows; the constant basis consumes one and predicts the other nine.
    assert {row["holdout_confirmations"] for row in rows.values()} == {9}
    assert checked["counts"]["holdout_confirmations_total"] == 36


# ---------------------------------------------------------------------------
# Blindness: vocabulary, chronology, denied probe
# ---------------------------------------------------------------------------


def test_public_config_carries_no_forbidden_vocabulary() -> None:
    text = (ROOT / CONFIG_PATH).read_text(encoding="utf-8")
    assert config_vocabulary_violations(text) == []
    for token in ("kepler", "newton", "einstein", "au", "year", "gravity", "planet"):
        assert token in FORBIDDEN_VOCABULARY
    # The sealed fixture is where the answer lives, so the same grep must fail there.
    sealed = (ROOT / TARGETS_PATH).read_text(encoding="utf-8")
    assert config_vocabulary_violations(sealed) != []


def test_public_config_carries_no_identifying_paths_or_names() -> None:
    assert set(CONFIG) == {
        "data_declaration",
        "policies",
        "schema_version",
        "target_fixture_commitment_sha256",
        "view_families",
        "worlds",
    }
    assert [world["world_id"] for world in CONFIG["worlds"]] == list(WORLD_IDS)
    for world in CONFIG["worlds"]:
        assert set(world["columns"]) <= {"x1", "x2", "x3"}
        assert len(world["rows"]) == 10


def test_forbidden_vocabulary_guard_fails_closed(tmp_path: Path) -> None:
    leaked = copy.deepcopy(CONFIG)
    leaked["data_declaration"]["boundary"] = "these rows are orbital radii in au"
    config_path = tmp_path / "leaked.json"
    config_path.write_text(json.dumps(leaked), encoding="utf-8")
    with pytest.raises(BlindPlanetaryLawError, match="leaked target vocabulary"):
        build_artifacts(ROOT, config_path)


def test_chronology_denied_probe_and_zero_prereads(checked: dict) -> None:
    chronology = checked["chronology"]
    assert chronology["unseal_batches"] == 1
    probe = chronology["denied_probe"]
    assert probe["attempted_target_reads"] == 1
    assert probe["denied_target_reads"] == 1
    assert probe["denied_content_bytes_exposed"] == 0
    assert probe["denied_paths"] == [TARGETS_PATH]
    assert probe["denied_surfaces"] == ["pathlib.Path.open"]
    reads = [event["target_reads"] for event in chronology["events"]]
    assert reads == [0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert checked["counts"]["target_fixture_reads"] == 1
    assert checked["counts"]["target_fixture_reads_denied_before_unseal"] == 1
    assert checked["counts"]["post_unseal_generation_events"] == 0


def test_sealed_guard_denies_every_read_surface_in_process() -> None:
    target = ROOT / TARGETS_PATH
    guard = _SealedTargetsGuard(ROOT)
    with guard:
        with pytest.raises(PermissionError):
            target.read_bytes()
        # Each patched surface is exercised on purpose; the guard raises before any
        # file object exists, so no context manager can apply.
        with pytest.raises(PermissionError):
            open(target, "rb")  # noqa: SIM115
        with pytest.raises(PermissionError):
            io.open(target, "rb")  # noqa: SIM115, UP020
        # Non-target files stay readable while the guard is active.
        assert (ROOT / CONFIG_PATH).read_bytes()
    certificate = guard.certificate()
    assert certificate["attempted_target_reads"] == 3
    assert certificate["denied_target_reads"] == 3
    assert certificate["denied_content_bytes_exposed"] == 0
    assert certificate["denied_surfaces"] == ["pathlib.Path.open", "builtins.open", "io.open"]
    # The guard restores the originals on exit.
    assert target.read_bytes()[:1] == b"{"


def test_unseal_before_phase_a_seal_fails_closed() -> None:
    with pytest.raises(BlindPlanetaryLawError, match="before candidate freeze"):
        _unseal_targets(ROOT, CONFIG, "not-a-sealed-root")


# ---------------------------------------------------------------------------
# Commitments and tamper
# ---------------------------------------------------------------------------


def test_commitment_mismatch_fails_closed(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    digest = drifted["worlds"][0]["sealed_target_sha256"]
    drifted["worlds"][0]["sealed_target_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    config_path = tmp_path / "drifted.json"
    config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BlindPlanetaryLawError, match="commitment did not open"):
        build_artifacts(ROOT, config_path)


def test_fixture_binding_tamper_fails_closed(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    drifted["target_fixture_commitment_sha256"] = "0" * 64
    config_path = tmp_path / "binding.json"
    config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BlindPlanetaryLawError, match="target fixture content changed"):
        build_artifacts(ROOT, config_path)


def test_provenance_commitment_tamper_fails_closed(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    drifted["data_declaration"]["provenance_commitment_sha256"] = "1" * 64
    config_path = tmp_path / "provenance.json"
    config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BlindPlanetaryLawError, match="provenance commitment did not open"):
        build_artifacts(ROOT, config_path)


def test_row_tamper_breaks_the_sealed_generative_rule(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    row = drifted["worlds"][0]["rows"][9]
    row["x2"] = {"numerator": row["x2"]["numerator"] + 1, "denominator": row["x2"]["denominator"]}
    config_path = tmp_path / "rows.json"
    config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BlindPlanetaryLawError, match="do not replay from the sealed rule"):
        build_artifacts(ROOT, config_path)


def test_receipt_tamper_fails_closed(artifacts: dict) -> None:
    campaign = copy.deepcopy(artifacts["campaign"])
    campaign["world_results"][0]["verdict"] = "MISSED"
    with pytest.raises(BlindPlanetaryLawError, match="seal changed"):
        validate_campaign(campaign, root=ROOT)
    resealed = {key: value for key, value in campaign.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(
        {key: value for key, value in campaign.items() if key != "content_sha256"}
    )
    with pytest.raises(BlindPlanetaryLawError, match="exact replay changed"):
        validate_campaign(resealed, root=ROOT)
    worlds = copy.deepcopy(artifacts["worlds"])
    world = next(iter(worlds.values()))
    world["unseal"]["verdict"] = "MISSED"
    with pytest.raises(BlindPlanetaryLawError, match="world receipt seal changed"):
        validate_artifacts(artifacts["campaign"], worlds, root=ROOT)


def test_determinism_exact_replay(artifacts: dict) -> None:
    replayed = build_artifacts(ROOT)
    assert replayed["campaign"] == artifacts["campaign"]
    assert replayed["worlds"] == artifacts["worlds"]


# ---------------------------------------------------------------------------
# The declared derived-view search
# ---------------------------------------------------------------------------


def test_derived_view_search_space_is_bounded_and_pinned(checked: dict) -> None:
    assert len(VIEW_FAMILIES["power_pair"]) == DECLARED_SEARCH_SPACE["power_pair"]
    assert len(VIEW_FAMILIES["power_triple"]) == DECLARED_SEARCH_SPACE["power_triple"]
    assert checked["view_search_space"] == {
        "power_pair_views": DECLARED_SEARCH_SPACE["power_pair"],
        "power_triple_views": DECLARED_SEARCH_SPACE["power_triple"],
        "total_views_evaluated": TOTAL_DECLARED_VIEWS,
    }
    assert checked["counts"]["views_evaluated"] == TOTAL_DECLARED_VIEWS
    families = checked["declared_view_families"]
    assert families["power_pair"]["exponent_range"] == [
        -PAIR_PREDICTOR_EXPONENT_BOUND,
        PAIR_PREDICTOR_EXPONENT_BOUND,
    ]
    assert families["power_pair"]["response_exponent_range"] == [1, PAIR_RESPONSE_EXPONENT_BOUND]
    assert families["power_triple"]["exponent_range"] == [
        -TRIPLE_PREDICTOR_EXPONENT_BOUND,
        TRIPLE_PREDICTOR_EXPONENT_BOUND,
    ]
    assert families["power_triple"]["response_exponent_range"] == [
        1,
        TRIPLE_RESPONSE_EXPONENT_BOUND,
    ]
    assert families["power_triple"]["bases"] == ["x1", "(1 - x2^2)", "x2"]


def test_whole_view_search_is_logged_with_b1_and_b2_per_view(checked: dict) -> None:
    rows = _by_id(checked)
    for classical_id, row in rows.items():
        search = row["derived_view_search"]
        assert search["views_evaluated"] == search["views_declared"]
        assert len(search["log"]) == search["views_evaluated"]
        assert search["views_admitted"] == EXPECTED_ADMITTED_VIEWS[classical_id]
        ranks = [entry["rank"] for entry in search["log"]]
        assert ranks == sorted(ranks) == list(range(len(ranks)))
        for entry in search["log"]:
            assert entry["status"] in {"ADMITTED", "REJECTED", "UNDEFINED"}
            if entry["status"] == "UNDEFINED":
                continue
            assert entry["b1_decision"] in {"PASS", "BLOCK"}
            assert entry["b2_decision"] in {"PASS", "BLOCK"}
            assert len(entry["b1_receipt_sha256"]) == 64
            assert len(entry["b2_receipt_sha256"]) == 64
            if entry["status"] == "ADMITTED":
                assert entry["b1_family_id"] == "constant"
                assert entry["constant"] is not None
    # The relativistic world is the one that needed the derived variable.
    relativity = rows["einstein_perihelion_advance"]["derived_view_search"]
    assert relativity["family"] == "power_triple"
    assert relativity["views_evaluated"] == 250
    winner = next(
        entry for entry in relativity["log"] if entry["view_id"] == "v=1;i=1;j=1;k=0"
    )
    assert winner["exponents"] == {"i": 1, "j": 1, "k": 0, "v": 1}
    assert winner["rank"] < min(
        entry["rank"]
        for entry in relativity["log"]
        if entry["status"] == "ADMITTED" and entry["view_id"] != "v=1;i=1;j=1;k=0"
    )


def test_index_dependent_statements_are_recorded_but_never_selected(checked: dict) -> None:
    assert "never selected" in CANDIDATE_SELECTION_RULE
    assert checked["candidate_selection_rule"] == CANDIDATE_SELECTION_RULE
    for world in CONFIG["worlds"]:
        phase_a = _run_world_phase_a(world)
        raw = phase_a["index_dependent_raw_stage_decisions"]
        # B1 and B7 both refuse the raw index-indexed rows: no law lives in the row order.
        assert raw["b1_basis_synthesis"] == "BLOCK"
        assert raw["b7_structural_repair"] == "BLOCK"
        assert phase_a["candidate"]["source_stage"] == "b4_declared_view_search"


# ---------------------------------------------------------------------------
# Verdict reachability
# ---------------------------------------------------------------------------


def test_rediscovered_exact_verdict_is_reachable() -> None:
    comparison = _compare_candidate(
        _pair_candidate(3, 2, Fraction(1)),
        _pair_target(3, 2, Fraction(1), "x1**Rational(3, 2)"),
    )
    assert comparison["base_verdict"] == "REDISCOVERED_EXACT"
    assert comparison["equivalent"] is True


def test_partial_verdict_is_reachable_on_a_wrong_constant() -> None:
    comparison = _compare_candidate(
        _pair_candidate(-2, 1, Fraction(7, 2)),
        _pair_target(-2, 1, Fraction(1), "x1**Rational(-2, 1)"),
    )
    assert comparison["base_verdict"] == "PARTIAL"
    assert comparison["detail"]["exponent_match"] is True
    assert comparison["detail"]["constant_match"] is False


def test_missed_verdict_is_reachable_on_a_wrong_exponent_and_on_no_candidate() -> None:
    comparison = _compare_candidate(
        _pair_candidate(2, 1, Fraction(1)),
        _pair_target(3, 2, Fraction(1), "x1**Rational(3, 2)"),
    )
    assert comparison["base_verdict"] == "MISSED"
    assert comparison["detail"]["exponent_match"] is False
    empty = _compare_candidate(None, _pair_target(3, 2, Fraction(1), "x1**Rational(3, 2)"))
    assert empty["base_verdict"] == "MISSED"
    assert empty["method"] == "no_candidate"


def test_a_world_outside_the_declared_grammar_is_missed_not_guessed(tmp_path: Path) -> None:
    """A response column with no law in the declared view grammar must score MISSED."""

    world = copy.deepcopy(CONFIG["worlds"][0])
    for offset, row in enumerate(world["rows"]):
        row["x2"] = {"numerator": 7 + offset * offset * offset, "denominator": 11 + offset}
    phase_a = _run_world_phase_a(world)
    assert phase_a["candidate"] is None
    assert phase_a["derived_view_search"]["views_admitted"] == []
    assert phase_a["prover_routes"][0]["status"] == "NOT_APPLICABLE"
    target = json.loads((ROOT / TARGETS_PATH).read_text(encoding="utf-8"))["targets"][0]
    assert _compare_candidate(phase_a["candidate"], target)["base_verdict"] == "MISSED"


# ---------------------------------------------------------------------------
# Provers, generative rule, claims
# ---------------------------------------------------------------------------


def test_lean_is_emitted_for_the_monomial_companion_only(checked: dict) -> None:
    rows = _by_id(checked)
    assert checked["counts"]["lean_sources_emitted"] == 9
    assert checked["counts"]["prover_receipts"] == 9
    for row in rows.values():
        assert row["lean_emitted"] is True
        routes = {route["route"]: route for route in row["prover_trail"]}
        assert routes["b5_lemma_decomposition"]["decision"] == "DECOMPOSED"
        assert routes["b5_lemma_decomposition"]["lean_source_emitted"] is True
    # Only the increasing world gets the monotonicity route.
    increasing = [
        route
        for route in rows["kepler_harmonic_law"]["prover_trail"]
        if route["route"] == "b6_quantified_inequality"
    ]
    assert len(increasing) == 2
    decreasing = [
        route
        for route in rows["newton_inverse_square_law"]["prover_trail"]
        if route["route"] == "b6_quantified_inequality"
    ]
    assert len(decreasing) == 1
    assert all(route["decision"] == "PROVED_LOCALLY" for route in increasing + decreasing)


def test_sealed_generative_rule_replays_every_public_row(checked: dict) -> None:
    verification = checked["generative_rule_verification"]
    assert verification["public_rows_replayed_from_sealed_rule"] is True
    assert verification["quantized_roots_are_correctly_rounded"] is True
    assert verification["declared_constants_recomputed"] is True
    assert verification["rows_verified"] == 40
    fidelity = verification["anchor_fidelity"]
    axis = Fraction(fidelity["max_relative_deviation_semi_major_axis"])
    period = Fraction(fidelity["max_relative_deviation_sidereal_period"])
    # Pure quantization on the axes; the real two-body residual on the periods.
    assert axis < Fraction(1, 10**12)
    assert Fraction(1, 10**4) < period < Fraction(1, 10**3)


def test_claims_stay_inside_the_rediscovery_boundary(checked: dict) -> None:
    claims = checked["claims"]
    assert claims["rediscovery_of_classical_results"] is True
    assert claims["novelty_claimed"] is False
    assert claims["real_observational_data_opened"] is False
    assert claims["data_computed_from_declared_model"] is True
    assert claims["machine_found_laws_unaided"] is True
    assert claims["kernel_verified_lean"] is False
    assert claims["lean_proves_the_recovered_relation_itself"] is False
    assert claims["post_unseal_generation"] is False
    assert claims["target_records_read_before_candidate_freeze"] == 0
    assert {key: claims[key] for key in STATIC_CLAIMS} == STATIC_CLAIMS
    for phrase in ("no observational dataset is opened", "no novelty"):
        assert phrase in checked["scope"]


def test_attribution_is_revealed_only_after_the_unseal(checked: dict) -> None:
    rows = _by_id(checked)
    assert rows["kepler_harmonic_law"]["attribution_year"] == 1619
    assert rows["newton_inverse_square_law"]["attribution_year"] == 1687
    assert rows["einstein_perihelion_advance"]["attribution_year"] == 1915
    for row in rows.values():
        assert row["attribution"]
        assert set(row["column_meanings"]) <= {"x1", "x2", "x3"}
    config_text = (ROOT / CONFIG_PATH).read_text(encoding="utf-8").lower()
    for name in ("kepler", "newton", "einstein", "harmonic", "perihelion"):
        assert name not in config_text
    # The attribution years must not appear as standalone numbers either; digit
    # boundaries keep long exact numerators from producing false positives.
    for year in (1619, 1687, 1915):
        assert re.search(rf"(?<!\d){year}(?!\d)", config_text) is None


def test_written_receipt_matches_the_build(checked: dict) -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert stored == checked
    for row in checked["world_results"]:
        world = json.loads((ROOT / row["world_receipt_path"]).read_text(encoding="utf-8"))
        assert world["content_sha256"] == row["world_receipt_sha256"]
        assert world["unseal"]["verdict"] == row["verdict"]
