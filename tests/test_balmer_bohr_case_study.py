"""Gates for the Balmer/Bohr head-to-head case study.

The load-bearing tests here are the ones that could embarrass the claim: the committed
public config is grepped for every forbidden token, the sealed-fixture read guard is
exercised on all three patched surfaces, every commitment is tampered with in turn and
must fail closed, the 2916-view search space is pinned bound by bound, the holdout
predictions are recomputed here from the frozen candidate and checked against values the
blind phase provably could not read, and every symbolic step of the derivation is
re-derived independently rather than compared against a stored transcript.

Both negative controls are re-run here as well, and all three verdicts -- exact, partial
and missed -- are shown to be reachable, so a PASS means the grader could have failed.
"""

from __future__ import annotations

import copy
import io
import json
import re
import shutil
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from sigma_theory_compiler.balmer_bohr_case_study import (
    CITED_CONSTANTS,
    CITED_MEASUREMENT,
    CONFIG_PATH,
    DOC_PATH,
    EXTENDED_FORBIDDEN_VOCABULARY,
    FORBIDDEN_VOCABULARY,
    INDEX_EXPONENT_BOUND,
    OFFSET_BOUND,
    POSTULATES,
    QUADRATIC_EXPONENT_BOUND,
    RECEIPT_PATH,
    RESULT_SCHEMA,
    RUNTIME_PATH,
    SHIFT_BOUND,
    SOURCE_PATH,
    TARGETS_PATH,
    TEST_PATH,
    TOLERANCE_ROBUSTNESS_LADDER,
    VIEWS,
    BalmerBohrCaseStudyError,
    _CaseStudySealedGuard,
    _run_phase_a,
    _score,
    build_case_study,
    build_receipt,
    config_vocabulary_violations,
    main,
    rydberg_expression,
    solve_orbit,
    validate_receipt,
)
from sigma_theory_compiler.blind_planetary_law_rediscovery_campaign import (
    FORBIDDEN_VOCABULARY as PLANETARY_FORBIDDEN_VOCABULARY,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
FIXTURE = json.loads((ROOT / TARGETS_PATH).read_text(encoding="utf-8"))

#: The search space, pinned. Changing any bound changes what the engine could possibly
#: have found, so all five numbers are asserted together.
EXPECTED_SHIFT_BOUND = (0, 3)
EXPECTED_INDEX_EXPONENT_BOUND = (-4, 4)
EXPECTED_QUADRATIC_EXPONENT_BOUND = (-4, 4)
EXPECTED_OFFSET_BOUND = (0, 9)
EXPECTED_TOTAL_VIEWS = 2916

EXPECTED_VIEW_ID = "s=2;i=-2;j=1;c=4"
EXPECTED_VIEW_EXPONENTS = {"c": 4, "i": -2, "j": 1, "s": 2}
EXPECTED_CONSTANT = Fraction(131241251, 36000)
EXPECTED_ROUNDED_CONSTANT = "3645.6"
EXPECTED_HOLDOUT_LABELS = [5, 6, 7]

COPIED_FOR_TMP_ROOT = (CONFIG_PATH, TARGETS_PATH, SOURCE_PATH, TEST_PATH, DOC_PATH)


@pytest.fixture(scope="module")
def built() -> dict:
    return build_receipt(ROOT)


@pytest.fixture(scope="module")
def receipt(built: dict) -> dict:
    validate_receipt(built["receipt"], root=ROOT)
    return built["receipt"]


def _tmp_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in COPIED_FOR_TMP_ROOT:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
    return path


def _flip(digest: str) -> str:
    return ("0" if digest[0] != "0" else "1") + digest[1:]


# ---------------------------------------------------------------------------
# Blinding: the vocabulary guard
# ---------------------------------------------------------------------------


def test_public_config_carries_no_forbidden_vocabulary() -> None:
    text = (ROOT / CONFIG_PATH).read_text(encoding="utf-8")
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    leaked = sorted(tokens & set(FORBIDDEN_VOCABULARY))
    assert leaked == [], leaked
    assert config_vocabulary_violations(text) == []
    for mandated in ("hydrogen", "spectral", "wavelength", "line", "balmer", "atom", "light"):
        assert mandated in FORBIDDEN_VOCABULARY
    assert "angstrom" in FORBIDDEN_VOCABULARY


def test_forbidden_vocabulary_extends_the_planetary_list() -> None:
    assert set(PLANETARY_FORBIDDEN_VOCABULARY) < set(FORBIDDEN_VOCABULARY)
    assert set(EXTENDED_FORBIDDEN_VOCABULARY) <= set(FORBIDDEN_VOCABULARY)
    assert len(FORBIDDEN_VOCABULARY) == len(set(FORBIDDEN_VOCABULARY))
    assert list(FORBIDDEN_VOCABULARY) == sorted(FORBIDDEN_VOCABULARY)


def test_forbidden_vocabulary_guard_fails_closed(tmp_path: Path) -> None:
    leaked = copy.deepcopy(CONFIG)
    leaked["data_declaration"]["boundary"] = "these rows are hydrogen wavelengths"
    path = _write(tmp_path / "leaked.json", leaked)
    with pytest.raises(BalmerBohrCaseStudyError, match="leaked target vocabulary"):
        build_case_study(ROOT, path)


def test_public_config_hides_the_true_ordinal() -> None:
    labels = [row["m"] for row in CONFIG["rows"]["fit_rows"]]
    assert labels == [1, 2, 3, 4]
    assert CONFIG["rows"]["holdout_indices"] == EXPECTED_HOLDOUT_LABELS
    quantum = [row["quantum_index"] for row in FIXTURE["provenance"]["fit_source"]["rows"]]
    assert quantum == [3, 4, 5, 6], "the sealed ordinal must differ from the public label"
    assert "quantum_index" not in canonical_json_bytes(CONFIG).decode("utf-8")


def test_public_config_carries_no_holdout_values() -> None:
    """The engine sees holdout labels only; the values live behind the seal."""

    text = canonical_json_bytes(CONFIG).decode("utf-8")
    for row in FIXTURE["holdout"]["rows"]:
        assert str(row["v"]["numerator"]) not in text
    assert set(CONFIG["rows"]) >= {"holdout_indices", "sealed_holdout_commitment_sha256"}
    assert "holdout_rows" not in CONFIG["rows"]


# ---------------------------------------------------------------------------
# Blinding: the sealed-fixture read guard and the denied probe
# ---------------------------------------------------------------------------


def test_sealed_guard_denies_every_read_surface_in_process() -> None:
    target = ROOT / TARGETS_PATH
    guard = _CaseStudySealedGuard(ROOT)
    with guard:
        with pytest.raises(PermissionError):
            target.read_bytes()
        with pytest.raises(PermissionError):
            open(target, "rb")  # noqa: SIM115
        with pytest.raises(PermissionError):
            io.open(target, "rb")  # noqa: SIM115, UP020
        assert (ROOT / CONFIG_PATH).read_bytes()
    certificate = guard.certificate()
    assert certificate["attempted_target_reads"] == 3
    assert certificate["denied_target_reads"] == 3
    assert certificate["denied_content_bytes_exposed"] == 0
    assert certificate["denied_paths"] == [TARGETS_PATH]
    assert target.read_bytes()[:1] == b"{"


def test_denied_probe_fires_exactly_once_and_chronology_is_clean(receipt: dict) -> None:
    chronology = receipt["chronology"]
    probe = chronology["denied_probe"]
    assert probe["attempted_target_reads"] == 1
    assert probe["denied_target_reads"] == 1
    assert probe["denied_content_bytes_exposed"] == 0
    assert probe["denied_paths"] == [TARGETS_PATH]
    assert probe["denied_surfaces"] == ["pathlib.Path.open"]
    assert chronology["unseal_batches"] == 1
    reads = [event["target_reads"] for event in chronology["events"]]
    assert reads == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    freeze = next(
        event for event in chronology["events"] if event["event"] == "holdout_predictions_frozen"
    )
    unseal = next(
        event for event in chronology["events"] if event["event"] == "atomic_target_unseal"
    )
    assert freeze["sequence"] < unseal["sequence"]
    assert receipt["counts"]["target_fixture_reads"] == 1
    assert receipt["counts"]["target_fixture_reads_denied_before_unseal"] == 1
    assert receipt["counts"]["post_unseal_generation_events"] == 0


def test_phase_a_reproduces_under_an_active_guard(receipt: dict) -> None:
    """The frozen candidate and predictions are computed with the fixture unreadable."""

    guard = _CaseStudySealedGuard(ROOT)
    with guard:
        with pytest.raises(PermissionError):
            (ROOT / TARGETS_PATH).read_bytes()
        blind = _run_phase_a(CONFIG)
    assert blind["candidate"] == receipt["blind_race"]["candidate"]
    assert blind["holdout_predictions"] == receipt["blind_race"]["holdout_predictions"]
    assert blind["search_space"] == receipt["blind_race"]["search_space"]
    assert guard.certificate()["denied_target_reads"] == 1


def test_unseal_before_the_freeze_fails_closed() -> None:
    from sigma_theory_compiler.balmer_bohr_case_study import _unseal

    with pytest.raises(BalmerBohrCaseStudyError, match="before candidate freeze"):
        _unseal(ROOT, CONFIG, "not-a-sealed-root")


# ---------------------------------------------------------------------------
# Commitments fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("rows", "sealed_target_sha256"), "target commitment did not open"),
        (("rows", "sealed_holdout_commitment_sha256"), "holdout commitment did not open"),
        (
            ("data_declaration", "provenance_commitment_sha256"),
            "provenance commitment did not open",
        ),
        (("target_fixture_commitment_sha256",), "target fixture content changed"),
    ],
)
def test_commitment_mismatch_fails_closed(
    tmp_path: Path, path: tuple[str, ...], message: str
) -> None:
    drifted = copy.deepcopy(CONFIG)
    holder = drifted
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = _flip(holder[path[-1]])
    config_path = _write(tmp_path / "drifted.json", drifted)
    with pytest.raises(BalmerBohrCaseStudyError, match=message):
        build_case_study(ROOT, config_path)


def test_fixture_tamper_fails_closed(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    fixture = copy.deepcopy(FIXTURE)
    fixture["target"]["constant_decimal"] = "3645.7"
    _write(root / TARGETS_PATH, fixture)
    with pytest.raises(BalmerBohrCaseStudyError, match="target fixture content changed"):
        build_case_study(root)


def test_row_tamper_breaks_the_sealed_source_replay(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    drifted["rows"]["fit_rows"][0]["v"]["numerator"] += 1
    config_path = _write(tmp_path / "row.json", drifted)
    with pytest.raises(BalmerBohrCaseStudyError, match="do not replay from the sealed table"):
        build_case_study(ROOT, config_path)


def test_declared_policy_and_grammar_drift_fail_closed(tmp_path: Path) -> None:
    drifted = copy.deepcopy(CONFIG)
    drifted["view_family"]["offset_range"] = [0, 10]
    with pytest.raises(BalmerBohrCaseStudyError, match="declared view grammar changed"):
        build_case_study(ROOT, _write(tmp_path / "grammar.json", drifted))
    drifted = copy.deepcopy(CONFIG)
    drifted["policies"]["target_unseal_batches"] = 2
    with pytest.raises(BalmerBohrCaseStudyError, match="prospective policy changed"):
        build_case_study(ROOT, _write(tmp_path / "policy.json", drifted))
    drifted = copy.deepcopy(CONFIG)
    drifted["data_declaration"]["fit_relative_tolerance"] = "1e-1"
    with pytest.raises(BalmerBohrCaseStudyError, match="prospective envelope"):
        build_case_study(ROOT, _write(tmp_path / "tolerance.json", drifted))


# ---------------------------------------------------------------------------
# The declared search space, pinned
# ---------------------------------------------------------------------------


def test_search_space_bounds_are_pinned(receipt: dict) -> None:
    assert SHIFT_BOUND == EXPECTED_SHIFT_BOUND
    assert INDEX_EXPONENT_BOUND == EXPECTED_INDEX_EXPONENT_BOUND
    assert QUADRATIC_EXPONENT_BOUND == EXPECTED_QUADRATIC_EXPONENT_BOUND
    assert OFFSET_BOUND == EXPECTED_OFFSET_BOUND
    assert len(VIEWS) == EXPECTED_TOTAL_VIEWS

    shifts = EXPECTED_SHIFT_BOUND[1] - EXPECTED_SHIFT_BOUND[0] + 1
    indices = EXPECTED_INDEX_EXPONENT_BOUND[1] - EXPECTED_INDEX_EXPONENT_BOUND[0] + 1
    quadratics = EXPECTED_QUADRATIC_EXPONENT_BOUND[1] - EXPECTED_QUADRATIC_EXPONENT_BOUND[0] + 1
    offsets = EXPECTED_OFFSET_BOUND[1] - EXPECTED_OFFSET_BOUND[0] + 1
    assert shifts * indices * (1 + (quadratics - 1) * offsets) == EXPECTED_TOTAL_VIEWS

    declared = CONFIG["view_family"]
    assert declared["shift_range"] == list(EXPECTED_SHIFT_BOUND)
    assert declared["index_exponent_range"] == list(EXPECTED_INDEX_EXPONENT_BOUND)
    assert declared["quadratic_exponent_range"] == list(EXPECTED_QUADRATIC_EXPONENT_BOUND)
    assert declared["offset_range"] == list(EXPECTED_OFFSET_BOUND)
    assert declared["total_declared_views"] == EXPECTED_TOTAL_VIEWS

    space = receipt["blind_race"]["search_space"]
    assert space["total_declared_views"] == EXPECTED_TOTAL_VIEWS
    assert space["views_evaluated"] + space["views_undefined"] == EXPECTED_TOTAL_VIEWS
    assert space["views_rejected"] + space["views_admitted"] == space["views_evaluated"]
    assert space["views_admitted"] == 1


def test_every_declared_view_appears_in_the_published_log(receipt: dict) -> None:
    log = receipt["blind_race"]["search_log"]
    assert len(log) == EXPECTED_TOTAL_VIEWS
    assert [entry["rank"] for entry in log] == list(range(EXPECTED_TOTAL_VIEWS))
    assert {entry["view_id"] for entry in log} == {view["view_id"] for view in VIEWS}
    space = receipt["blind_race"]["search_space"]
    assert canonical_sha256(log) == space["search_log_sha256"]
    statuses = {entry["status"] for entry in log}
    assert statuses == {"ADMITTED", "REJECTED", "SKIP"}
    admitted = [entry for entry in log if entry["status"] == "ADMITTED"]
    assert [entry["view_id"] for entry in admitted] == [EXPECTED_VIEW_ID]
    reasons = space["rejection_reasons"]
    assert sum(1 for entry in log if entry.get("reason") == "not_constant") == (
        reasons["not_constant"]
    )
    assert sum(1 for entry in log if entry.get("reason") == "undefined") == reasons["undefined"]
    for key in reasons:
        assert key in space["rejection_reason_legend"]


def test_exact_arithmetic_refuses_every_view_including_the_winner(receipt: dict) -> None:
    log = receipt["blind_race"]["search_log"]
    decisions = {entry["b1_decision"] for entry in log if "b1_decision" in entry}
    assert decisions == {"BLOCK"}
    winner = receipt["blind_race"]["b1_on_the_winning_column"]
    assert winner["decision"] == "BLOCK"
    assert winner["first_blocker"]
    assert "tolerance-aware" in winner["note"]


# ---------------------------------------------------------------------------
# The recovered relation
# ---------------------------------------------------------------------------


def test_engine_recovered_the_relation_and_the_offset(receipt: dict) -> None:
    candidate = receipt["blind_race"]["candidate"]
    assert candidate["view_id"] == EXPECTED_VIEW_ID
    assert candidate["view_exponents"] == EXPECTED_VIEW_EXPONENTS
    assert candidate["recovered_index_offset"] == 2
    assert Fraction(**candidate["constant"]) == EXPECTED_CONSTANT
    assert candidate["rejected_earlier_views"] == 419
    unseal = receipt["blind_race"]["unseal"]
    assert unseal["verdict"] == "REDISCOVERED_EXACT"
    assert unseal["structure_match"] is True
    assert unseal["constant_match"] is True
    assert unseal["shapes_agree"] is True
    assert unseal["constant_rounded_to_published_places"] == EXPECTED_ROUNDED_CONSTANT
    assert unseal["sealed_constant_decimal"] == EXPECTED_ROUNDED_CONSTANT
    assert receipt["verdict"] == "REDISCOVERED_EXACT"
    assert receipt["decision"] == "PASS"
    assert receipt["claims"]["machine_found_the_relation_unaided"] is True


def test_receipt_carries_recomputed_latex_for_every_rendered_relation(receipt: dict) -> None:
    """The site renders MathML from these strings, so they are re-derived here."""

    m, upper, scale = (
        sp.Symbol("m", positive=True),
        sp.Symbol("M", positive=True),
        sp.Symbol("B", positive=True),
    )
    candidate = receipt["blind_race"]["candidate"]
    assert candidate["latex"] == sp.latex(
        sp.Eq(sp.Symbol("v", positive=True), scale * (m + 2) ** 2 * ((m + 2) ** 2 - 4) ** -1)
    )
    unseal = receipt["blind_race"]["unseal"]
    assert unseal["classical_latex"] == sp.latex(
        sp.Eq(sp.Symbol("lambda"), scale * upper**2 * (upper**2 - 4) ** -1)
    )
    assert unseal["discovered_latex"] == candidate["latex"]
    assert unseal["classical_upper_index_note"].startswith("M = m + 2")
    for step in receipt["derivation"]["steps"]:
        assert step["latex"].strip()
    assert receipt["derivation"]["loop_closure"]["constant_identity_latex"] == sp.latex(
        sp.Eq(scale, 4 / sp.Symbol("R", positive=True))
    )


def test_constant_is_the_exact_mean_of_the_recomputed_column(receipt: dict) -> None:
    """Re-derive the invariant here, from the committed config, with no engine help."""

    column = []
    for row in CONFIG["rows"]["fit_rows"]:
        upper = row["m"] + 2
        value = Fraction(row["v"]["numerator"], row["v"]["denominator"])
        column.append(value * Fraction(upper**2 - 4, upper**2))
    mean = sum(column, Fraction(0)) / len(column)
    assert mean == EXPECTED_CONSTANT
    spread = max(abs(entry - mean) for entry in column) / mean
    assert spread < Fraction(1, 10**4)
    assert Fraction(**receipt["blind_race"]["candidate"]["constant"]) == mean


def test_tolerance_ladder_shows_the_verdict_does_not_depend_on_the_tolerance(
    receipt: dict,
) -> None:
    ladder = receipt["blind_race"]["tolerance_robustness"]
    assert [rung["relative_tolerance"] for rung in ladder] == list(TOLERANCE_ROBUSTNESS_LADDER)
    declared = [rung for rung in ladder if rung["is_the_declared_tolerance"]]
    assert len(declared) == 1
    assert declared[0]["relative_tolerance"] == CONFIG["data_declaration"][
        "fit_relative_tolerance"
    ]
    by_tolerance = {rung["relative_tolerance"]: rung for rung in ladder}
    for tight in ("1e-6", "1e-5"):
        assert by_tolerance[tight]["views_admitted"] == 0
    for stable in ("1e-4", "3e-4", "1e-3", "1e-2"):
        assert by_tolerance[stable]["admitted_view_ids"] == [EXPECTED_VIEW_ID]
    assert by_tolerance["1e-1"]["views_admitted"] > 1
    assert EXPECTED_VIEW_ID in by_tolerance["1e-1"]["admitted_view_ids"]


# ---------------------------------------------------------------------------
# Holdout: predicted before it could be read, verified after
# ---------------------------------------------------------------------------


def test_holdout_predictions_match_the_sealed_rows(receipt: dict) -> None:
    predictions = receipt["blind_race"]["holdout_predictions"]
    assert [row["m"] for row in predictions] == EXPECTED_HOLDOUT_LABELS
    constant = Fraction(**receipt["blind_race"]["candidate"]["constant"])
    sealed = {row["index"]: Fraction(**row["v"]) for row in FIXTURE["holdout"]["rows"]}
    tolerance = Fraction(3, 10**4)
    scored = {row["m"]: row for row in receipt["blind_race"]["unseal"]["holdout"]}
    assert set(scored) == set(sealed)
    for prediction in predictions:
        upper = prediction["m"] + 2
        assert prediction["shifted_index"] == upper
        expected = constant * Fraction(upper**2, upper**2 - 4)
        assert Fraction(**prediction["predicted_value"]) == expected
        measured = sealed[prediction["m"]]
        relative = abs(expected - measured) / measured
        assert relative <= tolerance
        assert scored[prediction["m"]]["within_declared_tolerance"] is True
        assert scored[prediction["m"]]["relative_residual"].startswith("0.00010")
    assert receipt["blind_race"]["unseal"]["holdout_within_tolerance"] is True
    assert receipt["counts"]["holdout_rows_within_tolerance"] == 3
    assert receipt["counts"]["holdout_rows_predicted"] == 3


def test_sealed_rule_replays_on_every_row(receipt: dict) -> None:
    replay = receipt["blind_race"]["sealed_rule_replay"]
    assert replay["rows_replayed"] == 7
    assert [row["shifted_index"] for row in replay["rows"]] == [3, 4, 5, 6, 7, 8, 9]
    assert [row["source"] for row in replay["rows"]] == ["fit"] * 4 + ["holdout"] * 3
    constant = Fraction(FIXTURE["target"]["constant"])
    assert constant == Fraction(36456, 10)
    for row in replay["rows"]:
        upper = row["shifted_index"]
        value = constant * Fraction(upper**2, upper**2 - 4)
        assert Fraction(row["sealed_rule_decimal"]) == Fraction(round(value * 10**6), 10**6)
    assert Fraction(replay["max_relative_residual"]) < Fraction(2, 10**4)


# ---------------------------------------------------------------------------
# Verdicts are reachable in both directions
# ---------------------------------------------------------------------------


def _phase_a_stub(candidate: dict, predictions: list[dict]) -> dict:
    return {"candidate": candidate, "holdout_predictions": predictions}


def test_all_three_verdicts_are_reachable(receipt: dict) -> None:
    blind = receipt["blind_race"]
    target = FIXTURE["target"]
    exact = _score(
        _phase_a_stub(blind["candidate"], blind["holdout_predictions"]), CONFIG, target
    )
    assert exact["verdict"] == "REDISCOVERED_EXACT"

    off_constant = copy.deepcopy(blind["candidate"])
    off_constant["constant"] = {"numerator": 3700, "denominator": 1}
    off_constant["sympy_expression"] = "Rational(3700, 1)*(m + 2)**2*((m + 2)**2 - 4)**-1"
    partial = _score(
        _phase_a_stub(off_constant, blind["holdout_predictions"]), CONFIG, target
    )
    assert partial["verdict"] == "PARTIAL"
    assert partial["structure_match"] is True
    assert partial["constant_match"] is False

    off_shape = copy.deepcopy(blind["candidate"])
    off_shape["view_exponents"] = {"c": 1, "i": -2, "j": 1, "s": 2}
    off_shape["sympy_expression"] = "Rational(1, 1)*(m + 2)**2*((m + 2)**2 - 1)**-1"
    missed = _score(_phase_a_stub(off_shape, blind["holdout_predictions"]), CONFIG, target)
    assert missed["verdict"] == "MISSED"

    none_at_all = _score(_phase_a_stub(None, []), CONFIG, target)
    assert none_at_all["verdict"] == "MISSED"
    assert none_at_all["method"] == "no_candidate"


def test_a_row_outside_the_declared_grammar_is_missed_not_guessed(tmp_path: Path) -> None:
    """Rows the grammar cannot express must produce no candidate, never a near miss."""

    from sigma_theory_compiler.balmer_bohr_case_study import _run_view_search

    pairs = [(index, Fraction(2) ** index) for index in range(1, 5)]
    search = _run_view_search(pairs, Fraction(1, 10**4))
    assert search["admitted"] == []
    assert len(search["log"]) == EXPECTED_TOTAL_VIEWS


# ---------------------------------------------------------------------------
# The derivation, re-derived here rather than compared to a transcript
# ---------------------------------------------------------------------------


def test_every_derivation_step_is_recomputed(receipt: dict) -> None:
    n, hbar, m_e, k, e, h, c = sp.symbols("n hbar m_e k e h c", positive=True)
    n_1, n_2 = sp.symbols("n_1 n_2", positive=True)
    v, r = sp.symbols("v r", positive=True)

    solutions = sp.solve(
        [sp.Eq(m_e * v**2 / r, k * e**2 / r**2), sp.Eq(m_e * v * r, n * hbar)],
        [v, r],
        dict=True,
    )
    assert len(solutions) == 1
    radius = sp.simplify(solutions[0][r])
    speed = sp.simplify(solutions[0][v])
    assert sp.simplify(radius - n**2 * hbar**2 / (m_e * k * e**2)) == 0

    energy = sp.simplify(m_e * speed**2 / 2 - k * e**2 / radius)
    assert sp.simplify(energy + m_e * k**2 * e**4 / (2 * hbar**2 * n**2)) == 0

    difference = sp.simplify(energy.subs(n, n_2) - energy.subs(n, n_1))
    inverse = sp.simplify(
        sp.expand(sp.simplify(difference / (h * c)).subs(hbar, h / (2 * sp.pi)))
    )
    rydberg = sp.simplify(inverse / (1 / n_1**2 - 1 / n_2**2))
    assert sp.simplify(rydberg - 2 * sp.pi**2 * m_e * e**4 * k**2 / (h**3 * c)) == 0

    steps = receipt["derivation"]["steps"]
    assert [step["step"] for step in steps] == [1, 2, 3, 4]
    assert all(step["check_status"] == "pass" for step in steps)
    assert str(radius) in steps[0]["symbolic_result"]
    assert str(energy) in steps[1]["symbolic_result"]
    assert str(inverse) in steps[2]["symbolic_result"]
    assert receipt["derivation"]["symbolic_rydberg"] == str(rydberg)
    positives = {symbol.name: symbol for symbol in (n, hbar, m_e, k, e, h, c, n_1, n_2)}
    reparsed = sp.sympify(receipt["derivation"]["symbolic_rydberg"], locals=positives)
    assert sp.simplify(reparsed - rydberg) == 0
    assert len(receipt["derivation"]["postulates"]) == len(POSTULATES) == 2


def test_rydberg_matches_the_measured_constant(receipt: dict) -> None:
    mp.mp.dps = 60
    electron = mp.mpf(CITED_CONSTANTS["electron_mass_kg"]["value"])
    charge = mp.mpf(CITED_CONSTANTS["elementary_charge_C"]["value"])
    planck = mp.mpf(CITED_CONSTANTS["planck_constant_J_s"]["value"])
    light = mp.mpf(CITED_CONSTANTS["speed_of_light_m_s"]["value"])
    permittivity = mp.mpf(CITED_CONSTANTS["vacuum_electric_permittivity_F_m"]["value"])
    coulomb = 1 / (4 * mp.pi * permittivity)
    derived = 2 * mp.pi**2 * electron * charge**4 * coulomb**2 / (planck**3 * light)

    numerics = receipt["derivation"]["rydberg_numerics"]
    assert mp.almosteq(mp.mpf(numerics["derived_rydberg_per_m"]), derived, rel_eps=mp.mpf("1e-13"))
    measured = mp.mpf(CITED_MEASUREMENT["rydberg_constant_per_m"]["value"])
    relative = abs(derived - measured) / measured
    assert relative < mp.mpf("1e-10")
    assert mp.mpf(numerics["relative_error_vs_measured"]) < mp.mpf("1e-10")
    assert numerics["measured_rydberg_per_m"] == "1.0973731568160e7"
    assert numerics["quoted_rydberg_per_m"] == "1.0973731568e7"


def test_balmer_constant_is_four_over_the_rydberg_constant(receipt: dict) -> None:
    loop = receipt["derivation"]["loop_closure"]
    assert loop["the_identity_holds"] is True
    assert loop["residual_of_the_identity"] == "0"
    assert loop["constant_identity"].startswith("B = 4/R")
    mp.mp.dps = 60
    derived = mp.mpf(receipt["derivation"]["rydberg_numerics"]["derived_rydberg_per_m"])
    constant = 4 / derived * mp.mpf("1e10")
    assert mp.almosteq(
        mp.mpf(loop["constant_from_rydberg_1e-10_m"]), constant, rel_eps=mp.mpf("1e-12")
    )
    assert loop["published_balmer_constant_1e-10_m"] == "3645.6"
    assert mp.mpf(loop["relative_gap_vacuum_infinite_mass"]) < mp.mpf("2e-4")
    assert mp.mpf(loop["relative_gap_after_both_corrections"]) < mp.mpf("2e-4")
    assert [correction["id"] for correction in loop["corrections"]] == [
        "finite_nuclear_mass",
        "air_versus_vacuum_frame",
    ]


def test_negative_controls_fire(receipt: dict) -> None:
    n = sp.Symbol("n", positive=True)
    hbar, m_e, k, e = sp.symbols("hbar m_e k e", positive=True)
    correct = solve_orbit(1)["radius"]
    broken = solve_orbit(2)["radius"]
    assert sp.simplify(correct - broken) != 0
    assert sp.degree(sp.simplify(correct * m_e * k * e**2 / hbar**2), n) == 2
    assert sp.degree(sp.simplify(broken * m_e * k * e**2 / hbar**2), n) == 4

    broken_rydberg = rydberg_expression(hbar_power=3)
    assert sp.simplify(broken_rydberg - rydberg_expression(2)) != 0

    controls = {control["id"]: control for control in receipt["derivation"]["negative_controls"]}
    assert set(controls) == {"wrong_quantization_exponent", "wrong_power_of_hbar"}
    assert all(control["detected"] is True for control in controls.values())
    assert controls["wrong_quantization_exponent"]["observed_power_of_n"] == 4
    assert controls["wrong_quantization_exponent"]["required_power_of_n"] == 2
    mp.mp.dps = 60
    assert mp.mpf(controls["wrong_power_of_hbar"]["relative_error_vs_measured"]) > mp.mpf("1e30")


# ---------------------------------------------------------------------------
# The head-to-head block
# ---------------------------------------------------------------------------


def test_head_to_head_reports_cited_intervals_and_never_estimates_effort(
    receipt: dict,
) -> None:
    head = receipt["head_to_head"]
    assert set(head) == {
        "balmer_1885",
        "bohr_1913",
        "comparison_notes",
        "engine_derivation",
        "engine_empirical",
    }
    for key in ("balmer_1885", "bohr_1913"):
        entry = head[key]
        assert set(entry) >= {
            "human_timescale",
            "inputs_available",
            "method",
            "result",
            "role",
        }
        timescale = entry["human_timescale"]
        assert timescale["personal_effort_duration"] == "not precisely documented"
        assert timescale["interval_is_between_publications_not_a_working_time"] is True
        assert isinstance(timescale["documented_interval_years"], int)
        assert timescale["citation"]
    assert head["balmer_1885"]["human_timescale"]["documented_interval_years"] == 17
    assert head["bohr_1913"]["human_timescale"]["documented_interval_years"] == 28
    engine = head["engine_empirical"]
    assert engine["search_space_size"] == EXPECTED_TOTAL_VIEWS
    assert engine["measured_wall_clock"]["measurement_path"] == RUNTIME_PATH
    assert "not re-measured on replay" in engine["measured_wall_clock"]["boundary"]
    assert len(head["comparison_notes"]) >= 5
    joined = " ".join(head["comparison_notes"])
    assert "handed the grammar" in joined
    assert "not a comparison of difficulty" in joined


def test_claims_stay_inside_the_rediscovery_boundary(receipt: dict) -> None:
    claims = receipt["claims"]
    assert claims["rediscovery_of_classical_results"] is True
    assert claims["novelty_claimed"] is False
    assert claims["historical_timescales_cited_not_estimated"] is True
    assert claims["blinding_enforced_by_runtime_guard"] is True
    assert claims["real_observational_data_opened"] is False
    assert claims["post_unseal_generation"] is False
    assert claims["target_records_read_before_candidate_freeze"] == 0
    assert claims["grammar_and_postulates_are_human_declarations"] is True
    assert "no novelty" in receipt["scope"] or "novelty" in receipt["scope"]


def test_attribution_is_revealed_only_after_the_unseal(receipt: dict) -> None:
    unseal = receipt["blind_race"]["unseal"]
    assert unseal["attribution"].startswith("J. J. Balmer")
    assert unseal["attribution_year"] == 1885
    assert unseal["classical_id"] == "balmer_formula"
    blind_only = {
        key: value
        for key, value in receipt["blind_race"].items()
        if key not in {"unseal", "sealed_rule_replay", "sealed_holdout_rows"}
    }
    text = canonical_json_bytes(blind_only).decode("utf-8").lower()
    for word in ("balmer", "hydrogen", "angstrom", "attribution", "classical_id"):
        assert word not in text, word
    assert config_vocabulary_violations(text) == []


# ---------------------------------------------------------------------------
# Determinism, tamper, and the written receipt
# ---------------------------------------------------------------------------


def test_receipt_is_deterministic(receipt: dict) -> None:
    assert build_case_study(ROOT) == receipt
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert receipt["schema_version"] == RESULT_SCHEMA


def test_receipt_tamper_fails_closed(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["views_admitted"] = 2
    with pytest.raises(BalmerBohrCaseStudyError, match="seal changed"):
        validate_receipt(tampered, root=ROOT)
    resealed = {key: value for key, value in tampered.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(resealed)
    with pytest.raises(BalmerBohrCaseStudyError, match="exact replay changed"):
        validate_receipt(resealed, root=ROOT)


def test_written_receipt_matches_the_build(receipt: dict) -> None:
    written = json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    assert written == receipt
    validate_receipt(written, root=ROOT)


def test_runtime_measurement_is_outside_every_sealed_hash(receipt: dict) -> None:
    runtime = json.loads((ROOT / RUNTIME_PATH).read_text(encoding="utf-8"))
    assert runtime["receipt_path"] == RECEIPT_PATH
    assert set(runtime["measured_seconds"]) == {"blind_race", "derivation", "total"}
    for value in runtime["measured_seconds"].values():
        assert float(value) >= 0
    text = canonical_json_bytes(receipt).decode("utf-8")
    for value in runtime["measured_seconds"].values():
        assert value not in text
    assert "not a benchmark" in runtime["boundary"]


def test_no_float_reaches_the_receipt(receipt: dict) -> None:
    def walk(value: object, path: str = "$") -> None:
        assert not isinstance(value, float), path
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(receipt)


def test_source_bindings_cover_the_committed_files(receipt: dict) -> None:
    bindings = receipt["source_bindings"]
    assert set(bindings) == {"config", "doc", "source", "target_fixture", "test"}
    for binding in bindings.values():
        assert (ROOT / binding["path"]).is_file()
        assert len(binding["file_sha256"]) == 64
    assert bindings["target_fixture"]["path"] == TARGETS_PATH


def test_cli_builds_and_validates(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    assert main(["--root", str(root)]) == 0
    receipt_file = root / RECEIPT_PATH
    runtime_file = root / RUNTIME_PATH
    assert receipt_file.is_file()
    assert runtime_file.is_file()
    stamp = runtime_file.read_bytes()
    assert main(["--root", str(root), "--validate-checked"]) == 0
    assert main(["--root", str(root)]) == 0
    assert runtime_file.read_bytes() == stamp, "the one-time measurement must not be rewritten"
