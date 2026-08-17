"""Tests for the continued-fraction prior-art corpus and the adjudication screen.

The load-bearing checks are the ones that could let a wrong answer through:

* seeds must reproduce their *cited* closed forms numerically, so a mis-transcribed
  identity cannot enter the corpus;
* the corpus must be a real provenance forest -- every derived record traceable to a seed
  by declared transformations, and closed under its own equivalence check;
* the screen must recover the 182 already-labelled known formulas (a screen that cannot
  find known things is not fit to report an absence);
* a known formula in disguise -- a corpus record pushed through two transformations -- must
  come back ``KNOWN`` with an exhibited chain;
* value equality alone must never produce ``KNOWN``.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

from sigma_theory_compiler import cf_prior_art_corpus as corpus_module
from sigma_theory_compiler import cf_prior_art_screen as screen_module
from sigma_theory_compiler.cf_prior_art_corpus import (
    CFPattern,
    CorpusError,
    Poly,
    build_seeds,
    certify_seed,
    corpus_manifest,
    declared_equivalence_sequences,
    load_corpus,
    mobius_apply,
    mobius_compose,
    mobius_of,
    normal_form,
    resolve_to_seed,
    seq_constant,
    seq_from_poly,
    transform_contraction,
    transform_equivalence,
    transform_extension,
    transform_tail_shift,
    verify_forest_closure,
)
from sigma_theory_compiler.cf_prior_art_screen import (
    Candidate,
    ScreenError,
    load_candidates,
    run_screen,
    screen_candidate,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

REPO = Path(__file__).resolve().parents[1]
DATABASE = REPO / "runs/math/prior-art/cf-corpus-v1.sqlite"
MANIFEST = REPO / "runs/math/prior-art/cf-corpus-v1-manifest.json"
ENUMERATION = REPO / "runs/math/inverse-symbolic/cf-enumeration-v1.json"
ADJUDICATION = REPO / "runs/math/prior-art/cf-adjudication-v1.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATABASE, MANIFEST)


@pytest.fixture(scope="module")
def enumeration_receipt():
    return json.loads(ENUMERATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def adjudication_receipt():
    return json.loads(ADJUDICATION.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Exact arithmetic and the transformation group
# ---------------------------------------------------------------------------


def test_polynomial_and_rational_arithmetic_is_exact() -> None:
    left = Poly.of(1, 2, 3)
    right = Poly.of(-1, 1)
    assert (left * right).evaluate(5) == left.evaluate(5) * right.evaluate(5)
    assert left.substitute(Fraction(2), Fraction(1)).evaluate(3) == left.evaluate(7)
    ratio = corpus_module.Rat.of(left, right)
    assert ratio.evaluate(4) == Fraction(left.evaluate(4), right.evaluate(4))
    assert ratio.reciprocal().evaluate(4) == 1 / ratio.evaluate(4)


def test_equivalence_normal_form_is_a_class_invariant() -> None:
    """Every declared equivalence must leave ``r_n = b_n/(a_n a_{n-1})`` untouched."""

    base = CFPattern(seq_from_poly(Poly.of(2, 1)), seq_from_poly(Poly.of(0, -1)))
    reference = normal_form(base)
    checked = 0
    for _, sequence in declared_equivalence_sequences():
        try:
            moved, step = transform_equivalence(base, sequence)
        except CorpusError:
            continue
        moved_form = normal_form(moved)
        assert moved_form.key() == reference.key()
        # The normalized value is the class's second coordinate and must also agree.
        with mp.workdps(50):
            base_value = base.evaluate(400)
            moved_value = mobius_apply(step, base_value)
            assert abs(
                corpus_module.to_mpf(moved_form.scale) * moved_value
                - corpus_module.to_mpf(reference.scale) * base_value
            ) < mp.mpf(10) ** -40
        checked += 1
    assert checked >= 20


def test_tail_shift_moves_the_value_by_the_reported_mobius() -> None:
    base = CFPattern(seq_from_poly(Poly.of(1, 1)), seq_from_poly(Poly.of(1, 1)))
    with mp.workdps(50):
        value = base.evaluate(600)
        for levels in (1, 2, 3):
            moved, step = transform_tail_shift(base, levels)
            assert abs(moved.evaluate(600) - mobius_apply(step, value)) < mp.mpf(10) ** -40


def test_contraction_reproduces_the_even_and_odd_convergents() -> None:
    base = CFPattern(seq_from_poly(Poly.of(1, 2)), seq_from_poly(Poly.of(0, 0, 1)))
    numerators, denominators = corpus_module._convergents(base, 24)
    for parity, offset in (("even", 0), ("odd", 1)):
        contracted = transform_contraction(base, parity)
        contracted_numerators, contracted_denominators = corpus_module._convergents(
            contracted, 8
        )
        for index in range(6):
            top = 2 * index + offset
            assert (
                contracted_numerators[index + 1] / contracted_denominators[index + 1]
                == numerators[top + 1] / denominators[top + 1]
            )


def test_extension_round_trips_through_the_even_contraction() -> None:
    """``extend`` is the unit-denominator inverse of ``contract_even`` where it stays in class."""

    seed = next(item for item in build_seeds() if item.seed_id == "erfc_z_1_1")
    assert seed.pattern is not None
    contracted = transform_contraction(seed.pattern, "odd")
    extended = transform_extension(contracted)
    assert transform_contraction(extended, "even").key() == contracted.key()


def test_euler_minding_convergents_equal_the_series_partial_sums() -> None:
    for series in corpus_module._series_seeds():
        corpus_module.certify_series_correspondence(series, terms=14)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_seed_catalogue_is_deterministic_and_uniquely_identified() -> None:
    first = [seed.seed_id for seed in build_seeds()]
    second = [seed.seed_id for seed in build_seeds()]
    assert first == second == sorted(first)
    assert len(set(first)) == len(first)
    assert len(first) >= 200


SPOT_CHECKS = (
    ("euler_e_alternating", lambda: +mp.e),
    ("euler_e_minus_one", lambda: mp.e - 1),
    ("lambert_coth_half", lambda: mp.coth(mp.mpf(1) / 2)),
    ("lambert_tanh_1_over_3", lambda: mp.tanh(mp.mpf(1) / 3)),
    ("arctan_1_over_1", lambda: mp.pi / 4),
    ("gauss_log_z_1_1", lambda: mp.log(2)),
    ("simple_cf_sqrt_7", lambda: mp.sqrt(7)),
    ("periodic_surd_a1_b1", lambda: +mp.phi),
    ("apery_zeta3", lambda: 6 / mp.zeta(3)),
    ("cotangent_cf_iC_2_1", lambda: 2 * mp.coth(mp.pi / 2)),
    ("bessel_j_ratio_nu1_1_z2", lambda: mp.besselj(1, 2) / mp.besselj(0, 2)),
)


@pytest.mark.parametrize(("seed_id", "expected"), SPOT_CHECKS)
def test_spot_checked_seeds_reproduce_their_cited_closed_form(seed_id, expected) -> None:
    """Ten-plus hand-picked seeds must reproduce their stated value to 50 digits.

    The continued fraction is evaluated independently of the stored closed form, so a
    mis-encoded identity fails here rather than silently becoming "prior art".
    """

    seed = next(item for item in build_seeds() if item.seed_id == seed_id)
    assert seed.pattern is not None
    with mp.workdps(80):
        target = expected()
        approximation = seed.pattern.evaluate(8000)
        scale = max(mp.mpf(1), abs(target))
        assert abs(approximation - target) / scale < mp.mpf(10) ** -50


def test_every_seed_is_certified_against_its_stored_value() -> None:
    series_index = {
        f"euler_minding_{item.series_id}": item for item in corpus_module._series_seeds()
    }
    modes: dict[str, int] = {}
    for seed in build_seeds():
        report = certify_seed(seed)
        if seed.check_mode == "series_correspondence":
            corpus_module.certify_series_correspondence(series_index[seed.seed_id])
        modes[report["mode"]] = modes.get(report["mode"], 0) + 1
    assert modes["direct"] >= 150
    assert modes["series_correspondence"] >= 10


def test_a_wrong_seed_value_is_rejected() -> None:
    seed = next(item for item in build_seeds() if item.seed_id == "euler_e_alternating")
    broken = corpus_module.Seed(
        seed_id=seed.seed_id,
        family=seed.family,
        pattern=seed.pattern,
        value_expr="e + 1",
        value_fn=lambda: mp.e + 1,
        citation=seed.citation,
        validity_domain=seed.validity_domain,
    )
    with pytest.raises(CorpusError, match="failed certification"):
        certify_seed(broken)


# ---------------------------------------------------------------------------
# The sealed corpus artifact
# ---------------------------------------------------------------------------


def test_corpus_artifact_meets_the_declared_scale(corpus) -> None:
    counts = corpus.manifest["counts"]
    assert counts["records"] == len(corpus.records) >= 10_000
    assert counts["seeds"] >= 200
    assert counts["derived"] == counts["records"] - counts["seeds"]
    assert corpus.manifest["claims"]["corpus_absence_establishes_novelty"] is False
    assert corpus.manifest["claims"]["external_fetch_performed"] is False
    assert corpus.manifest["counts"]["seeds_by_family"]


def test_corpus_is_a_provenance_forest(corpus) -> None:
    closure = verify_forest_closure(list(corpus.records))
    assert closure["records"] == len(corpus.records)
    declared = set(corpus_module.DECLARED_TRANSFORMATIONS)
    for record in corpus.records:
        chain = resolve_to_seed(corpus.by_id, record.record_id)
        assert corpus.by_id[chain[0]].kind == "seed"
        if record.kind != "seed":
            assert dict(record.transform)["transformation"] in declared


def test_every_record_carries_a_resolvable_citation(corpus) -> None:
    for record in corpus.records:
        citation = record.citation
        assert citation.reference
        assert citation.author
        assert citation.confidence in corpus_module.CITATION_CONFIDENCES


def test_corpus_is_closed_under_its_own_equivalence_check(corpus) -> None:
    """Applying a declared equivalence to a corpus record must not leave its class."""

    sample = [
        record
        for record in corpus.records
        if record.kind == "seed" and record.pattern is not None
    ][:40]
    _, sequence = declared_equivalence_sequences()[1]
    checked = 0
    for record in sample:
        assert record.pattern is not None
        try:
            moved, _ = transform_equivalence(record.pattern, sequence)
            key = normal_form(moved).key()
        except CorpusError:
            continue
        if record.normal_form_key is None:
            continue
        assert key == record.normal_form_key
        assert record.record_id in corpus.normal_key_index[key]
        checked += 1
    assert checked >= 20


def test_corpus_rebuild_reproduces_the_sealed_record_stream(corpus) -> None:
    """The whole corpus is a pure function of the catalogue and the declared sets."""

    records, report = corpus_module.build_corpus()
    rebuilt = corpus_manifest(records, report)
    assert rebuilt["records_sha256"] == corpus.manifest["records_sha256"]
    assert rebuilt["counts"]["records"] == corpus.manifest["counts"]["records"]


def test_corpus_load_detects_a_manifest_mismatch(tmp_path) -> None:
    tampered = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tampered["counts"]["records"] += 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CorpusError, match="record count"):
        load_corpus(DATABASE, path)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def test_control_recovery_on_the_182_labelled_known(adjudication_receipt) -> None:
    controls = adjudication_receipt["controls"]
    assert controls["labelled_known_rediscovered"] == 182
    assert controls["passed"] is True
    assert float(controls["recovery_rate"]) >= 0.95
    assert controls["screened_KNOWN_with_resolvable_citation"] == controls["screened_KNOWN"]
    for row in adjudication_receipt["control_summaries"]:
        if row["verdict"] == "KNOWN":
            assert row["citation_reference"]
            assert row["matched_record_id"]


def test_all_thirty_two_candidates_are_classified(adjudication_receipt) -> None:
    candidates = adjudication_receipt["candidates"]
    assert len(candidates) == 32
    assert adjudication_receipt["input"]["labelled_not_in_builtin_table"] == 32
    for item in candidates:
        assert item["verdict"] in screen_module.VERDICTS
        assert item["test_that_fired"]
        if item["verdict"] == "KNOWN":
            assert item["matched_record"]["citation"]["reference"]
            assert item["chain_verified"] is True
        else:
            assert item["why_no_chain"]


def test_every_known_verdict_exhibits_a_chain_to_a_cited_seed(
    corpus, adjudication_receipt
) -> None:
    for item in adjudication_receipt["candidates"]:
        if item["verdict"] != "KNOWN":
            continue
        record = corpus.by_id[item["matched_record"]["record_id"]]
        chain = resolve_to_seed(corpus.by_id, record.record_id)
        assert corpus.by_id[chain[0]].kind == "seed"
        for step in item["transformation_chain"]:
            assert step["transformation"] in corpus_module.DECLARED_TRANSFORMATIONS


def test_each_verdict_class_is_reachable(corpus, enumeration_receipt) -> None:
    candidates = load_candidates(enumeration_receipt)
    verdicts = {
        screen_candidate(corpus, item)["verdict"]
        for item in candidates
        if item.source_label == "NOT_IN_BUILTIN_TABLE"
    }
    assert "KNOWN" in verdicts
    assert "INCONCLUSIVE_VALUE_MATCH" in verdicts

    # A classifier probe: a pattern outside the corpus reported against a constant the
    # corpus never produces must land in NOT_FOUND_IN_CORPUS.
    probe = CFPattern(seq_from_poly(Poly.of(3, 4)), seq_from_poly(Poly.of(0, 1, 3)))
    with mp.workdps(60):
        value = mp.nstr(probe.evaluate(3000), 60, strip_zeros=False)
    unknown = Candidate.from_polynomials(
        candidate_id="probe",
        target="euler_gamma",
        alpha=(3, 4, 0),
        beta=(0, 1, 3),
        wrap=mobius_of(1, 0, 0, 1),
        cf_value=value,
        formula_text="classifier probe",
        source_label="NOT_IN_BUILTIN_TABLE",
    )
    assert screen_candidate(corpus, unknown)["verdict"] == "NOT_FOUND_IN_CORPUS"


def test_planted_known_formula_in_disguise_comes_back_known_with_a_chain(corpus) -> None:
    """Take a cited seed, apply two declared transformations, feed it in as a stranger."""

    seed = corpus.by_id["seed:euler_e_minus_one"]
    assert seed.pattern is not None
    shifted, shift_map = transform_tail_shift(seed.pattern, 1)
    disguised, equivalence_map = transform_equivalence(shifted, seq_constant(2))
    total = mobius_compose(equivalence_map, shift_map)
    wrap = mobius_of(2, 4, 1, 0)
    with mp.workdps(110):
        cf_value = mobius_apply(total, mp.mpf(seed.cf_value))
        assert abs(mobius_apply(wrap, cf_value) - mp.e) < mp.mpf(10) ** -50
        text = mp.nstr(cf_value, 100, strip_zeros=False)
    planted = Candidate(
        candidate_id="planted",
        target="e",
        pattern=disguised,
        wrap=wrap,
        cf_value=text,
        formula_text="planted disguise of Euler's e = 2 + 2/(2 + 3/(3 + 4/(4 + ...)))",
        source_label="PLANTED",
    )
    report = screen_candidate(corpus, planted)
    assert report["verdict"] == "KNOWN"
    assert report["test_that_fired"] in {"exact_pattern_match", "equivalence_orbit_match"}
    assert report["chain_verified"] is True
    assert corpus.by_id[report["matched_record"]["record_id"]].seed_id == "euler_e_minus_one"


def test_planted_extension_is_recovered_by_the_orbit_search(corpus) -> None:
    """A candidate one *extension* away from a corpus record must be found by the search.

    The reported target is immaterial here: tests 1 and 2 are structural, so this exercises
    the orbit search itself rather than any value coincidence.
    """

    record = corpus.by_id["seed:erfc_z_1_1|contract_odd"]
    assert record.pattern is not None
    extended = transform_extension(record.pattern)
    assert corpus.lookup_pattern(extended) == []
    assert transform_contraction(extended, "even").key() == record.pattern.key()
    candidate = Candidate(
        candidate_id="planted-extension",
        target="pi",
        pattern=extended,
        wrap=mobius_of(1, 0, 0, 1),
        cf_value=record.cf_value,
        formula_text="planted extension of a contracted error-function continued fraction",
        source_label="PLANTED",
    )
    report = screen_candidate(corpus, candidate)
    assert report["verdict"] == "KNOWN"
    assert report["test_that_fired"] == "equivalence_orbit_match"
    assert report["transformation_chain"][0]["transformation"] == "contract_even"
    assert report["matched_record"]["record_id"] == record.record_id


def test_value_match_alone_is_never_known(corpus, adjudication_receipt) -> None:
    inconclusive = [
        item
        for item in adjudication_receipt["candidates"]
        if item["verdict"] == "INCONCLUSIVE_VALUE_MATCH"
    ]
    assert inconclusive, "the third verdict class must be exercised by the real run"
    for item in inconclusive:
        assert item["test_that_fired"] == "value_match_without_structural_confirmation"
        assert item["value_matches"]["records_with_the_same_reported_value"] or (
            item["value_matches"]["records_whose_continued_fraction_has_the_same_limit"]
        )
        assert "matched_record" not in item
        assert item["why_no_chain"]
    assert adjudication_receipt["claims"]["value_match_alone_is_not_membership"] is True


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_receipt_binds_the_enumeration_and_corpus_artifacts(
    adjudication_receipt, enumeration_receipt, corpus
) -> None:
    assert adjudication_receipt["input"]["content_sha256"] == enumeration_receipt["content_sha256"]
    assert (
        adjudication_receipt["input"]["result_core_sha256"]
        == enumeration_receipt["result_core_sha256"]
    )
    assert adjudication_receipt["corpus"]["content_sha256"] == corpus.manifest["content_sha256"]
    assert adjudication_receipt["corpus"]["records_sha256"] == corpus.manifest["records_sha256"]
    assert adjudication_receipt["corpus"]["records"] == len(corpus.records)
    assert adjudication_receipt["claims"] == screen_module.SCREEN_CLAIMS


def test_receipt_validates(adjudication_receipt) -> None:
    validate_receipt(adjudication_receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.__setitem__("content_sha256", "0" * 64), id="seal"),
        pytest.param(
            lambda r: r["config"].__setitem__("orbit_depth", 99), id="config-binding"
        ),
        pytest.param(
            lambda r: r["claims"].__setitem__("corpus_absence_establishes_novelty", True),
            id="claims",
        ),
        pytest.param(
            lambda r: r["candidates"][0].__setitem__("verdict", "NOVEL"), id="verdict-vocabulary"
        ),
        pytest.param(
            lambda r: r["controls"].__setitem__("screened_KNOWN", 1), id="control-gate"
        ),
    ],
)
def test_receipt_tamper_is_detected(adjudication_receipt, mutate) -> None:
    tampered = copy.deepcopy(adjudication_receipt)
    mutate(tampered)
    with pytest.raises(ScreenError):
        validate_receipt(tampered)


def _reseal(receipt: dict) -> None:
    """Re-seal a doctored receipt so the *content* check, not the hash, has to catch it."""

    measurement = receipt.pop("measurement", {"elapsed_seconds": "0.000"})
    for key in ("content_sha256", "result_core_sha256"):
        receipt.pop(key, None)
    receipt["result_core_sha256"] = canonical_sha256(receipt)
    receipt["measurement"] = measurement
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)


def test_known_verdict_without_a_citation_is_rejected(adjudication_receipt) -> None:
    tampered = copy.deepcopy(adjudication_receipt)
    for item in tampered["candidates"]:
        if item["verdict"] == "KNOWN":
            item["matched_record"]["citation"]["reference"] = ""
            break
    _reseal(tampered)
    with pytest.raises(ScreenError, match="citation"):
        validate_receipt(tampered)


def test_screen_is_deterministic(corpus, enumeration_receipt) -> None:
    first = run_screen(enumeration_receipt, corpus)
    second = run_screen(enumeration_receipt, corpus)
    assert first["result_core_sha256"] == second["result_core_sha256"]
    stored = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    assert first["result_core_sha256"] == stored["result_core_sha256"]


def test_control_gate_aborts_the_run_when_the_screen_cannot_recover_known_formulas(
    corpus, enumeration_receipt
) -> None:
    """Break the controls and the run must refuse to report anything at all."""

    doctored = copy.deepcopy(enumeration_receipt)
    for survivor in doctored["survivors"]:
        if survivor["prior_art"]["label"] == "KNOWN_REDISCOVERED":
            survivor["alpha"] = [4, 4, 4]
            survivor["beta"] = [3, 3, 3]
    with pytest.raises(ScreenError, match="control recovery rate"):
        run_screen(doctored, corpus)
