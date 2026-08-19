"""C3 gates -- self-generated conjectures, their obligations, and their adjudication.

The load-bearing test in this file is ``test_restating_generator_is_refused_by_the_gates``
and its neighbours: a generator that hands back what it was given is not conjecture
generation, and the admission gates exist to say so.  Everything else fixes the properties
that make the loop worth running -- the targets are generated rather than named, statements
are formed from a prefix and confronted with the whole domain, and a verdict is either an
exact certificate or an exact witness.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import pytest

from sigma_theory_compiler import math_proof as proof
from sigma_theory_compiler.self_generated_conjecture_adjudication import (
    ADMISSION_GATES,
    CLAIMS,
    CONGRUENCE_ROUTE,
    MAX_MODULUS,
    RELATION_REFUTE_ROUTE,
    ConjectureAdjudicationError,
    PoolConfig,
    adjudicate_congruence,
    adjudicate_relation,
    admit,
    binomial_poly,
    build_pool,
    congruence_quotient,
    emit_congruences,
    emit_relations,
    forward_differences,
    main,
    partial_sum,
    poly_eval,
    poly_from_values,
    relation_schema_counts,
    restating_generator,
    run_loop,
    seed_values,
    validate_receipt,
)

CONFIG = PoolConfig()


@pytest.fixture(scope="module")
def pool() -> list[dict]:
    return build_pool(CONFIG)["objects"]


@pytest.fixture(scope="module")
def schema_counts(pool: list[dict]) -> dict:
    return relation_schema_counts(pool, CONFIG)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return run_loop()


def _rational(record) -> list[Fraction]:
    return [Fraction(value, record["denominator"]) for value in record["numerator"]]


# ---------------------------------------------------------------------------
# The universe generates itself.
# ---------------------------------------------------------------------------


def test_pool_closed_forms_reproduce_the_summed_values_exactly(pool):
    """Each object's certificate representation is checked against its own construction."""

    assert len(pool) > 100
    for record in pool:
        seed = record["seed"]
        values = seed_values(seed["power"], seed["a"], seed["b"], CONFIG.window)
        for _ in range(seed["depth"]):
            values = partial_sum(values)
        assert values == record["values"]
        coefficients = _rational(record)
        for index, value in enumerate(values):
            assert poly_eval(coefficients, index) == value


def test_pool_objects_are_indexed_by_their_generating_parameters_only(pool):
    """No object carries a human-chosen name; the id *is* the point in the declared box."""

    for record in pool:
        seed = record["seed"]
        assert record["object_id"] == (
            f"S{seed['depth']}:p{seed['power']}:a{seed['a']}:b{seed['b']}"
        )
        assert seed["power"] in CONFIG.seed_powers
        assert seed["depth"] in CONFIG.sum_depths
        assert abs(seed["a"]) <= CONFIG.seed_box
        assert abs(seed["b"]) <= CONFIG.seed_box


def test_pool_is_deduplicated_by_value_vector(pool):
    vectors = [tuple(record["values"]) for record in pool]
    assert len(set(vectors)) == len(vectors)


def test_interpolation_and_binomial_basis_agree_on_a_known_partial_sum():
    values = partial_sum(seed_values(2, 0, 0, 12))
    coefficients = poly_from_values(values, 3)
    for index, value in enumerate(values):
        assert poly_eval(coefficients, index) == value
    assert poly_eval(coefficients, 8) == 204  # 8*9*17/6
    assert binomial_poly(2) == (Fraction(0), Fraction(-1, 2), Fraction(1, 2))
    assert forward_differences([Fraction(v) for v in (1, 3, 7, 13)]) == (
        Fraction(1),
        Fraction(2),
        Fraction(2),
        Fraction(0),
    )


# ---------------------------------------------------------------------------
# Statements are proposed from a prefix, never from the answer.
# ---------------------------------------------------------------------------


def test_emission_reads_the_formation_window_only(pool):
    """Scramble every value outside the formation window; emission must not notice."""

    tampered = copy.deepcopy(pool)
    for offset, record in enumerate(tampered):
        for index in range(CONFIG.prefix, CONFIG.window):
            record["values"][index] += 7 * (index + offset + 1)

    assert emit_congruences(tampered, CONFIG) == emit_congruences(pool, CONFIG)

    relation_tampered = copy.deepcopy(pool)
    for offset, record in enumerate(relation_tampered):
        for index in range(CONFIG.relation_window, CONFIG.window):
            record["values"][index] += 11 * (index + offset + 1)
    assert emit_relations(relation_tampered, CONFIG) == emit_relations(pool, CONFIG)


def test_emitted_statements_declare_a_claim_beyond_their_formation_window(pool):
    for candidate in emit_congruences(pool, CONFIG)[:50] + emit_relations(pool, CONFIG)[:50]:
        assert candidate["claim_range"] == "n >= 0"
        assert candidate["formation_window"][1] < CONFIG.window


def test_emitted_congruence_parameters_stay_inside_the_declared_lattice(pool):
    for candidate in emit_congruences(pool, CONFIG):
        parameters = candidate["parameters"]
        assert 2 <= parameters["m"] <= MAX_MODULUS
        assert 0 <= parameters["r"] < parameters["m"]
        assert 0 <= parameters["j"] < parameters["q"]


# ---------------------------------------------------------------------------
# THE DEGENERACY.  A generator that only restates its input must not get through.
# ---------------------------------------------------------------------------


def test_restating_generator_is_refused_by_the_gates(pool, schema_counts):
    """The control that must fail.  Every disguise of "here is what you gave me" dies."""

    candidates = restating_generator(pool, CONFIG)
    assert candidates, "the degenerate generator produced nothing to refuse"

    refusals: dict[str, set[str]] = {}
    for candidate in candidates:
        verdict = admit(candidate, pool, CONFIG, schema_counts)
        assert not verdict["admitted"], (
            f"restatement disguised as {candidate['disguise']} was admitted: "
            f"{candidate['statement']}"
        )
        assert verdict["refused_gates"]
        refusals.setdefault(candidate["disguise"], set()).update(verdict["refused_gates"])

    # Each disguise is caught by the gate it was built to probe, so the refusals are
    # specific rather than a blanket rejection of anything unfamiliar.
    assert "claim_extends_beyond_formation_window" in refusals["value_table"]
    assert "sides_share_no_object_slot" in refusals["self_relation"]
    assert "predicate_within_declared_lattice" in refusals["unbounded_modulus_echo"]
    assert "pool_share_at_least_two" in refusals["unbounded_modulus_echo"]
    assert "pool_separates_at_least_one" in refusals["tautology"]
    assert set(refusals) == {"value_table", "self_relation", "unbounded_modulus_echo", "tautology"}


def test_a_hand_written_echo_of_the_input_rows_is_refused(pool, schema_counts):
    """The simplest possible restating generator, written here rather than imported."""

    record = pool[len(pool) // 2]
    echo = {
        "kind": "residue_class_congruence",
        "object_id": record["object_id"],
        "parameters": {
            "q": 1,
            "j": 0,
            "m": max(abs(value) for value in record["values"]) * 2 + 3,
            "r": record["values"][0],
        },
        "statement": "the rows, restated",
        "formation_window": [0, CONFIG.prefix],
        "claim_range": "n >= 0",
    }
    verdict = admit(echo, pool, CONFIG, schema_counts)
    assert not verdict["admitted"]
    assert "predicate_within_declared_lattice" in verdict["refused_gates"]


def test_a_tautology_is_refused_for_separating_nothing(pool, schema_counts):
    tautology = {
        "kind": "residue_class_congruence",
        "object_id": pool[0]["object_id"],
        "parameters": {"q": 1, "j": 0, "m": 1, "r": 0},
        "statement": "for all n >= 0: a(n) = 0 (mod 1)",
        "formation_window": [0, CONFIG.prefix],
        "claim_range": "n >= 0",
    }
    verdict = admit(tautology, pool, CONFIG, schema_counts)
    assert not verdict["admitted"]
    assert verdict["content"]["separate"] == 0
    assert verdict["content"]["share"] == len(pool)
    assert "pool_separates_at_least_one" in verdict["refused_gates"]


def test_the_gates_are_not_vacuous(pool, schema_counts):
    """A gate that refuses everything proves nothing.  Honest candidates must survive."""

    admitted = 0
    for candidate in emit_congruences(pool, CONFIG)[:200]:
        if admit(candidate, pool, CONFIG, schema_counts)["admitted"]:
            admitted += 1
    assert admitted > 0
    assert set(ADMISSION_GATES) == {
        "claim_extends_beyond_formation_window",
        "predicate_within_declared_lattice",
        "sides_share_no_object_slot",
        "pool_share_at_least_two",
        "pool_separates_at_least_one",
    }


def test_an_unknown_statement_kind_is_refused(pool, schema_counts):
    verdict = admit(
        {
            "kind": "free_text_assertion",
            "object_id": pool[0]["object_id"],
            "parameters": {},
            "formation_window": [0, 1],
            "claim_range": "n >= 0",
        },
        pool,
        CONFIG,
        schema_counts,
    )
    assert not verdict["admitted"]
    assert len(verdict["refused_gates"]) == len(ADMISSION_GATES) - 1


# ---------------------------------------------------------------------------
# Adjudication: an exact certificate or an exact witness, never a shrug.
# ---------------------------------------------------------------------------


def test_planted_true_congruence_is_proved(pool):
    """n odd => n^2 = 1 (mod 8)."""

    record = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    verdict = adjudicate_congruence(record, {"q": 2, "j": 1, "m": 8, "r": 1})
    assert verdict["verdict"] == "PROVED"
    assert verdict["route"] == CONGRUENCE_ROUTE
    assert all(pair[1] == 1 for pair in verdict["binomial_coefficients"])


def test_planted_false_congruence_is_refuted_with_the_least_witness(pool):
    record = next(item for item in pool if item["object_id"] == "S1:p2:a0:b0")
    verdict = adjudicate_congruence(record, {"q": 4, "j": 0, "m": 10, "r": 0})
    assert verdict["verdict"] == "REFUTED"
    witness = verdict["witness"]
    coefficients = _rational(record)
    assert poly_eval(coefficients, witness["n"]) == witness["a_of_n"]
    assert witness["a_of_n"] % 10 != 0
    # Nothing smaller in the class survives as a counterexample.
    for smaller in range(witness["t"]):
        assert poly_eval(coefficients, 4 * smaller) % 10 == 0


def test_congruence_certificate_is_bound_to_its_statement(pool):
    record = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    verdict = adjudicate_congruence(record, {"q": 2, "j": 1, "m": 8, "r": 1})
    certificate = verdict["reconstruction_certificate"]
    assert certificate["certificate_kind"] == "exact_rational_identity"
    tampered = dict(certificate)
    tampered["decision"] = "proved_because_i_said_so"
    with pytest.raises(proof.ProofValidationError):
        proof.validate_rational_identity_certificate(tampered, _dummy_equation())


def _dummy_equation():
    from sigma_theory_compiler.math_expression_ir import Equation, literal

    return Equation(literal(0), literal(0))


def test_congruence_quotient_is_the_statement_it_claims_to_be(pool):
    """K(t) must literally equal (a(q t + j) - r)/m at every sampled index."""

    record = next(item for item in pool if item["object_id"] == "S2:p2:a1:b-1")
    q, j, m, r = 3, 2, 5, 4
    quotient = congruence_quotient(record, q, j, m, r)
    coefficients = _rational(record)
    for t in range(12):
        assert quotient and poly_eval(quotient, t) == (poly_eval(coefficients, q * t + j) - r) / m


def test_relation_refutation_routes_through_the_counterexample_engine(pool):
    target = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    left = next(item for item in pool if item["object_id"] == "S0:p1:a0:b0")
    right = next(item for item in pool if item["object_id"] == "S0:p1:a0:b1")
    verdict = adjudicate_relation(target, left, right, 1, 1)
    assert verdict["verdict"] == "REFUTED"
    assert verdict["route"] == RELATION_REFUTE_ROUTE
    witness = verdict["witness"]
    assert witness["target"] != witness["combination"]
    index = witness["n"]
    assert poly_eval(_rational(target), index) == witness["target"]
    assert (
        poly_eval(_rational(left), index) + poly_eval(_rational(right), index)
        == witness["combination"]
    )


def test_relation_proof_routes_through_the_identity_prover(pool):
    target = next(item for item in pool if item["object_id"] == "S0:p2:a1:b1")
    left = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    right = next(item for item in pool if item["object_id"] == "S0:p1:a0:b1")
    verdict = adjudicate_relation(target, left, right, 1, 1)
    assert verdict["verdict"] == "PROVED"
    assert verdict["route"] == CONGRUENCE_ROUTE
    assert verdict["certificate"]["decision"].startswith("proved_exact_rational_identity")


# ---------------------------------------------------------------------------
# The whole loop.
# ---------------------------------------------------------------------------


def test_loop_emits_adjudicates_and_reports(receipt):
    counts = receipt["counts"]
    assert counts["candidates_emitted"] > 0
    assert counts["refused_by_admission_gates"] > 0
    assert counts["adjudicated"] > 0
    assert counts["verdicts"]["PROVED"] > 0
    assert counts["verdicts"]["REFUTED"] > 0
    assert set(counts["verdicts_by_kind"]) == {"residue_class_congruence", "polynomial_relation"}
    assert receipt["controls_passed"]
    assert {entry["control"] for entry in receipt["controls"]} == {
        "restating_generator_must_be_refused",
        "planted_true_congruence_must_be_proved",
        "planted_false_congruence_must_be_refuted",
        "planted_false_relation_must_be_refuted",
        "gates_bite_on_the_honest_generator_too",
        "adjudicator_is_not_a_constant_function",
    }


def test_every_adjudicated_conjecture_carries_its_own_obligation(receipt):
    for record in receipt["conjectures"]:
        obligation = record["obligation"]
        assert obligation["obligation_kind"] in {
            "integer_valued_rational_polynomial",
            "exact_rational_identity",
        }
        assert obligation["completeness"]
        assert obligation["routed_to"]
        assert obligation["inputs"]
        assert obligation["obligation_sha256"]
        assert record["adjudication"]["verdict"] in {"PROVED", "REFUTED", "OPEN"}


def test_proved_congruences_survive_an_independent_brute_force_sweep(receipt, pool):
    """Recompute the claim directly from the integer sequence, ignoring the certificate."""

    by_id = {record["object_id"]: record for record in pool}
    checked = 0
    for record in receipt["conjectures"]:
        if record["kind"] != "residue_class_congruence":
            continue
        if record["adjudication"]["verdict"] != "PROVED":
            continue
        parameters = record["parameters"]
        coefficients = _rational(by_id[record["object_id"]])
        for t in range(120):
            value = poly_eval(coefficients, parameters["q"] * t + parameters["j"])
            assert value.denominator == 1
            assert int(value) % parameters["m"] == parameters["r"], record["statement"]
        checked += 1
    assert checked > 50


def test_refuted_conjectures_carry_a_witness_that_actually_kills_them(receipt, pool):
    by_id = {record["object_id"]: record for record in pool}
    refuted = 0
    for record in receipt["conjectures"]:
        adjudication = record["adjudication"]
        if adjudication["verdict"] != "REFUTED":
            continue
        refuted += 1
        witness = adjudication["witness"]
        if record["kind"] == "residue_class_congruence":
            parameters = record["parameters"]
            coefficients = _rational(by_id[record["object_id"]])
            value = poly_eval(coefficients, witness["n"])
            assert value == witness["a_of_n"]
            assert int(value) % parameters["m"] != parameters["r"]
        else:
            parameters = record["parameters"]
            index = witness["n"]
            combination = parameters["alpha"] * poly_eval(
                _rational(by_id[parameters["left_id"]]), index
            ) + parameters["beta"] * poly_eval(_rational(by_id[parameters["right_id"]]), index)
            assert combination == witness["combination"]
            assert poly_eval(_rational(by_id[record["object_id"]]), index) != combination
    assert refuted > 0


def test_loop_is_deterministic():
    assert run_loop()["content_sha256"] == run_loop()["content_sha256"]


def test_receipt_carries_no_floating_point(receipt):
    def walk(value):
        if isinstance(value, bool):
            return
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(receipt)


# ---------------------------------------------------------------------------
# Receipt validation is not a rubber stamp.
# ---------------------------------------------------------------------------


def test_valid_receipt_passes(receipt):
    validate_receipt(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.__setitem__("content_sha256", "0" * 64), id="seal"),
        pytest.param(lambda r: r["claims"].__setitem__("proved_means_novel", True), id="claims"),
        pytest.param(lambda r: r["controls"][0].__setitem__("passed", False), id="control-failed"),
        pytest.param(
            lambda r: r["counts"]["verdicts"].__setitem__("PROVED", 1), id="verdict-tally"
        ),
        pytest.param(lambda r: r["counts"].__setitem__("adjudicated", 3), id="adjudicated-count"),
        pytest.param(
            lambda r: r["counts"].__setitem__("refused_by_admission_gates", 0), id="vacuous-gate"
        ),
        pytest.param(
            lambda r: r["conjectures"][0]["obligation"].__setitem__("completeness", "trust me"),
            id="obligation-hash",
        ),
        pytest.param(
            lambda r: r["conjectures"][0]["admission"].__setitem__("admitted", False),
            id="unadmitted-conjecture",
        ),
        pytest.param(lambda r: r["pool"].__setitem__("objects_admitted", 0.5), id="float"),
    ],
)
def test_tampered_receipt_is_rejected(receipt, mutate):
    tampered = copy.deepcopy(receipt)
    mutate(tampered)
    with pytest.raises(ConjectureAdjudicationError):
        validate_receipt(tampered)


def test_a_refutation_without_a_witness_is_rejected(receipt):
    tampered = copy.deepcopy(receipt)
    for record in tampered["conjectures"]:
        if record["adjudication"]["verdict"] == "REFUTED":
            record["adjudication"].pop("witness")
            break
    else:  # pragma: no cover - the loop always produces refutations
        pytest.fail("no refutation to strip")
    tampered["content_sha256"] = _reseal(tampered)
    with pytest.raises(ConjectureAdjudicationError):
        validate_receipt(tampered)


def _reseal(value: dict) -> str:
    from sigma_theory_compiler.sigma_core import canonical_sha256

    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return canonical_sha256(body)


def test_claims_are_sealed():
    assert CLAIMS["proved_means_novel"] is False
    assert CLAIMS["targets_are_self_generated_not_caller_supplied"] is True
    assert CLAIMS["restating_generator_must_be_refused"] is True


def test_cli_runs_and_writes_a_valid_receipt(tmp_path, capsys):
    output = tmp_path / "receipt.json"
    assert (
        main(
            [
                "--summary",
                "--max-congruences",
                "40",
                "--max-relations",
                "20",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_receipt(payload)
    summary = json.loads(capsys.readouterr().out)
    assert summary["adjudicated"] == payload["counts"]["adjudicated"]
