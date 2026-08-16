"""M10 SAT-certificate-lane gates.

The lane's value is that its receipts cannot overstate what happened, so the
load-bearing tests are the certificate-discipline gates: a SAT model is re-verified
independently and a corrupted model fails closed; an UNSAT decision is labeled
solver-asserted unless a DRAT proof was really checked, and no forgery path upgrades
the label; caps trip into sealed receipts, never silence; receipts are deterministic;
and the six known-answer controls sit on their classical sides (R(3,3) = 6,
W(2,3) = 9, Pythagorean n = 20).
"""

from __future__ import annotations

import json
import shutil
from itertools import combinations
from pathlib import Path

import pytest

from sigma_theory_compiler.problem_queue import MACHINE_FORM_KINDS
from sigma_theory_compiler.sat_certificate_lane import (
    CONTROLS,
    DECISION_SAT,
    DECISION_UNSAT_ASSERTED,
    DECISION_UNSAT_DRAT,
    DEFAULT_CAPS,
    GENERIC_CONTROL_DIMACS,
    PROOF_EXTRACTION_USABLE,
    SOLVER_BACKEND,
    SYSTEM_LIMITS,
    SatCertificateLaneError,
    decide,
    main,
    run_controls,
    statement_from_machine_form,
    validate_receipt,
    verify_model,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SEALED_DIR = ROOT / "runs" / "math" / "sat-lane"

RAMSEY_5 = {"kind": "ramsey_edge_coloring", "n": 5, "k": 3}
RAMSEY_6 = {"kind": "ramsey_edge_coloring", "n": 6, "k": 3}
VDW_8 = {"kind": "vdw_arithmetic_progression", "n": 8, "k": 3}
VDW_9 = {"kind": "vdw_arithmetic_progression", "n": 9, "k": 3}
PYTH_20 = {"kind": "pythagorean_triple_coloring", "n": 20}

UNSAT_DECISIONS = (DECISION_UNSAT_ASSERTED, DECISION_UNSAT_DRAT)


def _reseal(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


class _Clock:
    """Injected monotonic source: every reading advances by a fixed number of seconds."""

    def __init__(self, step_seconds: int) -> None:
        self.ns = 0
        self.step = step_seconds * 1_000_000_000

    def __call__(self) -> int:
        self.ns += self.step
        return self.ns


@pytest.fixture(scope="module")
def ramsey5() -> dict:
    return decide(RAMSEY_5)


@pytest.fixture(scope="module")
def ramsey6() -> dict:
    return decide(RAMSEY_6)


@pytest.fixture(scope="module")
def vdw8() -> dict:
    return decide(VDW_8)


@pytest.fixture(scope="module")
def vdw9() -> dict:
    return decide(VDW_9)


@pytest.fixture(scope="module")
def pyth20() -> dict:
    return decide(PYTH_20)


@pytest.fixture(scope="module")
def generic_unsat() -> dict:
    return decide({"kind": "generic_cnf"}, dimacs_text=GENERIC_CONTROL_DIMACS)


# ---------------------------------------------------------------------------
# Known-answer controls: R(3,3) = 6, W(2,3) = 9, Pythagorean n = 20
# ---------------------------------------------------------------------------


def test_ramsey_r33_both_sides(ramsey5, ramsey6):
    assert ramsey5["decision"] == DECISION_SAT
    assert ramsey5["claims"]["model_independently_verified"] is True
    assert ramsey6["decision"] in UNSAT_DECISIONS
    validate_receipt(ramsey5)
    validate_receipt(ramsey6)


def test_vdw_w23_both_sides(vdw8, vdw9):
    assert vdw8["decision"] == DECISION_SAT
    assert vdw9["decision"] in UNSAT_DECISIONS
    validate_receipt(vdw8)
    validate_receipt(vdw9)


def test_pythagorean_n20_sat_and_scope_states_7825_boundary(pyth20):
    assert pyth20["decision"] == DECISION_SAT
    validate_receipt(pyth20)
    assert "7825" in pyth20["scope"]
    assert "Heule" in pyth20["scope"]
    assert "Heule" in pyth20["literature"]["citation"]
    assert "7825" in pyth20["literature"]["note"]


def test_generic_cnf_both_directions(generic_unsat):
    assert generic_unsat["decision"] in UNSAT_DECISIONS
    validate_receipt(generic_unsat, dimacs_text=GENERIC_CONTROL_DIMACS)
    text = "p cnf 2 2\n1 2 0\n-1 2 0\n"
    receipt = decide({"kind": "generic_cnf"}, dimacs_text=text)
    assert receipt["decision"] == DECISION_SAT
    validate_receipt(receipt, dimacs_text=text)


def test_sat_model_satisfies_every_clause_by_independent_recheck(ramsey5):
    """The test re-derives the K_5 clauses itself and checks the sealed coloring."""

    n = 5
    model = ramsey5["model"]
    assert sorted(abs(literal) for literal in model) == list(range(1, 11))
    assignment = {abs(literal): literal > 0 for literal in model}

    def edge(i: int, j: int) -> int:
        return i * (2 * n - i - 1) // 2 + (j - i)

    for subset in combinations(range(n), 3):
        colors = {assignment[edge(i, j)] for i, j in combinations(subset, 2)}
        assert len(colors) == 2, f"monochromatic triangle on {subset}"


# ---------------------------------------------------------------------------
# Encoder correctness: hand-computed counts and structures
# ---------------------------------------------------------------------------


def test_encoder_counts_match_hand_computed_values(ramsey5, ramsey6, vdw8, vdw9, pyth20):
    assert (ramsey5["encoding"]["variables"], ramsey5["encoding"]["clauses"]) == (10, 20)
    assert (ramsey6["encoding"]["variables"], ramsey6["encoding"]["clauses"]) == (15, 40)
    # vdw n=8, k=3: d=1 gives 6 APs, d=2 gives 4, d=3 gives 2 -> 12 APs -> 24 clauses.
    assert (vdw8["encoding"]["variables"], vdw8["encoding"]["clauses"]) == (8, 24)
    # vdw n=9, k=3: 7 + 5 + 3 + 1 = 16 APs -> 32 clauses.
    assert (vdw9["encoding"]["variables"], vdw9["encoding"]["clauses"]) == (9, 32)
    # Pythagorean n=20: (3,4,5) (6,8,10) (9,12,15) (12,16,20) (5,12,13) (8,15,17).
    assert (pyth20["encoding"]["variables"], pyth20["encoding"]["clauses"]) == (20, 12)


def test_pythagorean_triples_match_brute_force():
    from sigma_theory_compiler.sat_certificate_lane import _pythagorean_triples

    n = 200
    brute = sorted(
        (a, b, c)
        for c in range(1, n + 1)
        for b in range(1, c)
        for a in range(1, b)
        if a * a + b * b == c * c
    )
    assert _pythagorean_triples(n) == brute


def test_edge_variable_map_is_a_bijection():
    from sigma_theory_compiler.sat_certificate_lane import _edge_var

    n = 6
    ranks = [_edge_var(i, j, n) for i in range(n) for j in range(i + 1, n)]
    assert sorted(ranks) == list(range(1, n * (n - 1) // 2 + 1))


def test_variable_map_is_documented_in_the_receipt(ramsey5, vdw8, pyth20, generic_unsat):
    for receipt in (ramsey5, vdw8, pyth20, generic_unsat):
        variable_map = receipt["encoding"]["variable_map"]
        assert set(variable_map) == {"scheme", "description"}
        assert variable_map["description"].strip()


def test_vacuous_instances_are_sat_with_all_false_model():
    receipt = decide({"kind": "pythagorean_triple_coloring", "n": 4})
    assert receipt["decision"] == DECISION_SAT
    assert receipt["encoding"]["clauses"] == 0
    assert receipt["model"] == [-1, -2, -3, -4]
    validate_receipt(receipt)


# ---------------------------------------------------------------------------
# Certificate discipline: corrupted models fail closed
# ---------------------------------------------------------------------------


def test_verify_model_catches_a_flipped_literal(ramsey5):
    from sigma_theory_compiler.sat_certificate_lane import _build_clauses

    clauses = _build_clauses({"kind": "ramsey_edge_coloring", "n": 5, "k": 3})
    model = list(ramsey5["model"])
    ok, detail = verify_model(10, clauses, model)
    assert ok and detail["reason"] == "verified"
    # Every K_5 triangle-free 2-coloring is critical: flipping any one edge color
    # must create a monochromatic triangle, so every single-literal flip fails.
    for position in range(len(model)):
        corrupted = list(model)
        corrupted[position] = -corrupted[position]
        ok, detail = verify_model(10, clauses, corrupted)
        assert not ok, f"flip at position {position} was not caught"
        assert detail["reason"] == "unsatisfied_clause"


def test_verify_model_rejects_malformed_shapes():
    clauses = [[1]]
    assert verify_model(1, clauses, None)[0] is False
    assert verify_model(1, clauses, [])[0] is False
    assert verify_model(1, clauses, [2])[0] is False
    assert verify_model(1, clauses, [True])[0] is False
    assert verify_model(2, clauses, [1, 1])[0] is False


def test_corrupted_model_in_a_resealed_receipt_is_rejected(ramsey5):
    forged = json.loads(json.dumps(ramsey5))
    forged["model"][0] = -forged["model"][0]
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="fails independent verification"):
        validate_receipt(forged)


# ---------------------------------------------------------------------------
# UNSAT labeling honesty: asserted stays asserted, forgeries are rejected
# ---------------------------------------------------------------------------


def test_unsat_asserted_claims_are_honest(ramsey6, vdw9, generic_unsat):
    for receipt in (ramsey6, vdw9, generic_unsat):
        if receipt["decision"] != DECISION_UNSAT_ASSERTED:
            continue  # a platform with drat-trim may verify; honesty is per-label
        assert receipt["claims"]["unsat_independently_verified"] is False
        assert receipt["claims"]["unsat_solver_asserted_only"] is True
        assert receipt["drat"]["verified"] is False
        assert "NOT independently verified" in receipt["interpretation"]["conclusion"]


def test_forged_drat_label_is_rejected(ramsey6):
    if ramsey6["decision"] != DECISION_UNSAT_ASSERTED:
        pytest.skip("this environment produced a real DRAT verification")
    # Forgery 1: relabel the decision alone (claims now disagree).
    forged = json.loads(json.dumps(ramsey6))
    forged["decision"] = DECISION_UNSAT_DRAT
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="claims do not match"):
        validate_receipt(forged)
    # Forgery 2: also recompute claims and interpretation (drat block still says no).
    forged = json.loads(json.dumps(ramsey6))
    forged["decision"] = DECISION_UNSAT_DRAT
    forged["claims"]["unsat_independently_verified"] = True
    forged["claims"]["unsat_solver_asserted_only"] = False
    forged["interpretation"]["conclusion"] = (
        f"decision {DECISION_UNSAT_DRAT}: "
        + forged["interpretation"]["cnf_unsatisfiable_means"]
        + ". The DRAT refutation proof was independently checked by drat-trim."
    )
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="drat.verified"):
        validate_receipt(forged)
    # Forgery 3: also force the verified flag without a proof (empty proof rejected).
    forged["drat"].update({"available": True, "proof_extraction_usable": True})
    forged["drat"].update({"used": True, "verified": True})
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="nonempty proof"):
        validate_receipt(forged)


def test_empty_extracted_proof_never_upgrades_the_label():
    """A verifier that would say yes must never be reached with an empty proof."""

    def verifier(dimacs_text: str, lines: list[str]) -> bool:
        assert lines, "an empty proof must never be submitted for verification"
        return True

    receipt = decide(RAMSEY_6, drat_verifier=verifier)
    assert receipt["probes"]["drat_verifier_overridden"] is True
    if receipt["drat"]["proof_lines"] > 0:
        # A platform that really extracts proofs may verify through the override.
        assert receipt["decision"] == DECISION_UNSAT_DRAT
    else:
        assert receipt["decision"] == DECISION_UNSAT_ASSERTED
        assert receipt["drat"]["verified"] is False
        assert receipt["claims"]["unsat_independently_verified"] is False


def test_attempt_drat_guard_and_upgrade_wiring():
    from sigma_theory_compiler.sat_certificate_lane import _attempt_drat

    clauses = [[1], [-1]]
    # Empty proof: the verifier must not be called and nothing is verified.
    block = _attempt_drat(1, clauses, [], None, lambda text, lines: True, 1)
    assert block["verified"] is False
    assert block["proof_lines"] == 0
    assert block["proof_sha256"] is None
    # Nonempty proof: the injected verifier decides, and the proof is hashed.
    seen: dict = {}

    def verifier(dimacs_text: str, lines: list[str]) -> bool:
        seen["dimacs"] = dimacs_text
        seen["lines"] = lines
        return True

    block = _attempt_drat(1, clauses, ["0"], None, verifier, 1)
    assert block["verified"] is True
    assert block["proof_lines"] == 1
    assert isinstance(block["proof_sha256"], str)
    assert seen["dimacs"].startswith("p cnf 1 2\n")
    block = _attempt_drat(1, clauses, ["0"], None, lambda text, lines: False, 1)
    assert block["verified"] is False


@pytest.mark.skipif(
    shutil.which("drat-trim") is None or not PROOF_EXTRACTION_USABLE,
    reason="drat-trim not on PATH or proof extraction unusable on this platform",
)
def test_real_drat_path_verifies_k6():  # pragma: no cover - environment-gated
    receipt = decide(RAMSEY_6)
    assert receipt["decision"] == DECISION_UNSAT_DRAT
    assert receipt["claims"]["unsat_independently_verified"] is True
    validate_receipt(receipt)


# ---------------------------------------------------------------------------
# Caps: hard, sealed, and honest
# ---------------------------------------------------------------------------


def test_cap_trip_max_vars():
    receipt = decide(VDW_9, caps={"max_clauses": 1000, "max_seconds": 60, "max_vars": 5})
    assert receipt["decision"] == "CAP_TRIPPED:max_vars"
    assert receipt["model"] is None
    assert receipt["encoding"]["cnf_sha256"] is None
    assert receipt["claims"]["model_independently_verified"] is False
    assert receipt["claims"]["unsat_independently_verified"] is False
    assert receipt["interpretation"]["conclusion"] is None
    validate_receipt(receipt)


def test_cap_trip_max_clauses():
    receipt = decide(RAMSEY_6, caps={"max_clauses": 10, "max_seconds": 60, "max_vars": 1000})
    assert receipt["decision"] == "CAP_TRIPPED:max_clauses"
    assert receipt["encoding"]["clauses"] == 40
    assert receipt["encoding"]["cnf_sha256"] is None
    validate_receipt(receipt)


def test_cap_trip_max_vars_refuses_pythagorean_enumeration():
    receipt = decide(
        {"kind": "pythagorean_triple_coloring", "n": 100},
        caps={"max_clauses": 1000, "max_seconds": 60, "max_vars": 50},
    )
    assert receipt["decision"] == "CAP_TRIPPED:max_vars"
    assert receipt["encoding"]["clauses"] is None  # the refused enumeration is not faked
    validate_receipt(receipt)


def test_cap_trip_max_seconds_via_terminal_wall_audit():
    receipt = decide(
        RAMSEY_5,
        caps={"max_clauses": 1000, "max_seconds": 1, "max_vars": 1000},
        monotonic_ns=_Clock(2),
    )
    assert receipt["decision"] == "CAP_TRIPPED:max_seconds"
    assert receipt["model"] is None  # the over-budget answer was discarded
    assert receipt["probes"]["monotonic_ns_overridden"] is True
    assert receipt["encoding"]["cnf_sha256"] is not None  # the CNF was built before the solve
    validate_receipt(receipt)


def test_caps_are_validated_fail_closed():
    with pytest.raises(SatCertificateLaneError, match="exactly"):
        decide(RAMSEY_5, caps={"max_vars": 10})
    with pytest.raises(SatCertificateLaneError, match="nonnegative"):
        decide(RAMSEY_5, caps={"max_clauses": 10, "max_seconds": -1, "max_vars": 10})
    with pytest.raises(SatCertificateLaneError, match="system limit"):
        decide(
            RAMSEY_5,
            caps={
                "max_clauses": SYSTEM_LIMITS["max_clauses"] + 1,
                "max_seconds": 60,
                "max_vars": 10,
            },
        )
    with pytest.raises(SatCertificateLaneError, match="plain integer"):
        decide(RAMSEY_5, caps={"max_clauses": 10.0, "max_seconds": 60, "max_vars": 10})


# ---------------------------------------------------------------------------
# Determinism: same statement, byte-identical receipt
# ---------------------------------------------------------------------------


def test_receipts_are_deterministic(ramsey5, vdw8, pyth20, generic_unsat):
    for receipt, statement, text in (
        (ramsey5, RAMSEY_5, None),
        (vdw8, VDW_8, None),
        (pyth20, PYTH_20, None),
        (generic_unsat, {"kind": "generic_cnf"}, GENERIC_CONTROL_DIMACS),
    ):
        again = decide(statement, dimacs_text=text)
        assert canonical_json_bytes(again) == canonical_json_bytes(receipt)


# ---------------------------------------------------------------------------
# Tamper: seal replay plus semantic re-verification
# ---------------------------------------------------------------------------


def test_tampered_fields_without_reseal_are_rejected(ramsey5, ramsey6):
    for field, value in (
        ("decision", DECISION_UNSAT_ASSERTED),
        ("encoder_version", "sat-lane-encoder-9.9"),
        ("model", None),
        ("scope", "rewritten"),
    ):
        forged = json.loads(json.dumps(ramsey5))
        forged[field] = value
        with pytest.raises(SatCertificateLaneError):
            validate_receipt(forged)
    forged = json.loads(json.dumps(ramsey6))
    forged["statement"]["n"] = 7
    with pytest.raises(SatCertificateLaneError, match="seal changed"):
        validate_receipt(forged)


def test_decision_flip_sat_to_unsat_is_caught_by_re_solve(ramsey5):
    forged = json.loads(json.dumps(ramsey5))
    forged["decision"] = DECISION_UNSAT_ASSERTED
    forged["model"] = None
    forged["claims"]["model_independently_verified"] = False
    forged["claims"]["unsat_solver_asserted_only"] = True
    forged["interpretation"] = None  # replaced below by the real recompute path
    del forged["interpretation"]
    honest = decide(RAMSEY_6)  # borrow a coherent interpretation shape, then rewrite n
    forged["interpretation"] = json.loads(json.dumps(honest["interpretation"]))
    forged["interpretation"]["cnf_satisfiable_means"] = forged["interpretation"][
        "cnf_satisfiable_means"
    ].replace("K_6", "K_5").replace("n=6", "n=5")
    forged["interpretation"]["cnf_unsatisfiable_means"] = forged["interpretation"][
        "cnf_unsatisfiable_means"
    ].replace("n=6", "n=5")
    forged["interpretation"]["conclusion"] = forged["interpretation"]["conclusion"].replace(
        "n=6", "n=5"
    )
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="re-solve direction check"):
        validate_receipt(forged)


def test_decision_flip_unsat_to_sat_requires_an_impossible_model(ramsey6):
    forged = json.loads(json.dumps(ramsey6))
    forged["decision"] = DECISION_SAT
    forged["claims"]["model_independently_verified"] = True
    forged["claims"]["unsat_solver_asserted_only"] = False
    forged["model"] = [-(v + 1) for v in range(15)]
    honest_sat = decide(RAMSEY_5)
    forged["interpretation"] = json.loads(json.dumps(honest_sat["interpretation"]))
    with pytest.raises(SatCertificateLaneError):
        validate_receipt(_reseal(forged))


def test_statement_echo_tamper_is_rejected(ramsey5):
    forged = json.loads(json.dumps(ramsey5))
    forged["statement"]["n"] = 6
    forged = _reseal(forged)
    with pytest.raises(SatCertificateLaneError, match="text does not match"):
        validate_receipt(forged)


# ---------------------------------------------------------------------------
# Statement and DIMACS validation
# ---------------------------------------------------------------------------


def test_statement_rejections():
    with pytest.raises(SatCertificateLaneError, match="unknown statement kind"):
        decide({"kind": "graph_coloring", "n": 4})
    with pytest.raises(SatCertificateLaneError, match="must be at least 2"):
        decide({"kind": "ramsey_edge_coloring", "n": 1, "k": 3})
    with pytest.raises(SatCertificateLaneError, match="must be at least 2"):
        decide({"kind": "ramsey_edge_coloring", "n": 5, "k": 1})
    with pytest.raises(SatCertificateLaneError, match="plain integer"):
        decide({"kind": "vdw_arithmetic_progression", "n": 8.0, "k": 3})
    with pytest.raises(SatCertificateLaneError, match="plain integer"):
        decide({"kind": "vdw_arithmetic_progression", "n": True, "k": 3})
    with pytest.raises(SatCertificateLaneError, match="unknown statement keys"):
        decide({"kind": "pythagorean_triple_coloring", "n": 20, "k": 3})
    with pytest.raises(SatCertificateLaneError, match="parameter limit"):
        decide({"kind": "pythagorean_triple_coloring", "n": SYSTEM_LIMITS["max_vars"] + 1})
    with pytest.raises(SatCertificateLaneError, match="text does not match"):
        decide({"kind": "pythagorean_triple_coloring", "n": 20, "text": "something else"})
    with pytest.raises(SatCertificateLaneError, match="only applies to generic_cnf"):
        decide(RAMSEY_5, dimacs_text=GENERIC_CONTROL_DIMACS)


def test_dimacs_rejections():
    cases = (
        ("", "nonempty"),
        ("1 0\n", "before the header"),
        ("p cnf 1\n1 0\n", "header must be"),
        ("p cnf 1 2\n1 0\n", "declares 2 clauses but contains 1"),
        ("p cnf 1 1\n2 0\n", "exceeds the declared variable count"),
        ("p cnf 1 1\n1\n", "unterminated clause"),
        ("p cnf 1 1\nx 0\n", "non-integer DIMACS token"),
        ("p cnf 1 1\np cnf 1 1\n1 0\n", "duplicate DIMACS header"),
    )
    for text, match in cases:
        with pytest.raises(SatCertificateLaneError, match=match):
            decide({"kind": "generic_cnf"}, dimacs_text=text)
    with pytest.raises(SatCertificateLaneError, match="requires dimacs_text"):
        decide({"kind": "generic_cnf"})
    with pytest.raises(SatCertificateLaneError, match="does not match the supplied DIMACS"):
        decide(
            {"kind": "generic_cnf", "variables": 9},
            dimacs_text=GENERIC_CONTROL_DIMACS,
        )


def test_generic_receipt_validation_requires_the_dimacs_text(generic_unsat):
    with pytest.raises(SatCertificateLaneError, match="original DIMACS text"):
        validate_receipt(generic_unsat)


# ---------------------------------------------------------------------------
# Machine-form hook (A2 registry) and scheduler routing
# ---------------------------------------------------------------------------


def test_machine_form_kind_is_registered_in_the_problem_queue():
    assert MACHINE_FORM_KINDS["bounded_combinatorial_coloring"] == {
        "statement_kind": str,
        "n": int,
        "k": int,
    }


def test_statement_from_machine_form_round_trips_into_decide():
    statement = statement_from_machine_form(
        {
            "kind": "bounded_combinatorial_coloring",
            "statement_kind": "ramsey_edge_coloring",
            "n": 5,
            "k": 3,
        }
    )
    assert statement == RAMSEY_5
    assert decide(statement)["decision"] == DECISION_SAT
    assert statement_from_machine_form(
        {
            "kind": "bounded_combinatorial_coloring",
            "statement_kind": "pythagorean_triple_coloring",
            "n": 20,
            "k": 0,
        }
    ) == PYTH_20


def test_statement_from_machine_form_rejections():
    with pytest.raises(SatCertificateLaneError, match="machine_form.kind"):
        statement_from_machine_form({"kind": "sequence_rows"})
    with pytest.raises(SatCertificateLaneError, match="statement_kind"):
        statement_from_machine_form(
            {
                "kind": "bounded_combinatorial_coloring",
                "statement_kind": "generic_cnf",
                "n": 5,
                "k": 0,
            }
        )
    with pytest.raises(SatCertificateLaneError, match="k must be 0"):
        statement_from_machine_form(
            {
                "kind": "bounded_combinatorial_coloring",
                "statement_kind": "pythagorean_triple_coloring",
                "n": 20,
                "k": 3,
            }
        )
    with pytest.raises(SatCertificateLaneError, match="keys changed"):
        statement_from_machine_form(
            {"kind": "bounded_combinatorial_coloring", "statement_kind": "ramsey_edge_coloring"}
        )


# ---------------------------------------------------------------------------
# CLI, immutable receipts, and the sealed known-answer receipts in runs/
# ---------------------------------------------------------------------------


def test_cli_controls_seal_six_receipts_and_are_idempotent(tmp_path):
    output_dir = tmp_path / "sat-lane"
    assert main(["--controls", "--output-dir", str(output_dir)]) == 0
    receipts = sorted(path.name for path in output_dir.glob("*.json"))
    assert receipts == sorted(f"{name}.json" for name, _, _ in CONTROLS)
    assert (output_dir / "generic-cnf-contradiction-unsat.cnf").is_file()
    first_bytes = {path.name: path.read_bytes() for path in output_dir.glob("*.json")}
    assert main(["--controls", "--output-dir", str(output_dir)]) == 0
    for path in output_dir.glob("*.json"):
        assert path.read_bytes() == first_bytes[path.name]
    tampered = output_dir / "ramsey-r33-n5-sat.json"
    tampered.write_bytes(b"{}")
    with pytest.raises(SatCertificateLaneError, match="refusing to overwrite"):
        run_controls(output_dir)


def test_cli_single_statement_and_validate(tmp_path):
    statement_path = tmp_path / "statement.json"
    statement_path.write_text(json.dumps(RAMSEY_5), encoding="utf-8")
    output = tmp_path / "receipt.json"
    assert main(["--statement", str(statement_path), "--output", str(output)]) == 0
    assert main(["--validate", "--output", str(output)]) == 0
    dimacs_path = tmp_path / "instance.cnf"
    dimacs_path.write_text(GENERIC_CONTROL_DIMACS, encoding="utf-8")
    generic_statement = tmp_path / "generic.json"
    generic_statement.write_text(json.dumps({"kind": "generic_cnf"}), encoding="utf-8")
    generic_output = tmp_path / "generic-receipt.json"
    assert (
        main(
            [
                "--statement",
                str(generic_statement),
                "--dimacs",
                str(dimacs_path),
                "--output",
                str(generic_output),
            ]
        )
        == 0
    )
    assert (
        main(
            ["--validate", "--output", str(generic_output), "--dimacs", str(dimacs_path)]
        )
        == 0
    )


def test_sealed_control_receipts_are_valid_honest_and_expected():
    """The committed known-answer receipts under runs/math/sat-lane/ stay honest."""

    assert SEALED_DIR.is_dir(), "run the controls: python -m ... --controls"
    for name, statement, expected in CONTROLS:
        path = SEALED_DIR / f"{name}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        text = None
        if statement["kind"] == "generic_cnf":
            text = (SEALED_DIR / f"{name}.cnf").read_text(encoding="utf-8")
        validate_receipt(receipt, dimacs_text=text)
        if expected == "UNSAT":
            assert receipt["decision"] in UNSAT_DECISIONS
            assert (
                receipt["claims"]["unsat_independently_verified"]
                is (receipt["decision"] == DECISION_UNSAT_DRAT)
            )
        else:
            assert receipt["decision"] == expected
        assert receipt["statement"]["kind"] == statement["kind"]
        assert receipt["probes"] == {
            "drat_verifier_overridden": False,
            "monotonic_ns_overridden": False,
        }
        assert receipt["caps"] == DEFAULT_CAPS
        assert receipt["solver"]["backend"] == SOLVER_BACKEND
        stored = path.read_bytes()
        assert stored == canonical_json_bytes(receipt) + b"\n", name
