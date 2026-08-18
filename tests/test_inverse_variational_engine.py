"""Gates for the inverse variational engine.

The claim this module makes is that it *decides* whether equations of motion come from an action,
and *builds* the action when they do.  So these tests do not read verdicts back out of the
receipt.  Wherever the engine says something, the test recomputes it with its own independent
sympy code and compares.

The tests pin: that the Helmholtz conditions really are the identities a Lagrangian forces, by
building a nontrivial Lagrangian, forming its Euler-Lagrange expressions, and checking all three
conditions vanish; that adding a curl term breaks exactly H3, adding damping breaks exactly H2 and
breaking mass-matrix symmetry breaks exactly H1; that the class boundary really excludes what it
says it excludes; that the homotopy really returns the textbook oscillator Lagrangian and the
central ansatz really returns ``-A/r``; that the round trip is not vacuous, by perturbing a
constructed Lagrangian and watching it fail; that the damped oscillator's multiplier really is
``exp(gamma t)`` and really produces Caldirola-Kanai, re-derived here from scratch; that no
multiplier repairs a constant curl; that Noether really returns the energy and the three angular
momenta and really refuses the momenta of a central potential; that the screened-gravity potential
and its flat rotation speed recompute independently; that the coherence obstruction really is the
curl of the coherence-modulated force and really vanishes in the spherical limit; and that the
receipt is deterministic, float-free, claim-frozen and tamper-evident under both a plain edit and
a reseal.

The full-run fixture is session-scoped: one run does a few dozen ``simplify`` passes over
three-dimensional jets and two multiplier sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler import inverse_variational_engine as engine
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sigma_gravity_candidate_gate import CANDIDATE_CONFIG

ROOT = Path(__file__).resolve().parents[1]
_T = sp.Symbol("t", real=True)
_R = sp.Symbol("r", positive=True)


@pytest.fixture(scope="session")
def receipt() -> dict:
    return engine.run_inverse_variational_engine(ROOT)


@pytest.fixture(scope="session")
def sealed() -> dict:
    return json.loads((ROOT / engine.RECEIPT_PATH).read_text(encoding="utf-8"))


def _system(system_id: str, coordinates, equations, parameters=None):
    """Build a system straight from sympy expressions, bypassing the string parser."""

    ir = engine.system_ir(
        system_id,
        list(coordinates),
        [str(item) for item in equations],
        parameters=parameters or {},
    )
    return engine.SecondOrderSystem(ir, equations=list(equations))


def _jet(names):
    positions = tuple(sp.Symbol(name, real=True) for name in names)
    velocities = tuple(sp.Symbol(f"d{name}", real=True) for name in names)
    accelerations = tuple(sp.Symbol(f"dd{name}", real=True) for name in names)
    return positions, velocities, accelerations


def _report(receipt: dict, system_id: str) -> dict:
    return next(item for item in receipt["systems"] if item["system_id"] == system_id)


def _residuals(system) -> dict[tuple[str, int, int], sp.Expr]:
    return {
        (kind, i, j): sp.simplify(raw) for kind, i, j, raw in engine.helmholtz_residuals(system)
    }


# ---------------------------------------------------------------------------
# Step 1.  The Helmholtz conditions really are the identities a Lagrangian forces.
# ---------------------------------------------------------------------------


def test_a_nontrivial_lagrangian_satisfies_every_helmholtz_condition() -> None:
    # A Lagrangian with a position-dependent mass matrix, a gyroscopic cross term, explicit time
    # dependence and a potential.  Nothing about it is special-cased anywhere in the engine.
    (q1, q2), (v1, v2), _ = _jet(["q1", "q2"])
    lagrangian = (
        (1 + q1**2) * v1**2 / 2
        + q1 * q2 * v1 * v2
        + v2**2 / 2
        + _T * q1 * v2
        - q1**3 * q2
        + sp.exp(_T) * q2**2
    )
    placeholder = _system("placeholder", ["q1", "q2"], [sp.Integer(0), sp.Integer(0)])
    equations = engine.euler_lagrange(lagrangian, placeholder)
    built = _system("from_lagrangian", ["q1", "q2"], equations)
    assert all(value == 0 for value in _residuals(built).values())
    assert engine.helmholtz_test(built)["verdict"] == "VARIATIONAL"


def test_an_added_curl_term_breaks_exactly_condition_three() -> None:
    (q1, q2), _, (a1, a2) = _jet(["q1", "q2"])
    clean = _system("clean", ["q1", "q2"], [a1 + q1, a2 + q2])
    assert all(value == 0 for value in _residuals(clean).values())
    curled = _system("curled", ["q1", "q2"], [a1 + q1 - q2, a2 + q2 + q1])
    broken = {key for key, value in _residuals(curled).items() if value != 0}
    assert {key[0] for key in broken} == {"H3"}
    assert _residuals(curled)[("H3", 0, 1)] == -2


def test_an_added_damping_term_breaks_exactly_condition_two() -> None:
    gamma = sp.Symbol("gamma", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    damped = _system("damped", ["q1"], [a1 + gamma * v1 + q1], parameters={"gamma": {"positive": True}})
    residuals = _residuals(damped)
    broken = {key for key, value in residuals.items() if value != 0}
    assert {key[0] for key in broken} == {"H2"}
    assert sp.simplify(residuals[("H2", 0, 0)] - 2 * gamma) == 0


def test_an_asymmetric_mass_matrix_breaks_exactly_condition_one() -> None:
    (q1, q2), _, (a1, a2) = _jet(["q1", "q2"])
    skewed = _system("skewed", ["q1", "q2"], [a1 + 2 * a2 + q1, a2 + q2])
    broken = {key for key, value in _residuals(skewed).items() if value != 0}
    assert {key[0] for key in broken} == {"H1"}


def test_the_conditions_do_not_care_about_an_overall_sign() -> None:
    gamma = sp.Symbol("gamma", positive=True)
    _, (v1,), (a1,) = _jet(["q1"])
    forward = _system("forward", ["q1"], [a1 + gamma * v1], parameters={"gamma": {"positive": True}})
    flipped = _system("flipped", ["q1"], [-a1 - gamma * v1], parameters={"gamma": {"positive": True}})
    assert {key for key, value in _residuals(forward).items() if value != 0} == {
        key for key, value in _residuals(flipped).items() if value != 0
    }


def test_the_total_derivative_treats_the_jet_coordinates_as_independent() -> None:
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system("chain", ["q1"], [a1])
    assert sp.simplify(engine.total_derivative(q1, system) - v1) == 0
    assert sp.simplify(engine.total_derivative(v1, system) - a1) == 0
    jerk = sp.Symbol("dddq1", real=True)
    assert sp.simplify(engine.total_derivative(a1, system) - jerk) == 0
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.total_derivative(jerk, system)


def test_a_system_quadratic_in_the_acceleration_is_out_of_the_declared_class() -> None:
    (q1,), _, (a1,) = _jet(["q1"])
    system = _system("quadratic", ["q1"], [a1**2 - q1])
    verdict = engine.helmholtz_test(system)
    assert verdict["verdict"] == "OUT_OF_DECLARED_CLASS"
    assert verdict["conditions_checked"] == 0
    assert any("affine" in reason for reason in verdict["class_check"]["reasons_outside"])


def test_a_singular_acceleration_matrix_is_out_of_the_declared_class() -> None:
    (q1, q2), _, (a1, _a2) = _jet(["q1", "q2"])
    system = _system("degenerate", ["q1", "q2"], [a1 + q1, q2])
    verdict = engine.helmholtz_test(system)
    assert verdict["verdict"] == "OUT_OF_DECLARED_CLASS"
    assert any("singular" in reason for reason in verdict["class_check"]["reasons_outside"])


def test_a_third_order_equation_is_out_of_the_declared_class() -> None:
    (q1,), _, _ = _jet(["q1"])
    jerk = sp.Symbol("dddq1", real=True)
    system = _system("third_order", ["q1"], [jerk + q1])
    verdict = engine.helmholtz_test(system)
    assert verdict["verdict"] == "OUT_OF_DECLARED_CLASS"
    assert any("third-order" in reason for reason in verdict["class_check"]["reasons_outside"])


def test_a_mismatched_equation_count_is_refused_outright() -> None:
    ir = engine.system_ir("mismatched", ["q1", "q2"], ["ddq1"])
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.SecondOrderSystem(ir)


def test_every_failing_condition_carries_a_nonvanishing_witness(receipt: dict) -> None:
    failures = [
        failing
        for entry in receipt["systems"]
        for failing in entry["helmholtz"]["failing_conditions"]
    ]
    assert failures
    for failing in failures:
        assert failing["nonvanishing_witness"] not in (None, "0")
        assert sp.sympify(failing["nonvanishing_witness"]) != 0


# ---------------------------------------------------------------------------
# Step 2.  Construction, and the round trip that proves it.
# ---------------------------------------------------------------------------


def test_the_homotopy_returns_the_textbook_oscillator_lagrangian() -> None:
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system(
        "oscillator", ["q1"], [a1 + omega**2 * q1], parameters={"omega": {"positive": True}}
    )
    built = engine.construct_by_homotopy(system)
    assert sp.simplify(built - (v1**2 / 2 - omega**2 * q1**2 / 2)) == 0
    assert all(residual == 0 for residual in engine.round_trip_residuals(built, system))


def test_the_central_ansatz_returns_minus_a_over_r() -> None:
    amplitude = sp.Symbol("A", positive=True)
    (q1, q2, q3), (v1, v2, v3), (a1, a2, a3) = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    system = _system(
        "kepler",
        ["q1", "q2", "q3"],
        [
            a1 + amplitude * q1 / radius**3,
            a2 + amplitude * q2 / radius**3,
            a3 + amplitude * q3 / radius**3,
        ],
        parameters={"A": {"positive": True}},
    )
    built, potential = engine.construct_by_central_potential(system)
    assert sp.simplify(potential + amplitude / _R) == 0
    expected = (v1**2 + v2**2 + v3**2) / 2 + amplitude / radius
    assert sp.simplify(built - expected) == 0
    assert all(residual == 0 for residual in engine.round_trip_residuals(built, system))


def test_the_homotopy_is_declared_inapplicable_to_an_inverse_power_force() -> None:
    amplitude = sp.Symbol("A", positive=True)
    (q1, q2, q3), _, (a1, a2, a3) = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    system = _system(
        "kepler",
        ["q1", "q2", "q3"],
        [
            a1 + amplitude * q1 / radius**3,
            a2 + amplitude * q2 / radius**3,
            a3 + amplitude * q3 / radius**3,
        ],
        parameters={"A": {"positive": True}},
    )
    assert engine.construct_by_homotopy(system) is None


def test_the_round_trip_is_not_vacuous() -> None:
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system(
        "oscillator", ["q1"], [a1 + omega**2 * q1], parameters={"omega": {"positive": True}}
    )
    good = v1**2 / 2 - omega**2 * q1**2 / 2
    assert engine.round_trip_residuals(good, system) == [0]
    for perturbation in (q1**2, v1**3, _T * q1 * v1):
        assert engine.round_trip_residuals(good + perturbation, system) != [0]


def test_a_total_time_derivative_does_not_change_the_equations() -> None:
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system(
        "oscillator", ["q1"], [a1 + omega**2 * q1], parameters={"omega": {"positive": True}}
    )
    base = v1**2 / 2 - omega**2 * q1**2 / 2
    gauge = engine.total_derivative(_T * q1**2, system)
    assert engine.round_trip_residuals(base + gauge, system) == [0]


def test_every_constructed_lagrangian_round_trips_to_zero(receipt: dict) -> None:
    constructed = [
        entry for entry in receipt["systems"] if entry["construction"]["round_trip_verified"]
    ]
    assert len(constructed) == 4
    for entry in constructed:
        assert set(entry["construction"]["round_trip_residual"]) == {"0"}
        system = engine.SecondOrderSystem(entry["ir"])
        lagrangian = sp.sympify(
            entry["construction"]["lagrangian"], locals=system.expression_locals()
        )
        # Independent recomputation: differentiate the receipt's own Lagrangian back.
        assert engine.round_trip_residuals(lagrangian, system) == [0] * system.size


def test_a_variational_verdict_with_no_construction_aborts(monkeypatch) -> None:
    monkeypatch.setattr(engine, "construct_by_homotopy", lambda system: None)
    monkeypatch.setattr(engine, "construct_by_central_potential", lambda system: None)
    declared = {
        "role": "control",
        "ir": engine.system_ir("oscillator", ["q1"], ["ddq1 + q1"]),
        "expected": {"helmholtz_verdict": "VARIATIONAL"},
    }
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.analyze_system(declared)


# ---------------------------------------------------------------------------
# Step 3.  Integrating factors.
# ---------------------------------------------------------------------------


def test_the_caldirola_kanai_lagrangian_reproduces_the_damped_equation_from_scratch() -> None:
    gamma = sp.Symbol("gamma", positive=True)
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    multiplied = _system(
        "damped_times_factor",
        ["q1"],
        [sp.exp(gamma * _T) * (a1 + gamma * v1 + omega**2 * q1)],
        parameters={"gamma": {"positive": True}, "omega": {"positive": True}},
    )
    caldirola_kanai = sp.exp(gamma * _T) * (v1**2 / 2 - omega**2 * q1**2 / 2)
    assert engine.round_trip_residuals(caldirola_kanai, multiplied) == [0]
    assert engine.helmholtz_test(multiplied)["verdict"] == "VARIATIONAL"


def test_the_engine_finds_exp_gamma_t_for_the_damped_oscillator(receipt: dict) -> None:
    entry = _report(receipt, "damped_oscillator")
    assert entry["helmholtz"]["verdict"] == "NOT_VARIATIONAL"
    failing = entry["helmholtz"]["failing_conditions"]
    assert len(failing) == 1
    assert failing[0]["condition"] == "H2"
    gamma = sp.Symbol("gamma", positive=True)
    assert sp.simplify(sp.sympify(failing[0]["residual"], locals={"gamma": gamma}) - 2 * gamma) == 0

    search = entry["integrating_factor_search"]
    assert search["outcome"] == "MULTIPLIER_FOUND"
    assert search["space_size"] == 25
    found = search["found"]
    assert len(found) == 1
    assert sp.simplify(
        sp.sympify(found[0]["multiplier"], locals={"gamma": gamma, "t": _T}) - sp.exp(gamma * _T)
    ) == 0
    assert found[0]["helmholtz_verdict"] == "VARIATIONAL"
    assert set(found[0]["round_trip_residual"]) == {"0"}

    # The reported Lagrangian is Caldirola-Kanai, up to the constant rescaling a Lagrangian is
    # always free to carry.  Recomputed here, not read.
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), _ = _jet(["q1"])
    reported = sp.sympify(
        found[0]["lagrangian"], locals={"gamma": gamma, "omega": omega, "t": _T, "q1": q1, "dq1": v1}
    )
    expected = sp.exp(gamma * _T) * (v1**2 / 2 - omega**2 * q1**2 / 2)
    assert sp.simplify(reported - expected) == 0


def test_the_multiplier_grid_is_the_declared_finite_space() -> None:
    assert engine.MULTIPLIER_SPACE["total_degree_bound"] == 2
    assert len(engine.multiplier_exponent_tuples(1, 2)) == 25
    assert len(engine.multiplier_exponent_tuples(2, 2)) == 61
    assert len(engine.multiplier_exponent_tuples(3, 2)) == 113
    for entry in engine.multiplier_exponent_tuples(2, 2):
        assert sum(abs(value) for value in entry) <= 2
    assert tuple([0] * 5) in engine.multiplier_exponent_tuples(2, 2)


def test_no_multiplier_in_the_declared_space_repairs_a_constant_curl(receipt: dict) -> None:
    entry = _report(receipt, "nonconservative_planar_curl")
    search = entry["integrating_factor_search"]
    assert search["outcome"] == "NO_MULTIPLIER_IN_DECLARED_SPACE"
    assert search["space_size"] == 61
    assert search["candidates_rejected"] == 61
    assert search["found"] == []
    # Independent confirmation for the one multiplier a reader would try first: a constant.
    (q1, q2), _, (a1, a2) = _jet(["q1", "q2"])
    for constant in (sp.Integer(2), sp.Rational(-1, 3)):
        scaled = _system(
            "scaled", ["q1", "q2"], [constant * (a1 - q2), constant * (a2 + q1)]
        )
        assert engine.helmholtz_test(scaled)["verdict"] == "NOT_VARIATIONAL"


def test_every_reported_multiplier_really_makes_its_system_variational(receipt: dict) -> None:
    reported = 0
    for entry in receipt["systems"]:
        search = entry["integrating_factor_search"]
        if search is None:
            continue
        base = engine.SecondOrderSystem(entry["ir"])
        for found in search["found"]:
            reported += 1
            equations = [
                sp.sympify(text, locals=base.expression_locals())
                for text in found["multiplied_equations"]
            ]
            multiplied = base.with_equations(equations, "recheck")
            assert engine.helmholtz_test(multiplied)["verdict"] == "VARIATIONAL"
            lagrangian = sp.sympify(found["lagrangian"], locals=base.expression_locals())
            assert engine.round_trip_residuals(lagrangian, multiplied) == [0] * base.size
    assert reported == 1


# ---------------------------------------------------------------------------
# Step 4.  Symmetries and Noether charges.
# ---------------------------------------------------------------------------


def test_time_translation_gives_the_energy_and_it_is_conserved_on_shell() -> None:
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system(
        "oscillator", ["q1"], [a1 + omega**2 * q1], parameters={"omega": {"positive": True}}
    )
    lagrangian = v1**2 / 2 - omega**2 * q1**2 / 2
    charge = engine.noether_charge(lagrangian, sp.Integer(1), [sp.Integer(0)], system)
    energy = v1**2 / 2 + omega**2 * q1**2 / 2
    assert sp.simplify(charge + energy) == 0
    on_shell = engine.solve_accelerations(system)
    assert sp.simplify(engine.total_derivative(charge, system).subs(on_shell)) == 0
    assert engine.identify_charge(charge, lagrangian, system) == "minus_energy"


def test_rotations_give_the_three_angular_momenta_of_a_central_potential() -> None:
    amplitude = sp.Symbol("A", positive=True)
    (q1, q2, q3), (v1, v2, v3), (a1, a2, a3) = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    system = _system(
        "kepler",
        ["q1", "q2", "q3"],
        [
            a1 + amplitude * q1 / radius**3,
            a2 + amplitude * q2 / radius**3,
            a3 + amplitude * q3 / radius**3,
        ],
        parameters={"A": {"positive": True}},
    )
    lagrangian = (v1**2 + v2**2 + v3**2) / 2 + amplitude / radius
    analysis = engine.noether_analysis(lagrangian, system)
    names = {entry["physical_identification"] for entry in analysis["symmetries"]}
    assert names == {
        "minus_energy",
        "angular_momentum_q1_q2",
        "angular_momentum_q1_q3",
        "angular_momentum_q2_q3",
    }
    charge = engine.noether_charge(lagrangian, sp.Integer(0), [-q2, q1, sp.Integer(0)], system)
    assert sp.simplify(charge - (q1 * v2 - q2 * v1)) == 0


def test_space_translation_is_not_a_symmetry_of_a_central_potential() -> None:
    amplitude = sp.Symbol("A", positive=True)
    (q1, q2, q3), (v1, v2, v3), _ = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    system = _system("kepler", ["q1", "q2", "q3"], [sp.Integer(0)] * 3)
    lagrangian = (v1**2 + v2**2 + v3**2) / 2 + amplitude / radius
    defect = engine.invariance_defect(
        lagrangian, sp.Integer(0), [sp.Integer(1), sp.Integer(0), sp.Integer(0)], system
    )
    assert defect != 0
    assert sp.simplify(defect + amplitude * q1 / radius**3) == 0


def test_a_free_particle_conserves_all_three_momenta() -> None:
    _, (v1, v2, v3), (a1, a2, a3) = _jet(["q1", "q2", "q3"])
    system = _system("free", ["q1", "q2", "q3"], [a1, a2, a3])
    lagrangian = (v1**2 + v2**2 + v3**2) / 2
    analysis = engine.noether_analysis(lagrangian, system)
    names = {entry["physical_identification"] for entry in analysis["symmetries"]}
    assert {"momentum_q1", "momentum_q2", "momentum_q3", "minus_energy"} <= names


def test_the_caldirola_kanai_lagrangian_has_no_time_translation_symmetry() -> None:
    gamma = sp.Symbol("gamma", positive=True)
    omega = sp.Symbol("omega", positive=True)
    (q1,), (v1,), (a1,) = _jet(["q1"])
    system = _system(
        "damped_times_factor",
        ["q1"],
        [sp.exp(gamma * _T) * (a1 + gamma * v1 + omega**2 * q1)],
        parameters={"gamma": {"positive": True}, "omega": {"positive": True}},
    )
    lagrangian = sp.exp(gamma * _T) * (v1**2 / 2 - omega**2 * q1**2 / 2)
    assert engine.invariance_defect(lagrangian, sp.Integer(1), [sp.Integer(0)], system) != 0
    assert engine.noether_analysis(lagrangian, system)["symmetries_found"] == 0


def test_the_charge_identification_refuses_a_quantity_it_does_not_match() -> None:
    (q1,), (v1,), _ = _jet(["q1"])
    system = _system("free", ["q1"], [sp.Integer(0)])
    lagrangian = v1**2 / 2
    assert engine.identify_charge(v1, lagrangian, system) == "momentum_q1"
    assert engine.identify_charge(q1 * v1**3 + 7, lagrangian, system) is None


def test_every_reported_symmetry_is_invariant_and_conserved(receipt: dict) -> None:
    total = 0
    for entry in receipt["systems"]:
        if entry["noether"] is None:
            continue
        system = engine.SecondOrderSystem(entry["ir"])
        lagrangian = sp.sympify(
            entry["construction"]["lagrangian"], locals=system.expression_locals()
        )
        on_shell = engine.solve_accelerations(system)
        for symmetry in entry["noether"]["symmetries"]:
            total += 1
            tau = sp.sympify(symmetry["tau"], locals=system.expression_locals())
            xi = [
                sp.sympify(text, locals=system.expression_locals()) for text in symmetry["xi"]
            ]
            assert engine.invariance_defect(lagrangian, tau, xi, system) == 0
            charge = sp.sympify(
                symmetry["conserved_quantity"], locals=system.expression_locals()
            )
            assert sp.simplify(engine.total_derivative(charge, system).subs(on_shell)) == 0
    assert total == receipt["counts"]["conserved_quantities"] == 13


# ---------------------------------------------------------------------------
# Application (a): the blindly recovered planetary law.
# ---------------------------------------------------------------------------


def test_the_bound_blind_receipt_is_hash_and_seal_verified() -> None:
    law = engine.read_recovered_planetary_law(ROOT)
    binding = engine.BOUND_ARTIFACTS["blind_planetary_newton_world_receipt"]
    document = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    assert document["content_sha256"] == canonical_sha256(body)
    assert law["source_semantic_sha256"] == binding["semantic_sha256"]
    assert law["campaign_verdict"] == "REDISCOVERED_EXACT"
    assert law["recovered_before_the_target_was_unsealed"] is True


def test_a_tampered_bound_artifact_fails_closed(monkeypatch) -> None:
    binding = dict(engine.BOUND_ARTIFACTS["blind_planetary_newton_world_receipt"])
    binding["semantic_sha256"] = "0" * 64
    monkeypatch.setitem(engine.BOUND_ARTIFACTS, "blind_planetary_newton_world_receipt", binding)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.read_recovered_planetary_law(ROOT)


def test_the_recovered_amplitude_is_four_pi_squared() -> None:
    law = engine.read_recovered_planetary_law(ROOT)
    constant = sp.sympify(law["constant_text"])
    assert law["exponent"] == -2
    assert abs(sp.N(constant - 4 * sp.pi**2, 60)) < sp.Rational(1, 10**40)
    assert law["constant_matches_four_pi_squared_to_40_places"] is True


def test_the_recovered_law_yields_the_kepler_lagrangian_independently(receipt: dict) -> None:
    entry = _report(receipt, "blind_recovered_inverse_square")
    assert entry["helmholtz"]["verdict"] == "VARIATIONAL"
    law = entry["recovered_law"]
    amplitude = sp.sympify(law["constant_text"])
    (q1, q2, q3), (v1, v2, v3), _ = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    expected = (v1**2 + v2**2 + v3**2) / 2 + amplitude / radius
    system = engine.SecondOrderSystem(entry["ir"])
    reported = sp.sympify(
        entry["construction"]["lagrangian"], locals=system.expression_locals()
    )
    assert sp.simplify(reported - expected) == 0
    assert engine.round_trip_residuals(reported, system) == [0, 0, 0]


def test_the_data_to_theory_chain_runs_end_to_end(receipt: dict) -> None:
    chain = _report(receipt, "blind_recovered_inverse_square")["data_to_theory_chain"]
    assert [step["step"] for step in chain] == [
        "1_data",
        "2_empirical_law",
        "3_declared_lift_to_equations_of_motion",
        "4_helmholtz_test",
        "5_constructed_lagrangian",
        "6_noether_consequences",
    ]
    assert "exact rational rows" in chain[0]["produced"]
    assert "x_response" in chain[1]["produced"]
    assert "assumptions here, not findings" in chain[2]["by"]
    assert chain[3]["produced"].startswith("VARIATIONAL")
    assert "angular_momentum_q1_q2" in chain[5]["produced"]
    assert "minus_energy" in chain[5]["produced"]


def test_the_declared_lift_is_reported_as_an_assumption(receipt: dict) -> None:
    law = _report(receipt, "blind_recovered_inverse_square")["recovered_law"]
    assert "declared modelling assumption the data did not force" in law["declared_lift"]
    assert any(
        "assumes centrality and isotropy" in note
        for note in receipt["residual_gap_report"]["not_established"]
    )


# ---------------------------------------------------------------------------
# Application (b): the screened-gravity candidate.
# ---------------------------------------------------------------------------


def test_the_screened_gravity_binding_matches_the_imported_candidate_config() -> None:
    binding = engine.BOUND_ARTIFACTS["sigma_gravity_candidate_config"]
    assert canonical_sha256(CANDIDATE_CONFIG) == binding["semantic_sha256"]
    assert CANDIDATE_CONFIG["published_parameters"]["h_shape_p"] == "1/2"
    assert CANDIDATE_CONFIG["published_parameters"]["h_shape_q"] == "1"
    assert CANDIDATE_CONFIG["published_parameters"]["C_cluster_published"] == "1"


def test_the_spherical_screened_potential_recomputes_by_hand(receipt: dict) -> None:
    entry = _report(receipt, "sigma_gravity_spherical_quasistatic")
    assert entry["helmholtz"]["verdict"] == "VARIATIONAL"
    amplitude, big_g, mass, dagger = sp.symbols("A_amp G M g_dagger", positive=True)
    newtonian = big_g * mass / _R**2
    shape = sp.sqrt(dagger / newtonian) * dagger / (dagger + newtonian)
    effective = newtonian * (1 + amplitude * shape)
    # Integrating the effective acceleration is the whole content of the construction.
    potential = sp.simplify(sp.integrate(sp.simplify(effective), _R))
    reported = sp.sympify(
        entry["construction"]["radial_potential"],
        locals={"A_amp": amplitude, "G": big_g, "M": mass, "g_dagger": dagger, "r": _R},
    )
    assert sp.simplify(potential - reported) == 0
    assert sp.simplify(sp.diff(reported, _R) - effective) == 0
    assert reported.has(sp.log)


def test_the_screened_lagrangian_reduces_to_newton_when_the_amplitude_vanishes(
    receipt: dict,
) -> None:
    entry = _report(receipt, "sigma_gravity_spherical_quasistatic")
    system = engine.SecondOrderSystem(entry["ir"])
    lagrangian = sp.sympify(
        entry["construction"]["lagrangian"], locals=system.expression_locals()
    )
    amplitude = system.parameters["A_amp"]
    big_g, mass = system.parameters["G"], system.parameters["M"]
    (q1, q2, q3), (v1, v2, v3), _ = _jet(["q1", "q2", "q3"])
    radius = sp.sqrt(q1**2 + q2**2 + q3**2)
    newtonian = (v1**2 + v2**2 + v3**2) / 2 + big_g * mass / radius
    assert sp.simplify(lagrangian.subs(amplitude, 0) - newtonian) == 0


def test_the_screened_arm_predicts_a_flat_rotation_curve_with_v_fourth_in_mass(
    receipt: dict,
) -> None:
    entry = _report(receipt, "sigma_gravity_spherical_quasistatic")
    orbits = entry["consequences"]["circular_orbits"]
    assert orbits["asymptotically_flat_rotation_curve"] is True
    amplitude, big_g, mass, dagger = sp.symbols("A_amp G M g_dagger", positive=True)
    names = {"A_amp": amplitude, "G": big_g, "M": mass, "g_dagger": dagger, "r": _R}
    reported = sp.sympify(orbits["large_radius_limit_of_v_squared"], locals=names)
    assert sp.simplify(reported - amplitude * sp.sqrt(big_g * mass * dagger)) == 0
    fourth = sp.sympify(orbits["fourth_power_of_the_flat_speed"], locals=names)
    assert sp.simplify(fourth - amplitude**2 * big_g * mass * dagger) == 0
    # The independently recomputed circular-speed law really does tend to that limit.
    speed_squared = sp.sympify(orbits["circular_speed_squared"], locals=names)
    assert sp.simplify(sp.limit(speed_squared, _R, sp.oo) - reported) == 0


def test_the_coherence_arm_is_not_variational_and_the_obstruction_is_the_force_curl(
    receipt: dict,
) -> None:
    entry = _report(receipt, "sigma_gravity_meridional_coherence")
    assert entry["helmholtz"]["verdict"] == "NOT_VARIATIONAL"
    assert entry["construction"]["constructed"] is False
    assert entry["noether"] is None
    failing = entry["helmholtz"]["failing_conditions"]
    assert len(failing) == 1
    assert failing[0]["condition"] == "H3"
    assert (failing[0]["index_i"], failing[0]["index_j"]) == (1, 2)

    # Independent derivation: with no velocity dependence, H3 collapses to the planar curl of the
    # force, so the obstruction is exactly d(F_R)/dz - d(F_z)/dR.
    amplitude, big_g, mass, dagger, coherence_scale = sp.symbols(
        "A_amp G M g_dagger xi", positive=True
    )
    cylindrical, height = sp.symbols("qR qz", real=True)
    radius = sp.sqrt(cylindrical**2 + height**2)
    newtonian = big_g * mass / radius**2
    shape = sp.sqrt(dagger / newtonian) * dagger / (dagger + newtonian)
    effective = newtonian * (
        1 + amplitude * (cylindrical / (coherence_scale + cylindrical)) * shape
    )
    curl = sp.simplify(
        sp.diff(effective * cylindrical / radius, height)
        - sp.diff(effective * height / radius, cylindrical)
    )
    reported = sp.sympify(
        failing[0]["residual"],
        locals={
            "A_amp": amplitude,
            "G": big_g,
            "M": mass,
            "g_dagger": dagger,
            "xi": coherence_scale,
            "qR": cylindrical,
            "qz": height,
        },
    )
    assert sp.simplify(curl - reported) == 0
    assert reported != 0


def test_the_coherence_obstruction_vanishes_in_the_spherical_and_midplane_limits(
    receipt: dict,
) -> None:
    entry = _report(receipt, "sigma_gravity_meridional_coherence")
    coherence_scale = sp.Symbol("xi", positive=True)
    height = sp.Symbol("qz", real=True)
    names = {
        "A_amp": sp.Symbol("A_amp", positive=True),
        "G": sp.Symbol("G", positive=True),
        "M": sp.Symbol("M", positive=True),
        "g_dagger": sp.Symbol("g_dagger", positive=True),
        "xi": coherence_scale,
        "qR": sp.Symbol("qR", real=True),
        "qz": height,
    }
    residual = sp.sympify(entry["helmholtz"]["failing_conditions"][0]["residual"], locals=names)
    # xi -> 0 is C -> 1, which is exactly the spherical arm that came out VARIATIONAL.
    assert sp.simplify(residual.subs(coherence_scale, 0)) == 0
    # z = 0 is the mid-plane, where the coherence gradient is orthogonal to the motion.
    assert sp.simplify(residual.subs(height, 0)) == 0


def test_no_multiplier_repairs_the_coherence_arm(receipt: dict) -> None:
    search = _report(receipt, "sigma_gravity_meridional_coherence")["integrating_factor_search"]
    assert search["outcome"] == "NO_MULTIPLIER_IN_DECLARED_SPACE"
    assert search["space_size"] == 61
    assert search["found"] == []
    assert "Douglas multiplier problem" in " ".join(search["space"]["not_searched"])


def test_the_coherence_finding_states_its_scope(receipt: dict) -> None:
    finding = _report(receipt, "sigma_gravity_meridional_coherence")["finding"]
    assert "NOT VARIATIONAL" in finding
    assert "test-particle force law as published" in finding
    assert any(
        "derived functional of the source" in note
        for note in receipt["residual_gap_report"]["not_established"]
    )


# ---------------------------------------------------------------------------
# Controls as a set, and the class boundary as a declared object.
# ---------------------------------------------------------------------------


def test_the_four_declared_controls_land_where_they_are_declared_to(receipt: dict) -> None:
    assert receipt["verdicts"]["newtonian_inverse_square"] == "VARIATIONAL"
    assert receipt["verdicts"]["harmonic_oscillator"] == "VARIATIONAL"
    assert receipt["verdicts"]["damped_oscillator"] == "NOT_VARIATIONAL"
    assert receipt["verdicts"]["nonconservative_planar_curl"] == "NOT_VARIATIONAL"
    assert receipt["verdicts"]["quadratic_in_acceleration"] == "OUT_OF_DECLARED_CLASS"
    for entry in receipt["systems"]:
        assert entry["expectation_check"]["status"] == "pass"
        assert entry["expectation_check"]["blockers"] == []


def test_a_missed_expectation_aborts_the_run() -> None:
    declared = {
        "role": "control",
        "ir": engine.system_ir("oscillator", ["q1"], ["ddq1 + q1"]),
        "expected": {"helmholtz_verdict": "NOT_VARIATIONAL"},
    }
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.analyze_system(declared)


def test_a_missing_conserved_quantity_aborts_the_run() -> None:
    declared = {
        "role": "control",
        "ir": engine.system_ir("oscillator", ["q1"], ["ddq1 + q1"]),
        "expected": {
            "helmholtz_verdict": "VARIATIONAL",
            "required_conserved_quantities": ["momentum_q1"],
        },
    }
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.analyze_system(declared)


def test_the_declared_class_is_stated_before_any_verdict(receipt: dict) -> None:
    declared = receipt["declared_class"]
    assert declared["euler_lagrange_convention"] == "E_i = d/dt(dL/dq'_i) - dL/dq_i"
    assert set(declared["helmholtz_conditions"]) == {"H1", "H2", "H3"}
    assert declared["outside_verdict"].startswith("OUT_OF_DECLARED_CLASS")
    assert len(declared["regularity_requirements"]) == 3
    assert any("field theories" in item for item in declared["outside_this_class"])


def test_the_receipt_names_what_it_did_not_search(receipt: dict) -> None:
    notes = " ".join(receipt["residual_gap_report"]["not_established"])
    assert "structurally automatic" in notes
    assert "Divergence" in notes
    assert "local and star-shaped" in notes
    assert receipt["counts"]["multiplier_searches_run"] == 3
    assert receipt["counts"]["helmholtz_conditions_checked"] == 78


# ---------------------------------------------------------------------------
# Receipt: determinism, claims, bindings, tamper.
# ---------------------------------------------------------------------------


def test_the_receipt_is_deterministic(receipt: dict, sealed: dict) -> None:
    assert receipt == sealed
    assert receipt["content_sha256"] == sealed["content_sha256"]


def test_the_receipt_carries_no_floats_and_the_declared_claims(receipt: dict) -> None:
    engine._no_floats({key: item for key, item in receipt.items() if key != "content_sha256"})
    assert receipt["claims"] == {
        "construction_verified_by_round_trip": True,
        "helmholtz_verdict_is_within_declared_class": True,
        "not_variational_is_a_finding_not_a_failure": True,
        "novelty_claimed": False,
    }
    assert receipt["schema_version"] == engine.RESULT_SCHEMA
    assert receipt["decision"] == "INVERSE_VARIATIONAL_ANALYSIS_COMPLETE_NO_NOVELTY_CLAIMED"


def test_every_system_carries_its_own_reparseable_ir(receipt: dict) -> None:
    for entry in receipt["systems"]:
        assert entry["ir"]["schema_version"] == engine.SYSTEM_IR_SCHEMA
        rebuilt = engine.SecondOrderSystem(entry["ir"])
        assert rebuilt.system_id == entry["system_id"]
        assert rebuilt.size == entry["coordinate_count"]
        assert engine.helmholtz_test(rebuilt)["verdict"] == entry["helmholtz"]["verdict"]


def test_the_counts_replay_from_the_systems(receipt: dict) -> None:
    assert receipt["counts"] == json.loads(json.dumps(engine._counts(receipt["systems"])))
    assert receipt["counts"]["systems"] == 8
    assert receipt["counts"]["controls"] == 5
    assert receipt["counts"]["applications"] == 3
    assert receipt["counts"]["variational"] == 4
    assert receipt["counts"]["not_variational"] == 3
    assert receipt["counts"]["out_of_declared_class"] == 1
    assert receipt["counts"]["multipliers_found"] == 1


def test_a_plain_tamper_is_caught_by_the_seal(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["verdicts"]["sigma_gravity_meridional_coherence"] = "VARIATIONAL"
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_verdict_flip_is_still_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    for entry in tampered["systems"]:
        if entry["system_id"] == "sigma_gravity_meridional_coherence":
            entry["helmholtz"]["verdict"] = "VARIATIONAL"
    tampered["verdicts"]["sigma_gravity_meridional_coherence"] = "VARIATIONAL"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_claim_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["claims"]["novelty_claimed"] = True
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_class_boundary_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["declared_class"]["regularity_requirements"] = []
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_nonzero_round_trip_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    for entry in tampered["systems"]:
        if entry["construction"]["round_trip_verified"]:
            entry["construction"]["round_trip_residual"] = ["1"]
            break
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_missing_witness_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    for entry in tampered["systems"]:
        for failing in entry["helmholtz"]["failing_conditions"]:
            failing["nonvanishing_witness"] = "0"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_unconserved_charge_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    for entry in tampered["systems"]:
        if entry["noether"] and entry["noether"]["symmetries"]:
            entry["noether"]["symmetries"][0]["on_shell_time_derivative"] = "1"
            break
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_count_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["counts"]["multipliers_found"] = 99
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_a_resealed_config_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["config"]["multiplier_space"]["total_degree_bound"] = 9
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(engine.InverseVariationalEngineError):
        engine.validate_receipt(tampered)


def test_the_checked_receipt_validates(sealed: dict) -> None:
    engine.validate_receipt(sealed)


def test_the_cli_validates_the_sealed_receipt(capsys) -> None:
    assert engine.main(["--root", str(ROOT), "--validate-checked"]) == 0
    assert capsys.readouterr().out == ""


def test_the_cli_refuses_to_overwrite_an_immutable_receipt(tmp_path: Path, sealed: dict) -> None:
    target = tmp_path / "engine-v1.json"
    target.write_bytes(b'{"schema_version":"tampered"}\n')
    with pytest.raises(engine.InverseVariationalEngineError):
        engine._write(sealed, target)
