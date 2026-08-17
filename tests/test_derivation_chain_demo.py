"""Gates for the first-principles derivation chain demonstration.

The point of this module is that the chains are *derived*, not transcribed.  So these tests do
not merely read the receipt back: wherever a step claims a symbolic result, the test recomputes
that result with its own independent sympy code and compares.

The tests pin: that solving Laplace's equation really yields ``-A/r`` and that the boundary
condition really kills the additive constant; that the d-dimensional vacuum solution really is
``r^(2-d)`` with force ``r^(1-d)``, so d = 3 is what makes the exponent 2; that Euler-Lagrange on
the declared action really produces the inverse-square acceleration; that circular-orbit balance
really produces Kepler III and that a wrong force exponent destroys it; that the Schwarzschild
geodesic really reduces to ``u'' + u = GM/L^2 + 3 GM u^2/c^2``; that Poincare-Lindstedt really
produces ``6 pi GM/(c^2 a (1-e^2))``; that a perturbed metric really fails ``R_ab = 0``; that
Mercury's advance reproduces the cited 43 arcseconds per century; that each exclusion bound
recomputes from the cited residual budget; and that the receipt is deterministic, float-free,
claim-frozen, and tamper-evident under both a plain edit and a reseal.

The two heavy fixtures are session-scoped because a full chain run does several sympy
``dsolve``/``simplify`` passes and rebuilds two Ricci tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from sigma_theory_compiler import derivation_chain_demo as chain
from sigma_theory_compiler.relativity import C_SI, G_SI, JULIAN_YEAR_DAYS, M_SUN_KG
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def receipt() -> dict:
    return chain.run_derivation_chain(ROOT)


@pytest.fixture(scope="session")
def sealed() -> dict:
    return json.loads((ROOT / chain.RECEIPT_PATH).read_text(encoding="utf-8"))


def _chain(receipt: dict, chain_id: str) -> dict:
    return next(item for item in receipt["chains"] if item["chain_id"] == chain_id)


def _step(receipt: dict, chain_id: str, number: int) -> dict:
    return next(item for item in _chain(receipt, chain_id)["steps"] if item["step"] == number)


# ---------------------------------------------------------------------------
# Chain 1.  The derivation is live.
# ---------------------------------------------------------------------------


def test_solving_laplaces_equation_really_yields_minus_a_over_r(receipt: dict) -> None:
    radius = sp.Symbol("r", positive=True)
    potential = sp.Function("Phi")
    laplacian = sp.diff(radius**2 * sp.diff(potential(radius), radius), radius) / radius**2
    general = sp.dsolve(sp.Eq(sp.expand(laplacian), 0), potential(radius)).rhs

    constants = sorted(general.free_symbols - {radius}, key=lambda symbol: symbol.name)
    assert len(constants) == 2
    additive, scaling = constants
    # The independently solved general solution is an additive constant plus C/r.
    assert sp.simplify(general - additive - scaling / radius) == 0
    # It really solves the vacuum equation.
    assert sp.simplify(laplacian.subs(potential(radius), general).doit()) == 0
    # The boundary condition Phi -> 0 at infinity kills exactly the additive constant.
    assert sp.simplify(sp.limit(general, radius, sp.oo) - additive) == 0

    amplitude = sp.Symbol("A", positive=True)
    derived = general.subs({additive: 0, scaling: -amplitude})
    assert sp.simplify(derived + amplitude / radius) == 0

    step = _step(receipt, "newton_from_variational_principle", 2)
    assert step["derived_potential"] == "-A/r"
    assert step["vacuum_residual"] == "0"
    assert sp.simplify(sp.sympify(step["general_solution"], {"r": radius}) - general) == 0


def test_the_inverse_square_force_falls_out_of_the_solved_potential(receipt: dict) -> None:
    radius = sp.Symbol("r", positive=True)
    amplitude = sp.Symbol("A", positive=True)
    field = -sp.diff(-amplitude / radius, radius)
    assert sp.simplify(field + amplitude / radius**2) == 0

    step = _step(receipt, "newton_from_variational_principle", 2)
    reported = sp.sympify(step["derived_field_strength"], {"r": radius, "A": amplitude})
    assert sp.simplify(reported - field) == 0


def test_euler_lagrange_really_produces_the_inverse_square_acceleration(receipt: dict) -> None:
    time = sp.Symbol("t", real=True)
    amplitude, mass = sp.symbols("A m", positive=True)
    position = [sp.Function(name)(time) for name in ("x", "y", "z")]
    distance = sp.sqrt(sum(component**2 for component in position))
    lagrangian = sp.Rational(1, 2) * mass * sum(
        sp.diff(component, time) ** 2 for component in position
    ) + mass * amplitude / distance

    for component in position:
        equation = sp.diff(lagrangian, component) - sp.diff(
            sp.diff(lagrangian, sp.diff(component, time)), time
        )
        acceleration = sp.solve(sp.Eq(equation, 0), sp.diff(component, time, 2))[0]
        assert sp.simplify(acceleration + amplitude * component / distance**3) == 0

    step = _step(receipt, "newton_from_variational_principle", 3)
    assert step["acceleration_magnitude"] == "A/r^2"
    assert len(step["acceleration_components"]) == 3


def test_kepler_third_law_is_derived_and_a_wrong_exponent_destroys_it(receipt: dict) -> None:
    semi_major, period, amplitude = sp.symbols("a T A", positive=True)

    # Correct inverse-square force: T^2 = 4 pi^2 a^3 / A, an a-independent ratio.
    balance = sp.Eq((2 * sp.pi / period) ** 2 * semi_major, amplitude / semi_major**2)
    squared = sp.simplify(sp.solve(balance, period)[0] ** 2)
    assert sp.simplify(squared - 4 * sp.pi**2 * semi_major**3 / amplitude) == 0
    assert sp.simplify(sp.diff(squared / semi_major**3, semi_major)) == 0

    # Negative control: any other exponent leaves an explicit a-dependence in T^2/a^3.
    for exponent in (1, 3, 4):
        wrong = sp.Eq((2 * sp.pi / period) ** 2 * semi_major, amplitude / semi_major**exponent)
        wrong_squared = sp.simplify(sp.solve(wrong, period)[0] ** 2)
        ratio = sp.simplify(wrong_squared / semi_major**3)
        assert sp.simplify(sp.diff(ratio, semi_major)) != 0, exponent

    step = _step(receipt, "newton_from_variational_principle", 4)
    assert step["kepler_ratio"] == "4*pi**2/A"
    assert step["kepler_exponent"] == "3/2 power of the semi-major axis"
    verdicts = {
        row["force_exponent"]: row["kepler_third_law_holds"] for row in step["exponent_scan"]
    }
    assert verdicts == {"1": "no", "2": "yes", "3": "no", "4": "no"}


def test_the_dimension_generalization_gives_r_to_the_two_minus_d(receipt: dict) -> None:
    radius = sp.Symbol("r", positive=True)
    dimension = sp.Symbol("d", positive=True, integer=True)
    potential = sp.Function("Phi")
    laplacian = sp.simplify(
        sp.diff(radius ** (dimension - 1) * sp.diff(potential(radius), radius), radius)
        / radius ** (dimension - 1)
    )
    general = sp.simplify(sp.dsolve(sp.Eq(laplacian, 0), potential(radius)).rhs)
    constants = sorted(
        general.free_symbols - {radius, dimension}, key=lambda symbol: symbol.name
    )
    additive = next(
        symbol for symbol in constants if sp.simplify(sp.diff(general, symbol)) == 1
    )
    scaling = next(symbol for symbol in constants if symbol is not additive)

    # Potential goes like r^(2-d); the field strength therefore goes like r^(1-d).
    assert sp.simplify(sp.diff(general, scaling) - radius ** (2 - dimension)) == 0
    field = -sp.diff(general, radius)
    assert sp.simplify(sp.diff(field, scaling) - (dimension - 2) * radius ** (1 - dimension)) == 0

    # d = 3 is what makes the force exponent 2, and d = 2 is logarithmic rather than a power.
    assert sp.simplify((2 - dimension).subs(dimension, 3) + 1) == 0
    assert sp.simplify((1 - dimension).subs(dimension, 3) + 2) == 0

    step = _step(receipt, "newton_from_variational_principle", 5)
    assert step["potential_exponent"] == "2 - d"
    assert step["force_exponent"] == "1 - d"
    samples = {item["dimension"]: item for item in step["dimension_samples"]}
    assert "log(r)" in samples["2"]["general_solution"]
    assert samples["3"]["general_solution"] == "C1 + C2/r"
    assert samples["4"]["general_solution"] == "C1 + C2/r**2"
    for item in samples.values():
        assert item["vacuum_residual"] == "0", item["dimension"]


# ---------------------------------------------------------------------------
# Chain 2.  General Relativity.
# ---------------------------------------------------------------------------


def test_the_schwarzschild_geodesic_really_gives_the_orbit_equation(receipt: dict) -> None:
    angle = sp.Symbol("phi", real=True)
    inverse_radius = sp.Function("u")(angle)
    momentum, energy, light, newton_mass = sp.symbols("L E c GM", positive=True)

    # Independent re-derivation: differentiate the four-velocity normalization first integral.
    first_integral = (
        momentum**2 * sp.diff(inverse_radius, angle) ** 2
        - energy**2 / light**2
        + (1 - 2 * newton_mass * inverse_radius / light**2)
        * (light**2 + momentum**2 * inverse_radius**2)
    )
    reduced = sp.simplify(
        sp.expand(sp.diff(first_integral, angle))
        / (2 * momentum**2 * sp.diff(inverse_radius, angle))
    )
    target = (
        sp.diff(inverse_radius, angle, 2)
        + inverse_radius
        - newton_mass / momentum**2
        - 3 * newton_mass * inverse_radius**2 / light**2
    )
    assert sp.simplify(sp.expand(reduced - target)) == 0

    step = _step(receipt, "general_relativity_action_to_perihelion", 4)
    assert step["symbolic_result"] == "u'' + u = GM/L^2 + 3*GM*u^2/c^2"
    assert step["orbit_equation_residual"] == "0"
    assert step["newtonian_limit"] == "u'' + u = GM/L^2"


def test_perturbation_theory_really_gives_six_pi_gm_over_c2_a_one_minus_e2(receipt: dict) -> None:
    psi = sp.Symbol("psi", real=True)
    eccentricity, epsilon = sp.symbols("e epsilon", positive=True)
    detuning = sp.Symbol("k1", real=True)
    correction = sp.Function("w1")(psi)
    momentum, light, newton_mass, semi_major = sp.symbols("L c GM a", positive=True)

    # Independent Poincare-Lindstedt: k^2 w'' + w = 1 + epsilon w^2, k^2 = 1 + epsilon k1.
    zeroth = 1 + eccentricity * sp.cos(psi)
    residual = (
        (1 + epsilon * detuning) * (sp.diff(zeroth, psi, 2) + epsilon * sp.diff(correction, psi, 2))
        + (zeroth + epsilon * correction)
        - 1
        - epsilon * (zeroth + epsilon * correction) ** 2
    )
    order_one = sp.expand(
        sp.expand_trig(sp.expand(sp.series(residual, epsilon, 0, 2).removeO().coeff(epsilon, 1)))
    )
    detuning_value = sp.solve(sp.Eq(order_one.coeff(sp.cos(psi), 1), 0), detuning)[0]
    assert detuning_value == -2

    advance = sp.simplify(
        sp.series(
            2 * sp.pi / sp.sqrt(1 + epsilon * detuning_value) - 2 * sp.pi, epsilon, 0, 2
        ).removeO()
    )
    assert sp.simplify(advance - 2 * sp.pi * epsilon) == 0

    in_elements = advance.subs(
        epsilon, 3 * newton_mass**2 / (light**2 * momentum**2)
    ).subs(momentum**2, newton_mass * semi_major * (1 - eccentricity**2))
    expected = 6 * sp.pi * newton_mass / (light**2 * semi_major * (1 - eccentricity**2))
    assert sp.simplify(in_elements - expected) == 0

    step = _step(receipt, "general_relativity_action_to_perihelion", 5)
    assert step["advance_in_orbital_elements"] == "6*pi*GM/(c^2*a*(1 - e^2))"
    assert step["advance_residual_against_closed_form"] == "0"
    assert step["detuning"] == "-2"


def test_the_schwarzschild_metric_is_ricci_flat_and_a_wrong_metric_is_not(receipt: dict) -> None:
    step = _step(receipt, "general_relativity_action_to_perihelion", 3)
    assert step["components_calculated"] == 16
    assert step["nonzero_components"] == []
    assert step["provenance"] == "existing_repository_control"
    assert step["certified_by"] == "sigma_theory_compiler.relativity.schwarzschild_ricci_components"

    # Negative control: a perturbed lapse is not a vacuum solution.
    perturbed = chain.perturbed_schwarzschild_ricci()
    nonzero = {name: value for name, value in perturbed.items() if value != "0"}
    assert nonzero, "a non-Schwarzschild metric must not pass the vacuum check"
    assert set(step["negative_control_nonzero_components"]) == set(nonzero)


@pytest.mark.empirical_validation
def test_mercury_advance_reproduces_the_cited_forty_three_arcseconds(receipt: dict) -> None:
    mp.mp.dps = chain.WORKING_PRECISION_DIGITS
    elements = chain.CITED_OBSERVATIONS["mercury_orbital_elements"]
    semi_major = mp.mpf(elements["semi_major_axis_m"])
    eccentricity = mp.mpf(elements["eccentricity"])
    period_days = mp.mpf(elements["sidereal_period_days"])

    # Independent numeric recomputation of 6 pi GM/(c^2 a (1-e^2)) per orbit.
    advance = 6 * mp.pi * mp.mpf(G_SI) * mp.mpf(M_SUN_KG) / (
        mp.mpf(C_SI) ** 2 * semi_major * (1 - eccentricity**2)
    )
    per_century = advance * (100 * mp.mpf(JULIAN_YEAR_DAYS) / period_days) * (648000 / mp.pi)
    assert 42.5 < float(per_century) < 43.5

    cited = mp.mpf(
        chain.CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"][
            "value_arcsec_per_century"
        ]
    )
    agreement = mp.mpf(
        chain.CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"][
            "fractional_agreement_with_general_relativity"
        ]
    )
    assert abs(per_century - cited) / cited < agreement

    verification = receipt["numeric_verifications"][0]
    assert verification["verification_id"] == "mercury_perihelion_advance"
    assert float(verification["derived_arcsec_per_century"]) == pytest.approx(
        float(per_century), rel=1e-11
    )
    assert float(verification["relative_error"]) < float(agreement)


def test_the_recomputation_agrees_with_the_existing_relativity_control(receipt: dict) -> None:
    crosscheck = receipt["numeric_verifications"][1]
    assert crosscheck["certified_by"] == (
        "sigma_theory_compiler.relativity.solar_system_numeric_checks"
    )
    assert float(crosscheck["relative_difference"]) < float(crosscheck["tolerance"])


# ---------------------------------------------------------------------------
# Chain 3.  The alternatives and their exclusion.
# ---------------------------------------------------------------------------


def test_each_alternative_precession_is_rederived_independently(receipt: dict) -> None:
    radius, orbit_radius = sp.symbols("r a", positive=True)
    newton_mass, screening = sp.symbols("GM lamda", positive=True)
    compton = sp.Symbol("lambda_g", positive=True)
    delta = sp.Symbol("delta", real=True)
    alpha = sp.Symbol("alpha", real=True)

    def advance(force: sp.Expr) -> sp.Expr:
        slope = sp.simplify(
            orbit_radius * sp.diff(force, radius).subs(radius, orbit_radius)
            / force.subs(radius, orbit_radius)
        )
        return sp.simplify(2 * sp.pi / sp.sqrt(3 + slope) - 2 * sp.pi)

    # (a) modified exponent -> pi*delta at leading order.
    power = advance(newton_mass / radius ** (2 + delta))
    assert sp.simplify(sp.series(power, delta, 0, 2).removeO() - sp.pi * delta) == 0

    # (b) Yukawa -> pi*alpha*(a/lambda)^2*exp(-a/lambda).
    yukawa_potential = -(newton_mass / radius) * (1 + alpha * sp.exp(-radius / screening))
    yukawa_force = -sp.diff(yukawa_potential, radius)
    yukawa = advance(sp.simplify(yukawa_force))
    expected_yukawa = (
        sp.pi * alpha * orbit_radius**2 * sp.exp(-orbit_radius / screening) / screening**2
    )
    assert sp.simplify(sp.series(yukawa, alpha, 0, 2).removeO() - expected_yukawa) == 0

    # (c) graviton mass -> pi*(a/lambda_g)^2.
    graviton_force = -sp.diff(-(newton_mass / radius) * sp.exp(-radius / compton), radius)
    ratio = sp.Symbol("x", positive=True)
    graviton = advance(sp.simplify(graviton_force)).subs(compton, orbit_radius / ratio)
    assert sp.simplify(sp.series(graviton, ratio, 0, 3).removeO() - sp.pi * ratio**2) == 0

    derived = _chain(receipt, "alternatives_excluded_by_measurement")["derived_precessions"]
    assert derived["modified_exponent"]["advance_leading_order"] == "pi*delta"
    assert derived["graviton_mass"]["advance_leading_order_in_x"] == "pi*x**2"


def test_the_apsidal_machinery_reproduces_the_chain_two_result(receipt: dict) -> None:
    # The Chain 3 method, fed Chain 2's effective force, must return Chain 2's own answer.
    radius, orbit_radius = sp.symbols("r a", positive=True)
    newton_mass, light, momentum = sp.symbols("GM c L", positive=True)
    force = newton_mass / radius**2 + 3 * newton_mass * momentum**2 / (light**2 * radius**4)
    advance, _ = chain.apsidal_precession(force, radius, orbit_radius)
    leading = sp.simplify(
        sp.series(advance.subs(momentum**2, newton_mass * orbit_radius), light, sp.oo, 3).removeO()
    )
    assert sp.simplify(leading - 6 * sp.pi * newton_mass / (light**2 * orbit_radius)) == 0

    crosscheck = _chain(receipt, "alternatives_excluded_by_measurement")["derived_precessions"][
        "crosscheck_against_chain_two"
    ]
    assert crosscheck["residual"] == "0"
    assert crosscheck["status"] == "pass"


def test_the_exclusion_bounds_recompute_from_the_residual_budget(receipt: dict) -> None:
    mp.mp.dps = chain.WORKING_PRECISION_DIGITS
    bounds = receipt["exclusion_bounds"]
    residual = mp.mpf(bounds["residual_budget_rad_per_orbit"])
    semi_major = mp.mpf(
        chain.CITED_OBSERVATIONS["mercury_orbital_elements"]["semi_major_axis_m"]
    )

    # The budget itself: 0.1 percent of the cited 42.98 arcsec/century, per orbit.
    cited = chain.CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"]
    orbits = 100 * mp.mpf(JULIAN_YEAR_DAYS) / mp.mpf(
        chain.CITED_OBSERVATIONS["mercury_orbital_elements"]["sidereal_period_days"]
    )
    budget = (
        mp.mpf(cited["fractional_agreement_with_general_relativity"])
        * mp.mpf(cited["value_arcsec_per_century"])
        / (orbits * (648000 / mp.pi))
    )
    assert abs(budget - residual) / residual < mp.mpf("1e-11")

    # (a) |delta| < budget/pi
    assert abs(residual / mp.pi - mp.mpf(bounds["modified_exponent"]["absolute_upper_bound"])) / (
        residual / mp.pi
    ) < mp.mpf("1e-11")

    # (b) |alpha| < budget*exp(x)/(pi*x^2) at each declared sample point.
    for point in bounds["yukawa"]["exclusion_curve"]:
        x = 1 / mp.mpf(point["lambda_over_semi_major_axis"])
        expected = residual * mp.exp(x) / (mp.pi * x**2)
        assert abs(expected - mp.mpf(point["alpha_absolute_upper_bound"])) / expected < mp.mpf(
            "1e-11"
        )
    # exp(x)/x^2 is minimized at x = 2, so that sample must be the strongest constraint.
    strongest = min(
        bounds["yukawa"]["exclusion_curve"],
        key=lambda point: mp.mpf(point["alpha_absolute_upper_bound"]),
    )
    assert strongest["lambda_over_semi_major_axis"] == "0.5"
    assert strongest["alpha_absolute_upper_bound"] == (
        bounds["yukawa"]["tightest_point"]["alpha_absolute_upper_bound"]
    )

    # (c) lambda_g > a*sqrt(pi/budget), and the mass bound is its unit conversion.
    compton = semi_major * mp.sqrt(mp.pi / residual)
    reported = mp.mpf(bounds["graviton_mass"]["compton_wavelength_lower_bound_m"])
    assert abs(compton - reported) / compton < mp.mpf("1e-11")
    planck = mp.mpf(chain.CITED_OBSERVATIONS["defining_si_constants"]["planck_constant_J_s"])
    electron_volt = mp.mpf(chain.CITED_OBSERVATIONS["defining_si_constants"]["electron_volt_J"])
    mass_ev = planck * mp.mpf(C_SI) / (reported * electron_volt)
    assert abs(mass_ev - mp.mpf(bounds["graviton_mass"]["mass_upper_bound_eV"])) / mass_ev < mp.mpf(
        "1e-11"
    )
    # Sanity: these are the familiar solar-system orders of magnitude.
    assert mp.mpf("1e-10") < mp.mpf(bounds["modified_exponent"]["absolute_upper_bound"]) < mp.mpf(
        "1e-9"
    )
    assert mp.mpf("1e15") < reported < mp.mpf("1e16")
    assert mp.mpf("1e-22") < mass_ev < mp.mpf("1e-21")


# ---------------------------------------------------------------------------
# Receipt: determinism, claims, bindings, tamper.
# ---------------------------------------------------------------------------


def test_the_receipt_is_deterministic(receipt: dict, sealed: dict) -> None:
    assert receipt["content_sha256"] == sealed["content_sha256"]
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert chain.run_derivation_chain(ROOT)["content_sha256"] == receipt["content_sha256"]


def test_the_receipt_carries_no_floats_and_the_declared_claims(receipt: dict) -> None:
    chain._no_floats(receipt)
    assert receipt["claims"] == {
        "derivation_is_symbolic_and_checked": True,
        "novelty_claimed": False,
        "published_values_cited_not_fitted": True,
        "real_observational_data_opened": False,
        "reuses_existing_verified_controls": True,
    }
    assert receipt["claims"]["real_observational_data_opened"] is False
    assert receipt["claims"]["novelty_claimed"] is False


def test_every_step_is_attributed_to_new_or_reused_machinery(receipt: dict) -> None:
    counts = receipt["counts"]
    assert counts["chains"] == 3
    assert counts["steps_total"] == 13
    assert counts["steps_reusing_existing_controls"] == 3
    assert counts["steps_new_in_this_module"] == 10
    reused = [
        step["certified_by"]
        for item in receipt["chains"]
        for step in item["steps"]
        if step["provenance"] == "existing_repository_control"
    ]
    assert any("action-ir.json" in name for name in reused)
    assert any(chain.REQUIRED_VARIATION_CONTROL in name for name in reused)
    assert any("schwarzschild_ricci_components" in name for name in reused)


def test_a_changed_bound_artifact_fails_closed(tmp_path: Path, monkeypatch) -> None:
    poisoned = dict(chain.BOUND_ARTIFACTS)
    poisoned["relativity_module"] = {
        **chain.BOUND_ARTIFACTS["relativity_module"],
        "normalized_text_sha256": "0" * 64,
    }
    monkeypatch.setattr(chain, "BOUND_ARTIFACTS", poisoned)
    with pytest.raises(chain.DerivationChainDemoError, match="hash mismatch"):
        chain.run_derivation_chain(ROOT)


def test_a_missing_variation_certificate_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(chain, "REQUIRED_VARIATION_CONTROL", "not_a_registered_control")
    with pytest.raises(chain.DerivationChainDemoError, match="failed its check"):
        chain.run_derivation_chain(ROOT)


def test_a_non_discriminating_kepler_scan_aborts_the_run(monkeypatch) -> None:
    # The Kepler negative control gates its own step: if the scan stops discriminating between
    # force exponents, step 4 must not be allowed to report a pass.
    monkeypatch.setattr(chain, "_kepler_exponent_scan", list)
    with pytest.raises(chain.DerivationChainDemoError, match="failed its check"):
        chain.newton_chain()

    monkeypatch.setattr(
        chain,
        "_kepler_exponent_scan",
        lambda: [
            {"force_exponent": str(value), "kepler_third_law_holds": "yes"}
            for value in (1, 2, 3, 4)
        ],
    )
    with pytest.raises(chain.DerivationChainDemoError, match="failed its check"):
        chain.newton_chain()


def test_a_plain_tamper_is_caught_by_the_seal(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["exclusion_bounds"]["modified_exponent"]["absolute_upper_bound"] = "1.0"
    with pytest.raises(chain.DerivationChainDemoError, match="receipt seal changed"):
        chain.validate_receipt(tampered)


def test_a_reseal_after_tamper_is_still_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["counts"]["steps_new_in_this_module"] = 99
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(chain.DerivationChainDemoError, match="aggregate counts do not replay"):
        chain.validate_receipt(tampered)


def test_a_resealed_claim_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["claims"]["novelty_claimed"] = True
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(chain.DerivationChainDemoError, match="claims block changed"):
        chain.validate_receipt(tampered)


def test_a_resealed_failing_step_promoted_to_pass_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["chains"][0]["steps"][0]["check_status"] = "fail"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(chain.DerivationChainDemoError, match="not marked pass"):
        chain.validate_receipt(tampered)


def test_a_resealed_config_change_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["config"]["cited_observations"]["mercury_anomalous_perihelion_advance"][
        "fractional_agreement_with_general_relativity"
    ] = "1e-6"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(chain.DerivationChainDemoError, match="config binding changed"):
        chain.validate_receipt(tampered)


def test_the_checked_receipt_validates(sealed: dict) -> None:
    chain.validate_receipt(sealed)


def test_the_cli_validates_the_sealed_receipt(capsys) -> None:
    assert chain.main(["--root", str(ROOT), "--validate-checked"]) == 0
    assert capsys.readouterr().out == ""


def test_the_cli_refuses_to_overwrite_an_immutable_receipt(tmp_path: Path, sealed: dict) -> None:
    target = tmp_path / "chain-v1.json"
    chain._write(sealed, target)
    chain._write(sealed, target)
    with pytest.raises(chain.DerivationChainDemoError, match="immutable receipt"):
        chain._write({**sealed, "decision": "changed"}, target)
