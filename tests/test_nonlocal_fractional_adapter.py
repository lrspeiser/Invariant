"""Gates for the nonlocal ``(-Box)^alpha`` auxiliary-field localization adapter.

The tests pin: that all four known-answer controls fire *through this module's own adapter*
(a local d'Alembertian is a no-op that reproduces the un-adapted ladder rung for rung, the
Deser-Woodard ``1/Box`` operator reproduces its known one-auxiliary-field ghost, a
deliberately ghost-laden ``(-Box)^2`` rejects, and a declared spectral subtraction whose sign
flips between ``N = 4`` and ``N = 8`` is classified ``UNSTABLE_UNDER_LOCALIZATION``); that a
broken control aborts the run; that every verdict class is reachable; that the
convergence-study structure is enforced, so a receipt claiming stability over fewer than the
declared pole counts fails validation; that the emitted localization parameters are exact
where the weight admits closed form and carry a declared precision where it does not; that
the Gauss-Jacobi rule is exact on every moment up to degree ``2N-1`` and reproduces
``mu^-beta`` exactly at the reference scale; determinism, seal tamper, binding to the v3
formal-ladder receipt hash, the no-float rule, and the CLI.

The sealed run over all 71 families is shared through a session fixture because it re-runs the
v3 ladder's screening-sector rungs and its canonical-scalar control on an interval-certified
FLRW trajectory.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
import sympy as sp

from sigma_theory_compiler import nonlocal_fractional_adapter as adapter
from sigma_theory_compiler import v3_family_formal_ladder as ladder
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = adapter.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def receipt() -> dict:
    return adapter.run_adapter(ROOT)


@pytest.fixture(scope="session")
def sealed() -> dict:
    return json.loads((ROOT / adapter.RECEIPT_PATH).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Binding: this adapter only exists to discharge a blocker another receipt named.
# ---------------------------------------------------------------------------


def test_the_ladder_receipt_names_this_blocker_on_every_family() -> None:
    value = adapter.load_ladder_receipt(ROOT)
    assert adapter.DISCHARGED_BLOCKER in value["config"]["blockers"]
    assert value["counts"]["blocked_by_adapter"][adapter.DISCHARGED_BLOCKER] == 71
    assert value["counts"]["families_in"] == 71
    assert all(
        adapter.DISCHARGED_BLOCKER in family["materialization"]["full_lift_blockers"]
        for family in value["families"]
    )


def test_a_ladder_receipt_whose_seal_does_not_replay_is_refused(tmp_path) -> None:
    value = json.loads((ROOT / adapter.LADDER_RECEIPT_PATH).read_text(encoding="utf-8"))
    value["decision"] = "tampered"
    target = tmp_path / adapter.LADDER_RECEIPT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="seal does not replay"):
        adapter.load_ladder_receipt(tmp_path)


def test_the_receipt_binds_the_ladder_receipt_content_hash(receipt: dict) -> None:
    ladder_receipt = json.loads(
        (ROOT / adapter.LADDER_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    binding = receipt["inputs"]["formal_ladder_receipt"]
    assert binding["content_sha256"] == ladder_receipt["content_sha256"]
    assert binding["config_sha256"] == ladder_receipt["config_sha256"]
    assert binding["blocked_on_this_adapter"] == binding["families_in"] == 71


def test_alpha_is_read_off_the_lift_and_cross_checked_against_the_kernel_axis() -> None:
    screen = ladder.load_screen_receipt(ROOT)
    families = ladder.load_representatives(ROOT, screen)
    for family in families:
        alpha, component = adapter.read_declared_alpha(family)
        assert alpha == 1 - Fraction(str(family["representative_values"]["t"])) / 2
        assert component["mechanism"] == "nonlocal_propagator_correction"
    forged = json.loads(json.dumps(families[0]))
    forged["representative_values"]["t"] = "7"
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="disagrees with 1 - t/2"):
        adapter.read_declared_alpha(forged)


# ---------------------------------------------------------------------------
# The quadrature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pole_count", adapter.DECLARED_POLE_COUNTS)
def test_the_half_power_rule_is_closed_form_gauss_chebyshev(pole_count: int) -> None:
    rule = adapter.gauss_jacobi_rule(Fraction(1, 2), pole_count)
    assert rule["exactness"] == "closed_form_gauss_chebyshev"
    assert len(rule["exact_nodes"]) == pole_count
    expected = [
        str(sp.cos(sp.pi * (2 * index - 1) / (2 * pole_count)))
        for index in range(1, pole_count + 1)
    ]
    assert rule["exact_nodes"] == expected
    assert rule["exact_weights"] == [str(sp.pi / pole_count)] * pole_count


@pytest.mark.parametrize("beta", ["1/2", "3/4"])
@pytest.mark.parametrize("pole_count", adapter.DECLARED_POLE_COUNTS)
def test_the_rule_is_exact_on_every_moment_up_to_degree_two_n_minus_one(
    beta: str, pole_count: int
) -> None:
    rule = adapter.gauss_jacobi_rule(Fraction(beta), pole_count)
    bound = rule["moment_residual_bound_up_to_degree_2N_minus_1"]
    assert bound == "0" or int(bound.split("e")[1]) <= -40
    assert all(weight > 0 for weight in rule["_weights"])
    assert all(-1 < node < 1 for node in rule["_nodes"])


def test_the_localization_reproduces_the_operator_exactly_at_the_reference_scale() -> None:
    for alpha in ("1/2", "3/4", "3/2"):
        for pole_count in adapter.DECLARED_POLE_COUNTS:
            block = adapter.localize(
                Fraction(alpha), Fraction(1, 144), Fraction(11, 4), pole_count
            )
            bound = block["propagator_reproduction"]["exact_at_reference_scale_residual_bound"]
            assert bound == "0" or int(bound.split("e")[1]) <= -40


def test_the_localization_converges_pointwise_with_the_pole_count() -> None:
    errors = []
    for pole_count in adapter.DECLARED_POLE_COUNTS:
        block = adapter.localize(
            Fraction(1, 2), Fraction(1, 144), Fraction(11, 4), pole_count
        )
        row = block["propagator_reproduction"]["rows"][0]
        assert row["k_squared_over_mu"] == "1/100"
        errors.append(float(row["relative_error_decimal"]))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0] / 100


def test_an_out_of_range_fractional_exponent_is_refused() -> None:
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="0 < beta < 1"):
        adapter.gauss_jacobi_rule(Fraction(0), 4)
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="at least one pole"):
        adapter.gauss_jacobi_rule(Fraction(1, 2), 0)


# ---------------------------------------------------------------------------
# The physics of the localized arm.
# ---------------------------------------------------------------------------


def test_a_local_dalembertian_introduces_no_auxiliary_field() -> None:
    block = adapter.localize(Fraction(1), Fraction(1), Fraction(1), 8)
    assert block["route"] == adapter.ROUTE_LOCAL
    assert block["is_approximation"] is False
    assert block["auxiliary_field_count"] == "1"
    assert block["ghost_pole_count"] == "0"
    assert block["kinetic_matrix"] == "Matrix([[1]])"


@pytest.mark.parametrize("pole_count", adapter.DECLARED_POLE_COUNTS)
def test_alpha_above_one_puts_a_ghost_on_every_massive_pole(pole_count: int) -> None:
    block = adapter.localize(Fraction(3, 2), Fraction(1, 144), Fraction(11, 4), pole_count)
    assert block["ghost_pole_count"] == str(pole_count)
    assert block["residue_sign_sequence"] == "+" + "-" * pole_count
    assert block["ghost_kind"] == adapter.GHOST_PROPAGATOR
    # The exact Kallen-Lehmann sum rule is what makes the ghost N-independent, and therefore
    # a statement about the declared operator rather than about the quadrature.
    assert block["residue_sum_rule"]["is_zero"] is True
    assert block["ghost_localization_artifact_possible"] is False
    assert block["principal_symbol"]["ghost_free"] is False
    assert adapter.arm_rung_statuses(block)["statuses"]["ghost_freedom"] == "reject"


@pytest.mark.parametrize("alpha", ["1/2", "3/4"])
@pytest.mark.parametrize("pole_count", adapter.DECLARED_POLE_COUNTS)
def test_alpha_below_one_has_a_positive_spectral_density(alpha: str, pole_count: int) -> None:
    block = adapter.localize(
        Fraction(alpha), Fraction(1, 144), Fraction(11, 4), pole_count
    )
    assert block["ghost_pole_count"] == "0"
    assert block["residue_sign_sequence"] == "+" * pole_count
    assert block["residue_sum_rule"]["is_zero"] is False
    assert block["principal_symbol"]["ghost_free"] is True
    assert adapter.arm_rung_statuses(block)["statuses"]["ghost_freedom"] == "pass"


def test_the_finite_pole_total_spectral_weight_grows_with_the_pole_count() -> None:
    weights = [
        float(
            adapter.localize(Fraction(1, 2), Fraction(1), Fraction(1), pole_count)[
                "total_spectral_weight_decimal"
            ]
        )
        for pole_count in adapter.DECLARED_POLE_COUNTS
    ]
    # The exact operator's total spectral weight diverges, so a finite-pole approximant is a
    # different theory in the ultraviolet however large N is.  That is why alpha < 1 passes are
    # conditional and never a proof about the nonlocal limit.
    assert weights == sorted(weights)
    assert weights[-1] > weights[0]


def test_an_integer_power_above_two_is_a_typed_blocker_not_a_guess() -> None:
    block = adapter.localize(Fraction(3), Fraction(1), Fraction(1), 4)
    assert block["status"] == "blocked"
    assert block["blocker"] == "missing_adapter:integer_order_localization_above_quadratic"
    assert block["blocker"] in adapter.BLOCKERS
    statuses = adapter.arm_rung_statuses(block)["statuses"]
    assert set(statuses.values()) == {"blocked"}


def test_a_nonpositive_amplitude_or_scale_is_refused() -> None:
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="amplitude must be"):
        adapter.localize(Fraction(1, 2), Fraction(1), Fraction(0), 4)
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="mass scale must be"):
        adapter.localize(Fraction(1, 2), Fraction(0), Fraction(1), 4)


# ---------------------------------------------------------------------------
# Exact parameter emission.
# ---------------------------------------------------------------------------


def test_the_chebyshev_route_emits_exact_algebraic_masses_and_couplings() -> None:
    block = adapter.localize(Fraction(3, 2), Fraction(1, 144), Fraction(11, 4), 2)
    masses = [mode["mass_squared_exact"] for mode in block["modes"]]
    assert masses[0] == "0"
    for text in masses[1:]:
        parsed = sp.sympify(text)
        assert not parsed.atoms(sp.Float)
        assert parsed.is_rational is not True  # genuinely algebraic, not silently rounded
        assert sp.sqrt(2) in parsed.atoms(sp.Pow) or "sqrt(2)" in text
    for mode in block["modes"]:
        assert float(mode["mass_squared_decimal"]) == pytest.approx(
            float(sp.N(sp.sympify(mode["mass_squared_exact"]), 30)), rel=1e-20, abs=1e-30
        )


def test_the_generic_jacobi_route_declares_its_precision_and_its_minimal_polynomial() -> None:
    rule = adapter.gauss_jacobi_rule(Fraction(3, 4), 4)
    assert rule["exactness"] == "algebraic_root_of_the_declared_jacobi_polynomial"
    assert rule["precision_digits"] == str(adapter.EMITTED_PRECISION_DIGITS)
    polynomial = sp.Poly(sp.sympify(rule["minimal_polynomial"]), sp.Symbol("y"))
    assert polynomial.degree() == 4
    assert all(coefficient.is_rational for coefficient in polynomial.all_coeffs())
    block = adapter.localize(Fraction(3, 4), Fraction(1), Fraction(1), 4)
    assert block["quadrature"]["precision_digits"] == str(adapter.EMITTED_PRECISION_DIGITS)
    for mode in block["modes"]:
        digits = (
            mode["mass_squared_decimal"]
            .split("e")[0]
            .replace("-", "")
            .replace(".", "")
            .lstrip("0")
        )
        assert len(digits) <= adapter.EMITTED_PRECISION_DIGITS
        assert "_exact_nodes" not in block.get("quadrature", {})


def test_the_reference_scale_is_the_declared_kernel_length(receipt: dict) -> None:
    for entry in receipt["families"]:
        localization = receipt["localizations"][entry["localization_id"]]
        length = Fraction(entry["kernel_parameters"]["L2"])
        assert localization["reference_mass_squared"] == str(1 / (length * length))
        assert localization["arm_amplitude"] == entry["kernel_parameters"]["w_power"]
        assert localization["alpha"] == entry["alpha"]


# ---------------------------------------------------------------------------
# Known-answer controls.
# ---------------------------------------------------------------------------


def test_every_declared_control_fires(receipt: dict) -> None:
    controls = receipt["controls"]
    for name, declared in sorted(adapter.CONTROL_CASES.items()):
        assert controls[name]["observed_stability"] == declared["expect_stability"]
        assert sorted(controls[name]["per_pole_count_verdicts"]) == sorted(
            str(item) for item in adapter.DECLARED_POLE_COUNTS
        )


def test_a_local_theory_through_the_adapter_reproduces_the_unadapted_verdict(
    receipt: dict,
) -> None:
    control = receipt["controls"]["local_dalembertian_passthrough"]
    assert control["unadapted_ladder_verdict"] == "FORMAL_PASS"
    assert control["observed_stability"] == "STABLE_PASS"
    assert set(control["unadapted_rung_statuses"].values()) == {"pass"}
    assert set(control["per_pole_count_verdicts"].values()) == {"LOCALIZED_PASS"}
    assert set(control["per_pole_count_ghost_poles"].values()) == {"0"}


def test_the_deser_woodard_control_reproduces_its_known_ghost_structure() -> None:
    block = adapter.localize(Fraction(-1), Fraction(1), Fraction(1), 4)
    assert block["route"] == adapter.ROUTE_INVERSE
    assert block["is_approximation"] is False
    assert [mode["id"] for mode in block["modes"]] == ["psi", "xi"]
    assert block["auxiliary_field_count"] == "2"
    assert block["ghost_pole_count"] == "1"
    assert block["ghost_kind"] == adapter.GHOST_AUXILIARY
    assert block["undiagonalized_kinetic_matrix"] == "Matrix([[0, -1], [-1, 0]])"
    assert sp.Matrix([[0, -1], [-1, 0]]).eigenvals() == {sp.Integer(-1): 1, sp.Integer(1): 1}
    # The whole point of the Deser-Woodard case: this ghost may be an artifact of the
    # localized formulation, and the receipt has to say so rather than count it as a result.
    assert block["ghost_localization_artifact_possible"] is True
    assert adapter.arm_rung_statuses(block)["statuses"]["ghost_freedom"] == "reject"


def test_a_deliberately_ghost_laden_operator_is_rejected() -> None:
    block = adapter.localize(Fraction(2), Fraction(1), Fraction(1), 4)
    assert block["route"] == adapter.ROUTE_LOCAL
    assert block["ghost_pole_count"] == "1"
    assert block["ghost_kind"] == adapter.GHOST_AUXILIARY
    assert block["ghost_localization_artifact_possible"] is False
    assert block["undiagonalized_kinetic_matrix"] == "Matrix([[0, 1], [1, 0]])"
    assert adapter.arm_rung_statuses(block)["statuses"]["ghost_freedom"] == "reject"
    assert adapter.arm_rung_statuses(block)["statuses"]["positive_energy_hamiltonian"] == "reject"


def test_the_instability_control_flips_between_four_and_eight_poles(receipt: dict) -> None:
    control = receipt["controls"]["subtracted_pole_unstable"]
    assert control["observed_stability"] == "UNSTABLE_UNDER_LOCALIZATION"
    assert control["per_pole_count_verdicts"]["2"] == "LOCALIZED_PASS"
    assert control["per_pole_count_verdicts"]["4"] == "LOCALIZED_PASS"
    assert control["per_pole_count_verdicts"]["8"] == "LOCALIZED_REJECT:ghost_freedom"
    assert control["per_pole_count_verdicts"]["16"] == "LOCALIZED_REJECT:ghost_freedom"
    assert control["per_pole_count_ghost_poles"] == {"2": "0", "4": "0", "8": "1", "16": "1"}


def test_a_control_that_does_not_fire_aborts_the_run(monkeypatch) -> None:
    broken = {
        name: dict(case) for name, case in adapter.CONTROL_CASES.items()
    }
    broken["ostrogradsky_quadratic_box"]["expect_stability"] = "STABLE_PASS"
    monkeypatch.setattr(adapter, "CONTROL_CASES", broken)
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="but observed"):
        adapter.run_controls(ROOT)


def test_the_moment_controls_are_recorded_for_every_declared_pole_count(receipt: dict) -> None:
    controls = receipt["controls"]["gauss_jacobi_moment_controls"]["by_beta"]
    assert sorted(controls) == ["1/2", "3/4"]
    for rows in controls.values():
        assert sorted(rows) == sorted(str(item) for item in adapter.DECLARED_POLE_COUNTS)


# ---------------------------------------------------------------------------
# Verdict classes and the convergence-study contract.
# ---------------------------------------------------------------------------


def _study(*verdicts: str) -> dict[str, str]:
    return {
        str(count): verdict
        for count, verdict in zip(adapter.DECLARED_POLE_COUNTS, verdicts, strict=True)
    }


def test_every_verdict_class_is_reachable() -> None:
    passing = _study(*(["LOCALIZED_PASS"] * 4))
    rejecting = _study(*(["LOCALIZED_REJECT:ghost_freedom"] * 4))
    blocked = _study(
        *(["LOCALIZED_BLOCKED:missing_adapter:cubic_g3_uniform_weak_field_cone"] * 4)
    )
    flipping = _study(
        "LOCALIZED_PASS",
        "LOCALIZED_PASS",
        "LOCALIZED_REJECT:ghost_freedom",
        "LOCALIZED_REJECT:ghost_freedom",
    )
    assert adapter.classify_stability(passing) == "STABLE_PASS"
    assert adapter.classify_stability(rejecting) == "STABLE_REJECT:ghost_freedom"
    assert adapter.classify_stability(blocked) == (
        "STABLE_BLOCKED:missing_adapter:cubic_g3_uniform_weak_field_cone"
    )
    assert adapter.classify_stability(flipping) == "UNSTABLE_UNDER_LOCALIZATION"


def test_a_stability_class_needs_the_whole_declared_convergence_study() -> None:
    partial = {"2": "LOCALIZED_PASS", "4": "LOCALIZED_PASS"}
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="every declared pole count"):
        adapter.classify_stability(partial)


def test_a_stable_verdict_carries_the_exact_conditional_sentence() -> None:
    statement = adapter.stability_statement("STABLE_PASS")
    assert statement == (
        "holds for every N-pole localization tested, N in {2, 4, 8, 16}; "
        "the nonlocal limit is not proved"
    )
    assert adapter.stability_statement("STABLE_REJECT:ghost_freedom") == statement
    assert "no verdict is claimed" in adapter.stability_statement(
        "UNSTABLE_UNDER_LOCALIZATION"
    )


def test_a_receipt_claiming_stable_with_a_short_study_fails_validation(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    entry = forged["families"][0]
    entry["per_pole_count_verdicts"].pop("16")
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="verdict at every declared pole count"
    ):
        adapter.validate_receipt(forged)


def test_a_receipt_whose_declared_pole_counts_shrank_fails_validation(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    forged["config"]["declared_pole_counts"] = ["2", "4"]
    forged["config_sha256"] = canonical_sha256(forged["config"])
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="declared pole-count set changed"
    ):
        adapter.validate_receipt(forged)


def test_a_relabelled_stability_class_does_not_replay(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    victim = next(
        entry for entry in forged["families"] if entry["stability"].startswith("STABLE_REJECT")
    )
    victim["stability"] = "STABLE_PASS"
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="stability class does not replay"
    ):
        adapter.validate_receipt(forged)


def test_a_verdict_may_not_drop_its_residual_blockers(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    forged["families"][0]["residual_blockers"] = []
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="dropped its residual blockers"
    ):
        adapter.validate_receipt(forged)


# ---------------------------------------------------------------------------
# The 71-family result.
# ---------------------------------------------------------------------------


def test_every_family_is_decided_at_every_declared_pole_count(receipt: dict) -> None:
    assert len(receipt["families"]) == 71
    declared = sorted(str(item) for item in adapter.DECLARED_POLE_COUNTS)
    for entry in receipt["families"]:
        assert sorted(entry["per_pole_count_verdicts"]) == declared
        assert sorted(entry["per_pole_count_rungs"]) == declared
        for statuses in entry["per_pole_count_rungs"].values():
            assert list(statuses) == list(adapter.LADDER_RUNGS)


def test_the_ghost_result_is_exactly_the_alpha_above_one_families(receipt: dict) -> None:
    rejected = {
        entry["representative_ordinal"]
        for entry in receipt["families"]
        if entry["stability"] == "STABLE_REJECT:ghost_freedom"
    }
    above_one = {
        entry["representative_ordinal"]
        for entry in receipt["families"]
        if Fraction(entry["alpha"]) > 1
    }
    assert rejected == above_one
    assert receipt["counts"]["stable_reject_by_rung"] == {"ghost_freedom": 37}
    assert receipt["counts"]["ghost_kind_counts"]["propagator_residue_ghost"] == 37
    assert receipt["counts"]["ghost_localization_artifact_possible"] == 0


def test_the_counts_close_over_the_seventy_one_families(receipt: dict) -> None:
    counts = receipt["counts"]
    assert (
        counts["stable_pass"]
        + counts["stable_reject"]
        + counts["stable_blocked"]
        + counts["unstable_under_localization"]
        == counts["families_in"]
        == 71
    )
    assert counts["candidates_in"] == 45546


def test_the_remaining_block_is_a_different_named_adapter(receipt: dict) -> None:
    codes = set(receipt["counts"]["stable_blocked_by_adapter"])
    assert codes == {"missing_adapter:cubic_g3_uniform_weak_field_cone"}
    assert codes <= set(adapter.BLOCKERS)
    assert adapter.DISCHARGED_BLOCKER not in codes


def test_a_pass_never_claims_the_complete_lift(receipt: dict) -> None:
    for entry in receipt["families"]:
        assert list(entry["residual_blockers"]) == list(adapter.RESIDUAL_BLOCKERS)
        assert "missing_adapter:direct_scalar_matter_coupling" in entry["residual_blockers"]
        assert "missing_adapter:uv_form_factor_operator" in entry["residual_blockers"]
        assert "nonlocal_limit_of_the_finite_pole_localization_unproved" in (
            entry["residual_blockers"]
        )


# ---------------------------------------------------------------------------
# The localized action IR.
# ---------------------------------------------------------------------------


def test_the_localized_action_ir_carries_one_mode_per_pole(receipt: dict) -> None:
    for bundle in receipt["localizations"].values():
        for pole_count, action in sorted(bundle["localized_action_ir_by_pole_count"].items()):
            block = bundle["by_pole_count"][pole_count]
            assert action["schema_version"] == adapter.ACTION_IR_SCHEMA
            assert len(action["auxiliary_fields"]) == int(block["auxiliary_field_count"])
            assert action["content_sha256"] == canonical_sha256(
                {key: item for key, item in action.items() if key != "content_sha256"}
            )
            signs = {field["kinetic_sign"] for field in action["auxiliary_fields"]}
            assert signs <= {"1", "-1"}


def test_the_frozen_grammar_admits_one_mode_and_refuses_the_tower() -> None:
    single = adapter.grammar_admission(ROOT, 1)
    assert single["valid"] is True
    assert single["errors"] == []
    tower = adapter.grammar_admission(ROOT, 8)
    assert tower["valid"] is False
    assert any("unknown fields" in error for error in tower["errors"])
    probe = tower["extra_dynamical_field_bound_probe"]
    assert probe["valid"] is False
    assert any("extra-dynamical-field bound" in error for error in probe["errors"])


def test_the_field_contract_is_not_amended_by_this_adapter() -> None:
    contract = json.loads(
        (ROOT / adapter.FIELD_CONTRACT_PATH).read_text(encoding="utf-8")
    )
    assert contract["action_contract"]["matter_rule"].startswith(
        "Every matter species is minimally coupled"
    )
    assert [field["id"] for field in contract["fields"]] == [
        "g_mu_nu",
        "psi_m",
        "phi",
        "u_mu",
        "A_mu",
        "lambda_u",
        "J_b_mu",
    ]


def test_the_three_declared_nonlocality_controls_are_emitted(receipt: dict) -> None:
    for bundle in receipt["localizations"].values():
        for controls in bundle["nonlocality_controls_by_pole_count"].values():
            assert sorted(controls) == ["auxiliary_field", "causality", "initial_value"]
            assert controls["causality"]["status"] == "pass"
            assert "retarded branch" in controls["causality"]["open"]
            assert "more initial data" in controls["initial_value"]["open"]
            assert controls["auxiliary_field"]["status"] == "pass"


# ---------------------------------------------------------------------------
# Receipt discipline.
# ---------------------------------------------------------------------------


def test_the_claims_block_says_the_localization_is_not_the_nonlocal_theory(
    receipt: dict,
) -> None:
    assert receipt["claims"] == adapter.CLAIMS
    assert receipt["claims"]["approximation_is_not_the_nonlocal_theory"] is True
    assert receipt["claims"]["synthetic_controls_only"] is True
    assert receipt["claims"]["first_principles_derivation_pending"] is True
    assert receipt["claims"]["real_data_used"] is False


def test_a_receipt_without_the_approximation_claim_is_refused(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    forged["claims"]["approximation_is_not_the_nonlocal_theory"] = False
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="claims block changed"):
        adapter.validate_receipt(forged)


def test_no_float_reaches_the_receipt(receipt: dict) -> None:
    adapter._no_floats(receipt)
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="float in receipt"):
        adapter._no_floats({"families": [{"alpha": 0.5}]})


def test_the_receipt_seal_and_config_binding_replay(receipt: dict) -> None:
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert receipt["config_sha256"] == canonical_sha256(receipt["config"])
    adapter.validate_receipt(receipt)


def test_a_tampered_seal_is_refused(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    forged["decision"] = "everything passed"
    with pytest.raises(adapter.NonlocalFractionalAdapterError, match="receipt seal changed"):
        adapter.validate_receipt(forged)
    resealed = {key: item for key, item in forged.items() if key != "content_sha256"}
    resealed = {**resealed, "content_sha256": canonical_sha256(resealed)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="decision line does not replay"
    ):
        adapter.validate_receipt(resealed)


def test_a_doctored_headline_count_does_not_replay(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    forged["counts"]["stable_pass"] = 71
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="aggregate counts do not replay"
    ):
        adapter.validate_receipt(forged)


def test_a_forged_localization_id_does_not_replay(receipt: dict) -> None:
    forged = json.loads(json.dumps(receipt))
    key = min(forged["localizations"])
    forged["localizations"][key]["alpha"] = "99"
    forged.pop("content_sha256")
    forged = {**forged, "content_sha256": canonical_sha256(forged)}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="localization id does not replay"
    ):
        adapter.validate_receipt(forged)


def test_the_sealed_receipt_replays_the_run(receipt: dict, sealed: dict) -> None:
    # The sealed artifact was written by an earlier process; matching hashes is a determinism
    # check across processes, not only across calls.
    assert sealed["content_sha256"] == receipt["content_sha256"]
    adapter.validate_receipt(sealed)


def test_the_localization_is_deterministic_across_repeated_construction() -> None:
    adapter._LOCALIZATION_CACHE.clear()
    first = adapter._localization_bundle(ROOT, Fraction(3, 4), Fraction(1, 4), Fraction(3, 2))
    adapter._LOCALIZATION_CACHE.clear()
    second = adapter._localization_bundle(ROOT, Fraction(3, 4), Fraction(1, 4), Fraction(3, 2))
    assert first["localization_id"] == second["localization_id"]
    assert canonical_sha256(json.loads(json.dumps(first))) == canonical_sha256(
        json.loads(json.dumps(second))
    )


def test_the_immutable_receipt_write_refuses_to_overwrite(tmp_path, receipt: dict) -> None:
    target = tmp_path / "receipt.json"
    adapter._write(receipt, target)
    adapter._write(receipt, target)
    forged = {**receipt, "decision": "different"}
    with pytest.raises(
        adapter.NonlocalFractionalAdapterError, match="immutable receipt"
    ):
        adapter._write(forged, target)


def test_the_cli_validates_the_sealed_receipt(capsys) -> None:
    assert adapter.main(["--root", str(ROOT), "--validate-checked"]) == 0
    assert capsys.readouterr().out == ""
