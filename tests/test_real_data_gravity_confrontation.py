"""Tests for the real-data confrontation.

Each test names the thing that would embarrass the claim if it stopped holding.  The two
that matter most are the mandatory controls: Newtonian baryons alone must be certified
INFEASIBLE on these published rows, and a per-object mass parameter must still be rejected
by the amended field contract.  If either stopped holding, nothing else in this module means
anything.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.real_data_gravity_confrontation import (
    A0_GRID,
    CLAIMS,
    CONTRACT_PROBE_ACTIONS,
    COVERAGE_GRID,
    LENGTH_UNIT_GRID,
    QUADRATURE,
    RECEIPT_PATH,
    REFERENCE_GRID_POINT,
    RESULT_SCHEMA,
    TRIAL_TYPE,
    ColumnCache,
    RealDataGravityError,
    build_design,
    build_receipt,
    check_contract,
    clause_a_violations,
    contract_probe_report,
    derivation_chain,
    interval_reach,
    load_families,
    load_galaxies,
    measured_rows,
    newtonian_columns,
    pin_row,
    prepare_galaxy,
    render_action_latex,
    render_family_formulas,
    render_law_latex,
    render_observable_latex,
    run_controls,
    select_best_family,
    universal_parameter_width,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.tolerance_aware_fitting import (
    FEASIBLE,
    INFEASIBLE,
    ToleranceFittingError,
    parse_rows,
)

ROOT = Path(__file__).resolve().parents[1]
PIN_SOURCE = "declared structural constraint"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(ROOT)


@pytest.fixture(scope="module")
def sealed() -> dict:
    return json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def galaxies():
    return load_galaxies(ROOT)


@pytest.fixture(scope="module")
def families():
    return load_families(ROOT)


@pytest.fixture(scope="module")
def prepared(galaxies):
    items, provenance = galaxies
    convention = provenance["mass_to_light_convention"]
    return {
        galaxy.name: prepare_galaxy(
            galaxy,
            Fraction(convention["disk_3_6um"]),
            Fraction(convention["bulge_3_6um"]),
            QUADRATURE,
        )
        for galaxy in items
    }


@pytest.fixture(scope="module")
def rows(galaxies):
    items, _ = galaxies
    return {galaxy.name: measured_rows(galaxy, "test source citation") for galaxy in items}


# ---------------------------------------------------------------------------
# Step 0 -- the amended field contract
# ---------------------------------------------------------------------------


def test_the_contract_is_at_the_amended_version_and_clause_a_is_not_overridable() -> None:
    report = check_contract(ROOT)
    assert report["version"] == "sigma-covariant-field-contract-1.1"
    assert report["prior_version"] == "sigma-covariant-field-contract-1.0"
    assert report["clause_a_overridable"] is False
    assert report["amendment_id"].startswith("A1_")


def test_clause_a_still_fires_on_an_action_carrying_a_per_object_mass_parameter() -> None:
    """The amendment must not have opened a door for hidden mass."""

    findings = clause_a_violations(CONTRACT_PROBE_ACTIONS["per_object_halo_mass"])
    assert findings, "a per-object halo mass was accepted by the amended contract"
    assert any("per_object" in item for item in findings)
    assert any("unseen mass is forbidden absolutely" in item for item in findings)


def test_clause_a_permits_a_declared_universal_scalar_matter_coupling() -> None:
    assert clause_a_violations(CONTRACT_PROBE_ACTIONS["universal_scalar_coupling"]) == []


def test_the_probe_report_refuses_to_pass_if_either_probe_flips() -> None:
    report = contract_probe_report()
    assert report["per_object_halo_mass"]["rejected_by_clause_a"] is True
    assert report["universal_scalar_coupling"]["rejected_by_clause_a"] is False


def test_an_undeclared_parameter_scope_is_refused_rather_than_guessed() -> None:
    action = {"parameters": {"beta": {}}}
    with pytest.raises(RealDataGravityError, match="scope"):
        clause_a_violations(action)


def test_the_version_one_zero_prohibitions_are_all_carried_over() -> None:
    report = check_contract(ROOT)
    carried = report["version_one_zero_prohibitions_carried_over"]
    assert "object-specific parameters or metrics" in carried
    assert "a lensing-only metric or coupling" in carried
    assert any("on-shell field equation" in item for item in carried)


# ---------------------------------------------------------------------------
# Step 1 -- the measured data
# ---------------------------------------------------------------------------


def test_the_declared_data_is_six_published_galaxies_and_214_points(galaxies) -> None:
    items, provenance = galaxies
    assert len(items) == 6
    assert provenance["point_count"] == 214
    assert sum(galaxy.count for galaxy in items) == 214
    assert "Lelli" in provenance["source"]["primary_citation"]
    assert provenance["source"]["dataset_doi"].startswith("doi:")


def test_every_measured_row_cites_a_source_and_carries_a_declared_sigma_rule(rows) -> None:
    for galaxy_rows in rows.values():
        for row in galaxy_rows:
            assert row.source.strip()
            assert row.value_sigma_rule == "propagated_outward"
            assert row.value_sigma > 0
            assert row.value_citation and "e_Vobs" in row.value_citation
            assert row.point_sigma_rule == "half_ulp_of_last_published_digit"


def test_the_sigma_inflation_guard_refuses_a_sigma_its_rule_does_not_permit() -> None:
    """The instrument, not this module, is what makes sigma unfittable."""

    with pytest.raises(ToleranceFittingError, match="sigma_not_derivable_from_declared_rule"):
        parse_rows(
            [
                {
                    "label": "inflated",
                    "point": "1.00",
                    "point_sigma": "0.5",
                    "point_sigma_rule": "half_ulp_of_last_published_digit",
                    "source": "test",
                    "value": "2.00",
                    "value_sigma_rule": "exact",
                }
            ]
        )


def test_the_mass_to_light_convention_is_one_universal_pair_not_a_per_galaxy_table(
    galaxies,
) -> None:
    _, provenance = galaxies
    convention = provenance["mass_to_light_convention"]
    assert convention["disk_3_6um"] == "1/2"
    assert convention["bulge_3_6um"] == "7/10"
    assert "never per-object" in convention["status"]


# ---------------------------------------------------------------------------
# Step 2 -- formula generation
# ---------------------------------------------------------------------------


def test_the_twelve_surviving_families_carry_twenty_three_candidates(families) -> None:
    assert len(families) == 12
    assert sum(family.size for family in families) == 23
    for family in families:
        assert family.stability == "STABLE_PASS"
        assert family.screening_family == "curvature"
        assert family.sector_id == "kmouflage_convex_kessence"


def test_every_family_renders_its_own_law_action_and_observable(families) -> None:
    laws = set()
    for family in families:
        rendered = render_family_formulas(family)
        law = rendered["law_latex"]
        assert family.parameters["w_yukawa"].split("/")[0] in law
        assert "S(r)" in law
        assert rendered["free_quantities_per_galaxy"] == 0
        assert "K(X)" in rendered["action_latex"]
        assert "A(\\phi)=e^{\\beta\\phi/M_{\\rm Pl}}" in rendered["action_latex"]
        assert "v_{\\rm pred}^{2}(r)" in rendered["observable_latex"]
        laws.add(law)
    assert len(laws) > 1, "every family rendered the same law"


def test_the_rendered_law_matches_the_generating_receipts_own_parameters(families) -> None:
    family = select_best_family(families)
    assert family.ordinal == 673869399
    law = render_law_latex(family)
    assert "e^{-s/2}" in law  # L1 = 2
    assert "\\tfrac{9}{2}" in law  # w_power = 9/2
    assert "]^{4}\\right]^{-1}" in law or "\\right)^{4}\\right]^{-1}" in law  # sharpness k = 4
    assert "K(X)" in render_action_latex(family)
    assert "\\Upsilon_{d}" in render_observable_latex(family)


# ---------------------------------------------------------------------------
# Step 3 -- the derivation chain
# ---------------------------------------------------------------------------


def test_every_derivation_step_is_recomputed_and_checked() -> None:
    chain = derivation_chain()
    assert chain["all_steps_checked"] is True
    assert [step["step"] for step in chain["steps"]] == [1, 2, 3, 4, 5, 6, 7]
    for step in chain["steps"]:
        assert step["checked"] is True


def test_the_derivation_recovers_the_general_relativity_limit_two_ways() -> None:
    chain = derivation_chain()
    limit = next(step for step in chain["steps"] if step["step"] == 6)
    assert limit["coupling_switched_off"] is True
    assert limit["deep_screening_limit"] is True


def test_a_wrong_sign_kinetic_term_breaks_the_derivation() -> None:
    control = derivation_chain()["negative_controls"]["wrong_sign_kinetic_term"]
    assert control["broke"] is True
    assert control["healthy_kinetic_coefficient"] == "1"
    assert control["ghost_kinetic_coefficient"] == "-1"


def test_omitting_the_screening_factor_breaks_the_solar_system_limit() -> None:
    control = derivation_chain()["negative_controls"]["screening_factor_omitted"]
    assert control["broke"] is True
    assert control["cited_bound"]["value"] == "2.3e-5"
    assert "Cassini" in control["cited_bound"]["citation"]


def test_the_derivation_chain_is_deterministic() -> None:
    assert derivation_chain() == derivation_chain()


# ---------------------------------------------------------------------------
# Step 4 -- structural absence of per-object freedom, and the mandatory controls
# ---------------------------------------------------------------------------


def _newtonian_design(items, rows, prepared, pinned):
    return build_design(
        items,
        rows,
        ("baryonic_coefficient",),
        lambda galaxy: newtonian_columns(galaxy, prepared[galaxy.name]["v_bar_squared"]),
        pin_row(PIN_SOURCE),
        pinned,
    )


def test_the_pooled_width_does_not_grow_with_the_number_of_galaxies(
    galaxies, rows, prepared
) -> None:
    """This is what makes 'no per-galaxy parameter' structural rather than a promise."""

    items, _ = galaxies
    widths = set()
    counts = set()
    for take in (1, 2, 6):
        design = _newtonian_design(items[:take], rows, prepared, (0,))
        widths.add(universal_parameter_width(design))
        counts.add(len(design.rows))
    assert widths == {1}, "the design widened when galaxies were added"
    assert len(counts) == 3, "adding galaxies did not add rows"


def test_every_galaxy_is_indexed_by_the_same_universal_parameter_vector(
    galaxies, rows, prepared
) -> None:
    items, _ = galaxies
    design = _newtonian_design(items, rows, prepared, (0,))
    represented = {name for name in design.galaxy_of_row if name != "structural"}
    assert represented == {galaxy.name for galaxy in items}
    assert all(len(column) == design.width for column in design.columns)


def test_newtonian_baryons_alone_are_infeasible_on_these_published_rows(
    galaxies, rows, prepared, families
) -> None:
    """The mandatory control.  If this ever returns FEASIBLE the pipeline is broken."""

    items, _ = galaxies
    controls = run_controls(items, rows, prepared, families, ColumnCache(prepared))
    for coverage in COVERAGE_GRID:
        entry = controls[coverage]["newtonian_baryons_only"]
        assert entry["verdict"] == INFEASIBLE
        assert entry["certificate"]["checked_here"] is True
        assert entry["certificate"]["unreachable_rows"]


def test_a_universal_rescale_of_the_baryons_does_not_rescue_them(
    galaxies, rows, prepared, families
) -> None:
    items, _ = galaxies
    controls = run_controls(items, rows, prepared, families, ColumnCache(prepared))
    entry = controls["1"]["newtonian_baryons_with_one_universal_rescale"]
    assert entry["verdict"] == INFEASIBLE
    assert entry["universal_parameter_count"] == 1


def test_a_deliberately_wrong_law_is_infeasible(galaxies, rows, prepared, families) -> None:
    items, _ = galaxies
    controls = run_controls(items, rows, prepared, families, ColumnCache(prepared))
    entry = controls["1"]["deliberately_wrong_law"]
    assert entry["verdict"] == INFEASIBLE
    assert "wrong asymptotics" in entry["law"]


def test_the_newtonian_limit_of_the_law_is_exact(galaxies, rows, prepared, families) -> None:
    items, _ = galaxies
    controls = run_controls(items, rows, prepared, families, ColumnCache(prepared))
    entry = controls["1"]["newtonian_limit_of_the_law"]
    assert entry["verdict"] == FEASIBLE
    assert float(entry["worst_relative_deviation"]) == 0.0


def test_the_interval_reach_diagnostic_decides_nothing_and_counts_declared_intervals(
    galaxies, rows, prepared, families
) -> None:
    items, _ = galaxies
    family = select_best_family(families)
    reach = interval_reach(family, items, rows, prepared, ColumnCache(prepared))
    assert reach["decides_nothing"] is True
    assert reach["published_intervals"] == 214
    assert 0 <= reach["intervals_reached"] <= 214


# ---------------------------------------------------------------------------
# Grids and claim boundary
# ---------------------------------------------------------------------------


def test_the_universal_constant_grids_are_declared_finite_and_contain_the_reference() -> None:
    assert REFERENCE_GRID_POINT["a0"] in A0_GRID
    assert REFERENCE_GRID_POINT["length_unit"] in LENGTH_UNIT_GRID
    assert len(set(A0_GRID)) == len(A0_GRID)
    assert len(set(LENGTH_UNIT_GRID)) == len(LENGTH_UNIT_GRID)
    assert COVERAGE_GRID[0] == "1"
    assert all(1 <= int(Fraction(value)) <= 6 for value in COVERAGE_GRID)


def test_the_receipt_declares_itself_exploratory_and_uncitable_as_confirmation(
    receipt,
) -> None:
    assert receipt["trial_type"] == TRIAL_TYPE == "exploratory"
    assert receipt["claims"]["sealed_no_refit_trial"] is False
    assert receipt["claims"]["may_be_cited_as_confirmation"] is False
    assert receipt["claims"]["invisible_matter_used"] is False
    assert receipt["claims"]["per_object_free_parameters"] is False
    assert receipt["claims"]["uncertainties_declared_not_fitted"] is True
    assert receipt["claims"]["external_fetch_performed"] is True
    assert receipt["exploratory_caveat"]["sealed_no_refit_trial"] is False


def test_the_receipt_reports_a_verdict_for_every_family_at_every_coverage_factor(
    receipt,
) -> None:
    for coverage in COVERAGE_GRID:
        block = receipt["results_by_coverage_factor"][coverage]
        assert len(block["per_family"]) == 12
        for entry in block["per_family"].values():
            for mode in ("universal_arms", "zero_freedom"):
                assert entry[mode]["verdict"] in {FEASIBLE, INFEASIBLE}
                if entry[mode]["verdict"] == INFEASIBLE:
                    assert entry[mode]["certificate"]["checked_here"] is True
                    assert entry[mode]["certificate"]["unreachable_rows"]


def test_no_family_is_credited_with_more_freedom_than_the_declared_universal_constants(
    receipt,
) -> None:
    for coverage in COVERAGE_GRID:
        block = receipt["results_by_coverage_factor"][coverage]
        for entry in block["per_family"].values():
            assert entry["zero_freedom"]["universal_parameter_count"] == 1
            assert entry["universal_arms"]["universal_parameter_count"] == 3


def test_the_controls_hold_at_every_coverage_factor_in_the_receipt(receipt) -> None:
    assert receipt["controls_summary"]["newtonian_baryons_only_infeasible"] is True
    assert receipt["controls_summary"]["wrong_law_infeasible"] is True


def test_the_nonlocal_quadrature_is_converged_and_the_design_is_cross_checked(
    receipt,
) -> None:
    assert receipt["quadrature_convergence"]["within_tolerance"] is True
    assert receipt["design_crosscheck"]["within_tolerance"] is True
    assert receipt["design_crosscheck"]["working_precision_digits"] == 50


# ---------------------------------------------------------------------------
# Receipt determinism, seal, and tamper
# ---------------------------------------------------------------------------


def test_the_receipt_is_deterministic_and_sealed(receipt) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert receipt["schema_version"] == RESULT_SCHEMA
    assert build_receipt(ROOT) == receipt


def test_the_written_receipt_matches_the_build(receipt, sealed) -> None:
    assert sealed == receipt
    validate_receipt(sealed, root=ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        # The first mutation survives a reseal and is caught only by exact replay; the rest
        # are caught by the cheap guards, which is why they are cheap to test.
        lambda value: value.update({"decision": "PASS"}),
        lambda value: value.update({"trial_type": "confirmatory"}),
        lambda value: value["claims"].update({"may_be_cited_as_confirmation": True}),
        lambda value: value.update({"schema_version": "invariant-something-else-1.0"}),
    ],
)
def test_receipt_tamper_fails_closed(receipt, mutate) -> None:
    tampered = copy.deepcopy(receipt)
    mutate(tampered)
    with pytest.raises(RealDataGravityError):
        validate_receipt(tampered, root=ROOT)
    resealed = {key: value for key, value in tampered.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(resealed)
    with pytest.raises(RealDataGravityError):
        validate_receipt(resealed, root=ROOT)


def test_the_receipt_carries_no_scalar_goodness_key(receipt) -> None:
    from sigma_theory_compiler.tolerance_aware_fitting import forbidden_receipt_keys

    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert forbidden_receipt_keys(body) == []


def test_the_claims_block_is_pinned(receipt) -> None:
    assert receipt["claims"] == CLAIMS
