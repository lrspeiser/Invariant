"""Gates for the declared-framework machinery of the tensor-space constraint search.

The module under test used to carry ``declared_framework`` as inert prose: it appeared at exactly
two places in the source -- a key name and a verbatim ``dict(config["declared_framework"])`` copy
into the receipt -- and at zero places in the test file.  A config declaring flat Euclidean 3-space,
a displacement field and "polynomial in the linear strain tensor" still returned
``['g_mn', 'R_mn - (1/2) R g_mn']``, so a sealed receipt could carry an elasticity framework claim
over a general-relativity computation.

Pinned here: that the declaration now SELECTS the basis generator, so the two shipped frameworks
return genuinely different answers from the same engine; that the framework the receipt reports is
MEASURED off the arrays the generator built and not read back from the config, checked by
recomputing each measured field independently; that perturbing any single field of the declaration
is refused with a typed blocker naming that field; that the original false declaration is refused;
that prose-only and unimplemented-generator declarations are refused; that the isotropic rank-4
enumeration really collapses to two dimensions, cross-checked against an independent sympy rank
computation; that the Navier-Cauchy coefficients the module solves for match an independent
symbolic derivation; and that the elasticity receipt is deterministic, float-free, claim-frozen,
byte-canonical and replay-validated.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from sigma_theory_compiler import tensor_constraint_search as tcs
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CURVATURE_CONFIG = json.loads((ROOT / tcs.CONFIG_PATH).read_text(encoding="utf-8"))
ELASTIC_CONFIG = json.loads((ROOT / tcs.ELASTICITY_CONFIG_PATH).read_text(encoding="utf-8"))
ELASTIC_RECEIPT = ROOT / tcs.ELASTICITY_OUTPUT_PATH
PRIME = 1048573
SECOND_PRIME = 1048571


@pytest.fixture(scope="session")
def elastic_receipt() -> dict:
    return tcs.run_tensor_constraint_search(ELASTIC_CONFIG, ROOT)


@pytest.fixture(scope="session")
def sealed_elastic() -> dict:
    if not ELASTIC_RECEIPT.exists():  # pragma: no cover - the receipt is committed
        pytest.skip("committed elasticity receipt not present")
    return json.loads(ELASTIC_RECEIPT.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def curvature_realized() -> dict:
    bank = tcs.build_bank(4, 2, PRIME, "gate-test", 3)
    terms = [tcs.named_tensors(item, 2) for item in bank]
    return tcs.measure_realized_framework(bank, terms, dimension=4, order=2, modulus=PRIME)


@pytest.fixture(scope="session")
def elastic_realized() -> dict:
    bank = tcs._elastic_bank(3, PRIME, "gate-test", 3)
    patterns = tcs.enumerate_elasticity_patterns(3, PRIME)
    return tcs.measure_realized_elastic_framework(
        bank, patterns, dimension=3, modulus=PRIME, seed="gate-test"
    )


def _search(receipt: dict, search_id: str) -> dict:
    return next(item for item in receipt["searches"] if item["search_id"] == search_id)


# ---------------------------------------------------------------------------
# The declaration selects the generator.
# ---------------------------------------------------------------------------


def test_the_two_shipped_frameworks_declare_different_generators() -> None:
    assert tcs.declared_generator(CURVATURE_CONFIG) == "riemann_tensor"
    assert tcs.declared_generator(ELASTIC_CONFIG) == "linear_strain_tensor"


def test_a_different_declaration_produces_a_genuinely_different_basis(
    elastic_receipt: dict,
) -> None:
    """The whole point.  Same engine, same arithmetic, different declaration, different answer."""

    headline = _search(elastic_receipt, "d3-isotropic-lame")
    assert headline["admissible_space"]["basis"] == [
        "delta_ij_delta_kl",
        "delta_ik_delta_jl + delta_il_delta_jk",
    ]
    # ... and emphatically not the general-relativity answer the module used to return no matter
    # what was declared.
    assert "R_mn - (1/2) R g_mn" not in headline["admissible_space"]["basis"]
    assert "g_mn" not in headline["admissible_space"]["basis"]
    assert headline["surviving_dimension"] == 2
    assert headline["realized_framework"]["concomitant_generator"] == "linear_strain_tensor"


def test_prose_only_framework_declaration_is_refused() -> None:
    config = copy.deepcopy(ELASTIC_CONFIG)
    config["declared_framework"].pop("machine_checkable")
    with pytest.raises(tcs.FrameworkMismatch, match="machine_checkable"):
        tcs.declared_generator(config)
    with pytest.raises(tcs.FrameworkMismatch):
        tcs.run_tensor_constraint_search(config, ROOT)


def test_unimplemented_generator_declaration_is_refused() -> None:
    config = copy.deepcopy(ELASTIC_CONFIG)
    config["declared_framework"]["machine_checkable"]["concomitant_generator"] = "spinor_bilinear"
    with pytest.raises(tcs.FrameworkMismatch, match="spinor_bilinear"):
        tcs.declared_generator(config)


def test_a_missing_typed_field_is_refused() -> None:
    config = copy.deepcopy(ELASTIC_CONFIG)
    config["declared_framework"]["machine_checkable"].pop("curvature")
    with pytest.raises(tcs.FrameworkMismatch, match="curvature"):
        tcs.declared_generator(config)


# ---------------------------------------------------------------------------
# The gate: declared versus measured.
# ---------------------------------------------------------------------------


def test_the_original_false_declaration_is_refused(curvature_realized: dict) -> None:
    """The exact defect.  This declaration used to sail through and seal a receipt."""

    with pytest.raises(tcs.FrameworkMismatch) as blocker:
        tcs.check_framework_consistency(
            tcs.ELASTICITY_DECLARATION_OVER_CURVATURE,
            curvature_realized,
            context="regression",
        )
    message = str(blocker.value)
    for field in ("metric_signature", "curvature", "tensor_rank", "concomitant_generator"):
        assert field in message
    assert "linear_strain_tensor" in message and "riemann_tensor" in message


def test_the_mirror_false_declaration_is_refused(elastic_realized: dict) -> None:
    with pytest.raises(tcs.FrameworkMismatch) as blocker:
        tcs.check_framework_consistency(
            tcs.CURVATURE_DECLARATION_OVER_ELASTICITY,
            elastic_realized,
            context="regression",
        )
    assert "riemann_tensor" in str(blocker.value)


@pytest.mark.parametrize(
    ("field", "lie"),
    [
        ("metric_signature", "euclidean"),
        ("curvature", "flat"),
        ("primary_field_indices", 1),
        ("primary_field_symmetric", False),
        ("tensor_rank", 4),
        ("tensor_symmetry", "none"),
        ("concomitant_generator", "linear_strain_tensor"),
    ],
)
def test_every_single_declared_field_is_load_bearing(
    curvature_realized: dict, field: str, lie: object
) -> None:
    """Perturb one field of a true declaration and the gate must name exactly that field."""

    truthful = {name: curvature_realized[name] for name in tcs.FRAMEWORK_FIELDS}
    assert tcs.check_framework_consistency(truthful, curvature_realized, context="t")["agreed"]
    assert truthful[field] != lie, "the perturbation must actually change the field"
    with pytest.raises(tcs.FrameworkMismatch) as blocker:
        tcs.check_framework_consistency({**truthful, field: lie}, curvature_realized, context="t")
    assert field in str(blocker.value)


def test_a_curvature_lie_that_still_routes_to_the_curvature_generator_is_refused() -> None:
    """The subtle case: the generator name is honest, the rest of the framework is not."""

    config = copy.deepcopy(CURVATURE_CONFIG)
    config["declared_framework"]["machine_checkable"]["curvature"] = "flat"
    assert tcs.declared_generator(config) == "riemann_tensor"
    with pytest.raises(tcs.FrameworkMismatch, match="curvature"):
        tcs.run_tensor_constraint_search(config, ROOT)


def test_the_gate_runs_on_every_search_in_the_elasticity_receipt(elastic_receipt: dict) -> None:
    consistency = elastic_receipt["framework_consistency"]
    assert consistency["all_searches_agree"] is True
    assert consistency["searches_checked"] == [
        item["search_id"] for item in elastic_receipt["searches"]
    ]
    assert len(consistency["per_search"]) == len(elastic_receipt["searches"])
    for entry in consistency["per_search"]:
        assert entry["agreed"] is True
        assert list(entry["checked_fields"]) == list(tcs.FRAMEWORK_FIELDS)
        assert entry["declared"] == entry["realized"]


# ---------------------------------------------------------------------------
# The realized framework is MEASURED, not read back from the config.
# ---------------------------------------------------------------------------


def test_the_flat_jet_probe_really_has_zero_curvature() -> None:
    flat = tcs._Geometry(4, tcs.flat_metric_jet(4, 3, PRIME), PRIME, 3)
    curved = tcs._Geometry(4, tcs.metric_jet(4, 3, PRIME, "probe", 0), PRIME, 3)
    assert not np.any(flat.riemann % PRIME)
    assert np.any(curved.riemann % PRIME)


def test_curvature_generator_identity_is_measured_by_the_flat_jet(
    curvature_realized: dict,
) -> None:
    """Recompute the generator identification independently of the module's bookkeeping."""

    flat = tcs._Geometry(4, tcs.flat_metric_jet(4, 3, PRIME), PRIME, 3)
    flat_terms = tcs.named_tensors(flat, 2)
    dead_on_flat = sorted(
        name for name in tcs.named_term_names(2) if not np.any(flat_terms[name] % PRIME)
    )
    assert dead_on_flat == ["Rg", "Ric"], "R_mn and R g_mn are the curvature-driven terms"
    assert np.any(flat_terms["g"] % PRIME), "g_mn survives on the flat jet and is not curvature"
    evidence = curvature_realized["measurement_evidence"]
    assert sorted(evidence["terms_vanishing_on_the_exactly_flat_jet"]) == dead_on_flat
    assert curvature_realized["concomitant_generator"] == "riemann_tensor"


def test_curvature_signature_and_rank_are_measured(curvature_realized: dict) -> None:
    jet = tcs.metric_jet(4, 3, PRIME, "probe", 0)
    base = jet[..., 0] % PRIME
    assert int(base[0, 0]) == PRIME - 1 and all(int(base[a, a]) == 1 for a in (1, 2, 3))
    assert curvature_realized["metric_signature"] == "lorentzian"
    assert curvature_realized["primary_field_indices"] == jet.ndim - 1 == 2
    assert curvature_realized["primary_field_symmetric"] is True
    assert curvature_realized["tensor_rank"] == 2
    assert curvature_realized["curvature"] == "generic"


def test_elastic_framework_measurements_are_independently_reproducible(
    elastic_realized: dict,
) -> None:
    jet = tcs.displacement_jet(3, 3, PRIME, "gate-test", 0)
    assert jet.ndim - 1 == 1
    elastic = tcs._ElasticJet(3, jet, PRIME, 3)
    # the metric this generator uses is the Kronecker delta, and its curvature is COMPUTED
    assert np.array_equal(elastic.metric[..., 0] % PRIME, np.eye(3, dtype=np.int64))
    assert not np.any(elastic.christoffel % PRIME)
    assert not np.any(elastic.riemann % PRIME)
    assert {field: elastic_realized[field] for field in tcs.FRAMEWORK_FIELDS} == {
        "metric_signature": "euclidean",
        "curvature": "flat",
        "primary_field_indices": 1,
        "primary_field_symmetric": False,
        "tensor_rank": 4,
        "tensor_symmetry": "symmetric",
        "concomitant_generator": "linear_strain_tensor",
    }
    # and the two frameworks disagree on every field the gate checks except the stress symmetry
    curvature_bank = tcs.build_bank(4, 2, PRIME, "gate-test", 2)
    curvature = tcs.measure_realized_framework(
        curvature_bank,
        [tcs.named_tensors(item, 2) for item in curvature_bank],
        dimension=4,
        order=2,
        modulus=PRIME,
    )
    differing = {
        field for field in tcs.FRAMEWORK_FIELDS if curvature[field] != elastic_realized[field]
    }
    assert differing == set(tcs.FRAMEWORK_FIELDS) - {"tensor_symmetry"}


def test_a_rigid_motion_separates_the_strain_generator_from_a_gradient_generator() -> None:
    """The probe that makes the elasticity generator identity a measurement, not a label."""

    rigid = tcs._ElasticJet(3, tcs.rigid_motion_jet(3, 3, PRIME, "probe"), PRIME, 3)
    assert np.any(rigid.gradient % PRIME), "a rigid rotation has a non-zero displacement gradient"
    assert not np.any(rigid.strain % PRIME), "a rigid motion has identically zero linear strain"
    patterns = tcs.enumerate_elasticity_patterns(3, PRIME)
    assert all(
        not np.any(rigid.stress(item["array"], rigid.strain) % PRIME) for item in patterns
    )
    assert any(np.any(rigid.stress(item["array"], rigid.gradient) % PRIME) for item in patterns)


def test_the_strain_really_is_the_symmetrised_gradient() -> None:
    elastic = tcs._ElasticJet(
        3, tcs.displacement_jet(3, 3, PRIME, "probe", 1), PRIME, 3
    )
    expected = (
        (elastic.gradient + np.einsum("ikj->kij", elastic.gradient))
        * pow(2, PRIME - 2, PRIME)
    ) % PRIME
    assert np.array_equal(elastic.strain % PRIME, expected)
    assert np.array_equal(elastic.strain % PRIME, np.einsum("ikj->kij", elastic.strain) % PRIME)


# ---------------------------------------------------------------------------
# The elasticity derivation, cross-checked against independent code.
# ---------------------------------------------------------------------------


def test_the_three_delta_pairings_are_the_enumeration() -> None:
    patterns = tcs.enumerate_elasticity_patterns(3, PRIME)
    assert [item["label"] for item in patterns] == [
        "delta_ij_delta_kl",
        "delta_ik_delta_jl",
        "delta_il_delta_jk",
    ]
    delta = np.eye(3, dtype=np.int64)
    expected = [
        np.einsum("ij,kl->ijkl", delta, delta),
        np.einsum("ik,jl->ijkl", delta, delta),
        np.einsum("il,jk->ijkl", delta, delta),
    ]
    for item, reference in zip(patterns, expected, strict=True):
        assert np.array_equal(item["array"] % PRIME, reference % PRIME)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_isotropic_minor_symmetric_rank4_space_is_two_dimensional_by_sympy(
    dimension: int,
) -> None:
    """Independent rank computation over the rationals, not over the prime field."""

    delta = sp.eye(dimension)

    def pairing(spec: str) -> sp.Matrix:
        rows = []
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    for l in range(dimension):
                        index = {"i": i, "j": j, "k": k, "l": l}
                        value = delta[index[spec[0]], index[spec[1]]] * delta[
                            index[spec[2]], index[spec[3]]
                        ]
                        rows.append(value)
        return sp.Matrix(rows).T

    pairings = [pairing("ijkl"), pairing("ikjl"), pairing("iljk")]
    assert sp.Matrix.vstack(*pairings).rank() == 3

    # minor symmetry C_ijkl = C_jikl = C_ijlk, as a linear condition on the coefficients
    def minor_residual(vector: sp.Matrix) -> sp.Matrix:
        array = sp.zeros(1, dimension**4)
        for coefficient, row in zip(vector, pairings, strict=True):
            array += coefficient * row
        residual = []
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    for l in range(dimension):
                        here = ((i * dimension + j) * dimension + k) * dimension + l
                        swap_ij = ((j * dimension + i) * dimension + k) * dimension + l
                        swap_kl = ((i * dimension + j) * dimension + l) * dimension + k
                        residual.append(array[here] - array[swap_ij])
                        residual.append(array[here] - array[swap_kl])
        return sp.Matrix(residual)

    unknowns = sp.symbols("c0 c1 c2")
    residual = minor_residual(sp.Matrix(list(unknowns)))
    solution = sp.solve(list(residual), list(unknowns), dict=True)
    assert len(solution) == 1
    free = {value for entry in solution[0].values() for value in entry.free_symbols}
    free |= {symbol for symbol in unknowns if symbol not in solution[0]}
    assert len(free) == 2, "minor symmetry leaves exactly two free isotropic constants"

    engine = tcs.run_elasticity_search(
        dimension=dimension,
        constraints=list(tcs.SUPPORTED_ELASTIC_CONSTRAINTS[:-1]),
        modulus=PRIME,
        seed="crosscheck",
        bank_samples=3,
        holdout_samples=2,
    )
    assert engine["surviving_dimension"] == 2 == len(free)


def test_navier_cauchy_matches_an_independent_symbolic_derivation() -> None:
    """Derive d_j sigma_ij symbolically in sympy and compare to the module's solved coefficients."""

    coords = sp.symbols("x y z")
    lam, mu = sp.symbols("lambda mu")
    displacement = []
    for axis in range(3):
        poly = sp.Integer(0)
        for order, monomial in enumerate(
            [1, *coords, *[a * b for index, a in enumerate(coords) for b in coords[index:]]]
        ):
            poly += sp.Symbol(f"a_{axis}_{order}") * monomial
        displacement.append(poly)

    def d(expression: sp.Expr, axis: int) -> sp.Expr:
        return sp.diff(expression, coords[axis])

    strain = sp.Matrix(
        3, 3, lambda i, j: sp.Rational(1, 2) * (d(displacement[i], j) + d(displacement[j], i))
    )
    trace = sum(strain[a, a] for a in range(3))
    stress = sp.Matrix(
        3, 3, lambda i, j: lam * (1 if i == j else 0) * trace + 2 * mu * strain[i, j]
    )
    equilibrium = [sum(d(stress[i, j], j) for j in range(3)) for i in range(3)]
    laplacian = [sum(d(d(displacement[i], j), j) for j in range(3)) for i in range(3)]
    divergence = sum(d(displacement[j], j) for j in range(3))
    grad_div = [d(divergence, i) for i in range(3)]
    for i in range(3):
        residual = sp.expand(equilibrium[i] - mu * laplacian[i] - (lam + mu) * grad_div[i])
        assert residual == 0, "the symbolic derivation is not Navier-Cauchy"

    derived = tcs.navier_cauchy_derivation(
        dimension=3, modulus=PRIME, seed="crosscheck", bank_samples=3, holdout_samples=2
    )
    by_name = {item["direction"]: item for item in derived["directions"]}
    # C = d_ij d_kl carries lambda: it contributes only to grad div.
    assert by_name["lame_lambda"]["laplacian_coefficient"] == "0"
    assert by_name["lame_lambda"]["grad_div_coefficient"] == "1"
    # C = d_ik d_jl + d_il d_jk carries mu: it contributes to both, with equal weight.
    assert by_name["lame_mu"]["laplacian_coefficient"] == "1"
    assert by_name["lame_mu"]["grad_div_coefficient"] == "1"
    assert all(item["verified_on_holdout"] for item in derived["directions"])
    assert derived["assembled"] == (
        "d_j sigma_ij = (1 mu) lap u_i + (1 lambda + 1 mu) d_i (div u)"
    )


def test_fabricated_navier_cauchy_coefficient_is_rejected_independently() -> None:
    """The module's own control, recomputed here with independent arithmetic."""

    jet = tcs._ElasticJet(3, tcs.displacement_jet(3, 3, PRIME, "fabricate", 0), PRIME, 3)
    patterns = tcs.enumerate_elasticity_patterns(3, PRIME)
    lam, mu = 7, 11
    array = (
        lam * patterns[0]["array"]
        + mu * (patterns[1]["array"] + patterns[2]["array"])
    ) % PRIME
    actual = jet.equilibrium(jet.stress(array, jet.strain)) % PRIME
    truthful = (mu * jet.laplacian() + (lam + mu) * jet.gradient_of_divergence()) % PRIME
    fabricated = (mu * jet.laplacian() + (lam + 2 * mu) * jet.gradient_of_divergence()) % PRIME
    assert np.array_equal(actual, truthful)
    assert not np.array_equal(actual, fabricated)


def test_the_minor_antisymmetric_combination_is_invisible_to_elasticity() -> None:
    patterns = tcs.enumerate_elasticity_patterns(3, PRIME)
    trap = (patterns[1]["array"] - patterns[2]["array"]) % PRIME
    assert np.any(trap), "the trap must be a non-zero rank-4 tensor"
    jets = [
        tcs._ElasticJet(3, tcs.displacement_jet(3, 3, PRIME, "trap", sample), PRIME, 3)
        for sample in range(3)
    ]
    assert all(not np.any(jet.stress(trap, jet.strain) % PRIME) for jet in jets)
    assert any(np.any(jet.stress(trap, jet.gradient) % PRIME) for jet in jets)


def test_dropping_the_argument_symmetry_enlarges_and_the_cauchy_relation_shrinks(
    elastic_receipt: dict,
) -> None:
    headline = _search(elastic_receipt, "d3-isotropic-lame")["surviving_dimension"]
    grown = _search(elastic_receipt, "d3-major-symmetry-only")["surviving_dimension"]
    shrunk = _search(elastic_receipt, "d3-cauchy-relation")["surviving_dimension"]
    assert grown > headline > shrunk
    assert (headline, grown, shrunk) == (2, 3, 1)


def test_stress_symmetry_alone_already_forces_the_minor_symmetry(elastic_receipt: dict) -> None:
    stress_only = _search(elastic_receipt, "d3-stress-symmetry-only")
    headline = _search(elastic_receipt, "d3-isotropic-lame")
    assert stress_only["surviving_dimension"] == headline["surviving_dimension"] == 2
    assert stress_only["admissible_space"]["basis"] == headline["admissible_space"]["basis"]
    verdict = elastic_receipt["generalizations"][
        "minor_symmetry_is_not_an_independent_declaration"
    ]["verdict"]
    assert verdict.startswith("FORCED")


def test_the_reduction_table_reports_what_each_constraint_actually_cost(
    elastic_receipt: dict,
) -> None:
    """Each row is measured after applying that constraint, not a repeat of the final number."""

    def table(search_id: str) -> list[tuple[str, object]]:
        return [
            (str(row["constraint"]), row["dimension"])
            for row in elastic_receipt["reduction_tables"][search_id]
        ]

    assert table("d3-isotropic-lame") == [
        ("(declared framework only)", "infinite"),
        ("flat_euclidean + material_isotropy", 3),
        ("minor_symmetry", 2),
        ("stress_symmetry", 2),
        ("major_symmetry", 2),
        ("(final)", 2),
    ]
    # major symmetry alone costs nothing for an isotropic material: the row must say 3, not 2
    assert table("d3-major-symmetry-only") == [
        ("(declared framework only)", "infinite"),
        ("flat_euclidean + material_isotropy", 3),
        ("major_symmetry", 3),
        ("(final)", 3),
    ]
    # and the 3 -> 2 collapse is bought by the symmetry of the Cauchy stress on its own
    assert table("d3-stress-symmetry-only") == [
        ("(declared framework only)", "infinite"),
        ("flat_euclidean + material_isotropy", 3),
        ("stress_symmetry", 2),
        ("major_symmetry", 2),
        ("(final)", 2),
    ]
    assert table("d3-cauchy-relation")[-2:] == [("cauchy_relation", 1), ("(final)", 1)]


def test_the_enumeration_labels_are_pinned_to_the_coefficient_coordinate_system() -> None:
    for dimension in (1, 2, 3, 4):
        patterns = tcs.enumerate_elasticity_patterns(dimension, PRIME)
        assert [item["label"] for item in patterns] == list(tcs.ELASTICITY_PATTERN_TEXT)


def test_one_dimension_has_a_single_elastic_constant(elastic_receipt: dict) -> None:
    sweep = {
        item["dimension"]: item["elastic_constants"]
        for item in elastic_receipt["generalizations"]["dimension_sweep"]["by_dimension"]
    }
    assert sweep == {1: 1, 2: 2, 3: 2, 4: 2}


def test_dropping_the_flat_isotropic_background_is_refused() -> None:
    with pytest.raises(tcs.ConstraintOutOfScope, match="flat_euclidean"):
        tcs.run_elasticity_search(
            dimension=3,
            constraints=["minor_symmetry", "stress_symmetry"],
            modulus=PRIME,
            seed="refuse",
            bank_samples=2,
            holdout_samples=1,
        )
    with pytest.raises(tcs.TensorConstraintSearchError, match="unsupported"):
        tcs.run_elasticity_search(
            dimension=3,
            constraints=["flat_euclidean", "material_isotropy", "hyperbolic_background"],
            modulus=PRIME,
            seed="refuse",
            bank_samples=2,
            holdout_samples=1,
        )


def test_the_degenerate_one_dimensional_fit_reports_nothing() -> None:
    derived = tcs.navier_cauchy_derivation(
        dimension=1, modulus=PRIME, seed="degenerate", bank_samples=3, holdout_samples=2
    )
    assert derived["operator_basis_independent_on_the_bank"] is False
    assert derived["assembled"] is None
    assert all(item["unique_solution"] is False for item in derived["directions"])


def test_exact_solver_refuses_an_underdetermined_system() -> None:
    column = [1, 2, 3, 4]
    assert tcs._solve_exact([column, column], [2, 4, 6, 8], PRIME) is None
    solved = tcs._solve_exact([[1, 0, 0], [0, 1, 0]], [3, 5, 0], PRIME)
    assert solved == [Fraction(3), Fraction(5)]


# ---------------------------------------------------------------------------
# Receipt discipline for the second framework.
# ---------------------------------------------------------------------------


def test_elasticity_receipt_shape_and_claims(elastic_receipt: dict) -> None:
    assert set(elastic_receipt) == tcs._TOP_KEYS_STRAIN
    assert elastic_receipt["claims"] == tcs.CLAIMS
    assert tcs.CLAIMS["declared_framework_selects_the_basis_generator"] is True
    assert tcs.CLAIMS["declared_framework_is_gated_against_the_measured_computation"] is True
    assert elastic_receipt["schema_version"] == tcs.RESULT_SCHEMA


def test_elasticity_receipt_carries_no_floats_and_no_host_paths(elastic_receipt: dict) -> None:
    body = {key: value for key, value in elastic_receipt.items() if key != "content_sha256"}
    tcs._no_floats(body)
    assert not tcs._HOST_PATH.search(json.dumps(elastic_receipt, sort_keys=True))


def test_elasticity_two_primes_agree(elastic_receipt: dict) -> None:
    for item in elastic_receipt["searches"]:
        assert item["prime_replays"] == 2
    first = tcs.run_elasticity_search(
        dimension=3,
        constraints=list(ELASTIC_CONFIG["sweep_constraints"]),
        modulus=PRIME,
        seed="agree",
        bank_samples=3,
        holdout_samples=2,
    )
    second = tcs.run_elasticity_search(
        dimension=3,
        constraints=list(ELASTIC_CONFIG["sweep_constraints"]),
        modulus=SECOND_PRIME,
        seed="agree",
        bank_samples=3,
        holdout_samples=2,
    )
    assert first == second


def test_elasticity_negative_controls_all_pass(elastic_receipt: dict) -> None:
    controls = elastic_receipt["negative_controls"]
    assert controls, "the elasticity framework ships negative controls"
    for control in controls:
        assert control["status"] == "pass"
        assert control["verdict"] in {"REJECTED", "REFUSED"}
    names = {control["control"] for control in controls}
    assert "elasticity_framework_declared_over_a_curvature_computation_is_refused" in names
    assert "curvature_framework_declared_over_an_elasticity_computation_is_refused" in names
    assert "fabricated_navier_cauchy_coefficient_is_rejected" in names
    assert "a_rigid_motion_produces_no_stress" in names


def test_the_regression_control_records_the_original_defect(elastic_receipt: dict) -> None:
    control = next(
        item
        for item in elastic_receipt["negative_controls"]
        if item["control"] == "elasticity_framework_declared_over_a_curvature_computation_is_refused"
    )
    assert control["blocker_type"] == "FrameworkMismatch"
    assert set(control["mismatched_fields"]) >= {
        "concomitant_generator",
        "curvature",
        "metric_signature",
        "tensor_rank",
    }


def test_elasticity_receipt_is_deterministic(elastic_receipt: dict, sealed_elastic: dict) -> None:
    assert elastic_receipt == sealed_elastic
    again = tcs.run_tensor_constraint_search(ELASTIC_CONFIG, ROOT)
    assert again["content_sha256"] == elastic_receipt["content_sha256"]


def test_elasticity_receipt_is_byte_canonical(sealed_elastic: dict) -> None:
    encoded = ELASTIC_RECEIPT.read_bytes()
    assert encoded == canonical_json_bytes(sealed_elastic) + b"\n"
    body = {key: value for key, value in sealed_elastic.items() if key != "content_sha256"}
    assert sealed_elastic["content_sha256"] == canonical_sha256(body)


def test_committed_elasticity_receipt_passes_full_replay_validation(sealed_elastic: dict) -> None:
    tcs.validate_receipt(sealed_elastic, ELASTIC_CONFIG, ROOT)


def test_a_resealed_framework_tamper_is_caught_by_replay(sealed_elastic: dict) -> None:
    """Rewrite the sealed framework block to the false claim and reseal; replay must catch it."""

    tampered = copy.deepcopy(sealed_elastic)
    tampered["framework_consistency"]["typed_declaration"]["concomitant_generator"] = (
        "riemann_tensor"
    )
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.validate_receipt(tampered, ELASTIC_CONFIG, ROOT)


def test_validate_receipt_rejects_a_receipt_from_the_other_generator(
    sealed_elastic: dict,
) -> None:
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.validate_receipt(sealed_elastic, CURVATURE_CONFIG, ROOT)


def test_elasticity_source_bindings_point_at_its_own_config(elastic_receipt: dict) -> None:
    bindings = elastic_receipt["source_bindings"]
    assert bindings["config"]["path"] == tcs.ELASTICITY_CONFIG_PATH
    for binding in bindings.values():
        digest = binding.get("file_sha256") or binding.get("semantic_sha256")
        assert isinstance(digest, str) and len(digest) == 64


def test_cli_runs_and_validates_the_elasticity_framework() -> None:
    if not ELASTIC_RECEIPT.exists():  # pragma: no cover - the receipt is committed
        pytest.skip("committed elasticity receipt not present")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigma_theory_compiler.tensor_constraint_search",
            "--framework",
            "linear_strain_tensor",
            "--validate-checked",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validated"] is True
