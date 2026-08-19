"""Tensor-space constraint search: derive the Einstein tensor by exhaustion.

Einstein's 1915 result has a mechanizable core.  *Given* (i) the framework of pseudo-Riemannian
geometry, (ii) the restriction to generally covariant tensors built from the metric with at most
``k`` derivatives, (iii) the divergence-free requirement that conservation of the source forces on
the geometry side, and (iv) the Newtonian limit, the field equations stop being a choice.
Lovelock's theorem says that in four dimensions the only surviving tensor is ``a G_mn + L g_mn``.
That last step is a *uniqueness search over a constrained space*, and a uniqueness search over a
constrained space is a thing a machine can do.  This module does it.

What the module is given and what it derives
--------------------------------------------
GIVEN, declared in ``configs/tensor_constraint_search.json`` and never discovered here: the
manifold, the metric, the Levi-Civita connection, the Riemann tensor, the notion of a covariant
derivative, the class of concomitants searched (polynomial in curvature and its covariant
derivatives), the tensor rank and symmetry, the derivative order, and which constraints are on.

DERIVED, recomputed on every run and never transcribed: the basis of admissible terms at the
declared order, obtained by mechanically enumerating every contraction pattern and then *measuring*
how far the Riemann symmetries and the first Bianchi identity collapse it; the dimension of the
coefficient space after each constraint; the surviving family; the constant that the Newtonian
limit fixes; and what changes in five and six dimensions and at derivative order four.

Method
------
A generally covariant concomitant ``T(g, dg, ..., d^k g)`` evaluated at a point is a function of the
``k``-jet of the metric at that point, and that jet is an *unconstrained* point of jet space: any
symmetric-matrix-valued Taylor polynomial is the jet of some metric.  So ``T`` vanishes identically
if and only if it vanishes on every jet, and the search reduces to linear algebra over a bank of
randomly drawn metric jets.  Jets are carried exactly in a truncated multivariate Taylor ring over
a prime field, so no floating point appears anywhere in the derivation, and every run is replayed
under a second prime.

The sampling error is one-sided, and that is what makes the result honest.  A finite bank of jets
imposes a *subset* of the identical constraints, so the nullspace it reports can only ever come out
too *large*.  Uniqueness is therefore never manufactured by sampling.  It is closed from the other
side by exhibiting explicit members -- ``g_mn`` and ``G_mn`` -- and verifying each of them on
held-out jets that took no part in the rank computation.  When the exhibited count meets the sampled
nullity the sandwich is tight and the dimension is exact within the declared basis.

Keeping that guarantee across the subtraction
---------------------------------------------
The reported dimension is a *distinct-tensor* dimension, so a coefficient nullity has to have the
identically-vanishing directions taken out of it.  That subtraction is where a one-sided guarantee
goes to die: the sampled identically-vanishing dimension is itself an upper bound, so subtracting it
pushes the answer *down*, and the difference of two upper bounds has no guaranteed sign at all.  On
a one-jet bank in ``d = 4`` at order 2 that difference used to report 1 while the same call
exhibited 2 independent members -- below its own lower bound.

So nothing sampled is ever subtracted.  The only directions removed are **witnessed** ones:
declared vectors, each written from the literature and then confronted with the bank *and* the
holdout.  A witness is a lower bound on the true vanishing dimension, so taking it out leaves the
error still running upward.  If the sampled vanishing space is larger than the witnessed one there
is a direction nobody can name, and the search reports an open upper bound and refuses to publish a
dimension.  Every consumer of a dimension -- the reduction tables, the Gauss-Bonnet verdict, the
relaxation controls, the headline counts -- goes through :func:`published_dimension` and is refused
there rather than being handed a number that looks like a measurement and is not one.  Two open
upper bounds must never be differenced into a verdict.

What is emphatically NOT claimed
--------------------------------
1. Nothing here is novel.  Vermeil (1917), Weyl (1921), Cartan (1922), Lanczos (1938) and Lovelock
   (1971, 1972) are the mathematics; this is an engine capability demonstration reproducing them.
2. The *framework was declared, not discovered*.  The engine did not invent Riemannian geometry,
   did not decide the source should be conserved, and did not choose the Newtonian limit.  It was
   handed all four inputs and worked out what they force.  That is the boundary of the result and
   it is carried as a frozen claim in every receipt.  The declaration is now load-bearing rather
   than decorative -- see below.
3. Uniqueness is uniqueness *within the declared basis and derivative order*.  That the enumerated
   polynomial basis also exhausts the wider class of non-polynomial concomitants is a cited theorem
   (Lovelock 1972), not an output of this search.
4. The cosmological constant comes out **unforced**.  The Newtonian limit fixes the coefficient of
   ``G_mn`` and says nothing about the coefficient of ``g_mn``.  That is historically exactly right,
   and the module reports it rather than hiding it.

Sharp tests the search has to survive
-------------------------------------
* Drop ``divergence_free`` and the space must get strictly *larger*.
* Drop ``generally_covariant`` and the request must be *refused* with a typed blocker, because the
  enumeration has no meaning outside a covariant basis.
* A fabricated "divergence-free" tensor, ``R_mn - R g_mn / 3``, must be caught by the check.
* In ``d = 4`` the Gauss-Bonnet term is topological: the Euler-Lagrange derivative of
  ``R^2 - 4 R_ab R^ab + R_abcd R^abcd`` -- the Lanczos tensor -- must come out *identically zero*
  componentwise and contribute nothing to the field equations.  In ``d = 5`` and ``d = 6`` the same
  tensor must be non-zero and divergence-free, adding exactly one dimension.  That pair is the
  classic trap and it runs as an abort-on-failure control.

The declared framework drives the computation
---------------------------------------------
``declared_framework`` used to be inert.  It appeared at exactly two places in this file -- a key
name in the receipt key set, and ``dict(config["declared_framework"])``, which copied it verbatim
into the receipt -- and at zero places in the test file.  Handing the module a config that declared
flat Euclidean 3-space, a displacement field and "polynomial in the linear strain tensor" returned
``['g_mn', 'R_mn - (1/2) R g_mn']``, the general-relativity answer, because the basis was generated
from the Riemann tensor regardless of what had been declared.  A receipt could therefore carry an
elasticity framework claim over a general-relativity computation, and the paragraph above presented
that block as the boundary of the claim.  Two things now close that:

1. **The declaration selects the generator.**  ``declared_framework.machine_checkable`` is a typed
   sub-block, and its ``concomitant_generator`` field routes the run.  ``riemann_tensor`` runs the
   curvature search that derives ``a G_mn + L g_mn``.  ``linear_strain_tensor`` runs a genuinely
   different search on flat Euclidean space with a displacement field, which derives the
   two-dimensional isotropic elasticity tensor -- the Lame constants -- and the Navier-Cauchy
   equilibrium operator ``mu lap u + (lambda + mu) grad div u`` those constants force.  Neither
   answer is reachable from the other's declaration.  A framework block of prose alone, or one
   naming a generator this module does not implement, is REFUSED rather than served by whichever
   generator happens to be wired in.

2. **The declaration is then confronted with the computation.**  ``measure_realized_framework`` and
   ``measure_realized_elastic_framework`` read the framework back off the arrays the generator
   actually built: the signature of the base-point metric, whether the Riemann tensor of the
   sampled jets is identically zero, the index count and pair symmetry of the primary field jet,
   the rank and symmetry of the produced terms, and -- for the generator identity itself -- which
   terms die on an *exactly flat* metric jet, or whether the produced stress vanishes on an
   infinitesimal *rigid motion*.  Those two probes are what make the generator identity a
   measurement instead of a label.  Any field-level disagreement raises ``FrameworkMismatch`` and
   no receipt is sealed.  The original false declaration is pinned as an abort-on-failure control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256

#: Schemas.  Bump only with a receipt-shape change.
CONFIG_SCHEMA = "invariant-tensor-constraint-search-config-1.1"
RESULT_SCHEMA = "invariant-tensor-constraint-search-result-1.1"

#: Repository-relative paths this module binds to itself.
CONFIG_PATH = "configs/tensor_constraint_search.json"
ELASTICITY_CONFIG_PATH = "configs/tensor_constraint_elasticity.json"
SOURCE_PATH = "src/sigma_theory_compiler/tensor_constraint_search.py"
TEST_PATH = "tests/test_tensor_constraint_search.py"
FRAMEWORK_TEST_PATH = "tests/test_tensor_constraint_framework.py"
OUTPUT_PATH = "runs/math/tensor-constraint/lovelock-v1.json"
ELASTICITY_OUTPUT_PATH = "runs/math/tensor-constraint/elasticity-v1.json"

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.
CLAIMS: dict[str, bool] = {
    "declared_framework_selects_the_basis_generator": True,
    "declared_framework_is_gated_against_the_measured_computation": True,
    "derivation_is_symbolic_and_checked": True,
    "framework_was_declared_not_discovered": True,
    "novelty_claimed": False,
    "uniqueness_is_within_declared_basis_and_order": True,
}

#: Exact top-level receipt key set for the Riemann-generator (Lovelock) framework.
_TOP_KEYS = {
    "arithmetic",
    "basis_completeness_citations",
    "claims",
    "config_sha256",
    "content_sha256",
    "counts",
    "decision",
    "declared_framework",
    "framework_consistency",
    "generalizations",
    "negative_controls",
    "newtonian_limit",
    "reduction_tables",
    "relaxation_controls",
    "reused_repository_machinery",
    "schema_version",
    "scope",
    "searches",
    "source_bindings",
}

#: Exact top-level receipt key set for the linear-strain-generator (elasticity) framework.
_TOP_KEYS_STRAIN = {
    "arithmetic",
    "basis_completeness_citations",
    "claims",
    "config_sha256",
    "content_sha256",
    "counts",
    "decision",
    "declared_framework",
    "equilibrium_operator",
    "framework_consistency",
    "generalizations",
    "negative_controls",
    "reduction_tables",
    "relaxation_controls",
    "reused_repository_machinery",
    "schema_version",
    "scope",
    "searches",
    "source_bindings",
}

#: Refuse to persist an absolute host path into a receipt.
_HOST_PATH = re.compile(r"[A-Za-z]:\\|/(?:home|Users)/")


class TensorConstraintSearchError(ValueError):
    """Raised when the search, a control, or a receipt validation fails closed."""


class ConstraintOutOfScope(TensorConstraintSearchError):
    """Typed blocker: a declared constraint set the search refuses to run.

    Dropping general covariance is the motivating case.  Without it the admissible-term basis is
    not finite-dimensional and the enumeration means nothing, so the honest answer is a refusal
    rather than a confident answer over a basis that was never justified.
    """


class SandwichNotTight(TensorConstraintSearchError):
    """Typed blocker: a dimension was asked for that the run could only bound from one side.

    A search whose sandwich did not close reports an *open upper bound*, not a dimension.  Two open
    upper bounds cannot be subtracted, compared or differenced into a verdict, so every consumer of
    a search dimension goes through :func:`published_dimension` and is refused here rather than
    being handed a number that looks like a measurement and is not one.
    """


class FrameworkMismatch(TensorConstraintSearchError):
    """Typed blocker: the declared framework is not the framework the basis generator ran.

    Before this gate existed the ``declared_framework`` block was inert prose: it was copied
    verbatim into the receipt and referenced nowhere else, so a receipt could carry an elasticity
    framework claim over a general-relativity computation and nothing in the module would notice.
    The gate closes that hole by *measuring* the framework off the arrays the search actually
    built -- the signature of the base-point metric, whether the curvature of the sampled jets is
    identically zero, the index structure of the primary field jet, the rank and the pair symmetry
    of the produced basis terms, and whether the produced terms are curvature-driven -- and
    refusing to assemble a receipt when any measured field disagrees with what was declared.
    """


#: The typed, machine-checkable half of a ``declared_framework`` block.  Every one of these is
#: MEASURED off the running computation and compared; free-text keys alongside them are prose and
#: are carried but never trusted.
FRAMEWORK_FIELDS: tuple[str, ...] = (
    "metric_signature",
    "curvature",
    "primary_field_indices",
    "primary_field_symmetric",
    "tensor_rank",
    "tensor_symmetry",
    "concomitant_generator",
)

#: Generators this module can actually run.  A declaration naming anything else is refused rather
#: than silently served by whichever generator happens to be wired in.
SUPPORTED_GENERATORS: tuple[str, ...] = ("riemann_tensor", "linear_strain_tensor")


# ---------------------------------------------------------------------------
# Declared basis of admissible terms, and the classification that makes it complete.
# ---------------------------------------------------------------------------

#: Named rank-2 symmetric term generators, in order.  ``weight`` is the number of metric
#: derivatives the term carries.  These names index every coefficient vector in every receipt.
NAMED_TERMS: tuple[tuple[str, int, str], ...] = (
    ("g", 0, "g_mn"),
    ("Ric", 2, "R_mn"),
    ("Rg", 2, "R g_mn"),
    ("R2g", 4, "R^2 g_mn"),
    ("RRic", 4, "R R_mn"),
    ("RicSqg", 4, "R_ab R^ab g_mn"),
    ("RicRic", 4, "R_ma R^a_n"),
    ("RiemRic", 4, "R_manb R^ab"),
    ("RiemSqg", 4, "R_abcd R^abcd g_mn"),
    ("RiemRiem", 4, "R_mabc R_n^abc"),
    ("DDR", 4, "nabla_m nabla_n R"),
    ("BoxRg", 4, "box R g_mn"),
    ("BoxRic", 4, "box R_mn"),
)

_TERM_TEXT = {name: text for name, _, text in NAMED_TERMS}

#: Contraction-pattern generators used by the mechanical enumeration: ``(weight, slot_count)``.
PATTERN_GENERATORS: dict[str, tuple[int, int]] = {
    "RIEM": (2, 4),
    "DRIEM": (3, 5),
    "DDRIEM": (4, 6),
}

#: Declared coefficient vectors, in the ``NAMED_TERMS`` coordinate system.  These are the members
#: the search must *exhibit* to close the uniqueness sandwich from below, plus the two textbook
#: traps.  Each is checked, never assumed.
DECLARED_VECTORS: dict[str, dict[str, str]] = {
    "cosmological": {"g": "1"},
    "einstein": {"Ric": "1", "Rg": "-1/2"},
    "fabricated_third": {"Ric": "1", "Rg": "-1/3"},
    "gauss_bonnet_lanczos": {
        "R2g": "-1/2",
        "RRic": "2",
        "RicSqg": "2",
        "RicRic": "-4",
        "RiemRic": "-4",
        "RiemSqg": "-1/2",
        "RiemRiem": "2",
    },
    # The f(R) field equation f'(R) R_mn - f(R) g_mn/2 + (g_mn box - nabla_m nabla_n) f'(R),
    # evaluated at f(R) = R^2.  Written from the cited field-equation form, NOT read back off this
    # engine's own output, so that exhibiting it is an independent certificate.
    "quadratic_r_squared_euler_lagrange": {
        "R2g": "-1/2",
        "RRic": "2",
        "DDR": "-2",
        "BoxRg": "2",
    },
    # E2_mn: the metric Euler-Lagrange derivative of sqrt(-g) R_ab R^ab, written from the standard
    # variation and NOT read back off this engine.  Deriving it needs only the Palatini identity,
    # the contracted Bianchi identity nabla_a R^ab = (1/2) nabla^b R, and the Ricci commutator
    # [nabla_a, nabla_m] R^a_n = R_mb R^b_n - R_manb R^ab:
    #     E2_mn = -(1/2) R_ab R^ab g_mn + 2 R_manb R^ab + box R_mn + (1/2) box R g_mn
    #             - nabla_m nabla_n R
    # The R_ma R^a_n contributions from varying the two inverse metrics and from the commutator
    # cancel exactly, which is the well-known feature of this particular Euler-Lagrange derivative.
    "quadratic_ricci_squared_euler_lagrange": {
        "RicSqg": "-1/2",
        "RiemRic": "2",
        "DDR": "-1",
        "BoxRg": "1/2",
        "BoxRic": "1",
    },
    # E3_mn: the metric Euler-Lagrange derivative of sqrt(-g) R_abcd R^abcd, likewise written from
    # the standard variation:
    #     E3_mn = -(1/2) R_abcd R^abcd g_mn + 2 R_mabc R_n^abc - 4 R_ma R^a_n + 4 R_manb R^ab
    #             + 4 box R_mn - 2 nabla_m nabla_n R
    "quadratic_riemann_squared_euler_lagrange": {
        "RicRic": "-4",
        "RiemRic": "4",
        "RiemSqg": "-1/2",
        "RiemRiem": "2",
        "DDR": "-2",
        "BoxRic": "4",
    },
}

#: Declared vectors the search tries to exhibit, in receipt order.  These are the LOWER bound: each
#: is confronted with the bank and the holdout before it is credited, and their rank is what the
#: sampled nullity has to meet before any dimension is published.
EXHIBITED_LABELS: tuple[str, ...] = (
    "cosmological",
    "einstein",
    "gauss_bonnet_lanczos",
    "quadratic_r_squared_euler_lagrange",
    "quadratic_ricci_squared_euler_lagrange",
    "quadratic_riemann_squared_euler_lagrange",
)

#: The Gauss-Bonnet density is ``R^2 - 4 R_ab R^ab + R_abcd R^abcd``, so the Euler-Lagrange
#: operator -- which is linear -- must send it to the same combination of the three quadratic
#: Euler-Lagrange derivatives.  ``E1 - 4 E2 + E3`` must therefore equal the declared Lanczos vector
#: *exactly*, with all six derivative-carrying coefficients cancelling.  Each of the four vectors
#: was written independently from the literature, so this is a genuine cross-check of all four at
#: once and it is run as an abort-on-failure control in :func:`gauss_bonnet_decomposition`.
GAUSS_BONNET_DECOMPOSITION: tuple[tuple[str, str], ...] = (
    ("quadratic_r_squared_euler_lagrange", "1"),
    ("quadratic_ricci_squared_euler_lagrange", "-4"),
    ("quadratic_riemann_squared_euler_lagrange", "1"),
)

#: Cited classification results.  The enumeration below is mechanical, but the statement that the
#: enumerated polynomial basis exhausts the wider class of *non-polynomial* metric concomitants is
#: a theorem, not a search result, and is cited here rather than claimed.
BASIS_COMPLETENESS_CITATIONS: tuple[dict[str, str], ...] = (
    {
        "citation": "H. Vermeil, Nachr. Ges. Wiss. Goettingen (1917) 334",
        "statement": (
            "R and its metric multiples exhaust the scalar concomitants of g with at most two "
            "derivatives that are linear in the second derivatives."
        ),
    },
    {
        "citation": "H. Weyl, Raum-Zeit-Materie, 4th ed. (1921); E. Cartan, J. Math. Pures Appl. 1 "
        "(1922) 141",
        "statement": (
            "The only symmetric rank-2 concomitants of g with at most two derivatives, linear in "
            "the second derivatives, are spanned by g_mn, R_mn and R g_mn -- which is exactly the "
            "three-term basis this search re-derives by enumeration."
        ),
    },
    {
        "citation": "C. Lanczos, Ann. Math. 39 (1938) 842",
        "statement": (
            "The Euler-Lagrange derivative of the Gauss-Bonnet density vanishes identically in "
            "four dimensions."
        ),
    },
    {
        "citation": "D. Lovelock, J. Math. Phys. 12 (1971) 498",
        "statement": (
            "In dimension d the symmetric divergence-free rank-2 tensors that are second order in "
            "the metric are the Lanczos-Lovelock tensors; the Gauss-Bonnet member is the first one "
            "that is non-trivial, and it is non-trivial only for d > 4."
        ),
    },
    {
        "citation": "D. Lovelock, J. Math. Phys. 13 (1972) 874",
        "statement": (
            "In d = 4 the linearity-in-second-derivatives hypothesis can be dropped: any symmetric "
            "divergence-free A_mn(g, dg, ddg) equals a G_mn + L g_mn.  This is the step that lifts "
            "the search's within-basis uniqueness to uniqueness in the full concomitant class, and "
            "it is CITED here, not derived."
        ),
    },
    {
        "citation": "G. W. Horndeski, Int. J. Theor. Phys. 10 (1974) 363",
        "statement": (
            "With the declared field content relaxed from {metric} to {metric, scalar}, the same "
            "second-order-field-equation requirement yields the Horndeski family -- the axis along "
            "which this repository's screened-gravity candidates live."
        ),
    },
)

#: Completeness citations for the SECOND declared framework.  Same discipline: the enumeration is
#: mechanical, but the claim that the enumeration exhausts the invariant class is cited.
ELASTICITY_COMPLETENESS_CITATIONS: tuple[dict[str, str], ...] = (
    {
        "citation": "H. Weyl, The Classical Groups (1939), first fundamental theorem for O(n)",
        "statement": (
            "Every O(n)-invariant tensor is a linear combination of products of Kronecker deltas.  "
            "For rank 4 that is exactly the three delta pairings this search enumerates, so the "
            "enumeration is complete for isotropic materials.  CITED, not derived here."
        ),
    },
    {
        "citation": "A. L. Cauchy, Exercices de mathematiques 3 (1828) 160",
        "statement": (
            "The isotropic linear elastic law is sigma_ij = lambda d_ij tr(eps) + 2 mu eps_ij, and "
            "the static equilibrium operator it forces is mu lap u + (lambda + mu) grad div u -- "
            "the Navier-Cauchy equation.  This is what the elasticity search re-derives."
        ),
    },
    {
        "citation": "W. Voigt, Lehrbuch der Kristallphysik (1910), on the Cauchy relations",
        "statement": (
            "The rari-constant reduction lambda = mu is an additional hypothesis, not a "
            "consequence, and it is contradicted by measurement.  Imposing it here shrinks the "
            "space from two constants to one, which the search reports as a relaxation control."
        ),
    },
)

#: Repository machinery this module reuses rather than reimplementing.
REUSED_MACHINERY: tuple[dict[str, str], ...] = (
    {
        "artifact": "sigma_theory_compiler.relativity.schwarzschild_ricci_components",
        "kind": "python",
        "use": (
            "Cross-check of the curvature engine against an independent implementation: the "
            "repository's own sympy Schwarzschild Ricci computation is Ricci-flat, and the jet "
            "engine must agree that a Ricci-flat metric has vanishing Einstein tensor."
        ),
    },
    {
        "artifact": "formal/cadabra/contracted_bianchi.cdb",
        "kind": "text",
        "use": (
            "The contracted Bianchi identity nabla^a G_ab = 0 that makes the divergence-free "
            "constraint bite is already an established repository control.  It is hash-bound here "
            "and independently re-derived on the metric-jet bank, so the search does not lean on "
            "an unchecked import."
        ),
    },
    {
        "artifact": "formal/cadabra/einstein_hilbert_metric_variation.cdb",
        "kind": "text",
        "use": (
            "Varying the Einstein-Hilbert action is the complementary route to the same tensor.  "
            "Hash-bound as the control that this constraint search reproduces by a different "
            "method: no action is postulated here, only constraints."
        ),
    },
    {
        "artifact": "configs/actions/einstein_hilbert_control.json",
        "kind": "json",
        "use": "Declared Einstein-Hilbert action spec the variation control is registered against.",
    },
    {
        "artifact": "runs/gpu-baryonic-screen/nonlocal-localization-v1.json",
        "kind": "json",
        "use": (
            "The repository's surviving screened-gravity candidates.  Bound so the relaxation "
            "control can state exactly which declaration axis they live on."
        ),
    },
    {
        "artifact": "sigma_theory_compiler.sigma_core.canonical_json_bytes",
        "kind": "python",
        "use": "Float-rejecting canonical serialization used for every seal in this receipt.",
    },
)


# ---------------------------------------------------------------------------
# Truncated multivariate Taylor jets over a prime field.
# ---------------------------------------------------------------------------


def _compositions(dim: int, total: int) -> list[tuple[int, ...]]:
    """All exponent tuples of length ``dim`` summing to ``total``, lexicographically."""

    if dim == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for head in range(total, -1, -1):
        for tail in _compositions(dim - 1, total - head):
            out.append((head, *tail))
    return out


class _JetRing:
    """Taylor coefficients of a function of ``dim`` coordinates, truncated at total degree.

    A jet is the vector of Taylor coefficients ``f_a = d^a f(0) / a!`` indexed by multi-index ``a``
    with ``|a| <= degree``.  Exponents are ordered by total degree first, so a lower-degree ring's
    coefficient block is a *prefix* of a higher-degree ring's and truncation is a slice.
    """

    __slots__ = ("degree", "dim", "exponents", "index", "modulus", "products", "size")

    def __init__(self, dim: int, degree: int, modulus: int) -> None:
        self.dim = dim
        self.degree = degree
        self.modulus = modulus
        exponents: list[tuple[int, ...]] = []
        for total in range(degree + 1):
            exponents.extend(_compositions(dim, total))
        self.exponents = tuple(exponents)
        self.index = {exponent: position for position, exponent in enumerate(self.exponents)}
        self.size = len(self.exponents)
        products: list[tuple[int, int, int]] = []
        for left, left_exponent in enumerate(self.exponents):
            left_total = sum(left_exponent)
            for right, right_exponent in enumerate(self.exponents):
                if left_total + sum(right_exponent) > degree:
                    continue
                target = tuple(a + b for a, b in zip(left_exponent, right_exponent, strict=True))
                products.append((left, right, self.index[target]))
        self.products = tuple(products)


@cache
def _ring(dim: int, degree: int, modulus: int) -> _JetRing:
    """Rings are pure functions of ``(dim, degree, modulus)``; building the product table is
    quadratic in the ring size, so caching them is the difference between a fast search and a
    slow one when a bank holds a dozen metric jets."""

    return _JetRing(dim, degree, modulus)


class _JetCalculus:
    """Ring plumbing shared by every field carried as a truncated Taylor jet.

    Factored out so the elasticity generator can carry a displacement field with exactly the same
    exact arithmetic as the curvature generator carries a metric, *without* inheriting a
    connection or a Riemann tensor it never uses.  That separation is what makes the framework
    measurement meaningful: the two generators genuinely share only the arithmetic.
    """

    dim: int
    modulus: int
    rings: list[_JetRing]

    def ring(self, level: int) -> _JetRing:
        return self.rings[level]

    def cut(self, tensor: np.ndarray, level: int) -> np.ndarray:
        """Truncate a jet tensor to total degree ``level``."""

        return np.ascontiguousarray(tensor[..., : self.rings[level].size])

    def mul(self, spec: str, left: np.ndarray, right: np.ndarray, level: int) -> np.ndarray:
        """Contract two jet tensors with ``spec`` over tensor axes, convolving the jet axis."""

        ring = self.rings[level]
        left = self.cut(left, level)
        right = self.cut(right, level)
        probe = np.einsum(spec.replace("j", ""), left[..., 0], right[..., 0], optimize=True)
        out = np.zeros((*probe.shape, ring.size), dtype=np.int64)
        plain = spec.replace("j", "")
        for a, b, c in ring.products:
            out[..., c] += np.einsum(plain, left[..., a], right[..., b], optimize=True)
            out[..., c] %= self.modulus
        return out

    def diff(self, tensor: np.ndarray, level_in: int) -> np.ndarray:
        """Coordinate derivative, new index appended last; result lives one degree lower."""

        ring_in = self.rings[level_in]
        ring_out = self.rings[level_in - 1]
        out = np.zeros((*tensor.shape[:-1], self.dim, ring_out.size), dtype=np.int64)
        for position, exponent in enumerate(ring_out.exponents):
            for axis in range(self.dim):
                raised = list(exponent)
                raised[axis] += 1
                source = ring_in.index[tuple(raised)]
                out[..., axis, position] = (
                    tensor[..., source] * (exponent[axis] + 1)
                ) % self.modulus
        return out


class _Geometry(_JetCalculus):
    """Curvature of one metric jet, carried exactly over a prime field.

    Every tensor is a numpy ``int64`` array of shape ``(d,) * rank + (jet_size,)``.  Each derivative
    costs one jet degree, so the rings shrink as the pipeline advances and the expensive high-degree
    arithmetic stays confined to the metric itself.
    """

    def __init__(self, dim: int, metric_jet: np.ndarray, modulus: int, degree: int) -> None:
        if degree < 3:
            raise TensorConstraintSearchError("a divergence needs at least a 3-jet of the metric")
        self.dim = dim
        self.modulus = modulus
        # Facts about the primary field jet, READ OFF the array that was handed in rather than
        # asserted.  The framework gate consumes these, so they must never become declarations.
        self.primary_field_indices = int(metric_jet.ndim - 1)
        self.primary_field_symmetric = bool(
            self.primary_field_indices == 2
            and np.array_equal(metric_jet % modulus, np.einsum("abj->baj", metric_jet) % modulus)
        )
        self.half = pow(2, modulus - 2, modulus)
        self.rings = [_ring(dim, level, modulus) for level in range(degree + 1)]
        self.metric = metric_jet
        self.metric_level = degree
        self.inverse = self._invert_metric()
        self.gamma_level = degree - 1
        self.gamma = self._christoffel()
        self.riemann_level = degree - 2
        self.riemann, self.ricci, self.scalar = self._curvature()

    # -- curvature --------------------------------------------------------

    def _invert_metric(self) -> np.ndarray:
        """Neumann series inverse.

        ``g = eta + h`` with ``h`` purely higher order, so ``h`` is nilpotent in the truncated ring
        and ``g^-1 = sum_m (-eta h)^m eta`` terminates at the truncation degree.  The result is
        confronted with ``g g^-1 = I`` before it is used anywhere.
        """

        level = self.metric_level
        ring = self.rings[level]
        eta = np.zeros((self.dim, self.dim), dtype=np.int64)
        eta[0, 0] = self.modulus - 1
        for axis in range(1, self.dim):
            eta[axis, axis] = 1
        if not np.array_equal(self.metric[..., 0] % self.modulus, eta % self.modulus):
            raise TensorConstraintSearchError("metric jet must be Minkowskian at the base point")
        higher = self.metric.copy()
        higher[..., 0] = 0
        step = (-np.einsum("ab,bcj->acj", eta, higher, optimize=True)) % self.modulus
        inverse = np.zeros((self.dim, self.dim, ring.size), dtype=np.int64)
        inverse[..., 0] = eta % self.modulus
        term = inverse.copy()
        for _ in range(level):
            term = self.mul("abj,bcj->acj", step, term, level)
            inverse = (inverse + term) % self.modulus
        check = self.mul("abj,bcj->acj", self.metric, inverse, level)
        identity = np.zeros_like(check)
        identity[..., 0] = np.eye(self.dim, dtype=np.int64)
        if not np.array_equal(check % self.modulus, identity % self.modulus):
            raise TensorConstraintSearchError("metric inverse failed its own identity check")
        return inverse

    def _christoffel(self) -> np.ndarray:
        """``Gamma^a_bc`` with the raised index first."""

        level = self.gamma_level
        partial = self.diff(self.metric, self.metric_level)  # partial[i, j, k] = d_k g_ij
        combo = (
            np.einsum("dcbj->dbcj", partial)
            + np.einsum("bdcj->dbcj", partial)
            - np.einsum("bcdj->dbcj", partial)
        ) % self.modulus
        raised = self.mul("adj,dbcj->abcj", self.inverse, combo, level)
        return (raised * self.half) % self.modulus

    def _curvature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        level = self.riemann_level
        gamma = self.gamma
        d_gamma = self.diff(gamma, self.gamma_level)  # d_gamma[a, b, c, e] = d_e Gamma^a_bc
        first = self.cut(np.einsum("adbcj->abcdj", d_gamma), level)
        second = self.cut(np.einsum("acbdj->abcdj", d_gamma), level)
        quad = self.mul("acej,edbj->abcdj", gamma, gamma, level)
        mixed = (first - second + quad - np.einsum("abdcj->abcdj", quad)) % self.modulus
        lowered = self.mul("aej,ebcdj->abcdj", self.metric, mixed, level)
        ricci = np.einsum("abadj->bdj", mixed) % self.modulus
        scalar = self.mul("bdj,bdj->j", self.inverse, ricci, level)
        return lowered, ricci, scalar

    # -- covariant derivatives -------------------------------------------

    def covariant(self, tensor: np.ndarray, rank: int, level_in: int) -> np.ndarray:
        """``nabla_e T_{a1..an}`` with the derivative index FIRST; drops one jet degree."""

        level_out = level_in - 1
        if rank > 6:
            raise TensorConstraintSearchError("covariant derivative rank limit exceeded")
        letters = [chr(ord("a") + i) for i in range(rank)]
        joined = "".join(letters)
        # 'z' is the new derivative index and 'y' the Christoffel dummy; both sit outside the
        # slot alphabet so they cannot collide with a tensor index at any supported rank.
        out = np.einsum(f"{joined}zj->z{joined}j", self.diff(tensor, level_in)) % self.modulus
        gamma = self.cut(self.gamma, level_out)
        for slot in range(rank):
            replaced = list(letters)
            replaced[slot] = "y"
            spec = f"yz{letters[slot]}j,{''.join(replaced)}j->z{joined}j"
            out = (out - self.mul(spec, gamma, tensor, level_out)) % self.modulus
        return out

    def divergence(self, tensor: np.ndarray, level_in: int) -> np.ndarray:
        """``nabla^m T_mn`` for a rank-2 tensor; a rank-1 jet one degree lower."""

        nabla = self.covariant(tensor, 2, level_in)  # nabla[e, m, n]
        return self.mul("emj,emnj->nj", self.inverse, nabla, level_in - 1)


# ---------------------------------------------------------------------------
# Deterministic metric-jet bank.
# ---------------------------------------------------------------------------


def _stream(seed: str, counter: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{counter}".encode()).digest(), "big")


def metric_jet(dim: int, degree: int, modulus: int, seed: str, sample: int) -> np.ndarray:
    """A random symmetric metric jet that is exactly Minkowskian at the base point.

    Fixing ``g_ab(0) = eta_ab`` costs no generality -- a linear change of coordinates puts any
    metric in that form at a point -- and it guarantees invertibility over the prime field.  Every
    higher Taylor coefficient is drawn free, which is exactly the statement that the ``k``-jet of a
    metric is unconstrained, and therefore that a random jet probes an identity fully.
    """

    ring = _ring(dim, degree, modulus)
    jet = np.zeros((dim, dim, ring.size), dtype=np.int64)
    jet[0, 0, 0] = modulus - 1
    for axis in range(1, dim):
        jet[axis, axis, 0] = 1
    counter = 0
    label = f"{seed}|d{dim}|j{degree}|s{sample}"
    for left in range(dim):
        for right in range(left, dim):
            for position, exponent in enumerate(ring.exponents):
                if sum(exponent) == 0:
                    continue
                counter += 1
                value = _stream(label, counter) % modulus
                jet[left, right, position] = value
                jet[right, left, position] = value
    return jet


def build_bank(dim: int, order: int, modulus: int, seed: str, samples: int) -> list[_Geometry]:
    degree = order + 1
    return [
        _Geometry(dim, metric_jet(dim, degree, modulus, seed, sample), modulus, degree)
        for sample in range(samples)
    ]


# ---------------------------------------------------------------------------
# Named term construction.
# ---------------------------------------------------------------------------


def named_term_names(order: int) -> tuple[str, ...]:
    return tuple(name for name, weight, _ in NAMED_TERMS if weight <= order)


def named_tensors(geometry: _Geometry, order: int, level: int = 1) -> dict[str, np.ndarray]:
    """Build every named basis term of weight ``<= order``, as a jet tensor at degree ``level``."""

    if geometry.riemann_level < level:
        raise TensorConstraintSearchError("metric jet degree too low for the declared order")
    inverse = geometry.cut(geometry.inverse, level)
    metric = geometry.cut(geometry.metric, level)
    riemann = geometry.cut(geometry.riemann, level)
    ricci = geometry.cut(geometry.ricci, level)
    scalar = geometry.cut(geometry.scalar, level)

    terms: dict[str, np.ndarray] = {"g": metric, "Ric": ricci}
    terms["Rg"] = geometry.mul("j,mnj->mnj", scalar, metric, level)

    if order >= 4:
        if geometry.riemann_level != level + 2:
            raise TensorConstraintSearchError(
                "order-4 terms need the curvature two jet degrees above the term level"
            )
        ricci_mixed = geometry.mul("acj,cbj->abj", inverse, ricci, level)  # R^a_b
        ricci_up = geometry.mul("acj,cbj->abj", ricci_mixed, inverse, level)  # R^ab
        riem_up3 = geometry.mul("mabcj,adj->mdbcj", riemann, inverse, level)
        riem_up3 = geometry.mul("mdbcj,bej->mdecj", riem_up3, inverse, level)
        riem_up3 = geometry.mul("mdecj,cfj->mdefj", riem_up3, inverse, level)  # R_m^{abc}
        riem_up4 = geometry.mul("mdefj,mgj->gdefj", riem_up3, inverse, level)  # R^{abcd}

        scalar_sq = geometry.mul("j,j->j", scalar, scalar, level)
        ricci_sq = geometry.mul("abj,abj->j", ricci, ricci_up, level)
        riemann_sq = geometry.mul("mdefj,mdefj->j", riemann, riem_up4, level)

        terms["R2g"] = geometry.mul("j,mnj->mnj", scalar_sq, metric, level)
        terms["RRic"] = geometry.mul("j,mnj->mnj", scalar, ricci, level)
        terms["RicSqg"] = geometry.mul("j,mnj->mnj", ricci_sq, metric, level)
        terms["RicRic"] = geometry.mul("maj,anj->mnj", ricci, ricci_mixed, level)
        terms["RiemRic"] = geometry.mul("manbj,abj->mnj", riemann, ricci_up, level)
        terms["RiemSqg"] = geometry.mul("j,mnj->mnj", riemann_sq, metric, level)
        terms["RiemRiem"] = geometry.mul("mabcj,nabcj->mnj", riemann, riem_up3, level)

        top = geometry.riemann_level
        d_scalar = geometry.covariant(geometry.scalar, 0, top)
        dd_scalar = geometry.cut(geometry.covariant(d_scalar, 1, top - 1), level)
        d_ricci = geometry.covariant(geometry.ricci, 2, top)
        dd_ricci = geometry.cut(geometry.covariant(d_ricci, 3, top - 1), level)
        box_scalar = geometry.mul("mnj,mnj->j", inverse, dd_scalar, level)
        terms["DDR"] = dd_scalar
        terms["BoxRg"] = geometry.mul("j,mnj->mnj", box_scalar, metric, level)
        terms["BoxRic"] = geometry.mul("abj,abmnj->mnj", inverse, dd_ricci, level)

    return {
        name: ((tensor + np.einsum("mnj->nmj", tensor)) * geometry.half) % geometry.modulus
        for name, tensor in terms.items()
    }


# ---------------------------------------------------------------------------
# Framework measurement and the declared-vs-realized consistency gate.
# ---------------------------------------------------------------------------


def flat_metric_jet(dim: int, degree: int, modulus: int) -> np.ndarray:
    """The exactly flat jet: ``eta`` at the base point and every higher coefficient zero.

    Its Riemann tensor vanishes identically, so evaluating the named basis on it *separates* the
    terms that need curvature from the terms that do not.  That separation is the measurement the
    framework gate uses to decide which generator actually drove the basis, and it is a
    computation on real arrays rather than a label attached at the call site.
    """

    ring = _ring(dim, degree, modulus)
    jet = np.zeros((dim, dim, ring.size), dtype=np.int64)
    jet[0, 0, 0] = modulus - 1
    for axis in range(1, dim):
        jet[axis, axis, 0] = 1
    return jet


def _signature_name(diagonal: Sequence[int], modulus: int) -> str:
    negatives = sum(1 for value in diagonal if value == modulus - 1)
    positives = sum(1 for value in diagonal if value == 1)
    if negatives + positives != len(diagonal):
        return "non-normalised"
    if negatives == 0:
        return "euclidean"
    if negatives == 1:
        return "lorentzian"
    return f"indefinite-{negatives}-{positives}"


def measure_realized_framework(
    bank: Sequence[_Geometry],
    bank_terms: Sequence[Mapping[str, np.ndarray]],
    *,
    dimension: int,
    order: int,
    modulus: int,
) -> dict[str, Any]:
    """Measure the framework the basis generator actually ran, off its own arrays.

    Nothing in here reads the config.  Every reported field is a property of the metric jets that
    were sampled and the basis terms that were produced, so the record can be compared against the
    declaration without the comparison degenerating into a declaration checked against itself.
    """

    reference = bank[0]
    base = reference.metric[..., 0] % modulus
    diagonal = [int(base[axis, axis]) for axis in range(dimension)]
    off_diagonal = base.copy()
    np.fill_diagonal(off_diagonal, 0)
    signature = (
        _signature_name(diagonal, modulus) if not np.any(off_diagonal) else "non-normalised"
    )

    curved = any(bool(np.any(geometry.riemann % modulus)) for geometry in bank)

    names = named_term_names(order)
    ranks = {int(bank_terms[0][name].ndim - 1) for name in names}
    if len(ranks) != 1:
        raise FrameworkMismatch("the basis generator produced terms of mixed tensor rank")
    tensor_rank = ranks.pop()

    symmetric = True
    any_nonzero = False
    for terms in bank_terms:
        for name in names:
            tensor = terms[name] % modulus
            any_nonzero = any_nonzero or bool(np.any(tensor))
            if not np.array_equal(tensor, np.einsum("mnj->nmj", tensor) % modulus):
                symmetric = False
    tensor_symmetry = "symmetric" if (symmetric and any_nonzero) else "none"

    flat = _Geometry(
        dimension, flat_metric_jet(dimension, order + 1, modulus), modulus, order + 1
    )
    flat_terms = named_tensors(flat, order)
    curvature_driven: list[str] = []
    curvature_free: list[str] = []
    for name in names:
        zero_on_flat = not bool(np.any(flat_terms[name] % modulus))
        nonzero_on_bank = any(bool(np.any(terms[name] % modulus)) for terms in bank_terms)
        (curvature_driven if (zero_on_flat and nonzero_on_bank) else curvature_free).append(name)

    generator = "riemann_tensor" if curvature_driven else "curvature_free_unidentified"

    return {
        "metric_signature": signature,
        "curvature": "generic" if curved else "flat",
        "primary_field_indices": int(reference.primary_field_indices),
        "primary_field_symmetric": bool(reference.primary_field_symmetric),
        "tensor_rank": int(tensor_rank),
        "tensor_symmetry": tensor_symmetry,
        "concomitant_generator": generator,
        "measurement_evidence": {
            "base_point_metric_diagonal_is_pm_one": bool(signature != "non-normalised"),
            "riemann_nonzero_on_sampled_jets": bool(curved),
            "terms_vanishing_on_the_exactly_flat_jet": curvature_driven,
            "terms_surviving_on_the_exactly_flat_jet": curvature_free,
            "how_generator_was_identified": (
                "the named basis was evaluated on an exactly flat jet whose Riemann tensor is "
                "identically zero; terms that die there and live on the curved bank are driven by "
                "the Riemann tensor, and a non-empty set of them identifies the generator"
            ),
        },
    }


def check_framework_consistency(
    declared: Mapping[str, Any], realized: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Compare a typed declaration against a measured realization; refuse on any disagreement.

    Fails closed three ways: an absent typed block (a free-text framework cannot be checked, so it
    is refused rather than waved through), an unsupported generator name, and any field-level
    disagreement.  The mismatch report names every offending field with both values, because a
    blocker that does not say what is wrong is only marginally better than no blocker.
    """

    missing = [field for field in FRAMEWORK_FIELDS if field not in declared]
    if missing:
        raise FrameworkMismatch(
            f"[{context}] the declared framework carries no machine-checkable value for "
            f"{sorted(missing)}; a framework that cannot be checked against the computation is "
            "refused rather than copied into a receipt"
        )
    if declared["concomitant_generator"] not in SUPPORTED_GENERATORS:
        raise FrameworkMismatch(
            f"[{context}] declared concomitant_generator "
            f"{declared['concomitant_generator']!r} is not one this module implements "
            f"{list(SUPPORTED_GENERATORS)}; refusing to serve it with a different generator"
        )

    mismatches = [
        {
            "field": field,
            "declared": declared[field],
            "realized": realized[field],
        }
        for field in FRAMEWORK_FIELDS
        if declared[field] != realized[field]
    ]
    if mismatches:
        detail = "; ".join(
            f"{item['field']}: declared {item['declared']!r} but the computation realized "
            f"{item['realized']!r}"
            for item in mismatches
        )
        raise FrameworkMismatch(
            f"[{context}] the declared framework is not the framework the basis generator ran -- "
            f"{detail}.  Refusing to seal a receipt whose framework block describes a different "
            "computation from the one that produced its numbers."
        )
    return {
        "context": context,
        "checked_fields": list(FRAMEWORK_FIELDS),
        "declared": {field: declared[field] for field in FRAMEWORK_FIELDS},
        "realized": {field: realized[field] for field in FRAMEWORK_FIELDS},
        "measurement_evidence": dict(realized.get("measurement_evidence", {})),
        "agreed": True,
    }


# ---------------------------------------------------------------------------
# Mechanical contraction-pattern enumeration.
# ---------------------------------------------------------------------------


def _perfect_matchings(items: Sequence[int]) -> list[tuple[tuple[int, int], ...]]:
    if not items:
        return [()]
    head, *rest = items
    out: list[tuple[tuple[int, int], ...]] = []
    for position in range(len(rest)):
        remainder = rest[:position] + rest[position + 1 :]
        for tail in _perfect_matchings(remainder):
            out.append(((head, rest[position]), *tail))
    return out


def _factor_multisets(order: int) -> list[tuple[str, ...]]:
    """Every multiset of curvature generators whose total weight is at most ``order``."""

    names = sorted(PATTERN_GENERATORS)
    out: list[tuple[str, ...]] = [()]
    frontier: list[tuple[str, ...]] = [()]
    while frontier:
        nxt: list[tuple[str, ...]] = []
        for current in frontier:
            weight = sum(PATTERN_GENERATORS[name][0] for name in current)
            for name in names:
                if current and name < current[-1]:
                    continue
                if weight + PATTERN_GENERATORS[name][0] > order:
                    continue
                candidate = (*current, name)
                nxt.append(candidate)
                out.append(candidate)
        frontier = nxt
    return out


def enumerate_patterns(order: int) -> list[dict[str, Any]]:
    """Enumerate every rank-2 contraction pattern of weight ``<= order``.

    A pattern is a multiset of curvature factors with all indices down, a choice of either two free
    slots or a ``g_mn`` carrier, and a perfect matching of the remaining slots contracted with the
    inverse metric.  Nothing about the Riemann symmetries is used here.  That collapse is *measured*
    afterwards on the metric-jet bank, which is the point: the symmetries and the first Bianchi
    identity are discovered by the search, not asserted to it.
    """

    patterns: list[dict[str, Any]] = []
    for factors in _factor_multisets(order):
        slots: list[tuple[int, int]] = []
        for position, name in enumerate(factors):
            slots.extend((position, slot) for slot in range(PATTERN_GENERATORS[name][1]))
        total = len(slots)
        for carrier in (False, True):
            free_count = 0 if carrier else 2
            if total < free_count or (total - free_count) % 2 != 0:
                continue
            if not carrier and total == 0:
                continue
            for free in combinations(range(total), free_count):
                chosen = set(free)
                remaining = [i for i in range(total) if i not in chosen]
                for matching in _perfect_matchings(remaining):
                    patterns.append(
                        {
                            "factors": factors,
                            "carrier": carrier,
                            "free": free,
                            "matching": matching,
                            "slot_count": total,
                        }
                    )
    return patterns


_LETTERS = "abcdefghijklmnopqrstuvwx"


def _pattern_factor(geometry: _Geometry, name: str) -> np.ndarray:
    """Evaluate one curvature generator at jet degree zero."""

    if name == "RIEM":
        return geometry.cut(geometry.riemann, 0)
    if name == "DRIEM":
        return geometry.cut(geometry.covariant(geometry.riemann, 4, geometry.riemann_level), 0)
    if name == "DDRIEM":
        first = geometry.covariant(geometry.riemann, 4, geometry.riemann_level)
        return geometry.cut(geometry.covariant(first, 5, geometry.riemann_level - 1), 0)
    raise TensorConstraintSearchError(f"unknown pattern generator {name}")


def evaluate_pattern(
    geometry: _Geometry,
    pattern: Mapping[str, Any],
    cache: dict[str, np.ndarray],
    *,
    symmetrise: bool,
) -> np.ndarray:
    """Evaluate one contraction pattern as a rank-2 array at the base point."""

    operands: list[np.ndarray] = []
    specs: list[str] = []
    letters = [_LETTERS[i] for i in range(pattern["slot_count"])]
    cursor = 0
    for name in pattern["factors"]:
        count = PATTERN_GENERATORS[name][1]
        if name not in cache:
            cache[name] = _pattern_factor(geometry, name)[..., 0]
        specs.append("".join(letters[cursor : cursor + count]))
        operands.append(cache[name])
        cursor += count
    inverse = geometry.cut(geometry.inverse, 0)[..., 0]
    for left, right in pattern["matching"]:
        specs.append(letters[left] + letters[right])
        operands.append(inverse)
    if pattern["carrier"]:
        out_letters = "yz"
        specs.append("yz")
        operands.append(geometry.cut(geometry.metric, 0)[..., 0])
    else:
        out_letters = letters[pattern["free"][0]] + letters[pattern["free"][1]]
    value = _modular_einsum(specs, out_letters, operands, geometry.modulus)
    if not symmetrise:
        return value
    return ((value + value.T) * geometry.half) % geometry.modulus


def _modular_einsum(
    specs: Sequence[str], output: str, operands: Sequence[np.ndarray], modulus: int
) -> np.ndarray:
    """Multi-operand contraction folded pairwise, reducing modulo the prime at every step.

    A single ``np.einsum`` over five or six residue arrays silently overflows ``int64`` -- the
    entries are of order the prime, so a five-fold product is of order ``p^5``.  Folding the
    contraction two operands at a time and reducing after each step keeps every intermediate below
    ``p^2 * d^k``, which is comfortably inside ``int64`` for the dimensions searched here.
    """

    terms = [str(item) for item in specs]
    arrays = [np.asarray(item, dtype=np.int64) % modulus for item in operands]
    while len(arrays) > 1:
        best: tuple[tuple[int, int], int, int, str] | None = None
        for left in range(len(arrays)):
            for right in range(left + 1, len(arrays)):
                survivors = set(output)
                for other in range(len(terms)):
                    if other not in (left, right):
                        survivors |= set(terms[other])
                union = set(terms[left]) | set(terms[right])
                keep = "".join(sorted(union & survivors))
                score = (len(keep), len(union))
                if best is None or score < best[0]:
                    best = (score, left, right, keep)
        assert best is not None
        _, left, right, keep = best
        merged = (
            np.einsum(f"{terms[left]},{terms[right]}->{keep}", arrays[left], arrays[right],
                      optimize=True)
            % modulus
        )
        for position in sorted((left, right), reverse=True):
            terms.pop(position)
            arrays.pop(position)
        terms.append(keep)
        arrays.append(merged)
    if terms[0] != output:
        return np.einsum(f"{terms[0]}->{output}", arrays[0]) % modulus
    return arrays[0]


# ---------------------------------------------------------------------------
# Linear algebra over the prime field.
# ---------------------------------------------------------------------------


def _rref(matrix: Sequence[Sequence[int]], modulus: int) -> tuple[list[list[int]], list[int]]:
    rows = [list(row) for row in matrix]
    pivots: list[int] = []
    if not rows:
        return rows, pivots
    row_index = 0
    for column in range(len(rows[0])):
        pivot = next((r for r in range(row_index, len(rows)) if rows[r][column] % modulus), None)
        if pivot is None:
            continue
        rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
        scale = pow(rows[row_index][column], modulus - 2, modulus)
        rows[row_index] = [(value * scale) % modulus for value in rows[row_index]]
        for other in range(len(rows)):
            if other == row_index:
                continue
            factor = rows[other][column] % modulus
            if factor:
                rows[other] = [
                    (a - factor * b) % modulus
                    for a, b in zip(rows[other], rows[row_index], strict=True)
                ]
        pivots.append(column)
        row_index += 1
        if row_index == len(rows):
            break
    return rows, pivots


def _rank(matrix: Sequence[Sequence[int]], modulus: int) -> int:
    return len(_rref(matrix, modulus)[1])


def _nullspace(rows: Sequence[Sequence[int]], modulus: int) -> list[list[int]]:
    """Basis of ``{c : sum_i c_i * rows[i] == 0}``, i.e. the left nullspace of ``rows``."""

    if not rows:
        return []
    transposed = [list(column) for column in zip(*rows, strict=True)]
    reduced, pivots = _rref(transposed, modulus)
    width = len(rows)
    pivot_set = set(pivots)
    basis: list[list[int]] = []
    for column in range(width):
        if column in pivot_set:
            continue
        vector = [0] * width
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = (-reduced[row][column]) % modulus
        basis.append(vector)
    return basis


def _rational(value: int, modulus: int) -> Fraction:
    """Rational reconstruction of a residue, so receipts read ``-1/2`` and not a big integer."""

    bound = int((modulus // 2) ** 0.5)
    previous, current = modulus, value % modulus
    previous_s, current_s = 0, 1
    while current > bound:
        quotient = previous // current
        previous, current = current, previous - quotient * current
        previous_s, current_s = current_s, previous_s - quotient * current_s
    if current_s == 0 or abs(current_s) > bound:
        raise TensorConstraintSearchError("rational reconstruction failed")
    fraction = Fraction(current, current_s) if current_s > 0 else Fraction(-current, -current_s)
    check = fraction.numerator * pow(fraction.denominator, modulus - 2, modulus)
    if (check - value) % modulus:
        raise TensorConstraintSearchError("rational reconstruction did not verify")
    return fraction


def _normalise(vector: Sequence[int], modulus: int) -> list[Fraction]:
    lead = next((value for value in vector if value % modulus), None)
    if lead is None:
        return [Fraction(0)] * len(vector)
    scale = pow(lead, modulus - 2, modulus)
    return [_rational((value * scale) % modulus, modulus) for value in vector]


def _vector_text(vector: Sequence[Fraction], names: Sequence[str]) -> str:
    parts: list[str] = []
    for coefficient, name in zip(vector, names, strict=True):
        if coefficient == 0:
            continue
        label = _TERM_TEXT.get(name, name)
        if coefficient == 1:
            parts.append(f"+ {label}")
        elif coefficient == -1:
            parts.append(f"- {label}")
        elif coefficient > 0:
            parts.append(f"+ ({coefficient}) {label}")
        else:
            parts.append(f"- ({-coefficient}) {label}")
    text = " ".join(parts)
    return text.removeprefix("+ ")


def _declared_vector(name: str, names: Sequence[str], modulus: int) -> list[int]:
    spec = DECLARED_VECTORS[name]
    unknown = set(spec) - set(names)
    if unknown:
        raise TensorConstraintSearchError(f"declared vector {name} needs terms outside the basis")
    vector: list[int] = []
    for term in names:
        fraction = Fraction(spec.get(term, "0"))
        vector.append(
            (fraction.numerator * pow(fraction.denominator, modulus - 2, modulus)) % modulus
        )
    return vector


def _combine(vector: Sequence[int], tensors: Mapping[str, np.ndarray], names: Sequence[str],
             modulus: int) -> np.ndarray:
    total = np.zeros_like(tensors[names[0]])
    for coefficient, name in zip(vector, names, strict=True):
        if coefficient % modulus:
            total = (total + coefficient * tensors[name]) % modulus
    return total


def _in_span(vector: Sequence[int], basis: Sequence[Sequence[int]], modulus: int) -> bool:
    if not basis:
        return not any(value % modulus for value in vector)
    return _rank(list(basis), modulus) == _rank([*list(basis), list(vector)], modulus)


def _sample_rows(
    pairs: Sequence[tuple[_Geometry, Mapping[str, np.ndarray]]],
    names: Sequence[str],
    *,
    divergence: bool,
    full_jet: bool = False,
) -> list[list[int]]:
    """One row per named term, holding that term (or its divergence) over every sampled jet.

    ``full_jet`` keeps the whole truncated Taylor jet rather than only its value at the base point.
    A divergence already lives one jet degree lower, so for ``divergence=True`` the base point IS
    the whole jet and the flag makes no difference.  For the terms themselves it does: the linear
    Taylor coefficients say what the tensor is at neighbouring points of the same metric, and an
    identity has to hold there too.  Keeping them shrinks the sampled identically-vanishing space
    towards the true one, which is the safe direction for a quantity that gets subtracted.
    """

    rows: list[list[int]] = []
    for name in names:
        values: list[int] = []
        for geometry, terms in pairs:
            tensor = geometry.divergence(terms[name], 1) if divergence else terms[name]
            block = tensor if full_jet else tensor[..., 0]
            values.extend(int(value) for value in block.ravel())
        rows.append(values)
    return rows


def _vanishing_witnesses(
    pairs: Sequence[tuple[_Geometry, Mapping[str, np.ndarray]]],
    names: Sequence[str],
    modulus: int,
) -> list[tuple[str, list[int]]]:
    """Declared vectors that vanish componentwise on every sampled jet, bank and holdout alike.

    This is the ONLY thing the search is ever allowed to subtract from a nullity.  A sampled
    vanishing dimension is an upper bound and subtracting it tips the reported dimension below the
    truth; a witness is a named vector with a literature provenance, so crediting it is a lower
    bound on the true vanishing dimension and subtracting it keeps the error running upward.
    """

    witnesses: list[tuple[str, list[int]]] = []
    for label, spec in DECLARED_VECTORS.items():
        if label == "fabricated_third" or not set(spec) <= set(names):
            continue
        vector = _declared_vector(label, names, modulus)
        if all(not np.any(_combine(vector, terms, names, modulus) % modulus) for _, terms in pairs):
            witnesses.append((label, vector))
    return witnesses


def _vanishing_blocker(sampled: int, witnessed: int) -> dict[str, Any]:
    return {
        "type": "SandwichNotTight",
        "reason": "unwitnessed_identically_vanishing_directions",
        "sampled_identically_vanishing_dimension": sampled,
        "witnessed_identically_vanishing_dimension": witnessed,
        "detail": (
            f"the jet sample says {sampled} coefficient direction(s) vanish identically but only "
            f"{witnessed} of them are witnessed by a declared member.  A sampled vanishing "
            "dimension is an UPPER bound, so subtracting it would report the space too small.  "
            "Refusing to publish a dimension."
        ),
    }


def _sandwich_blocker(
    *,
    coefficient_nullity: int,
    exhibited_rank: int,
    sampled_vanishing: int,
    witnessed_vanishing: int,
) -> dict[str, Any]:
    if sampled_vanishing != witnessed_vanishing:
        return _vanishing_blocker(sampled_vanishing, witnessed_vanishing)
    return {
        "type": "SandwichNotTight",
        "reason": "exhibited_members_do_not_span_the_sampled_nullspace",
        "coefficient_nullity": coefficient_nullity,
        "exhibited_coefficient_rank": exhibited_rank,
        "detail": (
            f"the sampled coefficient nullspace has dimension {coefficient_nullity} but the "
            f"exhibited members span only {exhibited_rank} of it, so the reported dimension is an "
            "OPEN UPPER BOUND and not a measurement.  Refusing to publish it."
        ),
    }
def _solve_exact(
    columns: Sequence[Sequence[int]], target: Sequence[int], modulus: int
) -> list[Fraction] | None:
    """Solve ``sum_c x_c columns[c] == target`` exactly over the prime field.

    Returns ``None`` when the system has no solution or the solution is not unique, which is the
    honest answer for a degenerate operator basis rather than a coefficient picked out of an
    underdetermined family.  Residues are lifted to rationals by the same balanced-lift the rest
    of the module uses, so no float ever appears.
    """

    rows = [list(column) for column in columns] + [list(target)]
    kernel = _nullspace(rows, modulus)
    solutions: list[list[Fraction]] = []
    for vector in kernel:
        tail = vector[-1] % modulus
        if tail == 0:
            continue
        scale = (-pow(tail, modulus - 2, modulus)) % modulus
        solutions.append(
            [_rational((value * scale) % modulus, modulus) for value in vector[:-1]]
        )
    if len(solutions) != 1 or len(kernel) != 1:
        return None
    return solutions[0]


# ---------------------------------------------------------------------------
# A SECOND declared framework: linear elasticity on flat Euclidean space.
#
# Everything below shares the exact prime-field jet arithmetic with the curvature engine above and
# shares nothing else.  It never builds a connection, never builds a Riemann tensor, and consumes a
# displacement field rather than a metric, which is precisely why the framework measurement can
# tell the two apart by looking at arrays instead of at labels.
# ---------------------------------------------------------------------------

#: Rank-4 pattern labels, in enumeration order.  These index every elasticity coefficient vector.
ELASTICITY_PATTERN_TEXT: dict[str, str] = {
    "delta_ij_delta_kl": "d_ij d_kl",
    "delta_ik_delta_jl": "d_ik d_jl",
    "delta_il_delta_jk": "d_il d_jk",
}

#: Declared elasticity coefficient vectors that the search must *exhibit*, plus the traps.
DECLARED_ELASTIC_VECTORS: dict[str, dict[str, str]] = {
    # C = d_ij d_kl.  Acting on strain this is d_ij tr(eps): the Lame lambda direction.
    "lame_lambda": {"delta_ij_delta_kl": "1"},
    # C = d_ik d_jl + d_il d_jk.  Acting on strain this is 2 eps_ij: the Lame mu direction.
    "lame_mu": {"delta_ik_delta_jl": "1", "delta_il_delta_jk": "1"},
    # The minor-ANTIsymmetric combination.  It is a perfectly good rank-4 tensor and it annihilates
    # every symmetric strain, which is why the search must report it as identically zero rather
    # than as an extra elastic constant.
    "fabricated_minor_antisymmetric": {
        "delta_ik_delta_jl": "1",
        "delta_il_delta_jk": "-1",
    },
    # Cauchy / rari-constant relation lambda = mu: the historically WRONG 19th-century reduction.
    "cauchy_rari_constant": {
        "delta_ij_delta_kl": "1",
        "delta_ik_delta_jl": "1",
        "delta_il_delta_jk": "1",
    },
}

SUPPORTED_ELASTIC_CONSTRAINTS = (
    "flat_euclidean",
    "material_isotropy",
    "minor_symmetry",
    "stress_symmetry",
    "major_symmetry",
    "cauchy_relation",
)


def enumerate_elasticity_patterns(dim: int, modulus: int) -> list[dict[str, Any]]:
    """Every rank-4 tensor flat isotropic space offers: the perfect matchings of four slots.

    With no field content at all the only invariant available is the flat metric ``d_ab``, so a
    rank-4 concomitant with constant coefficients is a sum over ways of pairing the four free
    indices.  Nothing about elasticity is used here; the collapse under the declared symmetries is
    *measured* afterwards, exactly as the curvature side measures the Riemann collapse.
    """

    delta = np.eye(dim, dtype=np.int64) % modulus
    letters = "ijkl"
    out: list[dict[str, Any]] = []
    for matching in _perfect_matchings([0, 1, 2, 3]):
        specs = [letters[left] + letters[right] for left, right in matching]
        array = np.einsum(f"{specs[0]},{specs[1]}->ijkl", delta, delta) % modulus
        out.append(
            {
                "label": "delta_" + "_delta_".join(specs),
                "matching": [list(pair) for pair in matching],
                "array": array,
            }
        )
    if [item["label"] for item in out] != list(ELASTICITY_PATTERN_TEXT):
        raise TensorConstraintSearchError(
            "the enumerated pattern labels do not match the declared coefficient coordinate "
            "system; every elasticity vector in every receipt is indexed by that order"
        )
    return out


def displacement_jet(dim: int, degree: int, modulus: int, seed: str, sample: int) -> np.ndarray:
    """A random displacement jet: ``d`` free component functions, no symmetry, unconstrained.

    The jet of a displacement field is as unconstrained as the jet of a metric, so a random jet
    probes an elasticity identity fully and the same one-sided sampling argument applies.
    """

    ring = _ring(dim, degree, modulus)
    jet = np.zeros((dim, ring.size), dtype=np.int64)
    label = f"{seed}|elastic|d{dim}|j{degree}|s{sample}"
    counter = 0
    for axis in range(dim):
        for position in range(ring.size):
            counter += 1
            jet[axis, position] = _stream(label, counter) % modulus
    return jet


def rigid_motion_jet(dim: int, degree: int, modulus: int, seed: str) -> np.ndarray:
    """``u_i = c_i + w_ij x_j`` with ``w`` antisymmetric: an infinitesimal rigid motion.

    Its linear strain vanishes identically while its displacement gradient does not.  Evaluating
    the generator on it therefore *separates* a law built from the strain from a law built from
    the raw gradient, which is how the framework measurement identifies the strain generator
    without being told.
    """

    ring = _ring(dim, degree, modulus)
    jet = np.zeros((dim, ring.size), dtype=np.int64)
    counter = 0
    for axis in range(dim):
        counter += 1
        jet[axis, 0] = _stream(f"{seed}|rigid|translate|d{dim}", counter) % modulus
    for left in range(dim):
        for right in range(left + 1, dim):
            counter += 1
            value = _stream(f"{seed}|rigid|rotate|d{dim}", counter) % modulus
            jet[left, 1 + right] = value
            jet[right, 1 + left] = (-value) % modulus
    return jet


class _ElasticJet(_JetCalculus):
    """One displacement jet on flat Euclidean space, carried exactly over a prime field.

    Cartesian coordinates, so the metric is the constant Kronecker delta and partial derivatives
    *are* covariant derivatives.  The connection and curvature of that metric are computed rather
    than asserted to be zero, because the framework measurement reports the curvature it measured
    and an asserted zero would put the honesty hole straight back.
    """

    def __init__(self, dim: int, jet: np.ndarray, modulus: int, degree: int) -> None:
        if degree < 2:
            raise TensorConstraintSearchError(
                "the equilibrium operator needs at least a 2-jet of the displacement"
            )
        self.dim = dim
        self.modulus = modulus
        self.half = pow(2, modulus - 2, modulus)
        self.rings = [_ring(dim, level, modulus) for level in range(degree + 1)]
        self.displacement = jet % modulus
        self.displacement_level = degree
        self.primary_field_indices = int(jet.ndim - 1)
        self.primary_field_symmetric = bool(
            self.primary_field_indices == 2
            and np.array_equal(jet % modulus, np.einsum("abj->baj", jet) % modulus)
        )
        metric = np.zeros((dim, dim, self.rings[degree].size), dtype=np.int64)
        metric[..., 0] = np.eye(dim, dtype=np.int64)
        self.metric = metric
        self.inverse = metric.copy()
        self.gradient_level = degree - 1
        self.gradient = self.diff(self.displacement, degree)  # gradient[i, k] = d_k u_i
        self.strain = (
            (self.gradient + np.einsum("ikj->kij", self.gradient)) * self.half
        ) % modulus
        self.christoffel, self.riemann = self._curvature()

    def _curvature(self) -> tuple[np.ndarray, np.ndarray]:
        """Connection and Riemann tensor of the metric this generator actually uses."""

        level = self.displacement_level - 1
        partial = self.diff(self.metric, self.displacement_level)  # partial[i, j, k]
        combo = (
            np.einsum("dcbj->dbcj", partial)
            + np.einsum("bdcj->dbcj", partial)
            - np.einsum("bcdj->dbcj", partial)
        ) % self.modulus
        gamma = (self.mul("adj,dbcj->abcj", self.inverse, combo, level) * self.half) % self.modulus
        d_gamma = self.diff(gamma, level)
        quad = self.mul("acej,edbj->abcdj", gamma, gamma, level - 1)
        riemann = (
            np.einsum("adbcj->abcdj", self.cut(d_gamma, level - 1))
            - np.einsum("acbdj->abcdj", self.cut(d_gamma, level - 1))
            + quad
            - np.einsum("abdcj->abcdj", quad)
        ) % self.modulus
        return gamma, riemann

    def stress(self, tensor: np.ndarray, argument: np.ndarray) -> np.ndarray:
        """``sigma_ij = C_ijkl arg_kl`` for a constant-coefficient rank-4 ``C``."""

        return np.einsum("ijkl,klr->ijr", tensor, argument, optimize=True) % self.modulus

    def equilibrium(self, sigma: np.ndarray) -> np.ndarray:
        """``L_i = d_j sigma_ij``: the static equilibrium operator, one jet degree lower."""

        divided = self.diff(sigma, self.gradient_level)  # divided[i, j, k] = d_k sigma_ij
        return np.einsum("ijjr->ir", divided) % self.modulus

    def laplacian(self) -> np.ndarray:
        second = self.diff(self.gradient, self.gradient_level)  # second[i, k, m] = d_m d_k u_i
        return np.einsum("ikkr->ir", second) % self.modulus

    def gradient_of_divergence(self) -> np.ndarray:
        second = self.diff(self.gradient, self.gradient_level)
        return np.einsum("kkir->ir", second) % self.modulus


def _elastic_bank(
    dim: int, modulus: int, seed: str, samples: int, degree: int = 3
) -> list[_ElasticJet]:
    return [
        _ElasticJet(dim, displacement_jet(dim, degree, modulus, seed, sample), modulus, degree)
        for sample in range(samples)
    ]


def _elastic_combination(
    vector: Sequence[int], patterns: Sequence[Mapping[str, Any]], modulus: int
) -> np.ndarray:
    total = np.zeros_like(patterns[0]["array"])
    for coefficient, pattern in zip(vector, patterns, strict=True):
        total = (total + int(coefficient) * pattern["array"]) % modulus
    return total


def _declared_elastic_vector(name: str, labels: Sequence[str], modulus: int) -> list[int]:
    spec = DECLARED_ELASTIC_VECTORS[name]
    out: list[int] = []
    for label in labels:
        value = Fraction(spec.get(label, "0"))
        out.append(
            (int(value.numerator) * pow(int(value.denominator), modulus - 2, modulus)) % modulus
        )
    return out


def measure_realized_elastic_framework(
    bank: Sequence[_ElasticJet],
    patterns: Sequence[Mapping[str, Any]],
    *,
    dimension: int,
    modulus: int,
    seed: str,
) -> dict[str, Any]:
    """Measure the framework the elasticity generator actually ran, off its own arrays."""

    reference = bank[0]
    base = reference.metric[..., 0] % modulus
    diagonal = [int(base[axis, axis]) for axis in range(dimension)]
    off_diagonal = base.copy()
    np.fill_diagonal(off_diagonal, 0)
    signature = (
        _signature_name(diagonal, modulus) if not np.any(off_diagonal) else "non-normalised"
    )

    curved = any(bool(np.any(jet.riemann % modulus)) for jet in bank)

    ranks = {int(pattern["array"].ndim) for pattern in patterns}
    if len(ranks) != 1:
        raise FrameworkMismatch("the elasticity generator produced patterns of mixed rank")
    tensor_rank = ranks.pop()

    symmetric = True
    any_nonzero = False
    for jet in bank:
        for pattern in patterns:
            sigma = jet.stress(pattern["array"], jet.strain) % modulus
            any_nonzero = any_nonzero or bool(np.any(sigma))
            if not np.array_equal(sigma, np.einsum("ijr->jir", sigma) % modulus):
                symmetric = False
    tensor_symmetry = "symmetric" if (symmetric and any_nonzero) else "none"

    rigid = _ElasticJet(
        dimension,
        rigid_motion_jet(dimension, reference.displacement_level, modulus, seed),
        modulus,
        reference.displacement_level,
    )
    rigid_gradient_nonzero = bool(np.any(rigid.gradient % modulus))
    rigid_stress_zero = all(
        not bool(np.any(rigid.stress(pattern["array"], rigid.strain) % modulus))
        for pattern in patterns
    )
    rigid_gradient_stress_nonzero = any(
        bool(np.any(rigid.stress(pattern["array"], rigid.gradient) % modulus))
        for pattern in patterns
    )
    generator = (
        "linear_strain_tensor"
        if (rigid_stress_zero and rigid_gradient_nonzero and any_nonzero)
        else "strain_free_unidentified"
    )

    return {
        "metric_signature": signature,
        "curvature": "generic" if curved else "flat",
        "primary_field_indices": int(reference.primary_field_indices),
        "primary_field_symmetric": bool(reference.primary_field_symmetric),
        "tensor_rank": int(tensor_rank),
        "tensor_symmetry": tensor_symmetry,
        "concomitant_generator": generator,
        "measurement_evidence": {
            "base_point_metric_diagonal_is_pm_one": bool(signature != "non-normalised"),
            "riemann_of_the_metric_actually_used_is_zero": bool(not curved),
            "rigid_motion_has_nonzero_displacement_gradient": rigid_gradient_nonzero,
            "rigid_motion_produces_zero_stress": bool(rigid_stress_zero),
            "same_patterns_on_the_raw_gradient_would_stress_a_rigid_motion": bool(
                rigid_gradient_stress_nonzero
            ),
            "how_generator_was_identified": (
                "the patterns were evaluated on an infinitesimal rigid motion, whose linear strain "
                "vanishes identically while its displacement gradient does not.  Zero stress there "
                "and non-zero stress on the generic bank identifies the argument as the LINEAR "
                "STRAIN tensor; a law built from the raw gradient would stress a rigid rotation, "
                "and the same patterns on that argument are checked to confirm it would"
            ),
        },
    }


def run_elasticity_search(
    *,
    dimension: int,
    constraints: Sequence[str],
    modulus: int,
    seed: str,
    bank_samples: int,
    holdout_samples: int,
) -> dict[str, Any]:
    """Enumerate rank-4 isotropic patterns, measure the collapse, report the elastic constants."""

    declared = list(constraints)
    unknown = [name for name in declared if name not in SUPPORTED_ELASTIC_CONSTRAINTS]
    if unknown:
        raise TensorConstraintSearchError(
            f"unsupported elasticity constraints declared: {sorted(unknown)}"
        )
    for required in ("flat_euclidean", "material_isotropy"):
        if required not in declared:
            raise ConstraintOutOfScope(
                f"{required} is not optional in this search: the rank-4 basis is enumerated as "
                "pairings of the flat metric, and without a flat isotropic background that "
                "enumeration is not the admissible basis.  Refusing rather than returning an "
                "answer over an unjustified basis."
            )

    degree = 3
    bank = _elastic_bank(dimension, modulus, seed, bank_samples, degree)
    holdout = _elastic_bank(dimension, modulus, f"{seed}|holdout", holdout_samples, degree)
    patterns = enumerate_elasticity_patterns(dimension, modulus)
    labels = [str(pattern["label"]) for pattern in patterns]

    realized = measure_realized_elastic_framework(
        bank, patterns, dimension=dimension, modulus=modulus, seed=seed
    )

    minor = "minor_symmetry" in declared
    argument_name = "linear strain eps_kl" if minor else "raw displacement gradient d_l u_k"

    # -- what the patterns are, before any declared symmetry ---------------
    array_rows = [
        [int(value) for value in pattern["array"].ravel() % modulus] for pattern in patterns
    ]
    rank_as_arrays = _rank(array_rows, modulus)

    # -- each declared symmetry as its OWN block of linear conditions on the coefficients --
    per_constraint: dict[str, list[list[int]]] = {}
    if minor:
        rows: list[list[int]] = []
        for pattern in patterns:
            array = pattern["array"]
            first = (array - np.einsum("ijkl->jikl", array)) % modulus
            second = (array - np.einsum("ijkl->ijlk", array)) % modulus
            rows.append(
                [int(value) for value in first.ravel()]
                + [int(value) for value in second.ravel()]
            )
        per_constraint["minor_symmetry"] = rows
    if "stress_symmetry" in declared:
        rows = []
        for pattern in patterns:
            entry: list[int] = []
            for jet in bank:
                argument = jet.strain if minor else jet.gradient
                sigma = jet.stress(pattern["array"], argument) % modulus
                residual = (sigma - np.einsum("ijr->jir", sigma)) % modulus
                entry.extend(int(value) for value in residual.ravel())
            rows.append(entry)
        per_constraint["stress_symmetry"] = rows
    if "major_symmetry" in declared:
        rows = []
        for pattern in patterns:
            array = pattern["array"]
            major = (array - np.einsum("ijkl->klij", array)) % modulus
            rows.append([int(value) for value in major.ravel()])
        per_constraint["major_symmetry"] = rows
    if "cauchy_relation" in declared:
        # The rari-constant hypothesis lambda = mu, written in pattern coordinates.  lambda is the
        # coefficient of d_ij d_kl and mu the shared coefficient of the two remaining pairings.
        functional = {
            "delta_ij_delta_kl": 1,
            "delta_ik_delta_jl": modulus - 1,
            "delta_il_delta_jk": 0,
        }
        per_constraint["cauchy_relation"] = [
            [int(functional[label]) % modulus] for label in labels
        ]

    # -- collapse measured by ACTION on the declared argument ---------------
    def action_row(vector: Sequence[int]) -> list[int]:
        array = _elastic_combination(vector, patterns, modulus)
        out: list[int] = []
        for jet in bank:
            argument = jet.strain if minor else jet.gradient
            out.extend(int(value) for value in jet.stress(array, argument).ravel())
        return out

    identity = [
        [1 if a == b else 0 for b in range(len(patterns))] for a in range(len(patterns))
    ]
    pattern_actions = [action_row(vector) for vector in identity]
    rank_after_argument = _rank(pattern_actions, modulus)
    identically_vanishing = _nullspace(pattern_actions, modulus)

    # Each declared constraint is applied on top of the ones before it and the surviving dimension
    # is MEASURED after each, so the reduction table reports what each constraint actually cost
    # rather than repeating the final number once per row.
    applied: list[str] = []
    accumulated: list[list[int]] = [[] for _ in patterns]
    admissible = identity
    cascade: list[dict[str, Any]] = []
    for name in ("minor_symmetry", "stress_symmetry", "major_symmetry", "cauchy_relation"):
        if name not in per_constraint:
            continue
        applied.append(name)
        for index, segment in enumerate(per_constraint[name]):
            accumulated[index].extend(segment)
        admissible = _nullspace(accumulated, modulus)
        cascade.append(
            {
                "constraint": name,
                "coefficient_dimension": len(admissible),
                "distinct_law_dimension": _rank(
                    [action_row(vector) for vector in admissible], modulus
                ),
            }
        )

    admissible_actions = [action_row(vector) for vector in admissible]
    distinct_dimension = _rank(admissible_actions, modulus)

    # -- held-out verification ---------------------------------------------
    for vector in admissible:
        array = _elastic_combination(vector, patterns, modulus)
        for jet in holdout:
            argument = jet.strain if minor else jet.gradient
            sigma = jet.stress(array, argument) % modulus
            if "stress_symmetry" in declared and np.any(
                (sigma - np.einsum("ijr->jir", sigma)) % modulus
            ):
                raise TensorConstraintSearchError(
                    "a member of the reported elasticity space produced an asymmetric stress on a "
                    "held-out displacement jet"
                )
        if minor:
            array_minor = (array - np.einsum("ijkl->jikl", array)) % modulus
            if np.any(array_minor):
                raise TensorConstraintSearchError(
                    "a member of the reported elasticity space is not minor-symmetric"
                )

    reduced, _ = _rref(admissible, modulus)
    exhibited: list[dict[str, Any]] = []
    for label in ("lame_lambda", "lame_mu", "fabricated_minor_antisymmetric",
                  "cauchy_rari_constant"):
        vector = _declared_elastic_vector(label, labels, modulus)
        array = _elastic_combination(vector, patterns, modulus)
        acts_as_zero = all(
            not bool(np.any(jet.stress(array, jet.strain) % modulus)) for jet in bank
        )
        minor_symmetric = not bool(np.any((array - np.einsum("ijkl->jikl", array)) % modulus))
        major_symmetric = not bool(np.any((array - np.einsum("ijkl->klij", array)) % modulus))
        exhibited.append(
            {
                "name": label,
                "expression": _vector_text(
                    [Fraction(DECLARED_ELASTIC_VECTORS[label].get(item, "0")) for item in labels],
                    labels,
                ),
                "acts_as_zero_on_every_symmetric_strain": bool(acts_as_zero),
                "minor_symmetric": bool(minor_symmetric),
                "major_symmetric": bool(major_symmetric),
                "in_reported_space": bool(_in_span(vector, admissible, modulus)),
            }
        )

    return {
        "dimension": dimension,
        "declared_constraints": declared,
        "constraints_applied": applied,
        "declared_argument": argument_name,
        "realized_framework": realized,
        "basis_patterns": [
            {"name": label, "expression": ELASTICITY_PATTERN_TEXT[label]} for label in labels
        ],
        "enumeration": {
            "formal_pattern_count": len(patterns),
            "rank_as_rank4_arrays": rank_as_arrays,
            "rank_of_actions_on_declared_argument": rank_after_argument,
            "identically_vanishing": {
                "dimension": len(identically_vanishing),
                "vectors": [
                    _vector_text(_normalise(vector, modulus), labels)
                    for vector in identically_vanishing
                ],
            },
            "constraint_cascade": cascade,
        },
        "admissible_space": {
            "coefficient_dimension": len(admissible),
            "distinct_law_dimension": distinct_dimension,
            "basis": [_vector_text(_normalise(vector, modulus), labels) for vector in reduced],
            "holdout_samples_verified": len(holdout),
        },
        "exhibited_members": exhibited,
        "surviving_dimension": distinct_dimension,
    }


def navier_cauchy_derivation(
    *, dimension: int, modulus: int, seed: str, bank_samples: int, holdout_samples: int
) -> dict[str, Any]:
    """Derive the static equilibrium operator of the surviving elastic family, exactly.

    For each surviving direction of the elasticity tensor the divergence of the stress it produces
    is *fitted* on a bank of random displacement jets against the only two second-order isotropic
    vector operators available, ``lap u_i`` and ``d_i (div u)``, by exact linear algebra over the
    prime field.  The coefficients are an output, not an input, and the fit is re-verified on
    held-out jets.  The two directions together assemble the Navier-Cauchy operator.
    """

    degree = 3
    bank = _elastic_bank(dimension, modulus, seed, bank_samples, degree)
    holdout = _elastic_bank(dimension, modulus, f"{seed}|holdout", holdout_samples, degree)
    patterns = enumerate_elasticity_patterns(dimension, modulus)
    labels = [str(pattern["label"]) for pattern in patterns]

    def stack(jets: Sequence[_ElasticJet], pick: str, array: np.ndarray | None = None
              ) -> list[int]:
        out: list[int] = []
        for jet in jets:
            if pick == "laplacian":
                values = jet.laplacian()
            elif pick == "graddiv":
                values = jet.gradient_of_divergence()
            else:
                assert array is not None
                values = jet.equilibrium(jet.stress(array, jet.strain))
            out.extend(int(value) for value in values.ravel())
        return out

    laplacian = stack(bank, "laplacian")
    graddiv = stack(bank, "graddiv")
    operators_independent = _rank([laplacian, graddiv], modulus) == 2

    directions: list[dict[str, Any]] = []
    for name in ("lame_lambda", "lame_mu"):
        vector = _declared_elastic_vector(name, labels, modulus)
        array = _elastic_combination(vector, patterns, modulus)
        target = stack(bank, "equilibrium", array)
        solution = _solve_exact([laplacian, graddiv], target, modulus) if operators_independent \
            else None
        verified = False
        if solution is not None:
            held_lap = stack(holdout, "laplacian")
            held_grad = stack(holdout, "graddiv")
            held_target = stack(holdout, "equilibrium", array)
            verified = all(
                (
                    solution[0].numerator
                    * pow(int(solution[0].denominator), modulus - 2, modulus)
                    * a
                    + solution[1].numerator
                    * pow(int(solution[1].denominator), modulus - 2, modulus)
                    * b
                    - t
                )
                % modulus
                == 0
                for a, b, t in zip(held_lap, held_grad, held_target, strict=True)
            )
        directions.append(
            {
                "direction": name,
                "elasticity_tensor": _vector_text(
                    [Fraction(DECLARED_ELASTIC_VECTORS[name].get(item, "0")) for item in labels],
                    labels,
                ),
                "laplacian_coefficient": str(solution[0]) if solution else None,
                "grad_div_coefficient": str(solution[1]) if solution else None,
                "unique_solution": bool(solution is not None),
                "verified_on_holdout": bool(verified),
            }
        )

    lam = next(item for item in directions if item["direction"] == "lame_lambda")
    mu = next(item for item in directions if item["direction"] == "lame_mu")
    assembled = None
    if lam["unique_solution"] and mu["unique_solution"]:
        lap_coefficient = (
            f"{mu['laplacian_coefficient']} mu"
            if lam["laplacian_coefficient"] == "0"
            else f"{lam['laplacian_coefficient']} lambda + {mu['laplacian_coefficient']} mu"
        )
        assembled = (
            f"d_j sigma_ij = ({lap_coefficient}) lap u_i + "
            f"({lam['grad_div_coefficient']} lambda + {mu['grad_div_coefficient']} mu) "
            "d_i (div u)"
        )
    return {
        "question": (
            "what static equilibrium operator does the surviving elasticity family force on the "
            "displacement field?"
        ),
        "operator_basis": ["lap u_i", "d_i (div u)"],
        "operator_basis_independent_on_the_bank": bool(operators_independent),
        "directions": directions,
        "assembled": assembled,
        "identification": (
            "Navier-Cauchy: mu lap u + (lambda + mu) grad div u.  The coefficients above were "
            "SOLVED for on random displacement jets over a prime field and re-verified on held-out "
            "jets; they were not transcribed."
        )
        if assembled
        else None,
        "bank_samples": bank_samples,
        "holdout_samples": holdout_samples,
    }


# ---------------------------------------------------------------------------
# The search itself.
# ---------------------------------------------------------------------------

SUPPORTED_CONSTRAINTS = (
    "generally_covariant",
    "derivative_order",
    "symmetric",
    "divergence_free",
    "newtonian_limit",
)


def run_search(
    *,
    dimension: int,
    order: int,
    constraints: Sequence[str],
    modulus: int,
    seed: str,
    bank_samples: int,
    holdout_samples: int,
    enumerate_basis: bool,
) -> dict[str, Any]:
    """Enumerate, constrain, and report the surviving coefficient space."""

    declared = list(constraints)
    unknown = [name for name in declared if name not in SUPPORTED_CONSTRAINTS]
    if unknown:
        raise TensorConstraintSearchError(f"unsupported constraints declared: {sorted(unknown)}")
    if "generally_covariant" not in declared:
        raise ConstraintOutOfScope(
            "general covariance is not optional in this search: the admissible-term basis is "
            "enumerated from g, g^-1, the Riemann tensor and covariant derivatives, and without it "
            "the space is not finite-dimensional and no enumeration is defined.  Refusing rather "
            "than returning an answer over an unjustified basis."
        )
    if "derivative_order" not in declared:
        raise ConstraintOutOfScope(
            "a finite derivative order must be declared; without it the curvature-polynomial space "
            "is not finite-dimensional."
        )

    bank = build_bank(dimension, order, modulus, seed, bank_samples)
    holdout = build_bank(dimension, order, modulus, f"{seed}|holdout", holdout_samples)
    names = named_term_names(order)

    # -- enumeration -------------------------------------------------------
    enumeration: dict[str, Any] = {"performed": bool(enumerate_basis)}
    if enumerate_basis:
        patterns = enumerate_patterns(order)
        caches: list[dict[str, np.ndarray]] = [{} for _ in bank]
        raw_rows: list[list[int]] = []
        symmetric_rows: list[list[int]] = []
        half = pow(2, modulus - 2, modulus)
        for pattern in patterns:
            raw: list[int] = []
            symmetrised: list[int] = []
            for geometry, cache in zip(bank, caches, strict=True):
                value = evaluate_pattern(geometry, pattern, cache, symmetrise=False)
                raw.extend(int(item) for item in value.ravel())
                symmetrised.extend(
                    int(item) for item in (((value + value.T) * half) % modulus).ravel()
                )
            raw_rows.append(raw)
            symmetric_rows.append(symmetrised)
        enumeration["formal_pattern_count"] = len(patterns)
        enumeration["factor_multisets"] = [
            {
                "factors": list(multiset),
                "weight": sum(PATTERN_GENERATORS[name][0] for name in multiset),
                "slot_count": sum(PATTERN_GENERATORS[name][1] for name in multiset),
                "patterns": sum(1 for p in patterns if p["factors"] == multiset),
            }
            for multiset in _factor_multisets(order)
        ]
        enumeration["rank_before_symmetry"] = _rank(raw_rows, modulus)
        enumeration["rank_after_symmetry"] = _rank(symmetric_rows, modulus)
    else:
        enumeration["reason"] = (
            "the mechanical contraction enumeration is run where it is affordable; elsewhere the "
            "named basis it reduced to is reused, with the reuse recorded rather than hidden"
        )

    # -- named basis over the bank, then over bank AND holdout -------------
    #
    # Both sampled spaces below -- the identically-vanishing space and the divergence-free space --
    # are computed on the SAME jet set, and both are re-verified on the holdout.  That symmetry is
    # the whole point.  A finite jet set imposes a subset of the identical constraints, so each
    # sampled space comes out too LARGE.  The old code differenced a holdout-verified nullity
    # against a bank-only vanishing dimension, so the two errors ran in opposite directions and the
    # difference could land BELOW the truth -- the exact opposite of the advertised guarantee.
    bank_terms: list[dict[str, np.ndarray]] = [named_tensors(geometry, order) for geometry in bank]
    holdout_terms: list[dict[str, np.ndarray]] = [
        named_tensors(geometry, order) for geometry in holdout
    ]
    bank_pairs = list(zip(bank, bank_terms, strict=True))
    holdout_pairs = list(zip(holdout, holdout_terms, strict=True))
    sample_pairs = [*bank_pairs, *holdout_pairs]

    term_rows = _sample_rows(bank_pairs, names, divergence=False)
    divergence_rows = _sample_rows(bank_pairs, names, divergence=True)
    vanishing_rows = _sample_rows(bank_pairs, names, divergence=False, full_jet=True)
    vanishing_rows_all = _sample_rows(sample_pairs, names, divergence=False, full_jet=True)
    divergence_rows_all = _sample_rows(sample_pairs, names, divergence=True)

    if enumerate_basis:
        named_rank = _rank(term_rows, modulus)
        enumeration["named_basis_rank"] = named_rank
        enumeration["named_basis_spans_enumeration"] = bool(
            named_rank == enumeration["rank_after_symmetry"]
            and _rank([*symmetric_rows, *term_rows], modulus) == named_rank
        )
        if not enumeration["named_basis_spans_enumeration"]:
            raise TensorConstraintSearchError(
                "the named basis does not span the mechanically enumerated patterns; the basis is "
                "incomplete at this order and the search refuses to report a dimension"
            )

    vanishing_bank = _nullspace(vanishing_rows, modulus)
    identically_vanishing = _nullspace(vanishing_rows_all, modulus)
    for vector in identically_vanishing:
        for _, terms in holdout_pairs:
            if np.any(_combine(vector, terms, names, modulus) % modulus):  # pragma: no cover
                raise TensorConstraintSearchError(
                    "a member of the reported identically-vanishing space failed on a held-out jet"
                )

    # The subtracted quantity has to be bounded from BELOW, and no amount of sampling bounds an
    # identity from below.  So the only directions ever subtracted are *witnessed* ones: declared
    # vectors, written from the literature, each confronted with the bank and the holdout.  If the
    # sampled vanishing space is bigger than the witnessed one there is an unexplained direction in
    # it and the run refuses to publish a dimension rather than subtracting a sampled number.
    witnesses = _vanishing_witnesses(sample_pairs, names, modulus)
    witnessed_vectors = [vector for _, vector in witnesses]
    witnessed_dimension = _rank(witnessed_vectors, modulus) if witnessed_vectors else 0
    fully_witnessed = bool(witnessed_dimension == len(identically_vanishing))
    independent_dimension = len(names) - witnessed_dimension

    realized_framework = measure_realized_framework(
        bank, bank_terms, dimension=dimension, order=order, modulus=modulus
    )

    result: dict[str, Any] = {
        "dimension": dimension,
        "derivative_order": order,
        "declared_constraints": declared,
        "realized_framework": realized_framework,
        "basis_terms": [
            {"name": name, "expression": _TERM_TEXT[name], "metric_derivatives": weight}
            for name, weight, _ in NAMED_TERMS
            if name in set(names)
        ],
        "enumeration": enumeration,
        "identically_vanishing": {
            "dimension": len(identically_vanishing),
            "bank_only_dimension": len(vanishing_bank),
            "holdout_refuted_dimensions": len(vanishing_bank) - len(identically_vanishing),
            "holdout_samples_verified": len(holdout),
            "witnessed_dimension": witnessed_dimension,
            "witnesses": [label for label, _ in witnesses],
            "fully_witnessed": fully_witnessed,
            "vectors": [
                _vector_text(_normalise(vector, modulus), names)
                for vector in identically_vanishing
            ],
        },
        "independent_dimension": independent_dimension,
    }

    if "divergence_free" not in declared:
        result["divergence_free_applied"] = False
        result["dimension_published"] = fully_witnessed
        result["surviving_dimension"] = independent_dimension if fully_witnessed else None
        if not fully_witnessed:
            result["blocker"] = _vanishing_blocker(
                len(identically_vanishing), witnessed_dimension
            )
        result["surviving_family"] = (
            "the full symmetric basis, no conservation constraint imposed: "
            + ", ".join(_TERM_TEXT[name] for name in names)
        )
        return result

    nullspace_bank = _nullspace(divergence_rows, modulus)
    raw_nullspace = _nullspace(divergence_rows_all, modulus)

    # Held-out verification: every reported member must stay divergence-free on jets that took no
    # part in the bank rank computation.  Members the holdout refutes are *dropped*, which is what
    # recomputing the nullspace over bank+holdout already does -- the loop below is the assertion
    # that nothing survived it, and the count of refuted directions is published.
    for vector in raw_nullspace:
        for geometry, terms in holdout_pairs:
            tensor = _combine(vector, terms, names, modulus)
            if np.any(geometry.divergence(tensor, 1) % modulus):  # pragma: no cover
                raise TensorConstraintSearchError(
                    "a member of the reported divergence-free space failed on a held-out metric jet"
                )

    exhibited: list[dict[str, Any]] = []
    for label in EXHIBITED_LABELS:
        spec = DECLARED_VECTORS.get(label)
        if spec is None or not set(spec) <= set(names):
            continue
        vector = _declared_vector(label, names, modulus)
        # Both flags are decided on the bank AND the holdout.  An exhibited member is a LOWER
        # bound, so over-crediting it is the dangerous direction and the holdout is what stops it.
        vanishes = all(
            not np.any(_combine(vector, terms, names, modulus) % modulus)
            for _, terms in sample_pairs
        )
        divergence_free = all(
            not np.any(
                geometry.divergence(_combine(vector, terms, names, modulus), 1) % modulus
            )
            for geometry, terms in sample_pairs
        )
        exhibited.append(
            {
                "name": label,
                "expression": _vector_text(
                    [Fraction(spec.get(name, "0")) for name in names], names
                ),
                "identically_zero": bool(vanishes),
                "divergence_free": bool(divergence_free),
                "in_reported_space": bool(_in_span(vector, raw_nullspace, modulus)),
            }
        )

    reduced, _ = _rref(raw_nullspace, modulus)
    # Coefficient-space sandwich.  The sampled nullity is an upper bound on dim D_true (a finite
    # jet set imposes a subset of the constraints).  The exhibited members are genuinely
    # divergence-free, so their span sits inside D_true and their rank is a lower bound.  When the
    # two meet, the coefficient space is pinned EXACTLY -- no sampling error is left in it at all.
    exhibited_vectors = [
        _declared_vector(item["name"], names, modulus)
        for item in exhibited
        if item["in_reported_space"] and item["divergence_free"]
    ]
    exhibited_rank = _rank(exhibited_vectors, modulus) if exhibited_vectors else 0
    coefficient_tight = bool(exhibited_rank == len(raw_nullspace))

    # Distinct-tensor sandwich.  Only witnessed vanishing directions are ever subtracted, so the
    # upper bound is (upper bound on the nullity) - (lower bound on the vanishing dimension) and
    # therefore still errs upward.  The lower bound is the rank of the exhibited members modulo
    # those same witnessed directions.
    joint = [*exhibited_vectors, *witnessed_vectors]
    distinct_upper = len(raw_nullspace) - witnessed_dimension
    distinct_lower = (_rank(joint, modulus) if joint else 0) - witnessed_dimension
    tight = bool(coefficient_tight and fully_witnessed and distinct_upper == distinct_lower)

    result["divergence_free_applied"] = True
    result["divergence_free_space"] = {
        "coefficient_nullity": len(raw_nullspace),
        "coefficient_nullity_bank_only": len(nullspace_bank),
        "holdout_refuted_dimensions": len(nullspace_bank) - len(raw_nullspace),
        "identically_vanishing_inside": len(identically_vanishing),
        "witnessed_vanishing_inside": witnessed_dimension,
        "distinct_tensor_dimension": distinct_upper,
        "basis": [_vector_text(_normalise(vector, modulus), names) for vector in reduced],
        "holdout_samples_verified": len(holdout),
    }
    result["exhibited_members"] = exhibited
    result["uniqueness_certificate"] = {
        "upper_bound": (
            "ONE-SIDED UPWARD.  A finite jet set imposes a subset of the identical constraints, so "
            "the sampled coefficient nullspace CONTAINS the true one and its dimension can only be "
            "reported too large.  The distinct-tensor upper bound subtracts only WITNESSED "
            "identically-vanishing directions -- declared vectors confronted with the bank and the "
            "holdout -- never the sampled vanishing dimension, which is itself an upper bound and "
            "would tip the error the other way.  No uniqueness claim can be manufactured by "
            "sampling."
        ),
        "lower_bound": (
            "Every reported member was re-verified on held-out metric jets, and every exhibited "
            "member was confronted with the bank and the holdout before it was credited.  For the "
            "order-2 case the lower bound is additionally ALGEBRAIC: g_mn is divergence-free by "
            "metric compatibility and G_mn by the contracted Bianchi identity, which this run "
            "re-derives on the bank and which is hash-bound to "
            "formal/cadabra/contracted_bianchi.cdb."
        ),
        "coefficient_nullity": len(raw_nullspace),
        "exhibited_coefficient_rank": exhibited_rank,
        "coefficient_sandwich_tight": coefficient_tight,
        "sampled_identically_vanishing_dimension": len(identically_vanishing),
        "witnessed_identically_vanishing_dimension": witnessed_dimension,
        "vanishing_fully_witnessed": fully_witnessed,
        "independently_exhibited_dimension": distinct_lower,
        "sampled_dimension": distinct_upper,
        "sandwich_tight": tight,
        "algebraic_lower_bound_available": bool(order == 2),
    }
    result["dimension_published"] = tight
    result["surviving_dimension"] = distinct_upper if tight else None
    if not tight:
        result["blocker"] = _sandwich_blocker(
            coefficient_nullity=len(raw_nullspace),
            exhibited_rank=exhibited_rank,
            sampled_vanishing=len(identically_vanishing),
            witnessed_vanishing=witnessed_dimension,
        )
    return result


def published_dimension(search: Mapping[str, Any]) -> int:
    """The certified distinct-tensor dimension of a search, or a refusal.

    Every consumer of a search dimension -- the reduction tables, the Gauss-Bonnet verdict, the
    relaxation controls, the headline counts -- goes through here.  A search whose sandwich did not
    close reported an OPEN UPPER BOUND, and an open upper bound is not a dimension: it cannot be
    compared, differenced or published.  Refusing is the whole fix.
    """

    if not search.get("dimension_published", False):
        blocker = search.get("blocker") or {}
        raise SandwichNotTight(
            f"search {search.get('search_id', '(unnamed)')} did not close its uniqueness "
            f"sandwich, so it has no publishable dimension: "
            f"{blocker.get('detail', 'sandwich not tight')}"
        )
    value = search.get("surviving_dimension")
    if not isinstance(value, int):  # pragma: no cover - defensive
        raise SandwichNotTight("a published search carried no integer dimension")
    return value


# ---------------------------------------------------------------------------
# Newtonian limit: the constraint that fixes the constant.
# ---------------------------------------------------------------------------


def newtonian_limit_chain() -> dict[str, Any]:
    """Derive the coupling constant from the weak-field slow-motion limit, symbolically.

    Coordinates are ``x^0 = c t`` and the declared weak field is
    ``g_00 = -(1 + 2 Phi/c^2)``, ``g_ij = (1 - 2 Phi/c^2) delta_ij`` with static ``Phi``.  Nothing
    is asserted about ``G_00``: the Christoffel symbols, the Ricci tensor and the Einstein tensor
    are all built from that metric and linearised in the bookkeeping parameter.
    """

    epsilon = sp.Symbol("epsilon")
    light, newton, density = sp.symbols("c G rho", positive=True)
    coordinates = sp.symbols("x0 x1 x2 x3", real=True)
    potential = sp.Function("Phi")(*coordinates[1:])

    def linearise(expression: sp.Expr) -> sp.Expr:
        return sp.expand(sp.series(sp.expand(expression), epsilon, 0, 2).removeO())

    factor = 2 * epsilon * potential / light**2
    metric = sp.diag(-(1 + factor), 1 - factor, 1 - factor, 1 - factor)
    inverse = sp.diag(*[linearise(1 / metric[i, i]) for i in range(4)])

    christoffel = [[[sp.Integer(0)] * 4 for _ in range(4)] for _ in range(4)]
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                value = sp.Integer(0)
                for inner in range(4):
                    value += inverse[upper, inner] * (
                        sp.diff(metric[inner, right], coordinates[left])
                        + sp.diff(metric[inner, left], coordinates[right])
                        - sp.diff(metric[left, right], coordinates[inner])
                    )
                christoffel[upper][left][right] = linearise(value / 2)

    ricci = sp.zeros(4, 4)
    for left in range(4):
        for right in range(4):
            value = sp.Integer(0)
            for inner in range(4):
                value += sp.diff(christoffel[inner][left][right], coordinates[inner])
                value -= sp.diff(christoffel[inner][left][inner], coordinates[right])
                for other in range(4):
                    value += christoffel[inner][left][right] * christoffel[other][inner][other]
                    value -= christoffel[other][left][inner] * christoffel[inner][right][other]
            ricci[left, right] = linearise(value)

    curvature = linearise(sum(inverse[i, i] * ricci[i, i] for i in range(4)))
    einstein_00 = linearise(ricci[0, 0] - curvature * metric[0, 0] / 2)

    laplacian = sum(sp.diff(potential, coordinates[i], 2) for i in range(1, 4))
    ratio = sp.simplify(sp.expand(einstein_00.coeff(epsilon, 1)) / laplacian)
    if sp.simplify(ratio - 2 / light**2) != 0:
        raise TensorConstraintSearchError("the linearised G_00 did not reduce to 2 nabla^2 Phi/c^2")

    geodesic = sp.simplify(christoffel[1][0][0].coeff(epsilon, 1) * light**2)
    if sp.simplify(geodesic - sp.diff(potential, coordinates[1])) != 0:
        raise TensorConstraintSearchError("the weak-field geodesic did not reproduce -grad Phi")

    alpha = sp.Symbol("alpha", positive=True)
    stress_00 = density * light**2
    field_equation = sp.Eq(alpha * ratio * laplacian, stress_00)
    poisson = sp.Eq(laplacian, 4 * sp.pi * newton * density)
    solved = sp.solve(field_equation.subs(laplacian, poisson.rhs), alpha)
    if len(solved) != 1:
        raise TensorConstraintSearchError("the Newtonian limit did not fix a unique constant")
    alpha_value = sp.simplify(solved[0])
    coupling = sp.simplify(1 / alpha_value)
    if sp.simplify(coupling - 8 * sp.pi * newton / light**4) != 0:
        raise TensorConstraintSearchError("the fixed constant is not 8 pi G / c^4")

    return {
        "declared_convention": (
            "x^0 = c t, signature (-,+,+,+); field equation written alpha G_mn + Lambda g_mn = T_mn"
        ),
        "steps": [
            {
                "step": 1,
                "statement": "declared weak static field",
                "expression": (
                    "g_00 = -(1 + 2 Phi/c^2), g_ij = (1 - 2 Phi/c^2) delta_ij, "
                    "Phi = Phi(x1,x2,x3)"
                ),
                "provenance": "declared",
            },
            {
                "step": 2,
                "statement": "Christoffel symbols and Ricci tensor rebuilt from that metric and "
                "linearised in the bookkeeping parameter",
                "expression": f"R_00 = {sp.simplify(ricci[0, 0].coeff(epsilon, 1))} (order epsilon)",
                "provenance": "derived",
            },
            {
                "step": 3,
                "statement": "the geodesic equation identifies Phi as the Newtonian potential",
                "expression": "Gamma^1_00 = d_1 Phi / c^2 so d^2 x^i/dt^2 = -d_i Phi",
                "provenance": "derived",
            },
            {
                "step": 4,
                "statement": "linearised Einstein tensor",
                "expression": "G_00 = 2 nabla^2 Phi / c^2",
                "provenance": "derived",
            },
            {
                "step": 5,
                "statement": "static dust source in these coordinates",
                "expression": "T_mn = rho c^2 u_m u_n with u^m = (1,0,0,0), so T_00 = rho c^2",
                "provenance": "declared",
            },
            {
                "step": 6,
                "statement": "impose Poisson's equation",
                "expression": "nabla^2 Phi = 4 pi G rho",
                "provenance": "declared",
            },
            {
                "step": 7,
                "statement": "solve alpha * G_00 = T_00 for alpha",
                "expression": f"alpha = {alpha_value}",
                "provenance": "derived",
            },
            {
                "step": 8,
                "statement": "equivalently, dividing through by alpha",
                "expression": f"G_mn + (Lambda/alpha) g_mn = kappa T_mn with kappa = {coupling}",
                "provenance": "derived",
            },
        ],
        "alpha_coefficient_of_einstein_tensor": str(alpha_value),
        "coupling_constant_kappa": str(coupling),
        "coupling_constant_kappa_latex": "8\\pi G/c^{4}",
        "lambda_status": (
            "UNFORCED.  With Lambda kept, the 00 equation reads alpha G_00 + Lambda g_00 = T_00, "
            "and at leading order Lambda enters only as an additive constant in Poisson's "
            "equation.  The declared limit -- Minkowski background with Phi -> 0 at infinity -- "
            "bounds that constant but does not determine it.  The cosmological constant therefore "
            "survives the whole chain as a free parameter, which is historically exactly what "
            "happened."
        ),
        "final_family": "alpha G_mn + Lambda g_mn = T_mn, alpha = c^4/(8 pi G), Lambda free",
    }


# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------


def gauss_bonnet_decomposition() -> dict[str, Any]:
    """Confront the three quadratic Euler-Lagrange vectors with the declared Lanczos tensor.

    The Euler-Lagrange operator is linear and the Gauss-Bonnet density is ``R^2 - 4 R_ab R^ab +
    R_abcd R^abcd``, so ``E1 - 4 E2 + E3`` must reproduce the Lanczos vector exactly -- including
    the cancellation of every derivative-carrying coefficient, since the Lanczos tensor is second
    order.  All four vectors were written independently from the literature, so this is a check of
    all four at once and it runs in exact rational arithmetic, no field, no sampling.
    """

    names = named_term_names(4)
    combination = {name: Fraction(0) for name in names}
    for label, weight in GAUSS_BONNET_DECOMPOSITION:
        scale = Fraction(weight)
        for term, value in DECLARED_VECTORS[label].items():
            combination[term] += scale * Fraction(value)
    lanczos = DECLARED_VECTORS["gauss_bonnet_lanczos"]
    target = {name: Fraction(lanczos.get(name, "0")) for name in names}
    if combination != target:
        raise TensorConstraintSearchError(
            "control failure: E1 - 4 E2 + E3 did not reproduce the declared Lanczos tensor"
        )
    derivative_terms = [name for name in ("DDR", "BoxRg", "BoxRic") if combination[name] != 0]
    if derivative_terms:  # pragma: no cover - implied by the equality above
        raise TensorConstraintSearchError(
            "control failure: the Gauss-Bonnet combination kept a derivative-carrying term"
        )
    return {
        "control": "gauss_bonnet_is_the_1_minus4_1_combination_of_the_quadratic_variations",
        "claim_under_test": (
            "the three quadratic Euler-Lagrange vectors were fitted to this engine's output rather "
            "than written from the literature"
        ),
        "verdict": "REJECTED",
        "combination": " ".join(
            f"{weight}*{label}" for label, weight in GAUSS_BONNET_DECOMPOSITION
        ),
        "evidence": (
            "E1 - 4 E2 + E3 equals the declared Lanczos vector coefficient for coefficient in "
            "exact rational arithmetic, and all three derivative-carrying coefficients cancel to "
            "zero as they must for a second-order tensor.  Four independently written vectors "
            "cannot agree on ten rational coefficients by accident."
        ),
        "checked_coefficients": len(names),
        "status": "pass",
    }


def one_sided_guarantee_control(*, modulus: int) -> dict[str, Any]:
    """The reproduced defect, kept as a control: the old arithmetic must FAIL on a starved bank.

    The old code reported ``len(divergence_nullspace) - len(sampled_vanishing_space)``.  Both are
    upper bounds, so their difference has no guaranteed sign, and on a one-jet bank in d=4 at order
    2 it lands at 1 while the same call exhibits 2 independent members -- below the exhibited lower
    bound, the opposite of the documented direction.  This control re-runs that starved cell and
    demands that the OLD quantity violate the guarantee and the NEW one honour it.  If the naive
    difference ever stops violating it, the control fails and the receipt does not build: a control
    that cannot fail is not a control.
    """

    constraints = ["generally_covariant", "derivative_order", "symmetric", "divergence_free"]
    names = named_term_names(2)
    cases: list[dict[str, Any]] = []
    naive_violations = 0
    for bank_samples in (1, 2, 3):
        bank = build_bank(4, 2, modulus, "7", bank_samples)
        bank_pairs = [(geometry, named_tensors(geometry, 2)) for geometry in bank]
        naive_vanishing = len(
            _nullspace(_sample_rows(bank_pairs, names, divergence=False), modulus)
        )
        naive_nullity = len(_nullspace(_sample_rows(bank_pairs, names, divergence=True), modulus))
        naive = naive_nullity - naive_vanishing
        search = run_search(
            dimension=4,
            order=2,
            constraints=constraints,
            modulus=modulus,
            seed="7",
            bank_samples=bank_samples,
            holdout_samples=3,
            enumerate_basis=False,
        )
        exhibited = search["uniqueness_certificate"]["independently_exhibited_dimension"]
        fixed = published_dimension(search)
        if fixed < exhibited:
            raise TensorConstraintSearchError(
                "control failure: the repaired search reported a dimension BELOW the number of "
                "members it exhibits, which is the defect this control exists to catch"
            )
        if naive < exhibited:
            naive_violations += 1
        cases.append(
            {
                "bank_samples": bank_samples,
                "naive_difference": naive,
                "repaired_dimension": fixed,
                "independently_exhibited_dimension": exhibited,
                "naive_violates_the_guarantee": bool(naive < exhibited),
            }
        )
    if not naive_violations:
        raise TensorConstraintSearchError(
            "control failure: the naive nullity difference did not violate the one-sided "
            "guarantee on any starved bank, so this control is not testing anything"
        )
    return {
        "control": "naive_nullity_difference_violates_the_one_sided_sampling_guarantee",
        "claim_under_test": (
            "subtracting the SAMPLED identically-vanishing dimension from the sampled nullity "
            "preserves the documented direction, that a dimension can only be reported too large"
        ),
        "verdict": "REJECTED",
        "seed": "7",
        "cases": cases,
        "evidence": (
            "On a starved bank the naive difference falls BELOW the number of independent members "
            "the same call exhibits, because both terms are upper bounds and their errors run in "
            "opposite directions.  The repaired search subtracts only witnessed vanishing "
            "directions and stays at or above the exhibited lower bound at every bank size."
        ),
        "naive_violations": naive_violations,
        "status": "pass",
    }


def negative_controls(
    *, modulus: int, seed: str, samples: int, dimensions: Sequence[int]
) -> list[dict[str, Any]]:
    """Abort-on-failure controls.  Each one must FAIL in the way the physics says it should."""

    controls: list[dict[str, Any]] = []

    order2_names = named_term_names(2)
    bank2 = build_bank(4, 2, modulus, f"{seed}|control", samples)
    fabricated = _declared_vector("fabricated_third", order2_names, modulus)
    residuals: list[bool] = []
    for geometry in bank2:
        terms = named_tensors(geometry, 2)
        tensor = _combine(fabricated, terms, order2_names, modulus)
        residuals.append(bool(np.any(geometry.divergence(tensor, 1) % modulus)))
    if not all(residuals):
        raise TensorConstraintSearchError(
            "control failure: the fabricated tensor R_mn - R g_mn/3 was NOT rejected"
        )
    einstein = _declared_vector("einstein", order2_names, modulus)
    einstein_clean = all(
        not np.any(
            geometry.divergence(
                _combine(einstein, named_tensors(geometry, 2), order2_names, modulus), 1
            )
            % modulus
        )
        for geometry in bank2
    )
    if not einstein_clean:
        raise TensorConstraintSearchError(
            "control failure: the Einstein tensor was not divergence-free on the bank"
        )
    controls.append(
        {
            "control": "fabricated_divergence_free_tensor_is_rejected",
            "tensor": "R_mn - (1/3) R g_mn",
            "claim_under_test": "this tensor is divergence-free",
            "verdict": "REJECTED",
            "evidence": (
                "nabla^m T_mn is non-zero on every metric jet in the control bank, while the same "
                "check on R_mn - (1/2) R g_mn returns identically zero; the contracted Bianchi "
                "identity gives the residual (1/2 - 1/3) nabla_n R"
            ),
            "samples": len(bank2),
            "status": "pass",
        }
    )

    order4_names = named_term_names(4)
    lanczos = _declared_vector("gauss_bonnet_lanczos", order4_names, modulus)
    per_dimension: list[dict[str, Any]] = []
    for dimension in dimensions:
        bank = build_bank(dimension, 4, modulus, f"{seed}|gb|{dimension}", samples)
        vanishes = True
        divergence_free = True
        for geometry in bank:
            terms = named_tensors(geometry, 4)
            tensor = _combine(lanczos, terms, order4_names, modulus)
            if np.any(tensor % modulus):
                vanishes = False
            if np.any(geometry.divergence(tensor, 1) % modulus):
                divergence_free = False
        per_dimension.append(
            {
                "dimension": dimension,
                "identically_zero_componentwise": bool(vanishes),
                "divergence_free": bool(divergence_free),
                "samples": len(bank),
            }
        )
    four = next(item for item in per_dimension if item["dimension"] == 4)
    if not four["identically_zero_componentwise"]:
        raise TensorConstraintSearchError(
            "control failure: the d=4 Gauss-Bonnet Euler-Lagrange tensor did not vanish"
        )
    for item in per_dimension:
        if item["dimension"] > 4 and item["identically_zero_componentwise"]:
            raise TensorConstraintSearchError(
                f"control failure: the Gauss-Bonnet tensor vanished in d={item['dimension']}"
            )
        if not item["divergence_free"]:
            raise TensorConstraintSearchError(
                f"control failure: the Gauss-Bonnet tensor was not conserved in "
                f"d={item['dimension']}"
            )
    controls.append(
        {
            "control": "gauss_bonnet_is_topological_in_four_dimensions",
            "tensor": (
                "H_mn = Euler-Lagrange derivative of sqrt(-g)(R^2 - 4 R_ab R^ab + "
                "R_abcd R^abcd) = 2(R R_mn - 2 R_ma R^a_n - 2 R^ab R_manb + R_mabc R_n^abc) "
                "- (1/2) g_mn (R^2 - 4 R_ab R^ab + R_abcd R^abcd)"
            ),
            "claim_under_test": (
                "the Gauss-Bonnet invariant contributes to the field equations in four dimensions"
            ),
            "verdict": "REJECTED in d=4, CONFIRMED in d>4",
            "per_dimension": per_dimension,
            "status": "pass",
        }
    )

    blocked = None
    try:
        run_search(
            dimension=4,
            order=2,
            constraints=["derivative_order", "symmetric", "divergence_free"],
            modulus=modulus,
            seed=seed,
            bank_samples=1,
            holdout_samples=1,
            enumerate_basis=False,
        )
    except ConstraintOutOfScope as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: dropping general covariance was not refused"
        )
    controls.append(
        {
            "control": "dropping_general_covariance_is_refused",
            "claim_under_test": (
                "the search will answer a constraint set that omits general covariance"
            ),
            "verdict": "REFUSED",
            "blocker_type": "ConstraintOutOfScope",
            "blocker_message": blocked,
            "status": "pass",
        }
    )
    controls.extend(framework_gate_controls(modulus=modulus, seed=seed, samples=samples))
    return controls


#: The exact false declaration this gate exists to catch: an elasticity framework claimed over a
#: general-relativity computation.  Before the gate this block was copied verbatim into the receipt
#: and the search still returned ['g_mn', 'R_mn - (1/2) R g_mn'].
ELASTICITY_DECLARATION_OVER_CURVATURE = {
    "metric_signature": "euclidean",
    "curvature": "flat",
    "primary_field_indices": 1,
    "primary_field_symmetric": False,
    "tensor_rank": 4,
    "tensor_symmetry": "symmetric",
    "concomitant_generator": "linear_strain_tensor",
}

CURVATURE_DECLARATION_OVER_ELASTICITY = {
    "metric_signature": "lorentzian",
    "curvature": "generic",
    "primary_field_indices": 2,
    "primary_field_symmetric": True,
    "tensor_rank": 2,
    "tensor_symmetry": "symmetric",
    "concomitant_generator": "riemann_tensor",
}


def framework_gate_controls(
    *, modulus: int, seed: str, samples: int
) -> list[dict[str, Any]]:
    """Controls for the declared-vs-realized framework gate.  Each must be REFUSED."""

    controls: list[dict[str, Any]] = []

    bank = build_bank(4, 2, modulus, f"{seed}|framework-control", max(2, samples))
    bank_terms = [named_tensors(geometry, 2) for geometry in bank]
    realized_curvature = measure_realized_framework(
        bank, bank_terms, dimension=4, order=2, modulus=modulus
    )
    blocked: str | None = None
    try:
        check_framework_consistency(
            ELASTICITY_DECLARATION_OVER_CURVATURE,
            realized_curvature,
            context="control:elasticity-declared-over-curvature",
        )
    except FrameworkMismatch as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: an elasticity framework declared over the curvature computation was "
            "NOT refused"
        )
    controls.append(
        {
            "control": "elasticity_framework_declared_over_a_curvature_computation_is_refused",
            "claim_under_test": (
                "flat Euclidean space, a displacement field and a linear-strain concomitant class "
                "describe the computation that returned g_mn and R_mn - (1/2) R g_mn"
            ),
            "verdict": "REFUSED",
            "blocker_type": "FrameworkMismatch",
            "blocker_message": blocked,
            "mismatched_fields": sorted(
                field
                for field in FRAMEWORK_FIELDS
                if ELASTICITY_DECLARATION_OVER_CURVATURE[field] != realized_curvature[field]
            ),
            "history": (
                "This is the defect the gate was built for.  The declared_framework block used to "
                "appear at exactly one place in the computation -- a verbatim copy into the "
                "receipt -- so this false declaration produced a sealed receipt claiming "
                "elasticity over the Einstein-tensor derivation and nothing objected."
            ),
            "status": "pass",
        }
    )

    elastic_bank = _elastic_bank(3, modulus, f"{seed}|framework-control", max(2, samples))
    patterns = enumerate_elasticity_patterns(3, modulus)
    realized_elastic = measure_realized_elastic_framework(
        elastic_bank, patterns, dimension=3, modulus=modulus, seed=f"{seed}|framework-control"
    )
    blocked = None
    try:
        check_framework_consistency(
            CURVATURE_DECLARATION_OVER_ELASTICITY,
            realized_elastic,
            context="control:curvature-declared-over-elasticity",
        )
    except FrameworkMismatch as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: a curvature framework declared over the elasticity computation was "
            "NOT refused"
        )
    controls.append(
        {
            "control": "curvature_framework_declared_over_an_elasticity_computation_is_refused",
            "claim_under_test": (
                "pseudo-Riemannian geometry with a metric field and a Riemann-tensor concomitant "
                "class describes the computation that returned the two Lame constants"
            ),
            "verdict": "REFUSED",
            "blocker_type": "FrameworkMismatch",
            "blocker_message": blocked,
            "mismatched_fields": sorted(
                field
                for field in FRAMEWORK_FIELDS
                if CURVATURE_DECLARATION_OVER_ELASTICITY[field] != realized_elastic[field]
            ),
            "status": "pass",
        }
    )

    blocked = None
    try:
        check_framework_consistency(
            {**CURVATURE_DECLARATION_OVER_ELASTICITY, "concomitant_generator": "spinor_bilinear"},
            realized_curvature,
            context="control:unsupported-generator",
        )
    except FrameworkMismatch as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: an unsupported generator declaration was NOT refused"
        )
    controls.append(
        {
            "control": "unsupported_generator_declaration_is_refused",
            "claim_under_test": (
                "a framework naming a generator this module does not implement can still be served"
            ),
            "verdict": "REFUSED",
            "blocker_type": "FrameworkMismatch",
            "blocker_message": blocked,
            "status": "pass",
        }
    )

    blocked = None
    try:
        check_framework_consistency(
            {"geometry": "free text only, no machine-checkable fields"},
            realized_curvature,
            context="control:untyped-declaration",
        )
    except FrameworkMismatch as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: an untyped free-text framework declaration was NOT refused"
        )
    controls.append(
        {
            "control": "untyped_free_text_framework_declaration_is_refused",
            "claim_under_test": (
                "a framework block of prose alone is a sufficient declaration for a sealed receipt"
            ),
            "verdict": "REFUSED",
            "blocker_type": "FrameworkMismatch",
            "blocker_message": blocked,
            "status": "pass",
        }
    )
    return controls


def elasticity_negative_controls(
    *, modulus: int, seed: str, samples: int
) -> list[dict[str, Any]]:
    """Abort-on-failure controls for the elasticity framework."""

    controls: list[dict[str, Any]] = []
    bank = _elastic_bank(3, modulus, f"{seed}|elastic-control", samples)
    patterns = enumerate_elasticity_patterns(3, modulus)
    labels = [str(pattern["label"]) for pattern in patterns]

    # -- a fabricated Navier-Cauchy operator with the wrong grad-div coefficient --------------
    lam_value, mu_value = 3, 5
    vector = [
        (lam_value * a + mu_value * b) % modulus
        for a, b in zip(
            _declared_elastic_vector("lame_lambda", labels, modulus),
            _declared_elastic_vector("lame_mu", labels, modulus),
            strict=True,
        )
    ]
    array = _elastic_combination(vector, patterns, modulus)
    truthful_wrong: list[bool] = []
    truthful_right: list[bool] = []
    for jet in bank:
        actual = jet.equilibrium(jet.stress(array, jet.strain)) % modulus
        right = (
            mu_value * jet.laplacian() + (lam_value + mu_value) * jet.gradient_of_divergence()
        ) % modulus
        wrong = (
            mu_value * jet.laplacian()
            + (lam_value + 2 * mu_value) * jet.gradient_of_divergence()
        ) % modulus
        truthful_right.append(not bool(np.any((actual - right) % modulus)))
        truthful_wrong.append(bool(np.any((actual - wrong) % modulus)))
    if not all(truthful_right):
        raise TensorConstraintSearchError(
            "control failure: the derived Navier-Cauchy operator did not reproduce the stress "
            "divergence it was derived from"
        )
    if not all(truthful_wrong):
        raise TensorConstraintSearchError(
            "control failure: the fabricated (lambda + 2 mu) operator was NOT rejected"
        )
    controls.append(
        {
            "control": "fabricated_navier_cauchy_coefficient_is_rejected",
            "operator": "mu lap u_i + (lambda + 2 mu) d_i (div u)",
            "claim_under_test": "this is the equilibrium operator of an isotropic elastic solid",
            "verdict": "REJECTED",
            "evidence": (
                "with lambda = 3 and mu = 5 the fabricated operator differs from d_j sigma_ij on "
                "every displacement jet in the control bank, while mu lap u + (lambda + mu) grad "
                "div u matches it identically on the same bank"
            ),
            "samples": len(bank),
            "status": "pass",
        }
    )

    # -- rigid motions carry no stress ---------------------------------------------------------
    rigid = _ElasticJet(
        3, rigid_motion_jet(3, 3, modulus, f"{seed}|elastic-control"), modulus, 3
    )
    if not bool(np.any(rigid.gradient % modulus)):
        raise TensorConstraintSearchError(
            "control failure: the rigid-motion probe has a zero displacement gradient and proves "
            "nothing"
        )
    if bool(np.any(rigid.strain % modulus)):
        raise TensorConstraintSearchError(
            "control failure: the rigid-motion probe has a non-zero linear strain"
        )
    if not all(
        not bool(np.any(rigid.stress(pattern["array"], rigid.strain) % modulus))
        for pattern in patterns
    ):
        raise TensorConstraintSearchError(
            "control failure: a rigid motion produced a non-zero stress"
        )
    controls.append(
        {
            "control": "a_rigid_motion_produces_no_stress",
            "claim_under_test": (
                "the enumerated laws could be built from the raw displacement gradient rather than "
                "the linear strain"
            ),
            "verdict": "REJECTED",
            "evidence": (
                "the infinitesimal rigid motion u_i = c_i + w_ij x_j has a non-zero displacement "
                "gradient and an identically zero linear strain; every enumerated pattern returns "
                "zero stress on it, and the same patterns applied to its raw gradient do not"
            ),
            "status": "pass",
        }
    )

    # -- the minor-antisymmetric trap ------------------------------------------------------------
    trap = _elastic_combination(
        _declared_elastic_vector("fabricated_minor_antisymmetric", labels, modulus),
        patterns,
        modulus,
    )
    kills_strain = all(
        not bool(np.any(jet.stress(trap, jet.strain) % modulus)) for jet in bank
    )
    lives_on_gradient = any(
        bool(np.any(jet.stress(trap, jet.gradient) % modulus)) for jet in bank
    )
    if not (kills_strain and lives_on_gradient):
        raise TensorConstraintSearchError(
            "control failure: the minor-antisymmetric combination did not behave as the trap it is"
        )
    controls.append(
        {
            "control": "minor_antisymmetric_combination_is_not_an_extra_elastic_constant",
            "tensor": "d_ik d_jl - d_il d_jk",
            "claim_under_test": "this is a third independent isotropic elastic constant",
            "verdict": "REJECTED",
            "evidence": (
                "it is a non-zero rank-4 tensor that annihilates every symmetric strain in the "
                "control bank while acting non-trivially on the raw displacement gradient, so it "
                "is invisible to elasticity and the search reports it as identically vanishing "
                "rather than as a constant"
            ),
            "status": "pass",
        }
    )

    # -- one dimension collapses to a single constant ---------------------------------------------
    one_d = run_elasticity_search(
        dimension=1,
        constraints=list(SUPPORTED_ELASTIC_CONSTRAINTS),
        modulus=modulus,
        seed=f"{seed}|elastic-control",
        bank_samples=max(2, samples),
        holdout_samples=1,
    )
    if one_d["surviving_dimension"] != 1:
        raise TensorConstraintSearchError(
            "control failure: the one-dimensional elastic space is not one-dimensional"
        )
    controls.append(
        {
            "control": "one_dimension_collapses_to_a_single_elastic_constant",
            "claim_under_test": "the two Lame constants are a dimension-independent count",
            "verdict": "REJECTED",
            "evidence": (
                "in d = 1 all three delta pairings coincide and the surviving space is "
                "one-dimensional: a rod has one elastic constant, not two.  The count of two is "
                "derived per dimension, not assumed"
            ),
            "status": "pass",
        }
    )

    # -- the operator basis degenerates in one dimension ------------------------------------------
    degenerate = navier_cauchy_derivation(
        dimension=1,
        modulus=modulus,
        seed=f"{seed}|elastic-control",
        bank_samples=max(2, samples),
        holdout_samples=1,
    )
    if degenerate["operator_basis_independent_on_the_bank"] or degenerate["assembled"] is not None:
        raise TensorConstraintSearchError(
            "control failure: the d=1 equilibrium fit reported coefficients from a degenerate "
            "operator basis"
        )
    controls.append(
        {
            "control": "degenerate_operator_basis_reports_no_coefficients",
            "claim_under_test": (
                "the equilibrium fit will always return a lap / grad-div split"
            ),
            "verdict": "REFUSED",
            "evidence": (
                "in d = 1 the two operators lap u and grad div u are the same operator, the fit is "
                "underdetermined, and the derivation returns no coefficients rather than one "
                "member of an underdetermined family"
            ),
            "status": "pass",
        }
    )
    return controls


def schwarzschild_crosscheck() -> dict[str, Any]:
    """Confront the jet engine with the repository's own independent Ricci computation."""

    from .relativity import schwarzschild_ricci_components

    components = schwarzschild_ricci_components()
    nonzero = sorted(name for name, value in components.items() if sp.simplify(value) != 0)
    return {
        "control": "schwarzschild_crosscheck_against_repository_relativity_module",
        "source": "sigma_theory_compiler.relativity.schwarzschild_ricci_components",
        "component_count": len(components),
        "nonzero_components": nonzero,
        "statement": (
            "The repository's independent sympy Schwarzschild Ricci computation returns all "
            "sixteen components zero.  A Ricci-flat metric therefore has G_mn = 0 and Lambda g_mn "
            "as the only surviving member of the derived family, which is the vacuum equation the "
            "search's d=4 order-2 result predicts."
        ),
        "status": "pass" if not nonzero else "fail",
    }


# ---------------------------------------------------------------------------
# Receipt assembly.
# ---------------------------------------------------------------------------


def _resolve(root: Path, relative: str) -> Path:
    if "\\" in relative:
        raise TensorConstraintSearchError("repository paths use forward slashes")
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise TensorConstraintSearchError(f"path escapes the repository root: {relative}")
    return target


def _file_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    return canonical_sha256(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reduction_table(search: Mapping[str, Any], newtonian: bool) -> list[dict[str, Any]]:
    """The constraint-by-constraint collapse.  This table *is* the derivation."""

    enumeration = search["enumeration"]
    order = search["derivative_order"]
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "constraint": "(declared framework only)",
            "space": "rank-2 symmetric tensor fields on a pseudo-Riemannian manifold",
            "formal_terms": "not enumerable",
            "dimension": "infinite",
        },
        {
            "step": 1,
            "constraint": "generally_covariant",
            "space": "polynomials in g, g^-1, R_abcd and its covariant derivatives",
            "formal_terms": "not enumerable",
            "dimension": "infinite (all derivative orders)",
        },
    ]
    formal = enumeration.get("formal_pattern_count")
    rows.append(
        {
            "step": 2,
            "constraint": f"derivative_order <= {order}",
            "space": "every rank-2 contraction pattern at this weight",
            "formal_terms": str(formal) if formal is not None else "reused named basis",
            "dimension": (
                str(enumeration["rank_before_symmetry"])
                if "rank_before_symmetry" in enumeration
                else str(search["independent_dimension"])
            ),
        }
    )
    rows.append(
        {
            "step": 3,
            "constraint": "symmetric",
            "space": "T_mn = T_nm after the Riemann symmetries and first Bianchi collapse",
            "formal_terms": str(formal) if formal is not None else "reused named basis",
            "dimension": str(search["independent_dimension"]),
        }
    )
    if search.get("divergence_free_applied"):
        rows.append(
            {
                "step": 4,
                "constraint": "divergence_free",
                "space": "nabla^m T_mn = 0 identically (contracted Bianchi does the work)",
                "formal_terms": str(search["divergence_free_space"]["coefficient_nullity"]),
                "dimension": str(published_dimension(search)),
            }
        )
    if newtonian:
        rows.append(
            {
                "step": 5,
                "constraint": "newtonian_limit",
                "space": "weak-field slow-motion limit reproduces Poisson's equation",
                "formal_terms": "1 constant fixed",
                "dimension": str(published_dimension(search) - 1),
            }
        )
    return rows


def _elastic_reduction_table(search: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The constraint-by-constraint collapse for the elasticity framework."""

    enumeration = search["enumeration"]
    declared = search["declared_constraints"]
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "constraint": "(declared framework only)",
            "space": "rank-4 tensor fields on flat Euclidean space",
            "formal_terms": "not enumerable",
            "dimension": "infinite",
        },
        {
            "step": 1,
            "constraint": "flat_euclidean + material_isotropy",
            "space": "constant-coefficient rank-4 tensors built from the Kronecker delta alone",
            "formal_terms": enumeration["formal_pattern_count"],
            "dimension": enumeration["rank_as_rank4_arrays"],
        },
    ]
    step = 2
    space_text = {
        "minor_symmetry": "tensors contracting the LINEAR STRAIN, not the raw gradient",
        "stress_symmetry": "laws whose Cauchy stress is symmetric on every jet in the bank",
        "major_symmetry": "laws admitting a strain-energy function",
        "cauchy_relation": "laws additionally obeying the rari-constant hypothesis lambda = mu",
    }
    for entry in enumeration["constraint_cascade"]:
        if entry["constraint"] not in declared:
            raise TensorConstraintSearchError(
                "the constraint cascade reports a constraint that was never declared"
            )
        rows.append(
            {
                "step": step,
                "constraint": entry["constraint"],
                "space": space_text[entry["constraint"]],
                "formal_terms": enumeration["formal_pattern_count"],
                "dimension": entry["distinct_law_dimension"],
            }
        )
        step += 1
    rows.append(
        {
            "step": step,
            "constraint": "(final)",
            "space": search["admissible_space"]["basis"],
            "formal_terms": enumeration["formal_pattern_count"],
            "dimension": search["surviving_dimension"],
        }
    )
    return rows


def _framework_consistency_block(
    typed: Mapping[str, Any], searches: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Confront the typed declaration with every search's measured framework, or refuse."""

    per_search = [
        check_framework_consistency(
            typed,
            item["realized_framework"],
            context=str(item.get("search_id", "search")),
        )
        for item in searches
    ]
    return {
        "gate": "declared_framework vs the framework measured off the computation",
        "typed_declaration": {field: typed[field] for field in FRAMEWORK_FIELDS},
        "searches_checked": [str(item.get("search_id", "search")) for item in searches],
        "per_search": per_search,
        "all_searches_agree": True,
        "blocker_type_on_disagreement": "FrameworkMismatch",
        "what_this_closes": (
            "Before this gate the declared_framework block was inert: it was copied verbatim into "
            "the receipt and referenced nowhere in the computation, so a receipt could carry a "
            "framework claim describing a completely different theory from the one that produced "
            "its numbers.  Every field above is now MEASURED off the arrays the basis generator "
            "built -- the signature of the base-point metric, whether the sampled curvature is "
            "identically zero, the index structure of the primary field jet, the rank and pair "
            "symmetry of the produced terms, and which terms die on an exactly flat jet -- and a "
            "receipt is refused when any of them disagrees with what was declared."
        ),
    }


def declared_generator(config: Mapping[str, Any]) -> str:
    """Read the typed generator out of the declared framework, failing closed on prose.

    This is the one function that turns ``declared_framework`` from a receipt decoration into an
    input to the computation: the value it returns selects which basis generator runs.
    """

    declared = config.get("declared_framework")
    if not isinstance(declared, Mapping):
        raise FrameworkMismatch("the config carries no declared_framework block")
    typed = declared.get("machine_checkable")
    if not isinstance(typed, Mapping):
        raise FrameworkMismatch(
            "the declared_framework block carries no machine_checkable sub-block.  A framework "
            "stated only in prose cannot be checked against the computation it claims to describe, "
            f"so it is refused.  Required fields: {list(FRAMEWORK_FIELDS)}"
        )
    missing = [field for field in FRAMEWORK_FIELDS if field not in typed]
    if missing:
        raise FrameworkMismatch(
            f"the machine_checkable framework block is missing {sorted(missing)}"
        )
    generator = typed["concomitant_generator"]
    if generator not in SUPPORTED_GENERATORS:
        raise FrameworkMismatch(
            f"declared concomitant_generator {generator!r} is not one this module implements "
            f"{list(SUPPORTED_GENERATORS)}"
        )
    return str(generator)


def run_tensor_constraint_search(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Dispatch on the DECLARED framework and assemble the receipt for whichever it selects.

    The declaration is load-bearing twice over: it chooses the basis generator here, and it is
    re-checked field by field against the framework measured off the generator's own arrays before
    a receipt is sealed.
    """

    if config.get("schema_version") != CONFIG_SCHEMA:
        raise TensorConstraintSearchError("unexpected config schema")
    generator = declared_generator(config)
    if generator == "linear_strain_tensor":
        return _run_elasticity_campaign(config, root)
    return _run_curvature_campaign(config, root)


def _run_curvature_campaign(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Run every declared search, control and generalization, and assemble the receipt."""

    typed_framework = dict(config["declared_framework"]["machine_checkable"])
    arithmetic = config["arithmetic"]
    primes = [int(value) for value in arithmetic["primes"]]
    seed = str(arithmetic["jet_seed"])
    bank_samples = int(arithmetic["bank_samples"])
    holdout_samples = int(arithmetic["holdout_samples"])

    searches: list[dict[str, Any]] = []
    for spec in config["searches"]:
        replays: list[dict[str, Any]] = []
        for modulus in primes:
            replays.append(
                run_search(
                    dimension=int(spec["dimension"]),
                    order=int(spec["derivative_order"]),
                    constraints=list(spec["constraints"]),
                    modulus=modulus,
                    seed=seed,
                    bank_samples=bank_samples,
                    holdout_samples=holdout_samples,
                    enumerate_basis=bool(spec.get("enumerate_basis", False)),
                )
            )
        primary = replays[0]
        for replay in replays[1:]:
            if replay != primary:
                raise TensorConstraintSearchError(
                    f"search {spec['search_id']} disagreed between prime replays"
                )
        primary = dict(primary)
        primary["search_id"] = spec["search_id"]
        primary["purpose"] = spec["purpose"]
        primary["prime_replays"] = len(primes)
        searches.append(primary)

    # -- THE GATE.  Nothing downstream runs until the declared framework is confronted with the
    # framework measured off the arrays each search actually built.
    framework_consistency = _framework_consistency_block(typed_framework, searches)

    by_id = {item["search_id"]: item for item in searches}
    headline = by_id["d4-order2-einstein"]
    einstein_member = next(
        item for item in headline["exhibited_members"] if item["name"] == "einstein"
    )
    cosmological_member = next(
        item for item in headline["exhibited_members"] if item["name"] == "cosmological"
    )
    if not headline["uniqueness_certificate"]["sandwich_tight"]:
        raise TensorConstraintSearchError(
            "the exhibited members did not close the uniqueness sandwich in d=4 at order 2"
        )
    if published_dimension(headline) != 2:
        raise TensorConstraintSearchError(
            "the d=4 order-2 divergence-free space is not two-dimensional"
        )
    if not (einstein_member["in_reported_space"] and cosmological_member["in_reported_space"]):
        raise TensorConstraintSearchError("G_mn and g_mn are not both in the reported space")

    newtonian = newtonian_limit_chain()

    reduction_tables = {
        item["search_id"]: _reduction_table(
            item, "newtonian_limit" in item["declared_constraints"]
        )
        for item in searches
    }

    # -- generalizations ---------------------------------------------------
    gauss_bonnet = {
        "question": (
            "does the Gauss-Bonnet combination appear as an ADDITIONAL divergence-free "
            "contribution once d > 4?"
        ),
        "by_dimension": [],
        "verdict_depends_on": (
            "a DIFFERENCE of three dimensions.  Differencing open upper bounds is not a "
            "measurement, so each of the three searches must have closed its own sandwich before "
            "this verdict is computed at all; otherwise the verdict is refused, not published."
        ),
    }
    for search_id in ("d4-order4-relaxed", "d5-order4-gaussbonnet", "d6-order4-gaussbonnet"):
        item = by_id[search_id]
        gauss_bonnet["by_dimension"].append(
            {
                "dimension": item["dimension"],
                "identically_vanishing_dimension": item["identically_vanishing"]["dimension"],
                "identically_vanishing_vectors": item["identically_vanishing"]["vectors"],
                "identically_vanishing_witnesses": item["identically_vanishing"]["witnesses"],
                "sandwich_tight": item["uniqueness_certificate"]["sandwich_tight"],
                "divergence_free_distinct_dimension": published_dimension(item),
                "gauss_bonnet_member": next(
                    (
                        member
                        for member in item["exhibited_members"]
                        if member["name"] == "gauss_bonnet_lanczos"
                    ),
                    None,
                ),
            }
        )
    d4_order4 = published_dimension(by_id["d4-order4-relaxed"])
    d5_order4 = published_dimension(by_id["d5-order4-gaussbonnet"])
    d6_order4 = published_dimension(by_id["d6-order4-gaussbonnet"])
    if not (d5_order4 == d6_order4 == d4_order4 + 1):
        raise TensorConstraintSearchError(
            "the Gauss-Bonnet generalization did not add exactly one dimension for d>4"
        )
    gauss_bonnet["verdict"] = (
        f"YES.  The distinct-tensor divergence-free space at derivative order 4 is {d4_order4} in "
        f"d=4 and {d5_order4} in d=5 and d=6.  The extra direction is exactly the Lanczos-Lovelock "
        "Gauss-Bonnet tensor, which is identically zero in four dimensions and non-zero above.  "
        "All three dimensions are two-sided: each is an exhibited lower bound that met its sampled "
        "upper bound, so the difference is a difference of measurements and not of open bounds."
    )

    candidates = _load_json(_resolve(root, "runs/gpu-baryonic-screen/nonlocal-localization-v1.json"))
    passing = [item for item in candidates["families"] if item.get("stability") == "STABLE_PASS"]
    candidate_total = sum(int(item["size"]) for item in passing)
    sectors = sorted({item["sector_id"] for item in passing})

    relaxation = {
        "question": (
            "is the modified-gravity space this project has been enumerating by hand derivable "
            "from relaxing the declared constraints?"
        ),
        "axis_1_derivative_order": {
            "relaxation": "derivative_order <= 2 relaxed to <= 4, d = 4, all else unchanged",
            "dimension_before": published_dimension(by_id["d4-order2-einstein"]),
            "dimension_after": d4_order4,
            "new_directions": by_id["d4-order4-relaxed"]["divergence_free_space"]["basis"],
            "interpretation": (
                "The enlarged space contains the Euler-Lagrange derivative of R^2, which is the "
                "quadratic term of an f(R) theory: f(R) = R + a R^2 lands exactly on "
                "G_mn + a E1_mn.  So the f(R)-type and higher-curvature families are not separate "
                "inventions -- they are what the order-2 constraint was suppressing."
            ),
            "f_of_r_member_is_divergence_free": None,
        },
        "axis_2_field_content": {
            "relaxation": (
                "declared field content relaxed from {metric} to {metric, scalar}; derivative "
                "order and the second-order-field-equation requirement unchanged"
            ),
            "result": (
                "This axis is NOT computed by this module.  It is named here because it is where "
                "the repository's surviving screened-gravity candidates actually live, and "
                "pretending the metric-only search reaches them would be false."
            ),
            "cited_classification": "Horndeski (1974): the most general scalar-tensor theory with "
            "second-order field equations, parameterised by G2, G3, G4, G5",
            "repository_candidates": {
                "receipt": "runs/gpu-baryonic-screen/nonlocal-localization-v1.json",
                "surviving_families": len(passing),
                "surviving_candidates": candidate_total,
                "sector_ids": sectors,
                "declared_horndeski_slice": "G2 = X + c_K X^2, G3 = 0, G4 = 1/2",
                "reading": (
                    "G4 = 1/2 and G3 = 0 is precisely the slice in which the *tensor* sector is "
                    "untouched General Relativity -- so those candidates sit on top of the "
                    "a G_mn + Lambda g_mn this search derives, with all of the modification "
                    "carried by a nonlinear G2(X), the K-mouflage arm.  That is the honest "
                    "connection: this search derives the gravitational sector those candidates "
                    "assume, and shows that reaching the candidates themselves requires the "
                    "second declared relaxation, not the first."
                ),
            },
        },
    }
    f_of_r = next(
        (
            member
            for member in by_id["d4-order4-relaxed"].get("exhibited_members", [])
            if member["name"] == "quadratic_r_squared_euler_lagrange"
        ),
        None,
    )
    relaxation["axis_1_derivative_order"]["f_of_r_member_is_divergence_free"] = (
        f_of_r["divergence_free"] if f_of_r else None
    )

    relaxation_controls = {
        "dropping_divergence_free_enlarges_the_space": {
            "with_constraint": published_dimension(by_id["d4-order2-einstein"]),
            "without_constraint": published_dimension(by_id["d4-order2-no-conservation"]),
            "strictly_larger": bool(
                published_dimension(by_id["d4-order2-no-conservation"])
                > published_dimension(by_id["d4-order2-einstein"])
            ),
            "surviving_family_without_it": by_id["d4-order2-no-conservation"]["surviving_family"],
        },
        "dropping_general_covariance_is_refused": {
            "blocker_type": "ConstraintOutOfScope",
            "reported_in": "negative_controls",
        },
    }
    if not relaxation_controls["dropping_divergence_free_enlarges_the_space"]["strictly_larger"]:
        raise TensorConstraintSearchError(
            "dropping the conservation constraint did not enlarge the space"
        )

    controls = negative_controls(
        modulus=primes[0],
        seed=seed,
        samples=int(arithmetic["control_samples"]),
        dimensions=[int(value) for value in config["gauss_bonnet_dimensions"]],
    )
    controls.append(gauss_bonnet_decomposition())
    controls.append(one_sided_guarantee_control(modulus=primes[0]))
    controls.append(schwarzschild_crosscheck())

    bindings = {
        "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(_resolve(root, CONFIG_PATH))},
        "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_resolve(root, SOURCE_PATH))},
        "test": {"path": TEST_PATH, "file_sha256": _file_sha(_resolve(root, TEST_PATH))},
        "framework_test": {
            "path": FRAMEWORK_TEST_PATH,
            "file_sha256": _file_sha(_resolve(root, FRAMEWORK_TEST_PATH)),
        },
    }
    for entry in REUSED_MACHINERY:
        if entry["kind"] == "text":
            bindings[entry["artifact"]] = {
                "path": entry["artifact"],
                "file_sha256": _file_sha(_resolve(root, entry["artifact"])),
            }
        elif entry["kind"] == "json":
            bindings[entry["artifact"]] = {
                "path": entry["artifact"],
                "semantic_sha256": _json_sha(_resolve(root, entry["artifact"])),
            }

    final_family = headline["divergence_free_space"]["basis"]
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "declared_framework": dict(config["declared_framework"]),
        "framework_consistency": framework_consistency,
        "arithmetic": {
            "primes": [str(value) for value in primes],
            "jet_seed": seed,
            "bank_samples": bank_samples,
            "holdout_samples": holdout_samples,
            "control_samples": int(arithmetic["control_samples"]),
            "method": (
                "Every tensor identity is evaluated on random metric jets over a prime field.  A "
                "generally covariant concomitant of order k depends only on the k-jet of the "
                "metric, and that jet is unconstrained, so a random jet probes the identity fully. "
                " The sampling error is one-sided: a finite bank can only report the solution "
                "space too LARGE, never too small, so uniqueness is never manufactured by "
                "sampling.  It is closed from below by exhibiting members and verifying them on "
                "held-out jets.  The direction of that guarantee survives the SUBTRACTION only "
                "because the subtracted identically-vanishing dimension is never the sampled one "
                "-- which is itself an upper bound and would tip the error downward -- but the "
                "WITNESSED one: declared vectors, each confronted with the bank and the holdout.  "
                "When the sampled vanishing dimension exceeds the witnessed one, or when the "
                "exhibited members fail to span the sampled nullspace, the search reports an open "
                "upper bound and refuses to publish a dimension at all."
            ),
            "one_sided_guarantee": (
                "reported distinct-tensor dimension >= true dimension, always; and no dimension is "
                "published unless an independently exhibited lower bound meets it exactly"
            ),
        },
        "basis_completeness_citations": [dict(item) for item in BASIS_COMPLETENESS_CITATIONS],
        "reused_repository_machinery": [dict(item) for item in REUSED_MACHINERY],
        "searches": searches,
        "reduction_tables": reduction_tables,
        "newtonian_limit": newtonian,
        "generalizations": {"gauss_bonnet": gauss_bonnet},
        "relaxation_controls": {**relaxation_controls, "derivable_modified_gravity": relaxation},
        "negative_controls": controls,
        "counts": {
            "searches": len(searches),
            "negative_controls": len(controls),
            "prime_replays": len(primes),
            "enumerated_patterns_d4_order2": headline["enumeration"]["formal_pattern_count"],
            "enumerated_patterns_d4_order4": by_id["d4-order4-relaxed"]["enumeration"][
                "formal_pattern_count"
            ],
            "final_family_dimension": published_dimension(headline),
            "free_parameters_after_newtonian_limit": published_dimension(headline) - 1,
            "searches_with_a_closed_sandwich": sum(
                1 for item in searches if item.get("dimension_published")
            ),
            "screened_gravity_candidates_referenced": candidate_total,
        },
        "decision": (
            "DERIVED: with d=4, derivative order 2, rank-2 symmetric, generally covariant and "
            "divergence-free declared, the surviving space is exactly two-dimensional and is "
            "spanned by G_mn = R_mn - (1/2) R g_mn and g_mn.  The Newtonian limit fixes the "
            "coefficient of G_mn to c^4/(8 pi G), equivalently kappa = 8 pi G/c^4, and leaves the "
            "coefficient of g_mn free.  Nothing here is novel: the framework and the constraints "
            "were DECLARED, and the engine derived what they force."
        ),
        "scope": (
            "One declared framework (pseudo-Riemannian geometry), one declared concomitant class "
            "(polynomial in curvature and its covariant derivatives with constant coefficients), "
            "one declared rank and symmetry, and five declared derivative-order/dimension cells.  "
            "Uniqueness is uniqueness WITHIN that declared basis and order; its extension to the "
            "full non-polynomial concomitant class is Lovelock (1972), cited and not derived.  "
            "No observational data is opened, no novelty is claimed, and the framework was given "
            "to the search rather than found by it."
        ),
        "config_sha256": canonical_sha256(config),
        "source_bindings": bindings,
    }
    del final_family
    _no_floats(body)
    return {**body, "content_sha256": canonical_sha256(body)}


def _run_elasticity_campaign(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The SECOND declared framework, run end to end: linear elasticity on flat Euclidean space.

    Same engine, same exact prime-field arithmetic, same sandwich discipline -- and a genuinely
    different answer, because the declaration selected a genuinely different basis generator.  The
    curvature search returns ``g_mn`` and ``R_mn - (1/2) R g_mn``; this one returns the two Lame
    constants and the Navier-Cauchy operator, and neither can be reached from the other's config.
    """

    typed_framework = dict(config["declared_framework"]["machine_checkable"])
    arithmetic = config["arithmetic"]
    primes = [int(value) for value in arithmetic["primes"]]
    seed = str(arithmetic["jet_seed"])
    bank_samples = int(arithmetic["bank_samples"])
    holdout_samples = int(arithmetic["holdout_samples"])

    searches: list[dict[str, Any]] = []
    for spec in config["searches"]:
        replays = [
            run_elasticity_search(
                dimension=int(spec["dimension"]),
                constraints=list(spec["constraints"]),
                modulus=modulus,
                seed=seed,
                bank_samples=bank_samples,
                holdout_samples=holdout_samples,
            )
            for modulus in primes
        ]
        primary = replays[0]
        for replay in replays[1:]:
            if replay != primary:
                raise TensorConstraintSearchError(
                    f"search {spec['search_id']} disagreed between prime replays"
                )
        primary = dict(primary)
        primary["search_id"] = spec["search_id"]
        primary["purpose"] = spec["purpose"]
        primary["prime_replays"] = len(primes)
        searches.append(primary)

    # -- THE GATE, on this framework too -----------------------------------
    framework_consistency = _framework_consistency_block(typed_framework, searches)

    by_id = {item["search_id"]: item for item in searches}
    headline_id = str(config["headline_search_id"])
    headline = by_id[headline_id]
    if headline["surviving_dimension"] != 2:
        raise TensorConstraintSearchError(
            "the isotropic linear-elastic space is not two-dimensional in the headline cell"
        )
    lame = [item for item in headline["exhibited_members"] if item["name"] in
            ("lame_lambda", "lame_mu")]
    if not all(item["in_reported_space"] for item in lame):
        raise TensorConstraintSearchError(
            "the two Lame directions are not both in the reported space"
        )
    if _rank(
        [
            _declared_elastic_vector(item["name"], list(ELASTICITY_PATTERN_TEXT), primes[0])
            for item in lame
        ],
        primes[0],
    ) != headline["surviving_dimension"]:
        raise TensorConstraintSearchError(
            "the exhibited Lame members did not close the uniqueness sandwich"
        )

    equilibrium = navier_cauchy_derivation(
        dimension=headline["dimension"],
        modulus=primes[0],
        seed=seed,
        bank_samples=bank_samples,
        holdout_samples=holdout_samples,
    )
    if equilibrium["assembled"] is None:
        raise TensorConstraintSearchError(
            "the equilibrium operator did not solve uniquely in the headline cell"
        )
    for item in equilibrium["directions"]:
        if not item["verified_on_holdout"]:
            raise TensorConstraintSearchError(
                "a derived equilibrium coefficient failed on held-out displacement jets"
            )

    reduction_tables = {item["search_id"]: _elastic_reduction_table(item) for item in searches}

    sweep: list[dict[str, Any]] = []
    for dimension in [int(value) for value in config["dimension_sweep"]]:
        item = run_elasticity_search(
            dimension=dimension,
            constraints=list(config["sweep_constraints"]),
            modulus=primes[0],
            seed=seed,
            bank_samples=bank_samples,
            holdout_samples=holdout_samples,
        )
        sweep.append(
            {
                "dimension": dimension,
                "elastic_constants": item["surviving_dimension"],
                "basis": item["admissible_space"]["basis"],
            }
        )
    if not (
        sweep[0]["elastic_constants"] == 1
        and all(entry["elastic_constants"] == 2 for entry in sweep[1:])
    ):
        raise TensorConstraintSearchError(
            "the elastic-constant count is not 1 in d=1 and 2 in every higher dimension swept"
        )

    grown = by_id["d3-major-symmetry-only"]["surviving_dimension"]
    shrunk = by_id["d3-cauchy-relation"]["surviving_dimension"]
    stress_only = by_id["d3-stress-symmetry-only"]["surviving_dimension"]
    relaxation_controls = {
        "dropping_the_symmetry_of_the_argument_enlarges_the_space": {
            "with_constraint": headline["surviving_dimension"],
            "without_constraint": grown,
            "strictly_larger": bool(grown > headline["surviving_dimension"]),
            "surviving_family_without_it": by_id["d3-major-symmetry-only"]["admissible_space"][
                "basis"
            ],
        },
        "imposing_the_cauchy_relation_shrinks_the_space": {
            "with_constraint": shrunk,
            "without_constraint": headline["surviving_dimension"],
            "strictly_smaller": bool(shrunk < headline["surviving_dimension"]),
            "reading": (
                "lambda = mu is the 19th-century rari-constant hypothesis.  It is an extra "
                "declaration, it collapses the space to a single constant, and it is contradicted "
                "by measurement -- so the search reports it as a constraint that costs a "
                "dimension, not as a result."
            ),
        },
        "dropping_general_isotropy_is_refused": {
            "blocker_type": "ConstraintOutOfScope",
            "reported_in": "negative_controls",
        },
    }
    if not relaxation_controls["dropping_the_symmetry_of_the_argument_enlarges_the_space"][
        "strictly_larger"
    ]:
        raise TensorConstraintSearchError(
            "dropping the argument-symmetry constraint did not enlarge the space"
        )
    if not relaxation_controls["imposing_the_cauchy_relation_shrinks_the_space"][
        "strictly_smaller"
    ]:
        raise TensorConstraintSearchError("the Cauchy relation did not shrink the space")

    generalizations = {
        "dimension_sweep": {
            "question": "is the count of two elastic constants a dimension-independent fact?",
            "by_dimension": sweep,
            "verdict": (
                "NO.  In d = 1 all three delta pairings coincide and a rod has ONE elastic "
                "constant; from d = 2 upward the count is exactly two.  The count is derived per "
                "dimension by the same enumeration, not carried over."
            ),
        },
        "minor_symmetry_is_not_an_independent_declaration": {
            "question": (
                "is the minor symmetry of the elasticity tensor an extra assumption, or is it "
                "forced by the symmetry of the Cauchy stress?"
            ),
            "dimension_with_minor_symmetry_declared": headline["surviving_dimension"],
            "dimension_with_only_stress_symmetry_declared": stress_only,
            "basis_with_only_stress_symmetry_declared": by_id["d3-stress-symmetry-only"][
                "admissible_space"
            ]["basis"],
            "verdict": (
                "FORCED.  Declaring only that the Cauchy stress is symmetric -- angular momentum "
                "balance -- and letting the law contract the RAW displacement gradient still "
                "collapses the space onto the same two directions.  The minor symmetry did not "
                "need to be declared separately; it was already implied."
            )
            if stress_only == headline["surviving_dimension"]
            else "NOT FORCED at the sampled dimension",
        },
    }

    controls = elasticity_negative_controls(
        modulus=primes[0], seed=seed, samples=int(arithmetic["control_samples"])
    )
    blocked: str | None = None
    try:
        run_elasticity_search(
            dimension=3,
            constraints=["minor_symmetry", "stress_symmetry", "major_symmetry"],
            modulus=primes[0],
            seed=seed,
            bank_samples=2,
            holdout_samples=1,
        )
    except ConstraintOutOfScope as error:
        blocked = str(error)
    if blocked is None:
        raise TensorConstraintSearchError(
            "control failure: dropping flat_euclidean was not refused"
        )
    controls.append(
        {
            "control": "dropping_the_flat_isotropic_background_is_refused",
            "claim_under_test": (
                "the search will answer a constraint set that omits the flat isotropic background"
            ),
            "verdict": "REFUSED",
            "blocker_type": "ConstraintOutOfScope",
            "blocker_message": blocked,
            "status": "pass",
        }
    )
    controls.extend(
        framework_gate_controls(
            modulus=primes[0], seed=seed, samples=int(arithmetic["control_samples"])
        )
    )

    bindings = {
        "config": {
            "path": ELASTICITY_CONFIG_PATH,
            "file_sha256": _file_sha(_resolve(root, ELASTICITY_CONFIG_PATH)),
        },
        "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_resolve(root, SOURCE_PATH))},
        "test": {"path": TEST_PATH, "file_sha256": _file_sha(_resolve(root, TEST_PATH))},
        "framework_test": {
            "path": FRAMEWORK_TEST_PATH,
            "file_sha256": _file_sha(_resolve(root, FRAMEWORK_TEST_PATH)),
        },
    }

    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "declared_framework": dict(config["declared_framework"]),
        "framework_consistency": framework_consistency,
        "arithmetic": {
            "primes": [str(value) for value in primes],
            "jet_seed": seed,
            "bank_samples": bank_samples,
            "holdout_samples": holdout_samples,
            "control_samples": int(arithmetic["control_samples"]),
            "method": (
                "Every elasticity identity is evaluated on random displacement jets over a prime "
                "field.  The jet of a displacement field is unconstrained, so a random jet probes "
                "the identity fully, and the sampling error is one-sided in exactly the same way "
                "as on the curvature side: a finite bank can only report the admissible space too "
                "LARGE.  It is closed from below by exhibiting the two Lame directions and "
                "re-verifying them on held-out jets."
            ),
        },
        "basis_completeness_citations": [
            dict(item) for item in ELASTICITY_COMPLETENESS_CITATIONS
        ],
        "reused_repository_machinery": [dict(item) for item in REUSED_MACHINERY],
        "searches": searches,
        "reduction_tables": reduction_tables,
        "equilibrium_operator": equilibrium,
        "generalizations": generalizations,
        "relaxation_controls": relaxation_controls,
        "negative_controls": controls,
        "counts": {
            "searches": len(searches),
            "negative_controls": len(controls),
            "prime_replays": len(primes),
            "enumerated_patterns": headline["enumeration"]["formal_pattern_count"],
            "final_family_dimension": headline["surviving_dimension"],
            "dimensions_swept": len(sweep),
        },
        "decision": (
            "DERIVED: with flat Euclidean 3-space, a displacement field, a rank-4 concomitant "
            "linear in the LINEAR STRAIN tensor, material isotropy and a symmetric Cauchy stress "
            "declared, the surviving space is exactly two-dimensional and is spanned by "
            "d_ij d_kl and d_ik d_jl + d_il d_jk -- the Lame constants lambda and mu.  The static "
            "equilibrium operator those force is "
            f"{equilibrium['assembled']}, the Navier-Cauchy equation.  This is the SAME engine "
            "that returns g_mn and R_mn - (1/2) R g_mn from the curvature declaration; the "
            "different answer comes from the different declaration, which is the point."
        ),
        "scope": (
            "One declared framework (flat Euclidean space with a displacement field), one declared "
            "concomitant class (rank-4, constant-coefficient, linear in the linear strain), one "
            "declared material symmetry (isotropy) and four declared dimension cells.  That the "
            "delta pairings exhaust the O(n)-invariant rank-4 tensors is Weyl's first fundamental "
            "theorem, CITED and not derived.  Nothing here is novel: Cauchy (1828) and Navier "
            "(1821) are the mathematics.  No observational data is opened."
        ),
        "config_sha256": canonical_sha256(config),
        "source_bindings": bindings,
    }
    _no_floats(body)
    return {**body, "content_sha256": canonical_sha256(body)}


def _no_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise TensorConstraintSearchError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _no_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _no_floats(child, f"{path}[{index}]")


def validate_receipt(value: Mapping[str, Any], config: Mapping[str, Any], root: Path) -> None:
    """Fail closed on schema drift, seal drift, claim drift, host paths, or replay drift."""

    generator = declared_generator(config)
    expected_keys = _TOP_KEYS_STRAIN if generator == "linear_strain_tensor" else _TOP_KEYS
    if set(value) != expected_keys:
        raise TensorConstraintSearchError("tensor-constraint receipt schema changed")
    realized = value.get("framework_consistency", {}).get("per_search", [{}])
    if not realized or any(
        item.get("realized", {}).get("concomitant_generator") != generator for item in realized
    ):
        raise TensorConstraintSearchError(
            "tensor-constraint receipt was produced by a different generator than the config "
            "declares"
        )
    if value.get("schema_version") != RESULT_SCHEMA:
        raise TensorConstraintSearchError("tensor-constraint receipt schema version changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise TensorConstraintSearchError("tensor-constraint receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise TensorConstraintSearchError("tensor-constraint claim boundary changed")
    if _HOST_PATH.search(json.dumps(value, sort_keys=True)):
        raise TensorConstraintSearchError("tensor-constraint receipt persisted a host path")
    if value.get("config_sha256") != canonical_sha256(config):
        raise TensorConstraintSearchError("tensor-constraint config binding changed")
    _no_floats(body)
    for control in value.get("negative_controls", []):
        if control.get("status") != "pass":
            raise TensorConstraintSearchError("a negative control is not marked pass")
    expected = run_tensor_constraint_search(config, root)
    if dict(value) != expected:
        raise TensorConstraintSearchError("tensor-constraint exact replay changed")


def write_receipt(value: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output)
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise TensorConstraintSearchError("refusing to overwrite an immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate the space of generally covariant, symmetric, divergence-free rank-2 "
        "tensors and report what the declared constraints force."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--framework",
        choices=["riemann_tensor", "linear_strain_tensor"],
        default="riemann_tensor",
        help="which declared framework to run; selects the default config and receipt path",
    )
    parser.add_argument("--validate-checked", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    strain = arguments.framework == "linear_strain_tensor"
    config_path = arguments.config or _resolve(
        root, ELASTICITY_CONFIG_PATH if strain else CONFIG_PATH
    )
    output_path = arguments.output or _resolve(
        root, ELASTICITY_OUTPUT_PATH if strain else OUTPUT_PATH
    )
    config = _load_json(config_path)
    if arguments.validate_checked:
        validate_receipt(_load_json(output_path), config, root)
        print(json.dumps({"validated": True}, indent=2))
        return 0
    receipt = run_tensor_constraint_search(config, root)
    if not arguments.dry_run:
        write_receipt(receipt, output_path)
    if declared_generator(config) == "linear_strain_tensor":
        headline_id = str(config["headline_search_id"])
        headline = next(
            item for item in receipt["searches"] if item["search_id"] == headline_id
        )
        summary = {
            "declared_generator": "linear_strain_tensor",
            "enumerated_patterns": headline["enumeration"]["formal_pattern_count"],
            "reduction_table": [
                {"constraint": row["constraint"], "dimension": row["dimension"]}
                for row in receipt["reduction_tables"][headline_id]
            ],
            "family": headline["admissible_space"]["basis"],
            "equilibrium_operator": receipt["equilibrium_operator"]["assembled"],
            "dimension_sweep": receipt["generalizations"]["dimension_sweep"]["by_dimension"],
            "framework_gate": receipt["framework_consistency"]["all_searches_agree"],
            "decision": receipt["decision"],
        }
    else:
        headline = next(
            item for item in receipt["searches"] if item["search_id"] == "d4-order2-einstein"
        )
        summary = {
            "declared_generator": "riemann_tensor",
            "enumerated_patterns": headline["enumeration"]["formal_pattern_count"],
            "reduction_table": [
                {"constraint": row["constraint"], "dimension": row["dimension"]}
                for row in receipt["reduction_tables"]["d4-order2-einstein"]
            ],
            "family": headline["divergence_free_space"]["basis"],
            "coupling_constant_kappa": receipt["newtonian_limit"]["coupling_constant_kappa"],
            "lambda_status": "free (unforced by the Newtonian limit)",
            "gauss_bonnet_verdict": receipt["generalizations"]["gauss_bonnet"]["verdict"],
            "framework_gate": receipt["framework_consistency"]["all_searches_agree"],
            "decision": receipt["decision"],
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
