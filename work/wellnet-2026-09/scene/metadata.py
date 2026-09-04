"""metadata.py -- the charter's PARAMETER METADATA CONTRACT, machine-readable.

The charter (section "The parameter metadata contract") lists seventeen items
every quantity entering Invariant must carry.  This module makes each of them a
typed field with a closed vocabulary, so that the admissibility compiler in
`../compiler/` can prune candidate laws BEFORE any data fitting -- which is the
charter's stated reason for the contract existing at all:

    "This metadata is what allows the admissibility compiler to prune candidate
     laws before data fitting."

The seventeen contract items and where they live:

    1  physical name + operational definition   name, definition
    2  scalar/vector/tensor/graph/path/history  kind, rank
    3  units                                    dim  (exact exponent vector)
    4  coordinate and reference frame           frame
    5  transformation behaviour                 translation, rotation, boost,
                                                parity, time_reversal
    6  point/region/path/interval it belongs to support
    7  measurement or derivation source         source
    8  direct/derived/latent/nuisance           status
    9  resolution and smoothing scale           resolution_m, smoothing_m
   10  full uncertainty and covariance group    uncertainty, covariance_group
   11  boundary or gauge convention             gauge
   12  coarse-graining behaviour                coarse_grain
   13  causal availability                      causal
   14  completeness and selection function      completeness, selection
   15  allowed operations                       allowed_ops
   16  known algebraic dependencies             depends_on  (symbolic)
   17  independently measurable in test sample  independently_measurable (+why)

DESIGN NOTE ON UNITS.  Units are stored as an exponent vector over the SI base
dimensions actually needed here -- (M, L, T, Theta, Q) -- not as a string.  A
string cannot be checked; a vector can.  `Dim` supports exact arithmetic so
`dimensionally_consistent()` is a real test rather than a promise.

DESIGN NOTE ON GAUGE.  The charter is explicit: "Absolute Newtonian potential
must never be used without an operational boundary rule, because adding a
constant otherwise changes the supposed physical parameter without changing the
underlying force."  So `gauge` is not free text: a quantity whose translation
behaviour is `SHIFTS_BY_CONSTANT` is REQUIRED to name a boundary rule, and
`Quantity.__post_init__` raises if it does not.  This programme has already been
burned here -- two defensible global potential-depth rules differ by 0.87 dex
against a 0.9 dex gate margin.

NO OBSERVATIONAL DATA IS OPENED BY THIS MODULE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ===================================================================== units

#: base dimensions, in order.  Theta is thermodynamic temperature, Q is charge.
BASE_DIMS = ("M", "L", "T", "Theta", "Q")


@dataclass(frozen=True)
class Dim:
    """An exact dimension: exponents over (M, L, T, Theta, Q) as Fractions.

    Exact rational exponents matter.  A candidate law containing sqrt(G rho)
    has a half-integer time exponent, and float exponents accumulate error that
    makes a genuinely inconsistent law look consistent at 1e-16.
    """
    M: Fraction = Fraction(0)
    L: Fraction = Fraction(0)
    T: Fraction = Fraction(0)
    Theta: Fraction = Fraction(0)
    Q: Fraction = Fraction(0)

    @staticmethod
    def of(**kw) -> "Dim":
        return Dim(**{k: Fraction(v).limit_denominator(10**6)
                      for k, v in kw.items()})

    def __mul__(self, other: "Dim") -> "Dim":
        return Dim(*[getattr(self, d) + getattr(other, d) for d in BASE_DIMS])

    def __truediv__(self, other: "Dim") -> "Dim":
        return Dim(*[getattr(self, d) - getattr(other, d) for d in BASE_DIMS])

    def __pow__(self, p) -> "Dim":
        f = Fraction(p).limit_denominator(10**6)
        return Dim(*[getattr(self, d) * f for d in BASE_DIMS])

    def is_dimensionless(self) -> bool:
        return all(getattr(self, d) == 0 for d in BASE_DIMS)

    def as_dict(self) -> Dict[str, str]:
        return {d: str(getattr(self, d)) for d in BASE_DIMS}

    def __str__(self) -> str:
        if self.is_dimensionless():
            return "1"
        parts = []
        for d in BASE_DIMS:
            e = getattr(self, d)
            if e == 0:
                continue
            parts.append(d if e == 1 else f"{d}^{e}")
        return " ".join(parts)


DIMLESS = Dim()
DIM_M = Dim.of(M=1)
DIM_L = Dim.of(L=1)
DIM_T = Dim.of(T=1)
DIM_V = Dim.of(L=1, T=-1)
DIM_ACC = Dim.of(L=1, T=-2)
DIM_RHO = Dim.of(M=1, L=-3)
DIM_SIGMA = Dim.of(M=1, L=-2)              # surface density
DIM_PHI = Dim.of(L=2, T=-2)                # specific potential
DIM_TIDAL = Dim.of(T=-2)                   # d(acc)/dx, tidal tensor
DIM_PRESSURE = Dim.of(M=1, L=-1, T=-2)
DIM_TEMP = Dim.of(Theta=1)
DIM_ENERGY = Dim.of(M=1, L=2, T=-2)
DIM_ANGLE = DIMLESS                         # radians
DIM_FLUX = Dim.of(M=1, T=-3)               # energy / area / time
DIM_ANGMOM_SPEC = Dim.of(L=2, T=-1)

#: Newton's constant.  Kept here so a candidate law's dimensional check can
#: actually close: [G] = L^3 M^-1 T^-2.
DIM_G = Dim.of(M=-1, L=3, T=-2)
DIM_C = DIM_V


# ============================================================== vocabularies

KINDS = ("scalar", "vector", "tensor2", "tensor4", "pseudoscalar",
         "pseudovector", "graph", "path", "history", "distribution",
         "categorical")

#: How the quantity behaves under a TRANSLATION of the coordinate origin.
TRANSLATION = (
    "INVARIANT",              # unchanged.  The only safe default.
    "SHIFTS_BY_CONSTANT",     # e.g. absolute potential -- REQUIRES a gauge rule
    "COVARIANT",              # a position; transforms as a point
)

ROTATION = ("SCALAR", "VECTOR", "RANK2", "RANK4", "PSEUDOSCALAR",
            "PSEUDOVECTOR", "FRAME_DEPENDENT")

BOOST = (
    "INVARIANT",              # unchanged by a change of inertial frame
    "COVARIANT",              # transforms by a stated rule
    "FRAME_FIXED",            # only defined in one named frame -- a red flag
)

PARITY = ("EVEN", "ODD", "UNDEFINED")
TIME_REVERSAL = ("EVEN", "ODD", "DISSIPATIVE", "UNDEFINED")

SUPPORT = ("point", "region", "path", "interval", "pair", "graph", "global")

STATUS = ("direct", "derived", "invariant_descriptor", "latent", "nuisance",
          "constant")

#: Behaviour when the SAME physical system is re-represented -- one catalogue
#: row, ten deblended subcomponents, or a continuous density field.  This is
#: GATE 3 of the existing compiler, and the charter's rule that a valid network
#: law "cannot depend on how a cataloging algorithm happened to deblend the
#: image."
COARSE_GRAIN = (
    "EXTENSIVE",              # sums exactly (mass, flux, count)
    "INTENSIVE_LINEAR",       # a linear functional of density; averages exactly
    "NONLINEAR",              # does NOT commute with averaging; must be
                              # recomputed on the resolved scene
    "CATALOGUE_DEPENDENT",    # value depends on the row list itself -- unsafe
    "SCALE_DEFINED",          # only meaningful with its smoothing scale stated
    "TOPOLOGICAL",            # graph-valued; changes discretely under merges
)

#: Is the quantity available on the past light cone of the event being
#: predicted?  A law that reads a quantity marked NOT_CAUSAL is acausal.
CAUSAL = ("LOCAL_NOW", "PAST_LIGHT_CONE", "RETARDED", "SIMULTANEOUS_NONLOCAL",
          "FUTURE", "UNDEFINED")

#: Algebraic operations the metadata permits.  Used to reject e.g. taking the
#: logarithm of a dimensionful quantity, or adding two different dimensions.
OPS = ("add_same_dim", "multiply", "divide", "power_rational",
       "log_dimensionless", "exp_dimensionless", "gradient", "contract",
       "trace", "eigen", "project_on_axis", "path_integrate", "graph_reduce",
       "time_derivative", "smooth", "compare_to_scale")


IDENTIFIABILITY = ("measured", "constructible", "marginalisable",
                   "non_identifiable")


class ContractError(ValueError):
    """The metadata contract was violated.  Raised at construction time so a
    malformed quantity can never reach the compiler."""


# ================================================================== Quantity

@dataclass
class Quantity:
    """One entry in the parameter ontology, carrying the full contract.

    Constructing a Quantity that violates the contract raises ContractError.
    That is deliberate: the contract is only worth anything if it is enforced
    at the boundary rather than checked in a report afterwards.
    """
    # 1 -- name and operational definition
    name: str
    definition: str
    # 2 -- type
    kind: str
    # 3 -- units
    dim: Dim
    # 4 -- frame
    frame: str = "cluster_rest"
    # 5 -- transformation behaviour
    translation: str = "INVARIANT"
    rotation: str = "SCALAR"
    boost: str = "INVARIANT"
    parity: str = "EVEN"
    time_reversal: str = "EVEN"
    # 6 -- support
    support: str = "point"
    # 7 -- measurement or derivation source
    source: str = "unspecified"
    # 8 -- status
    status: str = "derived"
    # 9 -- resolution and smoothing scale, in metres.  None = not applicable.
    resolution_m: Optional[float] = None
    smoothing_m: Optional[float] = None
    # 10 -- uncertainty and covariance group
    uncertainty: str = "unquantified"
    covariance_group: str = "independent"
    # 11 -- boundary or gauge convention
    gauge: Optional[str] = None
    # 12 -- coarse-graining behaviour
    coarse_grain: str = "NONLINEAR"
    # 13 -- causal availability
    causal: str = "LOCAL_NOW"
    # 14 -- completeness and selection
    completeness: str = "unspecified"
    selection: str = "unspecified"
    # 15 -- allowed operations
    allowed_ops: Tuple[str, ...] = ("multiply", "divide", "power_rational")
    # 16 -- known algebraic dependencies, as symbolic relations between names
    depends_on: Tuple[str, ...] = ()
    exact_identities: Tuple[str, ...] = ()
    # 17 -- independent measurability in the test sample
    independently_measurable: bool = False
    measurability_note: str = ""
    #: A THREE-WAY refinement of item 17, added after the first prescreen run
    #: flagged every candidate -- including Newtonian gravity -- as
    #: non-identifiable, because g_N and r_3d are both constructed rather than
    #: directly observed.  A boolean cannot tell those cases apart:
    #:   measured        directly observed
    #:   constructible   determined by the resolved scene through a declared
    #:                   procedure; carries no free parameter of its own
    #:   marginalisable  NOT determined, but the scene ensemble integrates over
    #:                   it against a declared prior.  This is what Stage 1 is
    #:                   FOR, and line-of-sight depth is the type case.
    #:   non_identifiable  neither: a free latent field with no observational
    #:                   handle, admissible only through a generative law.
    identifiability: str = "measured"
    #: Is this quantity produced by INVERTING an assumed gravity law?  A
    #: convergence map, an NFW mass and an integrated Y inside an
    #: NFW-defined R500 all are.  Scoring a candidate law against one is
    #: circular, and the charter forbids it explicitly:
    #:   "Do not score ... a precomputed convergence map ... as though it
    #:    were the primitive observation."
    #: Distinct from `status="derived"`, which merely means "not measured
    #: directly": a deprojected gas density is derived and NOT contaminated.
    derived_under_theory: bool = False
    # bookkeeping
    rank: int = 0
    notes: str = ""

    # ------------------------------------------------------------- validation
    def __post_init__(self):
        if self.kind not in KINDS:
            raise ContractError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        if self.translation not in TRANSLATION:
            raise ContractError(f"{self.name}: translation {self.translation!r}")
        if self.rotation not in ROTATION:
            raise ContractError(f"{self.name}: rotation {self.rotation!r}")
        if self.boost not in BOOST:
            raise ContractError(f"{self.name}: boost {self.boost!r}")
        if self.parity not in PARITY:
            raise ContractError(f"{self.name}: parity {self.parity!r}")
        if self.time_reversal not in TIME_REVERSAL:
            raise ContractError(f"{self.name}: time_reversal {self.time_reversal!r}")
        if self.support not in SUPPORT:
            raise ContractError(f"{self.name}: support {self.support!r}")
        if self.status not in STATUS:
            raise ContractError(f"{self.name}: status {self.status!r}")
        if self.coarse_grain not in COARSE_GRAIN:
            raise ContractError(f"{self.name}: coarse_grain {self.coarse_grain!r}")
        if self.causal not in CAUSAL:
            raise ContractError(f"{self.name}: causal {self.causal!r}")
        for op in self.allowed_ops:
            if op not in OPS:
                raise ContractError(f"{self.name}: op {op!r} not in OPS")
        if self.identifiability not in IDENTIFIABILITY:
            raise ContractError(f"{self.name}: identifiability "
                                f"{self.identifiability!r} not in "
                                f"{IDENTIFIABILITY}")

        # --- THE GAUGE RULE.  The charter's explicit prohibition.
        if self.translation == "SHIFTS_BY_CONSTANT" and not self.gauge:
            raise ContractError(
                f"{self.name}: translation is SHIFTS_BY_CONSTANT so an "
                f"operational boundary rule is MANDATORY.  The charter: "
                f"'Absolute Newtonian potential must never be used without an "
                f"operational boundary rule.'")

        # --- THE LOG RULE.  log of a dimensionful quantity is not defined.
        if "log_dimensionless" in self.allowed_ops and not self.dim.is_dimensionless():
            raise ContractError(
                f"{self.name}: log_dimensionless allowed but dim={self.dim}. "
                f"A logarithm needs a declared reference scale first.")

        # --- THE SCALE RULE.  A SCALE_DEFINED quantity must state its scale.
        if self.coarse_grain == "SCALE_DEFINED" and self.smoothing_m is None:
            raise ContractError(
                f"{self.name}: coarse_grain SCALE_DEFINED requires smoothing_m")

        # --- rank consistency
        implied = {"scalar": 0, "pseudoscalar": 0, "vector": 1,
                   "pseudovector": 1, "tensor2": 2, "tensor4": 4}
        if self.kind in implied and self.rank == 0:
            object.__setattr__(self, "rank", implied[self.kind])
        if self.kind in implied and self.rank != implied[self.kind]:
            raise ContractError(f"{self.name}: rank {self.rank} != "
                                f"{implied[self.kind]} implied by kind")

    # ------------------------------------------------------------- interface
    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dim"] = self.dim.as_dict()
        d["dim_str"] = str(self.dim)
        d["allowed_ops"] = list(self.allowed_ops)
        d["depends_on"] = list(self.depends_on)
        d["exact_identities"] = list(self.exact_identities)
        return d

    def is_gauge_safe(self) -> bool:
        return self.translation != "SHIFTS_BY_CONSTANT" or bool(self.gauge)

    def commutes_with_averaging(self) -> bool:
        """Charter's root-data rule, at the level of one quantity."""
        return self.coarse_grain in ("EXTENSIVE", "INTENSIVE_LINEAR")


# ================================================================== registry

class Registry:
    """A named collection of Quantity objects with contract-level queries.

    The compiler consumes this.  Every query below answers one of the pre-data
    admissibility questions the charter's Stage 3 lists, using ONLY metadata --
    no data file is opened.
    """

    def __init__(self, quantities: Iterable[Quantity] = ()):
        self._q: Dict[str, Quantity] = {}
        for q in quantities:
            self.add(q)

    def add(self, q: Quantity) -> Quantity:
        if q.name in self._q:
            raise ContractError(f"duplicate quantity name {q.name!r}")
        self._q[q.name] = q
        return q

    def __len__(self):
        return len(self._q)

    def __contains__(self, name):
        return name in self._q

    def __getitem__(self, name) -> Quantity:
        if name not in self._q:
            raise KeyError(f"{name!r} is not in the registry "
                           f"({len(self._q)} quantities)")
        return self._q[name]

    def names(self) -> List[str]:
        return sorted(self._q)

    def all(self) -> List[Quantity]:
        return [self._q[n] for n in sorted(self._q)]

    def by_status(self, status: str) -> List[Quantity]:
        return [q for q in self.all() if q.status == status]

    # -------------------------------------------------- admissibility queries
    def gauge_unsafe(self) -> List[str]:
        return [q.name for q in self.all() if not q.is_gauge_safe()]

    def non_commuting(self) -> List[str]:
        """Quantities that must NOT be read off an averaged scene."""
        return [q.name for q in self.all() if not q.commutes_with_averaging()]

    def catalogue_dependent(self) -> List[str]:
        return [q.name for q in self.all()
                if q.coarse_grain == "CATALOGUE_DEPENDENT"]

    def not_independently_measurable(self) -> List[str]:
        return [q.name for q in self.all() if not q.independently_measurable]

    def by_identifiability(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {k: [] for k in IDENTIFIABILITY}
        for q in self.all():
            out[q.identifiability].append(q.name)
        return out

    def non_identifiable(self) -> List[str]:
        return [q.name for q in self.all()
                if q.identifiability == "non_identifiable"]

    def theory_contaminated(self) -> List[str]:
        return [q.name for q in self.all() if q.derived_under_theory]

    def acausal(self) -> List[str]:
        return [q.name for q in self.all() if q.causal in ("FUTURE",
                                                           "UNDEFINED")]

    def dimension_of_product(self, terms: Sequence[Tuple[str, Any]]) -> Dim:
        """dim of prod(name**power).  Raises KeyError on an unknown name."""
        d = DIMLESS
        for name, p in terms:
            d = d * (self[name].dim ** p)
        return d

    def identity_closure(self, names: Sequence[str]) -> Dict[str, List[str]]:
        """Which of `names` are exact algebraic functions of the others?

        The programme's `variable-lists-collapse` finding: five exact
        identities shrank rich-looking variable sets.  A candidate law reading
        both members of an identity pair has fewer independent directions than
        its term count suggests, and the search must know that BEFORE fitting.
        """
        s = set(names)
        red: Dict[str, List[str]] = {}
        for n in names:
            q = self[n]
            for ident in q.exact_identities:
                # an identity is stored as "n = f(a, b, ...)"; the RHS names
                # are the dependencies recorded in depends_on
                covered = [d for d in q.depends_on if d in s]
                if covered and len(covered) == len(q.depends_on):
                    red.setdefault(n, []).append(ident)
        return red

    def to_json(self) -> Dict[str, Any]:
        return {"n_quantities": len(self._q),
                "quantities": [q.to_json() for q in self.all()]}


# ==================================================== contract audit helpers

CONTRACT_FIELDS = (
    "name", "definition", "kind", "dim", "frame", "translation", "rotation",
    "boost", "parity", "time_reversal", "support", "source", "status",
    "resolution_m", "smoothing_m", "uncertainty", "covariance_group", "gauge",
    "coarse_grain", "causal", "completeness", "selection", "allowed_ops",
    "depends_on", "independently_measurable",
)

#: the seventeen charter items, mapped to the fields that carry them
CHARTER_ITEMS = {
    "physical name and operational definition": ("name", "definition"),
    "scalar/vector/tensor/graph/path/history type": ("kind", "rank"),
    "units": ("dim",),
    "coordinate and reference frame": ("frame",),
    "transformation under translation/rotation/boost/parity/time reversal":
        ("translation", "rotation", "boost", "parity", "time_reversal"),
    "point/region/path/interval it belongs to": ("support",),
    "measurement or derivation source": ("source",),
    "direct/derived/latent/nuisance status": ("status",),
    "resolution and smoothing scale": ("resolution_m", "smoothing_m"),
    "full uncertainty and covariance group": ("uncertainty",
                                              "covariance_group"),
    "boundary or gauge convention": ("gauge",),
    "coarse-graining behaviour": ("coarse_grain",),
    "causal availability": ("causal",),
    "completeness and selection function": ("completeness", "selection"),
    "allowed operations": ("allowed_ops",),
    "known algebraic dependencies": ("depends_on", "exact_identities"),
    "independently measurable in the test sample":
        ("independently_measurable", "measurability_note", "identifiability"),
}


#: Two contract items are CONDITIONALLY required rather than universally.
#: Recording that honestly is the point -- silently exempting them would make
#: the audit report a rubber stamp.
def _gauge_required(q: Quantity) -> bool:
    return q.translation == "SHIFTS_BY_CONSTANT"


def _scale_required(q: Quantity) -> bool:
    """A scale is required for anything defined by smoothing or by a finite
    aperture.  A point-supported directly-measured scalar has no scale."""
    return (q.coarse_grain == "SCALE_DEFINED"
            or q.support in ("region", "path")
            or q.kind in ("graph", "path"))


CONDITIONAL_ITEMS = {
    "boundary or gauge convention": _gauge_required,
    "resolution and smoothing scale": _scale_required,
}


def audit_contract(reg: Registry) -> Dict[str, Any]:
    """Is every charter contract item actually populated for every quantity?

    'unspecified' / 'unquantified' / None count as NOT populated.  This is the
    check that keeps the metadata from becoming decoration, which the task
    brief names explicitly.

    Two items are conditional (see CONDITIONAL_ITEMS): a gauge convention is
    required only of a quantity that shifts under a change of origin, and a
    resolution/smoothing scale only of a quantity whose value depends on one.
    Every OTHER item is required of every quantity without exception.
    """
    UNSET = {"unspecified", "unquantified", "", None}
    per_item: Dict[str, Dict[str, Any]] = {}
    for item, fields in CHARTER_ITEMS.items():
        cond = CONDITIONAL_ITEMS.get(item)
        missing, exempt = [], []
        for q in reg.all():
            if cond is not None and not cond(q):
                exempt.append(q.name)
                continue
            vals = [getattr(q, f, None) for f in fields]
            if all(v in UNSET for v in vals):
                missing.append(q.name)
        per_item[item] = {"fields": list(fields),
                          "conditional": cond is not None,
                          "n_applicable": len(reg) - len(exempt),
                          "n_exempt": len(exempt),
                          "n_missing": len(missing),
                          "missing": missing[:12],
                          "complete": not missing}
    return {"n_quantities": len(reg),
            "n_items": len(CHARTER_ITEMS),
            "items": per_item,
            "all_complete": all(v["complete"] for v in per_item.values())}
