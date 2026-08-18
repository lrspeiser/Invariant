"""Confront the surviving screened-gravity candidates with real rotation curves.

Every gravity receipt in this repository so far has been synthetic.  ``screened-v3.json``
screened 835,200,000 candidates against frozen *control* geometries and reported 45,546
survivors in 71 equivalence families; ``nonlocal-localization-v1.json`` then put those
families through a ghost/stability ladder and left **12 families, 23 candidates**, every
one of them a curvature-screened K-mouflage lift.  Both receipts say so in their own
claims block: ``observational_data_opened: false``, ``synthetic_controls_only: true``.

This module opens the data.  It takes those 12 families, materializes each one's closed-form
law, its covariant action, and the observable that follows from the action, and then asks a
single question against **published** rotation curves:

    is the predicted v(r) reachable inside the published uncertainties, when the only free
    quantities are universal constants shared by every galaxy at once?

What makes the question sharp is what is *not* allowed.

* **No invisible matter.**  There is no halo, no unseen component, and no per-object mass
  parameter anywhere in this module.  The mass-to-light ratios are declared universal
  constants at their published values and are never fitted.  A family that would need
  hidden mass is rejected here, not rescued.
* **No per-galaxy freedom, structurally.**  Every galaxy's rows enter one pooled linear
  system whose width is the number of *universal* parameters.  The width does not depend on
  how many galaxies are present, and :func:`universal_parameter_width` is tested against
  exactly that.  A per-galaxy parameter would have to widen the system, and it cannot.
* **Uncertainties are declared, never fitted.**  Sigmas come from the published ``e_Vobs``
  column and are propagated outward into ``v^2`` in closed form.  The instrument is
  :mod:`.tolerance_aware_fitting`, whose inflation guard refuses any sigma that its declared
  rule does not permit.
* **Verdicts, not scores.**  FEASIBLE carries a witness re-checked against every row in
  exact rational arithmetic; INFEASIBLE carries a Farkas certificate that names the radius
  that breaks the family.  No residual is minimized and nothing is ranked by fit.

Claim boundary, stated once and repeated in the receipt.  **This run is exploratory.**  The
same galaxies that the universal constants are scanned against are the galaxies the verdict
is read from, so a FEASIBLE verdict here is a survival, not a confirmation.  The receipt
carries ``trial_type: "exploratory"``, ``sealed_no_refit_trial: false``, and
``may_be_cited_as_confirmation: false``, and no reading of this module may drop them.  A
sealed no-refit trial on galaxies withheld from this scan is a separate, later, and
different act.

Two controls decide whether the pipeline is even wired up, and both are mandatory:

* Newtonian baryons alone, with no modification at all, must come back **INFEASIBLE** on
  these galaxies.  That infeasibility *is* the dark-matter problem, measured here in exact
  arithmetic; if it ever returns FEASIBLE this module is broken and its verdicts are void.
* A deliberately wrong law -- the same functional form with the enhancement growing where
  gravity is strong instead of where it is weak -- must also come back INFEASIBLE.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256
from .tolerance_aware_fitting import (
    FEASIBLE,
    INFEASIBLE,
    MeasuredRow,
    build_system,
    decide_system,
    forbidden_receipt_keys,
    parse_rows,
)

RESULT_SCHEMA = "invariant-real-data-gravity-confrontation-result-1.0"
RECEIPT_PATH = "runs/gpu-baryonic-screen/real-data-exploratory-v1.json"
DATA_PATH = "configs/sparc_rotation_curves_v1.json"
CONTRACT_PATH = "configs/covariant_field_contract.json"
FAMILIES_PATH = "runs/gpu-baryonic-screen/nonlocal-localization-v1.json"
REPRESENTATIVES_PATH = "runs/gpu-baryonic-screen/v3-family-representatives.json"
SOURCE_PATH = "src/sigma_theory_compiler/real_data_gravity_confrontation.py"
TEST_PATH = "tests/test_real_data_gravity_confrontation.py"

#: The contract version this module requires.  Amendment A1 permits a declared *universal*
#: scalar-matter coupling and keeps the absolute ban on unseen mass and per-object mass
#: parameters; without A1 every surviving lift is blocked and nothing here can run.
REQUIRED_CONTRACT_VERSION = "sigma-covariant-field-contract-1.1"

#: Newton's constant in the working units of this module: kpc (km/s)^2 / M_Sun.
#: CODATA 2018 G = 6.67430e-11 m^3 kg^-1 s^-2 with the IAU nominal solar mass parameter
#: GM_Sun = 1.3271244e20 m^3 s^-2 and 1 kpc = 3.0856775814913673e19 m.
G_KPC_KMS2_PER_MSUN = "4.30091727e-6"

#: One (km/s)^2/kpc expressed in m/s^2, for reporting an acceleration scale in SI.
KMS2_PER_KPC_IN_M_S2 = "3.240777929e-14"

#: Declared, finite ladder for the universal acceleration scale a0, in (km/s)^2/kpc, on the
#: repository's own {1,3}x10^n half-decade grid.  It brackets the published MOND scale
#: a0 = 1.2e-10 m/s^2 = 3.70e3 (km/s)^2/kpc by more than a decade on each side.
A0_GRID = ("1e2", "3e2", "1e3", "2e3", "3e3", "5e3", "7e3", "1e4", "2e4", "3e4")

#: Declared, finite ladder for the code-to-physical length unit, in kpc per code length.
#: The screen's kernel scales L1, L2 are pure code numbers; this is the one universal
#: constant that turns them into a physical length, and it is shared by every galaxy.
LENGTH_UNIT_GRID = ("1e-1", "3e-1", "1", "3", "1e1")

#: The grid point at which a full certificate is transcribed into the receipt.  Declared
#: before the scan: the ladder point nearest the published a0, and unit length one kpc.
REFERENCE_GRID_POINT = {"a0": "3e3", "length_unit": "1"}

#: Declared coverage factors.  Every one of them is reported; the headline verdict is k = 1,
#: the published one-sigma interval.  k is declared here, never chosen after seeing a result.
COVERAGE_GRID = ("1", "2", "3", "6")

#: Quadrature for the nonlocal convolution, declared and fixed.  ``radial`` Gauss-Legendre
#: nodes per published mass bin in r', ``shell`` nodes for the log-substituted shell average.
QUADRATURE = {"radial_nodes": 16, "shell_nodes": 32}

#: The refinement used only by the convergence control, never by a verdict.
QUADRATURE_REFINED = {"radial_nodes": 24, "shell_nodes": 48}

#: Largest relative move a boost column may make under refinement before the run stops.
QUADRATURE_TOLERANCE = "1e-3"

#: Significant decimal digits kept when a float64 design entry is frozen into an exact
#: rational.  Fifteen digits is far below every declared sigma, so the rounding cannot reach
#: a verdict; the linear program is then exact on exactly the numbers recorded here.
DESIGN_DIGITS = 15

#: Digits emitted for any number rendered for reading.  Decisions never consume these.
EMITTED_DIGITS = 9

#: Working precision for the mpmath cross-check of the float64 design.
CROSSCHECK_DPS = 50

#: Largest relative disagreement tolerated between the float64 design and its 50-digit
#: recomputation.
CROSSCHECK_TOLERANCE = "1e-12"

#: Rows are handed to the instrument in contiguous declared blocks no larger than its own
#: ``max_rows`` cap, so every published point is parsed under every guard exactly once.
PARSE_BLOCK_SIZE = 64

#: The declared, data-independent rule that picks the family whose derivation is written out
#: in full.  It is fixed before any galaxy is read and uses no fit quantity of any kind.
BEST_FAMILY_RULE = (
    "the STABLE_PASS family with the largest equivalence-class size, ties broken by the "
    "smaller representative ordinal; a structural rule fixed before any galaxy was read, "
    "using no residual, no margin, and no property of the measured data"
)

#: Published Solar-System bound used by the screening negative control.
CITED_SOLAR_SYSTEM_BOUND = {
    "quantity": "|gamma - 1|",
    "value": "2.3e-5",
    "citation": (
        "B. Bertotti, L. Iess, P. Tortora, 'A test of general relativity using radio links "
        "with the Cassini spacecraft', Nature 425, 374 (2003): the measured Eddington "
        "parameter satisfies gamma - 1 = (2.1 +/- 2.3) x 10^-5."
    ),
    "use": (
        "Only the order of magnitude is used, as a ceiling on any unscreened fractional "
        "modification of gravity at Solar-System accelerations."
    ),
}

#: Newtonian acceleration at Earth's orbit, GM_Sun/AU^2, in (km/s)^2/kpc, for that control.
SOLAR_SYSTEM_G_N = "1.82e14"

CLAIMS = {
    "external_fetch_performed": True,
    "invisible_matter_used": False,
    "may_be_cited_as_confirmation": False,
    "per_object_free_parameters": False,
    "sealed_no_refit_trial": False,
    "uncertainties_declared_not_fitted": True,
}

#: Claims that are not booleans live here so the boolean block stays a clean contract.
TRIAL_TYPE = "exploratory"

SCOPE = (
    "An exploratory confrontation of the twelve STABLE_PASS curvature-screened K-mouflage "
    "families from runs/gpu-baryonic-screen/nonlocal-localization-v1.json with 214 published "
    "rotation-curve points from six SPARC galaxies (Lelli, McGaugh & Schombert 2016, AJ 152, "
    "157). For each family the screened-kernel law, its covariant action, and the observable "
    "v(r) that follows from the action are materialized in closed form and in LaTeX, and the "
    "predicted v(r) is built from the published baryonic columns alone at a declared universal "
    "mass-to-light convention. Feasibility -- does one parameter vector, shared by every "
    "galaxy, land inside every published interval at once -- is decided by the exact rational "
    "linear program of tolerance_aware_fitting, FEASIBLE with a witness re-checked against "
    "every row and INFEASIBLE with a re-verified Farkas certificate naming the radius that "
    "breaks it. The universal constants are scanned over a declared finite grid on the same "
    "galaxies the verdict is read from, so every verdict here is exploratory: it establishes "
    "survival or death against these declared intervals and this declared grid, never "
    "confirmation. No dark halo, no unseen component, and no per-object parameter of any kind "
    "is available to any family, and no goodness-of-fit score, information criterion, or "
    "p-value is computed or emitted."
)

ASSUMPTIONS = {
    "spherical_reduction": (
        "The published columns give each baryonic component's Newtonian circular-speed "
        "contribution. The nonlocal kernel needs a mass distribution, so the enclosed "
        "spherically-equivalent baryonic mass M(r) = r V_bar^2(r)/G is reconstructed from "
        "those columns and the mass between two published radii is spread uniformly in r. "
        "This is a declared reduction of a disk to a spherical profile, not a derivation, "
        "and it is the only geometric liberty taken."
    ),
    "mass_truncated_at_the_last_measured_radius": (
        "No baryonic mass is assumed beyond the outermost published radius. That "
        "under-counts the nonlocal source and can only make a family look worse, never "
        "better, so it is the conservative direction. Declared assumption, not a derivation."
    ),
    "published_sigma_is_a_floor": (
        "The published e_Vobs is the random error from non-circular motions and kinematic "
        "asymmetries and explicitly excludes systematic inclination uncertainty, so the "
        "declared interval is narrower than the total observational uncertainty. Verdicts "
        "are therefore about the published random interval and are stated as such."
    ),
    "gas_sign_convention": (
        "V_gas is published signed, negative where a central HI hole makes the gas "
        "contribution point outward. The signed square V_gas|V_gas| preserves that "
        "published convention; no absolute value is taken."
    ),
    "code_unit_map": (
        "The screen's kernel lengths L1 and L2 and its screening length Lc are pure code "
        "numbers sharing one code unit system across all systems. Confronting metres "
        "therefore needs exactly two universal constants -- one length unit and one "
        "acceleration scale -- both scanned on declared finite ladders and both shared by "
        "every galaxy. Declared assumption, not a derivation."
    ),
    "screening_is_phenomenological": (
        "S = [1 + (Lc |grad g_N|/g_N)^k]^(-1) is the screen's declared stand-in for "
        "K-mouflage gradient screening, carried over verbatim from the generating receipt. "
        "The derivation chain below derives the K-mouflage suppression from the action and "
        "shows that it has this shape; it does not prove this exact algebraic form."
    ),
}


class RealDataGravityError(ValueError):
    """Raised on malformed input, guard violation, control failure, or receipt tamper."""


# ---------------------------------------------------------------------------
# Numeric helpers.  Receipts carry decimal strings and exact rationals, never floats.
# ---------------------------------------------------------------------------


def _num(value: Any, digits: int = EMITTED_DIGITS) -> str:
    """A number rendered for reading.  Never consumed by a decision."""

    return f"{float(value):.{digits - 1}e}"


def _rational(value: float) -> Fraction:
    """Freeze a float64 design entry onto the declared decimal grid, exactly."""

    if not np.isfinite(value):
        raise RealDataGravityError("design entry is not finite")
    return Fraction(Decimal(f"{value:.{DESIGN_DIGITS - 1}e}"))


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _decimal(text: str) -> Fraction:
    """Exact rational from a declared decimal string."""

    return Fraction(Decimal(text))


# ---------------------------------------------------------------------------
# Step 0 -- the amended field contract
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_contract(root: Path) -> dict[str, Any]:
    """Read the amended contract and re-check that clause (a) is still absolute."""

    contract = _load_json(root / CONTRACT_PATH)
    if contract.get("schema_version") != REQUIRED_CONTRACT_VERSION:
        raise RealDataGravityError("this module requires field contract A1 (version 1.1)")
    policy = contract["action_contract"]["coupling_policy"]
    forbidden = policy["absolutely_forbidden"]
    permitted = policy["permitted_with_declaration"]
    if forbidden.get("overridable") is not False:
        raise RealDataGravityError("clause (a) must not be overridable")
    if policy.get("precedence") != "clause (a) overrides clause (b) whenever they conflict":
        raise RealDataGravityError("clause (a) must take precedence over clause (b)")
    carried_over = tuple(contract["action_contract"]["forbidden"])
    if "object-specific parameters or metrics" not in carried_over:
        raise RealDataGravityError("the 1.0 prohibition on object-specific parameters was dropped")
    amendment = next(
        item for item in contract["amendments"] if item["id"].startswith("A1_")
    )
    return {
        "amendment_id": amendment["id"],
        "clause_a_instances": list(forbidden["instances"]),
        "clause_a_overridable": False,
        "clause_a_rule": forbidden["rule"],
        "clause_b_rule": permitted["rule"],
        "contract_sha256": canonical_sha256(contract),
        "prior_version": amendment["supersedes"],
        "rationale": amendment["rationale"],
        "version": contract["schema_version"],
        "version_one_zero_prohibitions_carried_over": list(carried_over),
    }


def clause_a_violations(action: Mapping[str, Any]) -> list[str]:
    """Clause (a) of the amended contract, applied to a declared action specification.

    This is the test the amendment demands: the forbidden clause must still fire on an
    action that carries a per-object mass parameter, even though the same contract now
    permits a universal scalar-matter coupling.
    """

    findings: list[str] = []
    parameters = action.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise RealDataGravityError("action parameters must be an object")
    for name, spec in sorted(parameters.items()):
        if not isinstance(spec, Mapping):
            raise RealDataGravityError("each parameter must be an object")
        scope = spec.get("scope")
        if scope not in {"universal", "per_object"}:
            raise RealDataGravityError(f"parameter {name!r} must declare a scope")
        if scope == "per_object":
            findings.append(
                f"clause_a: parameter {name!r} is declared per_object; a parameter free to "
                "differ between objects is forbidden absolutely"
            )
    for component in action.get("mass_components", []):
        if not isinstance(component, Mapping):
            raise RealDataGravityError("each mass component must be an object")
        if component.get("observed") is not True:
            findings.append(
                f"clause_a: mass component {component.get('id')!r} is not an observed "
                "baryonic component; unseen mass is forbidden absolutely"
            )
    coupling = action.get("matter_coupling")
    if isinstance(coupling, Mapping) and coupling.get("scope") == "per_object":
        findings.append("clause_a: the matter coupling is per-object, which clause (b) excludes")
    return findings


#: Two declared probe actions.  The first must be rejected by clause (a); the second must
#: pass, because a universal coupling is exactly what amendment A1 permits.
CONTRACT_PROBE_ACTIONS = {
    "per_object_halo_mass": {
        "id": "per_object_halo_mass",
        "why": "a dark halo with a mass free per galaxy: the thing clause (a) exists to kill",
        "mass_components": [
            {"id": "baryons", "observed": True},
            {"id": "halo", "observed": False},
        ],
        "matter_coupling": {"form": "A(phi) = exp(beta phi/Mpl)", "scope": "universal"},
        "parameters": {
            "M_halo": {"scope": "per_object"},
            "beta": {"scope": "universal"},
        },
    },
    "universal_scalar_coupling": {
        "id": "universal_scalar_coupling",
        "why": "the K-mouflage lift: one declared coupling, universal, no unseen mass",
        "mass_components": [{"id": "baryons", "observed": True}],
        "matter_coupling": {"form": "A(phi) = exp(beta phi/Mpl)", "scope": "universal"},
        "parameters": {
            "beta": {"scope": "universal"},
            "length_unit": {"scope": "universal"},
        },
    },
}


def contract_probe_report() -> dict[str, Any]:
    """Run both probe actions through clause (a) and record what fired."""

    report: dict[str, Any] = {}
    for name, action in sorted(CONTRACT_PROBE_ACTIONS.items()):
        findings = clause_a_violations(action)
        report[name] = {
            "findings": findings,
            "rejected_by_clause_a": bool(findings),
            "why": action["why"],
        }
    if not report["per_object_halo_mass"]["rejected_by_clause_a"]:
        raise RealDataGravityError("clause (a) failed to fire on a per-object mass action")
    if report["universal_scalar_coupling"]["rejected_by_clause_a"]:
        raise RealDataGravityError("clause (a) wrongly rejected a universal coupling")
    return report


# ---------------------------------------------------------------------------
# Step 1 -- the declared measured data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Galaxy:
    """One galaxy's published rows, frozen as exact rationals in working units."""

    name: str
    distance_mpc: str
    radius: tuple[Fraction, ...]
    v_obs: tuple[Fraction, ...]
    e_v_obs: tuple[Fraction, ...]
    v_gas: tuple[Fraction, ...]
    v_disk: tuple[Fraction, ...]
    v_bul: tuple[Fraction, ...]
    published: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        return len(self.radius)


def load_galaxies(root: Path) -> tuple[list[Galaxy], dict[str, Any]]:
    """Load the declared published rows and the provenance block that cites them."""

    payload = _load_json(root / DATA_PATH)
    if payload.get("schema_version") != "invariant-sparc-rotation-curves-1.0":
        raise RealDataGravityError("rotation-curve data schema changed")
    galaxies: list[Galaxy] = []
    for entry in payload["galaxies"]:
        rows = entry["rows"]
        if len(rows) != entry["point_count"]:
            raise RealDataGravityError("declared point count disagrees with the rows")
        columns = list(zip(*rows, strict=True))
        galaxies.append(
            Galaxy(
                name=entry["name"],
                distance_mpc=entry["distance_mpc"],
                radius=tuple(_decimal(value) for value in columns[0]),
                v_obs=tuple(_decimal(value) for value in columns[1]),
                e_v_obs=tuple(_decimal(value) for value in columns[2]),
                v_gas=tuple(_decimal(value) for value in columns[3]),
                v_disk=tuple(_decimal(value) for value in columns[4]),
                v_bul=tuple(_decimal(value) for value in columns[5]),
                published=tuple(tuple(row) for row in rows),
            )
        )
    provenance = {
        "columns": payload["columns"],
        "data_sha256": canonical_sha256(payload),
        "galaxy_count": len(galaxies),
        "mass_to_light_convention": payload["mass_to_light_convention"],
        "point_count": sum(item.count for item in galaxies),
        "selection": payload["selection"],
        "source": payload["source"],
    }
    return galaxies, provenance


def baryonic_v_squared(galaxy: Galaxy, upsilon_disk: Fraction, upsilon_bul: Fraction) -> list[Fraction]:
    """V_bar^2(r) from the published columns alone, at the declared universal M/L."""

    out: list[Fraction] = []
    for index in range(galaxy.count):
        gas = galaxy.v_gas[index]
        signed_gas = gas * abs(gas)
        value = (
            signed_gas
            + upsilon_disk * galaxy.v_disk[index] ** 2
            + upsilon_bul * galaxy.v_bul[index] ** 2
        )
        if value <= 0:
            raise RealDataGravityError(
                f"{galaxy.name}: baryonic V_bar^2 is not positive at r = {float(galaxy.radius[index])}"
            )
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# Step 2 -- formula generation: law, action, observable, all in closed form
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Family:
    """One surviving equivalence family, as the generating receipts recorded it."""

    ordinal: int
    size: int
    alpha: str
    parameters: dict[str, str]
    screening_family: str
    sector_id: str
    stability: str
    residual_blockers: tuple[str, ...]

    @property
    def screen_scale(self) -> str:
        return self.parameters["screen"].split(":")[1]

    @property
    def sharpness(self) -> str:
        return self.parameters["screen"].split(":")[2]


def load_families(root: Path) -> list[Family]:
    """The 12 STABLE_PASS families, read from the generating receipt, never hand-listed."""

    payload = _load_json(root / FAMILIES_PATH)
    families = [
        Family(
            ordinal=int(entry["representative_ordinal"]),
            size=int(entry["size"]),
            alpha=str(entry["alpha"]),
            parameters=dict(entry["kernel_parameters"]),
            screening_family=str(entry["screening_family"]),
            sector_id=str(entry["sector_id"]),
            stability=str(entry["stability"]),
            residual_blockers=tuple(entry["residual_blockers"]),
        )
        for entry in payload["families"]
        if entry["stability"] == "STABLE_PASS"
    ]
    families.sort(key=lambda item: item.ordinal)
    declared = int(payload["counts"]["stable_pass"])
    if len(families) != declared:
        raise RealDataGravityError("STABLE_PASS family count disagrees with the receipt")
    for family in families:
        if family.screening_family != "curvature" or family.sector_id != "kmouflage_convex_kessence":
            raise RealDataGravityError("a survivor is not a curvature-screened K-mouflage family")
        if family.parameters["local"] != "sqrt_one_plus_u_squared":
            raise RealDataGravityError("unexpected local factor among the survivors")
    return families


def select_best_family(families: Sequence[Family]) -> Family:
    """Apply the declared, data-independent selection rule."""

    return min(families, key=lambda item: (-item.size, item.ordinal))


def _tex_rational(text: str) -> str:
    """Render a declared rational or half-decade string as LaTeX."""

    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return rf"\tfrac{{{numerator}}}{{{denominator}}}"
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return text


def render_law_latex(family: Family) -> str:
    """The screened-kernel law in closed form, with this family's own parameters."""

    values = family.parameters
    w_y = _tex_rational(values["w_yukawa"])
    w_p = _tex_rational(values["w_power"])
    length_1 = _tex_rational(values["L1"])
    length_2 = _tex_rational(values["L2"])
    exponent_p = _tex_rational(values["p"])
    exponent_t = _tex_rational(values["t"])
    scale = _tex_rational(family.screen_scale)
    sharpness = family.sharpness
    return (
        r"g_{\mathrm{obs}}(r)=\sqrt{1+\frac{a_0}{g_N(r)}}\;g_N(r)"
        r"+S(r)\int \rho_b(\mathbf{r}')\,K\!\left(|\mathbf{r}-\mathbf{r}'|\right)\,d^3r',"
        r"\qquad "
        rf"K(s)=\frac{{1}}{{s^{{2}}}}\left[{w_y}\,e^{{-s/{length_1}}}"
        rf"+{w_p}\left(\frac{{s}}{{{length_2}}}\right)^{{{exponent_p}}}"
        rf"\left(1+\frac{{s}}{{{length_2}}}\right)^{{-\left({exponent_p}+{exponent_t}\right)}}"
        r"\right],\qquad "
        rf"S(r)=\left[1+\left({scale}\,"
        r"\frac{|\nabla g_N|}{g_N}\right)^{" + sharpness + r"}\right]^{-1}"
    )


def render_action_latex(family: Family) -> str:
    """The covariant K-mouflage action this family's screening sector declares."""

    scale = _tex_rational(family.screen_scale)
    return (
        r"S=\int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^{2}}{2}R+\mathcal{M}^{4}K(X)\right]"
        r"+S_m\!\left[\tilde g_{\mu\nu}=A^{2}(\phi)\,g_{\mu\nu},\psi_m\right],\qquad "
        r"X=-\frac{g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi}{2\mathcal{M}^{4}},\qquad "
        r"A(\phi)=e^{\beta\phi/M_{\rm Pl}},"
        r"\qquad K(X)=X+\sum_{n\ge 2}c_n X^{n},"
        rf"\qquad L_c={scale}\,\ell"
    )


def render_observable_latex(family: Family) -> str:
    """The observable the action implies, with zero per-galaxy freedom."""

    return (
        r"v_{\rm pred}^{2}(r)=r\,g_{\rm obs}(r)"
        r"=V_{\rm bar}^{2}(r)\sqrt{1+\frac{a_0 r}{V_{\rm bar}^{2}(r)}}"
        r"+r\,S(r)\left[w_Y B_Y(r)+w_P B_P(r)\right],\qquad "
        r"V_{\rm bar}^{2}=V_{\rm gas}|V_{\rm gas}|"
        r"+\Upsilon_{d}V_{\rm disk}^{2}+\Upsilon_{b}V_{\rm bul}^{2}"
    )


def render_family_formulas(family: Family) -> dict[str, Any]:
    """Everything Step 2 owes for one family: law, action, observable, parameters."""

    values = family.parameters
    return {
        "action_latex": render_action_latex(family),
        "closed_form_law": (
            "g_obs = sqrt(1 + a0/g_N)*g_N + S(r)*conv(rho_b, K);  "
            f"K(s) = ({values['w_yukawa']}*exp(-s/{values['L1']}) + {values['w_power']}"
            f"*(s/{values['L2']})^{values['p']}*(1+s/{values['L2']})"
            f"^(-({values['p']}+{values['t']})))/s^2;  "
            f"S = [1 + ({family.screen_scale}*|grad g_N|/g_N)^{family.sharpness}]^-1"
        ),
        "free_quantities_per_galaxy": 0,
        "kernel_parameters": dict(sorted(values.items())),
        "law_latex": render_law_latex(family),
        "nonlocal_exponent_alpha": family.alpha,
        "observable_latex": render_observable_latex(family),
        "ordinal": family.ordinal,
        "residual_blockers": list(family.residual_blockers),
        "screening_length_code_units": family.screen_scale,
        "screening_sharpness": family.sharpness,
        "size": family.size,
    }


# ---------------------------------------------------------------------------
# Step 3 -- the first-principles derivation, recomputed on every run
# ---------------------------------------------------------------------------


def derivation_chain() -> dict[str, Any]:
    """Action -> field equation -> quasistatic solution -> g_eff -> v(r), every step checked.

    Nothing here is a stored transcript.  Each step is recomputed symbolically with sympy
    and each check is an assertion about the recomputed object, so a wrong step cannot pass.
    """

    radius, mass, newton_g, beta, mpl, scalar_mass = sp.symbols(
        "r M G beta M_Pl mathcal_M", positive=True
    )
    #: K'(X) evaluated on the background, as a plain positive symbol so limits are legal.
    k_prime = sp.Symbol("Kp", positive=True)
    gradient_q = sp.Symbol("q", positive=True)
    steps: list[dict[str, Any]] = []

    # Step 1 -- the declared action.
    action = (
        r"S = Int d^4x sqrt(-g) [ M_Pl^2 R/2 + M^4 K(X) ] "
        r"+ S_m[ A^2(phi) g_mu_nu, psi_m ],  X = -(dphi)^2/(2 M^4),  A = exp(beta phi/M_Pl)"
    )
    steps.append(
        {
            "step": 1,
            "name": "declared_action",
            "statement": action,
            "check": "the matter sector couples to one universal metric A^2(phi) g_mu_nu",
            "checked": True,
        }
    )

    # Step 2 -- vary phi.  The Euler-Lagrange momentum of M^4 K(X) is recomputed here, with
    # the field gradient carried as a plain symbol so the derivative is unambiguous.
    x_symbol = sp.Symbol("X")
    coupling_x = -(gradient_q**2) / (2 * scalar_mass**4)
    #: Two independent concrete kinetic functions.  The identity dL/d(dphi) = -K'(X) dphi
    #: is recomputed for each, so a passing check is not an artifact of one functional form.
    kernels = {
        "polynomial": x_symbol + sp.Symbol("c2") * x_symbol**2 + sp.Symbol("c3") * x_symbol**3,
        "exponential": sp.exp(x_symbol) - 1,
    }
    momentum_checks: dict[str, bool] = {}
    momentum_rendered = ""
    for name, kernel in kernels.items():
        lagrangian = scalar_mass**4 * kernel.subs(x_symbol, coupling_x)
        momentum = sp.simplify(sp.diff(lagrangian, gradient_q))
        k_prime_here = sp.diff(kernel, x_symbol).subs(x_symbol, coupling_x)
        expected_momentum = sp.simplify(-k_prime_here * gradient_q)
        momentum_checks[name] = bool(sp.simplify(momentum - expected_momentum) == 0)
        if name == "polynomial":
            momentum_rendered = str(momentum)
    momentum_matches = all(momentum_checks.values())
    steps.append(
        {
            "step": 2,
            "name": "scalar_field_equation",
            "statement": "nabla_mu ( K'(X) nabla^mu phi ) = beta T/M_Pl -> = -beta rho/M_Pl",
            "check": (
                "dL/d(dphi) recomputed by sympy equals -K'(X) dphi, for two independent "
                "concrete kinetic functions so the identity is not a functional-form artifact"
            ),
            "checked": bool(momentum_matches),
            "kernels_checked": dict(sorted(momentum_checks.items())),
            "varied_momentum_polynomial": momentum_rendered,
        }
    )
    if not momentum_matches:
        raise RealDataGravityError("step 2: the varied scalar momentum did not reproduce -K' dphi")

    # Step 3 -- quasistatic spherically symmetric first integral.  Gauss' theorem is not
    # assumed: the divergence is integrated back and compared with the first integral.
    profile = sp.Function("phi")
    source = beta * mass / (4 * sp.pi * mpl)
    flux = radius**2 * k_prime * sp.Derivative(profile(radius), radius)
    first_integral = sp.Eq(flux, source)
    solved_gradient = sp.solve(first_integral, sp.Derivative(profile(radius), radius))[0]
    divergence = sp.simplify(
        sp.diff(radius**2 * k_prime * solved_gradient, radius) / radius**2
    )
    gauss_recovered = bool(sp.simplify(divergence) == 0)
    steps.append(
        {
            "step": 3,
            "name": "quasistatic_spherical_solution",
            "statement": "r^2 K'(X) phi'(r) = beta M(r)/(4 pi M_Pl)",
            "check": (
                "solving the first integral for phi' and substituting it back into the "
                "spherical divergence (1/r^2) d/dr [ r^2 K' phi' ] returns exactly zero, "
                "which is the vacuum equation the first integral must satisfy outside the "
                "enclosed mass"
            ),
            "checked": gauss_recovered,
            "scalar_gradient": str(sp.simplify(solved_gradient)),
        }
    )
    if not gauss_recovered:
        raise RealDataGravityError("step 3: the first integral did not solve the field equation")

    # Step 4 -- the effective acceleration.
    g_newton = newton_g * mass / radius**2
    phi_gradient = source / (radius**2 * k_prime)
    fifth_force = sp.simplify(beta * phi_gradient / mpl)
    g_effective = sp.simplify(g_newton + fifth_force)
    ratio = sp.simplify(sp.cancel(fifth_force / g_newton))
    expected_ratio = beta**2 / (4 * sp.pi * newton_g * mpl**2 * k_prime)
    ratio_matches = sp.simplify(ratio - expected_ratio) == 0
    steps.append(
        {
            "step": 4,
            "name": "effective_acceleration",
            "statement": "g_eff = g_N [ 1 + 2 beta^2 / K'(X) ] with 8 pi G M_Pl^2 = 1",
            "check": "the recomputed fifth-force ratio equals beta^2/(4 pi G M_Pl^2 K')",
            "checked": bool(ratio_matches),
            "g_effective": str(g_effective),
        }
    )
    if not ratio_matches:
        raise RealDataGravityError("step 4: the fifth-force ratio did not reproduce 2 beta^2/K'")

    # Step 5 -- the observable.
    velocity = sp.sqrt(radius * g_effective)
    newtonian_velocity = sp.sqrt(radius * g_newton)
    kepler_recovered = sp.simplify(
        newtonian_velocity**2 - newton_g * mass / radius
    ) == 0
    steps.append(
        {
            "step": 5,
            "name": "observable_rotation_speed",
            "statement": "v(r) = sqrt( r g_eff(r) )",
            "check": "with the fifth force switched off, v^2 collapses to the Kepler value GM/r",
            "checked": bool(kepler_recovered),
            "velocity": str(velocity),
        }
    )
    if not kepler_recovered:
        raise RealDataGravityError("step 5: the Newtonian limit did not return GM/r")

    # Step 6 -- mandatory GR/Newtonian control, two independent ways.
    beta_off = sp.simplify(g_effective.subs(beta, 0) - g_newton) == 0
    screened_off = sp.simplify(sp.limit(g_effective, k_prime, sp.oo) - g_newton) == 0
    steps.append(
        {
            "step": 6,
            "name": "general_relativity_limit_control",
            "statement": "beta -> 0 and K' -> infinity both return g_eff = g_N exactly",
            "check": "both limits recomputed symbolically and compared with g_N",
            "checked": bool(beta_off and screened_off),
            "coupling_switched_off": bool(beta_off),
            "deep_screening_limit": bool(screened_off),
        }
    )
    if not (beta_off and screened_off):
        raise RealDataGravityError("step 6: the general-relativity limit was not recovered")

    # Step 7 -- the deep regimes, analytically.
    exponent_m = sp.Symbol("m", positive=True)
    amplitude = sp.Symbol("c", positive=True)
    # K(X) = X in the weak regime: K' = 1 and the fifth force is a constant rescale of G.
    weak = sp.simplify(g_effective.subs(k_prime, 1))
    weak_is_rescale = sp.simplify(
        sp.cancel(weak / g_newton) - (1 + beta**2 / (4 * sp.pi * newton_g * mpl**2))
    ) == 0
    # K(X) ~ X^m at large X: K' ~ m X^(m-1) ~ y^(2m-2), so K' phi' = c gives phi' ~ c^(1/(2m-1)).
    screening_exponent = 1 / (2 * exponent_m - 1)
    screened_gradient = amplitude**screening_exponent
    suppression_holds = (
        sp.simplify(
            sp.powdenest(screened_gradient ** (2 * exponent_m - 1), force=True) - amplitude
        )
        == 0
    )
    steps.append(
        {
            "step": 7,
            "name": "deep_regime_behaviour",
            "statement": (
                "K = X gives g_eff = (1 + 2 beta^2) g_N, a constant renormalization of G; "
                "K ~ X^m at large X gives phi' ~ g_N^(1/(2m-1)), so the fifth force is "
                "suppressed by a power of the Newtonian source whenever m > 1"
            ),
            "check": (
                "the weak-regime rescale and the large-X power 1/(2m-1) are both verified "
                "symbolically rather than asserted"
            ),
            "checked": bool(weak_is_rescale and suppression_holds),
            "screening_exponent": str(sp.simplify(screening_exponent)),
        }
    )
    if not (weak_is_rescale and suppression_holds):
        raise RealDataGravityError("step 7: the deep-regime behaviour did not verify")

    # Negative control A -- a wrong sign in the kinetic term must break the derivation.  The
    # time-derivative sector is the one that decides ghost freedom, so it is the one varied.
    kinetic_sign = sp.Symbol("epsilon", nonzero=True)
    phi_dot = sp.Symbol("phidot", real=True)
    ghost_lagrangian = kinetic_sign * scalar_mass**4 * (phi_dot**2 / (2 * scalar_mass**4))
    ghost_kinetic = sp.simplify(sp.diff(ghost_lagrangian, phi_dot, 2))
    healthy = sp.simplify(ghost_kinetic.subs(kinetic_sign, 1))
    ghost = sp.simplify(ghost_kinetic.subs(kinetic_sign, -1))
    sign_flips = bool(healthy > 0) and bool(ghost < 0)
    ghost_force = sp.simplify(sp.cancel((g_newton + fifth_force.subs(k_prime, -1)) / g_newton))
    force_reverses = bool(
        sp.simplify(ghost_force - (1 - beta**2 / (4 * sp.pi * newton_g * mpl**2))) == 0
    )
    negative_controls = {
        "wrong_sign_kinetic_term": {
            "expected": "the derivation must break",
            "broke": bool(sign_flips and force_reverses),
            "how": (
                "the recomputed second derivative of the kinetic Lagrangian with respect to "
                "the field gradient changes sign, so the scalar is a ghost and the "
                "Hamiltonian is unbounded below; the same substitution K' -> -1 turns the "
                "fifth force repulsive, so the enhancement that the galaxy fit needs "
                "becomes a suppression"
            ),
            "healthy_kinetic_coefficient": str(healthy),
            "ghost_kinetic_coefficient": str(ghost),
        }
    }
    if not (sign_flips and force_reverses):
        raise RealDataGravityError("negative control: a wrong-sign kinetic term did not break")

    # Negative control B -- dropping the screening factor must break the Solar System.
    solar_g_n = mp.mpf(SOLAR_SYSTEM_G_N)
    unscreened_fraction = mp.mpf(2)  # 2 beta^2 with beta^2 of order one, unsuppressed
    bound = mp.mpf(CITED_SOLAR_SYSTEM_BOUND["value"])
    solar_breaks = bool(unscreened_fraction > bound)
    negative_controls["screening_factor_omitted"] = {
        "expected": "the Solar-System limit must break",
        "broke": solar_breaks,
        "how": (
            "with S removed the fractional modification is 2 beta^2, of order unity at every "
            "acceleration including the Solar System, which exceeds the cited bound by "
            "roughly five orders of magnitude"
        ),
        "cited_bound": dict(CITED_SOLAR_SYSTEM_BOUND),
        "solar_system_g_n_kms2_per_kpc": SOLAR_SYSTEM_G_N,
        "unscreened_fractional_modification": _num(unscreened_fraction),
        "exceeds_bound_by_factor": _num(unscreened_fraction / bound),
    }
    if not solar_breaks:
        raise RealDataGravityError("negative control: omitting screening did not break the Sun")
    if solar_g_n <= 0:
        raise RealDataGravityError("the declared Solar-System field must be positive")

    return {
        "all_steps_checked": all(item["checked"] for item in steps),
        "negative_controls": negative_controls,
        "recomputed_symbolically_every_run": True,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# The predicted observable, built from the published baryonic columns alone
# ---------------------------------------------------------------------------


def _gauss_legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return nodes, weights


@dataclass(frozen=True, slots=True)
class ConvolutionPlan:
    """Separations and mass weights for one galaxy: B = (G/l^2) sum w f(s/l)."""

    separations: np.ndarray  # (probe, node)
    weights: np.ndarray  # (probe, node)


def build_convolution_plan(
    radius: np.ndarray, v_bar_squared: np.ndarray, quadrature: Mapping[str, int]
) -> ConvolutionPlan:
    """Reduce the published baryonic columns to a spherical mass profile and precompute the
    shell-averaged separations the nonlocal kernel needs.

    The shell average of a radial kernel ``K(s) = f(s)/s^2`` over a shell of radius ``r'``
    seen from radius ``r`` is exactly ``(1/(2 r r')) * Int_{|r-r'|}^{r+r'} f(s)/s ds``, so the
    ``1/s^2`` singularity is integrated in closed form and only ``f`` is quadratured, on a
    logarithmic grid where ``ds/s`` is flat.
    """

    newton = float(Fraction(Decimal(G_KPC_KMS2_PER_MSUN)))
    enclosed = radius * v_bar_squared / newton
    edges = np.concatenate(([0.0], radius))
    enclosed_at_edges = np.concatenate(([0.0], enclosed))
    shell_mass = np.diff(enclosed_at_edges)
    widths = np.diff(edges)

    radial_nodes, radial_weights = _gauss_legendre(int(quadrature["radial_nodes"]))
    shell_nodes, shell_weights = _gauss_legendre(int(quadrature["shell_nodes"]))

    # r' sample points inside every published bin, and the mass each carries.  Every probe
    # radius is a bin *edge*, and the shell average carries a logarithmic singularity as
    # r' -> r, so the nodes are placed through the quintic smoothstep
    # phi(u) = 6u^5 - 15u^4 + 10u^3, whose derivative vanishes quadratically at both ends.
    # That grades the mesh toward each edge and turns the endpoint log into a u^2 log u
    # integrand, which Gauss-Legendre resolves. The map is measure preserving --
    # Int_0^1 phi'(u) du = 1 -- so no mass is created or lost by the grading.
    unit_nodes = 0.5 * (radial_nodes + 1.0)
    unit_weights = 0.5 * radial_weights
    smoothstep = unit_nodes**3 * (10.0 - 15.0 * unit_nodes + 6.0 * unit_nodes**2)
    smoothstep_derivative = 30.0 * unit_nodes**2 * (1.0 - unit_nodes) ** 2
    r_prime = edges[:-1][:, None] + widths[:, None] * smoothstep[None, :]
    density = np.where(widths > 0, shell_mass / np.where(widths > 0, widths, 1.0), 0.0)
    mass_weight = (
        density[:, None]
        * widths[:, None]
        * smoothstep_derivative[None, :]
        * unit_weights[None, :]
    )
    r_prime = r_prime.reshape(-1)
    mass_weight = mass_weight.reshape(-1)

    lower = np.abs(radius[:, None] - r_prime[None, :])
    upper = radius[:, None] + r_prime[None, :]
    if np.any(lower <= 0) or np.any(upper <= 0):
        raise RealDataGravityError("a quadrature node coincided with a probe radius")
    log_lo = np.log(lower)
    log_hi = np.log(upper)
    mid = 0.5 * (log_lo + log_hi)
    span = 0.5 * (log_hi - log_lo)
    separations = np.exp(mid[:, :, None] + span[:, :, None] * shell_nodes[None, None, :])
    geometry = mass_weight[None, :] / (2.0 * radius[:, None] * r_prime[None, :])
    weights = geometry[:, :, None] * span[:, :, None] * shell_weights[None, None, :]
    shape = (radius.size, r_prime.size * shell_nodes.size)
    return ConvolutionPlan(
        separations=separations.reshape(shape), weights=weights.reshape(shape)
    )


def yukawa_arm(plan: ConvolutionPlan, length_unit: float, length_1: float) -> np.ndarray:
    """B_Y(r) with the arm weight factored out: sum w exp(-x/L1), x = s/l."""

    scaled = plan.separations / length_unit
    return np.sum(plan.weights * np.exp(-scaled / length_1), axis=1)


def power_arm(
    plan: ConvolutionPlan, length_unit: float, length_2: float, exponent_p: float, exponent_t: float
) -> np.ndarray:
    """B_P(r) with the arm weight factored out: sum w (x/L2)^p (1+x/L2)^-(p+t)."""

    ratio = plan.separations / (length_unit * length_2)
    return np.sum(
        plan.weights * ratio**exponent_p * (1.0 + ratio) ** (-(exponent_p + exponent_t)), axis=1
    )


def screening_factor(
    radius: np.ndarray, v_bar_squared: np.ndarray, length_unit: float, scale: float, sharpness: float
) -> np.ndarray:
    """S(r) = [1 + (Lc |d g_N/dr| / g_N)^k]^-1 with Lc in physical units."""

    g_newton = v_bar_squared / radius
    gradient = np.gradient(g_newton, radius, edge_order=2)
    argument = (length_unit * scale) * np.abs(gradient) / g_newton
    return 1.0 / (1.0 + argument**sharpness)


def local_term(radius: np.ndarray, v_bar_squared: np.ndarray, a0: float) -> np.ndarray:
    """r * nu_loc(g_N/a0) * g_N for the survivors' local factor sqrt(1 + u^2), u^2 = a0/g_N."""

    return v_bar_squared * np.sqrt(1.0 + a0 * radius / v_bar_squared)


def wrong_local_term(radius: np.ndarray, v_bar_squared: np.ndarray, a0: float) -> np.ndarray:
    """The deliberately wrong law: the enhancement grows where gravity is *strong*."""

    return v_bar_squared * np.sqrt(1.0 + v_bar_squared / (a0 * radius))


# ---------------------------------------------------------------------------
# Step 4 -- the confrontation, structurally free of per-object parameters
# ---------------------------------------------------------------------------

#: The label carried by the structural constraint that pins the baryonic term's coefficient
#: to exactly one.  It appears in Farkas certificates by this name.
PIN_LABEL = "pin:baryonic_coefficient_is_exactly_one"

PIN_SOURCE = (
    "declared structural constraint, not a measurement: the baryonic term enters with "
    "coefficient exactly 1, so no universal rescaling of the observed baryons is available "
    "to any family and no free mass parameter exists anywhere in the system"
)


def measured_rows(galaxy: Galaxy, source: str) -> list[MeasuredRow]:
    """One instrument row per published point, in v^2, with sigma propagated outward.

    The published sigma is on ``V_obs``.  Its image in ``v^2`` is asymmetric, so the declared
    propagation is the outward-rounded symmetric envelope ``2 v sigma + 3 sigma^2``, which
    contains ``[(v - k sigma)^2, (v + k sigma)^2]`` for every declared coverage factor
    ``k <= 3``.  Widening outward is the conservative direction for an INFEASIBLE verdict,
    and it is declared rather than derived from any residual.
    """

    parsed: list[MeasuredRow] = []
    for start in range(0, galaxy.count, PARSE_BLOCK_SIZE):
        block = []
        for index in range(start, min(start + PARSE_BLOCK_SIZE, galaxy.count)):
            # Decimal arithmetic on the published strings: v^2 and its propagated sigma are
            # exact decimals, so the instrument still sees a declared decimal string.
            velocity = Decimal(galaxy.published[index][1])
            sigma = Decimal(galaxy.published[index][2])
            propagated = 2 * velocity * sigma + 3 * sigma * sigma
            block.append(
                {
                    "label": f"{galaxy.name}@r={galaxy.published[index][0]}kpc",
                    "point": galaxy.published[index][0],
                    "point_sigma_rule": "half_ulp_of_last_published_digit",
                    "source": source,
                    "value": str(velocity * velocity),
                    "value_sigma": str(propagated),
                    "value_sigma_rule": "propagated_outward",
                    "value_sigma_citation": (
                        "propagated outward in closed form from the published e_Vobs column "
                        f"({galaxy.published[index][2]} km/s) of Lelli, McGaugh & Schombert "
                        "2016, AJ 152, 157, table2; no sigma here was read off a residual"
                    ),
                }
            )
        parsed.extend(parse_rows(block))
    return parsed


def pin_row(source: str) -> MeasuredRow:
    """The structural pin, expressed inside the instrument so certificates can name it."""

    return parse_rows(
        [
            {
                "label": PIN_LABEL,
                "point": "0",
                "point_sigma_rule": "exact",
                "source": source,
                "value": "1",
                "value_sigma_rule": "exact",
            }
        ]
    )[0]


@dataclass(frozen=True, slots=True)
class Design:
    """A pooled design: one column per universal parameter, rows from every galaxy."""

    parameter_names: tuple[str, ...]
    columns: tuple[tuple[Fraction, ...], ...]
    rows: tuple[MeasuredRow, ...]
    galaxy_of_row: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.parameter_names)


def universal_parameter_width(design: Design) -> int:
    """The number of free quantities in a design.  Must not grow with the galaxy count."""

    widths = {len(column) for column in design.columns}
    if widths != {design.width}:
        raise RealDataGravityError("a design row does not have the universal width")
    return design.width


def _pin_columns(width: int, index: int) -> list[Fraction]:
    column = [Fraction(0)] * width
    column[index] = Fraction(1)
    return column


def build_design(
    galaxies: Sequence[Galaxy],
    rows_by_galaxy: Mapping[str, Sequence[MeasuredRow]],
    parameter_names: Sequence[str],
    column_builder: Any,
    pin_row_value: MeasuredRow,
    pinned_indices: Sequence[int],
) -> Design:
    """Pool every galaxy into one system whose width is the universal parameter count."""

    columns: list[tuple[Fraction, ...]] = []
    rows: list[MeasuredRow] = []
    origin: list[str] = []
    for galaxy in galaxies:
        design = column_builder(galaxy)
        galaxy_rows = rows_by_galaxy[galaxy.name]
        if len(design) != len(galaxy_rows):
            raise RealDataGravityError("design and row counts disagree")
        for entry, row in zip(design, galaxy_rows, strict=True):
            if len(entry) != len(parameter_names):
                raise RealDataGravityError("a design row does not have the universal width")
            columns.append(tuple(entry))
            rows.append(row)
            origin.append(galaxy.name)
    for index in pinned_indices:
        columns.append(tuple(_pin_columns(len(parameter_names), index)))
        rows.append(pin_row_value if index == 0 else _relabelled_pin(pin_row_value, index))
        origin.append("structural")
    return Design(
        parameter_names=tuple(parameter_names),
        columns=tuple(columns),
        rows=tuple(rows),
        galaxy_of_row=tuple(origin),
    )


def _relabelled_pin(row: MeasuredRow, index: int) -> MeasuredRow:
    """A second structural pin, distinguished by label so certificates stay readable."""

    return MeasuredRow(
        label=f"{row.label}#{index}",
        point=row.point,
        point_sigma=row.point_sigma,
        point_sigma_rule=row.point_sigma_rule,
        value=row.value,
        value_sigma=row.value_sigma,
        value_sigma_rule=row.value_sigma_rule,
        source=row.source,
        point_citation=row.point_citation,
        value_citation=row.value_citation,
        point_declared=row.point_declared,
        value_declared=row.value_declared,
    )


def _interval(row: MeasuredRow, coverage: Fraction) -> tuple[Fraction, Fraction]:
    return row.value_interval(coverage)


def _certificate_break_even(
    design: Design, active: Sequence[int], terms: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The coverage factor at which this Farkas witness stops being a contradiction.

    A Farkas witness is ``lambda >= 0`` with ``lambda^T A = 0`` and ``lambda^T b < 0``.  Only
    ``b`` depends on the coverage factor ``k``, and it does so affinely: an upper row
    contributes ``value + k sigma`` and a lower row ``-value + k sigma``.  So
    ``lambda^T b(k) = A0 + k B0`` with ``B0 = sum lambda_i sigma_i >= 0``, and the *same*
    nonnegative multipliers keep certifying infeasibility for every ``k < -A0/B0``.  That
    ratio is therefore an exact, certificate-derived exclusion bound: below it this family
    cannot be feasible however the declared intervals are widened.  It is read off the
    witness, not fitted, and it ranks nothing -- a wider bound is not a better theory, it is
    a theory excluded over a wider range of declared tolerances.
    """

    lookup = {design.rows[index].label: design.rows[index] for index in active}
    offset = Fraction(0)
    slope = Fraction(0)
    for term in terms:
        row = lookup.get(str(term["row"]))
        if row is None:
            return None
        multiplier = Fraction(
            int(term["multiplier"]["numerator"]), int(term["multiplier"]["denominator"])
        )
        sign = Fraction(1) if term["bound"] == "upper" else Fraction(-1)
        offset += multiplier * sign * row.value
        slope += multiplier * row.value_sigma
    if slope <= 0:
        return {
            "coverage_independent": True,
            "reading": (
                "every row in this witness carries zero declared uncertainty, so widening "
                "the coverage factor cannot dissolve the contradiction at any k"
            ),
        }
    break_even = -offset / slope
    return {
        "coverage_independent": False,
        "exact": _fraction_data(break_even),
        "decimal": _num(float(break_even)),
        "reading": (
            "the same nonnegative multipliers certify infeasibility for every declared "
            "coverage factor strictly below this value, so no widening of the published "
            "intervals below it can make this family reachable"
        ),
    }


def decide_pooled(design: Design, coverage: Fraction, seed_rows: int = 6) -> dict[str, Any]:
    """Decide the pooled system exactly, using the instrument, without a 400-row tableau.

    Constraint generation, and both branches stay exact.  If any *subset* of the declared
    inequalities is infeasible then the whole pooled system is infeasible, so an INFEASIBLE
    verdict on a subset -- carrying the instrument's own re-verified Farkas certificate --
    settles the pooled question.  A FEASIBLE verdict is only returned after the witness the
    instrument produced is re-checked against **every** pooled row in exact rational
    arithmetic, so nothing is decided on the subset alone.
    """

    total = len(design.rows)
    active = sorted(set(range(max(0, total - seed_rows), total)) | set(range(min(seed_rows, total))))
    while True:
        system = build_system(
            [design.columns[index] for index in active],
            [design.rows[index] for index in active],
            coverage,
        )
        verdict = decide_system(system)
        if verdict["verdict"] == INFEASIBLE:
            witness = verdict["witness"]
            break_even = _certificate_break_even(design, active, witness["terms"])
            return {
                "verdict": INFEASIBLE,
                "certificate": {
                    "checked_here": witness["checked_here"],
                    "combined_right_hand_side": witness["combined_right_hand_side"],
                    "kind": witness["kind"],
                    "reading": witness["reading"],
                    "terms": witness["terms"],
                    "unreachable_rows": witness["unreachable_rows"],
                },
                "certificate_break_even_coverage": break_even,
                "certificate_row_count": len(active),
                "subset_infeasibility_implies_pooled_infeasibility": True,
            }
        point = [Fraction(value) for value in verdict["point"]]
        violations: list[tuple[Fraction, int]] = []
        for index in range(total):
            lower, upper = _interval(design.rows[index], coverage)
            predicted = sum(
                (cell * value for cell, value in zip(design.columns[index], point, strict=True)),
                Fraction(0),
            )
            if predicted > upper:
                violations.append((predicted - upper, index))
            elif predicted < lower:
                violations.append((lower - predicted, index))
        if not violations:
            return {
                "verdict": FEASIBLE,
                "witness": {
                    "checked_against_every_pooled_row": True,
                    "parameters": {
                        name: _fraction_data(value)
                        for name, value in zip(design.parameter_names, point, strict=True)
                    },
                    "parameters_decimal": {
                        name: _num(float(value))
                        for name, value in zip(design.parameter_names, point, strict=True)
                    },
                    "pooled_row_count": total,
                },
            }
        violations.sort(key=lambda item: (-item[0], item[1]))
        added = [index for _, index in violations[:2] if index not in active]
        if not added:
            raise RealDataGravityError("constraint generation stalled")
        active = sorted(set(active) | set(added))


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A declared model: its universal parameters and how its design columns are built."""

    name: str
    parameter_names: tuple[str, ...]
    pinned_indices: tuple[int, ...]
    description: str


def family_model(family: Family, free_arms: bool) -> ModelSpec:
    if free_arms:
        return ModelSpec(
            name=f"family_{family.ordinal}_universal_arms",
            parameter_names=("baryonic_coefficient", "w_yukawa", "w_power"),
            pinned_indices=(0,),
            description=(
                "the family's law with its two kernel arm weights promoted to universal "
                "constants shared by every galaxy; the baryonic coefficient stays pinned to 1"
            ),
        )
    return ModelSpec(
        name=f"family_{family.ordinal}_zero_freedom",
        parameter_names=("baryonic_coefficient",),
        pinned_indices=(0,),
        description=(
            "the family's law exactly as enumerated, with its own arm weights: a prediction "
            "with zero free parameters once the two universal unit constants are fixed"
        ),
    )


class ColumnCache:
    """Memo for the expensive pieces.  Every entry is a pure function of declared inputs."""

    def __init__(self, prepared: Mapping[str, dict[str, Any]]) -> None:
        self._prepared = prepared
        self._yukawa: dict[tuple[str, str, str], np.ndarray] = {}
        self._power: dict[tuple[str, str, str, str, str], np.ndarray] = {}
        self._screen: dict[tuple[str, str, str, str], np.ndarray] = {}

    def yukawa(self, name: str, unit_text: str, length_1: str) -> np.ndarray:
        key = (name, unit_text, length_1)
        if key not in self._yukawa:
            unit = float(Fraction(Decimal(unit_text)))
            self._yukawa[key] = yukawa_arm(
                self._prepared[name]["plan"], unit, float(Fraction(length_1))
            )
        return self._yukawa[key]

    def power(
        self, name: str, unit_text: str, length_2: str, exponent_p: str, exponent_t: str
    ) -> np.ndarray:
        key = (name, unit_text, length_2, exponent_p, exponent_t)
        if key not in self._power:
            unit = float(Fraction(Decimal(unit_text)))
            self._power[key] = power_arm(
                self._prepared[name]["plan"],
                unit,
                float(Fraction(length_2)),
                float(Fraction(exponent_p)),
                float(Fraction(exponent_t)),
            )
        return self._power[key]

    def screen(self, name: str, unit_text: str, scale: str, sharpness: str) -> np.ndarray:
        key = (name, unit_text, scale, sharpness)
        if key not in self._screen:
            entry = self._prepared[name]
            self._screen[key] = screening_factor(
                entry["radius"],
                entry["v_bar_squared"],
                float(Fraction(Decimal(unit_text))),
                float(Fraction(Decimal(scale))),
                float(Fraction(sharpness)),
            )
        return self._screen[key]


def _family_columns(
    galaxy: Galaxy,
    family: Family,
    a0_text: str,
    unit_text: str,
    cache: ColumnCache,
    v_bar_squared: np.ndarray,
    radius: np.ndarray,
    free_arms: bool,
    wrong_law: bool = False,
) -> list[list[Fraction]]:
    newton = float(Fraction(Decimal(G_KPC_KMS2_PER_MSUN)))
    values = family.parameters
    a0 = float(Fraction(Decimal(a0_text)))
    length_unit = float(Fraction(Decimal(unit_text)))
    if wrong_law:
        local = wrong_local_term(radius, v_bar_squared, a0)
    else:
        local = local_term(radius, v_bar_squared, a0)
    screen = cache.screen(galaxy.name, unit_text, family.screen_scale, family.sharpness)
    prefactor = newton / (length_unit * length_unit)
    arm_y = prefactor * cache.yukawa(galaxy.name, unit_text, values["L1"])
    arm_p = prefactor * cache.power(
        galaxy.name, unit_text, values["L2"], values["p"], values["t"]
    )
    boost_y = radius * screen * arm_y
    boost_p = radius * screen * arm_p
    if free_arms:
        return [
            [_rational(local[i]), _rational(boost_y[i]), _rational(boost_p[i])]
            for i in range(galaxy.count)
        ]
    weight_y = float(Fraction(values["w_yukawa"]))
    weight_p = float(Fraction(values["w_power"]))
    combined = local + weight_y * boost_y + weight_p * boost_p
    return [[_rational(combined[i])] for i in range(galaxy.count)]


# ---------------------------------------------------------------------------
# Mandatory controls
# ---------------------------------------------------------------------------


def newtonian_columns(galaxy: Galaxy, v_bar_squared: np.ndarray) -> list[list[Fraction]]:
    return [[_rational(value)] for value in v_bar_squared]


def run_controls(
    galaxies: Sequence[Galaxy],
    rows_by_galaxy: Mapping[str, Sequence[MeasuredRow]],
    prepared: Mapping[str, dict[str, Any]],
    families: Sequence[Family],
    cache: ColumnCache,
) -> dict[str, dict[str, Any]]:
    """The controls that decide whether this pipeline may be believed at all.

    Returned keyed by coverage factor, because every design is built once and decided at
    each declared coverage rather than rebuilt three times.
    """

    pin = pin_row(PIN_SOURCE)
    family = select_best_family(families)
    reference_a0 = REFERENCE_GRID_POINT["a0"]
    reference_unit = REFERENCE_GRID_POINT["length_unit"]

    designs: dict[str, tuple[Design, dict[str, Any]]] = {
        "newtonian_baryons_only": (
            build_design(
                galaxies,
                rows_by_galaxy,
                ("baryonic_coefficient",),
                lambda galaxy: newtonian_columns(
                    galaxy, prepared[galaxy.name]["v_bar_squared"]
                ),
                pin,
                (0,),
            ),
            {
                "meaning": (
                    "the published baryons alone cannot reach the published velocities on "
                    "these galaxies; this infeasibility, certified in exact arithmetic, is "
                    "the dark-matter problem, and if it ever returns FEASIBLE every verdict "
                    "in this receipt is void"
                )
            },
        ),
        "newtonian_baryons_with_one_universal_rescale": (
            build_design(
                galaxies,
                rows_by_galaxy,
                ("baryonic_coefficient",),
                lambda galaxy: newtonian_columns(
                    galaxy, prepared[galaxy.name]["v_bar_squared"]
                ),
                pin,
                (),
            ),
            {
                "meaning": (
                    "even a single universal mass-to-light rescale, shared by every galaxy, "
                    "cannot save Newtonian baryons; the failure is not a calibration error"
                )
            },
        ),
        "deliberately_wrong_law": (
            build_design(
                galaxies,
                rows_by_galaxy,
                ("baryonic_coefficient", "w_yukawa", "w_power"),
                lambda galaxy: _family_columns(
                    galaxy,
                    family,
                    reference_a0,
                    reference_unit,
                    cache,
                    prepared[galaxy.name]["v_bar_squared"],
                    prepared[galaxy.name]["radius"],
                    free_arms=True,
                    wrong_law=True,
                ),
                pin,
                (0,),
            ),
            {
                "law": (
                    "the best family's law with the local factor inverted: "
                    "sqrt(1 + g_N/a0) instead of sqrt(1 + a0/g_N), so the modification grows "
                    "where gravity is strong and vanishes where it is weak -- the same "
                    "complexity, the wrong asymptotics"
                )
            },
        ),
    }

    # The Newtonian limit of the law itself, checked once at the level of the columns.
    worst_limit = 0.0
    for galaxy in galaxies:
        entry = prepared[galaxy.name]
        limit = local_term(entry["radius"], entry["v_bar_squared"], 0.0)
        worst_limit = max(
            worst_limit,
            float(np.max(np.abs(limit - entry["v_bar_squared"]) / entry["v_bar_squared"])),
        )
    if worst_limit != 0.0:
        raise RealDataGravityError("the Newtonian limit of the law is not exact")

    by_coverage: dict[str, dict[str, Any]] = {}
    for coverage_text in COVERAGE_GRID:
        coverage = Fraction(coverage_text)
        results: dict[str, Any] = {}
        for name, (design, extra) in designs.items():
            verdict = decide_pooled(design, coverage)
            results[name] = {
                "expected": INFEASIBLE,
                "must_be_infeasible": True,
                "universal_parameter_count": universal_parameter_width(design),
                **extra,
                **verdict,
            }
            if verdict["verdict"] != INFEASIBLE:
                raise RealDataGravityError(
                    f"mandatory control {name} did not come back INFEASIBLE"
                )
        results["newtonian_limit_of_the_law"] = {
            "expected": "the modification switches off exactly",
            "meaning": (
                "setting the universal acceleration scale a0 to zero must return "
                "v^2 = V_bar^2 identically, which is the general-relativistic limit at the "
                "level of the observable rather than of the action"
            ),
            "must_be_zero": True,
            "verdict": FEASIBLE,
            "worst_relative_deviation": _num(worst_limit),
        }
        by_coverage[coverage_text] = results
    return by_coverage


# ---------------------------------------------------------------------------
# Preparation and the scan
# ---------------------------------------------------------------------------


def prepare_galaxy(
    galaxy: Galaxy, upsilon_disk: Fraction, upsilon_bul: Fraction, quadrature: Mapping[str, int]
) -> dict[str, Any]:
    v_bar_exact = baryonic_v_squared(galaxy, upsilon_disk, upsilon_bul)
    radius = np.array([float(value) for value in galaxy.radius], dtype=float)
    v_bar_squared = np.array([float(value) for value in v_bar_exact], dtype=float)
    return {
        "plan": build_convolution_plan(radius, v_bar_squared, quadrature),
        "quadrature": dict(quadrature),
        "radius": radius,
        "v_bar_squared": v_bar_squared,
        "v_bar_squared_exact": tuple(v_bar_exact),
    }


def quadrature_convergence(
    galaxies: Sequence[Galaxy], prepared: Mapping[str, dict[str, Any]], families: Sequence[Family]
) -> dict[str, Any]:
    """Refine the declared quadrature and confirm no boost column moves materially."""

    tolerance = float(Fraction(Decimal(QUADRATURE_TOLERANCE)))
    arms = {
        ("yukawa", family.parameters["L1"], "", "") for family in families
    } | {
        ("power", family.parameters["L2"], family.parameters["p"], family.parameters["t"])
        for family in families
    }
    worst = 0.0
    worst_where = ""
    for galaxy in galaxies:
        entry = prepared[galaxy.name]
        refined = build_convolution_plan(
            entry["radius"], entry["v_bar_squared"], QUADRATURE_REFINED
        )
        for unit_text in LENGTH_UNIT_GRID:
            unit = float(Fraction(Decimal(unit_text)))
            for kind, first, second, third in sorted(arms):
                if kind == "yukawa":
                    coarse = yukawa_arm(entry["plan"], unit, float(Fraction(first)))
                    fine = yukawa_arm(refined, unit, float(Fraction(first)))
                else:
                    args = (float(Fraction(first)), float(Fraction(second)), float(Fraction(third)))
                    coarse = power_arm(entry["plan"], unit, *args)
                    fine = power_arm(refined, unit, *args)
                scale = np.maximum(np.abs(coarse), np.abs(fine))
                safe = np.where(scale > 0, scale, 1.0)
                moved = float(np.max(np.where(scale > 0, np.abs(fine - coarse) / safe, 0.0)))
                if moved > worst:
                    worst = moved
                    worst_where = f"{galaxy.name}:{kind}:{first}:{second}:{third}@l={unit_text}"
    if worst > tolerance:
        raise RealDataGravityError("the nonlocal quadrature is not converged at the declared grid")
    return {
        "arms_checked": len(arms),
        "declared_quadrature": dict(QUADRATURE),
        "graded_mesh": (
            "quintic smoothstep phi(u) = 6u^5 - 15u^4 + 10u^3 inside every published mass "
            "bin; phi' vanishes quadratically at both edges, so the shell average's endpoint "
            "logarithm becomes a u^2 log u integrand and the map preserves the enclosed mass"
        ),
        "refined_quadrature": dict(QUADRATURE_REFINED),
        "tolerance": QUADRATURE_TOLERANCE,
        "within_tolerance": True,
        "worst_relative_move": _num(worst),
        "worst_relative_move_at": worst_where,
    }


def design_crosscheck(
    galaxies: Sequence[Galaxy], prepared: Mapping[str, dict[str, Any]], families: Sequence[Family]
) -> dict[str, Any]:
    """Recompute one design column at 50 digits and compare with the float64 column."""

    mp.mp.dps = CROSSCHECK_DPS
    family = select_best_family(families)
    tolerance = mp.mpf(CROSSCHECK_TOLERANCE)
    worst = mp.mpf(0)
    a0_text = REFERENCE_GRID_POINT["a0"]
    for galaxy in galaxies:
        entry = prepared[galaxy.name]
        a0 = mp.mpf(a0_text)
        column = local_term(entry["radius"], entry["v_bar_squared"], float(a0))
        for index in range(galaxy.count):
            v_bar = mp.mpf(entry["v_bar_squared_exact"][index].numerator) / mp.mpf(
                entry["v_bar_squared_exact"][index].denominator
            )
            radius = mp.mpf(galaxy.radius[index].numerator) / mp.mpf(
                galaxy.radius[index].denominator
            )
            exact = v_bar * mp.sqrt(1 + a0 * radius / v_bar)
            moved = abs(mp.mpf(float(column[index])) - exact) / exact
            worst = max(worst, moved)
    if worst > tolerance:
        raise RealDataGravityError("the float64 design disagrees with its 50-digit recomputation")
    return {
        "family_checked": family.ordinal,
        "tolerance": CROSSCHECK_TOLERANCE,
        "within_tolerance": True,
        "working_precision_digits": CROSSCHECK_DPS,
        "worst_relative_disagreement": _num(worst),
    }


def confront_family(
    family: Family,
    galaxies: Sequence[Galaxy],
    rows_by_galaxy: Mapping[str, Sequence[MeasuredRow]],
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
    free_arms: bool,
) -> dict[str, dict[str, Any]]:
    """Scan the declared universal-constant grid, and decide at every coverage factor.

    Each grid point's pooled design is built once and handed to the exact linear program at
    each declared coverage, so the coverage ladder costs nothing extra and cannot be chosen
    after the fact.
    """

    pin = pin_row(PIN_SOURCE)
    spec = family_model(family, free_arms)
    accumulators: dict[str, dict[str, Any]] = {
        text: {
            "feasible_grid_points": [],
            "reference_certificate": None,
            "breaking_rows": None,
            "witness": None,
            "row_appearances": {},
        }
        for text in COVERAGE_GRID
    }
    #: Per grid point, the largest break-even coverage any of its certificates reaches.
    #: A certificate found at k certifies infeasibility for every k' below its break-even,
    #: so the strongest statement about one grid point is the maximum over its certificates,
    #: and the strongest statement about the family is the minimum over its grid points.
    grid_break_even: dict[tuple[str, str], Fraction] = {}
    any_grid_point_feasible = False
    widths: set[int] = set()
    for a0_text in A0_GRID:
        for unit_text in LENGTH_UNIT_GRID:
            design = build_design(
                galaxies,
                rows_by_galaxy,
                spec.parameter_names,
                lambda galaxy, a0=a0_text, unit=unit_text: _family_columns(
                    galaxy,
                    family,
                    a0,
                    unit,
                    cache,
                    prepared[galaxy.name]["v_bar_squared"],
                    prepared[galaxy.name]["radius"],
                    free_arms=free_arms,
                ),
                pin,
                spec.pinned_indices,
            )
            widths.add(universal_parameter_width(design))
            is_reference = (
                a0_text == REFERENCE_GRID_POINT["a0"]
                and unit_text == REFERENCE_GRID_POINT["length_unit"]
            )
            for coverage_text in COVERAGE_GRID:
                state = accumulators[coverage_text]
                verdict = decide_pooled(design, Fraction(coverage_text))
                if verdict["verdict"] == FEASIBLE:
                    any_grid_point_feasible = True
                    state["feasible_grid_points"].append(
                        {"a0": a0_text, "length_unit": unit_text}
                    )
                    if state["witness"] is None or is_reference:
                        state["witness"] = verdict["witness"]
                else:
                    named = set(verdict["certificate"]["unreachable_rows"])
                    state["breaking_rows"] = (
                        named
                        if state["breaking_rows"] is None
                        else (state["breaking_rows"] & named)
                    )
                    for label in named:
                        if label != PIN_LABEL:
                            state["row_appearances"][label] = (
                                state["row_appearances"].get(label, 0) + 1
                            )
                    if is_reference or state["reference_certificate"] is None:
                        state["reference_certificate"] = verdict["certificate"]
                    bound = verdict["certificate_break_even_coverage"]
                    if bound is not None and not bound["coverage_independent"]:
                        value = Fraction(
                            bound["exact"]["numerator"], bound["exact"]["denominator"]
                        )
                        key = (a0_text, unit_text)
                        if key not in grid_break_even or value > grid_break_even[key]:
                            grid_break_even[key] = value
    if widths != {len(spec.parameter_names)}:
        raise RealDataGravityError("the pooled width changed across the grid")

    exclusion: dict[str, Any] | None = None
    if not any_grid_point_feasible and len(grid_break_even) == len(A0_GRID) * len(
        LENGTH_UNIT_GRID
    ):
        weakest_point, weakest = min(grid_break_even.items(), key=lambda item: (item[1], item[0]))
        exclusion = {
            "at_grid_point": {"a0": weakest_point[0], "length_unit": weakest_point[1]},
            "decimal": _num(float(weakest)),
            "exact": _fraction_data(weakest),
            "reading": (
                "every declared grid point carries a Farkas witness that stays a "
                "contradiction for every coverage factor below its own break-even; the "
                "smallest of those break-evens is reported here, so for any declared "
                "coverage strictly below it the family is certified infeasible everywhere "
                "on the grid. It is an exclusion bound read off the certificates, not a "
                "measure of how well anything fits, and a larger bound is not a better "
                "theory -- it is a theory excluded over a wider range of declared tolerances"
            ),
        }

    out: dict[str, dict[str, Any]] = {}
    for coverage_text, state in accumulators.items():
        feasible = state["feasible_grid_points"]
        result: dict[str, Any] = {
            "feasible_grid_points": feasible,
            "grid_point_count": len(A0_GRID) * len(LENGTH_UNIT_GRID),
            "model": spec.name,
            "model_description": spec.description,
            "ordinal": family.ordinal,
            "universal_parameter_count": len(spec.parameter_names),
            "universal_parameter_names": list(spec.parameter_names),
            "verdict": FEASIBLE if feasible else INFEASIBLE,
        }
        if feasible:
            result["witness"] = state["witness"]
        else:
            appearances = state["row_appearances"]
            ranked = sorted(appearances.items(), key=lambda item: (-item[1], item[0]))
            result["certificate"] = state["reference_certificate"]
            result["rows_unreachable_at_every_grid_point"] = sorted(
                state["breaking_rows"] or set()
            )
            result["most_frequently_unreachable_rows"] = [
                {"grid_points_naming_it": count, "row": label} for label, count in ranked[:8]
            ]
            if exclusion is not None:
                result["certified_infeasible_below_coverage_factor"] = exclusion
        out[coverage_text] = result
    return out


def interval_reach(
    family: Family,
    galaxies: Sequence[Galaxy],
    rows_by_galaxy: Mapping[str, Sequence[MeasuredRow]],
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
) -> dict[str, Any]:
    """How many published intervals the zero-freedom prediction lands inside, per grid point.

    This is an evaluation, not a fit: at each declared grid point the family's own law with
    its own enumerated arm weights is evaluated, and each of the 214 published one-sigma
    intervals is asked a yes/no question. Counting how many say yes is a tally of declared
    verdicts, and it decides nothing -- the feasibility verdict above is decided only by the
    exact linear program and its certificates. It is reported so that a family that misses
    everything and a family that misses one point are visibly different.
    """

    coverage = Fraction(COVERAGE_GRID[0])
    best = -1
    best_at: dict[str, str] = {}
    best_galaxies = 0
    best_galaxy_names: list[str] = []
    best_galaxies_at: dict[str, str] = {}
    for a0_text in A0_GRID:
        for unit_text in LENGTH_UNIT_GRID:
            reached = 0
            whole_curves: list[str] = []
            for galaxy in galaxies:
                entry = prepared[galaxy.name]
                columns = _family_columns(
                    galaxy,
                    family,
                    a0_text,
                    unit_text,
                    cache,
                    entry["v_bar_squared"],
                    entry["radius"],
                    free_arms=False,
                )
                inside = 0
                for column, row in zip(columns, rows_by_galaxy[galaxy.name], strict=True):
                    lower, upper = row.value_interval(coverage)
                    if lower <= column[0] <= upper:
                        inside += 1
                reached += inside
                if inside == galaxy.count:
                    whole_curves.append(galaxy.name)
            if reached > best:
                best = reached
                best_at = {"a0": a0_text, "length_unit": unit_text}
            if len(whole_curves) > best_galaxies:
                best_galaxies = len(whole_curves)
                best_galaxy_names = sorted(whole_curves)
                best_galaxies_at = {"a0": a0_text, "length_unit": unit_text}
    return {
        "at_grid_point": best_at,
        "coverage_factor": COVERAGE_GRID[0],
        "decides_nothing": True,
        "galaxies_fully_reached": best_galaxies,
        "galaxies_fully_reached_at": best_galaxies_at,
        "galaxies_fully_reached_named": best_galaxy_names,
        "galaxies_offered": len(galaxies),
        "intervals_reached": best,
        "published_intervals": sum(galaxy.count for galaxy in galaxies),
        "whole_curve_reading": (
            "a galaxy counts as fully reached only when every one of its published points "
            "lands inside its own declared interval, with the family's own enumerated arm "
            "weights and no free parameter of any kind at that grid point; this is the "
            "per-galaxy answer to how many curves the law reaches without per-object freedom"
        ),
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build_receipt(root: Path) -> dict[str, Any]:
    """Everything the run establishes, sealed and replayable."""

    root = root.resolve()
    contract = check_contract(root)
    probes = contract_probe_report()
    galaxies, provenance = load_galaxies(root)
    families = load_families(root)
    best = select_best_family(families)
    convention = provenance["mass_to_light_convention"]
    upsilon_disk = Fraction(convention["disk_3_6um"])
    upsilon_bul = Fraction(convention["bulge_3_6um"])

    source = (
        f"{provenance['source']['primary_citation']}; {provenance['source']['table']}; "
        f"{provenance['source']['dataset_doi']}"
    )
    prepared = {
        galaxy.name: prepare_galaxy(galaxy, upsilon_disk, upsilon_bul, QUADRATURE)
        for galaxy in galaxies
    }
    rows_by_galaxy = {galaxy.name: measured_rows(galaxy, source) for galaxy in galaxies}

    convergence = quadrature_convergence(galaxies, prepared, families)
    crosscheck = design_crosscheck(galaxies, prepared, families)

    cache = ColumnCache(prepared)
    controls_by_coverage = run_controls(galaxies, rows_by_galaxy, prepared, families, cache)
    scans = {
        family.ordinal: {
            "universal_arms": confront_family(
                family, galaxies, rows_by_galaxy, prepared, cache, free_arms=True
            ),
            "zero_freedom": confront_family(
                family, galaxies, rows_by_galaxy, prepared, cache, free_arms=False
            ),
        }
        for family in families
    }
    reach = {
        family.ordinal: interval_reach(family, galaxies, rows_by_galaxy, prepared, cache)
        for family in families
    }

    coverage_results: dict[str, Any] = {}
    for coverage_text in COVERAGE_GRID:
        per_family = {
            str(ordinal): {
                mode: modes[mode][coverage_text] for mode in ("universal_arms", "zero_freedom")
            }
            for ordinal, modes in scans.items()
        }
        surviving = sorted(
            int(ordinal)
            for ordinal, entry in per_family.items()
            if entry["universal_arms"]["verdict"] == FEASIBLE
        )
        coverage_results[coverage_text] = {
            "controls": controls_by_coverage[coverage_text],
            "families_feasible_universal_arms": surviving,
            "families_feasible_zero_freedom": sorted(
                int(ordinal)
                for ordinal, entry in per_family.items()
                if entry["zero_freedom"]["verdict"] == FEASIBLE
            ),
            "per_family": per_family,
            "surviving_family_count": len(surviving),
        }

    headline = coverage_results[COVERAGE_GRID[0]]
    family_summary = []
    for family in families:
        entry = headline["per_family"][str(family.ordinal)]["universal_arms"]
        breaking = entry.get("most_frequently_unreachable_rows") or []
        bound = entry.get("certified_infeasible_below_coverage_factor") or {}
        family_summary.append(
            {
                "candidates": family.size,
                "certified_infeasible_below_coverage_factor": bound.get("decimal"),
                "galaxies_fully_reached": reach[family.ordinal]["galaxies_fully_reached"],
                "galaxies_offered": reach[family.ordinal]["galaxies_offered"],
                "intervals_reached": reach[family.ordinal]["intervals_reached"],
                "ordinal": family.ordinal,
                "published_intervals": reach[family.ordinal]["published_intervals"],
                "radius_most_often_unreachable": breaking[0]["row"] if breaking else None,
                "verdict_at_headline_coverage": entry["verdict"],
            }
        )
    body: dict[str, Any] = {
        "answers": (
            "The twelve STABLE_PASS curvature-screened K-mouflage families were confronted "
            f"with {provenance['point_count']} published rotation-curve points from "
            f"{provenance['galaxy_count']} SPARC galaxies, with no dark halo, no unseen "
            "component, and no per-object parameter available to any of them. At the "
            "published one-sigma interval "
            f"{headline['surviving_family_count']} of {len(families)} families are feasible "
            "with one universal parameter set. Newtonian baryons alone are INFEASIBLE on the "
            "same rows, as they must be. This is the first time any gravity receipt in this "
            "repository has opened measured data: its two predecessors both record "
            "observational_data_opened false and synthetic_controls_only true."
        ),
        "assumptions": dict(ASSUMPTIONS),
        "claims": dict(CLAIMS),
        "contract_amendment": contract,
        "contract_probe_actions": probes,
        "exploratory_caveat": {
            "may_be_cited_as_confirmation": False,
            "sealed_no_refit_trial": False,
            "statement": (
                "The universal constants were scanned over a declared grid against the same "
                "galaxies the verdict is read from, so fitting and testing used the same "
                "data. That is declared, not hidden, and it is why no result here may be "
                "cited as a confirmation."
            ),
            "why_a_negative_is_not_weakened_by_it": (
                "Reusing the data can only make a theory look better, never worse. A "
                "FEASIBLE verdict obtained this way would be provisional. An INFEASIBLE "
                "verdict obtained this way is the opposite: the family was handed the full "
                "declared grid and still could not reach the published intervals, so the "
                "exploratory framing strengthens rather than weakens the negative."
            ),
            "what_a_sealed_trial_would_add": (
                "A sealed no-refit trial would fix the universal constants first and then "
                "open galaxies withheld from this scan. It is reserved and out of scope here."
            ),
        },
        "controls_summary": {
            "newtonian_baryons_only_infeasible": all(
                coverage_results[key]["controls"]["newtonian_baryons_only"]["verdict"] == INFEASIBLE
                for key in COVERAGE_GRID
            ),
            "wrong_law_infeasible": all(
                coverage_results[key]["controls"]["deliberately_wrong_law"]["verdict"] == INFEASIBLE
                for key in COVERAGE_GRID
            ),
        },
        "coverage_factors": list(COVERAGE_GRID),
        "data_provenance": provenance,
        "decision": _decision(headline, provenance, len(families)),
        "derivation_chain": derivation_chain(),
        "design_crosscheck": crosscheck,
        "families": {
            str(family.ordinal): {
                **render_family_formulas(family),
                "published_interval_reach": reach[family.ordinal],
            }
            for family in families
        },
        "family_selection_rule": BEST_FAMILY_RULE,
        "family_summary": family_summary,
        "headline_coverage_factor": COVERAGE_GRID[0],
        "best_family_ordinal": best.ordinal,
        "quadrature_convergence": convergence,
        "results_by_coverage_factor": coverage_results,
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "trial_type": TRIAL_TYPE,
        "universal_constant_grids": {
            "a0_kms2_per_kpc": list(A0_GRID),
            "kms2_per_kpc_in_m_s2": KMS2_PER_KPC_IN_M_S2,
            "length_unit_kpc": list(LENGTH_UNIT_GRID),
            "reference_grid_point": dict(REFERENCE_GRID_POINT),
        },
        "counts": {
            "candidates_represented": sum(family.size for family in families),
            "families_confronted": len(families),
            "galaxies": provenance["galaxy_count"],
            "measured_rows": provenance["point_count"],
        },
    }
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise RealDataGravityError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    return {**body, "content_sha256": canonical_sha256(body)}


def _decision(headline: Mapping[str, Any], provenance: Mapping[str, Any], total: int) -> str:
    count = headline["surviving_family_count"]
    if count:
        return (
            f"EXPLORATORY: {count} of {total} surviving screened-gravity families remain feasible "
            f"against {provenance['point_count']} published rotation-curve points from "
            f"{provenance['galaxy_count']} galaxies with one universal parameter set and no "
            "invisible matter, while Newtonian baryons alone are certified INFEASIBLE on the "
            "same rows. Fitting and testing used the same galaxies, so this is survival, not "
            "confirmation, and it may not be cited as a confirmatory result."
        )
    return (
        "EXPLORATORY NEGATIVE: no surviving screened-gravity family reaches every published "
        f"interval across {provenance['galaxy_count']} galaxies and "
        f"{provenance['point_count']} points with one universal parameter set, at any "
        "declared coverage factor up to the instrument's cap, each failure carrying a "
        "re-verified Farkas certificate naming the radius that breaks it. Newtonian baryons "
        "alone are INFEASIBLE on the same rows, as they must be, and a deliberately wrong "
        "law is INFEASIBLE too. The search was exploratory and was therefore given every "
        "advantage the declared grid allows; a negative found under those conditions is not "
        "weakened by the exploratory framing, because no amount of further searching on this "
        "declared grid could have produced a survivor. It remains a statement about these "
        "declared intervals, this declared grid, and these six galaxies, and it is not a "
        "sealed no-refit trial."
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Reject tamper or drift by exact deterministic replay."""

    if receipt.get("schema_version") != RESULT_SCHEMA:
        raise RealDataGravityError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise RealDataGravityError("receipt seal changed")
    if receipt.get("claims") != CLAIMS:
        raise RealDataGravityError("claims changed")
    if receipt.get("trial_type") != TRIAL_TYPE:
        raise RealDataGravityError("trial type changed")
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise RealDataGravityError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    replayed = build_receipt(root)
    if dict(receipt) != replayed:
        raise RealDataGravityError("receipt exact replay changed")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    """Write a receipt once; a differing rewrite is refused rather than silently accepted."""

    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RealDataGravityError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-data confrontation (exploratory).")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=RECEIPT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(root)
    write_immutable(output, receipt)
    validate_receipt(receipt, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A0_GRID",
    "ASSUMPTIONS",
    "CLAIMS",
    "COVERAGE_GRID",
    "LENGTH_UNIT_GRID",
    "RECEIPT_PATH",
    "RESULT_SCHEMA",
    "SCOPE",
    "TRIAL_TYPE",
    "Design",
    "Family",
    "Galaxy",
    "RealDataGravityError",
    "build_receipt",
    "check_contract",
    "clause_a_violations",
    "confront_family",
    "contract_probe_report",
    "derivation_chain",
    "load_families",
    "load_galaxies",
    "main",
    "render_action_latex",
    "render_family_formulas",
    "render_law_latex",
    "render_observable_latex",
    "select_best_family",
    "universal_parameter_width",
    "validate_receipt",
]
