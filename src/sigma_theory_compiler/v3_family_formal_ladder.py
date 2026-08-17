"""Formal ladder over the 71 surviving screened-gravity families of ``screened-v3.json``.

The v3 GPU screen is a *static* screen: it decided 835.2 million ordinal-indexed
screened-kernel candidates against synthetic galaxy/cluster/lensing/Solar gates and left
45 546 all-gate passers in 71 grid-connected equivalence families.  Every one of those
families carries a ``covariant_lift_candidate`` block -- a declared field-theory ansatz for
the static kernel it fits, *not* a derivation.  This module asks the next question: of the
71 declared lifts, which ones can the repository's existing formal machinery even accept,
and of those, which survive ghost-freedom, gradient stability, tensor sanity,
principal-symbol hyperbolicity, and the positive-energy rung?

Three honesty rules are enforced structurally, not by comment:

1. **A rung this repository cannot execute is a typed blocker, never a silent pass.**  The
   nonlocal ``(-Box)^alpha`` arm of every v3 lift has no action-IR adapter anywhere in the
   tree, so the *complete* lift of every family is BLOCKED at materialization.  That split
   is itself a result and it is reported first.
2. **What is expressible is still run.**  The screening sector of each lift -- the cubic
   Galileon / kinetic braiding sector for the acceleration-screened families and the
   K-mouflage k-essence sector for the curvature-screened ones -- *is* inside scalar-tensor
   reach, and a ghost or a gradient instability there kills the family outright no matter
   what the blocked arms do.  The ladder therefore runs on that sector as a **necessary
   condition** and a rejection is a genuine scientific elimination.
3. **Controls must fire.**  Every run replays the repository's known-answer controls
   (canonical scalar passes, wrong-sign ghost rejects, gradient/superluminal negatives
   reject) *through this module's own ladder*.  A run whose controls do not fire aborts.

Reused machinery (nothing here re-derives what the tree already proves):

``scalar_tensor_pack.compile_scalar_tensor_pack``
    Compiles a normalized Horndeski ``L2-L4`` function family into exact symbolic
    ``G_T``/``F_T``/``Sigma``/``Theta``/``G_S``/``F_S``/k-essence blocks and runs ten generic
    covariant-variation, ADM, Dirac, tensor, scalar, k-essence, and cubic-BSSN controls.
``flrw_background.certify_flrw_background``
    Interval-certifies one *on-shell* FLRW trajectory of that IR and returns uniform health
    lower bounds plus the formulation route (generalized-harmonic vs modified-harmonic).
``principal_symbol.analyze_isotropic_second_order_symbol`` and
``principal_symbol.run_principal_symbol_controls``
    Reduced principal-symbol decision (ghost, gradient, real characteristics, strong
    hyperbolicity, cone policy) and its canonical/ghost/gradient/superluminal controls.
``horndeski.generic_kessence_nonlinear_adm_legendre_control``
    Nonlinear pointwise Legendre map and Hamiltonian density for arbitrary ``G2``.

Nothing in this module is a physical validation.  ``FORMAL_PASS`` means "every rung this
repository can execute passed on the declared background"; the global positive-energy
theorem, the arbitrary-inhomogeneous principal symbol, and the derivation of the lift
itself all remain open, and every receipt says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .flrw_background import BackgroundCertificationError, certify_flrw_background
from .horndeski import generic_kessence_nonlinear_adm_legendre_control
from .principal_symbol import (
    analyze_isotropic_second_order_symbol,
    run_principal_symbol_controls,
)
from .scalar_tensor_pack import SCHEMA_VERSION as PACK_SCHEMA
from .scalar_tensor_pack import compile_scalar_tensor_pack
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-v3-family-formal-ladder-result-1.0"
REPRESENTATIVES_SCHEMA = "invariant-v3-family-representatives-1.0"

#: The v3 screen receipt this ladder consumes, and the derived full-representative set.
SCREEN_RECEIPT_PATH = "runs/gpu-baryonic-screen/screened-v3.json"
REPRESENTATIVES_PATH = "runs/gpu-baryonic-screen/v3-family-representatives.json"
LADDER_RECEIPT_PATH = "runs/gpu-baryonic-screen/v3-formal-ladder.json"

#: The committed polynomial scalar-tensor IR the FLRW certifier consumes.  Its function
#: family ``g2 = x + c20 x^2 + c11 u x + c02 u^2``, ``g3 = d10 x + d01 u``,
#: ``g4 = (m2 + 2 a10 x + 2 a20 x^2 + 2 a01 u)/2`` covers both v3 screening sectors exactly.
POLYNOMIAL_IR_PATH = "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
POLYNOMIAL_IR_FILE_SHA256 = "abe74c8d95202b65748781f1793e78e5dfc55007ee3384d0fea06dc760c35bd8"
POLYNOMIAL_IR_CONTENT_SHA256 = (
    "5f59a96982ada839c59632e240c8d882ca59ccc809c77e81541622078d3a2667"
)

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.
CLAIMS: dict[str, bool] = {
    "corpus_absence_establishes_novelty": False,
    "first_principles_derivation_pending": True,
    "formal_pass_is_not_physical_validation": True,
    "real_data_used": False,
    "synthetic_controls_only": True,
}

#: Typed blocker codes.  Every one names the adapter that must be built; none of them is a
#: judgement about the physics, and none of them may ever be reported as a pass.
BLOCKERS: dict[str, dict[str, str]] = {
    "missing_adapter:nonlocal_fractional_operator": {
        "component_mechanism": "nonlocal_propagator_correction",
        "why": (
            "the lift's power arm is the static Green's function of (-Box)^alpha with "
            "alpha = 1 - t/2 together with a (s/L2)^p (1+s/L2)^-(p+t) UV form factor.  No "
            "action IR in this tree admits a fractional or form-factored d'Alembertian: "
            "action_ir's grammar is a closed local term library, scalar_tensor_pack's "
            "parser admits only local u,x functions, and no principal-symbol, ADM, or "
            "Dirac module accepts a nonlocal kernel"
        ),
        "adapter_to_build": (
            "a nonlocal/fractional-operator action IR with its own principal symbol, "
            "Cauchy formulation, and positivity contract (auxiliary-field localization of "
            "(-Box)^alpha, or a spectral-representation adapter)"
        ),
    },
    "missing_adapter:aqual_nu_to_kessence_inversion": {
        "component_mechanism": "pointwise_modified_dynamics",
        "why": (
            "the lift declares a pointwise nu_loc factor on g_N and names the AQUAL/"
            "k-essence lane, but it does not pin the K(X) whose static limit reproduces "
            "that nu.  The screen fixes nu, not the covariant function, so materializing a "
            "K here would be inventing the ansatz rather than reading it"
        ),
        "adapter_to_build": (
            "an inversion adapter nu(y) -> K(X) with an exactness certificate on the "
            "static branch, plus the branch-uniqueness argument that makes it well posed"
        ),
    },
    "missing_adapter:direct_scalar_matter_coupling": {
        "component_mechanism": "massive_scalar_exchange",
        "why": (
            "every lift arm couples the scalar directly to the source (S_int = g phi rho, "
            "phi T/Mpl).  configs/covariant_field_contract.json forbids it by rule -- every "
            "matter species is minimally coupled to g_mu_nu and to no candidate "
            "gravitational field -- and formal_backend.validate_covariant_action_spec "
            "enforces that, so the fifth-force sector cannot be written down at all"
        ),
        "adapter_to_build": (
            "a conformal/disformal matter-frame adapter that carries the coupling as a "
            "metric redefinition with its own PPN and equivalence-principle certificate"
        ),
    },
    "missing_adapter:cubic_g3_uniform_weak_field_cone": {
        "component_mechanism": "vainshtein_kinetic_braiding",
        "why": (
            "with canonical G3 != 0 the family leaves the generalized-harmonic k-essence "
            "class.  scalar_tensor_pack routes it to the cubic-Horndeski BSSN/CCZ4 "
            "weak-field theorem, whose own control records that the source supplies no "
            "universal numeric threshold for its 'much less than' hypothesis, and the FLRW "
            "certifier returns modified_harmonic_uniform_bound_required"
        ),
        "adapter_to_build": (
            "a candidate-specific uniform weak-field threshold and common scalar/metric "
            "cone bound for the declared braiding coefficient, or a modified-harmonic "
            "symmetrizer for cubic G3"
        ),
    },
}

#: Blocker precedence for the headline verdict: the arm that carries the kernel physics is
#: named first, then the other universal arms, then the sector-specific hyperbolicity route.
#: Alphabetical order would report an incidental blocker as the headline.
BLOCKER_ORDER: tuple[str, ...] = (
    "missing_adapter:nonlocal_fractional_operator",
    "missing_adapter:aqual_nu_to_kessence_inversion",
    "missing_adapter:direct_scalar_matter_coupling",
    "missing_adapter:cubic_g3_uniform_weak_field_cone",
)

#: Repository-wide bottlenecks that no family can clear because no code path returns a pass
#: for them.  They are attached to every verdict so that a FORMAL_PASS cannot be misread.
OUTSTANDING_REPO_BLOCKERS: tuple[str, ...] = (
    "global_positive_energy_on_general_nonmaximal_data_unresolved",
    "arbitrary_inhomogeneous_background_principal_symbol_unresolved",
    "first_principles_derivation_of_the_lift_pending",
    "background_domain_forward_invariance_under_nonlinear_evolution_unresolved",
)

#: The declared covariant sector each screening family reduces to, read off the lift text.
#:
#: acceleration -- ``L = -(dphi)^2/2 - (Box phi)(dphi)^2/Lambda^3 + phi T/Mpl``.  With the
#: pack normalization ``x = -nabla_phi_squared/(2 Lambda^4)`` the kinetic term is
#: ``Lambda^4 x`` so ``g2 = x``; the cubic term is ``+2 Lambda x Box(phi)`` and the pack
#: writes L3 as ``-Lambda g3 Box(phi)``, hence ``g3 = -2x`` exactly.  No free coefficient:
#: the lift pins it.
#:
#: curvature -- ``L = M^4 K(X)`` with ``K`` nonlinear at large ``X``.  The screen does *not*
#: pin ``K``, so the class representative is this tree's own convex-G2 k-essence cell
#: ``g2 = x + c_K x^2`` with ``c_K`` symbolic; the declared cell values are the repository's
#: existing G2 seeds ``c_K in {1/8, 1/4}``.  Verdicts are reported as conditions on ``c_K``.
SECTOR_ANSATZ: dict[str, dict[str, Any]] = {
    "acceleration": {
        "sector_id": "cubic_galileon_kinetic_braiding",
        "declared_lagrangian": "-(dphi)^2/2 - (Box phi)(dphi)^2/Lambda^3 + phi T/Mpl",
        "functions": {"g2": "x", "g3": "-2*x", "g4": "1/2"},
        "coefficients": [],
        "polynomial_coefficients": {"c20": "0", "d10": "-2"},
        "parameter_cells": [{}],
        "normalization_note": (
            "g3 = -2x is forced by the declared Lagrangian in the pack convention "
            "L3 = -Lambda_phi g3 Box(phi) with x = -nabla_phi_squared/(2 Lambda_phi^4)"
        ),
    },
    "curvature": {
        "sector_id": "kmouflage_convex_kessence",
        "declared_lagrangian": "M^4 K(X) with K nonlinear at large X",
        "functions": {"g2": "x + c_K*x**2", "g3": "0", "g4": "1/2"},
        "coefficients": ["c_K"],
        "polynomial_coefficients": {"c20": "c_K", "d10": "0"},
        "parameter_cells": [{"c_K": "1/8"}, {"c_K": "1/4"}],
        "normalization_note": (
            "the screen pins the k-essence class, not K itself; c_K is symbolic and the "
            "declared cell values 1/8 and 1/4 are this tree's existing convex-G2 seeds"
        ),
    },
}

#: One frozen on-shell FLRW run config shape for *every* family and every control.  It is
#: never tuned per family: only the coefficient assignment and the constraint-solved initial
#: Hubble rate change.  The canonical-scalar control must certify under it or the run aborts.
FLRW_RUN: dict[str, str] = {
    "schema_version": "sigma-flrw-background-run-1.0",
    "initial_u": "0",
    "initial_x": "3/100",
    "initial_radius": "1e-14",
    "tau_start": "0",
    "tau_end": "1/50",
    "step": "1/400",
    "precision_digits": "50",
    "determinant_floor": "1e-12",
    "health_margin": "1e-10",
    "constraint_tolerance": "1e-5",
    "picard_iterations": "20",
    "inflation_absolute": "1e-18",
    "inflation_relative": "1/10",
    "cubic_bssn_slicing_parameter": "1",
}

#: The ladder, in order.  ``stop_on`` is the verdict rule: the first rung that rejects ends
#: the ladder, and the first rung that blocks ends it too (a blocked rung cannot be walked
#: past without inventing evidence for the rungs above it).
LADDER_RUNGS: tuple[tuple[str, str], ...] = (
    ("ghost_freedom", "scalar kinetic sign: k-essence Legendre Jacobian and on-shell G_S"),
    ("gradient_stability", "c_s^2 >= 0: k-essence G2_X and on-shell F_S"),
    ("tensor_sector", "G_T > 0, F_T > 0 and c_T^2 = F_T/G_T finite"),
    (
        "principal_symbol_hyperbolicity",
        "formulation route plus the reduced isotropic principal-symbol decision",
    ),
    (
        "positive_energy_hamiltonian",
        "pointwise k-essence energy density and strict Legendre convexity",
    ),
)

#: Known-answer control actions, run through this module's own ladder every time.
CONTROL_ACTIONS: dict[str, dict[str, Any]] = {
    "canonical_scalar": {
        "expect": "FORMAL_PASS",
        "role": "positive",
        "functions": {"g2": "x", "g3": "0", "g4": "1/2"},
        "coefficients": [],
        "polynomial_coefficients": {"c20": "0", "d10": "0"},
        "why": "a canonical massless scalar in GR must clear every implemented rung",
    },
    "wrong_sign_ghost": {
        "expect": "FORMAL_REJECT:ghost_freedom",
        "role": "negative",
        "functions": {"g2": "x - 8*x**2", "g3": "0", "g4": "1/2"},
        "coefficients": [],
        "polynomial_coefficients": {"c20": "-8", "d10": "0"},
        "why": (
            "G2_X + 2X G2_XX = 1 - 48x is negative on the declared branch point, so the "
            "scalar is a ghost; the FLRW certifier independently rejects on G_S"
        ),
    },
    "tensor_ghost": {
        "expect": "FORMAL_REJECT:tensor_sector",
        "role": "negative",
        "functions": {"g2": "x", "g3": "0", "g4": "-1/2"},
        "coefficients": [],
        "polynomial_coefficients": {"c20": "0", "d10": "0", "m2": "-1"},
        "why": "a negative Einstein-Hilbert coefficient makes G_T = F_T = -1 a tensor ghost",
    },
    "superluminal_kessence": {
        "expect": "FORMAL_REJECT:principal_symbol_hyperbolicity",
        "role": "negative",
        "functions": {"g2": "x - x**2/8", "g3": "0", "g4": "1/2"},
        "coefficients": [],
        "polynomial_coefficients": {"c20": "-1/8", "d10": "0"},
        "why": (
            "a concave G2 keeps both signs positive at the declared point but drives "
            "c_s^2 = (1 - x/4)/(1 - 3x/4) above the cone policy bound"
        ),
    },
}


class V3FormalLadderError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact helpers.  No float ever reaches a receipt.
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    """Render a numeric as an exact decimal string; floats become fixed 9-digit text."""

    if isinstance(value, bool):
        raise V3FormalLadderError("booleans are not numeric receipt values")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.9e}"
    return str(value)


def _rational(text: str) -> Fraction:
    return Fraction(str(text))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "content_sha256": canonical_sha256(body)}


def _no_floats(value: Any, path: str = "$") -> None:
    """Fail closed if any float survives into a receipt."""

    if isinstance(value, float):
        raise V3FormalLadderError(f"float in receipt at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _no_floats(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_floats(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------
# Inputs: the sealed v3 receipt and the full 71-family representative set.
# ---------------------------------------------------------------------------


def load_screen_receipt(root: str | Path) -> dict[str, Any]:
    """Load ``screened-v3.json`` and verify its own seal before anything is derived."""

    path = Path(root) / SCREEN_RECEIPT_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise V3FormalLadderError("screened-v3 receipt seal does not replay")
    if value.get("schema_version") != "invariant-gpu-screened-kernel-screen-result-1.0":
        raise V3FormalLadderError("unexpected screened-v3 schema")
    return value


def build_representatives(
    screen: Mapping[str, Any], families: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Seal the full family-representative set against the v3 receipt it was derived from.

    ``screened-v3.json`` reports only ``SYSTEM_CAPS["max_families_reported"] = 16`` of its
    71 equivalence families, so the remaining 55 representatives have to be recovered by
    replaying the screen with that report cap lifted.  The replay is only trusted when it
    reproduces the sealed receipt exactly on everything the cap does not touch, and that
    reproduction evidence travels with the derived artifact.
    """

    reported = list(screen["passer_families_reported"])
    if len(families) != int(screen["counts"]["passer_families"]):
        raise V3FormalLadderError("representative count does not match the sealed receipt")
    if [dict(item) for item in families[: len(reported)]] != reported:
        raise V3FormalLadderError("regenerated representatives do not reproduce the sealed 16")
    body = {
        "schema_version": REPRESENTATIVES_SCHEMA,
        "source": {
            "path": SCREEN_RECEIPT_PATH,
            "content_sha256": screen["content_sha256"],
            "config_sha256": screen["config_sha256"],
            "gate_config_sha256": screen["gate_config_sha256"],
            "screen_config_sha256": screen["screen_config_sha256"],
            "geometry_sha256": screen["geometry_sha256"],
            "environment_sha256": screen["environment_sha256"],
        },
        "derivation": {
            "method": "gpu_screened_kernel_screen.run_screen replay with the report cap lifted",
            "report_cap_override": "SYSTEM_CAPS['max_families_reported'] 16 -> unbounded",
            "why": (
                "the sealed receipt caps its reported family block at 16 of 71; every other "
                "block it emits is uncapped and is reproduced byte-for-byte by the replay"
            ),
            "reproduction_checks": [
                "counts block identical",
                "pareto_front identical",
                "screening_family_breakdown identical",
                "config/gate/screen/geometry/environment hashes identical",
                "the sealed 16 reported families are a prefix of the regenerated 71",
            ],
        },
        "counts": {
            "families": len(families),
            "reported_in_sealed_receipt": len(reported),
            "recovered_by_replay": len(families) - len(reported),
            "all_gate_passers": int(screen["counts"]["all_gate_passers"]),
        },
        "families": [
            {
                "size": int(item["size"]),
                "screening_family": str(item["screening_family"]),
                "representative_ordinal": int(item["representative_ordinal"]),
                "representative_values": dict(item["representative_values"]),
                "representative_formula": str(item["representative_formula"]),
                "covariant_lift_candidate": json.loads(
                    json.dumps(item["covariant_lift_candidate"])
                ),
            }
            for item in families
        ],
    }
    return _seal(body)


def load_representatives(root: str | Path, screen: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load the derived representative set and re-check its seal and its v3 binding."""

    path = Path(root) / REPRESENTATIVES_PATH
    if not path.exists():
        raise V3FormalLadderError(
            f"{REPRESENTATIVES_PATH} is missing; run --regenerate-representatives (needs the GPU)"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != REPRESENTATIVES_SCHEMA:
        raise V3FormalLadderError("unexpected representative schema")
    if value.get("content_sha256") != canonical_sha256(body):
        raise V3FormalLadderError("representative artifact seal does not replay")
    if value["source"]["content_sha256"] != screen["content_sha256"]:
        raise V3FormalLadderError("representative artifact is bound to a different v3 receipt")
    families = list(value["families"])
    if len(families) != int(screen["counts"]["passer_families"]):
        raise V3FormalLadderError("representative artifact does not carry every family")
    return families


def regenerate_family_representatives(root: str | Path, *, use_gpu: bool = True) -> dict[str, Any]:
    """Replay the v3 screen with the family report cap lifted and seal all 71 representatives.

    This is the only place in this module that mutates another module's frozen caps, it
    restores the cap in a ``finally``, and the artifact it writes records that it did so.
    """

    from . import gpu_screened_kernel_screen as screen_module

    screen = load_screen_receipt(root)
    original = screen_module.SYSTEM_CAPS["max_families_reported"]
    try:
        screen_module.SYSTEM_CAPS["max_families_reported"] = 1 << 20
        replay = screen_module.run_screen(use_gpu=use_gpu)
    finally:
        screen_module.SYSTEM_CAPS["max_families_reported"] = original
    for key in (
        "counts",
        "pareto_front",
        "screening_family_breakdown",
        "config_sha256",
        "gate_config_sha256",
        "screen_config_sha256",
        "geometry_sha256",
        "environment_sha256",
    ):
        if replay[key] != screen[key]:
            raise V3FormalLadderError(f"screen replay diverged from the sealed receipt at {key}")
    return build_representatives(screen, replay["passer_families_reported"])


# ---------------------------------------------------------------------------
# Step 1: materialize the covariant ansatz, or emit a typed blocker.
# ---------------------------------------------------------------------------


def classify_lift(family: Mapping[str, Any]) -> dict[str, Any]:
    """Split a family's declared lift into expressible and blocked components."""

    lift = family["covariant_lift_candidate"]
    screening = str(family["screening_family"])
    mechanisms = [str(component["mechanism"]) for component in lift["components"]]
    blockers: list[str] = []
    for code, blocker in sorted(BLOCKERS.items()):
        if code == "missing_adapter:cubic_g3_uniform_weak_field_cone":
            continue
        if blocker["component_mechanism"] in mechanisms:
            blockers.append(code)
    expressible = [
        mechanism
        for mechanism in mechanisms
        if mechanism in {"vainshtein_kinetic_braiding", "kmouflage_gradient_screening"}
    ]
    if screening not in SECTOR_ANSATZ:
        raise V3FormalLadderError(f"no declared covariant sector for screening {screening!r}")
    ordered = [code for code in BLOCKER_ORDER if code in blockers]
    return {
        "screening_family": screening,
        "declared_mechanisms": sorted(mechanisms),
        "full_lift_expressible": not blockers,
        "full_lift_blockers": ordered,
        "expressible_sector_mechanisms": sorted(expressible),
        "sector_id": SECTOR_ANSATZ[screening]["sector_id"],
        "ladder_scope": (
            "complete declared lift"
            if not blockers
            else "necessary condition on the declared screening sector only"
        ),
    }


def sector_action_ir(screening: str) -> dict[str, Any]:
    """Compile the declared screening sector into the repository's typed scalar-tensor IR."""

    ansatz = SECTOR_ANSATZ[screening]
    return compiled_pack(ansatz["functions"], ansatz["coefficients"])


_PACK_CACHE: dict[str, dict[str, Any]] = {}


def compiled_pack(
    functions: Mapping[str, str], coefficients: Sequence[str]
) -> dict[str, Any]:
    """``compile_scalar_tensor_pack`` with a process cache (each call runs ten controls)."""

    spec = {
        "schema_version": PACK_SCHEMA,
        "name": "v3 screened-gravity family sector",
        "normalization": {
            "u": "phi/Lambda_phi",
            "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
            "Lambda_phi_positive": True,
        },
        "coefficients": list(coefficients),
        "functions": dict(functions),
        "derivative_overrides": {},
        "mutation_axes": [],
    }
    key = canonical_sha256(spec)
    if key not in _PACK_CACHE:
        compiled = compile_scalar_tensor_pack(spec)
        if compiled["errors"]:
            raise V3FormalLadderError(f"sector pack failed to compile: {compiled['errors']}")
        _PACK_CACHE[key] = compiled
    return _PACK_CACHE[key]


# ---------------------------------------------------------------------------
# Step 2: the on-shell background certificate, shared by every rung.
# ---------------------------------------------------------------------------


def _polynomial_ir(root: str | Path) -> dict[str, Any]:
    path = Path(root) / POLYNOMIAL_IR_PATH
    if _file_sha256(path) != POLYNOMIAL_IR_FILE_SHA256:
        raise V3FormalLadderError("polynomial scalar-tensor IR file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != POLYNOMIAL_IR_CONTENT_SHA256:
        raise V3FormalLadderError("polynomial scalar-tensor IR content hash changed")
    return value


def _initial_hubble(coefficients: Mapping[str, str]) -> str | None:
    """Solve the FLRW energy constraint ``E(u, x, h) = 0`` exactly for the declared branch.

    For the sector family ``g2 = x + c20 x^2``, ``g3 = d10 x``, ``g4 = m2/2`` the constraint
    reduces to ``3 m2 h^2 - 6 x sqrt(2x) d10 h - (x + 3 c20 x^2) = 0``; the expanding root is
    taken.  The result is a 40-digit decimal string, and the certifier is given a 1e-14
    interval radius so the true irrational root is enclosed.  ``None`` means the declared
    coefficients admit no expanding on-shell branch at all -- a fact about the candidate, not
    an error, so it is reported as a typed background status rather than raised.
    """

    x = sp.Rational(_rational(FLRW_RUN["initial_x"]))
    c20 = sp.Rational(_rational(coefficients.get("c20", "0")))
    d10 = sp.Rational(_rational(coefficients.get("d10", "0")))
    m2 = sp.Rational(_rational(coefficients.get("m2", "1")))
    h = sp.Symbol("h", real=True)
    constraint = (
        2 * x * (1 + 2 * c20 * x)
        - (x + c20 * x**2)
        + 6 * x * sp.sqrt(2 * x) * h * d10
        - 3 * m2 * h**2
    )
    expanding: list[sp.Expr] = []
    for root in sp.solve(sp.Eq(constraint, 0), h):
        numeric = sp.N(root, 40)
        if not numeric.is_real or numeric <= 0:
            continue
        expanding.append(numeric)
    if not expanding:
        return None
    return str(max(expanding))


def certify_background(root: str | Path, coefficients: Mapping[str, str]) -> dict[str, Any]:
    """Run the frozen on-shell FLRW interval certificate for one coefficient assignment."""

    ir = _polynomial_ir(root)
    assignment = {
        "a01": "0",
        "a10": "0",
        "a20": "0",
        "c02": "0",
        "c11": "0",
        "c20": "0",
        "d01": "0",
        "d10": "0",
        "m2": "1",
        **{key: str(value) for key, value in coefficients.items()},
    }
    hubble = _initial_hubble(assignment)
    if hubble is None:
        return {
            "status": "no_expanding_on_shell_branch",
            "errors": [
                "the declared coefficients admit no positive on-shell Hubble root at x = "
                + FLRW_RUN["initial_x"]
            ],
        }
    config = {
        "schema_version": FLRW_RUN["schema_version"],
        "name": "v3 screened-gravity sector on-shell FLRW certificate",
        "coefficients": assignment,
        "initial_state": {
            "u": FLRW_RUN["initial_u"],
            "x": str(float(_rational(FLRW_RUN["initial_x"]))),
            "h": hubble,
        },
        "initial_radius": float(FLRW_RUN["initial_radius"]),
        "tau_start": float(_rational(FLRW_RUN["tau_start"])),
        "tau_end": float(_rational(FLRW_RUN["tau_end"])),
        "step": float(_rational(FLRW_RUN["step"])),
        "precision_digits": int(FLRW_RUN["precision_digits"]),
        "determinant_floor": float(FLRW_RUN["determinant_floor"]),
        "health_margin": float(FLRW_RUN["health_margin"]),
        "constraint_tolerance": float(FLRW_RUN["constraint_tolerance"]),
        "picard_iterations": int(FLRW_RUN["picard_iterations"]),
        "inflation_absolute": float(FLRW_RUN["inflation_absolute"]),
        "inflation_relative": float(_rational(FLRW_RUN["inflation_relative"])),
        "cubic_bssn_slicing_parameter": float(_rational(FLRW_RUN["cubic_bssn_slicing_parameter"])),
    }
    try:
        report = certify_flrw_background(ir, config)
    except BackgroundCertificationError as error:  # pragma: no cover - fail closed
        return {"status": "certifier_error", "errors": [str(error)]}
    return report


def _background_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    uniform = report.get("uniform_certificate") or {}
    formulation = report.get("formulation_certificate") or {}
    bounds = uniform.get("health_lower_bounds") or {}
    summary = {
        "status": str(report.get("status")),
        "errors": [str(item) for item in report.get("errors", [])],
        "source_ir_sha256": str(report.get("source_ir_sha256", "")),
        "content_sha256": str(report.get("content_sha256", "")),
        "health_lower_bounds": {name: _text(bounds[name]) for name in sorted(bounds)},
        "formulation_route": str(formulation.get("route", "")),
        "formulation_status": str(formulation.get("status", "")),
    }
    if "Theta_min_abs" in uniform:
        summary["Theta_min_abs"] = _text(uniform["Theta_min_abs"])
    if "evolution_determinant_min_abs" in uniform:
        summary["evolution_determinant_min_abs"] = _text(uniform["evolution_determinant_min_abs"])
    kessence = formulation.get("uniform_kessence_health_lower_bounds") or {}
    if kessence:
        summary["kessence_health_lower_bounds"] = {
            name: _text(kessence[name]) for name in sorted(kessence)
        }
    return summary


# ---------------------------------------------------------------------------
# Step 3: the rungs.
# ---------------------------------------------------------------------------


def _domain_symbols(coefficients: Sequence[str], *, constrain: bool = True) -> dict[str, sp.Symbol]:
    symbols = {
        "u": sp.Symbol("u", real=True),
        "x": sp.Symbol("x", positive=True),
        "h": sp.Symbol("h", positive=True),
        "h_tau": sp.Symbol("h_tau", negative=True),
        "x_tau": sp.Symbol("x_tau", real=True),
    }
    for name in coefficients:
        symbols[name] = sp.Symbol(name, positive=True) if constrain else sp.Symbol(name, real=True)
    return symbols


def _status_of(expression: str, coefficients: Sequence[str], *, constrain: bool) -> str:
    parsed = sp.sympify(expression, locals=_domain_symbols(coefficients, constrain=constrain))
    simplified = sp.simplify(parsed)
    if simplified.is_positive:
        return "pass"
    if simplified.is_nonpositive:
        return "reject"
    return "conditional"


def _sign_verdict(expression: str, coefficients: Sequence[str]) -> dict[str, Any]:
    """Decide the sign of a compiled expression over the declared domain, fail-closed.

    ``pass`` needs a symbolic proof of positivity on the whole declared domain; ``reject``
    needs a symbolic proof that it is non-positive there.  Anything else is ``conditional``
    and carries the exact inequality forward instead of guessing.

    When the sector carries free coefficients the sign is decided twice: once on the declared
    domain (which assumes the coefficients positive) and once with the coefficients merely
    real.  The second decision is what *derives* the parameter inequality, rather than
    letting the declared domain quietly assume it.
    """

    parsed = sp.sympify(expression, locals=_domain_symbols(coefficients))
    factored = sp.factor(parsed)
    verdict = {
        "expression": str(factored),
        "condition": f"{factored} > 0",
        "domain": "x > 0, h > 0, h_tau < 0" + (
            ", " + ", ".join(f"{name} > 0" for name in coefficients) if coefficients else ""
        ),
        "symbolic_status": _status_of(expression, coefficients, constrain=True),
    }
    if coefficients:
        verdict["unconstrained_status"] = _status_of(expression, coefficients, constrain=False)
        verdict["unconstrained_domain"] = "x > 0, h > 0, h_tau < 0, " + ", ".join(
            f"{name} real" for name in coefficients
        )
    return verdict


def _rung(name: str, status: str, reason: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    if status not in {"pass", "reject", "blocked"}:
        raise V3FormalLadderError(f"illegal rung status {status!r}")
    return {
        "rung": name,
        "status": status,
        "reason": reason,
        "evidence": json.loads(json.dumps(evidence)),
    }


def _combine(symbolic: Mapping[str, Any], certified: str | None, margin: str | None) -> str:
    """A rung passes only when the symbolic sign and the certified lower bound agree.

    A symbolic proof of non-positivity on the declared domain rejects on its own, and so
    does an interval certificate that drives this rung's health coefficient through its
    non-positive margin on the declared on-shell trajectory.  Everything else that is not a
    closed certificate is ``blocked`` -- never a pass.
    """

    if symbolic["symbolic_status"] == "reject":
        return "reject"
    if margin == "violated":
        return "reject"
    if certified != "pass_interval_certified":
        return "blocked"
    return "pass" if symbolic["symbolic_status"] in {"pass", "conditional"} else "blocked"


def run_ladder(
    root: str | Path, screening_or_control: str, *, control: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Run every executable rung for one declared sector or one control action."""

    if control is not None:
        functions = control["functions"]
        coefficients = list(control["coefficients"])
        cells = [dict(control["polynomial_coefficients"])]
        label = "control"
    else:
        ansatz = SECTOR_ANSATZ[screening_or_control]
        functions = ansatz["functions"]
        coefficients = list(ansatz["coefficients"])
        cells = []
        for cell in ansatz["parameter_cells"]:
            assignment = dict(ansatz["polynomial_coefficients"])
            for key, value in assignment.items():
                if value in cell:
                    assignment[key] = cell[value]
            cells.append(assignment)
        label = ansatz["sector_id"]

    pack = compiled_pack(functions, coefficients)
    backgrounds = [
        {"coefficients": dict(cell), "certificate": _background_summary(certify_background(root, cell))}
        for cell in cells
    ]
    statuses = {entry["certificate"]["status"] for entry in backgrounds}
    certified = "pass_interval_certified" if statuses == {"pass_interval_certified"} else None
    joined_errors = sorted(
        {error for entry in backgrounds for error in entry["certificate"]["errors"]}
    )

    def bound(name: str) -> str | None:
        values = [
            entry["certificate"]["health_lower_bounds"].get(name) for entry in backgrounds
        ]
        return min(values) if all(value is not None for value in values) else None

    kinetic = _sign_verdict(pack["compiled_kessence_kinetic"], coefficients)
    gradient = _sign_verdict(pack["compiled_kessence_gradient"], coefficients)
    tensor_g = _sign_verdict(pack["compiled_tensor_G_T"], coefficients)
    tensor_f = _sign_verdict(pack["compiled_tensor_F_T"], coefficients)
    energy = _sign_verdict(pack["compiled_kessence_homogeneous_energy_density"], coefficients)
    # The FLRW no-ghost and no-gradient-instability coefficients are the conditions that
    # actually bite on a background: they are certified numerically on the declared
    # trajectory above, and their symbolic sign over the whole declared domain is the exact
    # inequality a family must satisfy anywhere else.
    scalar_g_s = _sign_verdict(pack["compiled_scalar_G_S"], coefficients)
    scalar_f_s = _sign_verdict(pack["compiled_scalar_F_S"], coefficients)

    ghost_margin = "violated" if any(
        "G_S" in error or "Legendre" in error for error in joined_errors
    ) else None
    gradient_margin = "violated" if any("F_S" in error for error in joined_errors) else None
    tensor_margin = "violated" if any(
        "G_T" in error or "F_T" in error for error in joined_errors
    ) else None

    rungs: list[dict[str, Any]] = []
    rungs.append(
        _rung(
            "ghost_freedom",
            _combine(kinetic, certified, ghost_margin),
            "k-essence Legendre Jacobian G2_X + 2X G2_XX and the on-shell G_S lower bound",
            {
                "symbolic": kinetic,
                "background_condition": scalar_g_s,
                "certified_G_S_lower_bound": bound("G_S"),
                "background_errors": joined_errors,
            },
        )
    )
    rungs.append(
        _rung(
            "gradient_stability",
            _combine(gradient, certified, gradient_margin),
            "k-essence gradient coefficient G2_X and the on-shell F_S lower bound",
            {
                "symbolic": gradient,
                "background_condition": scalar_f_s,
                "compiled_scalar_speed_squared": pack["compiled_scalar_speed_squared"],
                "compiled_kessence_speed_squared": pack["compiled_kessence_speed_squared"],
                "certified_F_S_lower_bound": bound("F_S"),
            },
        )
    )
    tensor_status = "reject" if "reject" in {
        tensor_g["symbolic_status"],
        tensor_f["symbolic_status"],
    } else _combine(tensor_g, certified, tensor_margin)
    rungs.append(
        _rung(
            "tensor_sector",
            tensor_status,
            "G_T = 2(g4 - 2x g4_x) > 0, F_T = 2 g4 > 0 and c_T^2 = F_T/G_T finite",
            {
                "G_T": tensor_g,
                "F_T": tensor_f,
                "compiled_tensor_speed_squared": pack["compiled_tensor_speed_squared"],
                "certified_G_T_lower_bound": bound("G_T"),
                "certified_F_T_lower_bound": bound("F_T"),
            },
        )
    )

    route = {entry["certificate"]["formulation_route"] for entry in backgrounds}
    reduced = _reduced_principal_symbol(pack, cells, coefficients)
    if certified is None:
        hyperbolic_status = "blocked"
        hyperbolic_reason = "the on-shell background certificate did not close"
        hyperbolic_blocker = "missing_adapter:on_shell_background_certificate"
    elif route == {"generalized_harmonic_kessence"}:
        hyperbolic_status = "pass" if reduced["passed"] else "reject"
        hyperbolic_reason = (
            "the certified trajectory is generalized-harmonic eligible and the reduced "
            "isotropic principal symbol is hyperbolic inside the cone policy"
            if reduced["passed"]
            else "the reduced isotropic principal symbol fails on the certified trajectory"
        )
        hyperbolic_blocker = None
    else:
        hyperbolic_status = "blocked"
        hyperbolic_reason = (
            "canonical G3 != 0 leaves the generalized-harmonic class; the certifier returns "
            + ", ".join(sorted(route))
        )
        hyperbolic_blocker = "missing_adapter:cubic_g3_uniform_weak_field_cone"
    rungs.append(
        _rung(
            "principal_symbol_hyperbolicity",
            hyperbolic_status,
            hyperbolic_reason,
            {
                "formulation_routes": sorted(route),
                "formulation_statuses": sorted(
                    {entry["certificate"]["formulation_status"] for entry in backgrounds}
                ),
                "reduced_principal_symbol": reduced,
                "blocker": hyperbolic_blocker,
                "scope": (
                    "gauge/constraint-reduced isotropic scalar sector on the certified "
                    "trajectory; the arbitrary-inhomogeneous coupled metric-scalar symbol "
                    "is unresolved tree-wide"
                ),
            },
        )
    )

    legendre_passed, legendre = generic_kessence_nonlinear_adm_legendre_control()
    if not legendre_passed:
        raise V3FormalLadderError("the k-essence nonlinear Legendre control did not fire")
    rungs.append(
        _rung(
            "positive_energy_hamiltonian",
            _combine(energy, certified, None),
            "pointwise k-essence energy density 2X G2_X - G2 and strict Legendre convexity",
            {
                "symbolic": energy,
                "hamiltonian_density": str(legendre["hamiltonian_density"]),
                "strict_convexity_condition": str(legendre["strict_convexity_condition"]),
                "certified_kessence_energy_lower_bound": _min_kessence_energy(backgrounds),
                "global_energy": "unresolved",
                "scope": (
                    "pointwise and homogeneous only; no adapter in this tree proves "
                    "E_ADM >= |P_ADM| on general nonmaximal data"
                ),
            },
        )
    )

    return {
        "label": label,
        "action_ir": {
            "schema_version": pack["schema_version"],
            "functions": dict(pack["functions"]),
            "normalized_action": pack["normalized_action"],
            "content_sha256": pack["content_sha256"],
            "status": pack["status"],
        },
        "parameter_cells": [dict(entry["coefficients"]) for entry in backgrounds],
        "backgrounds": backgrounds,
        "rungs": rungs,
    }


def _min_kessence_energy(backgrounds: Sequence[Mapping[str, Any]]) -> str | None:
    values = [
        entry["certificate"].get("kessence_health_lower_bounds", {}).get("kessence_energy_density")
        for entry in backgrounds
    ]
    return min(values) if values and all(value is not None for value in values) else None


def _reduced_principal_symbol(
    pack: Mapping[str, Any], cells: Sequence[Mapping[str, str]], coefficients: Sequence[str]
) -> dict[str, Any]:
    """Decide the reduced scalar principal symbol at the certified background point."""

    x = sp.Rational(_rational(FLRW_RUN["initial_x"]))
    results: list[dict[str, Any]] = []
    passed = True
    for cell in cells:
        substitution: dict[sp.Symbol, sp.Expr] = {sp.Symbol("x", positive=True): x}
        for name in coefficients:
            value = cell.get("c20" if name == "c_K" else name)
            if value is None:
                continue
            substitution[sp.Symbol(name, positive=True)] = sp.Rational(_rational(value))
        symbols = _domain_symbols(coefficients)
        kinetic = sp.simplify(sp.sympify(pack["compiled_kessence_kinetic"], locals=symbols).subs(substitution))
        gradient = sp.simplify(sp.sympify(pack["compiled_kessence_gradient"], locals=symbols).subs(substitution))
        if kinetic.free_symbols or gradient.free_symbols:
            return {
                "passed": False,
                "status": "unresolved_symbolic_block",
                "reason": "the reduced block still carries free symbols at the declared cell",
            }
        analysis = analyze_isotropic_second_order_symbol(
            sp.Matrix([[kinetic]]), sp.Matrix([[gradient]])
        )
        passed = passed and analysis.passed
        results.append(
            {
                "cell": dict(cell),
                "kinetic_matrix": str(sp.Matrix([[kinetic]])),
                "gradient_matrix": str(sp.Matrix([[gradient]])),
                "speed_squared": [str(value) for value in analysis.speed_squared],
                "ghost_free": bool(analysis.ghost_free),
                "gradient_stable": bool(analysis.gradient_stable),
                "real_characteristics": bool(analysis.real_characteristics),
                "strongly_hyperbolic": bool(analysis.strongly_hyperbolic),
                "cone_policy_pass": bool(analysis.cone_policy_pass),
                "passed": bool(analysis.passed),
            }
        )
    return {
        "passed": bool(passed),
        "status": "decided",
        "convention": "L2 = dot(u)^T K dot(u)/2 - partial_i(u)^T G partial_i(u)/2",
        "cone_policy": "0 <= c_mode^2 <= 1 relative to the physical metric cone",
        "cells": results,
    }


def ladder_verdict(materialization: Mapping[str, Any], rungs: Sequence[Mapping[str, Any]]) -> str:
    """FORMAL_PASS / FORMAL_REJECT:<rung> / BLOCKED:<code>, first failure wins."""

    for rung in rungs:
        if rung["status"] == "reject":
            return f"FORMAL_REJECT:{rung['rung']}"
    if materialization["full_lift_blockers"]:
        return f"BLOCKED:{materialization['full_lift_blockers'][0]}"
    for rung in rungs:
        if rung["status"] == "blocked":
            blocker = rung["evidence"].get("blocker") or "missing_adapter:unnamed"
            return f"BLOCKED:{blocker}"
    return "FORMAL_PASS"


def sector_verdict(rungs: Sequence[Mapping[str, Any]]) -> str:
    """The sector-only verdict: what the necessary-condition ladder alone concluded."""

    for rung in rungs:
        if rung["status"] == "reject":
            return f"SECTOR_REJECT:{rung['rung']}"
        if rung["status"] == "blocked":
            blocker = rung["evidence"].get("blocker") or "missing_adapter:unnamed"
            return f"SECTOR_BLOCKED:{rung['rung']}:{blocker}"
    return "SECTOR_PASS"


# ---------------------------------------------------------------------------
# Known-answer controls.  A run whose controls do not fire aborts.
# ---------------------------------------------------------------------------


def run_controls(root: str | Path) -> dict[str, Any]:
    """Replay the imported controls and drive this module's own ladder with known answers."""

    imported = run_principal_symbol_controls()
    if not imported["passed"]:
        raise V3FormalLadderError("imported principal-symbol controls did not pass")
    expected_negative = ("negative_kinetic_ghost", "negative_gradient", "superluminal_cone")
    for name in expected_negative:
        entry = imported["controls"].get(name)
        if entry is None or entry.get("passed"):
            raise V3FormalLadderError(f"negative control {name} did not fire")
    legendre_passed, legendre = generic_kessence_nonlinear_adm_legendre_control()
    if not legendre_passed:
        raise V3FormalLadderError("k-essence Legendre control did not fire")
    for name, entry in legendre["negative_controls"].items():
        if not entry.get("rejected"):
            raise V3FormalLadderError(f"Legendre negative control {name} did not reject")

    ladder_controls: dict[str, Any] = {}
    for name in sorted(CONTROL_ACTIONS):
        declared = CONTROL_ACTIONS[name]
        result = run_ladder(root, name, control=declared)
        materialization = {"full_lift_blockers": []}
        observed = ladder_verdict(materialization, result["rungs"])
        if observed != declared["expect"]:
            raise V3FormalLadderError(
                f"ladder control {name} expected {declared['expect']} but observed {observed}"
            )
        ladder_controls[name] = {
            "role": declared["role"],
            "why": declared["why"],
            "functions": dict(declared["functions"]),
            "expected_verdict": declared["expect"],
            "observed_verdict": observed,
            "rung_statuses": {rung["rung"]: rung["status"] for rung in result["rungs"]},
            "action_content_sha256": result["action_ir"]["content_sha256"],
        }
    return {
        "imported_principal_symbol_controls": {
            "passed": True,
            "positive": sorted(
                name
                for name, entry in imported["controls"].items()
                if isinstance(entry, dict) and entry.get("passed")
            ),
            "negative_fired": list(expected_negative),
        },
        "imported_kessence_legendre_control": {
            "passed": True,
            "negatives_rejected": sorted(legendre["negative_controls"]),
        },
        "ladder_controls": ladder_controls,
    }


# ---------------------------------------------------------------------------
# Aggregation and the receipt.
# ---------------------------------------------------------------------------


def _aggregate(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = list(verdicts)
    by_screening: dict[str, dict[str, Any]] = {}
    for entry in families:
        block = by_screening.setdefault(
            entry["screening_family"],
            {"families": 0, "candidates": 0, "verdicts": {}, "sector_verdicts": {}},
        )
        block["families"] += 1
        block["candidates"] += entry["size"]
        block["verdicts"][entry["verdict"]] = block["verdicts"].get(entry["verdict"], 0) + 1
        block["sector_verdicts"][entry["sector_verdict"]] = (
            block["sector_verdicts"].get(entry["sector_verdict"], 0) + 1
        )
    rung_eliminations = {name: 0 for name, _ in LADDER_RUNGS}
    rung_blocks = {name: 0 for name, _ in LADDER_RUNGS}
    for entry in families:
        for rung in entry["rungs"]:
            if rung["status"] == "reject":
                rung_eliminations[rung["rung"]] += 1
                break
            if rung["status"] == "blocked":
                rung_blocks[rung["rung"]] += 1
                break
    blocked_by_code: dict[str, int] = {}
    sector_blocked_by_code: dict[str, int] = {}
    for entry in families:
        for code in entry["materialization"]["full_lift_blockers"]:
            blocked_by_code[code] = blocked_by_code.get(code, 0) + 1
        for rung in entry["rungs"]:
            if rung["status"] != "blocked":
                continue
            code = rung["evidence"].get("blocker")
            if code:
                sector_blocked_by_code[code] = sector_blocked_by_code.get(code, 0) + 1
            break
    axis_correlation: dict[str, dict[str, dict[str, int]]] = {}
    for axis in ("screen", "local", "L1", "L2", "p", "t", "w_yukawa", "w_power"):
        rows: dict[str, dict[str, int]] = {}
        for entry in families:
            value = entry["representative_values"][axis]
            if axis == "screen":
                value = value.split(":")[0]
            row = rows.setdefault(str(value), {})
            row[entry["sector_verdict"]] = row.get(entry["sector_verdict"], 0) + 1
        axis_correlation[axis] = {key: rows[key] for key in sorted(rows)}
    survivors = [entry for entry in families if not entry["verdict"].startswith("FORMAL_REJECT")]
    eliminated_families = sorted(
        {
            entry["screening_family"]
            for entry in families
            if entry["verdict"].startswith("FORMAL_REJECT")
        }
    )
    fully_eliminated = [
        name
        for name in sorted(by_screening)
        if all(
            entry["verdict"].startswith("FORMAL_REJECT")
            for entry in families
            if entry["screening_family"] == name
        )
    ]
    return {
        "families_in": len(families),
        "not_eliminated": len(survivors),
        "eliminated": len(families) - len(survivors),
        "formal_pass": sum(1 for entry in families if entry["verdict"] == "FORMAL_PASS"),
        "blocked": sum(1 for entry in families if entry["verdict"].startswith("BLOCKED")),
        "sector_pass": sum(1 for entry in families if entry["sector_verdict"] == "SECTOR_PASS"),
        "per_rung_first_elimination": rung_eliminations,
        "per_rung_first_block": rung_blocks,
        "blocked_by_adapter": {key: blocked_by_code[key] for key in sorted(blocked_by_code)},
        "sector_blocked_by_adapter": {
            key: sector_blocked_by_code[key] for key in sorted(sector_blocked_by_code)
        },
        "by_screening_family": {key: by_screening[key] for key in sorted(by_screening)},
        "screening_families_with_any_elimination": eliminated_families,
        "screening_families_eliminated_entirely": fully_eliminated,
        "sector_verdict_by_kernel_axis": axis_correlation,
    }


def _surviving_lagrangians(verdicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The distinct surviving sector Lagrangians with the conditions they must satisfy."""

    survivors: dict[str, dict[str, Any]] = {}
    for entry in verdicts:
        if entry["verdict"].startswith("FORMAL_REJECT"):
            continue
        key = entry["sector_id"]
        block = survivors.setdefault(
            key,
            {
                "sector_id": key,
                "screening_families": set(),
                "families": 0,
                "candidates": 0,
                "declared_lagrangian": entry["sector"]["declared_lagrangian"],
                "normalized_functions": entry["sector"]["functions"],
                "normalization_note": entry["sector"]["normalization_note"],
                "action_content_sha256": entry["action_ir"]["content_sha256"],
                "sector_verdict": entry["sector_verdict"],
                "parameter_conditions": [],
                "background_conditions": [],
                "unconditional_results": [],
                "highest_rung_reached": entry["highest_rung_reached"],
            },
        )
        block["screening_families"].add(entry["screening_family"])
        block["families"] += 1
        block["candidates"] += entry["size"]
        for rung in entry["rungs"]:
            evidence = rung["evidence"]
            for key, bucket in (
                ("symbolic", "parameter_conditions"),
                ("background_condition", "background_conditions"),
            ):
                symbolic = evidence.get(key)
                if not symbolic:
                    continue
                unconstrained = symbolic.get("unconstrained_status", symbolic["symbolic_status"])
                if unconstrained == "conditional":
                    domain = symbolic.get("unconstrained_domain", symbolic["domain"])
                    text = f"{rung['rung']}: {symbolic['condition']} on {domain}"
                    if text not in block[bucket]:
                        block[bucket].append(text)
                elif key == "symbolic" and unconstrained == "pass":
                    text = f"{rung['rung']}: {symbolic['condition']} holds identically"
                    if text not in block["unconditional_results"]:
                        block["unconditional_results"].append(text)
    output = []
    for key in sorted(survivors):
        block = survivors[key]
        block["screening_families"] = sorted(block["screening_families"])
        block["parameter_conditions"] = sorted(block["parameter_conditions"])
        block["background_conditions"] = sorted(block["background_conditions"])
        block["unconditional_results"] = sorted(block["unconditional_results"])
        output.append(block)
    return output


def run_formal_ladder(root: str | Path) -> dict[str, Any]:
    """Run the full ladder over every family representative and build the sealed receipt."""

    root = Path(root).resolve()
    screen = load_screen_receipt(root)
    families = load_representatives(root, screen)
    controls = run_controls(root)

    sector_results: dict[str, dict[str, Any]] = {}
    verdicts: list[dict[str, Any]] = []
    for family in families:
        materialization = classify_lift(family)
        screening = materialization["screening_family"]
        if screening not in sector_results:
            sector_results[screening] = run_ladder(root, screening)
        result = sector_results[screening]
        rungs = result["rungs"]
        highest = "none"
        for rung in rungs:
            if rung["status"] != "pass":
                break
            highest = rung["rung"]
        verdicts.append(
            {
                "representative_ordinal": int(family["representative_ordinal"]),
                "screening_family": screening,
                "size": int(family["size"]),
                "representative_values": dict(family["representative_values"]),
                "representative_formula": str(family["representative_formula"]),
                "materialization": materialization,
                "sector_id": materialization["sector_id"],
                "sector": {
                    "declared_lagrangian": SECTOR_ANSATZ[screening]["declared_lagrangian"],
                    "functions": SECTOR_ANSATZ[screening]["functions"],
                    "normalization_note": SECTOR_ANSATZ[screening]["normalization_note"],
                },
                "action_ir": result["action_ir"],
                "parameter_cells": result["parameter_cells"],
                "background_certificates": result["backgrounds"],
                "rungs": rungs,
                "highest_rung_reached": highest,
                "sector_verdict": sector_verdict(rungs),
                "verdict": ladder_verdict(materialization, rungs),
                "outstanding_repo_blockers": list(OUTSTANDING_REPO_BLOCKERS),
            }
        )
    verdicts.sort(key=lambda entry: entry["representative_ordinal"])
    aggregate = _aggregate(verdicts)
    survivors = _surviving_lagrangians(verdicts)

    body = {
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "formal ladder over the 71 all-gate-passing equivalence families of the v3 "
            "screened-kernel gravity screen.  Every family's declared covariant lift is "
            "either materialized as a typed scalar-tensor action IR or given a typed "
            "missing-adapter blocker; the rungs that this tree can execute are then run on "
            "an interval-certified on-shell FLRW trajectory.  A pass is a formal "
            "necessary-condition result on synthetic controls, never a physical validation"
        ),
        "claims": CLAIMS,
        "assumptions": {
            "screening_sector_is_the_declared_ansatz": (
                "the sector Lagrangians are read off the lift's own field_theory_ansatz "
                "text; the acceleration sector is pinned exactly, the curvature sector only "
                "to the k-essence class, whose representative carries a symbolic coefficient"
            ),
            "background_is_declared_not_derived": (
                "health is certified on one frozen on-shell FLRW trajectory per parameter "
                "cell; no inhomogeneous or astrophysical background is certified"
            ),
            "sector_ladder_is_a_necessary_condition": (
                "for families whose complete lift is blocked, a sector rejection still "
                "eliminates the family, but a sector pass does not admit it"
            ),
            "cone_policy": "0 <= c_mode^2 <= 1 relative to the physical metric cone",
        },
        "inputs": {
            "screen_receipt": {
                "path": SCREEN_RECEIPT_PATH,
                "content_sha256": screen["content_sha256"],
                "passer_families": int(screen["counts"]["passer_families"]),
                "all_gate_passers": int(screen["counts"]["all_gate_passers"]),
            },
            "family_representatives": {
                "path": REPRESENTATIVES_PATH,
                "content_sha256": canonical_sha256(
                    {
                        key: item
                        for key, item in json.loads(
                            (root / REPRESENTATIVES_PATH).read_text(encoding="utf-8")
                        ).items()
                        if key != "content_sha256"
                    }
                ),
            },
            "polynomial_scalar_tensor_ir": {
                "path": POLYNOMIAL_IR_PATH,
                "file_sha256": POLYNOMIAL_IR_FILE_SHA256,
                "content_sha256": POLYNOMIAL_IR_CONTENT_SHA256,
            },
        },
        "config": {
            "ladder_rungs": [{"rung": name, "definition": text} for name, text in LADDER_RUNGS],
            "sector_ansatz": SECTOR_ANSATZ,
            "flrw_run": FLRW_RUN,
            "blockers": BLOCKERS,
            "outstanding_repo_blockers": list(OUTSTANDING_REPO_BLOCKERS),
        },
        "controls": controls,
        "counts": aggregate,
        "surviving_lagrangians": survivors,
        "families": verdicts,
        "decision": _decision(aggregate, survivors),
        "adapter_gap_report": _adapter_gap_report(aggregate),
    }
    body["config_sha256"] = canonical_sha256(body["config"])
    _no_floats(body)
    return _seal(body)


def _decision(aggregate: Mapping[str, Any], survivors: Sequence[Mapping[str, Any]]) -> str:
    blocked = aggregate["blocked"]
    total = aggregate["families_in"]
    eliminated = aggregate["eliminated"]
    sector_pass = aggregate["sector_pass"]
    distinct = len(survivors)
    return (
        f"BLOCKED-AT-MATERIALIZATION: {blocked} of {total} families cannot have their "
        f"complete declared lift written down in this tree at all -- every v3 lift carries a "
        f"nonlocal (-Box)^alpha arm, an unpinned AQUAL nu factor, and a direct scalar-matter "
        f"coupling the frozen field contract forbids.  The necessary-condition ladder over "
        f"the declared screening sector eliminates {eliminated} families outright and passes "
        f"{sector_pass}; the surviving sector Lagrangians collapse to {distinct} distinct "
        f"actions, so the ladder discriminates on the screening axis and on nothing else"
    )


def _adapter_gap_report(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "statement": (
            "the ladder's reach, not the families' health, is the binding constraint: no "
            "adapter in this tree accepts a nonlocal fractional operator, an AQUAL nu "
            "inversion, or a direct scalar-matter coupling, and cubic G3 has no "
            "candidate-specific hyperbolicity route"
        ),
        "adapters_to_build": [
            {
                "code": code,
                "families_blocked_at_materialization": aggregate["blocked_by_adapter"].get(
                    code, 0
                ),
                "families_blocked_at_a_ladder_rung": aggregate[
                    "sector_blocked_by_adapter"
                ].get(code, 0),
                "component_mechanism": BLOCKERS[code]["component_mechanism"],
                "why": BLOCKERS[code]["why"],
                "adapter_to_build": BLOCKERS[code]["adapter_to_build"],
            }
            for code in BLOCKER_ORDER
        ],
        "build_order": [
            "missing_adapter:nonlocal_fractional_operator",
            "missing_adapter:aqual_nu_to_kessence_inversion",
            "missing_adapter:direct_scalar_matter_coupling",
            "missing_adapter:cubic_g3_uniform_weak_field_cone",
        ],
        "why_that_order": (
            "the nonlocal arm is present in every one of the 71 lifts, so it gates every "
            "family; the AQUAL inversion is next because it is also universal; the matter "
            "coupling needs a field-contract amendment and therefore review; the cubic-G3 "
            "cone only gates the acceleration-screened sector"
        ),
    }


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, claim, control, and structural replay; fail closed."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise V3FormalLadderError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise V3FormalLadderError("receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise V3FormalLadderError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise V3FormalLadderError("config binding changed")
    _no_floats(body)
    inputs = value.get("inputs", {})
    screen = inputs.get("screen_receipt", {})
    if not isinstance(screen.get("content_sha256"), str) or len(screen["content_sha256"]) != 64:
        raise V3FormalLadderError("screen receipt binding is malformed")
    if inputs.get("polynomial_scalar_tensor_ir", {}).get("file_sha256") != (
        POLYNOMIAL_IR_FILE_SHA256
    ):
        raise V3FormalLadderError("polynomial IR binding changed")
    controls = value.get("controls", {}).get("ladder_controls", {})
    if set(controls) != set(CONTROL_ACTIONS):
        raise V3FormalLadderError("control set changed")
    for name, entry in sorted(controls.items()):
        if entry.get("observed_verdict") != CONTROL_ACTIONS[name]["expect"]:
            raise V3FormalLadderError(f"control {name} did not fire in the receipt")
    families = value.get("families", [])
    if len(families) != int(screen.get("passer_families", -1)):
        raise V3FormalLadderError("receipt does not carry every family")
    rung_names = [name for name, _ in LADDER_RUNGS]
    for entry in families:
        if [rung["rung"] for rung in entry["rungs"]] != rung_names:
            raise V3FormalLadderError("family ladder is not the declared rung sequence")
        verdict = entry.get("verdict", "")
        if verdict == "FORMAL_PASS":
            if entry["materialization"]["full_lift_blockers"]:
                raise V3FormalLadderError("a blocked family cannot be a FORMAL_PASS")
            if any(rung["status"] != "pass" for rung in entry["rungs"]):
                raise V3FormalLadderError("FORMAL_PASS with a non-passing rung")
        elif verdict.startswith("FORMAL_REJECT:"):
            if verdict.split(":", 1)[1] not in rung_names:
                raise V3FormalLadderError("rejection names an unknown rung")
        elif verdict.startswith("BLOCKED:"):
            if verdict.split(":", 1)[1] not in BLOCKERS:
                raise V3FormalLadderError("block names an unknown adapter")
        else:
            raise V3FormalLadderError(f"unknown verdict {verdict!r}")
    counts = value.get("counts", {})
    if counts.get("families_in") != len(families):
        raise V3FormalLadderError("aggregate family count does not replay")
    if counts.get("not_eliminated", 0) + counts.get("eliminated", 0) != len(families):
        raise V3FormalLadderError("survivor arithmetic does not close")
    # The aggregate is the part a reader quotes, so it must be recomputable from the family
    # list it claims to summarize.  A resealed receipt with a doctored headline fails here.
    if json.loads(json.dumps(_aggregate(families))) != counts:
        raise V3FormalLadderError("aggregate counts do not replay from the family list")
    if json.loads(json.dumps(_surviving_lagrangians(families))) != value.get(
        "surviving_lagrangians"
    ):
        raise V3FormalLadderError("surviving Lagrangians do not replay from the family list")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise V3FormalLadderError("refusing to overwrite an immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Formal ladder over the 71 screened-gravity families of v3."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument(
        "--regenerate-representatives",
        action="store_true",
        help="replay the GPU screen with the family report cap lifted (writes the derived set)",
    )
    parser.add_argument("--cpu", action="store_true", help="force the numpy screen path")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.regenerate_representatives:
        artifact = regenerate_family_representatives(root, use_gpu=not args.cpu)
        _write(artifact, root / REPRESENTATIVES_PATH)
        print(json.dumps({"families": artifact["counts"]["families"]}, indent=2))
        return 0
    if args.validate_checked:
        target = Path(args.output) if args.output else root / LADDER_RECEIPT_PATH
        validate_receipt(json.loads(target.read_text(encoding="utf-8")))
        return 0
    result = run_formal_ladder(root)
    if args.output:
        _write(result, args.output)
    print(
        json.dumps(
            {
                "families_in": result["counts"]["families_in"],
                "eliminated": result["counts"]["eliminated"],
                "not_eliminated": result["counts"]["not_eliminated"],
                "formal_pass": result["counts"]["formal_pass"],
                "blocked": result["counts"]["blocked"],
                "sector_pass": result["counts"]["sector_pass"],
                "blocked_by_adapter": result["counts"]["blocked_by_adapter"],
                "surviving_lagrangians": [
                    block["sector_id"] for block in result["surviving_lagrangians"]
                ],
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
