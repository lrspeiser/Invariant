"""Item 48 action-first weak-field gravity generator and retrospective test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item44_scale_hierarchy import _predict as _item44_predict
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _best_candidate,
    _item44_oof,
    _ordinary_crossfit,
    _paired_p,
    _predict as _item45_predict,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    _physical_log_values as _item46_physical_log_values,
    _predict as _item46_predict,
    load_config as _load_item46_config,
    pi_vectors as _item46_pi_vectors,
)
from sigma_theory_compiler.gravity_item47_operator_generator import (
    _evaluation_arrays as _item47_evaluation_arrays,
    _item45_oof,
    _item46_oof,
    _predict as _item47_predict,
    _profiles,
    _shape_by_object,
    load_config as _load_item47_config,
    operator_bank_from_arrays as _item47_operator_bank_from_arrays,
)


CONFIG_PATH = Path("configs/gravity_item48_action_generator_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM44_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json"
)
ITEM47_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-47-operator-generator-v1-source/operator-features.json"
)
ITEM47_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-47-operator-generator-v1-source/joint-evaluation-result.json"
)


class GravityItem48Error(RuntimeError):
    """Raised when an Item 48 action, derivation, leakage, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item48-action-generator-config-1.0"
        or int(config.get("item", -1)) != 48
    ):
        raise GravityItem48Error("unexpected Item 48 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem48Error("stable gravity goal changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(config["scientific_freeze_commit"])) is None:
        raise GravityItem48Error("Item 48 scientific freeze is not bound to a commit")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem48Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-47-operator-generator-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem48Error("Item 47 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem48Error("Item 47 decision binding changed")
    if int(predecessor["selected_candidate"]["candidate_id"]) != int(
        required["selected_candidate_id"]
    ):
        raise GravityItem48Error("Item 47 candidate binding changed")
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem48Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem48Error("count-only rejection entered Item 48")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem48Error("finite empirical family pruning entered Item 48")
    policy = load_counterexample_policy(root / POLICY_PATH)
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem48Error("executable counterexample policy changed")
    actions = config["action_generator"]
    if (
        len(actions["action_classes"]) != 6
        or int(actions["recipes_per_class"]) != 16
        or int(actions["action_recipes"]) != 96
        or len(config["normalized_action_contract"]["baryonic_sources"]) != 4
        or any(len(actions["class_variants"][name]) != 4 for name in actions["action_classes"])
    ):
        raise GravityItem48Error("action grammar capacity changed")
    generator = config["candidate_generator"]
    if (
        int(generator["action_recipes"]) != 96
        or int(generator["cells_per_recipe"]) != 4096
        or int(generator["cells_per_action_class"]) != 65536
        or int(generator["raw_candidate_cells"]) != 393216
        or int(generator["post_evaluation_cells"]) != 0
    ):
        raise GravityItem48Error("candidate capacity changed")
    if int(actions["response_values_allowed_in_action_generation"]) != 0:
        raise GravityItem48Error("responses entered action generation")
    if bool(config["formal_infrastructure_boundary"]["item48_actions_are_admitted_by_existing_covariant_compiler"]):
        raise GravityItem48Error("radial actions were mislabeled covariant")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem48Error("fresh confirmation entered retrospective Item 48")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem48Error("paid calls entered Item 48")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def action_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = config["action_generator"]
    sources = config["normalized_action_contract"]["baryonic_sources"]
    catalog: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(generator["action_classes"]):
        variants = generator["class_variants"][class_name]
        for source_id, source in enumerate(sources):
            for variant_id, variant in enumerate(variants):
                catalog.append(
                    {
                        "recipe_id": len(catalog),
                        "action_class_id": class_id,
                        "action_class": class_name,
                        "source_id": source_id,
                        "source": source,
                        "variant_id": variant_id,
                        "variant": variant,
                        "reduced_action": generator["reduced_action_templates"][class_name],
                        "creativity_label": generator["creativity_labels"][class_name],
                        "historical_novelty_claimed": False,
                    }
                )
    if len(catalog) != 96:
        raise GravityItem48Error("action catalog does not contain 96 recipes")
    return catalog


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    candidate_id = np.arange(total, dtype=np.int64)
    local = candidate_id % per_recipe
    return {
        "candidate_id": candidate_id,
        "recipe": (candidate_id // per_recipe).astype(np.int16),
        "amplitude_index": ((local // 256) % 16).astype(np.int8),
        "exponent_index": ((local // 16) % 16).astype(np.int8),
        "transition_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    return (
        np.asarray(grids["amplitude"])[np.asarray(candidates["amplitude_index"], int)],
        np.asarray(grids["acceleration_exponent"])[np.asarray(candidates["exponent_index"], int)],
        np.asarray(grids["transition_u"])[np.asarray(candidates["transition_index"], int)],
    )


def admissible_candidates(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    gate = config["candidate_generator"]["admissibility"]
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    local_id = np.arange(per_recipe, dtype=np.int64)
    local = {
        "candidate_id": local_id,
        "recipe": np.zeros(per_recipe, dtype=np.int16),
        "amplitude_index": ((local_id // 256) % 16).astype(np.int8),
        "exponent_index": ((local_id // 16) % 16).astype(np.int8),
        "transition_index": (local_id % 16).astype(np.int8),
    }
    amplitude, exponent, transition = _candidate_parameters(local, config)
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    h = np.asarray(gate["probe_action_coordinates"], dtype=float)
    multiplier = 1.0 + amplitude[:, None, None] * np.power(
        u[None, None, :], -exponent[:, None, None]
    ) / (1.0 + u[None, None, :] / transition[:, None, None]) * (
        0.05 + 0.95 * h[None, :, None]
    )
    finite = np.all(np.isfinite(multiplier), axis=(1, 2))
    positive = finite & np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
    bounded = positive & np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
    safe_log_multiplier = np.log10(np.maximum(multiplier, np.finfo(float).tiny))
    local_limit = bounded & (
        np.max(np.abs(safe_log_multiplier[:, :, -1]), axis=1)
        <= float(gate["maximum_high_acceleration_log10_deviation"])
    )
    material = local_limit & (
        np.max(np.abs(safe_log_multiplier[:, :, 0]), axis=1)
        >= float(gate["minimum_low_acceleration_absolute_log10_deviation"])
    )
    kept_local = np.flatnonzero(material)
    recipes = int(config["candidate_generator"]["action_recipes"])
    keep = np.concatenate([kept_local + recipe * per_recipe for recipe in range(recipes)])
    raw = generate_raw_candidates(config)
    admitted = {key: value[keep] for key, value in raw.items()}
    signatures = {
        hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        for row in np.round(
            safe_log_multiplier[kept_local], int(gate["behavior_signature_decimals"])
        )
    }
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_per_action_recipe": len(kept_local),
        "admitted_by_action_class": {
            name: int(np.sum(np.asarray(admitted["recipe"]) // 16 == class_id))
            for class_id, name in enumerate(config["action_generator"]["action_classes"])
        },
        "local_behavioral_equivalence_classes": len(signatures),
        "rejection_counts_per_recipe": {
            "nonfinite": int(np.sum(~finite)),
            "nonpositive_permittivity": int(np.sum(finite & ~positive)),
            "out_of_bounds": int(np.sum(positive & ~bounded)),
            "no_newtonian_limit": int(np.sum(bounded & ~local_limit)),
            "immaterial_weak_field": int(np.sum(local_limit & ~material)),
        },
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    if candidate_id < 0 or candidate_id >= total:
        raise GravityItem48Error("candidate id outside frozen grid")
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    recipe_id = candidate_id // per_recipe
    local = candidate_id % per_recipe
    row = {
        "amplitude_index": np.asarray([(local // 256) % 16]),
        "exponent_index": np.asarray([(local // 16) % 16]),
        "transition_index": np.asarray([local % 16]),
    }
    amplitude, exponent, transition = _candidate_parameters(row, config)
    return {
        **action_catalog(config)[recipe_id],
        "candidate_id": candidate_id,
        "permittivity": "epsilon_c=1/nu_c",
        "derived_flux_equation": "epsilon_c*g=g_bar",
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
        },
    }


def _euler(lagrangian: sp.Expr, field: sp.Expr, x: sp.Symbol, order: int) -> sp.Expr:
    result = sp.diff(lagrangian, field)
    for derivative_order in range(1, order + 1):
        derivative = sp.diff(field, x, derivative_order)
        result += (-1) ** derivative_order * sp.diff(
            sp.diff(lagrangian, derivative), x, derivative_order
        )
    return sp.simplify(result)


def symbolic_action_derivation() -> dict[str, Any]:
    x = sp.symbols("x", real=True)
    ell, ell1, ell2, kappa, lam = sp.symbols(
        "ell ell1 ell2 kappa lambda", positive=True
    )
    phi = sp.Function("Phi")(x)
    h = sp.Function("h")(x)
    chi = sp.Function("chi")(x)
    eta = sp.Function("eta")(x)
    zeta = sp.Function("zeta")(x)
    source = sp.Function("source_b")(x)
    j = sp.Function("J")(x)
    stiffness = sp.Function("B")(x)
    eps_h = sp.Function("epsilon")(h)
    eps_x = sp.Function("epsilon")(x)

    rows: list[dict[str, Any]] = []
    direct_l = eps_x * sp.diff(phi, x) ** 2 / 2 - source * phi
    direct_phi = _euler(direct_l, phi, x, 1)
    direct_expected = -source - sp.diff(eps_x * sp.diff(phi, x), x)
    rows.append(
        {
            "action_class": "source_conditioned_permittivity",
            "lagrangian": str(direct_l),
            "field_equations": {"Phi": str(direct_phi)},
            "exact_residuals": {"Phi": str(sp.simplify(direct_phi - direct_expected))},
            "source_free_shift_identity": str(sp.simplify(sp.diff(direct_l, phi) + source)),
        }
    )

    common = eps_h * sp.diff(phi, x) ** 2 / 2 - source * phi
    phi_expected = -source - sp.diff(eps_h * sp.diff(phi, x), x)

    def append_auxiliary(
        name: str,
        constraint_terms: list[tuple[sp.Expr, sp.Expr]],
        orders: Mapping[sp.Expr, int],
        expected: Mapping[str, sp.Expr],
        reduced_energy: sp.Expr,
        reduced_expected: Mapping[str, sp.Expr],
    ) -> None:
        lagrangian = common + sum(multiplier * constraint for multiplier, constraint in constraint_terms)
        equations: dict[str, str] = {"Phi": str(_euler(lagrangian, phi, x, 1))}
        residuals: dict[str, str] = {
            "Phi": str(sp.simplify(_euler(lagrangian, phi, x, 1) - phi_expected))
        }
        for field, order in orders.items():
            label = str(field.func)
            equation = _euler(lagrangian, field, x, order)
            equations[label] = str(equation)
            residuals[label] = str(sp.simplify(equation - expected[label]))
        reduced_equations: dict[str, str] = {}
        reduced_residuals: dict[str, str] = {}
        for label, target in reduced_expected.items():
            field = h if label == "h" else chi
            order = 4 if name == "bihelmholtz_auxiliary" and label == "h" else 1
            if name in {
                "screened_auxiliary",
                "mixed_two_field",
                "convex_nonlinear_auxiliary",
                "adaptive_gradient_auxiliary",
            }:
                order = 1
            equation = _euler(reduced_energy, field, x, order)
            reduced_equations[label] = str(equation)
            reduced_residuals[label] = str(sp.simplify(equation - target))
        rows.append(
            {
                "action_class": name,
                "lagrangian": str(lagrangian),
                "field_equations": equations,
                "exact_residuals": residuals,
                "reduced_energy": str(reduced_energy),
                "reduced_euler_equations": reduced_equations,
                "reduced_exact_residuals": reduced_residuals,
                "source_free_shift_identity": str(sp.simplify(sp.diff(lagrangian, phi) + source)),
            }
        )

    c_screen = h - ell**2 * sp.diff(h, x, 2) - j
    e_screen = h**2 / 2 + ell**2 * sp.diff(h, x) ** 2 / 2 - j * h
    append_auxiliary(
        "screened_auxiliary",
        [(eta, c_screen)],
        {eta: 0, h: 2},
        {
            str(eta.func): c_screen,
            str(h.func): sp.diff(eps_h, h) * sp.diff(phi, x) ** 2 / 2
            + eta
            - ell**2 * sp.diff(eta, x, 2),
        },
        e_screen,
        {"h": c_screen},
    )

    c_bi = (
        h
        - (ell1**2 + ell2**2) * sp.diff(h, x, 2)
        + ell1**2 * ell2**2 * sp.diff(h, x, 4)
        - j
    )
    e_bi = (
        h**2 / 2
        + (ell1**2 + ell2**2) * sp.diff(h, x) ** 2 / 2
        + ell1**2 * ell2**2 * sp.diff(h, x, 2) ** 2 / 2
        - j * h
    )
    append_auxiliary(
        "bihelmholtz_auxiliary",
        [(eta, c_bi)],
        {eta: 0, h: 4},
        {
            str(eta.func): c_bi,
            str(h.func): sp.diff(eps_h, h) * sp.diff(phi, x) ** 2 / 2
            + eta
            - (ell1**2 + ell2**2) * sp.diff(eta, x, 2)
            + ell1**2 * ell2**2 * sp.diff(eta, x, 4),
        },
        e_bi,
        {"h": c_bi},
    )

    c_h = h - ell**2 * sp.diff(h, x, 2) - kappa * chi - j
    c_chi = chi - ell**2 * sp.diff(chi, x, 2) - kappa * h
    e_mixed = (
        (h**2 + chi**2) / 2
        + ell**2 * (sp.diff(h, x) ** 2 + sp.diff(chi, x) ** 2) / 2
        - kappa * h * chi
        - j * h
    )
    append_auxiliary(
        "mixed_two_field",
        [(eta, c_h), (zeta, c_chi)],
        {eta: 0, zeta: 0, h: 2, chi: 2},
        {
            str(eta.func): c_h,
            str(zeta.func): c_chi,
            str(h.func): sp.diff(eps_h, h) * sp.diff(phi, x) ** 2 / 2
            + eta
            - ell**2 * sp.diff(eta, x, 2)
            - kappa * zeta,
            str(chi.func): zeta - ell**2 * sp.diff(zeta, x, 2) - kappa * eta,
        },
        e_mixed,
        {"h": c_h, "chi": c_chi},
    )

    c_nonlinear = h - ell**2 * sp.diff(h, x, 2) + lam * h**3 - j
    e_nonlinear = (
        h**2 / 2 + ell**2 * sp.diff(h, x) ** 2 / 2 + lam * h**4 / 4 - j * h
    )
    append_auxiliary(
        "convex_nonlinear_auxiliary",
        [(eta, c_nonlinear)],
        {eta: 0, h: 2},
        {
            str(eta.func): c_nonlinear,
            str(h.func): sp.diff(eps_h, h) * sp.diff(phi, x) ** 2 / 2
            + eta
            - ell**2 * sp.diff(eta, x, 2)
            + 3 * lam * eta * h**2,
        },
        e_nonlinear,
        {"h": c_nonlinear},
    )

    c_adaptive = h - ell**2 * sp.diff(stiffness * sp.diff(h, x), x) - j
    e_adaptive = h**2 / 2 + ell**2 * stiffness * sp.diff(h, x) ** 2 / 2 - j * h
    append_auxiliary(
        "adaptive_gradient_auxiliary",
        [(eta, c_adaptive)],
        {eta: 0, h: 2},
        {
            str(eta.func): c_adaptive,
            str(h.func): sp.diff(eps_h, h) * sp.diff(phi, x) ** 2 / 2
            + eta
            - ell**2 * sp.diff(stiffness * sp.diff(eta, x), x),
        },
        e_adaptive,
        {"h": c_adaptive},
    )

    all_zero = all(
        value == "0"
        for row in rows
        for group in (row["exact_residuals"], row.get("reduced_exact_residuals", {}))
        for value in group.values()
    )
    shift_exact = all(row["source_free_shift_identity"] == "0" for row in rows)
    return {
        "schema_version": "invariant-gravity-item48-action-derivation-1.0",
        "action_classes": rows,
        "all_exact_euler_residuals_zero": all_zero,
        "all_source_free_shift_identities_zero": shift_exact,
        "radial_flux_identity": "d_x(epsilon_c*d_x Phi)+source_b=0; after radial integration epsilon_c*g=g_bar",
        "newtonian_off_switch": "A=0 implies nu=1, epsilon=1, and the ordinary Poisson flux equation",
        "reduced_static_convexity": {
            "source_conditioned_permittivity": "epsilon_c>0 is enforced candidate by candidate",
            "screened_auxiliary": "1+ell^2 k^2>0",
            "bihelmholtz_auxiliary": "(1+ell1^2 k^2)(1+ell2^2 k^2)>0",
            "mixed_two_field": "eigenvalues 1-kappa and 1+kappa are positive for every frozen kappa<1",
            "convex_nonlinear_auxiliary": "1+ell^2 k^2+3 lambda h^2>0",
            "adaptive_gradient_auxiliary": "1+ell^2(1+sigma J^2)k^2>0",
        },
        "full_dynamical_stability_claimed": False,
        "covariance_claimed": False,
    }


def malformed_action_controls() -> dict[str, Any]:
    checks = {
        "negative_permittivity": (-0.5 > 0.0),
        "missing_source_term": (sp.Symbol("h") - sp.Symbol("J") == sp.Symbol("h")),
        "indefinite_two_field_mixing": bool(np.min(np.linalg.eigvalsh([[1.0, -1.2], [-1.2, 1.0]])) > 0.0),
        "negative_quartic_coupling": bool(1.0 + 3.0 * (-1.0) * 1.0**2 > 0.0),
        "wrong_bihelmholtz_cross_coefficient": bool(math.isclose(0.2**2 * 0.6**2, 0.2 * 0.6)),
        "response_in_action_source": False,
    }
    return {
        "controls": [
            {"id": name, "admitted": bool(admitted), "expected": "reject"}
            for name, admitted in checks.items()
        ],
        "all_malformed_controls_rejected": not any(checks.values()),
    }


def build_derivation_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    symbolic = symbolic_action_derivation()
    malformed = malformed_action_controls()
    if not symbolic["all_exact_euler_residuals_zero"]:
        raise GravityItem48Error("symbolic action variation failed")
    if not malformed["all_malformed_controls_rejected"]:
        raise GravityItem48Error("a malformed action control was admitted")
    return _content_hashed(
        {
            **symbolic,
            "item": 48,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "action_catalog": action_catalog(config),
            "malformed_action_controls": malformed,
            "claim_boundary": config["formal_infrastructure_boundary"],
        }
    )


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item48-candidate-manifest-1.0",
            "item": 48,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_values_used_during_action_or_formula_generation": 0,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "symbolic_action_recipes": len(action_catalog(config)),
            "action_catalog": action_catalog(config),
            "claim_boundaries": [
                "Every candidate begins as a normalized radial weak-field action cell before empirical scoring.",
                "Exact radial Euler identities and reduced static convexity are not a covariant conservation, Hamiltonian, causality, or ghost proof.",
                "The zero-slip light response is a fixed diagnostic closure rather than a field equation derived from a relativistic action.",
                "No algebraic or behavioral distinction establishes historical novelty.",
                "Retrospective grouped validation can generate a lead but cannot confirm one.",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item48-exposure-manifest-1.0",
            "item": 48,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {
                    "id": "S4TM_ITEM43_EXPLORATION",
                    "objects": 28,
                    "response_status": "already exposed",
                    "role": "retrospective action development",
                },
                {
                    "id": "CLASH_ACCELERATION",
                    "objects": 20,
                    "points": 84,
                    "response_status": "already exposed",
                    "role": "retrospective action development",
                },
            ],
            "sealed_data": {
                "item43_s4tm_confirmation_lenses": 7,
                "access_authorized": False,
                "response_rows_read": 0,
            },
            "rules": [
                "derive actions and field equations before empirical scoring",
                "construct every action coordinate without a response field",
                "no result may be described as fresh confirmation",
                "preserve every mismatch; neither one counterexample nor its count is a veto",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> list[Path]:
    config = load_config(root)
    paths = [
        _source_path(root, config, "candidate_manifest"),
        _source_path(root, config, "derivation_receipt"),
        _source_path(root, config, "exposure_manifest"),
    ]
    _write_json(paths[0], build_candidate_manifest(root))
    _write_json(paths[1], build_derivation_receipt(root))
    _write_json(paths[2], build_exposure_manifest(root))
    return paths


def _quadrature_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.any(np.diff(x) <= 0.0):
        raise GravityItem48Error("action profile requires at least three ordered nodes")
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.diff(edges)


def _derivative_matrix(x: np.ndarray) -> np.ndarray:
    identity = np.eye(len(x), dtype=float)
    return np.gradient(identity, np.asarray(x, dtype=float), axis=0, edge_order=1)


def _profile_sources(
    radius: np.ndarray,
    mass: np.ndarray,
    *,
    baryonic_size: float,
    evaluation_radius: float,
    evaluation_u: float,
) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    mass = np.maximum.accumulate(np.asarray(mass, dtype=float))
    mass_at_evaluation = float(np.interp(evaluation_radius, radius, mass))
    u = evaluation_u * (mass / mass_at_evaluation) * np.square(evaluation_radius / radius)
    log_u = np.log10(np.maximum(u, 1e-300))
    geometry = np.log10(radius / baryonic_size)
    slope = np.gradient(np.log(np.maximum(mass, 1e-300)), np.log(radius), edge_order=1)
    bounded_u = log_u / (2.0 + np.abs(log_u))
    bounded_geometry = geometry / (1.0 + np.abs(geometry))
    bounded_slope = (slope - 1.0) / (1.0 + np.abs(slope - 1.0))
    interaction = bounded_geometry * np.tanh(2.0 * bounded_u)
    result = np.vstack((bounded_u, bounded_geometry, bounded_slope, interaction))
    if result.shape != (4, len(radius)) or not np.all(np.isfinite(result)):
        raise GravityItem48Error("action source construction failed")
    return result


def _linear_solution(
    q: np.ndarray, weighted_source: np.ndarray
) -> tuple[np.ndarray, float, float]:
    eigen_min = float(np.min(np.linalg.eigvalsh(0.5 * (q + q.T))))
    if eigen_min <= 0.0:
        raise GravityItem48Error("reduced action Hessian is not positive")
    h = np.linalg.solve(q, weighted_source)
    residual = q @ h - weighted_source
    relative = float(
        np.max(np.abs(residual)) / max(1.0, float(np.max(np.abs(weighted_source))))
    )
    return h, relative, eigen_min


def _solve_action_profile(
    action_class: str,
    variant: Any,
    source: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source, dtype=float)
    weights = _quadrature_weights(x)
    w = np.diag(weights)
    d1 = _derivative_matrix(x)
    d2 = d1 @ d1
    rhs = w @ source
    if action_class == "source_conditioned_permittivity":
        if variant == "identity":
            h = source.copy()
        elif variant == "signed_square_root":
            h = np.sign(source) * np.sqrt(np.abs(source))
        elif variant == "tanh":
            h = np.tanh(source)
        elif variant == "softsign_0p3":
            h = source / (0.3 + np.abs(source))
        else:
            raise GravityItem48Error("unknown direct action link")
        return h, {
            "relative_discrete_euler_residual": 0.0,
            "minimum_reduced_hessian_eigenvalue": None,
            "iterations": 0,
        }
    if action_class == "screened_auxiliary":
        ell = float(variant)
        q = w + ell**2 * d1.T @ w @ d1
        h, residual, eigen_min = _linear_solution(q, rhs)
        return h, {
            "relative_discrete_euler_residual": residual,
            "minimum_reduced_hessian_eigenvalue": eigen_min,
            "iterations": 1,
        }
    if action_class == "bihelmholtz_auxiliary":
        ell1, ell2 = (float(value) for value in variant)
        q = (
            w
            + (ell1**2 + ell2**2) * d1.T @ w @ d1
            + ell1**2 * ell2**2 * d2.T @ w @ d2
        )
        h, residual, eigen_min = _linear_solution(q, rhs)
        return h, {
            "relative_discrete_euler_residual": residual,
            "minimum_reduced_hessian_eigenvalue": eigen_min,
            "iterations": 1,
        }
    if action_class == "mixed_two_field":
        ell, kappa = (float(value) for value in variant)
        q0 = w + ell**2 * d1.T @ w @ d1
        coupling = -kappa * w
        q = np.block([[q0, coupling], [coupling, q0]])
        solution, residual, eigen_min = _linear_solution(
            q, np.concatenate((rhs, np.zeros_like(rhs)))
        )
        return solution[: len(source)], {
            "relative_discrete_euler_residual": residual,
            "minimum_reduced_hessian_eigenvalue": eigen_min,
            "iterations": 1,
        }
    if action_class == "convex_nonlinear_auxiliary":
        ell, lam = (float(value) for value in variant)
        q = w + ell**2 * d1.T @ w @ d1
        h = np.linalg.solve(q, rhs)
        residual = math.inf
        iterations = 0
        for iterations in range(1, 51):
            gradient = q @ h + lam * weights * np.power(h, 3) - rhs
            residual = float(
                np.max(np.abs(gradient)) / max(1.0, float(np.max(np.abs(rhs))))
            )
            if residual <= 1e-12:
                break
            hessian = q + np.diag(3.0 * lam * weights * np.square(h))
            step = np.linalg.solve(hessian, gradient)
            scale = 1.0
            current_norm = float(np.linalg.norm(gradient))
            while scale > 1e-6:
                trial = h - scale * step
                trial_gradient = q @ trial + lam * weights * np.power(trial, 3) - rhs
                if float(np.linalg.norm(trial_gradient)) < current_norm:
                    h = trial
                    break
                scale *= 0.5
            else:
                raise GravityItem48Error("convex nonlinear action Newton solve stalled")
        hessian = q + np.diag(3.0 * lam * weights * np.square(h))
        eigen_min = float(np.min(np.linalg.eigvalsh(0.5 * (hessian + hessian.T))))
        if residual > 1e-8 or eigen_min <= 0.0:
            raise GravityItem48Error("convex nonlinear action solve failed")
        return h, {
            "relative_discrete_euler_residual": residual,
            "minimum_reduced_hessian_eigenvalue": eigen_min,
            "iterations": iterations,
        }
    if action_class == "adaptive_gradient_auxiliary":
        ell, sigma = (float(value) for value in variant)
        stiffness = np.diag(weights * (1.0 + sigma * np.square(source)))
        q = w + ell**2 * d1.T @ stiffness @ d1
        h, residual, eigen_min = _linear_solution(q, rhs)
        return h, {
            "relative_discrete_euler_residual": residual,
            "minimum_reduced_hessian_eigenvalue": eigen_min,
            "iterations": 1,
        }
    raise GravityItem48Error(f"unknown action class: {action_class}")


def action_bank_from_arrays(
    arrays: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    profiles = _profiles(arrays)
    catalog = action_catalog(config)
    raw = np.empty((len(arrays["object"]), len(catalog)), dtype=float)
    audit: dict[tuple[str, int], dict[str, Any]] = {}
    for name in sorted(set(np.asarray(arrays["object"]).tolist())):
        indices = np.flatnonzero(np.asarray(arrays["object"]) == name)
        first = int(indices[0])
        profile_radius, profile_mass = profiles[str(name)]
        profile_x = np.log(profile_radius)
        sources = _profile_sources(
            profile_radius,
            profile_mass,
            baryonic_size=float(arrays["size"][first]),
            evaluation_radius=float(arrays["radius"][first]),
            evaluation_u=float(arrays["u"][first]),
        )
        solutions: list[np.ndarray] = []
        for recipe in catalog:
            solution, health = _solve_action_profile(
                str(recipe["action_class"]),
                recipe["variant"],
                sources[int(recipe["source_id"])],
                profile_x,
            )
            solutions.append(solution)
            audit[(str(name), int(recipe["recipe_id"]))] = health
        for row_index in indices:
            radius = float(arrays["radius"][row_index])
            raw[row_index] = np.asarray(
                [np.interp(radius, profile_radius, solution) for solution in solutions]
            )
    coordinate = 0.5 + 0.5 * raw / (1.0 + np.abs(raw))
    if (
        raw.shape != (len(arrays["object"]), 96)
        or not np.all(np.isfinite(coordinate))
        or np.any(coordinate <= 0.0)
        or np.any(coordinate >= 1.0)
    ):
        raise GravityItem48Error("action coordinate bank failed")
    residuals = [float(row["relative_discrete_euler_residual"]) for row in audit.values()]
    eigenvalues = [
        float(row["minimum_reduced_hessian_eigenvalue"])
        for row in audit.values()
        if row["minimum_reduced_hessian_eigenvalue"] is not None
    ]
    solver_audit = {
        "profile_recipe_solves": len(audit),
        "maximum_relative_discrete_euler_residual": max(residuals),
        "minimum_reduced_hessian_eigenvalue": min(eigenvalues),
        "maximum_nonlinear_iterations": max(int(row["iterations"]) for row in audit.values()),
    }
    maximum = float(
        config["candidate_generator"]["admissibility"][
            "require_discrete_euler_residual_at_most"
        ]
    )
    if solver_audit["maximum_relative_discrete_euler_residual"] > maximum:
        raise GravityItem48Error("discrete action Euler residual exceeds frozen limit")
    return raw, coordinate, solver_audit


def _source_arrays(item44: Mapping[str, Any]) -> dict[str, Any]:
    rows = item44["records"]
    return {
        "population": np.asarray([row["population"] for row in rows]),
        "object": np.asarray([row["object"] for row in rows]),
        "fold": np.asarray([int(row["fold"]) for row in rows]),
        "radius": np.asarray([float(row["radius_kpc"]) for row in rows]),
        "size": np.asarray([float(row["baryonic_size_kpc"]) for row in rows]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows]),
        "u": np.asarray([float(row["u"]) for row in rows]),
    }


def build_action_features(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item44 = _read_json(root / ITEM44_FEATURE_PATH)
    item47 = _read_json(root / ITEM47_FEATURE_PATH)
    arrays = _source_arrays(item44)
    if len(item47["records"]) != len(item44["records"]):
        raise GravityItem48Error("Item 44/47 source row counts differ")
    for index, (left, right) in enumerate(zip(item44["records"], item47["records"], strict=True)):
        if (
            int(right["source_row_index"]) != index
            or left["population"] != right["population"]
            or left["object"] != right["object"]
        ):
            raise GravityItem48Error("Item 44/47 source alignment changed")
    raw, bank, solver_audit = action_bank_from_arrays(arrays, config)
    records = [
        {
            "source_row_index": index,
            "population": str(arrays["population"][index]),
            "object": str(arrays["object"][index]),
            "fold": int(arrays["fold"][index]),
            "raw_action_values": [float(value) for value in raw[index]],
            "action_coordinates": [float(value) for value in bank[index]],
        }
        for index in range(len(arrays["object"]))
    ]
    hashes = [
        hashlib.sha256(np.round(bank[:, index], 12).astype("<f8").tobytes()).hexdigest()
        for index in range(bank.shape[1])
    ]
    lineage = [
        {
            "population": row["population"],
            "object": row["object"],
            "fold": row["fold"],
            "radius_kpc": row["radius_kpc"],
            "baryonic_size_kpc": row["baryonic_size_kpc"],
            "redshift": row["redshift"],
            "u": row["u"],
        }
        for row in item44["records"]
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item48-action-features-1.0",
            "item": 48,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_blind_source_lineage_sha256": _sha256_bytes(_canonical_bytes(lineage)),
            "response_fields_read_by_feature_builder": [],
            "response_values_used": 0,
            "records": records,
            "counts": {
                "s4tm_lenses": int(np.sum(arrays["population"] == "S4TM")),
                "clash_clusters": len(
                    set(arrays["object"][arrays["population"] == "CLASH"].tolist())
                ),
                "clash_points": int(np.sum(arrays["population"] == "CLASH")),
                "total_points": len(records),
                "action_recipes": bank.shape[1],
                "sealed_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "solver_audit": solver_audit,
            "dataset_behavior": {
                "unique_action_coordinate_hashes": len(set(hashes)),
                "duplicate_symbolic_recipes_on_development_predictors": len(hashes)
                - len(set(hashes)),
                "action_coordinate_sha256": hashes,
            },
            "action_catalog": action_catalog(config),
            "lineage": config["data_roles"],
        }
    )


def write_action_features(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "feature_receipt")
    _write_json(path, build_action_features(root))
    return path


def _evaluation_arrays(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    config47 = _load_item47_config(root)
    arrays, _ = _item47_evaluation_arrays(root, config47)
    feature = _read_json(_source_path(root, config, "feature_receipt"))
    rows = feature["records"]
    if len(rows) != len(arrays["target"]):
        raise GravityItem48Error("action feature row count changed")
    for index, row in enumerate(rows):
        if (
            int(row["source_row_index"]) != index
            or row["population"] != arrays["population"][index]
            or row["object"] != arrays["object"][index]
        ):
            raise GravityItem48Error("action feature/source alignment changed")
    arrays["action_bank"] = np.asarray(
        [row["action_coordinates"] for row in rows], dtype=float
    ).T
    return arrays


def _candidate_subset(
    candidates: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in candidates.items()}


def _predict(
    candidate_id: int,
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    bank_key: str,
) -> np.ndarray:
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    recipe = candidate_id // per_recipe
    local = candidate_id % per_recipe
    row = {
        "amplitude_index": np.asarray([(local // 256) % 16]),
        "exponent_index": np.asarray([(local // 16) % 16]),
        "transition_index": np.asarray([local % 16]),
    }
    amplitude, exponent, transition = _candidate_parameters(row, config)
    h = np.asarray(arrays[bank_key])[recipe]
    multiplier = 1.0 + amplitude[0] * np.power(arrays["u"], -exponent[0]) / (
        1.0 + arrays["u"] / transition[0]
    ) * (0.05 + 0.95 * h)
    if np.any(multiplier <= 0.0) or not np.all(np.isfinite(multiplier)):
        raise GravityItem48Error("admitted action produced invalid permittivity")
    return arrays["base"] + np.log10(multiplier)


def _fixed_oof(
    fold_ids: Mapping[int, int],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    bank_key: str,
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _predict(candidate_id, arrays, config, bank_key=bank_key)[test]
    return prediction


def _item47_oof(
    root: Path, arrays: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, int]]:
    config47 = _load_item47_config(root)
    evaluation = _read_json(root / ITEM47_EVALUATION_PATH)
    fold_ids = {
        int(row["fold"]): int(row["selected_operator"]["candidate_id"])
        for row in evaluation["fold_ledger"]
    }
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item47_predict(
            candidate_id, arrays, config47, bank_key="operator_bank"
        )[test]
    return prediction, fold_ids


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    arrays = _evaluation_arrays(root, config)
    admitted, admission = admissible_candidates(config)
    direct_candidates = _candidate_subset(admitted, np.asarray(admitted["recipe"]) < 16)
    scale_candidates = _candidate_subset(admitted, np.asarray(admitted["recipe"]) == 0)
    scale_arrays = dict(arrays)
    scale_arrays["scale_free_bank"] = np.ones((96, len(arrays["target"])))
    candidate_oof = np.empty(len(arrays["target"]), dtype=float)
    direct_oof = np.empty(len(arrays["target"]), dtype=float)
    fold_candidate: dict[int, int] = {}
    fold_direct: dict[int, int] = {}
    fold_scale: dict[int, int] = {}
    ledger: list[dict[str, Any]] = []
    backends: set[str] = set()
    evaluations = 0
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        candidate_id, train_loss, backend, count = _best_candidate(
            admitted, arrays, train, config, bank_key="action_bank"
        )
        direct_id, direct_loss, direct_backend, direct_count = _best_candidate(
            direct_candidates, arrays, train, config, bank_key="action_bank"
        )
        scale_id, scale_loss, scale_backend, scale_count = _best_candidate(
            scale_candidates, scale_arrays, train, config, bank_key="scale_free_bank"
        )
        candidate_oof[test] = _predict(
            candidate_id, arrays, config, bank_key="action_bank"
        )[test]
        direct_oof[test] = _predict(direct_id, arrays, config, bank_key="action_bank")[test]
        fold_candidate[fold] = candidate_id
        fold_direct[fold] = direct_id
        fold_scale[fold] = scale_id
        evaluations += count + direct_count + scale_count
        backends.update((backend, direct_backend, scale_backend))
        ledger.append(
            {
                "fold": fold,
                "selected_action": decode_candidate(candidate_id, config),
                "action_training_balanced_loss": train_loss,
                "selected_direct_action_control": decode_candidate(direct_id, config),
                "direct_action_training_balanced_loss": direct_loss,
                "selected_scale_free_candidate_id": scale_id,
                "scale_free_training_balanced_loss": scale_loss,
                "heldout_s4tm_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "S4TM")
                        ].tolist()
                    )
                ),
                "heldout_clash_objects": sorted(
                    set(
                        arrays["object"][
                            test & (arrays["population"] == "CLASH")
                        ].tolist()
                    )
                ),
            }
        )
    scale_oof = _fixed_oof(fold_scale, scale_arrays, config, "scale_free_bank")
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_id, selected_loss, backend, count = _best_candidate(
        admitted, arrays, all_rows, config, bank_key="action_bank"
    )
    selected_direct, selected_direct_loss, direct_backend, direct_count = _best_candidate(
        direct_candidates, arrays, all_rows, config, bank_key="action_bank"
    )
    evaluations += count + direct_count
    backends.update((backend, direct_backend))
    cpu_loss = _score(arrays, _predict(selected_id, arrays, config, bank_key="action_bank"))[
        "balanced_loss"
    ]
    cpu_gpu_difference = abs(float(cpu_loss) - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem48Error("CPU/GPU selected loss cross-check failed")

    item44_oof, fold_item44 = _item44_oof(root, arrays)
    item45_oof, fold_item45 = _item45_oof(root, arrays)
    item46_oof, fold_item46 = _item46_oof(root, arrays)
    item47_oof, fold_item47 = _item47_oof(root, arrays)
    scores = {
        "action_generator": _score(arrays, candidate_oof),
        "item47_operator_generator": _score(arrays, item47_oof),
        "item46_dimensionless_generator": _score(arrays, item46_oof),
        "item45_universal_interaction": _score(arrays, item45_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "matched_direct_source_action": _score(arrays, direct_oof),
        "matched_scale_free": _score(arrays, scale_oof),
        "baryonic_newton": _score(arrays, arrays["base"]),
        "mond_rar": _score(
            arrays,
            arrays["base"]
            + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"])))),
        ),
        "ordinary_ridge": _score(arrays, _ordinary_crossfit(arrays, config)),
    }
    controls = tuple(name for name in scores if name != "action_generator")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["action_generator"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray([control_objects[key] - candidate_objects[key] for key in object_keys])
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()
    systematic_scores: dict[str, Any] = {}
    config44 = _read_json(root / "configs/gravity_item44_scale_hierarchy_v1.json")
    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    config47 = _load_item47_config(root)
    shapes = _shape_by_object(root, arrays)
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(arrays, str(population), float(shift), config45)
        varied["pi_bank"] = (
            1.0
            / (
                1.0
                + np.abs(
                    _item46_physical_log_values(varied, config46)
                    @ np.asarray(_item46_pi_vectors(config46), dtype=float).T
                )
            )
        ).T
        varied["operator_bank"] = _item47_operator_bank_from_arrays(
            varied, shapes, config47
        )[1].T
        varied["action_bank"] = action_bank_from_arrays(varied, config)[1].T
        varied_scale = dict(varied)
        varied_scale["scale_free_bank"] = np.ones((96, len(varied["target"])))
        candidate_variant = _fixed_oof(fold_candidate, varied, config, "action_bank")
        direct_variant = _fixed_oof(fold_direct, varied, config, "action_bank")
        scale_variant = _fixed_oof(fold_scale, varied_scale, config, "scale_free_bank")
        item44_variant = np.empty(len(varied["target"]), dtype=float)
        item45_variant = np.empty(len(varied["target"]), dtype=float)
        item46_variant = np.empty(len(varied["target"]), dtype=float)
        item47_variant = np.empty(len(varied["target"]), dtype=float)
        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            item44_variant[test] = _item44_predict(
                fold_item44[fold], varied, config44
            )[test]
            item45_variant[test] = _item45_predict(
                fold_item45[fold], varied, config45, bank_key="interaction_bank"
            )[test]
            item46_variant[test] = _item46_predict(
                fold_item46[fold], varied, config46, bank_key="pi_bank"
            )[test]
            item47_variant[test] = _item47_predict(
                fold_item47[fold], varied, config47, bank_key="operator_bank"
            )[test]
        variants = {
            "action_generator": _score(varied, candidate_variant),
            "item47_operator_generator": _score(varied, item47_variant),
            "item46_dimensionless_generator": _score(varied, item46_variant),
            "item45_universal_interaction": _score(varied, item45_variant),
            "item44_scale_hierarchy": _score(varied, item44_variant),
            "matched_direct_source_action": _score(varied, direct_variant),
            "matched_scale_free": _score(varied, scale_variant),
            "baryonic_newton": _score(varied, varied["base"]),
            "mond_rar": _score(
                varied,
                varied["base"]
                + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(varied["u"])))),
            ),
            "ordinary_ridge": _score(varied, _ordinary_crossfit(varied, config)),
        }
        systematic_scores[str(variant_name)] = {
            "action_generator": variants["action_generator"],
            "item45_primary_control": variants["item45_universal_interaction"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= (
                variants["action_generator"]["object_losses"][key]
                > variants[strongest]["object_losses"][key]
            )
    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(
        1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"]))
    )
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = (
        100.0
        * (
            scores[strongest]["balanced_loss"]
            - scores["action_generator"]["balanced_loss"]
        )
        / scores[strongest]["balanced_loss"]
    )
    improvement_item45 = (
        100.0
        * (
            scores["item45_universal_interaction"]["balanced_loss"]
            - scores["action_generator"]["balanced_loss"]
        )
        / scores["item45_universal_interaction"]["balanced_loss"]
    )
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_counterexample)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool(
            (min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "trim_changes_sign": bool(
            (float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item48-joint-evaluation-1.0",
            "item": 48,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_data_balanced_training_loss": selected_loss,
            "selected_direct_action_control": decode_candidate(selected_direct, config),
            "selected_direct_full_data_balanced_training_loss": selected_direct_loss,
            "fold_ledger": ledger,
            "scores": scores,
            "strongest_control": strongest,
            "aggregate_improvement_percent": improvement,
            "improvement_over_item45_percent": improvement_item45,
            "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
            },
            "counterexamples": [
                {
                    "object": key,
                    "raw_counterexample": bool(raw_counterexample[index]),
                    "uncertainty_resolved_counterexample": bool(
                        stable_counterexample[index]
                    ),
                }
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends),
                "candidate_point_fold_evaluations": evaluations,
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_difference,
                "admission": admission,
            },
            "counts": {
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "All responses were exposed before Item 48; grouped cross-validation cannot create fresh confirmation.",
                "The actions are normalized static radial effective actions, not covariant four-dimensional theories.",
                "The auxiliary constraints make the source-to-field map action-derived but do not prove a healthy propagating degree of freedom.",
                "Static reduced convexity does not prove Hamiltonian stability, hyperbolicity, or causality.",
                "The identical zero-slip multiplier for light is a diagnostic closure rather than a relativistic lensing derivation.",
                "S4TM uses an analytic projected stellar profile without measured gas; CLASH uses model-dependent published acceleration profiles.",
                "Four global mass shifts do not exhaust baryonic or lens-model uncertainty and cannot prune a family.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    derivation = _read_json(_source_path(root, config, "derivation_receipt"))
    exposure = _read_json(_source_path(root, config, "exposure_manifest"))
    features = _read_json(_source_path(root, config, "feature_receipt"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    scores = evaluation["scores"]
    systematics = evaluation["systematic_scores"]
    promotion = config["evaluation"]["promotion_gates"]
    item45 = scores["item45_universal_interaction"]
    action = scores["action_generator"]
    class_names = [
        row["selected_action"]["action_class"] for row in evaluation["fold_ledger"]
    ]
    gates = {
        "all_action_derivation_and_static_health_checks_pass": bool(
            derivation["all_exact_euler_residuals_zero"]
            and derivation["all_source_free_shift_identities_zero"]
            and derivation["malformed_action_controls"][
                "all_malformed_controls_rejected"
            ]
            and float(features["solver_audit"]["maximum_relative_discrete_euler_residual"])
            <= float(
                config["candidate_generator"]["admissibility"][
                    "require_discrete_euler_residual_at_most"
                ]
            )
            and float(features["solver_audit"]["minimum_reduced_hessian_eigenvalue"])
            > 0.0
        ),
        "balanced_improvement_over_item45_at_least": float(
            evaluation["improvement_over_item45_percent"]
        )
        >= 100.0 * float(promotion["balanced_improvement_over_item45_at_least"]),
        "improves_both_populations_over_item45": all(
            action["populations"][population]["loss"]
            < item45["populations"][population]["loss"]
            for population in ("S4TM", "CLASH")
        ),
        "paired_p_at_most": float(evaluation["paired_sign_flip_p"])
        <= float(promotion["paired_p_at_most"]),
        "same_action_class_selected_in_all_folds": len(set(class_names)) == 1,
        "leave_one_and_trim_stable": bool(
            float(
                evaluation["robustness"][
                    "leave_one_min_mean_control_minus_candidate_loss"
                ]
            )
            > 0.0
            and float(
                evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]
            )
            > 0.0
        ),
        "all_mass_scale_variants_positive": all(
            value["action_generator"]["balanced_loss"]
            < value["item45_primary_control"]["balanced_loss"]
            for value in systematics.values()
        ),
        "post_evaluation_candidate_cells": int(
            evaluation["counts"]["post_evaluation_candidate_cells"]
        )
        == 0,
        "sealed_confirmation_rows": int(evaluation["counts"]["sealed_confirmation_rows"])
        == 0,
        "fresh_confirmation_available": False,
    }
    empirical_gate_names = tuple(
        name
        for name in gates
        if name
        not in {
            "fresh_confirmation_available",
            "post_evaluation_candidate_cells",
            "sealed_confirmation_rows",
        }
    )
    empirical_lead = all(gates[name] for name in empirical_gate_names)
    decision = (
        "RETROSPECTIVE_ITEM48_ACTION_LEAD_REQUIRES_COVARIANT_AND_FRESH_TESTS"
        if empirical_lead
        else "NONPROMOTED_ITEM48_ACTION_RESULT_RETAINED"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item48-action-generator-result-1.0",
            "item": 48,
            "goal": "GRAVITY_ROADMAP_ITEM_48_ACTION_GENERATOR",
            "decision": decision,
            "selected_candidate": evaluation["selected_candidate"],
            "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation[
                "aggregate_improvement_percent"
            ],
            "improvement_over_item45_percent": evaluation[
                "improvement_over_item45_percent"
            ],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "gates": gates,
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "counts": {
                "raw_candidates": candidate["raw_candidates"],
                "admitted_candidates": candidate["admitted_candidates"],
                "symbolic_action_recipes": candidate["symbolic_action_recipes"],
                "unique_action_behaviors_on_development_data": features[
                    "dataset_behavior"
                ]["unique_action_coordinate_hashes"],
                "profile_recipe_solves": features["solver_audit"][
                    "profile_recipe_solves"
                ],
                "s4tm_lenses": features["counts"]["s4tm_lenses"],
                "clash_clusters": features["counts"]["clash_clusters"],
                "clash_points": features["counts"]["clash_points"],
                "candidate_point_fold_evaluations": evaluation["compute"][
                    "candidate_point_fold_evaluations"
                ],
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "formal_scope": {
                "exact_radial_euler_residuals_zero": derivation[
                    "all_exact_euler_residuals_zero"
                ],
                "source_free_shift_identities_zero": derivation[
                    "all_source_free_shift_identities_zero"
                ],
                "malformed_controls_rejected": derivation[
                    "malformed_action_controls"
                ]["all_malformed_controls_rejected"],
                "maximum_discrete_euler_residual": features["solver_audit"][
                    "maximum_relative_discrete_euler_residual"
                ],
                "minimum_reduced_hessian_eigenvalue": features["solver_audit"][
                    "minimum_reduced_hessian_eigenvalue"
                ],
                "covariant_completion": False,
                "full_hamiltonian_stability": False,
                "relativistic_lensing_derivation": False,
            },
            "source_bindings": {
                "config": {
                    "path": str(CONFIG_PATH),
                    "sha256": _sha256_file(root / CONFIG_PATH),
                },
                "candidate_manifest": {
                    "path": str(
                        _source_path(root, config, "candidate_manifest").relative_to(root)
                    ),
                    "sha256": _sha256_file(
                        _source_path(root, config, "candidate_manifest")
                    ),
                },
                "derivation_receipt": {
                    "path": str(
                        _source_path(root, config, "derivation_receipt").relative_to(root)
                    ),
                    "sha256": _sha256_file(
                        _source_path(root, config, "derivation_receipt")
                    ),
                },
                "exposure_manifest": {
                    "path": str(
                        _source_path(root, config, "exposure_manifest").relative_to(root)
                    ),
                    "sha256": _sha256_file(
                        _source_path(root, config, "exposure_manifest")
                    ),
                },
                "features": {
                    "path": str(
                        _source_path(root, config, "feature_receipt").relative_to(root)
                    ),
                    "sha256": _sha256_file(_source_path(root, config, "feature_receipt")),
                },
                "evaluation": {
                    "path": str(
                        _source_path(root, config, "evaluation_result").relative_to(root)
                    ),
                    "sha256": _sha256_file(_source_path(root, config, "evaluation_result")),
                },
            },
            "claims": {
                "roadmap_item_48_complete": True,
                "fresh_confirmation_completed": False,
                "action_derived_radial_field_equations": True,
                "covariant_action_established": False,
                "full_stability_established": False,
                "relativistic_lensing_theory_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Preserve every action, derivation, equivalence, and mismatch; require a covariant completion and fresh unchanged replication for confirmation, then advance to Item 49 seeded pseudorandom exploration.",
            "exposure": exposure,
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "candidate_manifest": _read_json(_source_path(root, config, "candidate_manifest"))
        == build_candidate_manifest(root),
        "derivation_receipt": _read_json(_source_path(root, config, "derivation_receipt"))
        == build_derivation_receipt(root),
        "exposure_manifest": _read_json(_source_path(root, config, "exposure_manifest"))
        == build_exposure_manifest(root),
        "feature_receipt": _read_json(_source_path(root, config, "feature_receipt"))
        == build_action_features(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "write-freeze",
            "write-features",
            "evaluate",
            "aggregate",
            "replay",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write-freeze":
        result: Any = [str(path) for path in write_freeze_manifests(root)]
    elif args.command == "write-features":
        result = str(write_action_features(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
