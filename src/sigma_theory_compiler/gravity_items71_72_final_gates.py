"""Items 71-72: independent-confirmation and novelty gates."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    _law_acceleration,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    load_config as load_item59_config,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    replay as replay_item59,
)
from sigma_theory_compiler.gravity_item60_direct_clash_lensing_gate import (
    replay as replay_item60,
)
from sigma_theory_compiler.gravity_item61_cross_scale_gate import replay as replay_item61
from sigma_theory_compiler.gravity_items62_70_formal_gates import (
    replay as replay_items62_70,
)

CONFIG_PATH = Path("configs/gravity_items71_72_confirmation_novelty_v1.json")
ITEM59_PATH = Path("runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json")


class GravityItems71To72Error(RuntimeError):
    """Raised when the frozen final-gate boundary changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-items71-72-confirmation-novelty-config-1.0"
        or config.get("items") != [71, 72]
        or config.get("status")
        != "scientific_freeze_before_final_replay_and_novelty_adjudication"
    ):
        raise GravityItems71To72Error("unsupported Items 71-72 config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItems71To72Error("Items 71-72 scientific freeze is not bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItems71To72Error("invalid Items 71-72 freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItems71To72Error(f"scientific dependency changed: {relative}")
    item59 = _read_json(root / ITEM59_PATH)
    selected = item59["selection"]["selected_qualifying"]["variant"]
    target = config["target"]
    if (
        selected["variant_id"] != target["variant_id"]
        or selected["family_id"] != target["family_id"]
        or selected["parameters"] != target["parameters"]
    ):
        raise GravityItems71To72Error("target candidate changed")
    confirmation = config["item71"]
    if (
        confirmation["sealed_sparc_confirmation_rows_allowed"] != 0
        or confirmation["new_observational_target_rows_allowed"] != 0
        or not confirmation["pass_requires_external_data_or_independent_implementation"]
    ):
        raise GravityItems71To72Error("independent confirmation boundary changed")
    novelty = config["item72"]
    if (
        len(novelty["prior_art"]) < 7
        or novelty["absence_from_scoped_search_proves_global_novelty"]
        or novelty["behavioral_witness"]["required_local_formula_difference"] != 0.0
    ):
        raise GravityItems71To72Error("novelty adjudication boundary changed")
    policy = config["counterexample_policy"]
    if (
        policy["single_empirical_counterexample_terminal"]
        or policy["counterexample_count_alone_terminal"]
        or policy["finite_sample_may_prune_formula_family"]
        or policy["novelty_label_may_override_failed_physics_gate"]
        or policy["lack_of_novelty_prunes_empirically_useful_formula"]
    ):
        raise GravityItems71To72Error("counterexample or novelty policy changed")


def build_item71(root: Path) -> dict[str, Any]:
    config = load_config(root)
    replay_checks = {
        "item59": replay_item59(root),
        "item60": replay_item60(root),
        "item61": replay_item61(root),
        "items62_70": replay_items62_70(root),
    }
    all_internal = all(bool(value["ok"]) for value in replay_checks.values())
    item59 = _read_json(root / ITEM59_PATH)
    confirmation = config["item71"]
    external = int(confirmation["external_survey_confirmation_clusters"])
    independent = int(confirmation["independent_external_implementation_replays"])
    gate_passed = all_internal and (external > 0 or independent > 0)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item71-independent-confirmation-gate-1.0",
            "goal": "GRAVITY_ROADMAP_ITEM_71_INDEPENDENT_CONFIRMATION",
            "item": 71,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "target": config["target"],
            "eligible_claim": confirmation["eligible_claim"],
            "ineligible_claims": confirmation["ineligible_claims"],
            "internal_exact_replays": replay_checks,
            "all_internal_exact_replays_passed": all_internal,
            "same_release_confirmation": {
                "clusters": confirmation["same_release_confirmation_clusters"],
                "rows": item59["counts"]["confirmation_rows"],
                "formula_or_nuisance_refit": False,
                "survey_or_reduction_independent": False,
            },
            "external_survey_confirmation_clusters": external,
            "independent_external_implementation_replays": independent,
            "new_target_rows_opened": 0,
            "sealed_sparc_confirmation_rows_opened": 0,
            "gate_passed": gate_passed,
            "result_class": "BLOCKED",
            "decision": "ITEM71_INTERNAL_REPRODUCIBILITY_PASSED_EXTERNAL_INDEPENDENT_CONFIRMATION_NOT_OBTAINED",
            "claims": {
                "internal_reproducibility_established": all_internal,
                "external_independent_confirmation_established": False,
                "item59_narrow_claim_retained": True,
                "universal_gravity_claim_promoted": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [
                "The four Item 59 confirmation clusters came from the same X-COP release and analysis family as development.",
                "A byte-identical replay checks reproducibility, not investigator or instrument independence.",
                "The exact representation already failed galaxy transfer and unscreened Solar-System extrapolation, so only the narrow X-COP profile claim remains eligible.",
            ],
            "compute": {"backend": "deterministic_exact_replay", "gpu_used": False, "paid_api_cost_usd": 0.0},
            "next_action": "Freeze an external X-ray-plus-SZ cluster sample or commission a genuinely independent implementation before making a confirmation claim.",
        }
    )


def _behavioral_witness(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    config59 = load_item59_config(root)
    a0 = float(config59["constants"]["transition_acceleration_m_s2"])
    radius = np.geomspace(1.0, 100.0, 21)
    witness_index = 10
    profile_a = a0 * np.geomspace(30.0, 0.03, len(radius))
    profile_b = profile_a.copy()
    profile_b[:witness_index] *= 30.0
    profile_b[witness_index + 1 :] *= 0.03
    profile_b[witness_index] = profile_a[witness_index]
    beta = float(config["target"]["parameters"]["beta"])
    candidate_a = _law_acceleration(
        "cross_scale_boundary", {"beta": beta}, radius, 100.0, profile_a, config59
    )
    candidate_b = _law_acceleration(
        "cross_scale_boundary", {"beta": beta}, radius, 100.0, profile_b, config59
    )
    local_a = profile_a[witness_index] / -np.expm1(
        -np.sqrt(profile_a[witness_index] / a0)
    )
    local_b = profile_b[witness_index] / -np.expm1(
        -np.sqrt(profile_b[witness_index] / a0)
    )
    candidate_fraction = abs(candidate_a[witness_index] - candidate_b[witness_index]) / abs(
        candidate_a[witness_index]
    )
    local_fraction = abs(local_a - local_b) / abs(local_a)
    required = config["item72"]["behavioral_witness"]
    passed = (
        candidate_fraction >= float(required["required_candidate_difference_fraction_minimum"])
        and local_fraction == float(required["required_local_formula_difference"])
    )
    return {
        "witness_radius": float(radius[witness_index]),
        "local_gbar_profile_a_m_s2": float(profile_a[witness_index]),
        "local_gbar_profile_b_m_s2": float(profile_b[witness_index]),
        "candidate_profile_a_m_s2": float(candidate_a[witness_index]),
        "candidate_profile_b_m_s2": float(candidate_b[witness_index]),
        "candidate_fractional_difference": float(candidate_fraction),
        "local_rar_profile_a_m_s2": float(local_a),
        "local_rar_profile_b_m_s2": float(local_b),
        "local_formula_fractional_difference": float(local_fraction),
        "witness_passed": bool(passed),
        "establishes": "behavioral non-equivalence to formulas depending only on local radius and local gbar",
        "does_not_establish": "non-equivalence to all nonlocal, permittivity, auxiliary-field, or action-level theories",
    }


def build_item72(root: Path) -> dict[str, Any]:
    config = load_config(root)
    novelty = config["item72"]
    witness = _behavioral_witness(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item72-novelty-adjudication-gate-1.0",
            "goal": "GRAVITY_ROADMAP_ITEM_72_NOVELTY_ADJUDICATION",
            "item": 72,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "target": config["target"],
            "search_as_of": novelty["search_as_of"],
            "search_scope": novelty["search_scope"],
            "prior_art": novelty["prior_art"],
            "exact_phrase_search_found_exact_formula": novelty[
                "exact_phrase_search_found_exact_formula"
            ],
            "behavioral_non_equivalence_witness": witness,
            "classification": {
                "known_formula": False,
                "algebraic_rewrite_of_purely_local_rar_or_mond": False,
                "known_family_combination": True,
                "potentially_new_synthesis": True,
                "historical_novelty_established": False,
                "new_fundamental_physics_established": False,
                "reason": "The exact radial occupancy-and-two-kernel expression was not located in the scoped search and is behaviorally nonlocal, but every main motif has substantial prior art and global absence cannot be proved by search.",
            },
            "gate_passed": False,
            "result_class": "INCONCLUSIVE",
            "decision": "ITEM72_POTENTIALLY_NEW_SYNTHESIS_OF_KNOWN_MOTIFS_HISTORICAL_NOVELTY_NOT_ESTABLISHED",
            "claims": {
                "exact_local_algebraic_rewrite_excluded_by_witness": witness[
                    "witness_passed"
                ],
                "equivalence_to_all_nonlocal_theories_excluded": False,
                "historical_novelty_established": False,
                "empirical_usefulness_erased_by_prior_art": False,
                "failed_physics_gates_overridden_by_novelty_label": False,
                "formula_family_pruned": False,
            },
            "compute": {"backend": "literature_scope_plus_numeric_witness", "gpu_used": False, "paid_api_cost_usd": 0.0},
            "limitations": [
                "A literature search cannot prove that no equivalent formula exists under different notation.",
                "The witness excludes only purely local algebraic response families, not general nonlocal field theories.",
                "The formula lacks an action, local screen, lensing rule, and cosmology, so novelty would not make it a viable gravity theory.",
            ],
            "next_action": "Use the formula only as a clearly labeled phenomenological synthesis; pursue novelty only after an action-level descendant survives the physical gates and a specialist prior-art review.",
        }
    )


def build_aggregate(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item71 = build_item71(root)
    item72 = build_item72(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-items71-72-final-gates-1.0",
            "goal": "GRAVITY_ROADMAP_ITEMS_71_AND_72",
            "items": [71, 72],
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "decisions": {"71": item71["decision"], "72": item72["decision"]},
            "passed_items": [],
            "claims": {
                "roadmap_items_71_72_attempts_complete": True,
                "external_independent_confirmation_established": False,
                "historical_novelty_established": False,
                "narrow_item59_empirical_claim_retained": True,
                "current_candidate_is_complete_gravity_theory": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
            },
            "compute": {"gpu_used": False, "paid_api_cost_usd": 0.0},
            "next_action": "Close the current 72-item audit as executed, preserve the narrow X-COP lead, and start a new action-level screened transition-law campaign before acquiring new confirmation data.",
        }
    )


def write_results(root: Path) -> tuple[Path, Path, Path]:
    config = load_config(root)
    item71_path = root / str(config["paths"]["item71"])
    item72_path = root / str(config["paths"]["item72"])
    aggregate_path = root / str(config["paths"]["aggregate"])
    _write_json(item71_path, build_item71(root))
    _write_json(item72_path, build_item72(root))
    _write_json(aggregate_path, build_aggregate(root))
    return item71_path, item72_path, aggregate_path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item71_path = root / str(config["paths"]["item71"])
    item72_path = root / str(config["paths"]["item72"])
    aggregate_path = root / str(config["paths"]["aggregate"])
    checks = {
        "item71": item71_path.is_file() and _read_json(item71_path) == build_item71(root),
        "item72": item72_path.is_file() and _read_json(item72_path) == build_item72(root),
        "aggregate": aggregate_path.is_file()
        and _read_json(aggregate_path) == build_aggregate(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "evaluate":
        print(json.dumps({"paths": [str(path) for path in write_results(root)]}, sort_keys=True))
        return 0
    result = replay(root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
