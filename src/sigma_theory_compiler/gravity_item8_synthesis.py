"""Deterministic synthesis for gravity roadmap Item 8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
RESULT_PATH = "runs/gravity/roadmap/item-08-field-gradients-curvature-v1.json"
RESULT_FILE_SHA256 = "2c97f701d4119fb5b8dc628c9f0b2579919cc2c8b3da38bb510e9d889bc3510b"
RESULT_CONTENT_SHA256 = "d07841c5a10415583613e3ce1583a8d5124995fce90f8fdd2bb411b1751783ff"
RESULT_DECISION = "REJECT_ITEM8_FIELD_CURVATURE_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-08-synthesis-v1.json"


class GravityItem8SynthesisError(RuntimeError):
    """Raised when an Item 8 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(root: Path) -> dict[str, Any]:
    path = root / RESULT_PATH
    if _sha256_file(path) != RESULT_FILE_SHA256:
        raise GravityItem8SynthesisError("Item 8 result file changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("content_sha256") != RESULT_CONTENT_SHA256:
        raise GravityItem8SynthesisError("Item 8 result content changed")
    if result.get("decision") != RESULT_DECISION:
        raise GravityItem8SynthesisError("Item 8 result decision changed")
    counts = result["counts"]
    if int(counts["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem8SynthesisError("Item 8 confirmation boundary was opened")
    if int(counts["post_response_formula_generation_allowed"]) != 0:
        raise GravityItem8SynthesisError("post-response generation boundary changed")
    if int(counts["paid_model_calls"]) != 0:
        raise GravityItem8SynthesisError("unexpected paid model call")
    return result


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem8SynthesisError("stable roadmap changed")
    result = _load_result(root)
    qualifying = result["primary"]["qualifying_selector"]
    baseline = result["primary"]["strongest_nonqualifying_selector"]
    unrestricted = result["primary"]["unrestricted"]
    relative_gain = float(
        qualifying["relative_mse_improvement_over_strongest_baseline"]
    )
    unrestricted_qualifying = sum(
        bool(fold["selected_qualifying"]) for fold in unrestricted["folds"]
    )
    if relative_gain >= 0:
        raise GravityItem8SynthesisError("frozen field family is not negative")
    if unrestricted_qualifying != 0:
        raise GravityItem8SynthesisError("unrestricted selector retained a field family")
    if int(result["gate_counts"]["passed"]) != 3:
        raise GravityItem8SynthesisError("Item 8 gate count changed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item8-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_08_FIELD_GRADIENTS_AND_CURVATURE_SYNTHESIS",
        "item_number": 8,
        "decision": (
            "REJECT_ITEM8_TESTED_PROJECTED_K_LIGHT_FIELD_DERIVATIVES_ADVANCE_ITEM9"
        ),
        "evidence": {
            "path": RESULT_PATH,
            "file_sha256": RESULT_FILE_SHA256,
            "content_sha256": RESULT_CONTENT_SHA256,
            "decision": RESULT_DECISION,
            "exploration_groups": result["counts"]["exploration_groups"],
            "reserved_confirmation_groups": result["counts"][
                "reserved_confirmation_groups"
            ],
            "confirmation_accesses": result["counts"][
                "reserved_confirmation_target_accesses"
            ],
            "baseline_mse": baseline["metrics"]["mse"],
            "baseline_r2": baseline["metrics"]["r2"],
            "qualifying_mse": qualifying["metrics"]["mse"],
            "qualifying_r2": qualifying["metrics"]["r2"],
            "relative_mse_improvement": qualifying[
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "unrestricted_qualifying_folds": unrestricted_qualifying,
            "permutation_p_value": result["permutation"]["p_value"],
            "gates": result["gate_counts"],
        },
        "scope_rejected": [
            "promotion of the exact projected K-light normalized center-field, Hessian-invariant, third-derivative/alignment, radial-field-curvature, and combined families on 2M++ group dispersion",
            "retuning those families on the 98 opened group-dispersion responses",
            "the claim that the tested derivative synthesis is an established gravity cause",
        ],
        "equivalence_regions_retained": [
            "sqrt(GM/R) is a known virial mass-size scaling",
            "GM/R^2 and GM/R^3 amplitudes are algebraic mass-size rewrites",
            "projected axis ratio and radial concentration are ordinary shape controls",
            "dispersion, member redshift, virial or dark mass, lensing mass, identity, and per-object coefficients are forbidden circular shortcuts",
        ],
        "not_rejected": [
            "complete gas, plasma, and stellar baryonic fields rather than projected K light",
            "three-dimensional, covariant, external-field, boundary, or nonlocal curvature laws",
            "interior/exterior baryonic balance or redirected field mechanisms",
            "history, resonance, modified inertia, alternative field equations, dark matter, or general relativity",
        ],
        "failure_space": {
            "label": "FAILED_PROJECTED_K_LIGHT_FIELD_DERIVATIVE_REGION",
            "relative_mse_improvement": qualifying[
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "permutation_p_value": result["permutation"]["p_value"],
            "unrestricted_qualifying_folds": unrestricted_qualifying,
            "reuse_rule": (
                "Use for equivalence rejection. Reconsider only with materially new baryonic "
                "information, an external or nonlocal operator, a different response, or a prior "
                "field-equation derivation; never by retuning the opened 2M++ responses."
            ),
        },
        "why_item_complete": (
            "All 98 exploration groups passed quality. The qualifying field-derivative selector "
            "increased held-out MSE by 9.65%, the unrestricted selector chose only nonqualifying "
            "controls in all five folds, every richness and concentration stratum was negative, "
            "brightest-member removal did not rescue the result, and the frozen permutation test "
            "gave p=0.762. Only three of eleven gates passed. The 33 confirmations remain sealed. "
            "Item 9 supplies the next materially distinct nonlocal variables: interior/exterior balance."
        ),
        "counts": {
            "attempts": 1,
            "exploration_groups": result["counts"]["exploration_groups"],
            "quality_passing_groups": result["counts"]["exploration_groups"],
            "permutations": result["counts"]["permutation_nested_cv_runs"],
            "preregistered_model_families": result["counts"][
                "preregistered_model_families"
            ],
            "qualifying_families": result["counts"]["qualifying_families"],
            "reserved_confirmation_groups": result["counts"][
                "reserved_confirmation_groups"
            ],
            "confirmation_accesses": result["counts"][
                "reserved_confirmation_target_accesses"
            ],
            "post_response_formula_generation": result["counts"][
                "post_response_formula_generation_allowed"
            ],
            "paid_model_calls": result["counts"]["paid_model_calls"],
        },
        "claim_boundaries": {
            "all_field_curvature_theories_rejected": False,
            "tested_projected_k_light_derivative_families_promoted": False,
            "confirmation_opened": False,
            "roadmap_item_8_complete": True,
            "roadmap_item_9_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 9 interior/exterior balance. Freeze nonlocal enclosed-versus-outer "
            "baryon operators on a fresh resolved real dataset before accessing its target response; "
            "do not retune the opened Item 8 group responses or open their confirmations."
        ),
        "content_sha256": None,
    }
    content = dict(receipt)
    content.pop("content_sha256")
    receipt["content_sha256"] = canonical_sha256(content)
    return receipt


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    path = root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_receipt(root)) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    stored = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
    if stored != build_receipt(root):
        raise GravityItem8SynthesisError("Item 8 synthesis receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_receipt(args.root)
    else:
        print(write_receipt(args.root))


if __name__ == "__main__":
    main()
