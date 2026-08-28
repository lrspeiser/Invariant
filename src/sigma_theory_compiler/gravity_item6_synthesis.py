"""Deterministic synthesis for the Item 6 thermodynamic-state experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-06-thermodynamic-state-v1.json"
ATTEMPT_FILE_SHA256 = "beed9e79a12ca53c62f3b6e8678dfab0e4f71f295bbe401a35598f4704d6085d"
ATTEMPT_CONTENT_SHA256 = "c111d943b654d7650864548494e9783edd2551a95c0e35f23f5cb96bcdae9c71"
ATTEMPT_DECISION = "REJECT_ITEM6_THERMODYNAMIC_STATE_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-06-synthesis-v1.json"


class GravityItem6SynthesisError(RuntimeError):
    """Raised when the Item 6 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem6SynthesisError("stable roadmap changed")
    attempt_path = root / ATTEMPT_PATH
    if _sha256_file(attempt_path) != ATTEMPT_FILE_SHA256:
        raise GravityItem6SynthesisError("Item 6 attempt file changed")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt.get("content_sha256") != ATTEMPT_CONTENT_SHA256:
        raise GravityItem6SynthesisError("Item 6 attempt content changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem6SynthesisError("Item 6 attempt decision changed")
    if int(attempt["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem6SynthesisError("Item 6 confirmation boundary was opened")
    if not attempt["gate_checks"]["all_20_exploration_clusters_pass_frozen_quality"]:
        raise GravityItem6SynthesisError("Item 6 is not a clean representation test")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item6-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_06_THERMODYNAMIC_STATE_SYNTHESIS",
        "item_number": 6,
        "decision": "REJECT_ITEM6_TESTED_ACCEPT_THERMODYNAMIC_PROMOTION_RETAIN_COOLING_LEAD_ADVANCE_ITEM7",
        "evidence": {
            "path": ATTEMPT_PATH,
            "file_sha256": ATTEMPT_FILE_SHA256,
            "content_sha256": ATTEMPT_CONTENT_SHA256,
            "decision": ATTEMPT_DECISION,
            "exploration_clusters": attempt["counts"]["exploration_clusters"],
            "confirmation_accesses": attempt["counts"]["reserved_confirmation_target_accesses"],
            "qualifying_r2": attempt["primary"]["qualifying_selector"]["metrics"]["r2"],
            "relative_mse_improvement": attempt["primary"]["qualifying_selector"][
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "permutation_p_value": attempt["permutation"]["p_value"],
            "gates": attempt["gate_counts"],
        },
        "scope_rejected": [
            "promotion of the frozen cooling-time, core-pressure, entropy-gradient, and phase-interaction family on the 20 opened ACCEPT/HeCS responses",
            "the claim that those interactions improve collisionless dispersion beyond temperature/luminosity controls with preregistered statistical significance",
            "the claim that thermodynamic state has been established as a gravity cause",
        ],
        "nonpromoted_positive_lead": {
            "label": "NONPROMOTED_POSITIVE_LEAD",
            "family": "cooling_state_coupling",
            "heldout_r2": attempt["primary"]["qualifying_selector"]["metrics"]["r2"],
            "relative_mse_improvement": attempt["primary"]["qualifying_selector"][
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "positive_cool_core_strata": attempt["gate_checks"][
                "qualifying_improvement_positive_in_both_cool_core_strata"
            ],
            "positive_response_error_envelopes": attempt["gate_checks"][
                "upper_and_lower_response_error_envelopes_do_not_reverse_improvement"
            ],
            "why_not_promoted": [
                "the unrestricted selector chose the temperature/luminosity nuisance model in all five folds",
                "the frozen temperature-stratified permutation p-value is 0.326 rather than at most 0.05",
            ],
            "reuse_rule": "Do not retune on the 20 opened responses or open the eight confirmations. Reconsider only after a materially independent frozen dataset or an action-derived prediction reproduces the same interaction.",
        },
        "equivalence_regions_retained": [
            "sigma proportional to sqrt(kT) is the known beta_spec/virial temperature family",
            "temperature, X-ray luminosity, redshift, and member-count main effects are nuisance baselines",
            "linear K0, K100, and alpha effects are a known entropy-profile combination",
            "caustic/NFW mass, lensing mass, identity, and per-object coefficients remain forbidden shortcuts",
        ],
        "not_rejected": [
            "direct cooling-time and pressure profiles rather than the frozen proxies",
            "independently measured turbulent, electron-ion, or nonequilibrium thermodynamics",
            "an action-derived thermodynamic coupling with a prior quantitative prediction",
            "all thermodynamic gravity theories, dark-matter theories, or general relativity",
        ],
        "why_item_complete": (
            "The response-blind ACCEPT/HeCS match yields a balanced 20-cluster exploration "
            "sample with no quality failures and a strong temperature/luminosity control. The "
            "qualifying cooling-state family shows a 14.7% held-out MSE gain and positive slice "
            "robustness, so it is preserved rather than pruned. It nevertheless fails the two "
            "promotion checks designed to distinguish a creative interaction from flexible-fit "
            "luck: unrestricted selection and the 499-permutation test. Retuning the opened "
            "responses would invalidate that evidence. The next distinct causal search is Item 7 "
            "baryonic composition; the cooling lead remains available for independent future replay."
        ),
        "counts": {
            "attempts": 1,
            "exploration_clusters": attempt["counts"]["exploration_clusters"],
            "quality_failures": 0,
            "permutations": attempt["counts"]["permutation_nested_cv_runs"],
            "reserved_confirmation_clusters": attempt["counts"]["reserved_confirmation_clusters"],
            "confirmation_accesses": attempt["counts"]["reserved_confirmation_target_accesses"],
            "paid_model_calls": attempt["counts"]["paid_model_calls"],
        },
        "claim_boundaries": {
            "all_thermodynamic_state_theories_rejected": False,
            "tested_accept_thermodynamic_family_promoted": False,
            "cooling_state_lead_retained": True,
            "cooling_state_lead_confirmed": False,
            "roadmap_item_6_complete": True,
            "roadmap_item_7_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 7 baryonic composition. Freeze a real-data test of gas, stars, "
            "plasma, molecular matter, and their spatial mixtures before any new response access, "
            "while retaining the Item 6 cooling-state family as a nonpromoted positive lead."
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
        raise GravityItem6SynthesisError("Item 6 synthesis receipt drifted")


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
