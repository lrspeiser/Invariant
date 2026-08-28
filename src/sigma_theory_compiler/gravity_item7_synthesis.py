"""Deterministic synthesis for both Item 7 baryonic-composition attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPTS = (
    {
        "path": "runs/gravity/roadmap/item-07-baryonic-composition-v1.json",
        "file_sha256": "05150827ad3a6e9bea48331fb8e00025af155c22d6352fa67b6a963d6e0f2a2a",
        "content_sha256": "a3dca4434cb1cc9b3969b2c84bedd8ff3f0a8f11495ef6b1fc0840e6f89d7066",
        "decision": "INCONCLUSIVE_ITEM7_BARYONIC_COMPOSITION_QUALITY_GATE",
    },
    {
        "path": "runs/gravity/roadmap/item-07-baryonic-composition-v2.json",
        "file_sha256": "a27089b5fc2fc5f7223b1ba67f91da7fade08d20efdf7ee2eccdc496a275b71e",
        "content_sha256": "12bf6ba7f5ff691daa7f3b28d1e4ca4a3394c2c80eb31afbe48d01f5fbe5b4b8",
        "decision": "INCONCLUSIVE_ITEM7_COMPOSITION_REPLAY_QUALITY_GATE",
    },
)
OUTPUT_PATH = "runs/gravity/roadmap/item-07-synthesis-v1.json"


class GravityItem7SynthesisError(RuntimeError):
    """Raised when an Item 7 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_attempts(root: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for frozen in ATTEMPTS:
        path = root / frozen["path"]
        if _sha256_file(path) != frozen["file_sha256"]:
            raise GravityItem7SynthesisError("Item 7 attempt file changed")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("content_sha256") != frozen["content_sha256"]:
            raise GravityItem7SynthesisError("Item 7 attempt content changed")
        if receipt.get("decision") != frozen["decision"]:
            raise GravityItem7SynthesisError("Item 7 attempt decision changed")
        if int(receipt["counts"]["reserved_confirmation_target_accesses"]) != 0:
            raise GravityItem7SynthesisError("Item 7 confirmation boundary was opened")
        loaded.append(receipt)
    if int(loaded[1]["counts"]["phangs_confirmation_target_accesses"]) != 0:
        raise GravityItem7SynthesisError("PHANGS confirmation boundary was opened")
    return loaded


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem7SynthesisError("stable roadmap changed")
    attempt1, attempt2 = _load_attempts(root)
    attempt2_gain = float(
        attempt2["primary"]["qualifying_selector"][
            "relative_mse_improvement_over_strongest_baseline"
        ]
    )
    if attempt2_gain >= 0:
        raise GravityItem7SynthesisError("independent replay is not negative")
    if any(
        bool(fold["selected_qualifying"])
        for fold in attempt2["primary"]["unrestricted"]["folds"]
    ):
        raise GravityItem7SynthesisError("independent replay selected a qualifying family")

    evidence = []
    for index, (frozen, attempt) in enumerate(zip(ATTEMPTS, (attempt1, attempt2)), start=1):
        evidence.append(
            {
                "attempt": index,
                "path": frozen["path"],
                "file_sha256": frozen["file_sha256"],
                "content_sha256": frozen["content_sha256"],
                "decision": frozen["decision"],
                "exploration_galaxies": attempt["counts"]["exploration_galaxies"],
                "confirmation_accesses": attempt["counts"][
                    "reserved_confirmation_target_accesses"
                ],
                "qualifying_r2": attempt["primary"]["qualifying_selector"]["metrics"]["r2"],
                "relative_mse_improvement": attempt["primary"]["qualifying_selector"][
                    "relative_mse_improvement_over_strongest_baseline"
                ],
                "permutation_p_value": attempt["permutation"]["p_value"],
                "gates": attempt["gate_counts"],
                "interpretation": (
                    "A nonpromoted positive PHANGS lead with two representation failures."
                    if index == 1
                    else "The frozen lead fails its materially independent H I-width replay in every predictive robustness slice; one zero published uncertainty keeps the receipt formally inconclusive."
                ),
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item7-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_07_BARYONIC_COMPOSITION_SYNTHESIS",
        "item_number": 7,
        "decision": "REJECT_ITEM7_TESTED_GLOBAL_PHASE_COMPOSITION_GENERALIZATION_ADVANCE_ITEM8",
        "evidence": evidence,
        "scope_rejected": [
            "promotion or generalization of the exact global phase-entropy, stellar-gas boundary, atomic-molecular boundary, ratio-curvature, and composition-by-structure family",
            "retuning that family on the opened PHANGS rotation responses or xGASS H I-width responses",
            "the claim that the tested global composition interactions are an established gravity cause",
        ],
        "failed_replay_lead": {
            "label": "FAILED_INDEPENDENT_REPLAY_LEAD",
            "origin": "PHANGS attempt 1",
            "origin_relative_mse_improvement": attempt1["primary"]["qualifying_selector"][
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "replay_relative_mse_improvement": attempt2["primary"]["qualifying_selector"][
                "relative_mse_improvement_over_strongest_baseline"
            ],
            "replay_permutation_p_value": attempt2["permutation"]["p_value"],
            "replay_unrestricted_qualifying_folds": sum(
                bool(fold["selected_qualifying"])
                for fold in attempt2["primary"]["unrestricted"]["folds"]
            ),
            "reuse_rule": "Keep as counterexample knowledge. Reconsider only from a materially new resolved composition representation or a prior action-derived prediction, never by retuning either opened response set.",
        },
        "equivalence_regions_retained": [
            "V proportional to total baryonic mass to the one-quarter power is the known baryonic Tully-Fisher family",
            "V squared proportional to baryonic mass divided by size is a Newtonian mass-size family",
            "raw stellar, atomic, and molecular phase main effects are nonqualifying",
            "line width, dynamical/dark mass, lensing mass, identity, and per-galaxy coefficients are forbidden circular shortcuts",
        ],
        "not_rejected": [
            "resolved multi-phase baryonic spatial geometry",
            "ionized or plasma composition absent from both galaxy samples",
            "an action-derived universal composition coupling with a prior quantitative prediction",
            "all composition-dependent gravity theories, dark-matter theories, or general relativity",
        ],
        "why_item_complete": (
            "Attempt 1 preserved a substantial but statistically nonpromoted PHANGS pattern. "
            "Attempt 2 transferred its exact interaction family without response-driven rewriting "
            "to 95 quality-passing galaxies with an independent H I-width target. The family is "
            "23.23% worse than the fixed baryonic Tully-Fisher control, is selected in zero of five "
            "unrestricted folds, loses in every frozen stratum and uncertainty envelope, and has "
            "permutation p=0.882. The one zero published uncertainty prevents a formally clean pass "
            "or reject receipt but cannot turn this broad negative replay into evidence for promotion. "
            "Retuning either opened response set would invalidate the counterexample. Item 8 supplies "
            "the next materially distinct variables: field gradients and curvature."
        ),
        "counts": {
            "attempts": 2,
            "exploration_galaxies": 129,
            "quality_passing_galaxies": 126,
            "quality_failures": 3,
            "permutations": 998,
            "reserved_confirmation_galaxies": 45,
            "confirmation_accesses": 0,
            "paid_model_calls": 0,
        },
        "claim_boundaries": {
            "all_baryonic_composition_theories_rejected": False,
            "tested_global_phase_family_generalizes": False,
            "phangs_lead_independently_replicated": False,
            "phangs_lead_preserved_as_counterexample": True,
            "roadmap_item_7_complete": True,
            "roadmap_item_8_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 8 field gradients and curvature. Freeze acceleration-gradient, "
            "baryonic-potential-curvature, tidal-invariant, and higher-derivative candidates on "
            "fresh real systems before accessing their target responses."
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
        raise GravityItem7SynthesisError("Item 7 synthesis receipt drifted")


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

