"""Deterministic synthesis closing the tested Item 4 compactness families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT_PATH = "runs/gravity/roadmap/item-04-baryonic-compactness-v1.json"
ATTEMPT_FILE_SHA256 = "221370e242e64c4f78304209a555786e4e65a2798969dbb9622b006f69b762e0"
ATTEMPT_CONTENT_SHA256 = "9717a7ee9dd3b9275ff50dcc4f05d19299c7f0f7f77d0956c8cbc0f7e775096b"
ATTEMPT_DECISION = "REJECT_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-04-synthesis-v1.json"


class GravityItem4SynthesisError(RuntimeError):
    """Raised when an Item 4 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem4SynthesisError("stable roadmap changed")
    attempt_path = root / ATTEMPT_PATH
    if _sha256_file(attempt_path) != ATTEMPT_FILE_SHA256:
        raise GravityItem4SynthesisError("Item 4 attempt file changed")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt.get("content_sha256") != ATTEMPT_CONTENT_SHA256:
        raise GravityItem4SynthesisError("Item 4 attempt content changed")
    if attempt.get("decision") != ATTEMPT_DECISION:
        raise GravityItem4SynthesisError("Item 4 attempt decision changed")
    if int(attempt["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem4SynthesisError("Item 4 confirmation boundary was opened")
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item4-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_04_BARYONIC_COMPACTNESS_SYNTHESIS",
        "item_number": 4,
        "decision": "REJECT_ITEM4_TESTED_PROJECTED_BARYONIC_COMPACTNESS_FAMILIES_ADVANCE_ITEM5",
        "evidence": {
            "path": ATTEMPT_PATH,
            "file_sha256": ATTEMPT_FILE_SHA256,
            "content_sha256": ATTEMPT_CONTENT_SHA256,
            "decision": ATTEMPT_DECISION,
            "exploration_groups": 52,
            "confirmation_accesses": 0,
            "permutation_p_value": attempt["response"]["permutation_test"]["p_value"],
        },
        "scope_rejected": [
            "fixed-light-to-mass projected mass-to-Rrms and acceleration compactness rewrites",
            "softened pairwise projected binding energy and gravitational-radius structure",
            "light-centroid potential depth, inner-versus-outer potential contrast, potential dispersion, and their frozen interaction",
            "these variables as incremental predictors of fresh optical-group velocity dispersion beyond mass, size, richness, redshift, and environment",
        ],
        "equivalence_regions_retained": [
            "log(GM/(R*c^2)) and log(GM/(R^2*g_dagger)) are algebraic mass-size rewrites",
            "sqrt(G*K_pair/M) is the known pairwise virial velocity scale",
            "R_g=M^2/K_pair is the known gravitational-radius construction",
            "a fit driven by richness, catalog identity, or mass-size controls is not a new compactness law",
        ],
        "not_rejected": [
            "three-dimensional compactness from complete gas, stellar, and molecular baryons",
            "radially resolved galaxy or cluster concentration laws with independent dynamics",
            "nonlocal interior-exterior kernels, baryonic boundaries, and field curvature",
            "compactness terms derived from an action or field equation",
            "all modified-gravity or dark-matter theories",
        ],
        "why_item_complete": (
            "The frozen test uses 52 entirely unused real groups and retains all of them. "
            "It compares explicit algebraic rewrites, known virial constructions, and a "
            "potentially new inner-outer potential synthesis against strong target-blind "
            "controls under whole-group nested holdout and 499 stratified permutations. "
            "The unrestricted selector chooses a qualifying family in only one of five "
            "folds; the qualifying selector loses in the richer stratum and has p=0.348. "
            "Retuning the same projected-light structures would reuse opened responses; "
            "materially nonlocal, boundary, or field-level constructions belong to later items."
        ),
        "counts": {
            "attempts": 1,
            "declared_model_families": 8,
            "exploration_groups": 52,
            "quality_failures": 0,
            "permutations": 499,
            "paid_model_calls": 0,
            "confirmation_accesses": 0,
        },
        "claim_boundaries": {
            "all_baryonic_compactness_theories_rejected": False,
            "tested_projected_light_compactness_families_rejected": True,
            "roadmap_item_4_complete": True,
            "roadmap_item_5_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 5 pressure support. Freeze a unified observable support "
            "relation and its real-data split before opening a new response; retain the "
            "Item 4 equivalence regions in the failure-space database."
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
        raise GravityItem4SynthesisError("Item 4 synthesis receipt drifted")


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
