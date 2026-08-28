"""Deterministic synthesis closing the tested Item 5 pressure-support families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
ATTEMPT1_PATH = "runs/gravity/roadmap/item-05-pressure-support-v1.json"
ATTEMPT1_FILE_SHA256 = "bf7527d3ad72e2b3cdcbcef770ec2a051eaf4d58a200e40a9bbfbf53bf7de23c"
ATTEMPT1_CONTENT_SHA256 = "0a66e48046ad2eb35ee23a1710f825c8a61f26254691da8a846f90e7aaad891e"
ATTEMPT1_DECISION = "INCONCLUSIVE_ITEM5_PRESSURE_SUPPORT_QUALITY_GATE"
ATTEMPT2_PATH = "runs/gravity/roadmap/item-05-pressure-cross-support-v2.json"
ATTEMPT2_FILE_SHA256 = "f37bb3d5deb941e01fe2d1034892fab6626a9ef02d515a2de4c78009b3167835"
ATTEMPT2_CONTENT_SHA256 = "957ce9138b7b22bae5d7fba46284b8fb9147360b40a3ae2836dc7599b7afbfdb"
ATTEMPT2_DECISION = "REJECT_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION"
OUTPUT_PATH = "runs/gravity/roadmap/item-05-synthesis-v1.json"


class GravityItem5SynthesisError(RuntimeError):
    """Raised when an Item 5 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(
    root: Path,
    *,
    path: str,
    file_sha256: str,
    content_sha256: str,
    decision: str,
) -> dict[str, Any]:
    source = root / path
    if _sha256_file(source) != file_sha256:
        raise GravityItem5SynthesisError(f"bound Item 5 input changed: {path}")
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if receipt.get("content_sha256") != content_sha256:
        raise GravityItem5SynthesisError(f"bound Item 5 content changed: {path}")
    if receipt.get("decision") != decision:
        raise GravityItem5SynthesisError(f"bound Item 5 decision changed: {path}")
    return receipt


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem5SynthesisError("stable roadmap changed")
    attempt1 = _load_bound(
        root,
        path=ATTEMPT1_PATH,
        file_sha256=ATTEMPT1_FILE_SHA256,
        content_sha256=ATTEMPT1_CONTENT_SHA256,
        decision=ATTEMPT1_DECISION,
    )
    attempt2 = _load_bound(
        root,
        path=ATTEMPT2_PATH,
        file_sha256=ATTEMPT2_FILE_SHA256,
        content_sha256=ATTEMPT2_CONTENT_SHA256,
        decision=ATTEMPT2_DECISION,
    )
    confirmation_accesses = int(attempt1["counts"]["reserved_confirmation_target_accesses"]) + int(
        attempt2["counts"]["reserved_confirmation_target_accesses"]
    )
    if confirmation_accesses != 0:
        raise GravityItem5SynthesisError("Item 5 confirmation boundary was opened")
    if not attempt2["gate_checks"]["all_44_exploration_clusters_pass_frozen_quality"]:
        raise GravityItem5SynthesisError("Item 5 attempt 2 is not a clean quality result")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item5-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_05_PRESSURE_SUPPORT_SYNTHESIS",
        "item_number": 5,
        "decision": "REJECT_ITEM5_TESTED_OBSERVABLE_PRESSURE_SUPPORT_FAMILIES_ADVANCE_ITEM6",
        "evidence": [
            {
                "attempt": 1,
                "path": ATTEMPT1_PATH,
                "file_sha256": ATTEMPT1_FILE_SHA256,
                "content_sha256": ATTEMPT1_CONTENT_SHA256,
                "decision": ATTEMPT1_DECISION,
                "interpretation": "The dwarf-galaxy local/nonlocal lane is inconclusive because six of eleven profiles fail the frozen unsmoothed pressure representation.",
                "confirmation_accesses": attempt1["counts"][
                    "reserved_confirmation_target_accesses"
                ],
            },
            {
                "attempt": 2,
                "path": ATTEMPT2_PATH,
                "file_sha256": ATTEMPT2_FILE_SHA256,
                "content_sha256": ATTEMPT2_CONTENT_SHA256,
                "decision": ATTEMPT2_DECISION,
                "interpretation": "The materially different cluster thermal-to-collisionless lane passes representation quality but rejects every frozen pressure-coherence promotion gate.",
                "confirmation_accesses": attempt2["counts"][
                    "reserved_confirmation_target_accesses"
                ],
                "qualifying_r2": attempt2["primary"]["qualifying_selector"]["metrics"]["r2"],
                "permutation_p_value": attempt2["permutation"]["p_value"],
                "alternative_response_mse_improvement": attempt2["alternative_response_robustness"][
                    "qualifying_mse_improvement"
                ],
            },
        ],
        "scope_rejected": [
            "the exact unsmoothed LITTLE THINGS local pressure-slope representation as a reliable universal discovery pipeline",
            "the frozen 0.35-dex nonlocal slope, cumulative pressure-memory, and pressure-curvature extensions on the quality-passing dwarf subset",
            "fixed-aperture SPT thermal-energy scaling as an object-by-object predictor of held-out cluster galaxy dispersion",
            "pressure-per-filter-area compactness and the exact coherence/extent/phase interactions C*L, C*Q, C*R, Q*R, and C^2",
            "promotion of those families beyond self-similar, redshift, flexible-SZ, raw-observable, and compactness controls",
        ],
        "equivalence_regions_retained": [
            "the classical asymmetric-drift equation is known physics rather than a new gravity formula",
            "sigma proportional to (Y D_A^2 E(z))^(1/5) is a known self-similar scaling family",
            "pressure amplitude divided by an observed size is a compactness-style rewrite",
            "M500c, sigmaSPT, lensing mass, fitted halo mass, identity, and per-object coefficients are forbidden circular shortcuts",
        ],
        "not_rejected": [
            "classical pressure support when evaluated with an independently frozen smooth physical pressure model",
            "full radial X-ray/SZ pressure and temperature profiles with complete baryonic measurements",
            "separately measured thermal, turbulent, rotational, and collisionless support in one system",
            "action-derived nonlocal or modified-gravity pressure couplings",
            "all thermodynamic-state variables, modified-gravity theories, dark-matter theories, or general relativity",
        ],
        "why_item_complete": (
            "Attempt 1 exposes and preserves a concrete representation failure rather than silently "
            "smoothing after response access. Attempt 2 then supplies the required materially different "
            "real-data test: all 44 clusters pass quality, but qualifying coherence models have negative "
            "held-out R2, permutation p=0.43, no unrestricted-fold selections, and worse performance on "
            "92 alternative dispersion estimates. Retuning either opened response set would invalidate "
            "the gates. Radially resolved temperature, entropy, cooling, and pressure-gradient causes "
            "are the explicit scope of Item 6, so the exact tested Item 5 regions are closed and retained "
            "as counterexamples while materially new thermodynamic variables advance there."
        ),
        "counts": {
            "attempts": 2,
            "exploration_objects_selected": 55,
            "quality_passing_objects": 49,
            "quality_failures": 6,
            "radial_rows_attempt1": attempt1["counts"]["radial_rows"],
            "alternative_response_rows_attempt2": attempt2["counts"][
                "pooled_alternative_response_rows"
            ],
            "permutations": int(attempt1["counts"]["permutations"])
            + int(attempt2["counts"]["permutation_nested_cv_runs"]),
            "reserved_confirmation_objects": int(
                attempt1["counts"]["reserved_confirmation_galaxies"]
            )
            + int(attempt2["counts"]["reserved_confirmation_clusters"]),
            "confirmation_accesses": confirmation_accesses,
            "paid_model_calls": int(attempt1["counts"]["paid_model_calls"])
            + int(attempt2["counts"]["paid_model_calls"]),
        },
        "claim_boundaries": {
            "all_pressure_support_theories_rejected": False,
            "tested_observable_pressure_support_families_rejected": True,
            "attempt1_is_conclusive": False,
            "attempt2_is_clean_scoped_rejection": True,
            "roadmap_item_5_complete": True,
            "roadmap_item_6_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": (
            "Advance to Item 6 thermodynamic state. Before any new response access, freeze direct "
            "temperature, entropy, cooling-time, and pressure-gradient variables on a fresh real system; "
            "retain both Item 5 response sets and all 23 confirmations without retuning or opening them."
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
        raise GravityItem5SynthesisError("Item 5 synthesis receipt drifted")


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
