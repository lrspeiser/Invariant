"""Deterministic synthesis closing the tested local Item 3 density families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
OUTPUT_PATH = "runs/gravity/roadmap/item-03-synthesis-v1.json"
ATTEMPTS = (
    {
        "attempt": 1,
        "path": "runs/gravity/roadmap/item-03-surface-volume-density-v1.json",
        "file_sha256": "5742559f50abb7939ef68f909870c9d8c66e0d9788d34aa7f7e1b0696d762bf4",
        "content_sha256": "a0dff681242e6264f95433833d19dede6394af5014f73165695cfafe1c9addd9",
        "decision": "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE",
    },
    {
        "attempt": 2,
        "path": "runs/gravity/roadmap/item-03-smooth-density-profiles-v2.json",
        "file_sha256": "8ec8309ee3e36b2d008b8687fa3c779b58de2a3fde187d0151cb726657521a64",
        "content_sha256": "31c992465afd3cbd5230412694856510621da0687632c35ead152cf05bce94ea",
        "decision": "REJECT_ITEM3_SMOOTH_DENSITY_CROSSOVER_EXPLORATION",
    },
)


class GravityItem3SynthesisError(RuntimeError):
    """Raised when an Item 3 input or synthesis invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem3SynthesisError("stable roadmap changed")
    evidence: list[dict[str, Any]] = []
    for expected in ATTEMPTS:
        path = root / str(expected["path"])
        if _sha256_file(path) != expected["file_sha256"]:
            raise GravityItem3SynthesisError(
                f"Item 3 attempt-{expected['attempt']} file changed"
            )
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("content_sha256") != expected["content_sha256"]:
            raise GravityItem3SynthesisError(
                f"Item 3 attempt-{expected['attempt']} content changed"
            )
        if receipt.get("decision") != expected["decision"]:
            raise GravityItem3SynthesisError(
                f"Item 3 attempt-{expected['attempt']} decision changed"
            )
        confirmation_accesses = int(
            receipt.get("compute", {}).get(
                "confirmation_accesses",
                receipt.get("compute_and_access", {}).get("confirmation_accesses", 0),
            )
        )
        if confirmation_accesses != 0:
            raise GravityItem3SynthesisError("an Item 3 confirmation boundary was opened")
        evidence.append(
            {
                "attempt": expected["attempt"],
                "path": expected["path"],
                "file_sha256": expected["file_sha256"],
                "content_sha256": expected["content_sha256"],
                "decision": expected["decision"],
                "confirmation_accesses": confirmation_accesses,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-synthesis-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_03_SURFACE_VERSUS_VOLUME_DENSITY_SYNTHESIS",
        "item_number": 3,
        "decision": "REJECT_ITEM3_TESTED_LOCAL_SURFACE_VOLUME_DENSITY_FAMILIES_ADVANCE_ITEM4",
        "evidence": evidence,
        "scope_rejected": [
            "equivalent-mass dual surface/volume transition locations, widths, overlaps, and area asymmetries",
            "group-light surface/volume quantile-density features for direct velocity dispersion",
            "smooth local effective surface and volume amplitudes derived from directly observed galaxy and cluster baryonic profiles",
            "local density amplitude, scale contrast, acceleration interaction, and transition-crossover combinations",
        ],
        "equivalence_regions_retained": [
            "u_surface/u_volume=3/D_M is an Item 1 mass-dimension rewrite",
            "smooth log(u_surface/u_volume)=log(3H/(2r)) is a local profile-scale rewrite",
            "binary galaxy/cluster labels are population proxies and never qualifying causes",
        ],
        "not_rejected": [
            "nonlocal interior/exterior baryonic kernels",
            "baryonic boundary or edge terms",
            "environmental density and external fields",
            "density-dependent action or field-equation operators",
            "all modified-gravity or dark-matter theories",
        ],
        "why_item_complete": "Attempt 1 tests coarse object summaries and fresh direct group dynamics; attempt 2 repairs the representation with smooth direct radial galaxy and X-COP profiles. The repaired local families still fail within both populations, lose to baselines, select a forbidden population label, fail permutation, and fail stellar-baryon robustness. Further local retuning would reuse opened responses; materially nonlocal density ideas belong to later roadmap items.",
        "counts": {
            "attempts": 2,
            "declared_model_families": 20,
            "paid_model_calls": 0,
            "confirmation_accesses": 0,
            "pseudorandom_candidates": 0,
        },
        "claim_boundaries": {
            "all_surface_volume_density_theories_rejected": False,
            "local_frozen_families_rejected": True,
            "roadmap_item_3_complete": True,
            "roadmap_item_4_authorized_next": True,
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
        },
        "next_action": "Advance to Item 4 baryonic compactness. Freeze mass-to-size and potential-depth variables before accessing a new real response, while retaining Item 3 equivalence regions in the failure-space database.",
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
    existing = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
    if existing != build_receipt(root):
        raise GravityItem3SynthesisError("Item 3 synthesis receipt drifted")


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
