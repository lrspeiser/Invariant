"""Hash-bound five-attempt closure for gravity roadmap Item 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

ROADMAP_PATH = "docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md"
ROADMAP_SHA256 = "b9db7e1aa76deaf28ce2e840cd3d7db4cf7a086b1c7ca473800780c264a0e772"
OUTPUT_PATH = "runs/gravity/roadmap/item-02-synthesis-v1.json"

ATTEMPT_BINDINGS = (
    {
        "attempt": 1,
        "path": "runs/gravity/roadmap/item-02-shape-anisotropy-v1.json",
        "file_sha256": "6a257d6086da4c6aaa0bbf0dc2db03be8ea89ff1b1755848a4d45dd57f29f76a",
        "content_sha256": "13bf111dac994e5d97661fd0d00a789796b7ccf627596b06cb7bc7b3bfdb6835",
        "decision": "INCONCLUSIVE_ITEM2_SHAPE_ANISOTROPY",
    },
    {
        "attempt": 2,
        "path": "runs/gravity/roadmap/item-02-wise-multipoles-v2.json",
        "file_sha256": "4b550a901a133440869b7db8b3248c1252fc168144d13fd36f91f2a6defbc745",
        "content_sha256": "5bee70308b84d35d324ef938e4c162437a841d03b4905565a222a2f2c2490b35",
        "decision": "INCONCLUSIVE_ITEM2_WISE_MULTIPOLES",
    },
    {
        "attempt": 3,
        "path": "runs/gravity/roadmap/item-02-stellar-multipoles-v3.json",
        "file_sha256": "018656dd24b667b4b06fc13808216e17780286e26372d23139ac062dcc250465",
        "content_sha256": "7683167d0cbaa2a0adcff5ac61de503196c5facc7a7ad76fb0533c004344ec6e",
        "decision": "INCONCLUSIVE_ITEM2_STELLAR_MULTIPOLES",
    },
    {
        "attempt": 4,
        "path": "runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4.json",
        "file_sha256": "fea2bd9461242e3ea14a393f97250e1972b41c84ad0c09d4900275e131a76818",
        "content_sha256": "3046daec435b42fcd8381c7664f642f58a2905c3c3531dced8472b552430d521",
        "decision": "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE",
    },
    {
        "attempt": 5,
        "path": "runs/gravity/roadmap/item-02-axes-group-geometry-v5.json",
        "file_sha256": "bff8bc67c27ba27cd1d4c9ac4662920ccc8248a342e9771e7169f71def298f6e",
        "content_sha256": "bb4df4b26bbf211de84837a99f44bbcfe7f6e805c41ce0e5c393265c2975e44d",
        "decision": "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY_QUALITY_GATE",
    },
)


class GravityItem2SynthesisError(RuntimeError):
    """Raised when an Item 2 evidence binding or scoped conclusion drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_attempts(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for binding in ATTEMPT_BINDINGS:
        path = root / binding["path"]
        if _sha256_file(path) != binding["file_sha256"]:
            raise GravityItem2SynthesisError(f"attempt file changed: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("content_sha256") != binding["content_sha256"]:
            raise GravityItem2SynthesisError(f"attempt content changed: {path}")
        if value.get("decision") != binding["decision"]:
            raise GravityItem2SynthesisError(f"attempt decision changed: {path}")
        values.append(value)
    return values


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _sha256_file(root / ROADMAP_PATH) != ROADMAP_SHA256:
        raise GravityItem2SynthesisError("stable roadmap binding changed")
    attempts = _validate_attempts(root)
    if any(value["claims"].get("alternative_to_gr_established") for value in attempts):
        raise GravityItem2SynthesisError("an attempt contains an alternative-to-GR overclaim")
    if any(value["claims"].get("roadmap_item_2_complete") for value in attempts):
        raise GravityItem2SynthesisError("an attempt prematurely claims Item 2 completion")
    confirmation_accesses = (
        int(attempts[0]["counts"]["sparc_confirmation_evaluator_accesses"])
        + int(attempts[1]["counts"]["sparc_confirmation_evaluator_accesses"])
        + int(attempts[2]["counts"]["sparc_confirmation_evaluator_accesses"])
        + int(attempts[3]["counts"]["reserved_confirmation_target_accesses"])
        + int(attempts[4]["counts"]["reserved_confirmation_target_accesses"])
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item2-synthesis-receipt-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_02_SHAPE_ANISOTROPY_SCOPED_CLOSURE",
        "item_number": 2,
        "item_title": "Shape and anisotropy",
        "status": "REJECT",
        "decision": "REJECT_ITEM2_TESTED_PROJECTED_SHAPE_FAMILIES_ADVANCE_ITEM3",
        "roadmap_binding": {
            "path": ROADMAP_PATH,
            "file_sha256": ROADMAP_SHA256,
        },
        "attempt_bindings": [dict(binding) for binding in ATTEMPT_BINDINGS],
        "tested_scope": {
            "populations": [
                "SPARC disk galaxies",
                "CLASH galaxy clusters",
                "MaNGA ellipticals and S0 galaxies",
                "AXES-SDSS galaxy groups",
            ],
            "tracers": [
                "SPARC inclination and 3.6-micron concentration",
                "unWISE W1 stellar light",
                "CLASH X-ray morphology",
                "CLASH member stellar mass",
                "MaNGA resolved stellar mass",
                "AXES member r-band luminosity",
            ],
            "responses": [
                "Item 1 galaxy/cluster development coefficient",
                "MaNGA direct aperture stellar dynamics",
                "AXES member-redshift group dynamics",
            ],
            "feature_families": [
                "global projected axis ratio and concentration",
                "two-dimensional quadrupole, m3, and m4",
                "target-blind comparable stellar multipoles",
                "radially resolved and nonlocal profile differences",
                "group graph filamentarity and boundary geometry",
            ],
            "model_scope": "the frozen finite linear, quadratic, interaction, ridge, and nested-selection families in attempts 1 through 5",
        },
        "coverage_checks": {
            "attempt_receipts_hash_bound": True,
            "comparable_stellar_tracer_control_tested": True,
            "direct_dynamics_response_tested": True,
            "global_projected_shape_tested": True,
            "graph_filamentarity_tested": True,
            "intermediate_scale_groups_tested": True,
            "radial_nonlocal_shape_tested": True,
            "real_clusters_tested": True,
            "real_galaxies_tested": True,
        },
        "survivor_checks": {
            "any_attempt_eligible_for_confirmation": False,
            "any_family_beats_strongest_nonshape_baseline_in_every_required_population": False,
            "any_family_positive_within_every_required_population_and_overlap": False,
            "any_family_survives_all_frozen_robustness_controls": False,
            "any_projected_shape_family_establishes_a_cause": False,
        },
        "boundary_checks": {
            "all_confirmation_boundaries_untouched": confirmation_accesses == 0,
            "direct_lensing_likelihood_evaluations": 0,
            "paid_model_calls": 0,
            "sequential_G6_G7_G8_advanced": False,
            "total_confirmation_target_accesses": confirmation_accesses,
        },
        "failure_space": [
            {
                "family": "global_axis_ratio_and_concentration",
                "reason": "separates populations but has negative within-population prediction and fails shared-feature overlap",
                "attempts": [1],
            },
            {
                "family": "global_low_order_multipoles_from_mismatched_tracers",
                "reason": "loses to the support proxy within galaxies, clusters, overlap, and the independent bar-sign check",
                "attempts": [2],
            },
            {
                "family": "global_low_order_multipoles_from_comparable_stellar_tracers",
                "reason": "representation is independently validated but response prediction remains negative within both populations",
                "attempts": [3],
            },
            {
                "family": "radially_resolved_and_nonlocal_projected_stellar_shape",
                "reason": "loses to mass-size and morphology controls in direct MaNGA dynamics and fails all response variants",
                "attempts": [4],
            },
            {
                "family": "intermediate_group_multipoles_and_graph_filamentarity",
                "reason": "adds no significant robust increment beyond luminosity, size, richness, redshift, and environment",
                "attempts": [5],
            },
        ],
        "scoped_rejection": {
            "rejected": "Projected baryonic shape summaries in the five frozen families are not a sufficient universal hidden variable for the tested galaxy-to-group-to-cluster gravitational responses.",
            "not_rejected": [
                "intrinsic three-dimensional shape or velocity anisotropy not measured by these projections",
                "action-level tensor, torsion, nonmetric, or polarization theories",
                "new nonlocal operators not behaviorally equivalent to the tested summaries",
                "filament dynamics or lensing responses not represented by cleaned group membership",
                "surface-density, volume-density, pressure, thermodynamic, time, or environmental causes in later roadmap items",
            ],
            "reason_to_advance": "Five target-blind real-data attempts now cover the roadmap's required global, comparable-tracer, radial/nonlocal, and intermediate/filamentary branches without a qualifying survivor; further tuning on the same responses would mine noise rather than test a materially new shape cause.",
        },
        "creativity": {
            "label": "scoped_failure_space_synthesis_of_known_shape_and_dynamics_tests",
            "historical_novelty_established": False,
            "known_formula_or_rewrite": False,
            "new_theory_claimed": False,
        },
        "claims": {
            "all_anisotropic_gravity_rejected": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
            "historical_novelty_established": False,
            "item_2_tested_scope_complete": True,
            "roadmap_item_3_authorized_by_order": True,
            "shape_can_never_matter": False,
        },
        "next_action": "Begin Item 3 surface-versus-volume density with a frozen dimensionless derivation and a real galaxy/group/cluster test; retain all Item 2 failures as forbidden/equivalent search regions and keep every Item 2 confirmation set sealed.",
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    validate_receipt(receipt, root=root)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2SynthesisError("synthesis content hash changed")
    if value.get("decision") != "REJECT_ITEM2_TESTED_PROJECTED_SHAPE_FAMILIES_ADVANCE_ITEM3":
        raise GravityItem2SynthesisError("unexpected Item 2 synthesis decision")
    if value.get("status") != "REJECT":
        raise GravityItem2SynthesisError("Item 2 closure is not a scoped REJECT")
    if not all(bool(item) for item in value["coverage_checks"].values()):
        raise GravityItem2SynthesisError("Item 2 coverage is incomplete")
    if any(bool(item) for item in value["survivor_checks"].values()):
        raise GravityItem2SynthesisError("a projected-shape survivor forbids closure")
    boundaries = value["boundary_checks"]
    if not boundaries["all_confirmation_boundaries_untouched"]:
        raise GravityItem2SynthesisError("an Item 2 confirmation boundary was opened")
    if boundaries["total_confirmation_target_accesses"] != 0:
        raise GravityItem2SynthesisError("confirmation access count is nonzero")
    if value["claims"]["all_anisotropic_gravity_rejected"]:
        raise GravityItem2SynthesisError("synthesis overclaims all anisotropic gravity")
    if value["claims"]["alternative_to_gr_established"]:
        raise GravityItem2SynthesisError("synthesis overclaims an alternative to GR")
    _validate_attempts(root.resolve())


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_receipt(root)
    path = root / OUTPUT_PATH
    path.write_bytes(canonical_json_bytes(receipt))
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.check:
        path = root / OUTPUT_PATH
        stored = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(stored, root=root)
        if build_receipt(root) != stored:
            raise GravityItem2SynthesisError("synthesis is not an exact rebuild")
        return 0
    path = write_receipt(root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
