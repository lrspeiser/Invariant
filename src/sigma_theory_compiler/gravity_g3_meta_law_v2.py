"""G3 fixed-shrinkage repair after nested shrinkage failed to transfer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from .gravity_g0_experiment import load_config as load_g0_config
from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g3_meta_law import (
    _best_local_templates,
    _fit_model,
    _fold_map,
    _nearest_training_galaxy,
    _template_prediction,
    prepare_packets,
    project_formula,
)
from .gravity_g3_meta_law import (
    validate_receipt as validate_v1_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g3-fixed-shrinkage-receipt-2.0"
CONFIG_SCHEMA = "invariant-gravity-g3-fixed-shrinkage-config-2.0"
CONFIG_PATH = "configs/gravity_g3_meta_law_v2.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g3_meta_law_v2.py"
TEST_PATH = "tests/test_gravity_g3_meta_law_v2.py"
OUTPUT_PATH = "runs/gravity/g3/galaxy-formula-meta-law-v2.json"


class GravityG3FixedShrinkageError(ValueError):
    """The fixed-shrinkage G3 repair or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the G3-v2 contract and validate its blocked v1 predecessor."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG3FixedShrinkageError("G3-v2 config schema changed")
    predecessor = config.get("predecessor_binding")
    if not isinstance(predecessor, Mapping):
        raise GravityG3FixedShrinkageError("G3-v2 predecessor is missing")
    path = root / str(predecessor["path"])
    if _file_sha256(path) != predecessor.get("file_sha256"):
        raise GravityG3FixedShrinkageError("G3-v2 predecessor file changed")
    receipt = _load_json(path)
    validate_v1_receipt(receipt, root=root)
    if receipt.get("content_sha256") != predecessor.get("content_sha256"):
        raise GravityG3FixedShrinkageError("G3-v2 predecessor content changed")
    if receipt.get("decision") != predecessor.get("required_decision"):
        raise GravityG3FixedShrinkageError("G3-v2 predecessor decision changed")
    if receipt.get("counts", {}).get("predicted_galaxies") != predecessor.get(
        "required_predicted_galaxies"
    ):
        raise GravityG3FixedShrinkageError("G3-v2 predecessor population changed")
    model = config.get("learned_residual", {})
    if model.get("fixed_shrinkage") != "0.3" or model.get("n_estimators") != 512:
        raise GravityG3FixedShrinkageError("G3-v2 fixed model changed")
    cv = config.get("whole_galaxy_cross_validation", {})
    if (
        cv.get("galaxy_id_available_to_model") is not False
        or cv.get("heldout_vobs_available_to_model_or_formula_projection") is not False
    ):
        raise GravityG3FixedShrinkageError("G3-v2 target-blind boundary changed")
    disclosure = config.get("diagnostic_disclosure", {})
    if (
        disclosure.get("fixed_shrinkage_selected_after_inspecting_g3_v1_outer_fold_results")
        is not True
        or disclosure.get("result_is_independent_confirmation") is not False
    ):
        raise GravityG3FixedShrinkageError("G3-v2 disclosure changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG3FixedShrinkageError("G3-v2 permits confirmation access")
    return config


def build_receipt(root: Path, *, outer_fold_limit: int | None = None) -> dict[str, Any]:
    """Evaluate the disclosed fixed-shrinkage repair on all whole-galaxy folds."""

    root = root.resolve()
    config = load_config(root)
    packets = prepare_packets(root, config)
    g0 = load_g0_config(root)
    a0 = float(
        next(item for item in g0["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    alpha = float(config["learned_residual"]["fixed_shrinkage"])
    g2 = _load_json(root / str(config["g2_binding"]["path"]))
    classes = g2["structural_classes"]
    templates = _best_local_templates(root)
    fold_count = int(config["whole_galaxy_cross_validation"]["outer_folds"])
    assignments = _fold_map(
        [row["galaxy"].name for row in packets],
        str(config["whole_galaxy_cross_validation"]["outer_salt"]),
        fold_count,
    )
    folds_to_run = fold_count if outer_fold_limit is None else min(outer_fold_limit, fold_count)
    results: dict[str, dict[str, Any]] = {}
    fold_results = []
    for outer_fold in range(folds_to_run):
        training = [row for row in packets if assignments[row["galaxy"].name] != outer_fold]
        heldout = [row for row in packets if assignments[row["galaxy"].name] == outer_fold]
        model = _fit_model(training, config)
        training_names = {row["galaxy"].name for row in training}
        library = [
            row for row in classes if training_names.intersection(row["member_galaxies"])
        ]
        constant_delta = float(np.median(np.concatenate([row["delta"] for row in training])))
        class_counts: Counter[str] = Counter()
        for packet in heldout:
            learned_delta = model.predict(packet["model_matrix"])
            target_v2 = (
                packet["rar2"] + packet["arrays"]["radius"] * a0 * alpha * learned_delta
            )
            direct_invalid = int(np.sum(~np.isfinite(target_v2) | (target_v2 <= 0)))
            projected = project_formula(packet, target_v2, library)
            candidate_prediction = np.sqrt(projected.pop("prediction2"))
            nearest = _nearest_training_galaxy(packet, training)
            nearest_name = nearest["galaxy"].name
            nearest_prediction, nearest_invalid = _template_prediction(
                packet, templates[nearest_name]
            )
            constant_v2 = (
                packet["rar2"]
                + packet["arrays"]["radius"] * a0 * constant_delta
            )
            constant_invalid = int(np.sum(~np.isfinite(constant_v2) | (constant_v2 <= 0)))
            predictions = {
                "constant_residual": np.sqrt(
                    np.maximum(constant_v2, np.finfo(np.float64).tiny)
                ),
                "empirical_rar": np.sqrt(packet["rar2"]),
                "meta_direct": np.sqrt(
                    np.maximum(target_v2, np.finfo(np.float64).tiny)
                ),
                "meta_projected": candidate_prediction,
                "nearest_galaxy": nearest_prediction,
                "newtonian_baryons": np.sqrt(packet["arrays"]["vbar2"]),
            }
            scores = {
                name: score_predictions(
                    prediction,
                    packet["arrays"]["vobs"],
                    packet["arrays"]["sigma"],
                )
                for name, prediction in predictions.items()
            }
            predicted_class = str(projected["class_id"])
            if not any(
                predicted_class == row["class_id"]
                and training_names.intersection(row["member_galaxies"])
                for row in library
            ):
                raise GravityG3FixedShrinkageError(
                    "G3-v2 projected a class absent from training"
                )
            class_counts[predicted_class] += 1
            results[packet["galaxy"].name] = {
                "candidate_prediction_sha256": canonical_sha256(
                    [format(float(value), ".15e") for value in candidate_prediction]
                ),
                "constant_invalid_prediction2": constant_invalid,
                "direct_invalid_prediction2": direct_invalid,
                "fold": outer_fold,
                "nearest_galaxy_baseline_source": nearest_name,
                "nearest_galaxy_invalid_prediction2": nearest_invalid,
                "point_count": packet["galaxy"].count,
                "predicted_formula": projected,
                "scores": scores,
                "target_best_g1_class_id_evaluator_only": templates[packet["galaxy"].name][
                    "class_id"
                ],
                "target_class_exact_match": (
                    predicted_class == templates[packet["galaxy"].name]["class_id"]
                ),
            }
        fold_results.append(
            {
                "fold": outer_fold,
                "fixed_alpha": _metric(alpha),
                "heldout_galaxies": len(heldout),
                "predicted_class_counts": dict(sorted(class_counts.items())),
                "training_galaxies": len(training),
                "training_only_class_library": len(library),
            }
        )
    ordered_names = [row["galaxy"].name for row in packets if row["galaxy"].name in results]
    aggregate_scores = {}
    for prediction_name in (
        "meta_projected",
        "meta_direct",
        "newtonian_baryons",
        "empirical_rar",
        "constant_residual",
        "nearest_galaxy",
    ):
        chi = sum(float(results[name]["scores"][prediction_name]["chi_square"]) for name in ordered_names)
        rows = sum(int(results[name]["scores"][prediction_name]["row_count"]) for name in ordered_names)
        aggregate_scores[prediction_name] = {
            "chi_square": _metric(chi),
            "row_count": rows,
        }
    candidate_chi = float(aggregate_scores["meta_projected"]["chi_square"])
    gains = {
        name: _metric(1.0 - candidate_chi / float(aggregate_scores[name]["chi_square"]))
        for name in config["baselines"]
    }
    minimum_gain = float(config["admission"]["minimum_fractional_chi_square_gain_over_every_baseline"])
    full_run = outer_fold_limit is None and len(results) == len(packets)
    invalid = sum(row["direct_invalid_prediction2"] for row in results.values())
    passed = (
        full_run
        and invalid == 0
        and all(float(value) >= minimum_gain for value in gains.values())
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G3",
        "decision": "PASS_G3_FIXED_SHRINKAGE_META_LAW" if passed else "BLOCK_G3_FIXED_SHRINKAGE",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "g4_universal_law_authorized": passed,
            "historical_novelty_established": False,
            "independent_confirmation_completed": False,
            "universal_zero_local_constant_law_found": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "confirmation_evaluator_accesses": 0,
            "exact_target_class_matches": sum(
                int(row["target_class_exact_match"]) for row in results.values()
            ),
            "invalid_direct_prediction2": invalid,
            "predicted_galaxies": len(results),
            "predicted_points": sum(row["point_count"] for row in results.values()),
            "unique_predicted_classes": len(
                {row["predicted_formula"]["class_id"] for row in results.values()}
            ),
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "estimator": {
            **config["learned_residual"],
            "runtime_sklearn_version": sklearn.__version__,
        },
        "folds": fold_results,
        "galaxies": [{"galaxy": name, **results[name]} for name in sorted(results)],
        "predecessor": {
            "content_sha256": config["predecessor_binding"]["content_sha256"],
            "decision": config["predecessor_binding"]["required_decision"],
            "path": config["predecessor_binding"]["path"],
        },
        "scores": {
            "aggregate": aggregate_scores,
            "fractional_gain_of_projected_meta_law": gains,
            "minimum_required_gain": _metric(minimum_gain),
        },
        "limitations": [
            "Alpha=0.3 was selected after inspecting the same exploration outer folds, so this is a disclosed reproducibility/model-development result rather than independent validation.",
            "The meta-law is a phenomenological tree ensemble around a known RAR base, not a compact first-principles equation.",
            "The generated per-galaxy formulas still carry two coefficients and therefore are not the G4 zero-local-constant law.",
            "Any PASS authorizes G4 construction only; confirmation remains sealed.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Validate the checked G3 fixed-shrinkage result."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG3FixedShrinkageError("G3-v2 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG3FixedShrinkageError("G3-v2 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG3FixedShrinkageError("G3-v2 config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG3FixedShrinkageError(f"G3-v2 {key} binding changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG3FixedShrinkageError("G3-v2 records confirmation access")
    claims = receipt.get("claims", {})
    if claims.get("historical_novelty_established") is not False:
        raise GravityG3FixedShrinkageError("G3-v2 overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG3FixedShrinkageError("G3-v2 overstates confirmation")
    passed = receipt.get("decision") == "PASS_G3_FIXED_SHRINKAGE_META_LAW"
    if passed and (
        counts.get("predicted_galaxies") != int(config["g2_binding"]["required_galaxies"])
        or counts.get("invalid_direct_prediction2") != 0
        or claims.get("g4_universal_law_authorized") is not True
        or any(
            float(value)
            < float(config["admission"]["minimum_fractional_chi_square_gain_over_every_baseline"])
            for value in receipt.get("scores", {})
            .get("fractional_gain_of_projected_meta_law", {})
            .values()
        )
    ):
        raise GravityG3FixedShrinkageError("G3-v2 PASS is unsupported")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG3FixedShrinkageError(
                f"refusing to overwrite immutable receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--outer-fold-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root, outer_fold_limit=args.outer_fold_limit)
    if args.outer_fold_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "decision": receipt["decision"],
                "predicted_galaxies": receipt["counts"]["predicted_galaxies"],
                "projected_chi_square": receipt["scores"]["aggregate"]["meta_projected"][
                    "chi_square"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G3_FIXED_SHRINKAGE_META_LAW" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG3FixedShrinkageError",
    "build_receipt",
    "load_config",
    "validate_receipt",
]
