"""G3 whole-galaxy meta-law and training-only G2 formula projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from .gravity_g0_experiment import (
    _empirical_rar,
    score_predictions,
)
from .gravity_g0_experiment import (
    load_config as load_g0_config,
)
from .gravity_g1_pilot import _baseline_contract, _binding, _file_sha256, _load_json, _metric
from .gravity_g1_pilot_v3 import FEATURE_IDS, baryonic_features
from .gravity_g2_equivalence import (
    canonical_formula_ir,
    collect_survivors,
    evaluate_component,
    structural_signature,
)
from .gravity_g2_equivalence import (
    load_config as load_g2_config,
)
from .gravity_g2_equivalence import (
    validate_receipt as validate_g2_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, assemble

SCHEMA = "invariant-gravity-g3-meta-law-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g3-meta-law-config-1.0"
CONFIG_PATH = "configs/gravity_g3_meta_law.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g3_meta_law.py"
TEST_PATH = "tests/test_gravity_g3_meta_law.py"
OUTPUT_PATH = "runs/gravity/g3/galaxy-formula-meta-law-v1.json"


class GravityG3MetaLawError(ValueError):
    """The G3 meta-law experiment or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load G3 and validate its G2 predecessor."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG3MetaLawError("G3 config schema changed")
    binding = config.get("g2_binding")
    if not isinstance(binding, Mapping):
        raise GravityG3MetaLawError("G3 G2 binding is missing")
    path = root / str(binding["path"])
    if _file_sha256(path) != binding.get("file_sha256"):
        raise GravityG3MetaLawError("G3 G2 file binding changed")
    receipt = _load_json(path)
    validate_g2_receipt(receipt, root=root)
    if receipt.get("content_sha256") != binding.get("content_sha256"):
        raise GravityG3MetaLawError("G3 G2 content binding changed")
    if receipt.get("decision") != binding.get("required_decision"):
        raise GravityG3MetaLawError("G3 G2 decision changed")
    cross_validation = config.get("whole_galaxy_cross_validation", {})
    if (
        cross_validation.get("galaxy_id_available_to_model") is not False
        or cross_validation.get("heldout_vobs_available_to_model_or_formula_projection")
        is not False
    ):
        raise GravityG3MetaLawError("G3 target-blind boundary changed")
    model = config.get("learned_residual", {})
    if model.get("n_estimators") != 512 or model.get("random_state") != 20260827:
        raise GravityG3MetaLawError("G3 estimator contract changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG3MetaLawError("G3 permits confirmation access")
    disclosure = config.get("diagnostic_disclosure", {})
    if disclosure.get("same_exploration_folds_inspected_during_model_development") is not True:
        raise GravityG3MetaLawError("G3 diagnostic disclosure changed")
    return config


def _fold_map(names: Sequence[str], salt: str, fold_count: int) -> dict[str, int]:
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
    )
    return {name: rank % fold_count for rank, name in enumerate(ordered)}


def _feature_summary(features: Mapping[str, np.ndarray], galaxy: Galaxy) -> np.ndarray:
    values: list[float] = []
    for feature_id in FEATURE_IDS:
        feature = features[feature_id]
        values.extend(
            [
                float(np.mean(feature)),
                float(np.std(feature)),
                *[float(value) for value in np.quantile(feature, [0.1, 0.5, 0.9])],
            ]
        )
    radius = np.asarray([float(value) for value in galaxy.radius], dtype=np.float64)
    values.extend(
        [
            float(np.log(radius[0])),
            float(np.log(radius[-1])),
            float(np.log(radius[-1] / radius[0])),
            float(np.log(float(galaxy.distance_mpc))),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def target_blind_matrix(galaxy: Galaxy, a0: float) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return local+global baryonic inputs; this function never reads V_obs or its error."""

    features = baryonic_features(galaxy, a0)
    local = np.column_stack([features[name] for name in FEATURE_IDS])
    summary = _feature_summary(features, galaxy)
    matrix = np.column_stack([local, np.tile(summary, (galaxy.count, 1))])
    if matrix.shape[1] != 52 or np.any(~np.isfinite(matrix)):
        raise GravityG3MetaLawError(f"invalid G3 target-blind matrix for {galaxy.name}")
    return matrix, summary, features


def prepare_packets(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prepare exploration packets while keeping evaluator targets out of model inputs."""

    root = root.resolve()
    g0 = load_g0_config(root)
    a0 = float(
        next(item for item in g0["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    packets = []
    for galaxy in assemble(root).exploration:
        contract = _baseline_contract(galaxy, g0)
        arrays = contract["arrays"]
        model_matrix, summary, features = target_blind_matrix(galaxy, a0)
        rar2 = _empirical_rar(arrays["radius"], arrays["vbar2"], a0) ** 2
        delta = (arrays["vobs"] ** 2 - rar2) / (arrays["radius"] * a0)
        packets.append(
            {
                "arrays": arrays,
                "a0": a0,
                "delta": delta,
                "features": features,
                "galaxy": galaxy,
                "model_matrix": model_matrix,
                "rar2": rar2,
                "summary": summary,
            }
        )
    if len(packets) != int(config["g2_binding"]["required_galaxies"]):
        raise GravityG3MetaLawError("G3 exploration galaxy count changed")
    if sum(row["galaxy"].count for row in packets) != int(
        config["g2_binding"]["required_points"]
    ):
        raise GravityG3MetaLawError("G3 exploration point count changed")
    return packets


def _model(config: Mapping[str, Any]) -> ExtraTreesRegressor:
    contract = config["learned_residual"]
    return ExtraTreesRegressor(
        n_estimators=int(contract["n_estimators"]),
        min_samples_leaf=int(contract["min_samples_leaf"]),
        max_features=float(contract["max_features"]),
        random_state=int(contract["random_state"]),
        n_jobs=int(contract["n_jobs"]),
    )


def _fit_model(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Any:
    matrix = np.vstack([row["model_matrix"] for row in rows])
    target = np.concatenate([row["delta"] for row in rows])
    return _model(config).fit(matrix, target)


def select_shrinkage(
    outer_train: Sequence[Mapping[str, Any]],
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    """Choose residual shrinkage using inner whole-galaxy predictions only."""

    cv = config["whole_galaxy_cross_validation"]
    salt = f"{cv['inner_salt']}|outer={outer_fold}"
    assignments = _fold_map(
        [row["galaxy"].name for row in outer_train], salt, int(cv["inner_folds"])
    )
    predictions: dict[str, np.ndarray] = {}
    for inner_fold in range(int(cv["inner_folds"])):
        training = [
            row for row in outer_train if assignments[row["galaxy"].name] != inner_fold
        ]
        validation = [
            row for row in outer_train if assignments[row["galaxy"].name] == inner_fold
        ]
        model = _fit_model(training, config)
        for row in validation:
            predictions[row["galaxy"].name] = model.predict(row["model_matrix"])
    if len(predictions) != len(outer_train):
        raise GravityG3MetaLawError("G3 inner predictions are incomplete")
    observed = np.concatenate([row["arrays"]["vobs"] for row in outer_train])
    sigma = np.concatenate([row["arrays"]["sigma"] for row in outer_train])
    a0 = float(outer_train[0]["a0"])
    scores = []
    for alpha in config["learned_residual"]["shrinkage_grid"]:
        predicted = []
        invalid = 0
        for row in outer_train:
            delta = predictions[row["galaxy"].name]
            prediction2 = (
                row["rar2"] + row["arrays"]["radius"] * a0 * float(alpha) * delta
            )
            invalid += int(np.sum(~np.isfinite(prediction2) | (prediction2 <= 0)))
            predicted.append(np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny)))
        score = score_predictions(np.concatenate(predicted), observed, sigma)
        scores.append(
            {
                "alpha": _metric(float(alpha)),
                "chi_square": score["chi_square"],
                "invalid_prediction2": invalid,
            }
        )
    eligible = [row for row in scores if row["invalid_prediction2"] == 0]
    if not eligible:
        raise GravityG3MetaLawError("all G3 shrinkage values make invalid predictions")
    selected = min(eligible, key=lambda row: (float(row["chi_square"]), float(row["alpha"])))
    return float(selected["alpha"]), scores


def _component_values(
    component: Mapping[str, Any],
    packet: Mapping[str, Any],
    normalized: bool,
) -> np.ndarray:
    if not normalized:
        features = packet["features"]
    else:
        features = {}
        for name, values in packet["features"].items():
            low = float(np.min(values))
            high = float(np.max(values))
            if high <= low:
                raise GravityG3MetaLawError("constant feature reached formula projection")
            features[name] = -1.0 + 2.0 * (values - low) / (high - low)
    return evaluate_component(component, features)


def formula_basis(
    formula_ir: Mapping[str, Any], packet: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    normalized = str(formula_ir["feature_normalization"]).startswith("within_galaxy")
    values = [
        _component_values(component, packet, normalized)
        for component in formula_ir["components"]
    ]
    columns = packet["arrays"]["radius"][:, None] * np.column_stack(values)
    base = (
        packet["rar2"]
        if formula_ir["base"] == "empirical_RAR"
        else packet["arrays"]["vbar2"]
    )
    return base, columns


def project_formula(
    packet: Mapping[str, Any],
    target_v2: np.ndarray,
    library: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose and fit a training-only G2 class to a learned target, never V_obs."""

    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    scale = max(float(np.mean(target_v2 * target_v2)), np.finfo(np.float64).tiny)
    component_cache: dict[str, np.ndarray] = {}
    for row in library:
        ir = row["canonical_ir"]
        normalized = str(ir["feature_normalization"]).startswith("within_galaxy")
        columns_list = []
        try:
            for component in ir["components"]:
                key = canonical_sha256(
                    {"component": component, "normalized": normalized}
                )
                if key not in component_cache:
                    component_cache[key] = _component_values(component, packet, normalized)
                columns_list.append(component_cache[key])
        except GravityG3MetaLawError:
            continue
        columns = packet["arrays"]["radius"][:, None] * np.column_stack(columns_list)
        base = packet["rar2"] if ir["base"] == "empirical_RAR" else packet["arrays"]["vbar2"]
        gram = columns.T @ columns
        determinant = float(np.linalg.det(gram))
        if determinant <= 1e-12 * float(gram[0, 0] * gram[1, 1]):
            continue
        coefficients = np.linalg.solve(gram, columns.T @ (target_v2 - base))
        prediction2 = base + columns @ coefficients
        if np.any(~np.isfinite(prediction2)) or np.any(prediction2 <= 0):
            continue
        error = float(np.mean((prediction2 - target_v2) ** 2) / scale)
        candidate = (error, str(row["class_id"]), coefficients, prediction2)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    if best is None:
        raise GravityG3MetaLawError(f"no G2 class projects for {packet['galaxy'].name}")
    return {
        "A_km2_s2_kpc": _metric(float(best[2][0])),
        "B_km2_s2_kpc": _metric(float(best[2][1])),
        "class_id": best[1],
        "normalized_projection_error": _metric(best[0]),
        "prediction2": best[3],
    }


def _best_local_templates(root: Path) -> dict[str, dict[str, Any]]:
    g2_config = load_g2_config(root)
    survivors = collect_survivors(root, g2_config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in survivors:
        grouped[row["galaxy"]].append(row)
    templates = {}
    for galaxy, rows in grouped.items():
        selected = min(
            rows,
            key=lambda row: (
                float(row["candidate"]["aggregate_score"]["chi_square"]),
                int(row["candidate"]["description_length"]["total_bits"]),
                row["candidate_id"],
            ),
        )
        candidate = selected["candidate"]
        semantic = [
            {key: value for key, value in component.items() if key != "llm_origin_assessment"}
            for component in candidate["components"]
        ]
        order = sorted(range(2), key=lambda index: canonical_json_bytes(semantic[index]))
        coefficients = []
        for index in order:
            key = "A_km2_s2_kpc" if index == 0 else "B_km2_s2_kpc"
            coefficients.append(float(np.median([float(fold[key]) for fold in candidate["folds"]])))
        templates[galaxy] = {
            "class_id": structural_signature(candidate),
            "coefficients": np.asarray(coefficients, dtype=np.float64),
            "formula_ir": canonical_formula_ir(candidate),
        }
    return templates


def _nearest_training_galaxy(
    packet: Mapping[str, Any], training: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    matrix = np.vstack([row["summary"] for row in training])
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    distances = np.sum(((matrix - packet["summary"][None, :]) / scale[None, :]) ** 2, axis=1)
    return training[int(np.argmin(distances))]


def _template_prediction(
    packet: Mapping[str, Any], template: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    base, columns = formula_basis(template["formula_ir"], packet)
    prediction2 = base + columns @ template["coefficients"]
    invalid = int(np.sum(~np.isfinite(prediction2) | (prediction2 <= 0)))
    return np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny)), invalid


def build_receipt(root: Path, *, outer_fold_limit: int | None = None) -> dict[str, Any]:
    """Run nested whole-galaxy meta-law evaluation and project to G2 classes."""

    root = root.resolve()
    config = load_config(root)
    packets = prepare_packets(root, config)
    g0 = load_g0_config(root)
    a0 = float(
        next(item for item in g0["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    g2 = _load_json(root / str(config["g2_binding"]["path"]))
    classes = g2["structural_classes"]
    templates = _best_local_templates(root)
    outer_count = int(config["whole_galaxy_cross_validation"]["outer_folds"])
    assignments = _fold_map(
        [row["galaxy"].name for row in packets],
        str(config["whole_galaxy_cross_validation"]["outer_salt"]),
        outer_count,
    )
    folds_to_run = outer_count if outer_fold_limit is None else min(outer_fold_limit, outer_count)
    results: dict[str, dict[str, Any]] = {}
    fold_results = []
    for outer_fold in range(folds_to_run):
        training = [row for row in packets if assignments[row["galaxy"].name] != outer_fold]
        heldout = [row for row in packets if assignments[row["galaxy"].name] == outer_fold]
        alpha, inner_scores = select_shrinkage(training, outer_fold, config)
        model = _fit_model(training, config)
        training_names = {row["galaxy"].name for row in training}
        library = [
            row for row in classes if training_names.intersection(row["member_galaxies"])
        ]
        if not library:
            raise GravityG3MetaLawError("G3 training-only class library is empty")
        constant_delta = float(np.median(np.concatenate([row["delta"] for row in training])))
        fold_class_counts: Counter[str] = Counter()
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
                raise GravityG3MetaLawError("heldout class was not supplied by training galaxies")
            fold_class_counts[predicted_class] += 1
            results[packet["galaxy"].name] = {
                "candidate_prediction_sha256": canonical_sha256(
                    [format(float(value), ".15e") for value in candidate_prediction]
                ),
                "direct_invalid_prediction2": direct_invalid,
                "fold": outer_fold,
                "nearest_galaxy_baseline_source": nearest_name,
                "nearest_galaxy_invalid_prediction2": nearest_invalid,
                "outer_training_galaxies": len(training),
                "point_count": packet["galaxy"].count,
                "predicted_formula": projected,
                "scores": scores,
                "target_best_g1_class_id_evaluator_only": templates[packet["galaxy"].name][
                    "class_id"
                ],
                "target_class_exact_match": (
                    predicted_class == templates[packet["galaxy"].name]["class_id"]
                ),
                "constant_invalid_prediction2": constant_invalid,
            }
        fold_results.append(
            {
                "fold": outer_fold,
                "heldout_galaxies": len(heldout),
                "inner_shrinkage_scores": inner_scores,
                "predicted_class_counts": dict(sorted(fold_class_counts.items())),
                "selected_alpha": _metric(alpha),
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
    baseline_names = list(config["baselines"])
    gains = {
        name: _metric(1.0 - candidate_chi / float(aggregate_scores[name]["chi_square"]))
        for name in baseline_names
    }
    minimum_gain = float(config["admission"]["minimum_fractional_chi_square_gain_over_every_baseline"])
    full_run = outer_fold_limit is None and len(results) == len(packets)
    invalid_candidate = sum(
        row["direct_invalid_prediction2"] for row in results.values()
    )
    passed = (
        full_run
        and invalid_candidate == 0
        and all(float(value) >= minimum_gain for value in gains.values())
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G3",
        "decision": "PASS_G3_WHOLE_GALAXY_META_LAW" if passed else "BLOCK_G3_META_LAW",
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
        "galaxies": [
            {"galaxy": name, **results[name]} for name in sorted(results)
        ],
        "scores": {
            "aggregate": aggregate_scores,
            "fractional_gain_of_projected_meta_law": gains,
            "minimum_required_gain": _metric(minimum_gain),
        },
        "limitations": [
            "The same exploration folds were inspected during model development; this is not independent confirmation.",
            "The ensemble predicts a phenomenological residual around the known RAR relation and is not a first-principles gravity equation.",
            "Each reported G2 class and two coefficients are generated from the learned target using baryonic inputs only, but the ensemble itself is not a compact analytic law.",
            "Exact recovery of a galaxy's evaluator-only best G1 class is diagnostic and not required because G2 found thousands of behaviorally redundant local solutions.",
            "A G3 PASS authorizes construction of a zero-local-constant G4 law; it is not itself that law.",
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
    """Validate the sealed G3 result and its full source chain."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG3MetaLawError("G3 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG3MetaLawError("G3 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG3MetaLawError("G3 config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG3MetaLawError(f"G3 {key} binding changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG3MetaLawError("G3 records confirmation access")
    claims = receipt.get("claims", {})
    if claims.get("historical_novelty_established") is not False:
        raise GravityG3MetaLawError("G3 overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG3MetaLawError("G3 overstates confirmation")
    passed = receipt.get("decision") == "PASS_G3_WHOLE_GALAXY_META_LAW"
    if passed and (
        counts.get("predicted_galaxies") != int(config["g2_binding"]["required_galaxies"])
        or claims.get("g4_universal_law_authorized") is not True
        or any(
            float(value)
            < float(config["admission"]["minimum_fractional_chi_square_gain_over_every_baseline"])
            for value in receipt.get("scores", {})
            .get("fractional_gain_of_projected_meta_law", {})
            .values()
        )
    ):
        raise GravityG3MetaLawError("G3 PASS is unsupported")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG3MetaLawError(f"refusing to overwrite immutable receipt: {path}")
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
    return 0 if receipt["decision"] == "PASS_G3_WHOLE_GALAXY_META_LAW" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG3MetaLawError",
    "build_receipt",
    "formula_basis",
    "load_config",
    "prepare_packets",
    "project_formula",
    "select_shrinkage",
    "target_blind_matrix",
    "validate_receipt",
]
