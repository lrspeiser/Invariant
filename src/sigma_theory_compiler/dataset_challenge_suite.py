"""Executable exact controls for intervention, noise, shift, and identifiability."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import prod
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creative_dataset_challenges.json"
OUTPUT_PATH = "runs/math/dataset-challenges/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/dataset_challenge_suite.py"
TEST_PATH = "tests/test_dataset_challenge_suite.py"
CONFIG_SCHEMA = "invariant-creative-dataset-challenges-2.0"
RESULT_SCHEMA = "invariant-creative-dataset-challenge-result-2.0"
_KINDS = ("intervention", "noisy", "shifted", "unidentifiable")
_TREATMENTS = {"do_treatment_0": Fraction(0), "do_treatment_1": Fraction(1)}


class DatasetChallengeError(ValueError):
    """A dataset challenge lost its executable calibration evidence."""


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DatasetChallengeError(f"{label} keys changed")


def _fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, str):
        raise DatasetChallengeError(f"{label} is not a rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise DatasetChallengeError(f"{label} is not rational") from error


def _normalized_sha256(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _affine_model(
    model: Mapping[str, Any], covariate: Fraction, treatment: Fraction = Fraction(0)
) -> Fraction:
    _strict_keys(
        model,
        {"covariate_coefficient", "intercept", "model_id", "treatment_effect"},
        "affine model",
    )
    return (
        _fraction(model["covariate_coefficient"], "covariate coefficient") * covariate
        + _fraction(model["intercept"], "affine intercept")
        + _fraction(model["treatment_effect"], "treatment effect") * treatment
    )


def _intervention_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        challenge,
        {"challenge_id", "kind", "mutation_model", "positive_model", "provenance", "records"},
        "intervention challenge",
    )
    records = challenge["records"]
    if not isinstance(records, list) or len(records) < 4:
        raise DatasetChallengeError("intervention challenge lacks records")
    parsed: dict[tuple[Fraction, str], Fraction] = {}
    for row in records:
        _strict_keys(row, {"covariate", "intervention", "outcome"}, "intervention record")
        intervention = row["intervention"]
        if intervention not in _TREATMENTS:
            raise DatasetChallengeError("intervention label is unsupported")
        key = (_fraction(row["covariate"], "intervention covariate"), intervention)
        if key in parsed:
            raise DatasetChallengeError("intervention records contain a duplicate cell")
        parsed[key] = _fraction(row["outcome"], "intervention outcome")
    covariates = sorted({covariate for covariate, _ in parsed})
    if set(parsed) != {(covariate, label) for covariate in covariates for label in _TREATMENTS}:
        raise DatasetChallengeError("interventions are not crossed at fixed covariates")

    positive = challenge["positive_model"]
    mutation = challenge["mutation_model"]
    positive_rows = sum(
        _affine_model(positive, covariate, _TREATMENTS[label]) == outcome
        for (covariate, label), outcome in parsed.items()
    )
    mutation_rows = sum(
        _affine_model(mutation, covariate, _TREATMENTS[label]) == outcome
        for (covariate, label), outcome in parsed.items()
    )
    contrasts = [
        parsed[(covariate, "do_treatment_1")] - parsed[(covariate, "do_treatment_0")]
        for covariate in covariates
    ]
    declared_effect = _fraction(positive["treatment_effect"], "positive treatment effect")
    if positive_rows != len(records) or set(contrasts) != {declared_effect}:
        raise DatasetChallengeError("positive intervention model or exact do-contrast failed")
    if mutation_rows == len(records):
        raise DatasetChallengeError("intervention mutation was not rejected")
    return {
        "actual_do_labeled_rows": len(records),
        "constant_do_contrast": str(declared_effect),
        "crossed_at_fixed_covariates": True,
        "mutation_exact_rows": mutation_rows,
        "observational_rows_relabelled_as_interventions": False,
        "positive_exact_rows": positive_rows,
    }


def _noisy_model(model: Mapping[str, Any], input_value: Fraction) -> Fraction:
    _strict_keys(model, {"intercept", "model_id", "slope"}, "noisy affine model")
    return _fraction(model["slope"], "noisy slope") * input_value + _fraction(
        model["intercept"], "noisy intercept"
    )


def _noise_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        challenge,
        {"challenge_id", "compatible_models", "kind", "mutation_model", "provenance", "records"},
        "noisy challenge",
    )
    records = challenge["records"]
    models = challenge["compatible_models"]
    if not isinstance(records, list) or not records or not isinstance(models, list) or len(models) < 2:
        raise DatasetChallengeError("noisy challenge lacks records or compatible models")
    parsed = []
    for row in records:
        _strict_keys(row, {"center", "input", "radius"}, "noisy record")
        radius = _fraction(row["radius"], "noise radius")
        if radius <= 0:
            raise DatasetChallengeError("noise intervals must have positive radii")
        parsed.append(
            (
                _fraction(row["input"], "noisy input"),
                _fraction(row["center"], "noisy center"),
                radius,
            )
        )
    model_ids = [model.get("model_id") for model in models if isinstance(model, Mapping)]
    if len(model_ids) != len(models) or len(model_ids) != len(set(model_ids)):
        raise DatasetChallengeError("noisy compatible model IDs are invalid")
    compatible = {}
    center_exact = {}
    prediction_vectors = []
    for model in models:
        predictions = tuple(_noisy_model(model, input_value) for input_value, _, _ in parsed)
        prediction_vectors.append(predictions)
        compatible[str(model["model_id"])] = all(
            center - radius <= prediction <= center + radius
            for prediction, (_, center, radius) in zip(predictions, parsed, strict=True)
        )
        center_exact[str(model["model_id"])] = sum(
            prediction == center
            for prediction, (_, center, _) in zip(predictions, parsed, strict=True)
        )
    mutation = challenge["mutation_model"]
    mutation_predictions = tuple(
        _noisy_model(mutation, input_value) for input_value, _, _ in parsed
    )
    mutation_compatible_rows = sum(
        center - radius <= prediction <= center + radius
        for prediction, (_, center, radius) in zip(mutation_predictions, parsed, strict=True)
    )
    if not all(compatible.values()) or len(set(prediction_vectors)) < 2:
        raise DatasetChallengeError("noise intervals do not retain distinct compatible models")
    if mutation_compatible_rows == len(parsed):
        raise DatasetChallengeError("noise mutation was not rejected")
    return {
        "center_exact_rows_by_model": center_exact,
        "distinct_point_predictions_retained": len(set(prediction_vectors)),
        "interval_compatible_models": sorted(compatible),
        "interval_rows": len(parsed),
        "mutation_compatible_rows": mutation_compatible_rows,
        "point_centers_treated_as_unique_truth": False,
        "positive_uncertainty_declared": True,
    }


def _shift_model(model: Mapping[str, Any], input_value: Fraction) -> Fraction:
    _strict_keys(
        model,
        {"alias_coefficient", "alias_roots", "intercept", "linear", "model_id", "quadratic"},
        "shift model",
    )
    roots = model["alias_roots"]
    if not isinstance(roots, list):
        raise DatasetChallengeError("shift alias roots are not an array")
    alias = _fraction(model["alias_coefficient"], "shift alias coefficient") * prod(
        input_value - _fraction(root, "shift alias root") for root in roots
    )
    return (
        _fraction(model["quadratic"], "shift quadratic coefficient") * input_value**2
        + _fraction(model["linear"], "shift linear coefficient") * input_value
        + _fraction(model["intercept"], "shift intercept")
        + alias
    )


def _shift_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        challenge,
        {
            "challenge_id",
            "invariant_model",
            "kind",
            "provenance",
            "records",
            "train_only_alias_model",
        },
        "shifted challenge",
    )
    records = challenge["records"]
    if not isinstance(records, list):
        raise DatasetChallengeError("shift records are not an array")
    parsed: dict[str, list[tuple[Fraction, Fraction]]] = {"deployment": [], "train": []}
    for row in records:
        _strict_keys(row, {"input", "outcome", "split"}, "shift record")
        split = row["split"]
        if split not in parsed:
            raise DatasetChallengeError("shift split is unsupported")
        parsed[split].append(
            (
                _fraction(row["input"], "shift input"),
                _fraction(row["outcome"], "shift outcome"),
            )
        )
    if not all(parsed.values()) or max(x for x, _ in parsed["train"]) >= min(
        x for x, _ in parsed["deployment"]
    ):
        raise DatasetChallengeError("deployment domain is not an explicit covariate shift")
    positive = challenge["invariant_model"]
    alias = challenge["train_only_alias_model"]
    positive_counts = {
        split: sum(_shift_model(positive, x) == y for x, y in rows)
        for split, rows in parsed.items()
    }
    alias_counts = {
        split: sum(_shift_model(alias, x) == y for x, y in rows)
        for split, rows in parsed.items()
    }
    if any(positive_counts[split] != len(parsed[split]) for split in parsed):
        raise DatasetChallengeError("shift-invariant positive model failed exact replay")
    if alias_counts["train"] != len(parsed["train"]) or alias_counts["deployment"] == len(
        parsed["deployment"]
    ):
        raise DatasetChallengeError("train-only shift alias was not exposed by deployment rows")
    return {
        "deployment_exact_rows": positive_counts["deployment"],
        "deployment_rows": len(parsed["deployment"]),
        "domain_shift_explicit": True,
        "train_exact_rows": positive_counts["train"],
        "train_only_alias_deployment_exact_rows": alias_counts["deployment"],
        "train_only_alias_train_exact_rows": alias_counts["train"],
        "train_rows": len(parsed["train"]),
    }


def _rational_vector(value: Any, label: str) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetChallengeError(f"{label} is not a nonempty array")
    return tuple(_fraction(item, label) for item in value)


def _unidentifiable_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        challenge,
        {
            "challenge_id",
            "compatible_models",
            "distinguishing_intervention_observed",
            "forced_choice_control",
            "kind",
            "observational_outcomes",
            "provenance",
        },
        "unidentifiable challenge",
    )
    observed = _rational_vector(challenge["observational_outcomes"], "observational outcome")
    models = challenge["compatible_models"]
    if not isinstance(models, list) or len(models) < 2:
        raise DatasetChallengeError("unidentifiable challenge lacks compatible mechanisms")
    observational = []
    interventions = []
    model_ids = []
    for model in models:
        _strict_keys(
            model,
            {"intervention_predictions", "model_id", "observational_predictions"},
            "compatible mechanism",
        )
        model_ids.append(model["model_id"])
        observational.append(
            _rational_vector(model["observational_predictions"], "model observation")
        )
        interventions.append(
            _rational_vector(model["intervention_predictions"], "model intervention")
        )
    if any(predictions != observed for predictions in observational):
        raise DatasetChallengeError("mechanisms are not observationally equivalent")
    if len(set(interventions)) < 2:
        raise DatasetChallengeError("mechanisms do not differ interventionally")
    if len(model_ids) != len(set(model_ids)) or challenge["forced_choice_control"] not in model_ids:
        raise DatasetChallengeError("unidentifiable model IDs or forced-choice control changed")
    if challenge["distinguishing_intervention_observed"] is not False:
        raise DatasetChallengeError("unidentifiable control contains distinguishing intervention data")
    return {
        "compatible_mechanisms": len(models),
        "forced_unique_mechanism_rejected": True,
        "intervention_predictions_diverge": True,
        "observational_predictions_equal": True,
        "required_conclusion": "UNDERDETERMINED_RETAIN_MULTIPLE_MECHANISMS",
    }


_EXECUTORS = {
    "intervention": _intervention_evidence,
    "noisy": _noise_evidence,
    "shifted": _shift_evidence,
    "unidentifiable": _unidentifiable_evidence,
}


def _build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _strict_keys(config, {"challenges", "schema_version"}, "dataset config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise DatasetChallengeError("dataset challenge config identity changed")
    challenges = config["challenges"]
    if not isinstance(challenges, list):
        raise DatasetChallengeError("dataset challenges are not an array")
    results = []
    kinds = set()
    challenge_ids = set()
    for challenge in challenges:
        if not isinstance(challenge, Mapping):
            raise DatasetChallengeError("dataset challenge is not an object")
        kind = challenge.get("kind")
        challenge_id = challenge.get("challenge_id")
        provenance = challenge.get("provenance")
        if kind not in _EXECUTORS:
            raise DatasetChallengeError("dataset challenge kind is unsupported")
        if (
            not isinstance(challenge_id, str)
            or not challenge_id.startswith("dataset.")
            or challenge_id in challenge_ids
            or not isinstance(provenance, str)
            or not provenance.startswith("synthetic_")
        ):
            raise DatasetChallengeError("dataset challenge identity or provenance changed")
        kinds.add(kind)
        challenge_ids.add(challenge_id)
        evidence = _EXECUTORS[kind](challenge)
        results.append(
            {
                "challenge_id": challenge_id,
                "evidence": evidence,
                "kind": kind,
                "mutation_control_rejected": True,
                "positive_control_passed": True,
                "provenance": provenance,
                "status": "PASS_EXECUTABLE_DATASET_CONTROL",
            }
        )
    if kinds != set(_KINDS) or len(results) != len(_KINDS):
        raise DatasetChallengeError("dataset challenge coverage changed")
    paths = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "source_bindings": {
            key: {"normalized_file_sha256": _normalized_sha256(root / path), "path": path}
            for key, path in sorted(paths.items())
        },
        "results": sorted(results, key=lambda item: item["kind"]),
        "summary": {
            "challenge_kinds": list(_KINDS),
            "mutation_controls_rejected": len(results),
            "positive_controls_passed": len(results),
            "status": "PASS_EXECUTABLE_DATASET_CHALLENGES",
            "total": len(results),
        },
        "claims": {
            "causal_effect_inferred_from_observational_rows": False,
            "dataset_formulas_established": False,
            "interval_centers_treated_as_exact_truth": False,
            "train_fit_establishes_deployment_validity": False,
            "unidentifiable_control_forced_to_one_model": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def run_dataset_challenges(root: Path) -> dict[str, Any]:
    """Execute and seal all four exact positive/mutation dataset controls."""

    receipt = _build_receipt(root)
    validate_dataset_challenges(receipt, root)
    return receipt


def validate_dataset_challenges(
    value: Mapping[str, Any], root: Path | None = None
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise DatasetChallengeError("dataset challenge result seal changed")
    _strict_keys(
        value,
        {"claims", "content_sha256", "results", "schema_version", "source_bindings", "summary"},
        "dataset receipt",
    )
    if value.get("schema_version") != RESULT_SCHEMA:
        raise DatasetChallengeError("dataset challenge result schema changed")
    summary = value.get("summary", {})
    _strict_keys(
        summary,
        {
            "challenge_kinds",
            "mutation_controls_rejected",
            "positive_controls_passed",
            "status",
            "total",
        },
        "dataset summary",
    )
    claims = value.get("claims", {})
    expected_claims = {
        "causal_effect_inferred_from_observational_rows",
        "dataset_formulas_established",
        "interval_centers_treated_as_exact_truth",
        "train_fit_establishes_deployment_validity",
        "unidentifiable_control_forced_to_one_model",
    }
    results = value.get("results")
    if (
        summary.get("challenge_kinds") != list(_KINDS)
        or summary.get("positive_controls_passed") != len(_KINDS)
        or summary.get("mutation_controls_rejected") != len(_KINDS)
        or summary.get("total") != len(_KINDS)
        or summary.get("status") != "PASS_EXECUTABLE_DATASET_CHALLENGES"
        or set(claims) != expected_claims
        or any(claims.get(key) is not False for key in expected_claims)
        or not isinstance(results, list)
        or [item.get("kind") for item in results] != list(_KINDS)
    ):
        raise DatasetChallengeError("dataset calibration boundary changed")
    for item in results:
        _strict_keys(
            item,
            {
                "challenge_id",
                "evidence",
                "kind",
                "mutation_control_rejected",
                "positive_control_passed",
                "provenance",
                "status",
            },
            "dataset result",
        )
        if (
            item["positive_control_passed"] is not True
            or item["mutation_control_rejected"] is not True
            or item["status"] != "PASS_EXECUTABLE_DATASET_CONTROL"
        ):
            raise DatasetChallengeError("dataset executable control outcome changed")
    expected_sources = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    if set(value.get("source_bindings", {})) != set(expected_sources):
        raise DatasetChallengeError("dataset source bindings changed")
    for key, path in expected_sources.items():
        binding = value["source_bindings"][key]
        _strict_keys(binding, {"normalized_file_sha256", "path"}, "dataset source binding")
        if binding["path"] != path:
            raise DatasetChallengeError("dataset source path changed")
    if root is not None:
        root = root.resolve()
        for key, path in expected_sources.items():
            bound_path = (root / path).resolve()
            try:
                bound_path.relative_to(root)
            except ValueError as error:
                raise DatasetChallengeError("dataset source path escapes root") from error
            if (
                value["source_bindings"][key]["normalized_file_sha256"]
                != _normalized_sha256(bound_path)
            ):
                raise DatasetChallengeError("dataset source hash changed")
        if dict(value) != _build_receipt(root):
            raise DatasetChallengeError("dataset receipt does not exactly replay")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run":
        receipt = run_dataset_challenges(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path = args.receipt if args.receipt.is_absolute() else root / args.receipt
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_dataset_challenges(receipt, root)
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
