"""Executable calibration suite for intervention, noise, shift, and identifiability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creative_dataset_challenges.json"
CONFIG_SCHEMA = "invariant-creative-dataset-challenges-1.0"
RESULT_SCHEMA = "invariant-creative-dataset-challenge-result-1.0"


class DatasetChallengeError(ValueError):
    """A dataset challenge lost its typed calibration evidence."""


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


def _intervention_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    records = challenge.get("records", [])
    if not isinstance(records, list) or len(records) < 4:
        raise DatasetChallengeError("intervention challenge lacks records")
    interventions = {row.get("intervention") for row in records if isinstance(row, Mapping)}
    covariates = {row.get("covariate") for row in records if isinstance(row, Mapping)}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {"covariate", "intervention", "outcome"}:
            raise DatasetChallengeError("intervention record schema changed")
        _fraction(row["covariate"], "intervention covariate")
        _fraction(row["outcome"], "intervention outcome")
    crossed = all(
        any(row["covariate"] == covariate and row["intervention"] == intervention for row in records)
        for covariate in covariates
        for intervention in interventions
    )
    if interventions != {"do_treatment_0", "do_treatment_1"} or not crossed:
        raise DatasetChallengeError("intervention labels are not crossed at fixed covariates")
    return {
        "actual_do_labeled_rows": len(records),
        "crossed_at_fixed_covariates": True,
        "observational_rows_relabelled_as_interventions": False,
    }


def _noise_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    records = challenge.get("records", [])
    radii = []
    for row in records if isinstance(records, list) else []:
        if not isinstance(row, Mapping) or set(row) != {"center", "input", "radius"}:
            raise DatasetChallengeError("noisy record schema changed")
        _fraction(row["input"], "noisy input")
        _fraction(row["center"], "noisy center")
        radii.append(_fraction(row["radius"], "noise radius"))
    if not radii or any(radius <= 0 for radius in radii):
        raise DatasetChallengeError("noise intervals must have positive radii")
    return {
        "interval_rows": len(radii),
        "positive_uncertainty_declared": True,
        "point_values_misreported_as_exact": False,
    }


def _shift_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    records = challenge.get("records", [])
    splits: dict[str, list[Fraction]] = {"deployment": [], "train": []}
    for row in records if isinstance(records, list) else []:
        if not isinstance(row, Mapping) or set(row) != {"input", "outcome", "split"}:
            raise DatasetChallengeError("shift record schema changed")
        if row["split"] not in splits:
            raise DatasetChallengeError("shift split is unsupported")
        splits[row["split"]].append(_fraction(row["input"], "shift input"))
        _fraction(row["outcome"], "shift outcome")
    if not all(splits.values()) or max(splits["train"]) >= min(splits["deployment"]):
        raise DatasetChallengeError("deployment domain is not an explicit covariate shift")
    return {
        "deployment_rows": len(splits["deployment"]),
        "domain_shift_explicit": True,
        "train_rows": len(splits["train"]),
    }


def _unidentifiable_evidence(challenge: Mapping[str, Any]) -> dict[str, Any]:
    observed = challenge.get("observational_outcomes")
    models = challenge.get("compatible_models")
    if not isinstance(observed, list) or not isinstance(models, list) or len(models) < 2:
        raise DatasetChallengeError("unidentifiable challenge is incomplete")
    observational = []
    interventions = []
    for model in models:
        if not isinstance(model, Mapping) or set(model) != {
            "intervention_predictions",
            "model_id",
            "observational_predictions",
        }:
            raise DatasetChallengeError("compatible model schema changed")
        observational.append(model["observational_predictions"])
        interventions.append(model["intervention_predictions"])
    if any(predictions != observed for predictions in observational) or len(
        {canonical_sha256(item) for item in interventions}
    ) < 2:
        raise DatasetChallengeError("models are not observationally equal and interventionally distinct")
    return {
        "compatible_mechanisms": len(models),
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


def run_dataset_challenges(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"challenges", "schema_version"} or config["schema_version"] != CONFIG_SCHEMA:
        raise DatasetChallengeError("dataset challenge config identity changed")
    challenges = config["challenges"]
    if not isinstance(challenges, list):
        raise DatasetChallengeError("dataset challenges are not an array")
    results = []
    kinds = set()
    for challenge in challenges:
        if not isinstance(challenge, Mapping):
            raise DatasetChallengeError("dataset challenge is not an object")
        kind = challenge.get("kind")
        if kind not in _EXECUTORS:
            raise DatasetChallengeError("dataset challenge kind is unsupported")
        kinds.add(kind)
        evidence = _EXECUTORS[kind](challenge)
        results.append(
            {
                "challenge_id": challenge.get("challenge_id"),
                "evidence": evidence,
                "kind": kind,
                "provenance": challenge.get("provenance"),
                "status": "PASS_CALIBRATION_CHALLENGE",
            }
        )
    if kinds != set(_EXECUTORS):
        raise DatasetChallengeError("dataset challenge coverage changed")
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "source": {"path": CONFIG_PATH, "sha256": _normalized_sha256(path)},
        "results": results,
        "summary": {
            "challenge_kinds": sorted(kinds),
            "passed": len(results),
            "total": len(results),
        },
        "claims": {
            "causal_effect_inferred_from_observational_rows": False,
            "dataset_formulas_established": False,
            "unidentifiable_control_forced_to_one_model": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_dataset_challenges(body)
    return body


def validate_dataset_challenges(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise DatasetChallengeError("dataset challenge result seal changed")
    if value.get("schema_version") != RESULT_SCHEMA:
        raise DatasetChallengeError("dataset challenge result schema changed")
    summary = value.get("summary", {})
    claims = value.get("claims", {})
    if (
        summary.get("challenge_kinds") != sorted(_EXECUTORS)
        or summary.get("passed") != summary.get("total")
        or summary.get("passed", 0) < 4
        or any(claims.get(key) is not False for key in claims)
    ):
        raise DatasetChallengeError("dataset calibration boundary changed")
