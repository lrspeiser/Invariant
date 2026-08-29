"""Replay Item 30 with a narrow frozen-result assembly key correction.

The frozen evaluator computes slice improvements under the key
``improvement_vs_flexible_nuisance`` but its partial-pattern collector asks for
``improvement_vs_flexible``.  The first GPU run therefore failed closed with a
``KeyError`` after scoring and before writing a result.  This adapter supplies a
read-only alias for the already-frozen flexible prediction, verifies that both
slice metrics are numerically identical, and removes the redundant baseline from
the serialized result.  It changes no candidate, response, fit, null, or gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import sigma_theory_compiler.gravity_item30_screening_mechanisms as frozen


def _with_flexible_alias(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(predictions)
    if "flexible_nuisance" not in result:
        raise frozen.GravityItem30Error("frozen flexible nuisance prediction is missing")
    result["flexible"] = result["flexible_nuisance"]
    return result


def _clean_corrected_scientific(scientific: dict[str, Any]) -> dict[str, Any]:
    baseline_mse = scientific["metrics"]["baseline_mse"]
    alias_mse = float(baseline_mse.pop("flexible"))
    if not np.isclose(alias_mse, float(baseline_mse["flexible_nuisance"]), rtol=0.0, atol=0.0):
        raise frozen.GravityItem30Error("flexible baseline alias changed the overall score")
    for value in scientific["broad_slices"].values():
        slice_alias_mse = float(value.pop("flexible_mse"))
        if not np.isclose(
            slice_alias_mse, float(value["flexible_nuisance_mse"]), rtol=0.0, atol=0.0
        ):
            raise frozen.GravityItem30Error("flexible baseline alias changed a slice score")
        if not np.isclose(
            float(value["improvement_vs_flexible"]),
            float(value["improvement_vs_flexible_nuisance"]),
            rtol=0.0,
            atol=0.0,
        ):
            raise frozen.GravityItem30Error("flexible improvement alias changed a slice score")
    return scientific


def evaluate_with_key_correction(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = frozen._baseline_predictions

    def aliased(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return _with_flexible_alias(original(*args, **kwargs))

    frozen._baseline_predictions = aliased
    try:
        scientific, compute = frozen._evaluate(config, rows)
    finally:
        frozen._baseline_predictions = original
    return _clean_corrected_scientific(scientific), compute


def run_corrected_experiment(root: Path) -> Path:
    root = root.resolve()
    config = frozen.load_config(root)
    frozen.verify_science_freeze(root, config)
    frozen.verify_sample_freeze(root, config)
    rows, response_manifest, extraction = frozen._load_response_rows(root, config)
    scientific, compute = evaluate_with_key_correction(config, rows)
    paths = frozen._source_paths(root, config)

    correction_path = paths["compute_manifest"].with_name("evaluation-result-key-correction.json")
    correction = frozen._content_hashed(
        {
            "schema_version": "invariant-gravity-item30-evaluation-key-correction-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "failure": "The frozen partial-slice collector requested improvement_vs_flexible after the slice evaluator emitted improvement_vs_flexible_nuisance; the first run raised KeyError before writing a result or compute manifest.",
            "correction": "Add flexible as a read-only alias of the frozen flexible_nuisance prediction during evaluation, require bit-exact equality of every duplicate MSE and improvement, then remove only the redundant flexible MSE from serialization.",
            "counts": {
                "candidate_cells_changed": 0,
                "sample_roles_changed": 0,
                "response_values_changed": 0,
                "baseline_predictions_changed": 0,
                "null_trials_changed": 0,
                "gate_thresholds_changed": 0,
                "confirmation_values_read": 0,
                "paid_api_calls": 0,
            },
            "claims": {
                "scientific_contract_changed": False,
                "failed_run_wrote_result": False,
                "failed_run_wrote_compute_manifest": False,
            },
        }
    )
    frozen._write_json(correction_path, correction)
    compute["evaluation_result_key_correction"] = {
        "path": correction_path.relative_to(root).as_posix(),
        "sha256": frozen._sha256_file(correction_path),
        "content_sha256": correction["content_sha256"],
    }
    compute_manifest = frozen._content_hashed(
        {"schema_version": "invariant-gravity-item30-compute-1.0", **compute}
    )
    frozen._write_json(paths["compute_manifest"], compute_manifest)
    receipt = frozen._build_receipt(
        root, config, rows, response_manifest, extraction, scientific, compute
    )
    result_path = root / str(config["paths"]["result"])
    frozen._write_json(result_path, receipt)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(run_corrected_experiment(args.root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
